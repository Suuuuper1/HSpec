# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming table builder primitives for HSpec Phase 2.

This module is intentionally CPU/NumPy only. It reads descriptor-backed raw
hidden-state mmap files by tile, computes prompt-level PCA parameters, and
projects keys directly into mmap table-store arrays. It does not touch Ray
actors, torch tensors, NPU streams, or the decode hot path.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import logging
import time
from typing import Any, Iterator, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import fallback is environment dependent.
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from vllm_ascend.spec_decode.hspec_store import (
    HSpecTrajectoryDesc,
    coerce_hspec_desc,
    estimate_hspec_trajectory_bytes,
    hspec_record_store_metric,
    hspec_record_store_metric_max,
    load_hspec_trajectory,
)
from vllm_ascend.spec_decode.hspec_table_store import (
    HSpecPromptTableDesc,
    HSpecTableStoreWriter,
    TABLE_STORE_SCHEMA_VERSION,
    get_hspec_pca_accum_dtype,
    get_hspec_pca_cov_max_bytes,
    get_hspec_pca_method,
    get_hspec_pca_random_oversample,
    get_hspec_pca_random_seed,
    get_hspec_pca_tile_rows,
    get_hspec_table_keys_dtype,
)

logger = logging.getLogger(__name__)

_VALID_PCA_METHODS = frozenset({"randomized", "covariance", "auto", "svd_reference"})
_VALID_ACCUM_DTYPES = frozenset({"float32", "float64"})
_VALID_KEY_DTYPES = frozenset({"float16"})
_DEFAULT_REFERENCE_MAX_ROWS = 32768
_DEFAULT_REFERENCE_MAX_FP32_BYTES = 512 * 1024 * 1024


class HSpecPCAError(RuntimeError):
    """Base error for HSpec streaming PCA builder failures."""


class HSpecPCAInsufficientSamples(HSpecPCAError):
    """Raised when a prompt does not have enough rows to fit PCA."""


class HSpecPCADimensionMismatch(HSpecPCAError):
    """Raised when descriptor hidden dimensions are invalid or inconsistent."""


class HSpecPCAReferenceBudgetExceeded(HSpecPCAError):
    """Raised when full-SVD debug reference would exceed its safety budget."""


class HSpecTableBuildEmpty(HSpecPCAInsufficientSamples):
    """Raised when descriptors cannot produce any value-shifted table entry."""


@dataclass(frozen=True)
class HSpecPCAConfig:
    method: str
    n_components: int
    tile_rows: int
    randomized_oversample: int
    randomized_seed: int
    covariance_max_bytes: int
    accum_dtype: str
    keys_dtype: str = "float16"

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in _VALID_PCA_METHODS:
            raise ValueError(f"Unsupported HSpec PCA method: {self.method!r}")
        if int(self.n_components) <= 0:
            raise ValueError(f"n_components must be > 0, got {self.n_components}")
        if int(self.tile_rows) <= 0:
            raise ValueError(f"tile_rows must be > 0, got {self.tile_rows}")
        if int(self.randomized_oversample) < 0:
            raise ValueError(
                "randomized_oversample must be >= 0, "
                f"got {self.randomized_oversample}"
            )
        if int(self.covariance_max_bytes) < 0:
            raise ValueError(
                "covariance_max_bytes must be >= 0, "
                f"got {self.covariance_max_bytes}"
            )
        accum_dtype = str(self.accum_dtype).strip().lower()
        if accum_dtype not in _VALID_ACCUM_DTYPES:
            raise ValueError(f"Unsupported HSpec PCA accumulation dtype: {self.accum_dtype!r}")
        keys_dtype = str(self.keys_dtype).strip().lower()
        if keys_dtype not in _VALID_KEY_DTYPES:
            raise ValueError(f"Unsupported HSpec table keys dtype: {self.keys_dtype!r}")

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "n_components", int(self.n_components))
        object.__setattr__(self, "tile_rows", int(self.tile_rows))
        object.__setattr__(self, "randomized_oversample", int(self.randomized_oversample))
        object.__setattr__(self, "randomized_seed", int(self.randomized_seed))
        object.__setattr__(self, "covariance_max_bytes", int(self.covariance_max_bytes))
        object.__setattr__(self, "accum_dtype", accum_dtype)
        object.__setattr__(self, "keys_dtype", keys_dtype)

    @classmethod
    def from_env(cls, n_components: int) -> "HSpecPCAConfig":
        return cls(
            method=get_hspec_pca_method(),
            n_components=int(n_components),
            tile_rows=get_hspec_pca_tile_rows(),
            randomized_oversample=get_hspec_pca_random_oversample(),
            randomized_seed=get_hspec_pca_random_seed(),
            covariance_max_bytes=get_hspec_pca_cov_max_bytes(),
            accum_dtype=get_hspec_pca_accum_dtype(),
            keys_dtype=get_hspec_table_keys_dtype(),
        )


@dataclass
class HSpecPCAMetrics:
    method_requested: str
    method_used: str = ""
    method_fallback_count: int = 0
    input_desc_count: int = 0
    valid_desc_count: int = 0
    input_rows: int = 0
    hidden_dim: int = 0
    tile_rows: int = 0
    tile_count: int = 0
    processed_fp32_tile_bytes: int = 0
    mean_ms: float = 0.0
    basis_ms: float = 0.0
    total_ms: float = 0.0
    covariance_bytes: int = 0
    randomized_rank: int = 0
    insufficient_samples_count: int = 0
    pca_error_count: int = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return dict(asdict(self))


@dataclass(frozen=True)
class HSpecPCAResult:
    prompt_id: str
    mean: np.ndarray
    components: np.ndarray
    n_samples: int
    method: str
    metrics: HSpecPCAMetrics

    def __post_init__(self) -> None:
        mean = np.ascontiguousarray(self.mean, dtype=np.float32)
        components = np.ascontiguousarray(self.components, dtype=np.float32)
        if mean.ndim != 1:
            raise ValueError(f"HSpecPCAResult.mean must be 1-D, got {mean.shape}")
        if components.ndim != 2:
            raise ValueError(
                f"HSpecPCAResult.components must be 2-D, got {components.shape}"
            )
        if components.shape[1] != mean.shape[0]:
            raise ValueError(
                "HSpecPCAResult components hidden dim mismatch: "
                f"mean={mean.shape} components={components.shape}"
            )
        object.__setattr__(self, "prompt_id", str(self.prompt_id))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "method", str(self.method))

    def to_prompt_pca_params(self):
        from vllm_ascend.spec_decode.hspec_utils import PromptPCAParams

        return PromptPCAParams(
            prompt_id=self.prompt_id,
            mean=self.mean,
            components=self.components,
            n_samples=self.n_samples,
        )


@dataclass
class HSpecPromptTableBuildMetrics:
    input_desc_count: int = 0
    included_desc_count: int = 0
    input_rows: int = 0
    included_rows: int = 0
    n_entries: int = 0
    n_rollouts: int = 0
    token_count: int = 0
    hidden_dim: int = 0
    n_components: int = 0
    pca_method: str = ""
    pca_mean_ms: float = 0.0
    pca_basis_ms: float = 0.0
    pca_total_ms: float = 0.0
    projection_ms: float = 0.0
    table_write_ms: float = 0.0
    total_ms: float = 0.0
    tile_rows: int = 0
    projection_tile_count: int = 0
    processed_fp32_tile_bytes: int = 0
    covariance_bytes: int = 0
    randomized_rank: int = 0
    method_fallback_count: int = 0
    memory_error_count: int = 0
    build_error_count: int = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return dict(asdict(self))


@dataclass(frozen=True)
class _PromptTablePlanItem:
    desc: HSpecTrajectoryDesc
    rollout_idx: int
    length: int
    n_entries: int
    reward: float


@dataclass(frozen=True)
class _PromptTablePlan:
    items: tuple[_PromptTablePlanItem, ...]
    entry_count: int
    token_count: int
    rollout_count: int
    input_rows: int


def _close_memmap(array: Any) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _flush_close_memmap(array: Any) -> None:
    try:
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()
    finally:
        _close_memmap(array)


def _coerce_desc_list(
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
) -> list[HSpecTrajectoryDesc]:
    result = [coerce_hspec_desc(item) for item in descs if item is not None]
    if not result:
        raise HSpecPCAInsufficientSamples("empty HSpec descriptor list")
    return result


def _infer_hidden_dim(descs: Sequence[HSpecTrajectoryDesc]) -> int:
    hidden_dim: Optional[int] = None
    for desc in descs:
        desc_dim = int(desc.hidden_dim)
        if desc_dim <= 0:
            raise HSpecPCADimensionMismatch(
                f"HSpec descriptor has invalid hidden_dim={desc.hidden_dim}: "
                f"request_id={desc.request_id}"
            )
        if hidden_dim is None:
            hidden_dim = desc_dim
        elif desc_dim != hidden_dim:
            raise HSpecPCADimensionMismatch(
                "HSpec descriptor hidden_dim mismatch: "
                f"expected={hidden_dim}, got={desc_dim}, request_id={desc.request_id}"
            )
    if hidden_dim is None:
        raise HSpecPCAInsufficientSamples("empty HSpec descriptor list")
    return int(hidden_dim)


def _count_rows(descs: Sequence[HSpecTrajectoryDesc]) -> tuple[int, int]:
    rows = 0
    valid_descs = 0
    for desc in descs:
        length = int(desc.length)
        if length > 0:
            rows += length
            valid_descs += 1
    return rows, valid_descs


def _threadpool_context(blas_threads: int):
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=max(int(blas_threads), 1))


def _covariance_bytes(hidden_dim: int, accum_dtype: str) -> int:
    dtype = np.dtype(accum_dtype)
    return int(hidden_dim) * int(hidden_dim) * int(dtype.itemsize)


def _covariance_fits(hidden_dim: int, config: HSpecPCAConfig) -> bool:
    return _covariance_bytes(hidden_dim, config.accum_dtype) <= int(
        config.covariance_max_bytes
    )


def _record_builder_metrics(metrics: HSpecPCAMetrics) -> None:
    hspec_record_store_metric("pca_mean_ms_total", int(round(metrics.mean_ms)))
    hspec_record_store_metric("pca_basis_ms_total", int(round(metrics.basis_ms)))
    hspec_record_store_metric("pca_tile_count", int(metrics.tile_count))
    hspec_record_store_metric(
        "pca_processed_fp32_tile_bytes",
        int(metrics.processed_fp32_tile_bytes),
    )
    if metrics.method_used == "randomized":
        hspec_record_store_metric("pca_method_randomized_count", 1)
    elif metrics.method_used == "covariance":
        hspec_record_store_metric("pca_method_covariance_count", 1)
    elif metrics.method_used == "svd_reference":
        hspec_record_store_metric("pca_method_svd_reference_count", 1)
    if metrics.method_fallback_count:
        hspec_record_store_metric(
            "pca_method_fallback_count",
            int(metrics.method_fallback_count),
        )
    if metrics.insufficient_samples_count:
        hspec_record_store_metric(
            "pca_insufficient_samples_count",
            int(metrics.insufficient_samples_count),
        )
    if metrics.pca_error_count:
        hspec_record_store_metric("pca_error_count", int(metrics.pca_error_count))
    hspec_record_store_metric_max("pca_cov_bytes_max", int(metrics.covariance_bytes))
    hspec_record_store_metric_max("pca_randomized_rank_max", int(metrics.randomized_rank))


def _record_table_build_metrics(metrics: HSpecPromptTableBuildMetrics) -> None:
    hspec_record_store_metric("table_build_projection_ms_total",
                              int(round(metrics.projection_ms)))
    hspec_record_store_metric("table_build_write_ms_total",
                              int(round(metrics.table_write_ms)))
    hspec_record_store_metric("table_build_projection_tile_count",
                              int(metrics.projection_tile_count))
    hspec_record_store_metric("table_build_entry_count", int(metrics.n_entries))
    hspec_record_store_metric("table_build_rollout_count", int(metrics.n_rollouts))
    hspec_record_store_metric("table_build_token_count", int(metrics.token_count))
    if metrics.build_error_count:
        hspec_record_store_metric("table_build_error_count",
                                  int(metrics.build_error_count))


def _plan_prompt_table(
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    *,
    max_entries: int,
) -> _PromptTablePlan:
    desc_list = _coerce_desc_list(descs)
    remaining_entries = max(int(max_entries), 0)
    input_rows = 0
    token_count = 0
    entry_count = 0
    items: list[_PromptTablePlanItem] = []
    for desc_obj in desc_list:
        desc = coerce_hspec_desc(desc_obj)
        length = max(int(desc.length), 0)
        input_rows += length
        if remaining_entries <= 0:
            continue
        rollout_entries = max(length - 1, 0)
        if rollout_entries <= 0:
            continue
        n_add = min(rollout_entries, remaining_entries)
        items.append(
            _PromptTablePlanItem(
                desc=desc,
                rollout_idx=len(items),
                length=length,
                n_entries=n_add,
                reward=float(desc.reward or 0.0),
            ))
        entry_count += n_add
        token_count += length
        remaining_entries -= n_add
    return _PromptTablePlan(
        items=tuple(items),
        entry_count=int(entry_count),
        token_count=int(token_count),
        rollout_count=len(items),
        input_rows=int(input_rows),
    )


def iter_prompt_hidden_tiles(
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    tile_rows: int,
    *,
    dtype: np.dtype | str = np.float32,
) -> Iterator[tuple[HSpecTrajectoryDesc, int, np.ndarray]]:
    """Yield descriptor-backed hidden rows as small contiguous CPU tiles."""
    if int(tile_rows) <= 0:
        raise ValueError(f"tile_rows must be > 0, got {tile_rows}")
    dtype_np = np.dtype(dtype)
    for desc_obj in descs:
        desc = coerce_hspec_desc(desc_obj)
        if int(desc.length) <= 0:
            continue
        hs = None
        tokens = None
        try:
            hs, tokens = load_hspec_trajectory(desc)
            _close_memmap(tokens)
            tokens = None
            length = int(desc.length)
            for start in range(0, length, int(tile_rows)):
                stop = min(start + int(tile_rows), length)
                tile = np.asarray(hs[start:stop], dtype=dtype_np, order="C")
                yield desc, start, np.ascontiguousarray(tile)
        finally:
            if tokens is not None:
                _close_memmap(tokens)
            if hs is not None:
                _close_memmap(hs)


def compute_streaming_mean(
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    config: HSpecPCAConfig,
    *,
    metrics: Optional[HSpecPCAMetrics] = None,
) -> tuple[np.ndarray, int, HSpecPCAMetrics]:
    """Compute prompt-level mean with a single tiled pass over hidden rows."""
    t0 = time.perf_counter()
    desc_list = _coerce_desc_list(descs)
    hidden_dim = _infer_hidden_dim(desc_list)
    input_rows, valid_desc_count = _count_rows(desc_list)
    if metrics is None:
        metrics = HSpecPCAMetrics(method_requested=config.method)
    metrics.input_desc_count = len(desc_list)
    metrics.valid_desc_count = valid_desc_count
    metrics.input_rows = input_rows
    metrics.hidden_dim = hidden_dim
    metrics.tile_rows = int(config.tile_rows)

    sum_dtype = np.dtype(config.accum_dtype)
    sum_h = np.zeros((hidden_dim,), dtype=sum_dtype)
    n_samples = 0
    try:
        for _, _, tile in iter_prompt_hidden_tiles(desc_list, config.tile_rows, dtype=np.float32):
            if tile.ndim != 2 or tile.shape[1] != hidden_dim:
                raise HSpecPCADimensionMismatch(
                    f"tile hidden_dim mismatch for prompt_id={prompt_id}: "
                    f"tile_shape={tile.shape}, expected_dim={hidden_dim}"
                )
            sum_h += tile.sum(axis=0, dtype=sum_dtype)
            n_samples += int(tile.shape[0])
            metrics.tile_count += 1
            metrics.processed_fp32_tile_bytes += int(tile.nbytes)
        if n_samples < 2:
            metrics.insufficient_samples_count += 1
            raise HSpecPCAInsufficientSamples(
                f"HSpec PCA requires at least 2 samples for prompt_id={prompt_id}, "
                f"got {n_samples}"
            )
        mean = np.ascontiguousarray(sum_h / float(n_samples), dtype=np.float32)
        return mean, n_samples, metrics
    finally:
        metrics.mean_ms += float((time.perf_counter() - t0) * 1000.0)


def _canonicalize_and_pad_components(
    components: np.ndarray,
    n_components: int,
    hidden_dim: int,
) -> np.ndarray:
    comp = np.asarray(components, dtype=np.float32, order="C")
    if comp.ndim != 2:
        raise ValueError(f"components must be 2-D, got {comp.shape}")
    if comp.shape[1] != int(hidden_dim):
        raise ValueError(
            f"components hidden dim mismatch: shape={comp.shape}, hidden_dim={hidden_dim}"
        )
    rows = min(int(comp.shape[0]), int(n_components))
    if rows > 0:
        comp = np.ascontiguousarray(comp[:rows], dtype=np.float32)
        for row in range(rows):
            vec = comp[row]
            pivot = int(np.argmax(np.abs(vec)))
            if float(vec[pivot]) < 0.0:
                comp[row] = -vec
    else:
        comp = np.empty((0, int(hidden_dim)), dtype=np.float32)
    if rows < int(n_components):
        padding = np.zeros((int(n_components) - rows, int(hidden_dim)), dtype=np.float32)
        comp = np.vstack([comp, padding])
    return np.ascontiguousarray(comp, dtype=np.float32)


def compute_pca_tiled_covariance(
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    mean: np.ndarray,
    n_samples: int,
    config: HSpecPCAConfig,
    *,
    metrics: Optional[HSpecPCAMetrics] = None,
) -> HSpecPCAResult:
    """Compute exact PCA from a tiled covariance matrix, with memory fallback."""
    desc_list = _coerce_desc_list(descs)
    hidden_dim = _infer_hidden_dim(desc_list)
    if metrics is None:
        metrics = HSpecPCAMetrics(method_requested=config.method)
    metrics.hidden_dim = hidden_dim
    metrics.covariance_bytes = _covariance_bytes(hidden_dim, config.accum_dtype)
    if metrics.covariance_bytes > int(config.covariance_max_bytes):
        metrics.method_fallback_count += 1
        return compute_pca_randomized_cov(
            prompt_id,
            desc_list,
            mean,
            n_samples,
            config,
            metrics=metrics,
        )

    t0 = time.perf_counter()
    try:
        accum_dtype = np.dtype(config.accum_dtype)
        cov = np.zeros((hidden_dim, hidden_dim), dtype=accum_dtype)
        mean_fp32 = np.asarray(mean, dtype=np.float32)
        for _, _, tile in iter_prompt_hidden_tiles(desc_list, config.tile_rows, dtype=np.float32):
            if tile.shape[1] != hidden_dim:
                raise HSpecPCADimensionMismatch(
                    f"tile hidden_dim mismatch for prompt_id={prompt_id}: "
                    f"tile_shape={tile.shape}, expected_dim={hidden_dim}"
                )
            tile -= mean_fp32
            centered = tile.astype(accum_dtype, copy=False)
            cov += centered.T @ centered
        cov /= float(max(int(n_samples) - 1, 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        rank = min(int(config.n_components), int(hidden_dim), int(n_samples))
        components = eigvecs[:, order[:rank]].T.astype(np.float32, copy=False)
        components = _canonicalize_and_pad_components(
            components,
            config.n_components,
            hidden_dim,
        )
        metrics.method_used = "covariance"
        return HSpecPCAResult(
            prompt_id=prompt_id,
            mean=mean,
            components=components,
            n_samples=n_samples,
            method="covariance",
            metrics=metrics,
        )
    except HSpecPCAReferenceBudgetExceeded:
        raise
    except Exception:
        metrics.pca_error_count += 1
        raise
    finally:
        metrics.basis_ms += float((time.perf_counter() - t0) * 1000.0)


def _deterministic_seed(prompt_id: str, base_seed: int) -> int:
    payload = f"{prompt_id}:{int(base_seed)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & 0xFFFFFFFF


def _deterministic_normal_matrix(
    prompt_id: str,
    hidden_dim: int,
    rank: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(_deterministic_seed(prompt_id, seed))
    omega = rng.standard_normal((int(hidden_dim), int(rank))).astype(np.float32)
    return np.ascontiguousarray(omega, dtype=np.float32)


def compute_pca_randomized_cov(
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    mean: np.ndarray,
    n_samples: int,
    config: HSpecPCAConfig,
    *,
    metrics: Optional[HSpecPCAMetrics] = None,
) -> HSpecPCAResult:
    """Compute randomized covariance PCA using tiled passes over hidden rows."""
    desc_list = _coerce_desc_list(descs)
    hidden_dim = _infer_hidden_dim(desc_list)
    if metrics is None:
        metrics = HSpecPCAMetrics(method_requested=config.method)
    metrics.hidden_dim = hidden_dim
    rank = min(
        int(hidden_dim),
        max(1, int(config.n_components) + int(config.randomized_oversample)),
    )
    metrics.randomized_rank = rank

    t0 = time.perf_counter()
    try:
        mean_fp32 = np.asarray(mean, dtype=np.float32)
        omega = _deterministic_normal_matrix(
            prompt_id,
            hidden_dim,
            rank,
            config.randomized_seed,
        )
        y = np.zeros((hidden_dim, rank), dtype=np.float32)
        for _, _, tile in iter_prompt_hidden_tiles(desc_list, config.tile_rows, dtype=np.float32):
            if tile.shape[1] != hidden_dim:
                raise HSpecPCADimensionMismatch(
                    f"tile hidden_dim mismatch for prompt_id={prompt_id}: "
                    f"tile_shape={tile.shape}, expected_dim={hidden_dim}"
                )
            tile -= mean_fp32
            sketch = tile @ omega
            y += tile.T @ sketch

        q, _ = np.linalg.qr(y, mode="reduced")
        q = np.ascontiguousarray(q.astype(np.float32, copy=False))

        b = np.zeros((rank, rank), dtype=np.float32)
        for _, _, tile in iter_prompt_hidden_tiles(desc_list, config.tile_rows, dtype=np.float32):
            tile -= mean_fp32
            aq = tile @ q
            b += aq.T @ aq
        b /= float(max(int(n_samples) - 1, 1))
        eigvals, eigvecs = np.linalg.eigh(b)
        order = np.argsort(eigvals)[::-1]
        take = min(int(config.n_components), int(rank))
        components = (q @ eigvecs[:, order[:take]]).T.astype(np.float32, copy=False)
        components = _canonicalize_and_pad_components(
            components,
            config.n_components,
            hidden_dim,
        )
        metrics.method_used = "randomized"
        return HSpecPCAResult(
            prompt_id=prompt_id,
            mean=mean,
            components=components,
            n_samples=n_samples,
            method="randomized",
            metrics=metrics,
        )
    except Exception:
        metrics.pca_error_count += 1
        raise
    finally:
        metrics.basis_ms += float((time.perf_counter() - t0) * 1000.0)


def _fit_pca_svd_reference_guarded(
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    mean: np.ndarray,
    n_samples: int,
    config: HSpecPCAConfig,
    *,
    metrics: HSpecPCAMetrics,
    allow_svd_reference: bool,
    reference_max_rows: int,
    reference_max_fp32_bytes: int,
) -> HSpecPCAResult:
    if not allow_svd_reference:
        raise HSpecPCAReferenceBudgetExceeded(
            "HSPEC_PCA_METHOD=svd_reference requires allow_svd_reference=True"
        )
    desc_list = _coerce_desc_list(descs)
    hidden_dim = _infer_hidden_dim(desc_list)
    fp32_bytes = int(n_samples) * int(hidden_dim) * np.dtype(np.float32).itemsize
    raw_bytes = sum(estimate_hspec_trajectory_bytes(desc) for desc in desc_list)
    if int(n_samples) > int(reference_max_rows):
        raise HSpecPCAReferenceBudgetExceeded(
            f"svd_reference rows exceed budget: rows={n_samples} "
            f"budget={reference_max_rows}"
        )
    if fp32_bytes > int(reference_max_fp32_bytes) or raw_bytes > int(reference_max_fp32_bytes):
        raise HSpecPCAReferenceBudgetExceeded(
            "svd_reference bytes exceed budget: "
            f"fp32_bytes={fp32_bytes} raw_bytes={raw_bytes} "
            f"budget={reference_max_fp32_bytes}"
        )

    t0 = time.perf_counter()
    arrays: list[np.ndarray] = []
    try:
        for _, _, tile in iter_prompt_hidden_tiles(desc_list, config.tile_rows, dtype=np.float32):
            arrays.append(tile)
        if not arrays:
            raise HSpecPCAInsufficientSamples(
                f"empty svd_reference input for prompt_id={prompt_id}"
            )
        full = np.concatenate(arrays, axis=0)
        centered = full - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(int(config.n_components), int(hidden_dim), int(n_samples))
        components = vt[:rank].astype(np.float32, copy=False)
        components = _canonicalize_and_pad_components(
            components,
            config.n_components,
            hidden_dim,
        )
        metrics.method_used = "svd_reference"
        return HSpecPCAResult(
            prompt_id=prompt_id,
            mean=mean,
            components=components,
            n_samples=n_samples,
            method="svd_reference",
            metrics=metrics,
        )
    except Exception:
        metrics.pca_error_count += 1
        raise
    finally:
        arrays.clear()
        metrics.basis_ms += float((time.perf_counter() - t0) * 1000.0)


def fit_prompt_pca_streaming(
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    *,
    config: HSpecPCAConfig,
    blas_threads: int = 1,
    allow_svd_reference: bool = False,
    reference_max_rows: int = _DEFAULT_REFERENCE_MAX_ROWS,
    reference_max_fp32_bytes: int = _DEFAULT_REFERENCE_MAX_FP32_BYTES,
) -> HSpecPCAResult:
    """Fit prompt-level PCA from descriptor-backed raw hidden rows by tile."""
    t_total = time.perf_counter()
    metrics = HSpecPCAMetrics(method_requested=config.method)
    try:
        with _threadpool_context(blas_threads):
            mean, n_samples, metrics = compute_streaming_mean(
                prompt_id,
                descs,
                config,
                metrics=metrics,
            )
            if config.method == "randomized":
                result = compute_pca_randomized_cov(
                    prompt_id,
                    descs,
                    mean,
                    n_samples,
                    config,
                    metrics=metrics,
                )
            elif config.method == "covariance":
                result = compute_pca_tiled_covariance(
                    prompt_id,
                    descs,
                    mean,
                    n_samples,
                    config,
                    metrics=metrics,
                )
            elif config.method == "auto":
                hidden_dim = int(mean.shape[0])
                metrics.covariance_bytes = _covariance_bytes(
                    hidden_dim,
                    config.accum_dtype,
                )
                if _covariance_fits(hidden_dim, config):
                    result = compute_pca_tiled_covariance(
                        prompt_id,
                        descs,
                        mean,
                        n_samples,
                        config,
                        metrics=metrics,
                    )
                else:
                    metrics.method_fallback_count += 1
                    result = compute_pca_randomized_cov(
                        prompt_id,
                        descs,
                        mean,
                        n_samples,
                        config,
                        metrics=metrics,
                    )
            elif config.method == "svd_reference":
                result = _fit_pca_svd_reference_guarded(
                    prompt_id,
                    descs,
                    mean,
                    n_samples,
                    config,
                    metrics=metrics,
                    allow_svd_reference=allow_svd_reference,
                    reference_max_rows=reference_max_rows,
                    reference_max_fp32_bytes=reference_max_fp32_bytes,
                )
            else:  # guarded by HSpecPCAConfig, kept for defensive clarity.
                raise ValueError(f"Unsupported HSpec PCA method: {config.method}")
    except HSpecPCAInsufficientSamples:
        metrics.insufficient_samples_count = max(metrics.insufficient_samples_count, 1)
        _record_builder_metrics(metrics)
        raise
    except HSpecPCAReferenceBudgetExceeded:
        raise
    except Exception:
        metrics.pca_error_count = max(metrics.pca_error_count, 1)
        _record_builder_metrics(metrics)
        raise

    metrics.total_ms = float((time.perf_counter() - t_total) * 1000.0)
    result = replace(result, metrics=metrics)
    _record_builder_metrics(metrics)
    return result


def _open_writer_arrays(
    writer: HSpecTableStoreWriter,
    descs: dict[str, Any],
) -> dict[str, np.memmap]:
    opened: dict[str, np.memmap] = {}
    try:
        for name, desc in descs.items():
            opened[name] = writer.open_memmap(desc, mode="r+")
        return opened
    except Exception:
        for arr in opened.values():
            _flush_close_memmap(arr)
        raise


def _close_writer_arrays(arrays: dict[str, np.memmap]) -> None:
    for arr in arrays.values():
        _flush_close_memmap(arr)


def build_prompt_table_to_store(
    *,
    prompt_id: str,
    descs: Sequence[HSpecTrajectoryDesc | dict[str, Any]],
    writer: HSpecTableStoreWriter,
    n_components: int,
    max_entries: int,
    pca_config: Optional[HSpecPCAConfig] = None,
    blas_threads: int = 1,
    wnd_size: int = 8,
    max_wnd: int = 28,
    min_wnd: int = 2,
) -> tuple[HSpecPromptTableDesc, HSpecPromptTableBuildMetrics]:
    """Build one prompt table directly into a versioned mmap table store."""
    t_total = time.perf_counter()
    metrics = HSpecPromptTableBuildMetrics()
    try:
        desc_list = _coerce_desc_list(descs)
        plan = _plan_prompt_table(desc_list, max_entries=max_entries)
        metrics.input_desc_count = len(desc_list)
        metrics.input_rows = int(plan.input_rows)
        metrics.included_desc_count = int(plan.rollout_count)
        metrics.included_rows = int(sum(item.length for item in plan.items))
        metrics.n_entries = int(plan.entry_count)
        metrics.n_rollouts = int(plan.rollout_count)
        metrics.token_count = int(plan.token_count)
        if plan.entry_count <= 0 or plan.rollout_count <= 0:
            raise HSpecTableBuildEmpty(
                f"HSpec table build requires at least one value-shifted entry "
                f"for prompt_id={prompt_id}"
            )

        config = pca_config or HSpecPCAConfig.from_env(int(n_components))
        if int(config.n_components) != int(n_components):
            config = replace(config, n_components=int(n_components))
        metrics.tile_rows = int(config.tile_rows)
        pca_descs = [item.desc for item in plan.items]
        pca_result = fit_prompt_pca_streaming(
            str(prompt_id),
            pca_descs,
            config=config,
            blas_threads=blas_threads,
        )
        pca_metrics = pca_result.metrics
        metrics.pca_method = str(pca_result.method)
        metrics.pca_mean_ms = float(pca_metrics.mean_ms)
        metrics.pca_basis_ms = float(pca_metrics.basis_ms)
        metrics.pca_total_ms = float(pca_metrics.total_ms)
        metrics.covariance_bytes = int(pca_metrics.covariance_bytes)
        metrics.randomized_rank = int(pca_metrics.randomized_rank)
        metrics.method_fallback_count = int(pca_metrics.method_fallback_count)
        metrics.processed_fp32_tile_bytes += int(
            pca_metrics.processed_fp32_tile_bytes)

        mean = np.ascontiguousarray(pca_result.mean, dtype=np.float32)
        components = np.ascontiguousarray(pca_result.components, dtype=np.float32)
        hidden_dim = int(mean.shape[0])
        comp_count = int(components.shape[0])
        metrics.hidden_dim = hidden_dim
        metrics.n_components = comp_count

        t_table_write = time.perf_counter()
        mean_desc = writer.reserve_array((hidden_dim,), "float32")
        components_desc = writer.reserve_array((comp_count, hidden_dim), "float32")
        keys_desc = writer.reserve_array((plan.entry_count, comp_count),
                                         config.keys_dtype)
        token_buffer_desc = writer.reserve_array((plan.token_count,), "int32")
        rollout_offset_desc = writer.reserve_array((plan.rollout_count,), "int64")
        rollout_len_desc = writer.reserve_array((plan.rollout_count,), "int32")
        entry_rollout_idx_desc = writer.reserve_array((plan.entry_count,), "int32")
        entry_offset_desc = writer.reserve_array((plan.entry_count,), "int32")
        rewards_desc = writer.reserve_array((plan.entry_count,), "float32")

        arrays = _open_writer_arrays(
            writer,
            {
                "mean": mean_desc,
                "components": components_desc,
                "keys": keys_desc,
                "token_buffer": token_buffer_desc,
                "rollout_offset": rollout_offset_desc,
                "rollout_len": rollout_len_desc,
                "entry_rollout_idx": entry_rollout_idx_desc,
                "entry_offset": entry_offset_desc,
                "rewards": rewards_desc,
            },
        )
        try:
            arrays["mean"][...] = mean
            arrays["components"][...] = components

            mean_fp32 = mean
            components_t = np.ascontiguousarray(components.T, dtype=np.float32)
            keys_dtype = np.dtype(config.keys_dtype)
            row_cursor = 0
            token_cursor = 0
            with _threadpool_context(blas_threads):
                for item in plan.items:
                    raw_hs = None
                    raw_tokens = None
                    try:
                        raw_hs, raw_tokens = load_hspec_trajectory(item.desc)
                        length = int(item.length)
                        n_add = int(item.n_entries)
                        rollout_idx = int(item.rollout_idx)
                        token_slice = np.asarray(raw_tokens[:length], dtype=np.int32)
                        arrays["token_buffer"][token_cursor:token_cursor + length] = token_slice
                        arrays["rollout_offset"][rollout_idx] = int(token_cursor)
                        arrays["rollout_len"][rollout_idx] = int(length)

                        row_end = row_cursor + n_add
                        arrays["entry_rollout_idx"][row_cursor:row_end] = rollout_idx
                        arrays["entry_offset"][row_cursor:row_end] = np.arange(
                            1,
                            n_add + 1,
                            dtype=np.int32,
                        )
                        arrays["rewards"][row_cursor:row_end] = float(item.reward)

                        for tile_start in range(0, n_add, int(config.tile_rows)):
                            tile_stop = min(tile_start + int(config.tile_rows), n_add)
                            t_projection = time.perf_counter()
                            h_tile = np.asarray(
                                raw_hs[tile_start:tile_stop],
                                dtype=np.float32,
                                order="C",
                            )
                            h_tile = np.ascontiguousarray(h_tile)
                            z = (h_tile - mean_fp32) @ components_t
                            metrics.projection_ms += float(
                                (time.perf_counter() - t_projection) * 1000.0)
                            keys_start = row_cursor + tile_start
                            keys_stop = row_cursor + tile_stop
                            arrays["keys"][keys_start:keys_stop] = z.astype(
                                keys_dtype,
                                copy=False,
                            )
                            metrics.projection_tile_count += 1
                            metrics.processed_fp32_tile_bytes += int(h_tile.nbytes)
                        row_cursor = row_end
                        token_cursor += length
                    finally:
                        if raw_tokens is not None:
                            _close_memmap(raw_tokens)
                        if raw_hs is not None:
                            _close_memmap(raw_hs)
            if row_cursor != int(plan.entry_count):
                raise RuntimeError(
                    f"HSpec table row cursor mismatch for prompt_id={prompt_id}: "
                    f"row_cursor={row_cursor} entries={plan.entry_count}"
                )
            if token_cursor != int(plan.token_count):
                raise RuntimeError(
                    f"HSpec table token cursor mismatch for prompt_id={prompt_id}: "
                    f"token_cursor={token_cursor} token_count={plan.token_count}"
                )
        finally:
            _close_writer_arrays(arrays)
            metrics.table_write_ms = max(
                float((time.perf_counter() - t_table_write) * 1000.0)
                - float(metrics.projection_ms),
                0.0,
            )

        table_desc = HSpecPromptTableDesc(
            schema_version=TABLE_STORE_SCHEMA_VERSION,
            prompt_id=str(prompt_id),
            version=int(writer.version),
            shard_id=int(writer.shard_id),
            table_file=str(writer.table_file),
            n_entries=int(plan.entry_count),
            n_rollouts=int(plan.rollout_count),
            hidden_dim=hidden_dim,
            n_components=comp_count,
            n_samples=int(pca_result.n_samples),
            pca_method=str(pca_result.method),
            mean=mean_desc,
            components=components_desc,
            keys=keys_desc,
            token_buffer=token_buffer_desc,
            rollout_token_offset=rollout_offset_desc,
            rollout_token_len=rollout_len_desc,
            entry_rollout_idx=entry_rollout_idx_desc,
            entry_offset=entry_offset_desc,
            rewards=rewards_desc,
            wnd_size=int(wnd_size),
            max_wnd=int(max_wnd),
            min_wnd=int(min_wnd),
            created_time_ns=time.time_ns(),
        )
        writer.commit_prompt(table_desc)
        metrics.total_ms = float((time.perf_counter() - t_total) * 1000.0)
        _record_table_build_metrics(metrics)
        return table_desc, metrics
    except MemoryError:
        metrics.memory_error_count += 1
        metrics.build_error_count += 1
        metrics.total_ms = float((time.perf_counter() - t_total) * 1000.0)
        _record_table_build_metrics(metrics)
        raise
    except Exception:
        metrics.build_error_count += 1
        metrics.total_ms = float((time.perf_counter() - t_total) * 1000.0)
        _record_table_build_metrics(metrics)
        raise
