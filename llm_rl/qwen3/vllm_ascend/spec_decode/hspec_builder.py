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
"""
This module is intentionally CPU/NumPy only. It reads descriptor-backed raw
hidden-state mmap files by tile, computes prompt-level PCA parameters, and
does not touch Ray actors, torch tensors, NPU streams, table-store writers, or
the decode hot path.
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


def _close_memmap(array: Any) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


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
