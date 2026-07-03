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
This module intentionally stays off the decode hot path. Step 1 only provides
descriptor, manifest, and writer/reader skeletons so later Phase 2 steps can
move active HSpec tables out of Ray actor heap without changing build/query
behavior yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from vllm_ascend.spec_decode.hspec_store import (
    get_hspec_table_store_root,
    hspec_record_store_metric,
    hspec_record_store_metric_max,
    hspec_strict_descriptor_mode_enabled,
)

logger = logging.getLogger(__name__)

TABLE_STORE_SCHEMA_VERSION = 1
TABLE_BIN_NAME = "table.bin"
PROMPT_INDEX_NAME = "prompt_index.jsonl"
VERSION_MANIFEST_NAME = "manifest.json"
ACTIVE_VERSION_NAME = "active_version.json"
DEFAULT_ALIGN_BYTES = 4096

_VALID_PREFETCH_MODES = frozenset({"descriptor", "legacy_arrays"})
_VALID_PCA_METHODS = frozenset({"randomized", "covariance", "auto", "svd_reference"})
_VALID_PCA_ACCUM_DTYPES = frozenset({"float32", "float64"})
_VALID_TABLE_KEY_DTYPES = frozenset({"float16"})


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            logger.debug("Failed to remove temporary HSpec table manifest %s",
                         tmp,
                         exc_info=True)


def _parse_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%s; using %s", name, value, default)
        return int(default)
    if parsed <= 0:
        logger.warning("%s=%s must be > 0; using %s", name, value, default)
        return int(default)
    return parsed


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%s; using %s", name, value, default)
        return max(int(default), 0)
    if parsed < 0:
        logger.warning("%s=%s must be >= 0; using %s", name, value, default)
        return max(int(default), 0)
    return parsed


def _parse_choice_env(name: str, default: str, valid: frozenset[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value in valid:
        return value
    logger.warning("Ignoring invalid %s=%s; using %s", name, value, default)
    return default


def get_hspec_pca_method() -> str:
    return _parse_choice_env("HSPEC_PCA_METHOD", "randomized", _VALID_PCA_METHODS)


def get_hspec_pca_tile_rows() -> int:
    return _parse_positive_int_env("HSPEC_PCA_TILE_ROWS", 1024)


def get_hspec_pca_random_oversample() -> int:
    return _parse_nonnegative_int_env("HSPEC_PCA_RANDOM_OVERSAMPLE", 16)


def get_hspec_pca_random_seed() -> int:
    value = os.getenv("HSPEC_PCA_RANDOM_SEED", "202405")
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_PCA_RANDOM_SEED=%s; using 202405",
                       value)
        return 202405


def get_hspec_pca_cov_max_bytes() -> int:
    return _parse_nonnegative_int_env("HSPEC_PCA_COV_MAX_BYTES", 134217728)


def get_hspec_pca_accum_dtype() -> str:
    return _parse_choice_env("HSPEC_PCA_ACCUM_DTYPE", "float32",
                             _VALID_PCA_ACCUM_DTYPES)


def get_hspec_table_keys_dtype() -> str:
    return _parse_choice_env("HSPEC_TABLE_KEYS_DTYPE", "float16",
                             _VALID_TABLE_KEY_DTYPES)


def get_hspec_table_file_align_bytes() -> int:
    return _parse_positive_int_env("HSPEC_TABLE_FILE_ALIGN_BYTES",
                                   DEFAULT_ALIGN_BYTES)


def get_hspec_table_prefetch_mode() -> str:
    return _parse_choice_env("HSPEC_TABLE_PREFETCH_MODE", "descriptor",
                             _VALID_PREFETCH_MODES)


def get_hspec_table_store_retain_versions() -> int:
    return _parse_positive_int_env("HSPEC_TABLE_STORE_RETAIN_VERSIONS", 2)


def hspec_table_store_gc_after_swap_enabled() -> bool:
    return os.getenv("HSPEC_TABLE_STORE_GC_AFTER_SWAP", "1") != "0"


def hspec_table_store_fsync_on_seal_enabled() -> bool:
    return os.getenv("HSPEC_TABLE_STORE_FSYNC_ON_SEAL", "0") != "0"


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int):
        values = (shape,)
    else:
        try:
            values = tuple(shape)
        except TypeError as exc:
            raise TypeError(f"shape must be an int or iterable of ints, got {shape!r}") from exc
    normalized: list[int] = []
    for dim in values:
        try:
            parsed = int(dim)
        except Exception as exc:
            raise TypeError(f"shape dimension must be int-like, got {dim!r}") from exc
        if parsed < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {values!r}")
        normalized.append(parsed)
    return tuple(normalized)


def _array_nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return int(count) * int(dtype.itemsize)


def _align_up(value: int, align: int) -> int:
    if align <= 0:
        align = DEFAULT_ALIGN_BYTES
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


@dataclass(frozen=True)
class HSpecArrayDesc:
    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: str
    order: str = "C"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("HSpecArrayDesc.path must be a non-empty string")
        if int(self.offset) < 0:
            raise ValueError(f"HSpecArrayDesc.offset must be >= 0, got {self.offset}")
        shape = _normalize_shape(self.shape)
        dtype = np.dtype(self.dtype)
        order = str(self.order)
        if order != "C":
            raise ValueError(f"HSpecArrayDesc.order must be 'C', got {order!r}")
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "order", order)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "HSpecArrayDesc":
        if not isinstance(obj, dict):
            raise TypeError(f"HSpecArrayDesc.from_dict expects dict, got {type(obj)!r}")
        return cls(
            path=str(obj["path"]),
            offset=int(obj["offset"]),
            shape=_normalize_shape(obj["shape"]),
            dtype=str(obj["dtype"]),
            order=str(obj.get("order", "C")),
        )


def coerce_array_desc(obj: Any, *, field_name: str = "array") -> HSpecArrayDesc:
    if isinstance(obj, HSpecArrayDesc):
        return obj
    if isinstance(obj, dict):
        try:
            return HSpecArrayDesc.from_dict(obj)
        except Exception as exc:
            raise type(exc)(f"{field_name}: {exc}") from exc
    if isinstance(obj, np.ndarray):
        raise TypeError(f"{field_name}: ndarray is not a valid HSpecArrayDesc")
    raise TypeError(f"{field_name}: expected HSpecArrayDesc or dict, got {type(obj)!r}")


@dataclass(frozen=True)
class HSpecPromptTableDesc:
    schema_version: int
    prompt_id: str
    version: int
    shard_id: int
    table_file: str
    n_entries: int
    n_rollouts: int
    hidden_dim: int
    n_components: int
    n_samples: int
    pca_method: str
    mean: HSpecArrayDesc
    components: HSpecArrayDesc
    keys: HSpecArrayDesc
    token_buffer: HSpecArrayDesc
    rollout_token_offset: HSpecArrayDesc
    rollout_token_len: HSpecArrayDesc
    entry_rollout_idx: HSpecArrayDesc
    entry_offset: HSpecArrayDesc
    rewards: Optional[HSpecArrayDesc]
    wnd_size: int = 8
    max_wnd: int = 28
    min_wnd: int = 2
    created_time_ns: int = 0

    def __post_init__(self) -> None:
        if int(self.schema_version) != TABLE_STORE_SCHEMA_VERSION:
            raise ValueError(
                f"HSpecPromptTableDesc schema_version must be {TABLE_STORE_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("HSpecPromptTableDesc.prompt_id must be non-empty")
        if int(self.version) < 0:
            raise ValueError("HSpecPromptTableDesc.version must be >= 0")
        if int(self.shard_id) < 0:
            raise ValueError("HSpecPromptTableDesc.shard_id must be >= 0")
        if not isinstance(self.table_file, str) or not self.table_file:
            raise ValueError("HSpecPromptTableDesc.table_file must be non-empty")
        for name in ("n_entries", "n_rollouts", "n_samples"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"HSpecPromptTableDesc.{name} must be >= 0")
        for name in ("hidden_dim", "n_components"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"HSpecPromptTableDesc.{name} must be > 0")
        if not isinstance(self.pca_method, str) or not self.pca_method:
            raise ValueError("HSpecPromptTableDesc.pca_method must be non-empty")
        if int(self.min_wnd) <= 0 or int(self.max_wnd) < int(self.min_wnd):
            raise ValueError(
                "HSpecPromptTableDesc window bounds are invalid: "
                f"min_wnd={self.min_wnd} max_wnd={self.max_wnd}"
            )
        wnd_size = max(int(self.min_wnd), min(int(self.wnd_size), int(self.max_wnd)))

        array_fields = (
            "mean",
            "components",
            "keys",
            "token_buffer",
            "rollout_token_offset",
            "rollout_token_len",
            "entry_rollout_idx",
            "entry_offset",
        )
        for field in array_fields:
            object.__setattr__(
                self,
                field,
                coerce_array_desc(getattr(self, field), field_name=field),
            )
        if self.rewards is not None:
            object.__setattr__(
                self,
                "rewards",
                coerce_array_desc(self.rewards, field_name="rewards"),
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "shard_id", int(self.shard_id))
        object.__setattr__(self, "n_entries", int(self.n_entries))
        object.__setattr__(self, "n_rollouts", int(self.n_rollouts))
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))
        object.__setattr__(self, "n_components", int(self.n_components))
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "wnd_size", wnd_size)
        object.__setattr__(self, "max_wnd", int(self.max_wnd))
        object.__setattr__(self, "min_wnd", int(self.min_wnd))
        object.__setattr__(self, "created_time_ns", int(self.created_time_ns))

        table_file = str(Path(self.table_file).resolve())
        for field in array_fields + (("rewards",) if self.rewards is not None else ()):
            arr = getattr(self, field)
            if arr is not None and str(Path(arr.path).resolve()) != table_file:
                hspec_record_store_metric("table_store_descriptor_path_mismatch", 1)
                message = (
                    "HSpecPromptTableDesc array path differs from table_file: "
                    f"prompt_id={self.prompt_id!r} version={self.version} "
                    f"shard_id={self.shard_id} field={field} "
                    f"path={arr.path!r} table_file={self.table_file!r}"
                )
                if hspec_strict_descriptor_mode_enabled():
                    hspec_record_store_metric("strict_descriptor_violation", 1)
                    raise ValueError(message)
                logger.warning(message)

    def to_dict(self) -> dict[str, Any]:
        return prompt_table_desc_to_dict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "HSpecPromptTableDesc":
        return coerce_prompt_table_desc(obj)


def prompt_table_desc_to_dict(desc: HSpecPromptTableDesc) -> dict[str, Any]:
    if not isinstance(desc, HSpecPromptTableDesc):
        desc = coerce_prompt_table_desc(desc)
    payload: dict[str, Any] = {
        "schema_version": int(desc.schema_version),
        "prompt_id": str(desc.prompt_id),
        "version": int(desc.version),
        "shard_id": int(desc.shard_id),
        "table_file": str(desc.table_file),
        "n_entries": int(desc.n_entries),
        "n_rollouts": int(desc.n_rollouts),
        "hidden_dim": int(desc.hidden_dim),
        "n_components": int(desc.n_components),
        "n_samples": int(desc.n_samples),
        "pca_method": str(desc.pca_method),
        "mean": desc.mean.to_dict(),
        "components": desc.components.to_dict(),
        "keys": desc.keys.to_dict(),
        "token_buffer": desc.token_buffer.to_dict(),
        "rollout_token_offset": desc.rollout_token_offset.to_dict(),
        "rollout_token_len": desc.rollout_token_len.to_dict(),
        "entry_rollout_idx": desc.entry_rollout_idx.to_dict(),
        "entry_offset": desc.entry_offset.to_dict(),
        "rewards": desc.rewards.to_dict() if desc.rewards is not None else None,
        "wnd_size": int(desc.wnd_size),
        "max_wnd": int(desc.max_wnd),
        "min_wnd": int(desc.min_wnd),
        "created_time_ns": int(desc.created_time_ns),
    }
    return payload


def coerce_prompt_table_desc(obj: Any) -> HSpecPromptTableDesc:
    if isinstance(obj, HSpecPromptTableDesc):
        return obj
    if not isinstance(obj, dict):
        if isinstance(obj, np.ndarray):
            raise TypeError("ndarray is not a valid HSpecPromptTableDesc")
        raise TypeError(f"Expected HSpecPromptTableDesc or dict, got {type(obj)!r}")
    required_fields = (
        "schema_version",
        "prompt_id",
        "version",
        "shard_id",
        "table_file",
        "n_entries",
        "n_rollouts",
        "hidden_dim",
        "n_components",
        "n_samples",
        "pca_method",
        "mean",
        "components",
        "keys",
        "token_buffer",
        "rollout_token_offset",
        "rollout_token_len",
        "entry_rollout_idx",
        "entry_offset",
        "rewards",
    )
    missing = [name for name in required_fields if name not in obj]
    if missing:
        raise ValueError(f"HSpecPromptTableDesc missing fields: {missing}")
    return HSpecPromptTableDesc(
        schema_version=int(obj["schema_version"]),
        prompt_id=str(obj["prompt_id"]),
        version=int(obj["version"]),
        shard_id=int(obj["shard_id"]),
        table_file=str(obj["table_file"]),
        n_entries=int(obj["n_entries"]),
        n_rollouts=int(obj["n_rollouts"]),
        hidden_dim=int(obj["hidden_dim"]),
        n_components=int(obj["n_components"]),
        n_samples=int(obj["n_samples"]),
        pca_method=str(obj["pca_method"]),
        mean=coerce_array_desc(obj["mean"], field_name="mean"),
        components=coerce_array_desc(obj["components"], field_name="components"),
        keys=coerce_array_desc(obj["keys"], field_name="keys"),
        token_buffer=coerce_array_desc(obj["token_buffer"], field_name="token_buffer"),
        rollout_token_offset=coerce_array_desc(
            obj["rollout_token_offset"], field_name="rollout_token_offset"),
        rollout_token_len=coerce_array_desc(
            obj["rollout_token_len"], field_name="rollout_token_len"),
        entry_rollout_idx=coerce_array_desc(
            obj["entry_rollout_idx"], field_name="entry_rollout_idx"),
        entry_offset=coerce_array_desc(obj["entry_offset"], field_name="entry_offset"),
        rewards=(coerce_array_desc(obj["rewards"], field_name="rewards")
                 if obj.get("rewards") is not None else None),
        wnd_size=int(obj.get("wnd_size", 8)),
        max_wnd=int(obj.get("max_wnd", 28)),
        min_wnd=int(obj.get("min_wnd", 2)),
        created_time_ns=int(obj.get("created_time_ns", 0)),
    )


def open_array(desc_obj: HSpecArrayDesc | dict[str, Any], mode: str = "r") -> np.memmap:
    desc = coerce_array_desc(desc_obj)
    if mode not in {"r", "r+", "w+"}:
        raise ValueError(f"Unsupported HSpec table array mmap mode: {mode!r}")
    dtype = np.dtype(desc.dtype)
    if desc.offset % dtype.itemsize != 0:
        raise ValueError(
            f"HSpec array offset must be aligned to dtype itemsize: "
            f"offset={desc.offset}, itemsize={dtype.itemsize}"
        )
    nbytes = _array_nbytes(desc.shape, dtype)
    if nbytes <= 0:
        raise ValueError(f"HSpec array has zero bytes: shape={desc.shape} dtype={dtype}")
    path = Path(desc.path)
    if mode in {"r", "r+"}:
        if not path.exists():
            raise FileNotFoundError(str(path))
        size = path.stat().st_size
        need = int(desc.offset) + int(nbytes)
        if size < need:
            raise ValueError(f"HSpec table file too small: path={path} size={size} need={need}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(
        str(path),
        dtype=dtype,
        mode=mode,
        offset=int(desc.offset),
        shape=tuple(desc.shape),
        order=desc.order,
    )


def _close_memmap(array: np.memmap) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _copy_array(desc: HSpecArrayDesc,
                *,
                dtype: np.dtype | type | str | None = None) -> np.ndarray:
    mmap_arr = open_array(desc)
    try:
        return np.array(mmap_arr, dtype=dtype, copy=True)
    finally:
        _close_memmap(mmap_arr)


def materialize_prompt_table(desc_obj: HSpecPromptTableDesc | dict[str, Any]) -> dict[str, Any]:
    """Materialize table arrays for debug or legacy prefetch fallback only.

    This copies table arrays into process memory and must not be used in the
    build hot path or decode hot loop.
    """
    desc = coerce_prompt_table_desc(desc_obj)
    hspec_record_store_metric("table_store_materialize_count", 1)

    mean = _copy_array(desc.mean, dtype=np.float32)
    components = _copy_array(desc.components, dtype=np.float32)
    keys = _copy_array(desc.keys)
    entry_rollout_idx = _copy_array(desc.entry_rollout_idx, dtype=np.int32)
    entry_offset = _copy_array(desc.entry_offset, dtype=np.int32)

    token_buffer = _copy_array(desc.token_buffer, dtype=np.int32)
    rollout_offsets = _copy_array(desc.rollout_token_offset, dtype=np.int64)
    rollout_lens = _copy_array(desc.rollout_token_len, dtype=np.int32)
    rollout_seqs: list[np.ndarray] = []
    for offset, length in zip(rollout_offsets.tolist(), rollout_lens.tolist()):
        start = int(offset)
        end = start + int(length)
        rollout_seqs.append(np.ascontiguousarray(token_buffer[start:end],
                                                 dtype=np.int32))

    result = {
        "mean": mean,
        "components": components,
        "keys": keys,
        "rollout_seqs": rollout_seqs,
        "entry_rollout_idx": entry_rollout_idx,
        "entry_offset": entry_offset,
        "rewards": (
            _copy_array(desc.rewards, dtype=np.float32)
            if desc.rewards is not None
            else None
        ),
        "n_entries": int(desc.n_entries),
        "wnd_size": int(desc.wnd_size),
        "max_wnd": int(desc.max_wnd),
        "min_wnd": int(desc.min_wnd),
    }
    return result


class HSpecTableStoreWriter:
    """Append-only writer skeleton for one shard/version table.bin."""

    def __init__(
        self,
        root: str | Path | None = None,
        shard_id: int = 0,
        version: int = 0,
        *,
        align_bytes: int | None = None,
    ) -> None:
        self.root = Path(root or get_hspec_table_store_root()).resolve()
        self.shard_id = int(shard_id)
        self.version = int(version)
        if self.shard_id < 0:
            raise ValueError(f"shard_id must be >= 0, got {self.shard_id}")
        if self.version < 0:
            raise ValueError(f"version must be >= 0, got {self.version}")
        self.align_bytes = int(align_bytes or get_hspec_table_file_align_bytes())
        if self.align_bytes <= 0:
            self.align_bytes = DEFAULT_ALIGN_BYTES
        self.shard_root = self.root / f"shard_{self.shard_id:03d}"
        self.version_dir = self.shard_root / f"version_{self.version:06d}"
        self.table_file = self.version_dir / TABLE_BIN_NAME
        self.prompt_index_path = self.version_dir / PROMPT_INDEX_NAME
        self.manifest_path = self.version_dir / VERSION_MANIFEST_NAME
        self.version_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._offset = self.table_file.stat().st_size if self.table_file.exists() else 0
        self._bytes_reserved = 0
        self._bytes_committed = 0
        self._prompt_index: dict[str, HSpecPromptTableDesc] = {}
        self._sealed = False
        self._manifest: Optional[dict[str, Any]] = None
        self._created_time_ns = time.time_ns()

    def reserve_array(self, shape: tuple[int, ...], dtype: str) -> HSpecArrayDesc:
        dtype_np = np.dtype(dtype)
        shape_t = _normalize_shape(shape)
        nbytes = _array_nbytes(shape_t, dtype_np)
        if nbytes <= 0:
            raise ValueError(f"Cannot reserve zero-byte HSpec table array: {shape_t}, {dtype_np}")
        with self._lock:
            if self._sealed:
                raise RuntimeError("Cannot reserve array after HSpec table writer is sealed")
            offset = _align_up(self._offset, max(self.align_bytes, dtype_np.itemsize))
            end = offset + nbytes
            self.table_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.table_file, "ab") as f:
                f.truncate(end)
            self._offset = end
            self._bytes_reserved += nbytes
        hspec_record_store_metric("table_store_reserved_bytes", nbytes)
        hspec_record_store_metric("table_store_array_descriptor_count", 1)
        return HSpecArrayDesc(
            path=str(self.table_file),
            offset=offset,
            shape=shape_t,
            dtype=dtype_np.name,
        )

    def open_memmap(self,
                    array_desc: HSpecArrayDesc | dict[str, Any],
                    mode: str = "r+") -> np.memmap:
        desc = coerce_array_desc(array_desc)
        if Path(desc.path).resolve() != self.table_file.resolve():
            raise ValueError(
                f"Array path does not belong to this writer: {desc.path} != {self.table_file}"
            )
        return open_array(desc, mode=mode)

    def commit_prompt(self, desc_obj: HSpecPromptTableDesc | dict[str, Any]) -> None:
        desc = coerce_prompt_table_desc(desc_obj)
        with self._lock:
            if self._sealed:
                raise RuntimeError("Cannot commit prompt after HSpec table writer is sealed")
            if desc.prompt_id in self._prompt_index:
                raise ValueError(f"Duplicate HSpec prompt table commit: {desc.prompt_id}")
            if desc.shard_id != self.shard_id or desc.version != self.version:
                raise ValueError(
                    "Prompt table descriptor targets a different shard/version: "
                    f"desc=({desc.shard_id}, {desc.version}) writer=({self.shard_id}, {self.version})"
                )
            self.prompt_index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.prompt_index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(prompt_table_desc_to_dict(desc), ensure_ascii=False) + "\n")
                f.flush()
            self._prompt_index[desc.prompt_id] = desc
            self._bytes_committed += _prompt_desc_live_bytes(desc)
        hspec_record_store_metric("table_store_committed_prompts", 1)
        hspec_record_store_metric("table_store_descriptor_count", 1)

    def seal(self, extra_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._sealed and self._manifest is not None:
                return dict(self._manifest)
            fsync_on_seal = hspec_table_store_fsync_on_seal_enabled()
            if fsync_on_seal:
                for path in (self.table_file, self.prompt_index_path):
                    if not path.exists():
                        continue
                    with open(path, "ab") as f:
                        f.flush()
                        os.fsync(f.fileno())
                        hspec_record_store_metric("table_store_fsync_count", 1)
            entry_count = sum(desc.n_entries for desc in self._prompt_index.values())
            manifest: dict[str, Any] = {
                "schema_version": TABLE_STORE_SCHEMA_VERSION,
                "status": "sealed",
                "version": int(self.version),
                "shard_id": int(self.shard_id),
                "version_dir": str(self.version_dir),
                "manifest_path": str(self.manifest_path),
                "table_file": str(self.table_file),
                "prompt_index_path": str(self.prompt_index_path),
                "prompt_count": len(self._prompt_index),
                "entry_count": int(entry_count),
                "bytes_reserved": int(self._bytes_reserved),
                "bytes_committed": int(self._bytes_committed),
                "table_file_size": int(self.table_file.stat().st_size) if self.table_file.exists() else 0,
                "align_bytes": int(self.align_bytes),
                "created_time_ns": int(self._created_time_ns),
                "sealed_time_ns": time.time_ns(),
                "fsync_on_seal": bool(fsync_on_seal),
            }
            if extra_metrics:
                manifest.update(extra_metrics)
            try:
                _write_json_atomic(self.manifest_path, manifest)
            except Exception:
                hspec_record_store_metric("table_store_manifest_write_error", 1)
                raise
            hspec_record_store_metric("table_store_bytes_written",
                                      int(manifest["table_file_size"]))
            hspec_record_store_metric("table_store_prompt_count",
                                      int(manifest["prompt_count"]))
            hspec_record_store_metric("table_store_entry_count",
                                      int(manifest["entry_count"]))
            hspec_record_store_metric_max("table_store_version",
                                          int(self.version))
            self._sealed = True
            self._manifest = dict(manifest)
        return dict(manifest)

    @property
    def prompt_index(self) -> dict[str, HSpecPromptTableDesc]:
        return dict(self._prompt_index)


class HSpecTableStoreReader:
    """Lightweight metadata reader for one shard table store."""

    def __init__(self, root: str | Path | None = None, shard_id: int = 0) -> None:
        self.root = Path(root or get_hspec_table_store_root()).resolve()
        self.shard_id = int(shard_id)
        if self.shard_id < 0:
            raise ValueError(f"shard_id must be >= 0, got {self.shard_id}")
        self.shard_root = self.root / f"shard_{self.shard_id:03d}"
        self.active_version_path = self.shard_root / ACTIVE_VERSION_NAME

    def read_active_manifest(self) -> dict[str, Any] | None:
        if not self.active_version_path.exists():
            return None
        try:
            with open(self.active_version_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            hspec_record_store_metric("table_store_reader_load_error", 1)
            raise

    def _version_dir(self, version: int) -> Path:
        return self.shard_root / f"version_{int(version):06d}"

    def load_version_manifest(self, version: int | None = None) -> dict[str, Any]:
        if version is None:
            active = self.read_active_manifest()
            if active is None:
                raise FileNotFoundError(str(self.active_version_path))
            manifest_path = Path(active["manifest_path"])
        else:
            manifest_path = self._version_dir(int(version)) / VERSION_MANIFEST_NAME
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            hspec_record_store_metric("table_store_reader_load_error", 1)
            raise

    def load_prompt_index(self, version: int | None = None) -> dict[str, HSpecPromptTableDesc]:
        if version is None:
            manifest = self.load_version_manifest(None)
        else:
            manifest = self.load_version_manifest(int(version))
        index_path = Path(manifest["prompt_index_path"])
        result: dict[str, HSpecPromptTableDesc] = {}
        if not index_path.exists():
            return result
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    desc = coerce_prompt_table_desc(json.loads(line))
                    result[desc.prompt_id] = desc
        except Exception:
            hspec_record_store_metric("table_store_reader_load_error", 1)
            raise
        return result

    def load_active_index(self) -> dict[str, HSpecPromptTableDesc]:
        active = self.read_active_manifest()
        if active is None:
            return {}
        return self.load_prompt_index(int(active["active_version"]))

    def get_prompt(self, prompt_id: str) -> HSpecPromptTableDesc | None:
        return self.load_active_index().get(str(prompt_id))

    def materialize_for_legacy_prefetch(self, prompt_id: str) -> dict[str, Any] | None:
        desc = self.get_prompt(prompt_id)
        if desc is None:
            return None
        return materialize_prompt_table(desc)


def write_active_version_manifest(
    shard_root: str | Path,
    *,
    active_version: int,
    shard_id: int,
    version_dir: str | Path,
    manifest_path: str | Path,
    prompt_count: int,
    entry_count: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": TABLE_STORE_SCHEMA_VERSION,
        "active_version": int(active_version),
        "shard_id": int(shard_id),
        "version_dir": str(version_dir),
        "manifest_path": str(manifest_path),
        "prompt_count": int(prompt_count),
        "entry_count": int(entry_count),
        "updated_time_ns": time.time_ns(),
    }
    try:
        _write_json_atomic(Path(shard_root) / ACTIVE_VERSION_NAME, payload)
    except Exception:
        hspec_record_store_metric("table_store_active_manifest_write_error", 1)
        raise
    return payload


def list_table_store_versions(
    root: str | Path | None = None,
    shard_id: int = 0,
) -> list[int]:
    """Return sealed/building version ids present for one shard.

    This helper is intentionally low-frequency: it is used when opening a new
    building writer or after swap for GC, never on the decode hot path.
    """
    base_root = Path(root or get_hspec_table_store_root()).resolve()
    shard_root = base_root / f"shard_{int(shard_id):03d}"
    if not shard_root.exists():
        return []
    versions: list[int] = []
    for child in shard_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("version_"):
            continue
        suffix = name[len("version_"):]
        if not suffix.isdigit():
            continue
        versions.append(int(suffix))
    return sorted(versions)


def _read_version_status(version_dir: Path) -> str:
    manifest_path = version_dir / VERSION_MANIFEST_NAME
    if not manifest_path.exists():
        return ""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return str(manifest.get("status", ""))
    except Exception:
        hspec_record_store_metric("table_store_reader_load_error", 1)
        logger.debug("Failed to read HSpec table version manifest %s",
                     manifest_path,
                     exc_info=True)
        return ""


def gc_table_store_versions(
    root: str | Path | None = None,
    shard_id: int = 0,
    *,
    active_version: int,
    retain_versions: int,
) -> dict[str, Any]:
    """Best-effort GC for old table-store versions.

    Active and recent versions are always retained. Delete failures are
    reported as metrics and never raised, which keeps Windows mmap lifecycle
    quirks from blocking the training loop.
    """
    base_root = Path(root or get_hspec_table_store_root()).resolve()
    shard_id_i = int(shard_id)
    active_version_i = int(active_version)
    retain_versions_i = max(int(retain_versions), 1)
    shard_root = base_root / f"shard_{shard_id_i:03d}"
    versions = list_table_store_versions(base_root, shard_id_i)
    hspec_record_store_metric("table_store_gc_scanned_versions", len(versions))
    if not versions:
        return {
            "scanned_versions": 0,
            "retained_versions": 0,
            "deleted_versions": 0,
            "delete_errors": 0,
            "kept": [],
            "deleted": [],
        }

    statuses: dict[int, str] = {
        version: _read_version_status(shard_root / f"version_{version:06d}")
        for version in versions
    }
    recent_candidates = [
        version for version in versions
        if statuses.get(version, "") != "gc_deletable"
    ]
    recent = set(recent_candidates[-retain_versions_i:])

    deleted: list[int] = []
    delete_errors = 0
    for version in versions:
        version_dir = shard_root / f"version_{version:06d}"
        status = statuses.get(version, "")
        if active_version_i > 0 and version == active_version_i:
            continue
        if status != "gc_deletable" and (
            version in recent or version > active_version_i
        ):
            continue
        if status and status not in {"sealed", "gc_deletable"}:
            continue
        try:
            shutil.rmtree(version_dir)
            deleted.append(version)
            hspec_record_store_metric("table_store_gc_deleted_versions", 1)
        except Exception:
            delete_errors += 1
            hspec_record_store_metric("table_store_gc_delete_error", 1)
            logger.debug("Failed to GC HSpec table store version %s",
                         version_dir,
                         exc_info=True)

    retained_count = len([version for version in versions if version not in deleted])
    hspec_record_store_metric("table_store_gc_retained_versions", retained_count)
    return {
        "scanned_versions": len(versions),
        "retained_versions": retained_count,
        "deleted_versions": len(deleted),
        "delete_errors": delete_errors,
        "kept": sorted(version for version in versions if version not in deleted),
        "deleted": deleted,
    }


def clear_active_version_manifest(shard_root: str | Path) -> None:
    """Remove active_version.json if present.

    Clear is a management/debug operation. Failure should not leave the actor
    unusable, but it is observable through metrics.
    """
    active_path = Path(shard_root) / ACTIVE_VERSION_NAME
    try:
        if active_path.exists():
            active_path.unlink()
    except Exception:
        hspec_record_store_metric("table_store_active_manifest_clear_error", 1)
        logger.debug("Failed to clear HSpec active version manifest %s",
                     active_path,
                     exc_info=True)


def _prompt_desc_live_bytes(desc: HSpecPromptTableDesc) -> int:
    arrays = [
        desc.mean,
        desc.components,
        desc.keys,
        desc.token_buffer,
        desc.rollout_token_offset,
        desc.rollout_token_len,
        desc.entry_rollout_idx,
        desc.entry_offset,
    ]
    if desc.rewards is not None:
        arrays.append(desc.rewards)
    total = 0
    for arr in arrays:
        total += _array_nbytes(arr.shape, np.dtype(arr.dtype))
    return total
