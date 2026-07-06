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
"""Local HSpec trajectory store for descriptor-based table building.

Phase 1 keeps hidden-state bytes out of ``DataProto`` and Ray object store.
Each rollout worker writes request-local binary segments and returns only a
small, immutable descriptor to the trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SEGMENT_MANIFEST_NAME = "segment.json"
_DESC_MANIFEST_NAME = "desc.jsonl"
_SEGMENT_SAFE_DELETE_STATES = frozenset({"gc_deletable", "aborted", "epoch_build_done"})
_SEGMENT_CONFIRMED_DELETE_STATES = _SEGMENT_SAFE_DELETE_STATES | frozenset({"sealed", "submitted_to_build"})
_unsafe_delete_trajectory_warned = False

_store_metrics_lock = threading.Lock()
_store_metrics: Dict[str, int] = {
    "raw_store_bytes": 0,
    "desc_count": 0,
    "collect_dropped": 0,
    "strict_descriptor_violation": 0,
    "legacy_payload_count": 0,
    "descriptor_payload_count": 0,
    "validation_collect_skip": 0,
    "collect_dropped_empty": 0,
    "collect_dropped_invalid_dim": 0,
    "collect_dropped_missing_offset": 0,
    "collect_dropped_align_mismatch": 0,
    "store_fp16_rows": 0,
    "source_dtype_fp16_rows": 0,
    "source_dtype_bf16_rows": 0,
    "source_dtype_other_rows": 0,
    "segment_sealed": 0,
    "segment_sealed_empty": 0,
    "segment_rotated": 0,
    "segment_manifest_write_error": 0,
    "segment_fsync_count": 0,
    "segment_close_error": 0,
    "segment_aborted": 0,
    "raw_store_budget_bytes": 0,
    "raw_store_budget_files": 0,
    "raw_store_budget_over_bytes": 0,
    "raw_store_budget_over_files": 0,
    "raw_store_budget_gc_skipped": 0,
    "raw_store_budget_gc_deleted": 0,
    "raw_store_epoch_bytes": 0,
    "raw_store_epoch_budget_bytes": 0,
    "raw_store_collect_budget_blocked": 0,
    "raw_store_collect_budget_unblocked": 0,
    "raw_store_collect_drop_bytes": 0,
    "raw_store_budget_active": 0,
    "collect_dropped_budget_worker_bytes": 0,
    "collect_dropped_budget_epoch_bytes": 0,
    "collect_dropped_raw_store_over_budget": 0,
    "unsafe_descriptor_cleanup_suppressed": 0,
    "segment_delete_count": 0,
    "segment_delete_bytes": 0,
    "segment_delete_error": 0,
    "raw_store_epoch_gc_segments": 0,
    "raw_store_epoch_gc_deleted": 0,
    "raw_store_epoch_gc_skipped": 0,
    "raw_store_epoch_gc_error": 0,
    "build_submission_count": 0,
    "build_submission_segments": 0,
    "topology_actor_count": 0,
    "topology_actor_init_error": 0,
    "topology_actor_reuse_mismatch": 0,
    "topology_actor_shard_mismatch": 0,
    "topology_actor_num_groups_mismatch": 0,
    "topology_actor_node_mismatch": 0,
    "descriptor_topology_violation": 0,
    "descriptor_node_mismatch": 0,
    "descriptor_shard_mismatch": 0,
    "descriptor_prompt_mismatch": 0,
    "descriptor_shard_normalized": 0,
    "table_store_descriptor_count": 0,
    "table_store_array_descriptor_count": 0,
    "table_store_reserved_bytes": 0,
    "table_store_committed_prompts": 0,
    "table_store_manifest_write_error": 0,
    "table_store_active_manifest_write_error": 0,
    "table_store_fsync_count": 0,
    "table_store_reader_load_error": 0,
    "table_store_materialize_count": 0,
    "table_store_stale_version_error": 0,
    "table_store_descriptor_path_mismatch": 0,
    "table_store_bytes_written": 0,
    "table_store_prompt_count": 0,
    "table_store_entry_count": 0,
    "table_store_version": 0,
    "table_swap_empty_count": 0,
    "table_store_gc_scanned_versions": 0,
    "table_store_gc_retained_versions": 0,
    "table_store_gc_deleted_versions": 0,
    "table_store_gc_delete_error": 0,
    "table_store_active_manifest_clear_error": 0,
    "table_prefetch_descriptor_count": 0,
    "table_prefetch_legacy_array_count": 0,
    "pca_mean_ms_total": 0,
    "pca_basis_ms_total": 0,
    "pca_method_randomized_count": 0,
    "pca_method_covariance_count": 0,
    "pca_method_svd_reference_count": 0,
    "pca_method_fallback_count": 0,
    "pca_cov_bytes_max": 0,
    "pca_randomized_rank_max": 0,
    "pca_tile_count": 0,
    "pca_mean_processed_fp32_tile_bytes": 0,
    "pca_basis_processed_fp32_tile_bytes": 0,
    "pca_reference_processed_fp32_tile_bytes": 0,
    "pca_processed_fp32_tile_bytes": 0,
    "pca_insufficient_samples_count": 0,
    "pca_error_count": 0,
    "table_build_projection_ms_total": 0,
    "table_build_write_ms_total": 0,
    "table_build_projection_tile_count": 0,
    "table_build_entry_count": 0,
    "table_build_rollout_count": 0,
    "table_build_token_count": 0,
    "table_build_pca_mean_processed_fp32_tile_bytes": 0,
    "table_build_pca_basis_processed_fp32_tile_bytes": 0,
    "table_build_pca_reference_processed_fp32_tile_bytes": 0,
    "table_build_projection_processed_fp32_tile_bytes": 0,
    "table_build_processed_fp32_tile_bytes": 0,
    "table_build_error_count": 0,
}


def _stable_partition_id(key: str, num_partitions: int) -> int:
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be > 0, got {num_partitions}")
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    hv = int.from_bytes(h, byteorder="little", signed=False)
    return hv % num_partitions


def hspec_legacy_dataproto_hs_enabled() -> bool:
    """Whether to keep the old DataProto ndarray transport path."""
    return os.getenv("HSPEC_LEGACY_DATAPROTO_HS", "0") != "0"


def hspec_default_descriptor_mode_enabled() -> bool:
    """Whether the default descriptor transport path is active."""
    return not hspec_legacy_dataproto_hs_enabled()


def hspec_strict_descriptor_mode_enabled() -> bool:
    """Whether descriptor mode should reject legacy payloads by default.

    This is a configuration helper for Step 5's hard API boundary. Step 1 only
    centralizes the policy so rollout/trainer/table code do not invent their
    own interpretation of the env knobs.
    """
    return (
        os.getenv("HSPEC_STRICT_DESCRIPTOR_MODE", "1") != "0"
        and not hspec_legacy_dataproto_hs_enabled()
    )


def hspec_step0_runtime_asserts_enabled() -> bool:
    """Enable cheap Step-0 key-level invariants outside decode hot paths."""
    return os.getenv("HSPEC_STEP0_RUNTIME_ASSERTS", "0") != "0"


def hspec_raw_store_gc_after_epoch_enabled() -> bool:
    """Whether trainer epoch barriers should delete successfully built raw segments."""
    return os.getenv("HSPEC_RAW_STORE_GC_AFTER_EPOCH", "1") != "0"


def hspec_single_node_only_enabled() -> bool:
    """Whether descriptor raw-store paths must remain on one logical node."""
    return os.getenv("HSPEC_SINGLE_NODE_ONLY", "1") != "0"


def hspec_topology_strict_enabled() -> bool:
    """Whether descriptor topology/routing drift should fail fast."""
    return os.getenv("HSPEC_TOPOLOGY_STRICT", "1") != "0"


def hspec_record_store_metric(name: str, value: int = 1) -> None:
    """Record a lightweight HSpec store/runtime metric.

    Step 0 metrics are process-local counters only. They intentionally avoid
    filesystem scans, object-store queries, or ndarray size traversal.
    """
    _metric_add(str(name), int(value))


def hspec_record_store_metric_max(name: str, value: int) -> None:
    """Record a process-local HSpec metric with max aggregation semantics."""
    with _store_metrics_lock:
        key = str(name)
        _store_metrics[key] = max(int(_store_metrics.get(key, 0)), int(value))


def get_hspec_store_dtype() -> str:
    """Return HSpec raw-store on-disk dtype.

    Phase 1 stores hidden rows as float16 by default to minimize local IO and
    page-cache pressure. This dtype describes bytes on disk, not the model's
    original hidden-state dtype.
    """
    value = os.getenv("HSPEC_STORE_DTYPE", "float16").strip().lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(
        f"Unsupported HSPEC_STORE_DTYPE={value!r}. "
        "HSpec Phase 1 currently supports only 'float16' on disk."
    )


def hspec_require_explicit_num_shards_enabled() -> bool:
    return os.getenv("HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS", "1") != "0"


def assert_hspec_num_shards_configured_for_production() -> None:
    """Require explicit shard count only on the HSpec production init path."""
    if hspec_require_explicit_num_shards_enabled() and not os.getenv("HSPEC_NUM_SHARDS"):
        raise RuntimeError(
            "HSPEC_NUM_SHARDS must be set when HSpec decode is enabled. "
            "Set HSPEC_NUM_SHARDS to match rollout TP/build shard policy, "
            "for example HSPEC_NUM_SHARDS=${HSPEC_INFER_TP}."
        )


def get_hspec_build_actor_num_cpus() -> float:
    value = os.getenv("HSPEC_BUILD_ACTOR_NUM_CPUS", "1")
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_BUILD_ACTOR_NUM_CPUS=%s; using 1", value)
        return 1.0
    if parsed <= 0:
        logger.warning("HSPEC_BUILD_ACTOR_NUM_CPUS=%s must be > 0; using 1", value)
        return 1.0
    return parsed


def get_hspec_build_blas_threads() -> int:
    value = os.getenv("HSPEC_BUILD_BLAS_THREADS", "1")
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_BUILD_BLAS_THREADS=%s; using 1", value)
        return 1
    if parsed <= 0:
        logger.warning("HSPEC_BUILD_BLAS_THREADS=%s must be > 0; using 1", value)
        return 1
    return parsed


def _parse_nonnegative_int_env(name: str, default: int = 0) -> int:
    value = os.getenv(name, str(default))
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Ignoring invalid %s=%s; using %s", name, value, default)
        return max(int(default), 0)


def _parse_nonnegative_float_env(name: str, default: float = 0.0) -> float:
    value = os.getenv(name, str(default))
    try:
        return max(float(value), 0.0)
    except ValueError:
        logger.warning("Ignoring invalid %s=%s; using %s", name, value, default)
        return max(float(default), 0.0)


def get_hspec_build_max_prompt_rows() -> int:
    """Maximum input rows materialized per prompt build, 0 means unlimited."""
    return _parse_nonnegative_int_env("HSPEC_BUILD_MAX_PROMPT_ROWS", 0)


def get_hspec_build_max_prompt_raw_bytes() -> int:
    """Maximum descriptor raw bytes materialized per prompt build, 0 means unlimited."""
    return _parse_nonnegative_int_env("HSPEC_BUILD_MAX_PROMPT_RAW_BYTES", 0)


def get_hspec_build_max_prompt_descs() -> int:
    """Maximum descriptors selected per prompt build, 0 means unlimited."""
    return _parse_nonnegative_int_env("HSPEC_BUILD_MAX_PROMPT_DESCS", 0)


def get_hspec_build_max_rss_mb() -> float:
    """Best-effort build actor RSS hard guard in MiB, 0 means disabled."""
    return _parse_nonnegative_float_env("HSPEC_BUILD_MAX_RSS_MB", 0.0)


def get_hspec_build_max_pending_epochs() -> int:
    """Maximum pending HSpec build epochs on the trainer, 0 means unlimited."""
    return _parse_nonnegative_int_env("HSPEC_BUILD_MAX_PENDING_EPOCHS", 0)


def get_hspec_build_queue_max_lag_s() -> float:
    """Soft build-queue lag threshold in seconds, 0 means disabled."""
    return _parse_nonnegative_float_env("HSPEC_BUILD_QUEUE_MAX_LAG_S", 0.0)


def get_hspec_epoch_build_barrier_timeout_s() -> float:
    """Epoch-boundary HSpec build wait timeout, 0 preserves Phase-3 wait-all."""
    return _parse_nonnegative_float_env("HSPEC_EPOCH_BUILD_BARRIER_TIMEOUT_S", 0.0)


def hspec_swap_partial_on_timeout_enabled() -> bool:
    """Whether Phase-4 timeout may publish a prompt-level partial swap."""
    return os.getenv("HSPEC_SWAP_PARTIAL_ON_TIMEOUT", "0") != "0"


def hspec_build_timeout_discard_unfinished_enabled() -> bool:
    """Whether unfinished descs are discarded after a build-barrier timeout."""
    return os.getenv("HSPEC_BUILD_TIMEOUT_DISCARD_UNFINISHED", "1") != "0"


def get_hspec_build_actor_name_prefix() -> str:
    return os.getenv("HSPEC_BUILD_ACTOR_NAME_PREFIX", "hspec_build_shard").strip() or "hspec_build_shard"


def get_hspec_node_id() -> str:
    """Best-effort stable node id for descriptor locality checks."""
    for key in ("NODE_RANK", "RAY_NODE_ID", "HOSTNAME", "COMPUTERNAME"):
        value = os.getenv(key)
        if value:
            return str(value)
    return socket.gethostname()


def get_hspec_worker_rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "WORKER_RANK"):
        value = os.getenv(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return int(os.getpid())


def get_hspec_num_shards() -> int:
    value = os.getenv("HSPEC_NUM_SHARDS")
    if value:
        try:
            return max(int(value), 1)
        except ValueError:
            logger.warning("Ignoring invalid HSPEC_NUM_SHARDS=%s", value)
    infer_tp = os.getenv("HSPEC_INFER_TP") or os.getenv("INFER_TP")
    if infer_tp:
        try:
            return max(int(infer_tp), 1)
        except ValueError:
            logger.warning("Ignoring invalid HSPEC_INFER_TP=%s", infer_tp)
    logger.warning(
        "HSPEC_NUM_SHARDS is not set; falling back to 5 outside strict HSpec production init."
    )
    return 5


def get_hspec_tp_group_id() -> int:
    value = os.getenv("HSPEC_TP_GROUP_ID")
    if value not in (None, ""):
        try:
            return max(int(value), 0)
        except ValueError:
            logger.warning("Ignoring invalid HSPEC_TP_GROUP_ID=%s", value)

    rank_value = os.getenv("RANK")
    infer_tp = os.getenv("HSPEC_INFER_TP") or os.getenv("INFER_TP")
    if rank_value is not None and infer_tp:
        try:
            rank = int(rank_value)
            tp = max(int(infer_tp), 1)
            return max(rank // tp, 0)
        except ValueError:
            logger.warning(
                "Failed to derive HSPEC_TP_GROUP_ID from RANK=%s HSPEC_INFER_TP=%s",
                rank_value,
                infer_tp,
            )
    return 0


def get_hspec_store_root() -> str:
    default_root = os.path.abspath(
        os.path.join(os.getcwd(), "outputs", "hspec_store"),
    )
    return os.getenv("HSPEC_STORE_DIR", default_root)


def get_hspec_table_store_root() -> str:
    default_root = os.path.join(get_hspec_store_root(), "table_store")
    return os.getenv("HSPEC_TABLE_STORE_DIR", default_root)


def get_hspec_store_isolation_mode() -> str:
    value = os.getenv("HSPEC_STORE_ISOLATION_MODE", "clean").strip().lower()
    if value in {"clean", "unique", "reuse"}:
        return value
    logger.warning(
        "Ignoring invalid HSPEC_STORE_ISOLATION_MODE=%s; using clean", value)
    return "clean"


def hspec_require_fresh_table_store_enabled() -> bool:
    return os.getenv("HSPEC_REQUIRE_FRESH_TABLE_STORE", "0") != "0"


def get_hspec_raw_store_max_bytes() -> int:
    value = os.getenv("HSPEC_RAW_STORE_MAX_BYTES", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_RAW_STORE_MAX_BYTES=%s", value)
        return 0


def get_hspec_collect_max_bytes_per_worker() -> int:
    """Maximum descriptor-mode collect bytes reserved by one rollout worker."""
    return _parse_nonnegative_int_env("HSPEC_COLLECT_MAX_BYTES_PER_WORKER", 0)


def get_hspec_raw_store_max_files() -> int:
    value = os.getenv("HSPEC_RAW_STORE_MAX_FILES", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_RAW_STORE_MAX_FILES=%s", value)
        return 0


def get_hspec_raw_store_max_bytes_per_epoch() -> int:
    """Return the Phase-4 per-epoch raw-store byte budget.

    Step 1 only centralizes the env contract. Step 2 wires this into
    collection-time backpressure. A value of 0 disables the budget.
    """
    value = os.getenv("HSPEC_RAW_STORE_MAX_BYTES_PER_EPOCH", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning(
            "Ignoring invalid HSPEC_RAW_STORE_MAX_BYTES_PER_EPOCH=%s", value)
        return 0


def hspec_raw_store_stop_collect_on_budget_enabled() -> bool:
    """Whether raw-store budget pressure should stop new collection."""
    return os.getenv("HSPEC_RAW_STORE_STOP_COLLECT_ON_BUDGET", "1") != "0"


def get_hspec_store_retain_batches() -> int:
    value = os.getenv("HSPEC_STORE_RETAIN_BATCHES", "128")
    try:
        return max(int(value), 1)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_STORE_RETAIN_BATCHES=%s", value)
        return 128


def hspec_segment_fsync_on_seal_enabled() -> bool:
    return os.getenv("HSPEC_SEGMENT_FSYNC_ON_SEAL", "0") != "0"


def hspec_raw_store_budget_delete_enabled() -> bool:
    """Whether budget enforcement may delete manifest-marked raw segments.

    Epoch-level GC is the normal production cleanup path. Budget deletion is
    intentionally opt-in and only touches gc_deletable/aborted segments.
    """
    return os.getenv("HSPEC_RAW_STORE_BUDGET_DELETE", "0") != "0"


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
            logger.debug("Failed to remove temporary HSpec manifest %s", tmp, exc_info=True)


def _metric_add(name: str, value: int) -> None:
    with _store_metrics_lock:
        _store_metrics[name] = _store_metrics.get(name, 0) + int(value)


def _metric_set(name: str, value: int) -> None:
    with _store_metrics_lock:
        _store_metrics[name] = int(value)


def _record_collect_drop(reason: str, count: int = 1) -> None:
    _metric_add("collect_dropped", count)
    _metric_add(f"collect_dropped_{reason}", count)


def collect_hspec_store_metrics(reset: bool = True) -> Dict[str, int]:
    with _store_metrics_lock:
        metrics = dict(_store_metrics)
        if reset:
            for key in list(_store_metrics.keys()):
                _store_metrics[key] = 0
    return metrics


@dataclass(frozen=True)
class HSpecTrajectoryDesc:
    epoch: int
    global_step: int
    node_id: str
    worker_rank: int
    tp_group_id: int
    shard_id: int
    request_id: str
    prompt_id: str
    hs_path: str
    hs_offset_rows: int
    token_path: str
    token_offset: int
    length: int
    hidden_dim: int
    hs_dtype: str
    token_dtype: str
    reward: float | None = None

    def with_updates(self, **kwargs: Any) -> "HSpecTrajectoryDesc":
        return replace(self, **kwargs)


def coerce_hspec_desc(obj: Any) -> HSpecTrajectoryDesc:
    if isinstance(obj, HSpecTrajectoryDesc):
        return obj
    if isinstance(obj, dict):
        return HSpecTrajectoryDesc(**obj)
    raise TypeError(f"Expected HSpecTrajectoryDesc or dict, got {type(obj)!r}")


@dataclass(frozen=True)
class HSpecSegmentKey:
    node_id: str
    worker_rank: int
    tp_group_id: int
    segment_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hspec_segment_key_from_desc(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> HSpecSegmentKey:
    desc = coerce_hspec_desc(desc_obj)
    hs_dir = Path(desc.hs_path).resolve().parent
    token_dir = Path(desc.token_path).resolve().parent
    if hs_dir != token_dir:
        raise ValueError(
            f"HSpec descriptor spans different segment dirs: hs={hs_dir} token={token_dir}"
        )
    return HSpecSegmentKey(
        node_id=str(desc.node_id),
        worker_rank=int(desc.worker_rank),
        tp_group_id=int(desc.tp_group_id),
        segment_dir=str(hs_dir),
    )


def read_hspec_segment_manifest(segment_dir: str | Path) -> Dict[str, Any]:
    manifest_path = Path(segment_dir).resolve() / _SEGMENT_MANIFEST_NAME
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_hspec_segment_manifest_status(
    segment_dir: str | Path,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    segment_path = Path(segment_dir).resolve()
    manifest = read_hspec_segment_manifest(segment_path)
    manifest["status"] = str(status)
    manifest["status_time_ns"] = time.time_ns()
    if extra:
        manifest.update(extra)
    _write_json_atomic(segment_path / _SEGMENT_MANIFEST_NAME, manifest)


class HSpecLocalCollector:
    """Process-local writer for HSpec rollout trajectories.

    Hidden states are written as fp16 rows. Tokens are written as int32.
    Storage is per-worker append-only segment per batch, while each request
    keeps a contiguous row/token slice via offsets in the descriptor.
    """

    def __init__(self) -> None:
        self.node_id = get_hspec_node_id()
        self.worker_rank = get_hspec_worker_rank()
        self.tp_group_id = get_hspec_tp_group_id()
        self.store_root = Path(get_hspec_store_root()).resolve()
        self.table_store_root = Path(get_hspec_table_store_root()).resolve()
        self._lock = threading.Lock()
        self._batch_counter = 0
        self._req_states: Dict[str, Dict[str, Any]] = {}
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.table_store_root.mkdir(parents=True, exist_ok=True)
        if "HSPEC_STORE_DIR" not in os.environ and str(self.store_root).startswith(tempfile.gettempdir()):
            logger.warning(
                "HSpec store root defaulted to a temp directory: %s. "
                "Set HSPEC_STORE_DIR to a large local NVMe or controlled shared path.",
                self.store_root,
            )
        self._segment_dir = self._batch_dir()
        self._segment_hs_path = str(self._segment_dir / "hs.fp16.bin")
        self._segment_token_path = str(self._segment_dir / "tokens.i32.bin")
        self._segment_hs_fh = None
        self._segment_token_fh = None
        self._segment_hs_rows = 0
        self._segment_token_len = 0
        self._epoch_bytes: Dict[int, int] = {}
        self._worker_collect_bytes = 0
        self._current_collect_epoch = -1
        self._budget_blocked_epoch: set[int] = set()
        self._raw_store_budget_blocked = False

    def _batch_dir(self) -> Path:
        path = (
            self.store_root
            / f"node_{self.node_id}"
            / f"tp_{self.tp_group_id:03d}"
            / f"worker_{self.worker_rank:05d}"
            / f"pid_{os.getpid()}"
            / f"batch_{self._batch_counter:08d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _count_existing_files(self) -> int:
        try:
            return sum(1 for _ in self.store_root.rglob("*") if _.is_file())
        except Exception:
            return 0

    @staticmethod
    def _segment_dir_bytes(segment_dir: Path) -> int:
        total = 0
        for file_path in segment_dir.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                total += int(file_path.stat().st_size)
            except FileNotFoundError:
                continue
        return total

    @staticmethod
    def _resolve_collect_epoch(epoch: int | None) -> int:
        if epoch is not None:
            return int(epoch)
        try:
            from vllm_ascend.spec_decode.hspec_utils import hspec_collection_context_epoch

            return int(hspec_collection_context_epoch())
        except Exception:
            return -1

    @staticmethod
    def _estimate_payload_bytes(rows_count: int, hidden_dim: int, token_count: int = 0) -> int:
        hidden_bytes = max(int(rows_count), 0) * max(int(hidden_dim), 0) * np.dtype(np.float16).itemsize
        token_bytes = max(int(token_count), 0) * np.dtype(np.int32).itemsize
        return int(hidden_bytes + token_bytes)

    def _maybe_rotate_collect_epoch_locked(self, collect_epoch: int) -> None:
        if collect_epoch < 0 or int(collect_epoch) == int(self._current_collect_epoch):
            return
        self._current_collect_epoch = int(collect_epoch)
        self._worker_collect_bytes = 0
        self._epoch_bytes = {
            int(collect_epoch): int(self._epoch_bytes.get(int(collect_epoch), 0))
        }
        self._budget_blocked_epoch = {
            int(collect_epoch)
        } if int(collect_epoch) in self._budget_blocked_epoch else set()

    def _collect_budget_decision_locked(
        self,
        estimated_bytes: int,
        epoch: int | None = None,
    ) -> Dict[str, Any]:
        bytes_est = max(int(estimated_bytes), 0)
        collect_epoch = self._resolve_collect_epoch(epoch)
        self._maybe_rotate_collect_epoch_locked(collect_epoch)

        if hspec_raw_store_stop_collect_on_budget_enabled() and self._raw_store_budget_blocked:
            _metric_set("raw_store_budget_active", 1)
            return {
                "allow": False,
                "reason": "raw_store_over_budget",
                "epoch": int(collect_epoch),
                "estimated_bytes": int(bytes_est),
                "worker_bytes": int(self._worker_collect_bytes),
                "epoch_bytes": int(self._epoch_bytes.get(collect_epoch, 0)),
            }

        worker_limit = get_hspec_collect_max_bytes_per_worker()
        if worker_limit > 0 and self._worker_collect_bytes + bytes_est > worker_limit:
            return {
                "allow": False,
                "reason": "budget_worker_bytes",
                "epoch": int(collect_epoch),
                "estimated_bytes": int(bytes_est),
                "limit_bytes": int(worker_limit),
                "current_bytes": int(self._worker_collect_bytes),
                "over_bytes": int(self._worker_collect_bytes + bytes_est - worker_limit),
            }

        epoch_limit = get_hspec_raw_store_max_bytes_per_epoch()
        if epoch_limit > 0:
            _metric_set("raw_store_epoch_budget_bytes", epoch_limit)
            epoch_bytes = int(self._epoch_bytes.get(collect_epoch, 0))
            if collect_epoch in self._budget_blocked_epoch or epoch_bytes + bytes_est > epoch_limit:
                if collect_epoch not in self._budget_blocked_epoch:
                    self._budget_blocked_epoch.add(collect_epoch)
                    _metric_add("raw_store_collect_budget_blocked", 1)
                return {
                    "allow": False,
                    "reason": "budget_epoch_bytes",
                    "epoch": int(collect_epoch),
                    "estimated_bytes": int(bytes_est),
                    "limit_bytes": int(epoch_limit),
                    "current_bytes": int(epoch_bytes),
                    "over_bytes": int(epoch_bytes + bytes_est - epoch_limit),
                }

        return {
            "allow": True,
            "reason": "",
            "epoch": int(collect_epoch),
            "estimated_bytes": int(bytes_est),
            "worker_bytes": int(self._worker_collect_bytes),
            "epoch_bytes": int(self._epoch_bytes.get(collect_epoch, 0)),
        }

    def collect_budget_decision(
        self,
        estimated_bytes: int,
        epoch: int | None = None,
    ) -> Dict[str, Any]:
        with self._lock:
            return dict(self._collect_budget_decision_locked(estimated_bytes, epoch))

    def _record_budget_reject_locked(
        self,
        decision: Dict[str, Any],
        *,
        estimated_bytes: int,
        reqs: int = 1,
    ) -> None:
        reason = str(decision.get("reason", "collect_budget"))
        bytes_est = max(int(estimated_bytes), 0)
        _record_collect_drop(reason, max(int(reqs), 1))
        _metric_add("raw_store_collect_drop_bytes", bytes_est)
        if reason == "budget_worker_bytes":
            _metric_add("collect_dropped_budget_worker_bytes", bytes_est)
        elif reason == "budget_epoch_bytes":
            _metric_add("collect_dropped_budget_epoch_bytes", bytes_est)
        elif reason == "raw_store_over_budget":
            _metric_add("collect_dropped_raw_store_over_budget", bytes_est)

    def try_reserve_collect_budget(
        self,
        estimated_bytes: int,
        epoch: int | None = None,
        reqs: int = 1,
    ) -> bool:
        bytes_est = max(int(estimated_bytes), 0)
        with self._lock:
            decision = self._collect_budget_decision_locked(bytes_est, epoch)
            if not bool(decision.get("allow", True)):
                self._record_budget_reject_locked(
                    decision,
                    estimated_bytes=bytes_est,
                    reqs=reqs,
                )
                return False
            collect_epoch = int(decision.get("epoch", self._resolve_collect_epoch(epoch)))
            self._worker_collect_bytes += bytes_est
            self._epoch_bytes[collect_epoch] = int(self._epoch_bytes.get(collect_epoch, 0)) + bytes_est
            _metric_add("raw_store_epoch_bytes", bytes_est)
            return True

    def _enforce_store_budget(self) -> None:
        max_bytes = get_hspec_raw_store_max_bytes()
        max_files = get_hspec_raw_store_max_files()
        if max_bytes <= 0 and max_files <= 0:
            if self._raw_store_budget_blocked:
                self._raw_store_budget_blocked = False
                _metric_add("raw_store_collect_budget_unblocked", 1)
            _metric_set("raw_store_budget_active", 0)
            return

        try:
            batch_dirs = sorted(
                [p for p in self.store_root.rglob("batch_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
        except Exception:
            logger.debug("Failed to list HSpec batch directories for GC", exc_info=True)
            return

        total_bytes = 0
        deletable_dirs: list[Path] = []
        for batch_dir in batch_dirs:
            if not batch_dir.exists():
                continue
            total_bytes += self._segment_dir_bytes(batch_dir)
            manifest_path = batch_dir / _SEGMENT_MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = read_hspec_segment_manifest(batch_dir)
            except Exception:
                continue
            if str(manifest.get("status", "")) in {"gc_deletable", "aborted"}:
                deletable_dirs.append(batch_dir)

        total_files = self._count_existing_files()
        _metric_add("raw_store_budget_bytes", total_bytes)
        _metric_add("raw_store_budget_files", total_files)

        over_bytes = max(total_bytes - max_bytes, 0) if max_bytes > 0 else 0
        over_files = max(total_files - max_files, 0) if max_files > 0 else 0
        if over_bytes <= 0 and over_files <= 0:
            if self._raw_store_budget_blocked:
                self._raw_store_budget_blocked = False
                _metric_add("raw_store_collect_budget_unblocked", 1)
            _metric_set("raw_store_budget_active", 0)
            return

        _metric_set("raw_store_budget_active", 1)
        if hspec_raw_store_stop_collect_on_budget_enabled() and not self._raw_store_budget_blocked:
            self._raw_store_budget_blocked = True
            _metric_add("raw_store_collect_budget_blocked", 1)
        _metric_add("raw_store_budget_over_bytes", over_bytes)
        _metric_add("raw_store_budget_over_files", over_files)
        if not hspec_raw_store_budget_delete_enabled():
            _metric_add("raw_store_budget_gc_skipped", 1)
            logger.warning(
                "HSpec raw store budget exceeded but budget deletion is disabled: "
                "bytes=%d max_bytes=%d files=%d max_files=%d",
                total_bytes,
                max_bytes,
                total_files,
                max_files,
            )
            return

        for batch_dir in deletable_dirs:
            if (max_bytes <= 0 or total_bytes <= max_bytes) and (max_files <= 0 or total_files <= max_files):
                break
            before_bytes = self._segment_dir_bytes(batch_dir)
            before_files = sum(1 for p in batch_dir.rglob("*") if p.is_file())
            if delete_hspec_segment(batch_dir, caller_confirmed_safe=False):
                _metric_add("raw_store_budget_gc_deleted", 1)
                total_bytes = max(total_bytes - before_bytes, 0)
                total_files = max(total_files - before_files, 0)
        if (max_bytes <= 0 or total_bytes <= max_bytes) and (max_files <= 0 or total_files <= max_files):
            if self._raw_store_budget_blocked:
                self._raw_store_budget_blocked = False
                _metric_add("raw_store_collect_budget_unblocked", 1)
            _metric_set("raw_store_budget_active", 0)

    @staticmethod
    def _safe_request_key(req_id: str) -> str:
        digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).hexdigest()
        return f"r{digest}"

    def _state_for_req(self, req_id: str) -> Dict[str, Any]:
        req_id = str(req_id)
        state = self._req_states.get(req_id)
        if state is not None:
            return state

        state = {
            "request_id": req_id,
            "hs_path": self._segment_hs_path,
            "token_path": self._segment_token_path,
            "hidden_dim": 0,
            "hs_dtype": "",
            "hs_rows": 0,
            "token_len": 0,
            "hs_offset_rows": None,
            "token_offset": None,
        }
        self._req_states[req_id] = state
        return state

    @staticmethod
    def _close_state_files(state: Dict[str, Any]) -> None:
        return

    def _segment_has_work_locked(self) -> bool:
        return (
            bool(self._req_states)
            or int(self._segment_hs_rows) > 0
            or int(self._segment_token_len) > 0
            or self._segment_hs_fh is not None
            or self._segment_token_fh is not None
        )

    def _flush_and_close_segment_files_locked(self) -> None:
        for fh in (self._segment_hs_fh, self._segment_token_fh):
            if fh is None:
                continue
            try:
                fh.flush()
                if hspec_segment_fsync_on_seal_enabled():
                    os.fsync(fh.fileno())
                    _metric_add("segment_fsync_count", 1)
            finally:
                try:
                    fh.close()
                except Exception:
                    _metric_add("segment_close_error", 1)
                    logger.debug("Failed to close HSpec segment file", exc_info=True)
        self._segment_hs_fh = None
        self._segment_token_fh = None

    def _write_segment_manifest_locked(
        self,
        *,
        status: str,
        desc_count: int,
        epoch: int,
        global_step: int,
        dropped_count: int = 0,
    ) -> None:
        payload = {
            "schema_version": 1,
            "status": str(status),
            "node_id": self.node_id,
            "worker_rank": int(self.worker_rank),
            "tp_group_id": int(self.tp_group_id),
            "pid": int(os.getpid()),
            "batch_id": int(self._batch_counter),
            "segment_dir": str(self._segment_dir),
            "hs_path": str(self._segment_hs_path),
            "token_path": str(self._segment_token_path),
            "hs_rows": int(self._segment_hs_rows),
            "token_len": int(self._segment_token_len),
            "desc_count": int(desc_count),
            "dropped_count": int(dropped_count),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "store_dtype": get_hspec_store_dtype(),
            "token_dtype": "int32",
            "sealed_time_ns": time.time_ns(),
            "fsync_on_seal": hspec_segment_fsync_on_seal_enabled(),
        }
        try:
            _write_json_atomic(self._segment_dir / _SEGMENT_MANIFEST_NAME, payload)
        except Exception:
            _metric_add("segment_manifest_write_error", 1)
            raise

    def _rotate_to_next_segment_locked(self) -> None:
        self._batch_counter += 1
        self._segment_dir = self._batch_dir()
        self._segment_hs_path = str(self._segment_dir / "hs.fp16.bin")
        self._segment_token_path = str(self._segment_dir / "tokens.i32.bin")
        self._segment_hs_fh = None
        self._segment_token_fh = None
        self._segment_hs_rows = 0
        self._segment_token_len = 0
        _metric_add("segment_rotated", 1)

    def clear_batch(self) -> None:
        """Forget in-flight request state without deleting flushed files."""
        with self._lock:
            for state in self._req_states.values():
                self._close_state_files(state)
            has_work = self._segment_has_work_locked()
            self._req_states.clear()
            if not has_work:
                return
            self._flush_and_close_segment_files_locked()
            self._write_segment_manifest_locked(
                status="gc_deletable",
                desc_count=0,
                epoch=-1,
                global_step=-1,
                dropped_count=0,
            )
            _metric_add("segment_aborted", 1)
            self._rotate_to_next_segment_locked()

    def append_hidden_rows(
        self,
        req_id: str,
        rows: torch.Tensor,
        epoch: int | None = None,
    ) -> None:
        if rows is None or rows.numel() == 0:
            return
        if rows.ndim != 2:
            raise ValueError(f"HSpec hidden rows must be 2-D, got {tuple(rows.shape)}")
        rows_count = int(rows.shape[0])
        hidden_dim = int(rows.shape[1])
        estimated_bytes = self._estimate_payload_bytes(rows_count, hidden_dim, 0)
        if not self.try_reserve_collect_budget(estimated_bytes, epoch=epoch, reqs=1):
            return

        store_dtype = get_hspec_store_dtype()
        if store_dtype == "float16":
            rows_cpu = rows.detach().to(device="cpu", dtype=torch.float16).contiguous()
            hs_dtype = "float16"
        else:
            raise AssertionError(f"Unsupported HSpec store dtype after validation: {store_dtype}")
        rows_np = rows_cpu.numpy()
        source_dtype = str(getattr(rows, "dtype", ""))

        with self._lock:
            state = self._state_for_req(str(req_id))
            self._append_hidden_rows_locked(
                str(req_id),
                state,
                rows_np,
                hs_dtype,
                source_dtype,
            )

    def extend_tokens(
        self,
        req_id: str,
        token_ids: Iterable[int],
        epoch: int | None = None,
    ) -> None:
        token_list = [int(t) for t in token_ids]
        if not token_list:
            return
        estimated_bytes = self._estimate_payload_bytes(0, 0, len(token_list))
        if not self.try_reserve_collect_budget(estimated_bytes, epoch=epoch, reqs=1):
            return
        token_np = np.ascontiguousarray(token_list, dtype=np.int32)

        with self._lock:
            state = self._state_for_req(str(req_id))
            self._extend_tokens_locked(state, token_np)

    def append_hidden_and_tokens(
        self,
        req_id: str,
        rows: torch.Tensor,
        token_ids: Iterable[int],
        epoch: int | None = None,
    ) -> None:
        token_list = [int(t) for t in token_ids]
        if (rows is None or rows.numel() == 0) and not token_list:
            return
        if rows is None or rows.numel() == 0:
            raise ValueError("HSpec token payload cannot be written without hidden rows")
        if rows.ndim != 2:
            raise ValueError(f"HSpec hidden rows must be 2-D, got {tuple(rows.shape)}")
        rows_count = int(rows.shape[0])
        hidden_dim = int(rows.shape[1])
        if rows_count != len(token_list):
            raise ValueError(
                f"HSpec token/hidden length mismatch for {req_id}: "
                f"hidden_rows={rows_count} token_len={len(token_list)}"
            )
        estimated_bytes = self._estimate_payload_bytes(rows_count, hidden_dim, len(token_list))
        if not self.try_reserve_collect_budget(estimated_bytes, epoch=epoch, reqs=1):
            return

        store_dtype = get_hspec_store_dtype()
        if store_dtype == "float16":
            rows_cpu = rows.detach().to(device="cpu", dtype=torch.float16).contiguous()
            hs_dtype = "float16"
        else:
            raise AssertionError(f"Unsupported HSpec store dtype after validation: {store_dtype}")
        rows_np = rows_cpu.numpy()
        token_np = np.ascontiguousarray(token_list, dtype=np.int32)
        source_dtype = str(getattr(rows, "dtype", ""))

        with self._lock:
            state = self._state_for_req(str(req_id))
            self._append_hidden_rows_locked(
                str(req_id),
                state,
                rows_np,
                hs_dtype,
                source_dtype,
            )
            self._extend_tokens_locked(state, token_np)

    def _append_hidden_rows_locked(
        self,
        req_id: str,
        state: Dict[str, Any],
        rows_np: np.ndarray,
        hs_dtype: str,
        source_dtype: str,
    ) -> None:
        hidden_dim = int(rows_np.shape[1])
        rows_count = int(rows_np.shape[0])
        if state["hidden_dim"] == 0:
            state["hidden_dim"] = hidden_dim
        elif int(state["hidden_dim"]) != hidden_dim:
            raise ValueError(
                f"HSpec hidden dim mismatch for {req_id}: "
                f"{state['hidden_dim']} vs {hidden_dim}"
            )
        if not state.get("hs_dtype"):
            state["hs_dtype"] = hs_dtype
        elif str(state["hs_dtype"]) != hs_dtype:
            raise ValueError(
                f"HSpec hidden dtype mismatch for {req_id}: "
                f"{state['hs_dtype']} vs {hs_dtype}"
            )
        if state["hs_offset_rows"] is None:
            state["hs_offset_rows"] = int(self._segment_hs_rows)
        if self._segment_hs_fh is None:
            self._segment_hs_fh = open(self._segment_hs_path, "ab")
        self._segment_hs_fh.write(rows_np.tobytes(order="C"))
        state["hs_rows"] += rows_count
        self._segment_hs_rows += rows_count
        _metric_add("raw_store_bytes", int(rows_np.nbytes))
        _metric_add("store_fp16_rows", rows_count)
        if "bfloat16" in source_dtype:
            _metric_add("source_dtype_bf16_rows", rows_count)
        elif "float16" in source_dtype:
            _metric_add("source_dtype_fp16_rows", rows_count)
        else:
            _metric_add("source_dtype_other_rows", rows_count)

    def _extend_tokens_locked(
        self,
        state: Dict[str, Any],
        token_np: np.ndarray,
    ) -> None:
        if state["token_offset"] is None:
            state["token_offset"] = int(self._segment_token_len)
        if self._segment_token_fh is None:
            self._segment_token_fh = open(self._segment_token_path, "ab")
        self._segment_token_fh.write(token_np.tobytes(order="C"))
        state["token_len"] += int(token_np.shape[0])
        self._segment_token_len += int(token_np.shape[0])
        _metric_add("raw_store_bytes", int(token_np.nbytes))

    def flush_descriptors(
        self,
        request_id_to_prompt_id: Optional[Dict[str, str]] = None,
        epoch: int = -1,
        global_step: int = -1,
    ) -> Dict[str, HSpecTrajectoryDesc]:
        request_id_to_prompt_id = request_id_to_prompt_id or {}
        descs: Dict[str, HSpecTrajectoryDesc] = {}

        with self._lock:
            if not self._segment_has_work_locked():
                return {}
            states = self._req_states
            self._req_states = {}

            dropped_count = 0
            for req_id, state in states.items():
                self._close_state_files(state)
                hs_rows = int(state.get("hs_rows", 0))
                token_len = int(state.get("token_len", 0))
                hidden_dim = int(state.get("hidden_dim", 0))
                hs_dtype = str(state.get("hs_dtype") or get_hspec_store_dtype())
                hs_offset_rows = state.get("hs_offset_rows")
                token_offset = state.get("token_offset")
                if hs_rows <= 0 or token_len <= 0:
                    _record_collect_drop("empty")
                    dropped_count += 1
                    continue
                if hidden_dim <= 0:
                    logger.warning(
                        "HSpec descriptor invalid hidden dim: req_id=%s hidden_dim=%d; dropping",
                        req_id,
                        hidden_dim,
                    )
                    _record_collect_drop("invalid_dim")
                    dropped_count += 1
                    continue
                if hs_offset_rows is None or token_offset is None:
                    logger.warning(
                        "HSpec descriptor missing offset: req_id=%s hs_offset_rows=%s token_offset=%s; dropping",
                        req_id,
                        hs_offset_rows,
                        token_offset,
                    )
                    _record_collect_drop("missing_offset")
                    dropped_count += 1
                    continue
                if hs_rows != token_len:
                    logger.warning(
                        "HSpec descriptor alignment mismatch: req_id=%s hs_rows=%d token_len=%d; dropping",
                        req_id,
                        hs_rows,
                        token_len,
                    )
                    _record_collect_drop("align_mismatch")
                    dropped_count += 1
                    continue

                prompt_id = str(request_id_to_prompt_id.get(req_id, ""))
                shard_id = _stable_partition_id(prompt_id or req_id, get_hspec_num_shards())
                desc = HSpecTrajectoryDesc(
                    epoch=int(epoch),
                    global_step=int(global_step),
                    node_id=self.node_id,
                    worker_rank=int(self.worker_rank),
                    tp_group_id=int(self.tp_group_id),
                    shard_id=int(shard_id),
                    request_id=str(req_id),
                    prompt_id=prompt_id,
                    hs_path=str(state["hs_path"]),
                    hs_offset_rows=int(hs_offset_rows),
                    token_path=str(state["token_path"]),
                    token_offset=int(token_offset),
                    length=hs_rows,
                    hidden_dim=hidden_dim,
                    hs_dtype=hs_dtype,
                    token_dtype="int32",
                    reward=None,
                )
                descs[req_id] = desc
                self._append_manifest(desc)

            self._flush_and_close_segment_files_locked()
            segment_status = "sealed" if descs else "gc_deletable"
            self._write_segment_manifest_locked(
                status=segment_status,
                desc_count=len(descs),
                epoch=int(epoch),
                global_step=int(global_step),
                dropped_count=dropped_count,
            )
            if descs:
                _metric_add("segment_sealed", 1)
            else:
                _metric_add("segment_sealed_empty", 1)
            self._rotate_to_next_segment_locked()

        _metric_add("desc_count", len(descs))
        self._enforce_store_budget()
        return descs

    def _append_manifest(self, desc: HSpecTrajectoryDesc) -> None:
        try:
            manifest_path = Path(desc.hs_path).with_name(_DESC_MANIFEST_NAME)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(desc), ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to append HSpec descriptor manifest", exc_info=True)


_collector: HSpecLocalCollector | None = None
_collector_lock = threading.Lock()


def get_hspec_local_collector() -> HSpecLocalCollector:
    global _collector
    if _collector is not None:
        return _collector
    with _collector_lock:
        if _collector is None:
            _collector = HSpecLocalCollector()
    return _collector


def hspec_collect_budget_decision(
    estimated_bytes: int = 0,
    epoch: int | None = None,
) -> Dict[str, Any]:
    """Cheap collect-budget precheck for rollout workers.

    This function never scans the raw-store directory. Directory-derived raw
    store pressure is reflected through the collector's sticky flag, which is
    updated only by low-frequency budget enforcement after segment flushes.
    """
    bytes_est = max(int(estimated_bytes), 0)
    if bytes_est <= 0:
        return {"allow": True, "reason": "", "estimated_bytes": 0}
    if (
        get_hspec_collect_max_bytes_per_worker() <= 0
        and get_hspec_raw_store_max_bytes_per_epoch() <= 0
        and not (
            hspec_raw_store_stop_collect_on_budget_enabled()
            and (get_hspec_raw_store_max_bytes() > 0 or get_hspec_raw_store_max_files() > 0)
            and _collector is not None
        )
    ):
        return {"allow": True, "reason": "", "estimated_bytes": int(bytes_est)}
    return get_hspec_local_collector().collect_budget_decision(bytes_est, epoch)


def hspec_record_collect_budget_reject(
    decision: Dict[str, Any],
    *,
    estimated_bytes: int = 0,
    reqs: int = 1,
) -> None:
    """Record store-side metrics for a collect-budget precheck rejection."""
    if _collector is None:
        reason = str(decision.get("reason", "collect_budget"))
        bytes_est = max(int(estimated_bytes), 0)
        _record_collect_drop(reason, max(int(reqs), 1))
        _metric_add("raw_store_collect_drop_bytes", bytes_est)
        if reason == "budget_worker_bytes":
            _metric_add("collect_dropped_budget_worker_bytes", bytes_est)
        elif reason == "budget_epoch_bytes":
            _metric_add("collect_dropped_budget_epoch_bytes", bytes_est)
        elif reason == "raw_store_over_budget":
            _metric_add("collect_dropped_raw_store_over_budget", bytes_est)
        return
    with _collector._lock:
        _collector._record_budget_reject_locked(
            dict(decision),
            estimated_bytes=max(int(estimated_bytes), 0),
            reqs=max(int(reqs), 1),
        )


def load_hspec_trajectory(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mmap-backed hidden states and token ids for a descriptor."""
    desc = coerce_hspec_desc(desc_obj)
    if desc.length <= 0:
        raise ValueError(f"HSpec descriptor has non-positive length: {desc.length}")
    if desc.hidden_dim <= 0:
        raise ValueError(f"HSpec descriptor has invalid hidden_dim: {desc.hidden_dim}")
    if int(desc.hs_offset_rows) < 0:
        raise ValueError(f"HSpec descriptor has negative hs_offset_rows: {desc.hs_offset_rows}")
    if int(desc.token_offset) < 0:
        raise ValueError(f"HSpec descriptor has negative token_offset: {desc.token_offset}")

    hs_dtype = np.dtype(desc.hs_dtype)
    token_dtype = np.dtype(desc.token_dtype)
    hs_offset = int(desc.hs_offset_rows) * int(desc.hidden_dim) * hs_dtype.itemsize
    token_offset = int(desc.token_offset) * token_dtype.itemsize
    hs_bytes = int(desc.length) * int(desc.hidden_dim) * hs_dtype.itemsize
    token_bytes = int(desc.length) * token_dtype.itemsize
    if not os.path.exists(desc.hs_path):
        raise FileNotFoundError(desc.hs_path)
    if not os.path.exists(desc.token_path):
        raise FileNotFoundError(desc.token_path)
    if os.path.getsize(desc.hs_path) < hs_offset + hs_bytes:
        raise ValueError(
            f"HSpec hs file too small: path={desc.hs_path} "
            f"size={os.path.getsize(desc.hs_path)} need={hs_offset + hs_bytes}"
        )
    if os.path.getsize(desc.token_path) < token_offset + token_bytes:
        raise ValueError(
            f"HSpec token file too small: path={desc.token_path} "
            f"size={os.path.getsize(desc.token_path)} need={token_offset + token_bytes}"
        )
    hs = np.memmap(
        desc.hs_path,
        dtype=hs_dtype,
        mode="r",
        offset=hs_offset,
        shape=(int(desc.length), int(desc.hidden_dim)),
        order="C",
    )
    tokens = np.memmap(
        desc.token_path,
        dtype=token_dtype,
        mode="r",
        offset=token_offset,
        shape=(int(desc.length),),
        order="C",
    )
    return hs, tokens


def estimate_hspec_trajectory_bytes(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> int:
    desc = coerce_hspec_desc(desc_obj)
    hs_dtype = np.dtype(desc.hs_dtype)
    token_dtype = np.dtype(desc.token_dtype)
    hs_bytes = int(desc.length) * int(desc.hidden_dim) * hs_dtype.itemsize
    token_bytes = int(desc.length) * token_dtype.itemsize
    return hs_bytes + token_bytes


def _cleanup_empty_segment_parents(seg: Path, root: Path) -> None:
    parent = seg.parent
    while parent != root and parent != parent.parent:
        if not parent.name.startswith(("pid_", "worker_", "tp_", "node_")):
            break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def delete_hspec_segment(
    segment: HSpecSegmentKey | str | Path,
    *,
    caller_confirmed_safe: bool = False,
) -> bool:
    """Delete a sealed HSpec segment directory after lifecycle checks.

    The caller must only pass ``caller_confirmed_safe=True`` after all build
    refs that may read the segment have completed successfully.
    """
    try:
        if isinstance(segment, HSpecSegmentKey):
            seg = Path(segment.segment_dir).resolve()
        else:
            seg = Path(segment).resolve()
        root = Path(get_hspec_store_root()).resolve()
        if seg == root or root not in seg.parents:
            raise ValueError(f"refuse to delete outside HSPEC_STORE_DIR: {seg}")
        if not seg.name.startswith("batch_"):
            raise ValueError(f"refuse to delete non-segment directory: {seg}")
        if not seg.exists():
            return False
        manifest = read_hspec_segment_manifest(seg)
        status = str(manifest.get("status", ""))
        allowed = _SEGMENT_CONFIRMED_DELETE_STATES if caller_confirmed_safe else _SEGMENT_SAFE_DELETE_STATES
        if status not in allowed:
            raise ValueError(f"refuse to delete segment with status={status!r}: {seg}")

        bytes_deleted = HSpecLocalCollector._segment_dir_bytes(seg)
        shutil.rmtree(seg)
        _metric_add("segment_delete_count", 1)
        _metric_add("segment_delete_bytes", bytes_deleted)
        _cleanup_empty_segment_parents(seg, root)
        return True
    except Exception:
        _metric_add("segment_delete_error", 1)
        logger.debug("Failed to delete HSpec segment %s", segment, exc_info=True)
        raise


def delete_hspec_trajectory(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> bool:
    """Unsafe legacy per-descriptor delete helper.

    Descriptor-mode HSpec stores many descriptors in one shared batch segment.
    Deleting ``desc.hs_path`` or ``desc.token_path`` can break other pending
    descriptors. Use ``delete_hspec_segment`` after epoch-level build success.
    """
    global _unsafe_delete_trajectory_warned
    if os.getenv("HSPEC_UNSAFE_DELETE_TRAJECTORY_FILES", "0") == "0":
        _metric_add("unsafe_descriptor_cleanup_suppressed", 1)
        if not _unsafe_delete_trajectory_warned:
            logger.warning(
                "Ignoring unsafe delete_hspec_trajectory(); use delete_hspec_segment() "
                "after epoch-level HSpec GC."
            )
            _unsafe_delete_trajectory_warned = True
        return False

    desc = coerce_hspec_desc(desc_obj)
    for path_str in (desc.hs_path, desc.token_path):
        try:
            path = Path(path_str)
            if path.exists():
                path.unlink()
            parent = path.parent
            while parent != parent.parent and parent.name.startswith(("batch_", "pid_", "worker_", "node_")):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        except Exception:
            logger.debug("Failed to delete HSpec trajectory file %s", path_str, exc_info=True)
            return False
    return True
