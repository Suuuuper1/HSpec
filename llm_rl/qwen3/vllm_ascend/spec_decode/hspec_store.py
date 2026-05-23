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
import socket
import tempfile
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_store_metrics_lock = threading.Lock()
_store_metrics: Dict[str, int] = {
    "raw_store_bytes": 0,
    "desc_count": 0,
    "collect_dropped": 0,
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


def get_hspec_raw_store_max_bytes() -> int:
    value = os.getenv("HSPEC_RAW_STORE_MAX_BYTES", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_RAW_STORE_MAX_BYTES=%s", value)
        return 0


def get_hspec_raw_store_max_files() -> int:
    value = os.getenv("HSPEC_RAW_STORE_MAX_FILES", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_RAW_STORE_MAX_FILES=%s", value)
        return 0


def get_hspec_store_retain_batches() -> int:
    value = os.getenv("HSPEC_STORE_RETAIN_BATCHES", "128")
    try:
        return max(int(value), 1)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_STORE_RETAIN_BATCHES=%s", value)
        return 128


def _metric_add(name: str, value: int) -> None:
    with _store_metrics_lock:
        _store_metrics[name] = _store_metrics.get(name, 0) + int(value)


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


class HSpecLocalCollector:
    """Process-local writer for HSpec rollout trajectories.

    Hidden states are written as fp16 rows. Tokens are written as int32. Files
    are per-request, which keeps each descriptor expressible as one contiguous
    segment even when decode steps interleave requests.
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

    def _batch_dir(self) -> Path:
        path = (
            self.store_root
            / f"node_{self.node_id}"
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

    def _enforce_store_budget(self) -> None:
        max_bytes = get_hspec_raw_store_max_bytes()
        max_files = get_hspec_raw_store_max_files()
        if max_bytes <= 0 and max_files <= 0:
            return

        try:
            batch_dirs = sorted(
                [p for p in self.store_root.rglob("batch_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
        except Exception:
            logger.debug("Failed to list HSpec batch directories for GC", exc_info=True)
            return

        retain = get_hspec_store_retain_batches()
        protected = set(batch_dirs[-retain:])

        def current_bytes() -> int:
            total = 0
            for batch_dir in batch_dirs:
                if not batch_dir.exists():
                    continue
                for file_path in batch_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            total += file_path.stat().st_size
                        except FileNotFoundError:
                            continue
            return total

        total_bytes = current_bytes()
        total_files = self._count_existing_files()
        for batch_dir in batch_dirs:
            if batch_dir in protected:
                continue
            if (max_bytes <= 0 or total_bytes <= max_bytes) and (max_files <= 0 or total_files <= max_files):
                break
            removed_bytes = 0
            removed_files = 0
            for file_path in sorted(batch_dir.rglob("*"), reverse=True):
                if not file_path.exists():
                    continue
                if file_path.is_file():
                    try:
                        removed_bytes += file_path.stat().st_size
                    except FileNotFoundError:
                        pass
                    try:
                        file_path.unlink()
                        removed_files += 1
                    except FileNotFoundError:
                        pass
                elif file_path.is_dir():
                    try:
                        file_path.rmdir()
                    except OSError:
                        pass
            try:
                batch_dir.rmdir()
            except OSError:
                pass
            total_bytes = max(total_bytes - removed_bytes, 0)
            total_files = max(total_files - removed_files, 0)

    @staticmethod
    def _safe_request_key(req_id: str) -> str:
        digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).hexdigest()
        return f"r{digest}"

    def _state_for_req(self, req_id: str) -> Dict[str, Any]:
        req_id = str(req_id)
        state = self._req_states.get(req_id)
        if state is not None:
            return state

        req_key = self._safe_request_key(req_id)
        base = self._batch_dir() / req_key
        state = {
            "request_id": req_id,
            "hs_path": str(base.with_suffix(".hs.fp16.bin")),
            "token_path": str(base.with_suffix(".tokens.i32.bin")),
            "hs_fh": None,
            "token_fh": None,
            "hidden_dim": 0,
            "hs_rows": 0,
            "token_len": 0,
        }
        self._req_states[req_id] = state
        return state

    @staticmethod
    def _close_state_files(state: Dict[str, Any]) -> None:
        for key in ("hs_fh", "token_fh"):
            fh = state.get(key)
            if fh is None:
                continue
            try:
                fh.close()
            except Exception:
                logger.debug("Failed to close HSpec collector file handle", exc_info=True)
            finally:
                state[key] = None

    def clear_batch(self) -> None:
        """Forget in-flight request state without deleting flushed files."""
        with self._lock:
            for state in self._req_states.values():
                self._close_state_files(state)
            self._req_states.clear()
            self._batch_counter += 1

    def append_hidden_rows(self, req_id: str, rows: torch.Tensor) -> None:
        if rows is None or rows.numel() == 0:
            return
        if rows.ndim != 2:
            raise ValueError(f"HSpec hidden rows must be 2-D, got {tuple(rows.shape)}")

        rows_cpu = rows.detach().to(device="cpu", dtype=torch.float16).contiguous()
        rows_np = rows_cpu.numpy()
        hidden_dim = int(rows_np.shape[1])

        with self._lock:
            state = self._state_for_req(str(req_id))
            if state["hidden_dim"] == 0:
                state["hidden_dim"] = hidden_dim
            elif int(state["hidden_dim"]) != hidden_dim:
                raise ValueError(
                    f"HSpec hidden dim mismatch for {req_id}: "
                    f"{state['hidden_dim']} vs {hidden_dim}"
                )
            if state.get("hs_fh") is None:
                state["hs_fh"] = open(state["hs_path"], "ab")
            state["hs_fh"].write(rows_np.tobytes(order="C"))
            state["hs_rows"] += int(rows_np.shape[0])
            _metric_add("raw_store_bytes", int(rows_np.nbytes))

    def extend_tokens(self, req_id: str, token_ids: Iterable[int]) -> None:
        token_list = [int(t) for t in token_ids]
        if not token_list:
            return
        token_np = np.ascontiguousarray(token_list, dtype=np.int32)

        with self._lock:
            state = self._state_for_req(str(req_id))
            if state.get("token_fh") is None:
                state["token_fh"] = open(state["token_path"], "ab")
            state["token_fh"].write(token_np.tobytes(order="C"))
            state["token_len"] += int(token_np.shape[0])
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
            states = self._req_states
            self._req_states = {}
            self._batch_counter += 1

        for req_id, state in states.items():
            self._close_state_files(state)
            hs_rows = int(state.get("hs_rows", 0))
            token_len = int(state.get("token_len", 0))
            hidden_dim = int(state.get("hidden_dim", 0))
            if hs_rows <= 0 or token_len <= 0:
                _metric_add("collect_dropped", 1)
                continue
            if hs_rows != token_len:
                logger.warning(
                    "HSpec descriptor alignment mismatch: req_id=%s hs_rows=%d token_len=%d; dropping",
                    req_id,
                    hs_rows,
                    token_len,
                )
                _metric_add("collect_dropped", 1)
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
                hs_offset_rows=0,
                token_path=str(state["token_path"]),
                token_offset=0,
                length=hs_rows,
                hidden_dim=hidden_dim,
                hs_dtype="float16",
                token_dtype="int32",
                reward=None,
            )
            descs[req_id] = desc
            self._append_manifest(desc)

        _metric_add("desc_count", len(descs))
        self._enforce_store_budget()
        return descs

    def _append_manifest(self, desc: HSpecTrajectoryDesc) -> None:
        try:
            manifest_path = Path(desc.hs_path).with_name("desc.jsonl")
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


def load_hspec_trajectory(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mmap-backed hidden states and token ids for a descriptor."""
    desc = coerce_hspec_desc(desc_obj)
    if desc.length <= 0:
        raise ValueError(f"HSpec descriptor has non-positive length: {desc.length}")
    if desc.hidden_dim <= 0:
        raise ValueError(f"HSpec descriptor has invalid hidden_dim: {desc.hidden_dim}")

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


def delete_hspec_trajectory(
    desc_obj: HSpecTrajectoryDesc | Dict[str, Any],
) -> None:
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
