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
    return 5


def get_hspec_store_root() -> str:
    return os.getenv(
        "HSPEC_STORE_DIR",
        os.path.join(tempfile.gettempdir(), "hspec_store"),
    )


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
        self.tp_group_id = int(os.getenv("HSPEC_TP_GROUP_ID", "0"))
        self.store_root = Path(get_hspec_store_root()).resolve()
        self._lock = threading.Lock()
        self._batch_counter = 0
        self._req_states: Dict[str, Dict[str, Any]] = {}

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
                continue
            if hs_rows != token_len:
                logger.warning(
                    "HSpec descriptor alignment mismatch: req_id=%s hs_rows=%d token_len=%d; dropping",
                    req_id,
                    hs_rows,
                    token_len,
                )
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
