# Copyright 2025 HSpec Authors
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
HSpec: Hidden State based Speculative Decoding query table.
"""

import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import ray
import zmq
from threadpoolctl import threadpool_limits

from vllm_ascend.spec_decode.hspec_utils import (
    PromptPCAParams,
    fit_pca_multi_sequence,
    fit_pca_single_sequence,
    stable_partition_id,
)
from vllm_ascend.spec_decode.hspec_store import (
    HSpecSegmentKey,
    HSpecTrajectoryDesc,
    coerce_hspec_desc,
    collect_hspec_store_metrics,
    estimate_hspec_trajectory_bytes,
    get_hspec_build_actor_name_prefix,
    get_hspec_build_actor_num_cpus,
    get_hspec_build_blas_threads,
    get_hspec_node_id,
    get_hspec_num_shards,
    get_hspec_store_root,
    get_hspec_table_store_root,
    hspec_legacy_dataproto_hs_enabled,
    hspec_record_store_metric,
    hspec_segment_key_from_desc,
    hspec_single_node_only_enabled,
    hspec_strict_descriptor_mode_enabled,
    hspec_topology_strict_enabled,
    load_hspec_trajectory,
)

logger = logging.getLogger(__name__)
_unsafe_descriptor_cleanup_warned = False
_LEGACY_HSPEC_PAYLOAD_KEYS = frozenset({"hidden_states", "tokens", "rewards", "prompt_token_ids"})


@dataclass(frozen=True)
class HSpecBuildSubmission:
    ref: ray.ObjectRef
    shard_id: int
    segments: frozenset[HSpecSegmentKey]
    prompt_ids: tuple[str, ...]
    legacy: bool = False


@dataclass(frozen=True)
class HSpecBuildActorTopology:
    actor_name: str
    shard_id: int
    num_groups: int
    logical_node_id: str
    hostname: str
    pid: int
    ray_node_id: str
    table_store_root: str
    table_store_base_root: str
    hspec_store_root: str
    similarity_threshold: float
    max_entries_per_prompt: int
    n_components: int
    build_actor_num_cpus: float
    build_blas_threads: int


def _build_actor_name(shard_id: int) -> str:
    prefix = get_hspec_build_actor_name_prefix()
    return f"{prefix}_{int(shard_id)}"


def _actor_topology_error(
    message: str,
    *,
    topologies: List[Dict[str, Any]],
    expected_num_groups: int,
    expected_node_id: str,
) -> RuntimeError:
    summary = [
        {
            "actor_name": item.get("actor_name"),
            "shard_id": item.get("shard_id"),
            "num_groups": item.get("num_groups"),
            "logical_node_id": item.get("logical_node_id"),
            "ray_node_id": item.get("ray_node_id"),
            "table_store_root": item.get("table_store_root"),
            "table_store_base_root": item.get("table_store_base_root"),
            "build_actor_num_cpus": item.get("build_actor_num_cpus"),
            "build_blas_threads": item.get("build_blas_threads"),
            "similarity_threshold": item.get("similarity_threshold"),
            "max_entries_per_prompt": item.get("max_entries_per_prompt"),
            "n_components": item.get("n_components"),
        }
        for item in topologies
    ]
    return RuntimeError(
        "HSpec build actor topology mismatch: "
        f"{message}; expected_num_groups={expected_num_groups}, "
        f"expected_node_id={expected_node_id!r}, "
        f"actor_name_prefix={get_hspec_build_actor_name_prefix()!r}, "
        f"topologies={summary}. "
        "This usually means stale Ray named actors or inconsistent HSPEC_NUM_SHARDS. "
        "Restart Ray or set a unique HSPEC_BUILD_ACTOR_NAME_PREFIX."
    )


def _validate_actor_topologies(
    topologies: List[Dict[str, Any]],
    *,
    expected_num_groups: int,
    expected_node_id: str,
    expected_similarity_threshold: Optional[float] = None,
    expected_max_entries_per_prompt: Optional[int] = None,
    expected_n_components: Optional[int] = None,
) -> None:
    if len(topologies) != expected_num_groups:
        hspec_record_store_metric("topology_actor_init_error", 1)
        raise _actor_topology_error(
            f"actor_count={len(topologies)}",
            topologies=topologies,
            expected_num_groups=expected_num_groups,
            expected_node_id=expected_node_id,
        )

    shard_ids: List[int] = []
    for item in topologies:
        try:
            shard_ids.append(int(item.get("shard_id")))
        except Exception as exc:
            hspec_record_store_metric("topology_actor_shard_mismatch", 1)
            raise _actor_topology_error(
                f"invalid shard_id in topology: {exc}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            ) from exc

    expected_shards = list(range(expected_num_groups))
    if sorted(shard_ids) != expected_shards:
        hspec_record_store_metric("topology_actor_shard_mismatch", 1)
        raise _actor_topology_error(
            f"shard_ids={sorted(shard_ids)} expected={expected_shards}",
            topologies=topologies,
            expected_num_groups=expected_num_groups,
            expected_node_id=expected_node_id,
        )

    for item in topologies:
        shard_id = int(item["shard_id"])
        if int(item.get("num_groups", -1)) != expected_num_groups:
            hspec_record_store_metric("topology_actor_num_groups_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports num_groups={item.get('num_groups')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if str(item.get("actor_name")) != _build_actor_name(shard_id):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports actor_name={item.get('actor_name')!r}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if hspec_single_node_only_enabled() and str(item.get("logical_node_id")) != str(expected_node_id):
            hspec_record_store_metric("topology_actor_node_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports node_id={item.get('logical_node_id')!r}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if int(item.get("build_blas_threads", 0)) < 1:
            hspec_record_store_metric("topology_actor_init_error", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports invalid build_blas_threads={item.get('build_blas_threads')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if str(item.get("hspec_store_root")) != str(get_hspec_store_root()):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports hspec_store_root={item.get('hspec_store_root')!r}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if str(item.get("table_store_base_root")) != str(get_hspec_table_store_root()):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports table_store_base_root={item.get('table_store_base_root')!r}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if expected_n_components is not None and int(item.get("n_components", -1)) != int(expected_n_components):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports n_components={item.get('n_components')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if (
            expected_max_entries_per_prompt is not None
            and int(item.get("max_entries_per_prompt", -1)) != int(expected_max_entries_per_prompt)
        ):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports max_entries_per_prompt={item.get('max_entries_per_prompt')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if expected_similarity_threshold is not None:
            try:
                actor_threshold = float(item.get("similarity_threshold"))
            except Exception as exc:
                hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
                raise _actor_topology_error(
                    f"actor shard={shard_id} reports invalid similarity_threshold={item.get('similarity_threshold')}",
                    topologies=topologies,
                    expected_num_groups=expected_num_groups,
                    expected_node_id=expected_node_id,
                ) from exc
            if abs(actor_threshold - float(expected_similarity_threshold)) > 1e-12:
                hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
                raise _actor_topology_error(
                    f"actor shard={shard_id} reports similarity_threshold={actor_threshold}",
                    topologies=topologies,
                    expected_num_groups=expected_num_groups,
                    expected_node_id=expected_node_id,
                )


def _get_actor_topologies(handles: List[ray.actor.ActorHandle]) -> List[Dict[str, Any]]:
    try:
        return [dict(item) for item in ray.get([handle.get_topology.remote() for handle in handles])]
    except Exception:
        hspec_record_store_metric("topology_actor_init_error", 1)
        logger.exception("Failed to fetch HSpec build actor topology")
        raise


def _looks_like_legacy_hspec_payload(data: Any) -> bool:
    return isinstance(data, dict) and bool(_LEGACY_HSPEC_PAYLOAD_KEYS.intersection(data.keys()))


def _raise_legacy_payload_forbidden(prompt_id: str, data: Any) -> None:
    if _looks_like_legacy_hspec_payload(data):
        hspec_record_store_metric("strict_descriptor_violation", 1)
        raise ValueError(
            "Legacy HSpec ndarray payload is forbidden in descriptor build API "
            "when HSPEC_LEGACY_DATAPROTO_HS=0. "
            f"prompt_id={prompt_id!r}; use build_tables_batch_legacy() only for explicit A/B."
        )
    raise TypeError(
        "HSpec descriptor build API expects a list of HSpecTrajectoryDesc "
        f"for prompt_id={prompt_id!r}, got {type(data)!r}"
    )


def _maybe_process_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return -1.0


# Per-prompt table data  (continuous-array, reference-value storage)


class PromptTableData:

    __slots__ = (
        "pca_params",
        "keys",
        "rollout_seqs",
        "entry_rollout_idx",
        "entry_offset",
        "rewards",
        "n_entries",
        "max_entries",
        "wnd_size",
        "max_wnd",
        "min_wnd",
    )

    def __init__(
        self,
        pca_params: PromptPCAParams,
        max_entries: int = 10_000,
        initial_wnd: int = 8,
        max_wnd: int = 28,
        min_wnd: int = 2,
    ):
        self.pca_params = pca_params
        self.max_entries = max_entries
        num_components = pca_params.n_components

        # Pre-allocate contiguous arrays (compacted after build)
        self.keys = np.empty((max_entries, num_components), dtype=np.float16)
        self.entry_rollout_idx = np.empty(max_entries, dtype=np.int32)
        self.entry_offset = np.empty(max_entries, dtype=np.int32)
        self.rewards = np.empty(max_entries, dtype=np.float32)
        self.rollout_seqs: List[np.ndarray] = []
        self.n_entries = 0

        # Adaptive window control
        self.wnd_size = initial_wnd
        self.max_wnd = max_wnd
        self.min_wnd = min_wnd

    # build

    def add_rollout(
        self,
        projected_keys: np.ndarray,
        token_sequence: np.ndarray,
        reward: float = 0.0,
    ) -> int:
        """Append entries from one rollout with **pre-projected** keys.

        Args:
            projected_keys:  (L, K)  float – output of PromptPCAParams.project()
            token_sequence:  (L,)    int32 – response tokens y[0 .. L-1]
            reward:          scalar reward for this rollout

        Returns:
            Number of entries actually written (may be < L when table is full).
        """
        sequence_len = len(token_sequence)
        room = self.max_entries - self.n_entries
        # Value shift (critical): entry at position t stores value starting from
        # y[t+1:], so the draft produced *after* accepting y[t] begins at the
        # next token and does not repeat the just-accepted token.
        if room <= 0 or sequence_len <= 1:
            return 0
        n_add = min(sequence_len - 1, room)

        rollout_idx = len(self.rollout_seqs)
        self.rollout_seqs.append(np.ascontiguousarray(token_sequence[:sequence_len], dtype=np.int32))

        # Store projected keys (raw dot-product similarity).
        keys_fp32 = projected_keys[:n_add].astype(np.float32, copy=False)
        # norms = np.linalg.norm(kf, axis=1, keepdims=True)
        # np.maximum(norms, 1e-8, out=norms)
        # kf /= norms
        start = self.n_entries
        end = start + n_add
        self.keys[start:end] = keys_fp32.astype(np.float16)
        self.entry_rollout_idx[start:end] = rollout_idx
        # Value shift: offset starts at 1 (y[1]) for t=0, ... , y[L-1] is dropped.
        self.entry_offset[start:end] = np.arange(1, n_add + 1, dtype=np.int32)
        self.rewards[start:end] = reward
        self.n_entries = end
        return n_add

    def compact(self):
        """Shrink pre-allocated arrays to populated size (saves memory)."""
        n_entries = self.n_entries
        if n_entries < self.max_entries:
            self.keys = np.ascontiguousarray(self.keys[:n_entries])
            self.entry_rollout_idx = np.ascontiguousarray(self.entry_rollout_idx[:n_entries])
            self.entry_offset = np.ascontiguousarray(self.entry_offset[:n_entries])
            self.rewards = np.ascontiguousarray(self.rewards[:n_entries])
            self.max_entries = n_entries

    def query(
        self,
        query_z: np.ndarray,
        threshold: float,
        accept_length: int = 1,
    ) -> Tuple[List[int], float]:
        """Find best match and return draft tokens.

        Args:
            query_z:            (K,) PCA-projected query (no normalisation).
            threshold:          raw dot-product threshold.
            accept_length:      previous accept length (for window control).

        Returns:
            (draft_tokens, best_similarity).
        """
        if self.n_entries == 0:
            return [], 0.0
        self._update_wnd(accept_length)

        # Dot-product similarity
        sims = self.keys[:self.n_entries].astype(np.float32).dot(
            query_z.astype(np.float32, copy=False))

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            draft = self.get_draft_tokens(best_idx, self.wnd_size)
            return draft, best_sim
        return [], best_sim

    def get_draft_tokens(self, entry_idx: int, max_tokens: int) -> List[int]:
        """Retrieve draft tokens via reference  (O(1) slice)."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return []
        rollout_idx = int(self.entry_rollout_idx[entry_idx])
        offset = int(self.entry_offset[entry_idx])
        seq = self.rollout_seqs[rollout_idx]
        return seq[offset:offset + max_tokens].tolist()

    def _update_wnd(self, accept_length: int):
        if accept_length >= self.wnd_size:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)
        elif accept_length <= 1:
            self.wnd_size = max(self.wnd_size // 2, self.min_wnd)

    # accessors

    def get_pca_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, components) as contiguous float32 numpy."""
        return self.pca_params.mean, self.pca_params.components

    def get_keys_numpy(self) -> np.ndarray:
        """Return active keys as contiguous (n_entries, K) float16."""
        return np.ascontiguousarray(self.keys[:self.n_entries])


# Partitioned Ray actor

def _resolve_num_groups() -> int:
    return get_hspec_num_shards()


_num_groups: int = _resolve_num_groups()


@ray.remote(num_cpus=1)
class HSpecTableGroup:
    """Ray actor managing HSpec build/query state for one shard."""

    def __init__(
        self,
        port: int = 6555,
        similarity_threshold: float = 0.9,
        max_entries_per_prompt: int = 10_000,
        n_components: int = 64,
        shard_id: int = 0,
    ):
        # Double-buffered tables
        #   _active  : read-only during decode (online query)
        #   _building: write-only during build phase
        #   swap()   : building → active at epoch boundary
        self._active: Dict[str, PromptTableData] = {}
        self._building: Dict[str, PromptTableData] = {}
        self._active_version: int = 0

        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries_per_prompt
        self.n_components = n_components
        self.port = port
        self.shard_id = int(shard_id)
        self.num_groups = _resolve_num_groups()
        self.logical_node_id = get_hspec_node_id()
        self.build_actor_num_cpus = get_hspec_build_actor_num_cpus()
        self.build_blas_threads = get_hspec_build_blas_threads()
        self.actor_name = _build_actor_name(self.shard_id)
        try:
            self.ray_node_id = str(ray.get_runtime_context().get_node_id())
        except Exception:
            self.ray_node_id = ""
        self.table_store_root = os.path.join(
            get_hspec_table_store_root(),
            f"shard_{self.shard_id:03d}",
        )
        os.makedirs(self.table_store_root, exist_ok=True)
        self._active_version_path = os.path.join(self.table_store_root, "active_version.json")
        self._build_queue_max_descs = max(
            int(os.getenv("HSPEC_BUILD_QUEUE_MAX_DESCS", "0")),
            0,
        )
        self._build_queue_max_bytes = max(
            int(os.getenv("HSPEC_BUILD_QUEUE_MAX_BYTES", "0")),
            0,
        )
        self._build_pending_descs = 0
        self._build_pending_bytes = 0

        # Metrics (track queries on active tables)
        self._query_count = 0
        self._match_count = 0
        self._total_draft_len = 0
        self._build_count = 0
        self._discard_count = 0
        # Verification metrics (post rejection-sampling)
        # verify_times: number of requests that entered verification with a
        # non-empty draft (i.e., spec decode attempted).
        # accept_times: number of those requests whose accepted prefix length >= 1.
        # accept_length_sum: sum(accepted_prefix_len) over accept_times.
        self._verify_count = 0
        self._accept_count = 0
        self._accept_len_sum = 0
        self._accept_advan_count = 0
        self._reject_advan_count = 0
        # Entry-position study metrics (reported asynchronously from the
        # worker-local proposer).
        self._entry_match_count = 0
        self._entry_delta_sum = 0
        self._entry_abs_delta_sum = 0
        self._entry_verify_count = 0
        self._entry_accept_count = 0
        self._entry_accept_len_sum = 0
        self._entry_abs_delta_verify: Dict[int, int] = {}
        self._entry_abs_delta_accept: Dict[int, int] = {}
        self._entry_abs_delta_accept_len_sum: Dict[int, int] = {}

        # ZMQ state
        self.running = False

    def get_topology(self) -> Dict[str, Any]:
        return {
            "actor_name": str(self.actor_name),
            "shard_id": int(self.shard_id),
            "num_groups": int(self.num_groups),
            "logical_node_id": str(self.logical_node_id),
            "hostname": socket.gethostname(),
            "pid": int(os.getpid()),
            "ray_node_id": str(self.ray_node_id),
            "table_store_root": str(self.table_store_root),
            "table_store_base_root": str(get_hspec_table_store_root()),
            "hspec_store_root": str(get_hspec_store_root()),
            "similarity_threshold": float(self.similarity_threshold),
            "max_entries_per_prompt": int(self.max_entries),
            "n_components": int(self.n_components),
            "build_actor_num_cpus": float(self.build_actor_num_cpus),
            "build_blas_threads": int(self.build_blas_threads),
        }

    def _can_accept_descriptor_batch(self, descs: List[HSpecTrajectoryDesc]) -> bool:
        pending_descs = self._build_pending_descs + len(descs)
        pending_bytes = self._build_pending_bytes + sum(estimate_hspec_trajectory_bytes(desc) for desc in descs)
        if self._build_queue_max_descs > 0 and pending_descs > self._build_queue_max_descs:
            return False
        if self._build_queue_max_bytes > 0 and pending_bytes > self._build_queue_max_bytes:
            return False
        return True

    def _mark_descriptor_batch_pending(self, descs: List[HSpecTrajectoryDesc]) -> None:
        self._build_pending_descs += len(descs)
        self._build_pending_bytes += sum(estimate_hspec_trajectory_bytes(desc) for desc in descs)

    def _mark_descriptor_batch_finished(self, descs: List[HSpecTrajectoryDesc]) -> None:
        self._build_pending_descs = max(self._build_pending_descs - len(descs), 0)
        self._build_pending_bytes = max(
            self._build_pending_bytes - sum(estimate_hspec_trajectory_bytes(desc) for desc in descs),
            0,
        )

    def _load_prompt_build_inputs_from_descs(
        self,
        prompt_id: str,
        descs: List[HSpecTrajectoryDesc],
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        hidden_states_list: List[np.ndarray] = []
        token_seq_list: List[np.ndarray] = []
        rewards: List[float] = []

        for desc in descs:
            if desc is None:
                self._discard_count += 1
                continue
            if not self._validate_descriptor_topology(prompt_id, desc):
                self._discard_count += 1
                continue
            try:
                hs, tokens = load_hspec_trajectory(desc)
            except Exception as exc:
                logger.warning(
                    "Failed to load HSpec trajectory for prompt_id=%s request_id=%s: %s",
                    prompt_id,
                    desc.request_id,
                    exc,
                )
                self._discard_count += 1
                continue
            if hs is None or tokens is None:
                self._discard_count += 1
                continue
            hidden_states_list.append(np.asarray(hs, dtype=np.float32))
            token_seq_list.append(np.asarray(tokens, dtype=np.int32))
            rewards.append(float(desc.reward or 0.0))

        return hidden_states_list, token_seq_list, rewards

    def _cleanup_trajectory_descs(self, descs: List[HSpecTrajectoryDesc]) -> None:
        cleanup = os.getenv("HSPEC_DELETE_TRAJECTORY_AFTER_BUILD", "0") != "0"
        if not cleanup or not descs:
            return
        hspec_record_store_metric("unsafe_descriptor_cleanup_suppressed", len(descs))
        global _unsafe_descriptor_cleanup_warned
        if not _unsafe_descriptor_cleanup_warned:
            logger.warning(
                "HSPEC_DELETE_TRAJECTORY_AFTER_BUILD is disabled for shared HSpec "
                "segment store; raw files are cleaned only by segment/epoch GC."
            )
            _unsafe_descriptor_cleanup_warned = True

    def _validate_descriptor_topology(
        self,
        prompt_id: str,
        desc: HSpecTrajectoryDesc,
    ) -> bool:
        expected_shard = stable_partition_id(prompt_id, self.num_groups)
        if desc.prompt_id and str(desc.prompt_id) != str(prompt_id):
            hspec_record_store_metric("descriptor_prompt_mismatch", 1)
            hspec_record_store_metric("descriptor_topology_violation", 1)
            logger.warning(
                "HSpec descriptor prompt mismatch on actor: prompt_id=%s desc.prompt_id=%s "
                "request_id=%s shard=%s expected_shard=%s",
                prompt_id,
                desc.prompt_id,
                desc.request_id,
                self.shard_id,
                expected_shard,
            )
            return False
        if hspec_single_node_only_enabled():
            actor_node = self.logical_node_id
            if str(desc.node_id) != str(actor_node):
                hspec_record_store_metric("descriptor_node_mismatch", 1)
                hspec_record_store_metric("descriptor_topology_violation", 1)
                logger.warning(
                    "HSpec descriptor node mismatch in single-node mode: prompt_id=%s "
                    "request_id=%s desc.node_id=%s actor.node_id=%s desc.shard_id=%s actor.shard_id=%s expected=%s",
                    prompt_id,
                    desc.request_id,
                    desc.node_id,
                    actor_node,
                    desc.shard_id,
                    self.shard_id,
                    expected_shard,
                )
                return False
        if int(desc.shard_id) != self.shard_id:
            hspec_record_store_metric("descriptor_shard_mismatch", 1)
            hspec_record_store_metric("descriptor_topology_violation", 1)
            logger.warning(
                "HSpec descriptor shard mismatch: prompt_id=%s request_id=%s "
                "desc.shard_id=%s actor.shard_id=%s expected=%s",
                prompt_id,
                desc.request_id,
                desc.shard_id,
                self.shard_id,
                expected_shard,
            )
            return False
        if int(desc.shard_id) != int(expected_shard):
            hspec_record_store_metric("descriptor_shard_mismatch", 1)
            hspec_record_store_metric("descriptor_topology_violation", 1)
            logger.warning(
                "HSpec descriptor stable shard mismatch: prompt_id=%s request_id=%s "
                "desc.shard_id=%s actor.shard_id=%s expected=%s",
                prompt_id,
                desc.request_id,
                desc.shard_id,
                self.shard_id,
                expected_shard,
            )
            return False
        return True

    def _write_active_version_manifest(self) -> None:
        payload = {
            "active_version": int(self._active_version),
            "shard_id": int(self.shard_id),
            "num_prompts": int(len(self._active)),
            "total_entries": int(self.total_entries()),
        }
        try:
            with open(self._active_version_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Failed to write HSpec active version manifest", exc_info=True)

    # Build

    def build_prompt_table(
        self,
        prompt_id: str,
        hidden_states_list: List[np.ndarray],
        token_seq_list: List[Any],
        rewards: List[float],
    ):
        """Build a complete table for one prompt.

        Pipeline:  validate → PCA fit → project → store entries.

        Args:
            prompt_id:          Stable prompt identifier.
            hidden_states_list: [(L_i, D) ndarray] per rollout.
            token_seq_list:     [list[int] | ndarray] per rollout.
            rewards:            [float] per rollout.
        """
        # ① Validate & filter
        valid_hs: List[np.ndarray] = []
        valid_tok: List[Any] = []
        valid_rew: List[float] = []
        for hs, tok, rew in zip(hidden_states_list, token_seq_list, rewards):
            if hs is None or len(tok) == 0:
                self._discard_count += 1
                continue
            if hs.ndim != 2:
                self._discard_count += 1
                continue
            if hs.shape[0] != len(tok):
                logger.warning(
                    "HSpec alignment mismatch for %s: hs=%d vs tok=%d, discarding this trajectory",
                    prompt_id,
                    hs.shape[0],
                    len(tok),
                )
                self._discard_count += 1
                continue
            valid_hs.append(hs if hs.dtype == np.float32 else hs.astype(np.float32))
            valid_tok.append(tok)
            valid_rew.append(rew)

        if not valid_hs:
            return

        # PCA fit  (single-sequence for PPO, multi for GRPO)
        num_components = self.n_components
        try:
            if len(valid_hs) == 1:
                pca_params, proj = fit_pca_single_sequence(prompt_id, valid_hs[0], num_components)
                proj_list = [proj]
            else:
                pca_params, proj_list = fit_pca_multi_sequence(prompt_id, valid_hs, num_components)
        except Exception as exc:
            logger.warning("HSpec PCA failed for %s: %s", prompt_id, exc)
            self._discard_count += len(valid_hs)
            return

        # Create table & populate with projected keys + token refs
        table = PromptTableData(pca_params=pca_params, max_entries=self.max_entries)
        for proj, tok, rew in zip(proj_list, valid_tok, valid_rew):
            tok_arr = (np.asarray(tok, dtype=np.int32)
                       if not isinstance(tok, np.ndarray)
                       else tok.astype(np.int32, copy=False))
            table.add_rollout(proj, tok_arr, rew)

        table.compact()
        self._building[prompt_id] = table
        self._build_count += 1

    def _build_result_metrics(
        self,
        *,
        t0: float,
        prompt_count: int,
        desc_count: int,
        legacy_payload_count: int,
        build_count_before: int,
        discard_count_before: int,
    ) -> Dict[str, float]:
        return {
            "shard_id": float(self.shard_id),
            "prompt_count": float(prompt_count),
            "desc_count": float(desc_count),
            "legacy_payload_count": float(legacy_payload_count),
            "build_count_delta": float(self._build_count - build_count_before),
            "discard_count_delta": float(self._discard_count - discard_count_before),
            "build_total_ms": float((time.perf_counter() - t0) * 1000.0),
            "build_actor_rss_mb": _maybe_process_rss_mb(),
        }

    def build_tables_batch(
        self,
        prompt_data_dict: Dict[str, List[HSpecTrajectoryDesc | Dict[str, Any]]],
    ) -> Dict[str, float]:
        """Build descriptor-only prompt payloads for one partition.

        Payload shape is ``{prompt_id: [HSpecTrajectoryDesc | dict, ...]}``.
        Legacy ndarray dict payloads must use ``build_tables_batch_legacy``.
        """
        t0 = time.perf_counter()
        prompt_count = 0
        desc_count = 0
        build_count_before = self._build_count
        discard_count_before = self._discard_count
        with threadpool_limits(limits=self.build_blas_threads):
            for prompt_id, data in prompt_data_dict.items():
                if not isinstance(data, list):
                    if hspec_strict_descriptor_mode_enabled() or _looks_like_legacy_hspec_payload(data):
                        _raise_legacy_payload_forbidden(prompt_id, data)
                    _raise_legacy_payload_forbidden(prompt_id, data)
                descs = [coerce_hspec_desc(item) for item in data if item is not None]
                if not descs:
                    continue
                prompt_count += 1
                desc_count += len(descs)
                hspec_record_store_metric("descriptor_payload_count", len(descs))
                if not self._can_accept_descriptor_batch(descs):
                    logger.warning(
                        "HSpec build queue budget exceeded on shard=%s, dropping %s descriptors for prompt_id=%s",
                        self.shard_id,
                        len(descs),
                        prompt_id,
                    )
                    self._discard_count += len(descs)
                    continue
                self._mark_descriptor_batch_pending(descs)
                try:
                    hidden_states_list, token_seq_list, rewards = self._load_prompt_build_inputs_from_descs(
                        prompt_id,
                        descs,
                    )
                    self.build_prompt_table(
                        prompt_id,
                        hidden_states_list,
                        token_seq_list,
                        rewards,
                    )
                finally:
                    self._cleanup_trajectory_descs(descs)
                    self._mark_descriptor_batch_finished(descs)

        return self._build_result_metrics(
            t0=t0,
            prompt_count=prompt_count,
            desc_count=desc_count,
            legacy_payload_count=0,
            build_count_before=build_count_before,
            discard_count_before=discard_count_before,
        )

    def build_tables_batch_legacy(self, prompt_data_dict: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Build old ndarray payloads for explicit HSPEC_LEGACY_DATAPROTO_HS=1 A/B only."""
        if not hspec_legacy_dataproto_hs_enabled():
            hspec_record_store_metric("strict_descriptor_violation", 1)
            raise RuntimeError("build_tables_batch_legacy() requires HSPEC_LEGACY_DATAPROTO_HS=1")

        t0 = time.perf_counter()
        prompt_count = 0
        legacy_payload_count = 0
        build_count_before = self._build_count
        discard_count_before = self._discard_count
        with threadpool_limits(limits=self.build_blas_threads):
            for prompt_id, data in prompt_data_dict.items():
                if not _looks_like_legacy_hspec_payload(data):
                    raise TypeError(
                        "Legacy HSpec build API expects dict payload with "
                        f"hidden_states/tokens/rewards for prompt_id={prompt_id!r}"
                    )
                prompt_count += 1
                hidden_states = data["hidden_states"]
                legacy_payload_count += len(hidden_states) if hasattr(hidden_states, "__len__") else 1
                self.build_prompt_table(
                    prompt_id,
                    hidden_states,
                    data["tokens"],
                    data["rewards"],
                )
        hspec_record_store_metric("legacy_payload_count", legacy_payload_count)
        return self._build_result_metrics(
            t0=t0,
            prompt_count=prompt_count,
            desc_count=0,
            legacy_payload_count=legacy_payload_count,
            build_count_before=build_count_before,
            discard_count_before=discard_count_before,
        )

    # Query

    def query(
        self,
        prompt_id: str,
        hidden_state: np.ndarray,
        accept_length: int = 1,
    ) -> List[int]:
        """Query with a *raw* (D-dim) hidden state.

        Internally projects via the prompt's PCA params, then matches
        against the stored PCA-projected keys.
        """
        self._query_count += 1
        if prompt_id not in self._active:
            return []

        table = self._active[prompt_id]
        # Project  → (K,)
        z = table.pca_params.project(hidden_state.reshape(1, -1).astype(np.float32, copy=False)).squeeze(0)

        draft, sim = table.query(z, self.similarity_threshold, accept_length)
        if draft:
            self._match_count += 1
            self._total_draft_len += len(draft)
        return draft

    def query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query for multiple prompts."""
        return [
            self.query(pid, hs, al)
            for pid, hs, al in zip(prompt_id_list, hidden_state_list, accept_length_list)
        ]

    # Table data access  (for proposer prefetch / cache)

    def get_prompt_pca(self, prompt_id: str):
        """Return ``(mean, components)`` numpy arrays, or ``None``."""
        if prompt_id not in self._active:
            return None
        table = self._active[prompt_id]
        return (table.pca_params.mean.copy(), table.pca_params.components.copy())

    def get_prompt_keys(self, prompt_id: str):
        """Return keys ndarray ``(n_entries, K)`` for prompt, or ``None``."""
        if prompt_id not in self._active:
            return None
        return self._active[prompt_id].get_keys_numpy()

    def get_prompt_table_data(self, prompt_id: str):
        """Return serialisable dict with full table data for proposer cache."""
        if prompt_id not in self._active:
            return None
        table = self._active[prompt_id]
        return {
            "mean": table.pca_params.mean,
            "components": table.pca_params.components,
            "keys": table.get_keys_numpy(),
            "rollout_seqs": [s.tolist() for s in table.rollout_seqs],
            "entry_rollout_idx": np.ascontiguousarray(table.entry_rollout_idx[:table.n_entries]),
            "entry_offset": np.ascontiguousarray(table.entry_offset[:table.n_entries]),
            "n_entries": table.n_entries,
            "wnd_size": table.wnd_size,
        }

    # Management

    def delete(self, prompt_id: str):
        """Delete a prompt's table from building side."""
        self._building.pop(prompt_id, None)

    def clear(self):
        """Clear all tables (both active and building) and reset metrics."""
        self._active.clear()
        self._building.clear()
        self._reset_metrics()
        self._write_active_version_manifest()

    def exist(self, prompt_id: str) -> bool:
        return prompt_id in self._active

    def get_prompt_ids(self) -> List[str]:
        return list(self._active.keys())

    def num_prompts(self) -> int:
        return len(self._active)

    def total_entries(self) -> int:
        return sum(t.n_entries for t in self._active.values())

    # Double-buffer version management

    def swap(self):
        """Swap building → active.  Resets query metrics for new epoch."""
        self._active = self._building
        self._building = {}
        self._active_version += 1
        self._reset_metrics()
        logger.info(
            "HSpec swap: active_version=%d, prompts=%d, entries=%d",
            self._active_version,
            len(self._active),
            self.total_entries(),
        )

    def get_active_version(self) -> int:
        """Return current active table version (epoch counter)."""
        return self._active_version

    def get_active_table_data_batch(
        self,
        prompt_ids: List[str],
    ) -> Tuple[int, Dict[str, Optional[Dict]]]:
        """Batch fetch table data from *active* tables for proposer prefetch.

        Returns ``(active_version, {prompt_id: table_data | None})``.
        The version is piggy-backed so the proposer can detect epoch
        swaps without a separate RPC – one fewer round trip.
        """
        result: Dict[str, Optional[Dict]] = {}
        for pid in prompt_ids:
            if pid not in self._active:
                result[pid] = None
                continue
            table = self._active[pid]
            result[pid] = {
                "mean": table.pca_params.mean.copy(),
                "components": table.pca_params.components.copy(),
                "keys": table.get_keys_numpy(),
                "rollout_seqs": [np.ascontiguousarray(s) for s in table.rollout_seqs],
                "entry_rollout_idx": np.ascontiguousarray(table.entry_rollout_idx[:table.n_entries]),
                "entry_offset": np.ascontiguousarray(table.entry_offset[:table.n_entries]),
                "n_entries": table.n_entries,
                "wnd_size": table.wnd_size,
                "max_wnd": table.max_wnd,
                "min_wnd": table.min_wnd,
            }
        return (self._active_version, result)

    # Metrics

    def compute_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "query_times": self._query_count,
            "match_times": self._match_count,
            "total_draft_length": self._total_draft_len,
            "verify_times": self._verify_count,
            "accept_times": self._accept_count,
            "accept_length_sum": self._accept_len_sum,
            "accept_times_advan": self._accept_advan_count,
            "reject_times_advan": self._reject_advan_count,
            "build_count": self._build_count,
            "discard_count": self._discard_count,
            "num_prompts": len(self._active),
            "total_entries": self.total_entries(),
            "entry_match_count": self._entry_match_count,
            "entry_delta_sum": self._entry_delta_sum,
            "entry_abs_delta_sum": self._entry_abs_delta_sum,
            "entry_verify_count": self._entry_verify_count,
            "entry_accept_count": self._entry_accept_count,
            "entry_accept_len_sum": self._entry_accept_len_sum,
        }
        for abs_delta, count in self._entry_abs_delta_verify.items():
            metrics[f"entry_abs_delta_verify_{abs_delta}"] = float(count)
        for abs_delta, count in self._entry_abs_delta_accept.items():
            metrics[f"entry_abs_delta_accept_{abs_delta}"] = float(count)
        for abs_delta, total_len in self._entry_abs_delta_accept_len_sum.items():
            metrics[f"entry_abs_delta_accept_len_sum_{abs_delta}"] = float(total_len)
        return metrics

    def _reset_metrics(self):
        self._query_count = 0
        self._match_count = 0
        self._total_draft_len = 0
        self._build_count = 0
        self._discard_count = 0
        self._verify_count = 0
        self._accept_count = 0
        self._accept_len_sum = 0
        self._accept_advan_count = 0
        self._reject_advan_count = 0
        self._entry_match_count = 0
        self._entry_delta_sum = 0
        self._entry_abs_delta_sum = 0
        self._entry_verify_count = 0
        self._entry_accept_count = 0
        self._entry_accept_len_sum = 0
        self._entry_abs_delta_verify = {}
        self._entry_abs_delta_accept = {}
        self._entry_abs_delta_accept_len_sum = {}

    # Online metrics reporting (from worker-local proposer)

    def report_online_metrics(
        self,
        query_times: int = 0,
        match_times: int = 0,
        total_draft_length: int = 0,
    ) -> None:
        """Accumulate online query stats reported by worker-local proposers.

        In HSPEC's fast path, similarity matching is executed entirely on the
        vLLM worker (device + local cache) and does not call the table actor's
        query methods. Therefore, actor-side query counters would remain zero
        unless we explicitly report these stats.

        Callers should invoke this at low frequency to avoid overhead.
        """
        try:
            self._query_count += int(query_times)
            self._match_count += int(match_times)
            self._total_draft_len += int(total_draft_length)
        except Exception:
            pass

    def report_verification_metrics(
        self,
        verify_times: int = 0,
        accept_times: int = 0,
        accept_length_sum: int = 0,
        accept_times_advan: int = 0,
        reject_times_advan: int = 0,
    ) -> None:
        """Accumulate post-verification stats (after rejection sampling).

        These are reported from the vLLM worker hot loop. Callers should never
        block on ray.get.
        """
        try:
            self._verify_count += int(verify_times)
            self._accept_count += int(accept_times)
            self._accept_len_sum += int(accept_length_sum)
            self._accept_advan_count += int(accept_times_advan)
            self._reject_advan_count += int(reject_times_advan)
        except Exception:
            pass

    def report_entry_metrics(
        self,
        match_count: int = 0,
        delta_sum: int = 0,
        abs_delta_sum: int = 0,
        verify_count: int = 0,
        accept_count: int = 0,
        accept_len_sum: int = 0,
        abs_delta_verify: Optional[Dict[int, int]] = None,
        abs_delta_accept: Optional[Dict[int, int]] = None,
        abs_delta_accept_len_sum: Optional[Dict[int, int]] = None,
    ) -> None:
        """Aggregate entry-position study metrics from worker-local proposers."""
        try:
            self._entry_match_count += int(match_count)
            self._entry_delta_sum += int(delta_sum)
            self._entry_abs_delta_sum += int(abs_delta_sum)
            self._entry_verify_count += int(verify_count)
            self._entry_accept_count += int(accept_count)
            # Global entry_avg_accept_length uses verify_count as denominator,
            # so this sum is over *all* verified drafts' accepted_prefix_len.
            self._entry_accept_len_sum += int(accept_len_sum)

            for abs_delta, count in (abs_delta_verify or {}).items():
                key = int(abs_delta)
                self._entry_abs_delta_verify[key] = self._entry_abs_delta_verify.get(key, 0) + int(count)
            for abs_delta, count in (abs_delta_accept or {}).items():
                key = int(abs_delta)
                self._entry_abs_delta_accept[key] = self._entry_abs_delta_accept.get(key, 0) + int(count)
            for abs_delta, total_len in (abs_delta_accept_len_sum or {}).items():
                key = int(abs_delta)
                self._entry_abs_delta_accept_len_sum[key] = (
                    self._entry_abs_delta_accept_len_sum.get(key, 0) + int(total_len))
        except Exception:
            pass

    # Debug

    def debug_table_info(self, prompt_id: str) -> Dict:
        """Return detailed debug info for a prompt's tables (building + active)."""
        info: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "active_version": self._active_version,
            "building_prompt_count": len(self._building),
            "active_prompt_count": len(self._active),
        }
        for label, store in [("building", self._building),
                             ("active", self._active)]:
            if prompt_id not in store:
                info[label] = None
                continue
            t = store[prompt_id]
            keys_np = t.get_keys_numpy()
            # Sample first key for sanity check
            key_sample = None
            key_norm_sample = None
            if t.n_entries > 0 and keys_np.shape[0] > 0:
                key_sample = keys_np[0][:8].tolist()  # first 8 dims
                key_norm_sample = float(np.linalg.norm(keys_np[0]))
            # Sample value (first entry's draft tokens)
            draft_sample = None
            if t.n_entries > 0:
                ridx = int(t.entry_rollout_idx[0])
                off = int(t.entry_offset[0])
                seq = t.rollout_seqs[ridx] if ridx < len(t.rollout_seqs) else None
                if seq is not None:
                    draft_sample = seq[off: off + min(5, t.wnd_size)].tolist()
            info[label] = {
                "n_entries": t.n_entries,
                "pca_mean_shape": list(t.pca_params.mean.shape),
                "pca_components_shape": list(t.pca_params.components.shape),
                "pca_mean_norm": float(np.linalg.norm(t.pca_params.mean)),
                "keys_shape": list(keys_np.shape),
                "key_sample_first8": key_sample,
                "key_norm_sample": key_norm_sample,
                "rollout_seqs_count": len(t.rollout_seqs),
                "rollout_seq_lens": [len(s) for s in t.rollout_seqs],
                "wnd_size": t.wnd_size,
                "max_wnd": t.max_wnd,
                "min_wnd": t.min_wnd,
                "draft_sample_entry0": draft_sample,
            }
        return info

    # ZMQ server  (for decode hot-loop queries)

    def run(self):
        """Run blocking ZMQ REP server (call via ``actor.run.remote()``)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{self.port}")
        self.running = True
        logger.info("HSpecTableGroup ZMQ server started on port %d", self.port)

        while self.running:
            try:
                msg = sock.recv()
                req = msgpack.unpackb(msg, raw=False)
                resp = self._handle_zmq(req)
                sock.send(msgpack.packb(resp, use_bin_type=True))
            except Exception as exc:
                try:
                    sock.send(msgpack.packb({"status": "error", "message": str(exc)},
                                            use_bin_type=True))
                except Exception:
                    pass

    def _handle_zmq(self, request: Dict) -> Any:
        method = request.get("method")
        params = request.get("params", {})

        if method == "query":
            hs = np.array(params["hidden_state"], dtype=np.float32)
            return self.query(params["prompt_id"], hs, params.get("accept_length", 1))
        if method == "query_batch":
            hs_list = [np.array(h, dtype=np.float32) for h in params["hidden_state_list"]]
            return self.query_batch(
                params["prompt_id_list"],
                hs_list,
                params["accept_length_list"],
            )
        if method == "get_prompt_pca":
            result = self.get_prompt_pca(params["prompt_id"])
            if result is None:
                return None
            mean, comp = result
            return {"mean": mean.tolist(), "components": comp.tolist()}
        if method == "stop":
            self.running = False
            return True
        return {"error": f"Unknown method: {method}"}


# Global client  (partition routing + async build interface)


class GlobalHSpecTableGroup:
    """Client-side interface for distributed HSpec tables."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_entries_per_prompt: int = 10_000,
        n_components: int = 64,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries_per_prompt
        self.n_components = n_components
        self.num_groups = _resolve_num_groups()
        self.logical_node_id = get_hspec_node_id()
        self.single_node_only = hspec_single_node_only_enabled()
        self.topology_strict = hspec_topology_strict_enabled()

        # Discover existing Ray actors
        self.groups: List[ray.actor.ActorHandle] = []
        for i in range(self.num_groups):
            try:
                self.groups.append(ray.get_actor(_build_actor_name(i)))
            except ValueError:
                logger.warning("HSpec build shard actor %s not found", _build_actor_name(i))

        if len(self.groups) != self.num_groups:
            raise RuntimeError(
                "HSpec build actors are incomplete: "
                f"found={len(self.groups)} expected={self.num_groups}. "
                "Call init_hspec_tables() before constructing GlobalHSpecTableGroup, "
                "or check HSPEC_BUILD_ACTOR_NAME_PREFIX/HSPEC_NUM_SHARDS."
            )
        self.actor_topologies = _get_actor_topologies(self.groups)
        _validate_actor_topologies(
            self.actor_topologies,
            expected_num_groups=self.num_groups,
            expected_node_id=self.logical_node_id,
            expected_similarity_threshold=self.similarity_threshold,
            expected_max_entries_per_prompt=self.max_entries,
            expected_n_components=self.n_components,
        )

        # ZMQ connections (lazy-initialised on first query)
        self._zmq_ctx: Optional[zmq.Context] = None
        self._zmq_sockets: Dict[int, zmq.Socket] = {}

        logger.info(
            "HSpec: GlobalHSpecTableGroup connected to %d actors, node_id=%s, topology_strict=%s",
            len(self.groups),
            self.logical_node_id,
            self.topology_strict,
        )

    def __len__(self):
        return len(self.groups)

    def _get_partition_id(self, prompt_id: str) -> int:
        return stable_partition_id(prompt_id, self.num_groups)

    def _get_partition(self, prompt_id: str) -> ray.actor.ActorHandle:
        return self.groups[self._get_partition_id(prompt_id)]

    def _topology_violation(self, metric_name: str, message: str) -> None:
        hspec_record_store_metric(metric_name, 1)
        hspec_record_store_metric("descriptor_topology_violation", 1)
        raise RuntimeError(message)

    def _validate_and_normalize_descriptor_for_routing(
        self,
        prompt_id: str,
        desc: HSpecTrajectoryDesc,
        expected_pid: int,
    ) -> HSpecTrajectoryDesc:
        recomputed_pid = stable_partition_id(prompt_id, self.num_groups)
        if int(expected_pid) != int(recomputed_pid):
            raise RuntimeError(
                "HSpec routing partition mismatch inside GlobalHSpecTableGroup: "
                f"prompt_id={prompt_id!r}, expected_pid={expected_pid}, recomputed={recomputed_pid}, "
                f"num_groups={self.num_groups}"
            )
        if expected_pid >= len(self.actor_topologies):
            raise RuntimeError(
                "HSpec routing points to missing build actor: "
                f"prompt_id={prompt_id!r}, expected_pid={expected_pid}, "
                f"actor_count={len(self.actor_topologies)}, num_groups={self.num_groups}"
            )
        actor_topology = self.actor_topologies[expected_pid]
        if int(actor_topology.get("shard_id", -1)) != int(expected_pid):
            raise RuntimeError(
                "HSpec cached actor topology is inconsistent: "
                f"prompt_id={prompt_id!r}, expected_pid={expected_pid}, topology={actor_topology}"
            )
        if desc.prompt_id and str(desc.prompt_id) != str(prompt_id):
            self._topology_violation(
                "descriptor_prompt_mismatch",
                "HSpec descriptor prompt mismatch before build routing: "
                f"prompt_id={prompt_id!r}, desc.prompt_id={desc.prompt_id!r}, "
                f"request_id={desc.request_id!r}, desc.node_id={desc.node_id!r}, "
                f"desc.shard_id={desc.shard_id}, expected_pid={expected_pid}, "
                f"num_groups={self.num_groups}, logical_node_id={self.logical_node_id!r}, "
                f"topology_strict={self.topology_strict}",
            )
        if self.single_node_only and str(desc.node_id) != str(self.logical_node_id):
            self._topology_violation(
                "descriptor_node_mismatch",
                "HSpec descriptor node mismatch in single-node mode before build routing: "
                f"prompt_id={prompt_id!r}, request_id={desc.request_id!r}, "
                f"desc.node_id={desc.node_id!r}, driver.node_id={self.logical_node_id!r}, "
                f"desc.shard_id={desc.shard_id}, expected_pid={expected_pid}, "
                f"num_groups={self.num_groups}, topology_strict={self.topology_strict}. "
                "Descriptor raw-store files are local; Phase 1 does not support cross-node build.",
            )
        if int(desc.shard_id) != int(expected_pid):
            hspec_record_store_metric("descriptor_shard_mismatch", 1)
            hspec_record_store_metric("descriptor_topology_violation", 1)
            message = (
                "HSpec descriptor shard mismatch before build routing: "
                f"prompt_id={prompt_id!r}, request_id={desc.request_id!r}, "
                f"desc.node_id={desc.node_id!r}, desc.shard_id={desc.shard_id}, "
                f"expected_pid={expected_pid}, num_groups={self.num_groups}, "
                f"logical_node_id={self.logical_node_id!r}, topology_strict={self.topology_strict}. "
                "This indicates HSPEC_NUM_SHARDS drift or stale descriptor."
            )
            if self.topology_strict:
                raise RuntimeError(message)
            logger.warning("%s Normalizing descriptor shard because HSPEC_TOPOLOGY_STRICT=0.", message)
            hspec_record_store_metric("descriptor_shard_normalized", 1)
            desc = desc.with_updates(shard_id=int(expected_pid))
        return desc

    # Online metrics reporting (worker-local fast path)

    def report_online_metrics_async(
        self,
        query_times: int,
        match_times: int,
        total_draft_length: int,
    ) -> Optional[ray.ObjectRef]:
        """Fire-and-forget reporting of online query stats.

        We intentionally do NOT block (no ray.get) to keep this off the hot
        decode loop. The stats are aggregated into existing actor-side counters
        so trainer-side `compute_metrics()` can reflect the true online match
        rate even when queries are executed locally on the worker.
        """
        if not self.groups:
            return None
        # Aggregate into a single actor to minimize overhead.
        try:
            return self.groups[0].report_online_metrics.remote(
                query_times=query_times,
                match_times=match_times,
                total_draft_length=total_draft_length,
            )
        except Exception:
            return None

    def report_verification_metrics_async(
        self,
        verify_times: int,
        accept_times: int,
        accept_length_sum: int,
        accept_times_advan: int = 0,
        reject_times_advan: int = 0,
    ) -> Optional[ray.ObjectRef]:
        """Fire-and-forget reporting of verification stats from vLLM workers."""
        if not self.groups:
            return None
        # Aggregate into a single actor to minimize overhead.
        try:
            return self.groups[0].report_verification_metrics.remote(
                verify_times=verify_times,
                accept_times=accept_times,
                accept_length_sum=accept_length_sum,
                accept_times_advan=accept_times_advan,
                reject_times_advan=reject_times_advan,
            )
        except Exception:
            return None

    def report_entry_metrics_async(
        self,
        match_count: int,
        delta_sum: int,
        abs_delta_sum: int,
        verify_count: int,
        accept_count: int,
        accept_len_sum: int,
        abs_delta_verify: Dict[int, int],
        abs_delta_accept: Dict[int, int],
        abs_delta_accept_len_sum: Dict[int, int],
    ) -> Optional[ray.ObjectRef]:
        """Fire-and-forget reporting for entry-position study metrics."""
        if not self.groups:
            return None
        try:
            return self.groups[0].report_entry_metrics.remote(
                match_count=match_count,
                delta_sum=delta_sum,
                abs_delta_sum=abs_delta_sum,
                verify_count=verify_count,
                accept_count=accept_count,
                accept_len_sum=accept_len_sum,
                abs_delta_verify=abs_delta_verify,
                abs_delta_accept=abs_delta_accept,
                abs_delta_accept_len_sum=abs_delta_accept_len_sum,
            )
        except Exception:
            return None

    # Build  (async, non-blocking)
    def build_tables_async(
        self,
        prompt_data: Dict[str, List[HSpecTrajectoryDesc | Dict[str, Any]]],
    ) -> List[HSpecBuildSubmission]:
        """Submit descriptor-only rollout data to partition actors.

        Phase 1 default payload is ``{prompt_id: List[HSpecTrajectoryDesc]}``.
        Legacy ndarray dict payloads must use ``build_tables_async_legacy``.
        PCA fitting + table construction runs inside Ray actors and does not
        block the caller. The returned submission records retain the raw
        segment keys required for epoch-level GC.
        """
        partition_payloads: Dict[int, Dict[str, Any]] = {i: {} for i in range(self.num_groups)}
        partition_segments: Dict[int, set[HSpecSegmentKey]] = {i: set() for i in range(self.num_groups)}
        partition_prompt_ids: Dict[int, List[str]] = {i: [] for i in range(self.num_groups)}
        for prompt_id, data in prompt_data.items():
            if not isinstance(data, list):
                if hspec_strict_descriptor_mode_enabled() or _looks_like_legacy_hspec_payload(data):
                    _raise_legacy_payload_forbidden(prompt_id, data)
                _raise_legacy_payload_forbidden(prompt_id, data)

            pid = self._get_partition_id(prompt_id)
            descs: List[HSpecTrajectoryDesc] = []
            for desc in data:
                if desc is None:
                    continue
                desc_obj = coerce_hspec_desc(desc)
                desc_obj = self._validate_and_normalize_descriptor_for_routing(
                    prompt_id,
                    desc_obj,
                    pid,
                )
                descs.append(desc_obj)
                try:
                    partition_segments[pid].add(hspec_segment_key_from_desc(desc_obj))
                except Exception:
                    logger.debug(
                        "Failed to inspect HSpec descriptor segment/shard hint for prompt_id=%s",
                        prompt_id,
                        exc_info=True,
                    )
                    raise
            if not descs:
                continue
            partition_payloads[pid][prompt_id] = descs
            partition_prompt_ids[pid].append(prompt_id)

        submissions: List[HSpecBuildSubmission] = []
        for pid, payload in partition_payloads.items():
            if not payload:
                continue
            if pid >= len(self.groups):
                raise RuntimeError(
                    "HSpec build routing points to missing actor: "
                    f"pid={pid}, actor_count={len(self.groups)}, num_groups={self.num_groups}"
                )
            ref = self.groups[pid].build_tables_batch.remote(payload)
            submissions.append(
                HSpecBuildSubmission(
                    ref=ref,
                    shard_id=pid,
                    segments=frozenset(partition_segments[pid]),
                    prompt_ids=tuple(partition_prompt_ids[pid]),
                    legacy=False,
                )
            )
        if submissions:
            hspec_record_store_metric("build_submission_count", len(submissions))
            hspec_record_store_metric(
                "build_submission_segments",
                len({segment for item in submissions for segment in item.segments}),
            )
        return submissions

    def build_tables_async_legacy(
        self,
        prompt_data: Dict[str, Dict[str, Any]],
    ) -> List[HSpecBuildSubmission]:
        """Submit old ndarray payloads for explicit HSPEC_LEGACY_DATAPROTO_HS=1 A/B only."""
        if not hspec_legacy_dataproto_hs_enabled():
            hspec_record_store_metric("strict_descriptor_violation", 1)
            raise RuntimeError("build_tables_async_legacy() requires HSPEC_LEGACY_DATAPROTO_HS=1")

        partition_payloads: Dict[int, Dict[str, Any]] = {i: {} for i in range(self.num_groups)}
        partition_prompt_ids: Dict[int, List[str]] = {i: [] for i in range(self.num_groups)}
        for prompt_id, data in prompt_data.items():
            if not _looks_like_legacy_hspec_payload(data):
                raise TypeError(
                    "Legacy HSpec build API expects dict payload with "
                    f"hidden_states/tokens/rewards for prompt_id={prompt_id!r}"
                )
            pid = self._get_partition_id(prompt_id)
            partition_payloads[pid][prompt_id] = data
            partition_prompt_ids[pid].append(prompt_id)

        submissions: List[HSpecBuildSubmission] = []
        for pid, payload in partition_payloads.items():
            if not payload:
                continue
            if pid >= len(self.groups):
                raise RuntimeError(
                    "HSpec legacy build routing points to missing actor: "
                    f"pid={pid}, actor_count={len(self.groups)}, num_groups={self.num_groups}"
                )
            ref = self.groups[pid].build_tables_batch_legacy.remote(payload)
            submissions.append(
                HSpecBuildSubmission(
                    ref=ref,
                    shard_id=pid,
                    segments=frozenset(),
                    prompt_ids=tuple(partition_prompt_ids[pid]),
                    legacy=True,
                )
            )
        if submissions:
            hspec_record_store_metric("build_submission_count", len(submissions))
        return submissions

    def query(self, prompt_id: str, hidden_state: np.ndarray, accept_length: int = 1):
        actor = self._get_partition(prompt_id)
        return actor.query.remote(prompt_id, hidden_state, accept_length)

    def query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query via Ray actors (blocking – collects results)."""
        parts: Dict[int, Dict[str, list]] = {
            i: {"pids": [], "hss": [], "als": [], "pos": []}
            for i in range(self.num_groups)
        }
        for idx, (pid, hs, al) in enumerate(zip(prompt_id_list, hidden_state_list, accept_length_list)):
            part = self._get_partition_id(pid)
            parts[part]["pids"].append(pid)
            parts[part]["hss"].append(hs)
            parts[part]["als"].append(al)
            parts[part]["pos"].append(idx)

        futures = {}
        for part, data in parts.items():
            if data["pids"] and part < len(self.groups):
                futures[part] = self.groups[part].query_batch.remote(
                    data["pids"], data["hss"], data["als"])

        results: List[List[int]] = [[] for _ in range(len(prompt_id_list))]
        for part, future in futures.items():
            batch_res = ray.get(future)
            for pos, draft in zip(parts[part]["pos"], batch_res):
                results[pos] = draft
        return results

    def post_query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query via ZMQ  (lower latency for decode hot path).

        Hidden states are sent as raw D-dim vectors; the table actor
        projects them internally via PCA.  Step 2 will optimise this by
        sending pre-projected K-dim vectors from on-device cache.
        """
        self._ensure_zmq()

        parts: Dict[int, Dict[str, list]] = {
            i: {"pids": [], "hss": [], "als": [], "pos": []}
            for i in range(self.num_groups)
        }
        for idx, (pid, hs, al) in enumerate(zip(prompt_id_list, hidden_state_list, accept_length_list)):
            part = self._get_partition_id(pid)
            parts[part]["pids"].append(pid)
            parts[part]["hss"].append(hs.tolist() if isinstance(hs, np.ndarray) else hs)
            parts[part]["als"].append(al)
            parts[part]["pos"].append(idx)

        responses: Dict[int, Any] = {}
        for server_id, data in parts.items():
            if data["pids"] and server_id in self._zmq_sockets:
                req = {
                    "method": "query_batch",
                    "params": {
                        "prompt_id_list": data["pids"],
                        "hidden_state_list": data["hss"],
                        "accept_length_list": data["als"],
                    },
                }
                responses[server_id] = self._zmq_send(server_id, req)

        results: List[List[int]] = [[] for _ in range(len(prompt_id_list))]
        for server_id, resp in responses.items():
            if isinstance(resp, list):
                for pos, draft in zip(parts[server_id]["pos"], resp):
                    results[pos] = draft
        return results

    # Table data access  (for proposer prefetch / cache)

    def get_prompt_pca(self, prompt_id: str):
        """Return future resolving to ``(mean, components)`` or ``None``."""
        return self._get_partition(prompt_id).get_prompt_pca.remote(prompt_id)

    def get_prompt_table_data(self, prompt_id: str):
        """Return future resolving to serialisable table data dict."""
        return self._get_partition(prompt_id).get_prompt_table_data.remote(prompt_id)

    def get_prompt_table_data_batch(self, prompt_id_list: List[str]) -> Dict[str, ray.ObjectRef]:
        """Batch fetch table data  (returns ``{prompt_id: future}``)."""
        futures: Dict[str, ray.ObjectRef] = {}
        for pid in prompt_id_list:
            part = self._get_partition_id(pid)
            if part < len(self.groups):
                futures[pid] = self.groups[part].get_prompt_table_data.remote(pid)
        return futures

    # Double-buffer version management

    def swap(self):
        """Swap building → active on all actors.  **Blocking.**"""
        if self.groups:
            ray.get([g.swap.remote() for g in self.groups])

    def swap_async(self) -> List[ray.ObjectRef]:
        """Queue swap on all actors.  Non-blocking; returns futures."""
        return [g.swap.remote() for g in self.groups]

    def get_active_version(self) -> int:
        """Return active table version (queries one actor)."""
        if not self.groups:
            return 0
        return ray.get(self.groups[0].get_active_version.remote())

    def prefetch_batch(self, prompt_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch-fetch table data from active tables (for proposer cache).

        One Ray call per partition → minimal overhead.  Returns
        ``{prompt_id: serialised_table_data | None}``.
        """
        from collections import defaultdict

        partition_prompts: Dict[int, List[str]] = defaultdict(list)
        for pid in prompt_ids:
            partition_prompts[self._get_partition_id(pid)].append(pid)

        futures: Dict[int, ray.ObjectRef] = {}
        for part, pids in partition_prompts.items():
            if part < len(self.groups):
                futures[part] = self.groups[part].get_active_table_data_batch.remote(pids)

        result: Dict[str, Optional[Dict]] = {}
        latest_version = -1
        for part, future in futures.items():
            version, batch_data = ray.get(future)
            latest_version = max(latest_version, version)
            result.update(batch_data)
        return latest_version, result

    def prefetch_batch_async(self, prompt_ids: List[str]) -> List[Tuple[ray.ObjectRef, List[str]]]:
        """Fire async prefetch – **non-blocking**, returns immediately.

        Returns ``[(ObjectRef, [prompt_ids]), ...]`` where each
        ObjectRef resolves to ``(active_version, {pid: data | None})``.
        Callers should poll with ``ray.wait(timeout=0)`` instead of
        blocking on ``ray.get()``.
        """
        from collections import defaultdict

        partition_prompts: Dict[int, List[str]] = defaultdict(list)
        for pid in prompt_ids:
            partition_prompts[self._get_partition_id(pid)].append(pid)

        result: List[Tuple[ray.ObjectRef, List[str]]] = []
        for part, pids in partition_prompts.items():
            if part < len(self.groups):
                future = self.groups[part].get_active_table_data_batch.remote(pids)
                result.append((future, pids))
        return result

    # Management

    def clear(self):
        """Clear all tables (returns list of futures)."""
        return [g.clear.remote() for g in self.groups]

    def delete(self, prompt_id: str):
        return self._get_partition(prompt_id).delete.remote(prompt_id)

    # Debug

    def debug_table_info(self, prompt_id: str) -> Dict:
        """Query a partition actor for debug info about a prompt's tables."""
        actor = self._get_partition(prompt_id)
        return ray.get(actor.debug_table_info.remote(prompt_id))

    # Metrics

    def compute_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from all partition actors."""
        if not self.groups:
            return {
                "hspec/match_rate": 0.0,
                "hspec/avg_draft_length": 0.0,
                "hspec/avg_accept_length": 0.0,
                "hspec/query_times": 0,
                "hspec/match_times": 0,
                "hspec/verify_times": 0,
                "hspec/accept_times": 0,
                "hspec/cache_match_rate": 0.0,
                "hspec/cache_query_times": 0,
                "hspec/cache_match_times": 0,
                "hspec/build_count": 0,
                "hspec/discard_count": 0,
                "hspec/num_prompts": 0,
                "hspec/total_entries": 0,
                "hspec/accept_times_advan": 0,
                "hspec/accept_times_advan_ratio": 0.0,
                "hspec/reject_times_advan": 0,
                "hspec/reject_times_advan_ratio": 0.0,
                "hspec/entry_match_avg_signed_delta": 0.0,
                "hspec/entry_match_avg_abs_delta": 0.0,
                "hspec/entry_verify_times": 0,
                "hspec/entry_accept_times": 0,
                "hspec/entry_avg_accept_length": 0.0,
            }
        tasks = [g.compute_metrics.remote() for g in self.groups]
        metrics_list = ray.get(tasks)

        agg: Dict[str, float] = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                agg[key] = agg.get(key, 0.0) + float(value)

        # Cache-match metrics (reported from worker-local proposer).
        cache_qt = agg.get("query_times", 0)
        cache_mt = agg.get("match_times", 0)
        total_draft_len = agg.get("total_draft_length", 0)
        verify_times = agg.get("verify_times", 0)
        accept_times = agg.get("accept_times", 0)
        accept_len_sum = agg.get("accept_length_sum", 0)
        accept_times_advan = agg.get("accept_times_advan", 0)
        reject_times_advan = agg.get("reject_times_advan", 0)
        reject_times = max(verify_times - accept_times, 0)
        entry_match_count = agg.get("entry_match_count", 0)
        entry_delta_sum = agg.get("entry_delta_sum", 0)
        entry_abs_delta_sum = agg.get("entry_abs_delta_sum", 0)
        entry_verify_count = agg.get("entry_verify_count", 0)
        entry_accept_count = agg.get("entry_accept_count", 0)
        entry_accept_len_sum = agg.get("entry_accept_len_sum", 0)

        result = {
            "hspec/match_rate": accept_times / verify_times if verify_times > 0 else 0.0,
            "hspec/avg_accept_length": accept_len_sum / accept_times if accept_times > 0 else 0.0,
            "hspec/query_times": verify_times,
            "hspec/match_times": accept_times,
            "hspec/verify_times": verify_times,
            "hspec/accept_times": accept_times,
            "hspec/cache_match_rate": cache_mt / cache_qt if cache_qt > 0 else 0.0,
            "hspec/cache_query_times": cache_qt,
            "hspec/cache_match_times": cache_mt,
            "hspec/avg_draft_length": total_draft_len / cache_mt if cache_mt > 0 else 0.0,
            "hspec/build_count": agg.get("build_count", 0),
            "hspec/discard_count": agg.get("discard_count", 0),
            "hspec/num_prompts": agg.get("num_prompts", 0),
            "hspec/total_entries": agg.get("total_entries", 0),
            "hspec/accept_times_advan": accept_times_advan,
            "hspec/accept_times_advan_ratio": accept_times_advan / accept_times if accept_times > 0 else 0.0,
            "hspec/reject_times_advan": reject_times_advan,
            "hspec/reject_times_advan_ratio": reject_times_advan / reject_times if reject_times > 0 else 0.0,
            "hspec/entry_match_avg_signed_delta": (
                entry_delta_sum / entry_match_count if entry_match_count > 0 else 0.0),
            "hspec/entry_match_avg_abs_delta": (
                entry_abs_delta_sum / entry_match_count if entry_match_count > 0 else 0.0),
            "hspec/entry_verify_times": entry_verify_count,
            "hspec/entry_accept_times": entry_accept_count,
            "hspec/entry_avg_accept_length": (
                entry_accept_len_sum / entry_verify_count if entry_verify_count > 0 else 0.0),
        }

        store_metrics = collect_hspec_store_metrics(reset=True)
        result["hspec/raw_store_bytes"] = float(store_metrics.get("raw_store_bytes", 0))
        result["hspec/desc_count"] = float(store_metrics.get("desc_count", 0))
        result["hspec/collect_dropped"] = float(store_metrics.get("collect_dropped", 0))
        result["hspec/segment_sealed"] = float(store_metrics.get("segment_sealed", 0))
        result["hspec/segment_rotated"] = float(store_metrics.get("segment_rotated", 0))
        result["hspec/segment_manifest_write_error"] = float(
            store_metrics.get("segment_manifest_write_error", 0))
        result["hspec/raw_store_budget_gc_skipped"] = float(
            store_metrics.get("raw_store_budget_gc_skipped", 0))
        result["hspec/unsafe_descriptor_cleanup_suppressed"] = float(
            store_metrics.get("unsafe_descriptor_cleanup_suppressed", 0))
        result["hspec/segment_delete_count"] = float(store_metrics.get("segment_delete_count", 0))
        result["hspec/segment_delete_bytes"] = float(store_metrics.get("segment_delete_bytes", 0))
        result["hspec/raw_store_epoch_gc_segments"] = float(
            store_metrics.get("raw_store_epoch_gc_segments", 0))
        result["hspec/raw_store_epoch_gc_deleted"] = float(
            store_metrics.get("raw_store_epoch_gc_deleted", 0))
        result["hspec/raw_store_epoch_gc_skipped"] = float(
            store_metrics.get("raw_store_epoch_gc_skipped", 0))
        result["hspec/raw_store_epoch_gc_error"] = float(
            store_metrics.get("raw_store_epoch_gc_error", 0))
        result["hspec/build_submission_count"] = float(store_metrics.get("build_submission_count", 0))
        result["hspec/build_submission_segments"] = float(store_metrics.get("build_submission_segments", 0))
        result["hspec/topology_actor_count"] = float(len(self.actor_topologies))
        result["hspec/topology_actor_init_error"] = float(
            store_metrics.get("topology_actor_init_error", 0))
        result["hspec/topology_actor_reuse_mismatch"] = float(
            store_metrics.get("topology_actor_reuse_mismatch", 0))
        result["hspec/topology_actor_node_mismatch"] = float(
            store_metrics.get("topology_actor_node_mismatch", 0))
        result["hspec/topology_actor_shard_mismatch"] = float(
            store_metrics.get("topology_actor_shard_mismatch", 0))
        result["hspec/topology_actor_num_groups_mismatch"] = float(
            store_metrics.get("topology_actor_num_groups_mismatch", 0))
        result["hspec/descriptor_topology_violation"] = float(
            store_metrics.get("descriptor_topology_violation", 0))
        result["hspec/descriptor_node_mismatch"] = float(
            store_metrics.get("descriptor_node_mismatch", 0))
        result["hspec/descriptor_shard_mismatch"] = float(
            store_metrics.get("descriptor_shard_mismatch", 0))
        result["hspec/descriptor_prompt_mismatch"] = float(
            store_metrics.get("descriptor_prompt_mismatch", 0))
        result["hspec/descriptor_shard_normalized"] = float(
            store_metrics.get("descriptor_shard_normalized", 0))

        abs_deltas = set()
        for key in agg:
            if key.startswith("entry_abs_delta_verify_"):
                abs_deltas.add(int(key.rsplit("_", 1)[-1]))
            elif key.startswith("entry_abs_delta_accept_"):
                abs_deltas.add(int(key.rsplit("_", 1)[-1]))
            elif key.startswith("entry_abs_delta_accept_len_sum_"):
                abs_deltas.add(int(key.rsplit("_", 1)[-1]))

        for abs_delta in sorted(abs_deltas):
            verify_count = agg.get(f"entry_abs_delta_verify_{abs_delta}", 0.0)
            accept_count = agg.get(f"entry_abs_delta_accept_{abs_delta}", 0.0)
            accept_len_total = agg.get(f"entry_abs_delta_accept_len_sum_{abs_delta}", 0.0)
            prefix = f"hspec/entry_abs_delta_{abs_delta}"
            result[f"{prefix}_verify_times"] = verify_count
            result[f"{prefix}_accept_times"] = accept_count
            result[f"{prefix}_match_rate"] = accept_count / verify_count if verify_count > 0 else 0.0
            result[f"{prefix}_avg_accept_length"] = (
                accept_len_total / accept_count if accept_count > 0 else 0.0)

        return result

    # ZMQ helpers

    def _ensure_zmq(self):
        if self._zmq_sockets:
            return
        self._zmq_ctx = zmq.Context()
        for i in range(_num_groups):
            sock = self._zmq_ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.connect(f"tcp://localhost:{6555 + i}")
            self._zmq_sockets[i] = sock

    def _zmq_send(self, server_id: int, request: Dict) -> Any:
        sock = self._zmq_sockets.get(server_id)
        if sock is None:
            return []
        try:
            sock.send(msgpack.packb(request, use_bin_type=True))
            resp = sock.recv()
            return msgpack.unpackb(resp, raw=False)
        except Exception:
            return []

    def run_server(self):
        """Start ZMQ servers on all actors (non-blocking futures)."""
        return [g.run.remote() for g in self.groups]

    def stop_server(self):
        """Send stop command to all ZMQ servers."""
        self._ensure_zmq()
        for sid in list(self._zmq_sockets):
            self._zmq_send(sid, {"method": "stop", "params": {}})


# Init / get  (module-level helpers)

_hspec_table_handles: List = []


def init_hspec_tables(
    similarity_threshold: float = 0.9,
    max_entries_per_prompt: int = 10_000,
    n_components: int = 64,
):
    """Create and register HSpec Ray actors.  Call once at training start."""
    global _hspec_table_handles
    global _num_groups
    _num_groups = _resolve_num_groups()
    _hspec_table_handles = []

    num_cpus = get_hspec_build_actor_num_cpus()
    total_actor_cpus = float(_num_groups) * float(num_cpus)
    host_cpus = os.cpu_count() or 0
    if host_cpus and total_actor_cpus > max(host_cpus - 2, 1):
        logger.warning(
            "HSpec build actor CPU reservation may oversubscribe host CPUs: "
            "num_groups=%s actor_num_cpus=%s total_actor_cpus=%s host_cpus=%s",
            _num_groups,
            num_cpus,
            total_actor_cpus,
            host_cpus,
        )

    existing_handles: List[ray.actor.ActorHandle] = []
    missing_existing = False
    for i in range(_num_groups):
        try:
            existing_handles.append(ray.get_actor(_build_actor_name(i)))
        except ValueError:
            missing_existing = True
            break
    if existing_handles and not missing_existing:
        topologies = _get_actor_topologies(existing_handles)
        _validate_actor_topologies(
            topologies,
            expected_num_groups=_num_groups,
            expected_node_id=get_hspec_node_id(),
            expected_similarity_threshold=similarity_threshold,
            expected_max_entries_per_prompt=max_entries_per_prompt,
            expected_n_components=n_components,
        )
        _hspec_table_handles = existing_handles
        logger.info("HSpec: reusing %d validated build shard actors.", len(existing_handles))
        return
    if existing_handles and missing_existing:
        hspec_record_store_metric("topology_actor_init_error", 1)
        raise RuntimeError(
            "HSpec found a partial set of existing build shard actors: "
            f"found={len(existing_handles)} expected={_num_groups}. "
            "Restart Ray or set a unique HSPEC_BUILD_ACTOR_NAME_PREFIX."
        )

    for i in range(_num_groups):
        handle = HSpecTableGroup.options(
            name=_build_actor_name(i),
            num_cpus=num_cpus,
            num_gpus=0,
        ).remote(
            port=6555 + i,
            similarity_threshold=similarity_threshold,
            max_entries_per_prompt=max_entries_per_prompt,
            n_components=n_components,
            shard_id=i,
        )
        _hspec_table_handles.append(handle)

    topologies = _get_actor_topologies(_hspec_table_handles)
    _validate_actor_topologies(
        topologies,
        expected_num_groups=_num_groups,
        expected_node_id=get_hspec_node_id(),
        expected_similarity_threshold=similarity_threshold,
        expected_max_entries_per_prompt=max_entries_per_prompt,
        expected_n_components=n_components,
    )
    hspec_record_store_metric("topology_actor_count", len(topologies))
    for item in topologies:
        logger.info(
            "HSpec build shard actor %d registered successfully: node_id=%s ray_node_id=%s "
            "build_actor_num_cpus=%s build_blas_threads=%s",
            int(item["shard_id"]),
            item.get("logical_node_id"),
            item.get("ray_node_id"),
            item.get("build_actor_num_cpus"),
            item.get("build_blas_threads"),
        )


def get_hspec_tables(
    similarity_threshold: float = 0.9,
    max_entries_per_prompt: int = 10_000,
    n_components: int = 64,
) -> GlobalHSpecTableGroup:
    """Get or create the global HSpec table manager.

    If actors do not exist yet they are created automatically.
    """
    global _num_groups
    expected_groups = _resolve_num_groups()
    if _num_groups != expected_groups:
        _num_groups = expected_groups

    # Ensure actors exist and match the current topology.
    needs_init = False
    existing_handles: List[ray.actor.ActorHandle] = []
    for i in range(_num_groups):
        try:
            existing_handles.append(ray.get_actor(_build_actor_name(i)))
        except ValueError:
            needs_init = True
            break

    if needs_init:
        init_hspec_tables(similarity_threshold, max_entries_per_prompt, n_components)
    else:
        topologies = _get_actor_topologies(existing_handles)
        _validate_actor_topologies(
            topologies,
            expected_num_groups=_num_groups,
            expected_node_id=get_hspec_node_id(),
            expected_similarity_threshold=similarity_threshold,
            expected_max_entries_per_prompt=max_entries_per_prompt,
            expected_n_components=n_components,
        )

    return GlobalHSpecTableGroup(
        similarity_threshold=similarity_threshold,
        max_entries_per_prompt=max_entries_per_prompt,
        n_components=n_components,
    )
