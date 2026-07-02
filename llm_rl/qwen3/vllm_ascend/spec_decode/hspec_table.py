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

import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import ray
import zmq
from threadpoolctl import threadpool_limits

from vllm_ascend.spec_decode.hspec_builder import (
    HSpecPCAConfig,
    HSpecPCAError,
    HSpecPCAInsufficientSamples,
    build_prompt_table_to_store,
)
from vllm_ascend.spec_decode.hspec_table_store import (
    HSpecPromptTableDesc,
    HSpecTableStoreWriter,
    clear_active_version_manifest,
    gc_table_store_versions,
    get_hspec_table_prefetch_mode,
    get_hspec_table_store_retain_versions,
    hspec_table_store_gc_after_swap_enabled,
    list_table_store_versions,
    materialize_prompt_table,
    open_array as open_table_array,
    write_active_version_manifest,
)
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
    get_hspec_build_max_prompt_descs,
    get_hspec_build_max_prompt_raw_bytes,
    get_hspec_build_max_prompt_rows,
    get_hspec_build_max_rss_mb,
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
    build_max_prompt_rows: int
    build_max_prompt_raw_bytes: int
    build_max_prompt_descs: int
    build_max_rss_mb: float


@dataclass
class _PromptBuildBudgetResult:
    selected: List[HSpecTrajectoryDesc] = field(default_factory=list)
    dropped: List[HSpecTrajectoryDesc] = field(default_factory=list)
    input_desc_count: int = 0
    input_rows: int = 0
    input_raw_bytes: int = 0
    selected_rows: int = 0
    selected_raw_bytes: int = 0
    dropped_rows: int = 0
    dropped_raw_bytes: int = 0
    oversize_drop_count: int = 0


@dataclass
class _PromptBuildInputs:
    hidden_states_list: List[np.ndarray] = field(default_factory=list)
    token_seq_list: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    input_desc_count: int = 0
    selected_desc_count: int = 0
    input_rows: int = 0
    selected_rows: int = 0
    input_raw_bytes: int = 0
    selected_raw_bytes: int = 0
    loaded_raw_bytes: int = 0
    loaded_fp32_bytes: int = 0
    budget_drop_count: int = 0
    budget_drop_rows: int = 0
    budget_drop_raw_bytes: int = 0
    budget_drop_oversize_count: int = 0
    materialize_ms: float = 0.0
    rss_before_bytes: int = 0
    rss_after_materialize_bytes: int = 0


@dataclass
class _PromptTableBuildMetrics:
    input_desc_count: int = 0
    selected_desc_count: int = 0
    input_rows: int = 0
    selected_rows: int = 0
    input_raw_bytes: int = 0
    selected_raw_bytes: int = 0
    budget_drop_count: int = 0
    budget_drop_rows: int = 0
    budget_drop_raw_bytes: int = 0
    budget_drop_oversize_count: int = 0
    validation_ms: float = 0.0
    pca_ms: float = 0.0
    table_add_ms: float = 0.0
    pca_mean_ms: float = 0.0
    pca_basis_ms: float = 0.0
    projection_ms: float = 0.0
    table_write_ms: float = 0.0
    processed_fp32_tile_bytes: int = 0
    pca_method: str = ""
    pca_method_fallback_count: int = 0
    pca_cov_bytes: int = 0
    pca_randomized_rank: int = 0
    projection_tile_count: int = 0
    table_entry_count: int = 0
    table_rollout_count: int = 0
    table_token_count: int = 0
    rss_before_pca_bytes: int = 0
    rss_after_pca_bytes: int = 0
    valid_rows: int = 0
    valid_desc_count: int = 0
    pca_error_count: int = 0
    memory_error_count: int = 0
    built: bool = False


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
            "build_max_prompt_rows": item.get("build_max_prompt_rows"),
            "build_max_prompt_raw_bytes": item.get("build_max_prompt_raw_bytes"),
            "build_max_prompt_descs": item.get("build_max_prompt_descs"),
            "build_max_rss_mb": item.get("build_max_rss_mb"),
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
    expected_build_max_prompt_rows: Optional[int] = None,
    expected_build_max_prompt_raw_bytes: Optional[int] = None,
    expected_build_max_prompt_descs: Optional[int] = None,
    expected_build_max_rss_mb: Optional[float] = None,
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
        if (
            expected_build_max_prompt_rows is not None
            and int(item.get("build_max_prompt_rows", -1)) != int(expected_build_max_prompt_rows)
        ):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports build_max_prompt_rows={item.get('build_max_prompt_rows')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if (
            expected_build_max_prompt_raw_bytes is not None
            and int(item.get("build_max_prompt_raw_bytes", -1)) != int(expected_build_max_prompt_raw_bytes)
        ):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                "actor shard=%s reports build_max_prompt_raw_bytes=%s"
                % (shard_id, item.get("build_max_prompt_raw_bytes")),
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if (
            expected_build_max_prompt_descs is not None
            and int(item.get("build_max_prompt_descs", -1)) != int(expected_build_max_prompt_descs)
        ):
            hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
            raise _actor_topology_error(
                f"actor shard={shard_id} reports build_max_prompt_descs={item.get('build_max_prompt_descs')}",
                topologies=topologies,
                expected_num_groups=expected_num_groups,
                expected_node_id=expected_node_id,
            )
        if expected_build_max_rss_mb is not None:
            try:
                actor_max_rss = float(item.get("build_max_rss_mb"))
            except Exception as exc:
                hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
                raise _actor_topology_error(
                    f"actor shard={shard_id} reports invalid build_max_rss_mb={item.get('build_max_rss_mb')}",
                    topologies=topologies,
                    expected_num_groups=expected_num_groups,
                    expected_node_id=expected_node_id,
                ) from exc
            if abs(actor_max_rss - float(expected_build_max_rss_mb)) > 1e-9:
                hspec_record_store_metric("topology_actor_reuse_mismatch", 1)
                raise _actor_topology_error(
                    f"actor shard={shard_id} reports build_max_rss_mb={actor_max_rss}",
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


def _close_memmap(array: Any) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _get_process_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value * 1024 if sys.platform.startswith("linux") else value
        except Exception:
            return 0


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / float(1024 * 1024)


def _maybe_process_rss_mb() -> float:
    rss = _get_process_rss_bytes()
    return _bytes_to_mb(rss) if rss > 0 else -1.0


def _desc_reward_sort_value(desc: HSpecTrajectoryDesc) -> float:
    try:
        if desc.reward is None:
            return float("-inf")
        return float(desc.reward)
    except Exception:
        return float("-inf")


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
        self._active: Dict[str, HSpecPromptTableDesc] = {}
        self._building: Dict[str, HSpecPromptTableDesc] = {}
        self._active_version: int = 0
        self._table_writer: Optional[HSpecTableStoreWriter] = None
        self._building_version: Optional[int] = None

        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries_per_prompt
        self.n_components = n_components
        self.port = port
        self.shard_id = int(shard_id)
        self.num_groups = _resolve_num_groups()
        self.logical_node_id = get_hspec_node_id()
        self.build_actor_num_cpus = get_hspec_build_actor_num_cpus()
        self.build_blas_threads = get_hspec_build_blas_threads()
        self.build_max_prompt_rows = get_hspec_build_max_prompt_rows()
        self.build_max_prompt_raw_bytes = get_hspec_build_max_prompt_raw_bytes()
        self.build_max_prompt_descs = get_hspec_build_max_prompt_descs()
        self.build_max_rss_mb = get_hspec_build_max_rss_mb()
        self._pca_config = HSpecPCAConfig.from_env(self.n_components)
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
        self._build_loaded_raw_bytes = 0
        self._build_loaded_fp32_bytes = 0
        self._build_input_rows = 0
        self._build_selected_rows = 0
        self._build_selected_desc_count = 0
        self._build_budget_drop_count = 0
        self._build_budget_drop_rows = 0
        self._build_budget_drop_raw_bytes = 0
        self._build_budget_drop_oversize_count = 0
        self._build_rss_cap_skip_count = 0
        self._build_memory_error_count = 0
        self._build_validation_ms = 0.0
        self._build_materialize_ms = 0.0
        self._build_pca_ms = 0.0
        self._build_table_add_ms = 0.0
        self._build_pca_mean_ms = 0.0
        self._build_pca_basis_ms = 0.0
        self._build_projection_ms = 0.0
        self._build_table_write_ms = 0.0
        self._build_processed_fp32_tile_bytes = 0
        self._build_projection_tile_count = 0
        self._build_pca_method_randomized_count = 0
        self._build_pca_method_covariance_count = 0
        self._build_pca_method_svd_reference_count = 0
        self._build_pca_method_fallback_count = 0
        self._build_pca_cov_bytes_max = 0
        self._build_pca_randomized_rank_max = 0
        self._build_total_ms = 0.0
        self._build_rss_peak_bytes = 0
        self._build_rss_after_materialize_peak_bytes = 0
        self._build_rss_after_pca_peak_bytes = 0
        self._build_rss_delta_peak_bytes = 0
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
            "build_max_prompt_rows": int(self.build_max_prompt_rows),
            "build_max_prompt_raw_bytes": int(self.build_max_prompt_raw_bytes),
            "build_max_prompt_descs": int(self.build_max_prompt_descs),
            "build_max_rss_mb": float(self.build_max_rss_mb),
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

    def _rss_cap_exceeded(self, rss_bytes: int) -> bool:
        return self.build_max_rss_mb > 0 and rss_bytes > 0 and _bytes_to_mb(rss_bytes) >= self.build_max_rss_mb

    def _select_prompt_descs_for_build(
        self,
        prompt_id: str,
        descs: List[HSpecTrajectoryDesc],
    ) -> _PromptBuildBudgetResult:
        budget = _PromptBuildBudgetResult(input_desc_count=len(descs))
        valid_items: List[tuple[HSpecTrajectoryDesc, int, int]] = []
        for desc in descs:
            if desc is None:
                self._discard_count += 1
                continue
            if not self._validate_descriptor_topology(prompt_id, desc):
                self._discard_count += 1
                continue
            rows = max(int(desc.length), 0)
            try:
                raw_bytes = max(int(estimate_hspec_trajectory_bytes(desc)), 0)
            except Exception as exc:
                logger.warning(
                    "Failed to estimate HSpec trajectory bytes for prompt_id=%s request_id=%s: %s",
                    prompt_id,
                    desc.request_id,
                    exc,
                )
                self._discard_count += 1
                continue
            budget.input_rows += rows
            budget.input_raw_bytes += raw_bytes
            valid_items.append((desc, rows, raw_bytes))

        if (
            self.build_max_prompt_rows <= 0
            and self.build_max_prompt_raw_bytes <= 0
            and self.build_max_prompt_descs <= 0
        ):
            budget.selected = [item[0] for item in valid_items]
            budget.selected_rows = budget.input_rows
            budget.selected_raw_bytes = budget.input_raw_bytes
            return budget

        def _sort_key(item: tuple[HSpecTrajectoryDesc, int, int]) -> tuple[float, int, str, int, int]:
            desc, _, _ = item
            return (
                -_desc_reward_sort_value(desc),
                -int(desc.global_step),
                str(desc.request_id),
                int(desc.hs_offset_rows),
                int(desc.token_offset),
            )

        selected_descs: List[HSpecTrajectoryDesc] = []
        selected_rows = 0
        selected_raw_bytes = 0
        for desc, rows, raw_bytes in sorted(valid_items, key=_sort_key):
            would_descs = len(selected_descs) + 1
            would_rows = selected_rows + rows
            would_raw_bytes = selected_raw_bytes + raw_bytes
            desc_oversize = (
                (self.build_max_prompt_descs > 0 and 1 > self.build_max_prompt_descs)
                or (self.build_max_prompt_rows > 0 and rows > self.build_max_prompt_rows)
                or (self.build_max_prompt_raw_bytes > 0 and raw_bytes > self.build_max_prompt_raw_bytes)
            )
            would_exceed = (
                (self.build_max_prompt_descs > 0 and would_descs > self.build_max_prompt_descs)
                or (self.build_max_prompt_rows > 0 and would_rows > self.build_max_prompt_rows)
                or (self.build_max_prompt_raw_bytes > 0 and would_raw_bytes > self.build_max_prompt_raw_bytes)
            )
            if desc_oversize or would_exceed:
                budget.dropped.append(desc)
                budget.dropped_rows += rows
                budget.dropped_raw_bytes += raw_bytes
                if desc_oversize:
                    budget.oversize_drop_count += 1
                continue
            selected_descs.append(desc)
            selected_rows = would_rows
            selected_raw_bytes = would_raw_bytes

        budget.selected = selected_descs
        budget.selected_rows = selected_rows
        budget.selected_raw_bytes = selected_raw_bytes
        if budget.dropped:
            self._discard_count += len(budget.dropped)
            logger.warning(
                "HSpec build prompt budget drop: shard=%s prompt_id=%s input_descs=%s selected_descs=%s "
                "input_rows=%s selected_rows=%s dropped_rows=%s input_raw_bytes=%s selected_raw_bytes=%s "
                "dropped_raw_bytes=%s caps(descs=%s rows=%s raw_bytes=%s)",
                self.shard_id,
                prompt_id,
                budget.input_desc_count,
                len(budget.selected),
                budget.input_rows,
                budget.selected_rows,
                budget.dropped_rows,
                budget.input_raw_bytes,
                budget.selected_raw_bytes,
                budget.dropped_raw_bytes,
                self.build_max_prompt_descs,
                self.build_max_prompt_rows,
                self.build_max_prompt_raw_bytes,
            )
        return budget

    def _load_prompt_build_inputs_from_descs(
        self,
        prompt_id: str,
        descs: List[HSpecTrajectoryDesc],
    ) -> _PromptBuildInputs:
        result = _PromptBuildInputs()
        budget = self._select_prompt_descs_for_build(prompt_id, descs)
        result.input_desc_count = budget.input_desc_count
        result.selected_desc_count = len(budget.selected)
        result.input_rows = budget.input_rows
        result.selected_rows = budget.selected_rows
        result.input_raw_bytes = budget.input_raw_bytes
        result.selected_raw_bytes = budget.selected_raw_bytes
        result.budget_drop_count = len(budget.dropped)
        result.budget_drop_rows = budget.dropped_rows
        result.budget_drop_raw_bytes = budget.dropped_raw_bytes
        result.budget_drop_oversize_count = budget.oversize_drop_count

        if not budget.selected:
            return result

        result.rss_before_bytes = _get_process_rss_bytes()
        t_materialize = time.perf_counter()
        for desc in budget.selected:
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
            try:
                hs_fp32 = np.asarray(hs, dtype=np.float32)
                tok_i32 = np.asarray(tokens, dtype=np.int32)
            except MemoryError:
                self._discard_count += 1
                raise
            result.hidden_states_list.append(hs_fp32)
            result.token_seq_list.append(tok_i32)
            result.rewards.append(float(desc.reward or 0.0))
            result.loaded_raw_bytes += estimate_hspec_trajectory_bytes(desc)
            result.loaded_fp32_bytes += int(hs_fp32.nbytes)
        result.materialize_ms = float((time.perf_counter() - t_materialize) * 1000.0)
        result.rss_after_materialize_bytes = _get_process_rss_bytes()

        return result

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

    def _next_table_store_version(self) -> int:
        versions = list_table_store_versions(
            root=get_hspec_table_store_root(),
            shard_id=self.shard_id,
        )
        max_disk_version = max(versions, default=0)
        return max(int(self._active_version), int(max_disk_version)) + 1

    def _maybe_gc_table_store_versions(self) -> None:
        if not hspec_table_store_gc_after_swap_enabled():
            return
        try:
            gc_table_store_versions(
                root=get_hspec_table_store_root(),
                shard_id=self.shard_id,
                active_version=self._active_version,
                retain_versions=get_hspec_table_store_retain_versions(),
            )
        except Exception:
            logger.debug("Failed during best-effort HSpec table-store GC",
                         exc_info=True)

    # Build

    def _get_or_create_table_writer(self) -> HSpecTableStoreWriter:
        if self._table_writer is not None:
            return self._table_writer
        version = self._next_table_store_version()
        self._building_version = version
        self._table_writer = HSpecTableStoreWriter(
            root=get_hspec_table_store_root(),
            shard_id=self.shard_id,
            version=version,
        )
        return self._table_writer

    def build_prompt_table_from_descs(
        self,
        prompt_id: str,
        descs: List[HSpecTrajectoryDesc],
    ) -> _PromptTableBuildMetrics:
        """Build one prompt table from raw-store descriptors into mmap table store."""
        metrics = _PromptTableBuildMetrics()
        t_validation = time.perf_counter()
        budget = self._select_prompt_descs_for_build(prompt_id, descs)
        metrics.validation_ms = float((time.perf_counter() - t_validation) * 1000.0)
        metrics.input_desc_count = budget.input_desc_count
        metrics.selected_desc_count = len(budget.selected)
        metrics.input_rows = budget.input_rows
        metrics.selected_rows = budget.selected_rows
        metrics.input_raw_bytes = budget.input_raw_bytes
        metrics.selected_raw_bytes = budget.selected_raw_bytes
        metrics.budget_drop_count = len(budget.dropped)
        metrics.budget_drop_rows = budget.dropped_rows
        metrics.budget_drop_raw_bytes = budget.dropped_raw_bytes
        metrics.budget_drop_oversize_count = budget.oversize_drop_count
        metrics.valid_desc_count = len(budget.selected)
        metrics.valid_rows = budget.selected_rows
        if not budget.selected:
            return metrics

        metrics.rss_before_pca_bytes = _get_process_rss_bytes()
        writer = self._get_or_create_table_writer()
        try:
            table_desc, build_metrics = build_prompt_table_to_store(
                prompt_id=prompt_id,
                descs=budget.selected,
                writer=writer,
                n_components=self.n_components,
                max_entries=self.max_entries,
                pca_config=self._pca_config,
                blas_threads=self.build_blas_threads,
            )
        except HSpecPCAInsufficientSamples as exc:
            self._discard_count += len(budget.selected)
            logger.debug(
                "HSpec descriptor table skipped for prompt_id=%s: %s",
                prompt_id,
                exc,
            )
            return metrics
        except MemoryError:
            metrics.memory_error_count = 1
            self._discard_count += len(budget.selected)
            logger.warning(
                "HSpec descriptor table build ran out of memory: shard=%s prompt_id=%s selected_descs=%s",
                self.shard_id,
                prompt_id,
                len(budget.selected),
            )
            return metrics
        except HSpecPCAError as exc:
            metrics.pca_error_count = 1
            self._discard_count += len(budget.selected)
            logger.warning("HSpec descriptor PCA failed for %s: %s", prompt_id, exc)
            return metrics
        except Exception as exc:
            metrics.pca_error_count = 1
            self._discard_count += len(budget.selected)
            logger.warning("HSpec descriptor table build failed for %s: %s",
                           prompt_id,
                           exc)
            return metrics

        metrics.rss_after_pca_bytes = _get_process_rss_bytes()
        metrics.pca_mean_ms = float(build_metrics.pca_mean_ms)
        metrics.pca_basis_ms = float(build_metrics.pca_basis_ms)
        metrics.pca_ms = float(build_metrics.pca_total_ms)
        metrics.table_add_ms = float(build_metrics.projection_ms + build_metrics.table_write_ms)
        metrics.projection_ms = float(build_metrics.projection_ms)
        metrics.table_write_ms = float(build_metrics.table_write_ms)
        metrics.processed_fp32_tile_bytes = int(build_metrics.processed_fp32_tile_bytes)
        metrics.pca_method = str(build_metrics.pca_method)
        metrics.pca_method_fallback_count = int(build_metrics.method_fallback_count)
        metrics.pca_cov_bytes = int(build_metrics.covariance_bytes)
        metrics.pca_randomized_rank = int(build_metrics.randomized_rank)
        metrics.projection_tile_count = int(build_metrics.projection_tile_count)
        metrics.table_entry_count = int(build_metrics.n_entries)
        metrics.table_rollout_count = int(build_metrics.n_rollouts)
        metrics.table_token_count = int(build_metrics.token_count)
        self._building[prompt_id] = table_desc
        self._build_count += 1
        metrics.built = True
        return metrics

    def build_prompt_table(
        self,
        prompt_id: str,
        hidden_states_list: List[np.ndarray],
        token_seq_list: List[Any],
        rewards: List[float],
    ) -> _PromptTableBuildMetrics:
        """Build a complete table for one prompt.

        Pipeline:  validate → PCA fit → project → store entries.

        Args:
            prompt_id:          Stable prompt identifier.
            hidden_states_list: [(L_i, D) ndarray] per rollout.
            token_seq_list:     [list[int] | ndarray] per rollout.
            rewards:            [float] per rollout.
        """
        metrics = _PromptTableBuildMetrics()
        # ① Validate & filter
        t_validation = time.perf_counter()
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
        metrics.validation_ms = float((time.perf_counter() - t_validation) * 1000.0)
        metrics.valid_desc_count = len(valid_hs)
        metrics.valid_rows = int(sum(int(hs.shape[0]) for hs in valid_hs))

        if not valid_hs:
            return metrics

        # PCA fit  (single-sequence for PPO, multi for GRPO)
        num_components = self.n_components
        metrics.rss_before_pca_bytes = _get_process_rss_bytes()
        t_pca = time.perf_counter()
        try:
            if len(valid_hs) == 1:
                pca_params, proj = fit_pca_single_sequence(prompt_id, valid_hs[0], num_components)
                proj_list = [proj]
            else:
                pca_params, proj_list = fit_pca_multi_sequence(prompt_id, valid_hs, num_components)
        except MemoryError:
            metrics.pca_ms = float((time.perf_counter() - t_pca) * 1000.0)
            metrics.rss_after_pca_bytes = _get_process_rss_bytes()
            metrics.memory_error_count = 1
            self._discard_count += len(valid_hs)
            logger.warning("HSpec PCA ran out of memory for %s", prompt_id)
            return metrics
        except Exception as exc:
            metrics.pca_ms = float((time.perf_counter() - t_pca) * 1000.0)
            metrics.rss_after_pca_bytes = _get_process_rss_bytes()
            metrics.pca_error_count = 1
            logger.warning("HSpec PCA failed for %s: %s", prompt_id, exc)
            self._discard_count += len(valid_hs)
            return metrics
        metrics.pca_ms = float((time.perf_counter() - t_pca) * 1000.0)
        metrics.rss_after_pca_bytes = _get_process_rss_bytes()

        # Create table & populate with projected keys + token refs
        t_table_add = time.perf_counter()
        table = PromptTableData(pca_params=pca_params, max_entries=self.max_entries)
        for proj, tok, rew in zip(proj_list, valid_tok, valid_rew):
            tok_arr = (np.asarray(tok, dtype=np.int32)
                       if not isinstance(tok, np.ndarray)
                       else tok.astype(np.int32, copy=False))
            table.add_rollout(proj, tok_arr, rew)

        table.compact()
        metrics.table_add_ms = float((time.perf_counter() - t_table_add) * 1000.0)
        self._building[prompt_id] = table
        self._build_count += 1
        metrics.built = True
        return metrics

    def _build_result_metrics(
        self,
        *,
        t0: float,
        prompt_count: int,
        desc_count: int,
        legacy_payload_count: int,
        selected_desc_count: int,
        build_input_rows: int,
        build_selected_rows: int,
        build_loaded_raw_bytes: int,
        build_loaded_fp32_bytes: int,
        build_budget_drop_count: int,
        build_budget_drop_rows: int,
        build_budget_drop_raw_bytes: int,
        build_budget_drop_oversize_count: int,
        build_rss_cap_skip_count: int,
        build_memory_error_count: int,
        build_validation_ms: float,
        build_materialize_ms: float,
        build_pca_ms: float,
        build_table_add_ms: float,
        build_total_ms: float,
        rss_before_bytes: int,
        rss_after_materialize_peak_bytes: int,
        rss_after_pca_peak_bytes: int,
        rss_peak_bytes: int,
        rss_delta_peak_bytes: int,
        build_count_before: int,
        discard_count_before: int,
        build_pca_mean_ms: float = 0.0,
        build_pca_basis_ms: float = 0.0,
        build_projection_ms: float = 0.0,
        build_table_write_ms: float = 0.0,
        build_processed_fp32_tile_bytes: int = 0,
        build_projection_tile_count: int = 0,
        build_pca_method_randomized_count: int = 0,
        build_pca_method_covariance_count: int = 0,
        build_pca_method_svd_reference_count: int = 0,
        build_pca_method_fallback_count: int = 0,
        build_pca_cov_bytes_max: int = 0,
        build_pca_randomized_rank_max: int = 0,
    ) -> Dict[str, float]:
        rss_now = _get_process_rss_bytes()
        rss_peak_bytes = max(int(rss_peak_bytes), int(rss_now))
        if rss_before_bytes > 0 and rss_now > 0:
            rss_delta_peak_bytes = max(int(rss_delta_peak_bytes), max(int(rss_now) - int(rss_before_bytes), 0))
        return {
            "shard_id": float(self.shard_id),
            "prompt_count": float(prompt_count),
            "desc_count": float(desc_count),
            "selected_desc_count": float(selected_desc_count),
            "legacy_payload_count": float(legacy_payload_count),
            "build_count_delta": float(self._build_count - build_count_before),
            "discard_count_delta": float(self._discard_count - discard_count_before),
            "build_total_ms": build_total_ms,
            "build_validation_ms": float(build_validation_ms),
            "build_materialize_ms": float(build_materialize_ms),
            "build_pca_ms": float(build_pca_ms),
            "build_table_add_ms": float(build_table_add_ms),
            "build_pca_mean_ms": float(build_pca_mean_ms),
            "build_pca_basis_ms": float(build_pca_basis_ms),
            "build_projection_ms": float(build_projection_ms),
            "build_table_write_ms": float(build_table_write_ms),
            "build_processed_fp32_tile_bytes": float(build_processed_fp32_tile_bytes),
            "build_projection_tile_count": float(build_projection_tile_count),
            "build_pca_method_randomized_count": float(build_pca_method_randomized_count),
            "build_pca_method_covariance_count": float(build_pca_method_covariance_count),
            "build_pca_method_svd_reference_count": float(build_pca_method_svd_reference_count),
            "build_pca_method_fallback_count": float(build_pca_method_fallback_count),
            "build_pca_cov_bytes_max": float(build_pca_cov_bytes_max),
            "build_pca_randomized_rank_max": float(build_pca_randomized_rank_max),
            "build_input_rows": float(build_input_rows),
            "build_selected_rows": float(build_selected_rows),
            "build_loaded_raw_bytes": float(build_loaded_raw_bytes),
            "build_loaded_fp32_bytes": float(build_loaded_fp32_bytes),
            "build_budget_drop_count": float(build_budget_drop_count),
            "build_budget_drop_rows": float(build_budget_drop_rows),
            "build_budget_drop_raw_bytes": float(build_budget_drop_raw_bytes),
            "build_budget_drop_oversize_count": float(build_budget_drop_oversize_count),
            "build_rss_cap_skip_count": float(build_rss_cap_skip_count),
            "build_memory_error_count": float(build_memory_error_count),
            "build_actor_rss_mb": _bytes_to_mb(rss_now) if rss_now > 0 else -1.0,
            "build_actor_rss_before_mb": _bytes_to_mb(rss_before_bytes) if rss_before_bytes > 0 else -1.0,
            "build_actor_rss_after_materialize_mb_max": (
                _bytes_to_mb(rss_after_materialize_peak_bytes)
                if rss_after_materialize_peak_bytes > 0 else -1.0
            ),
            "build_actor_rss_after_pca_mb_max": (
                _bytes_to_mb(rss_after_pca_peak_bytes)
                if rss_after_pca_peak_bytes > 0 else -1.0
            ),
            "build_actor_rss_peak_mb": _bytes_to_mb(rss_peak_bytes) if rss_peak_bytes > 0 else -1.0,
            "build_actor_rss_delta_mb_max": _bytes_to_mb(rss_delta_peak_bytes) if rss_delta_peak_bytes > 0 else 0.0,
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
        selected_desc_count = 0
        build_input_rows = 0
        build_selected_rows = 0
        build_loaded_raw_bytes = 0
        build_loaded_fp32_bytes = 0
        build_budget_drop_count = 0
        build_budget_drop_rows = 0
        build_budget_drop_raw_bytes = 0
        build_budget_drop_oversize_count = 0
        build_rss_cap_skip_count = 0
        build_memory_error_count = 0
        build_validation_ms = 0.0
        build_materialize_ms = 0.0
        build_pca_ms = 0.0
        build_table_add_ms = 0.0
        build_pca_mean_ms = 0.0
        build_pca_basis_ms = 0.0
        build_projection_ms = 0.0
        build_table_write_ms = 0.0
        build_processed_fp32_tile_bytes = 0
        build_projection_tile_count = 0
        build_pca_method_randomized_count = 0
        build_pca_method_covariance_count = 0
        build_pca_method_svd_reference_count = 0
        build_pca_method_fallback_count = 0
        build_pca_cov_bytes_max = 0
        build_pca_randomized_rank_max = 0
        rss_before_bytes = _get_process_rss_bytes()
        rss_peak_bytes = rss_before_bytes
        rss_after_materialize_peak_bytes = 0
        rss_after_pca_peak_bytes = 0
        rss_delta_peak_bytes = 0

        def _observe_rss(rss_bytes: int) -> None:
            nonlocal rss_peak_bytes, rss_delta_peak_bytes
            if rss_bytes <= 0:
                return
            rss_peak_bytes = max(rss_peak_bytes, rss_bytes)
            if rss_before_bytes > 0:
                rss_delta_peak_bytes = max(rss_delta_peak_bytes, max(rss_bytes - rss_before_bytes, 0))

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
                pre_build_rss = _get_process_rss_bytes()
                _observe_rss(pre_build_rss)
                if self._rss_cap_exceeded(pre_build_rss):
                    build_rss_cap_skip_count += 1
                    self._discard_count += len(descs)
                    logger.warning(
                        "HSpec build actor RSS cap exceeded before descriptor table build: "
                        "shard=%s prompt_id=%s rss_mb=%.2f cap_mb=%.2f dropping_descs=%s",
                        self.shard_id,
                        prompt_id,
                        _bytes_to_mb(pre_build_rss),
                        self.build_max_rss_mb,
                        len(descs),
                    )
                    continue

                table_metrics = self.build_prompt_table_from_descs(prompt_id, descs)
                selected_desc_count += table_metrics.selected_desc_count
                build_input_rows += table_metrics.input_rows
                build_selected_rows += table_metrics.selected_rows
                build_loaded_raw_bytes += table_metrics.selected_raw_bytes
                build_budget_drop_count += table_metrics.budget_drop_count
                build_budget_drop_rows += table_metrics.budget_drop_rows
                build_budget_drop_raw_bytes += table_metrics.budget_drop_raw_bytes
                build_budget_drop_oversize_count += table_metrics.budget_drop_oversize_count
                build_validation_ms += table_metrics.validation_ms
                build_pca_ms += table_metrics.pca_ms
                build_table_add_ms += table_metrics.table_add_ms
                build_pca_mean_ms += table_metrics.pca_mean_ms
                build_pca_basis_ms += table_metrics.pca_basis_ms
                build_projection_ms += table_metrics.projection_ms
                build_table_write_ms += table_metrics.table_write_ms
                build_processed_fp32_tile_bytes += table_metrics.processed_fp32_tile_bytes
                build_projection_tile_count += table_metrics.projection_tile_count
                build_pca_method_fallback_count += table_metrics.pca_method_fallback_count
                build_pca_cov_bytes_max = max(build_pca_cov_bytes_max,
                                              table_metrics.pca_cov_bytes)
                build_pca_randomized_rank_max = max(
                    build_pca_randomized_rank_max,
                    table_metrics.pca_randomized_rank,
                )
                if table_metrics.pca_method == "randomized":
                    build_pca_method_randomized_count += 1
                elif table_metrics.pca_method == "covariance":
                    build_pca_method_covariance_count += 1
                elif table_metrics.pca_method == "svd_reference":
                    build_pca_method_svd_reference_count += 1
                build_memory_error_count += table_metrics.memory_error_count
                rss_after_pca_peak_bytes = max(
                    rss_after_pca_peak_bytes,
                    table_metrics.rss_after_pca_bytes,
                )
                _observe_rss(table_metrics.rss_before_pca_bytes)
                _observe_rss(table_metrics.rss_after_pca_bytes)
            finally:
                self._cleanup_trajectory_descs(descs)
                self._mark_descriptor_batch_finished(descs)

        build_total_ms = float((time.perf_counter() - t0) * 1000.0)
        self._build_selected_desc_count += selected_desc_count
        self._build_input_rows += build_input_rows
        self._build_selected_rows += build_selected_rows
        self._build_loaded_raw_bytes += build_loaded_raw_bytes
        self._build_loaded_fp32_bytes += build_loaded_fp32_bytes
        self._build_budget_drop_count += build_budget_drop_count
        self._build_budget_drop_rows += build_budget_drop_rows
        self._build_budget_drop_raw_bytes += build_budget_drop_raw_bytes
        self._build_budget_drop_oversize_count += build_budget_drop_oversize_count
        self._build_rss_cap_skip_count += build_rss_cap_skip_count
        self._build_memory_error_count += build_memory_error_count
        self._build_validation_ms += build_validation_ms
        self._build_materialize_ms += build_materialize_ms
        self._build_pca_ms += build_pca_ms
        self._build_table_add_ms += build_table_add_ms
        self._build_pca_mean_ms += build_pca_mean_ms
        self._build_pca_basis_ms += build_pca_basis_ms
        self._build_projection_ms += build_projection_ms
        self._build_table_write_ms += build_table_write_ms
        self._build_processed_fp32_tile_bytes += build_processed_fp32_tile_bytes
        self._build_projection_tile_count += build_projection_tile_count
        self._build_pca_method_randomized_count += build_pca_method_randomized_count
        self._build_pca_method_covariance_count += build_pca_method_covariance_count
        self._build_pca_method_svd_reference_count += build_pca_method_svd_reference_count
        self._build_pca_method_fallback_count += build_pca_method_fallback_count
        self._build_pca_cov_bytes_max = max(self._build_pca_cov_bytes_max,
                                            build_pca_cov_bytes_max)
        self._build_pca_randomized_rank_max = max(
            self._build_pca_randomized_rank_max,
            build_pca_randomized_rank_max,
        )
        self._build_total_ms += build_total_ms
        self._build_rss_peak_bytes = max(self._build_rss_peak_bytes, rss_peak_bytes)
        self._build_rss_after_materialize_peak_bytes = max(
            self._build_rss_after_materialize_peak_bytes,
            rss_after_materialize_peak_bytes,
        )
        self._build_rss_after_pca_peak_bytes = max(
            self._build_rss_after_pca_peak_bytes,
            rss_after_pca_peak_bytes,
        )
        self._build_rss_delta_peak_bytes = max(self._build_rss_delta_peak_bytes, rss_delta_peak_bytes)
        return self._build_result_metrics(
            t0=t0,
            prompt_count=prompt_count,
            desc_count=desc_count,
            legacy_payload_count=0,
            selected_desc_count=selected_desc_count,
            build_input_rows=build_input_rows,
            build_selected_rows=build_selected_rows,
            build_loaded_raw_bytes=build_loaded_raw_bytes,
            build_loaded_fp32_bytes=build_loaded_fp32_bytes,
            build_budget_drop_count=build_budget_drop_count,
            build_budget_drop_rows=build_budget_drop_rows,
            build_budget_drop_raw_bytes=build_budget_drop_raw_bytes,
            build_budget_drop_oversize_count=build_budget_drop_oversize_count,
            build_rss_cap_skip_count=build_rss_cap_skip_count,
            build_memory_error_count=build_memory_error_count,
            build_validation_ms=build_validation_ms,
            build_materialize_ms=build_materialize_ms,
            build_pca_ms=build_pca_ms,
            build_table_add_ms=build_table_add_ms,
            build_total_ms=build_total_ms,
            rss_before_bytes=rss_before_bytes,
            rss_after_materialize_peak_bytes=rss_after_materialize_peak_bytes,
            rss_after_pca_peak_bytes=rss_after_pca_peak_bytes,
            rss_peak_bytes=rss_peak_bytes,
            rss_delta_peak_bytes=rss_delta_peak_bytes,
            build_count_before=build_count_before,
            discard_count_before=discard_count_before,
            build_pca_mean_ms=build_pca_mean_ms,
            build_pca_basis_ms=build_pca_basis_ms,
            build_projection_ms=build_projection_ms,
            build_table_write_ms=build_table_write_ms,
            build_processed_fp32_tile_bytes=build_processed_fp32_tile_bytes,
            build_projection_tile_count=build_projection_tile_count,
            build_pca_method_randomized_count=build_pca_method_randomized_count,
            build_pca_method_covariance_count=build_pca_method_covariance_count,
            build_pca_method_svd_reference_count=build_pca_method_svd_reference_count,
            build_pca_method_fallback_count=build_pca_method_fallback_count,
            build_pca_cov_bytes_max=build_pca_cov_bytes_max,
            build_pca_randomized_rank_max=build_pca_randomized_rank_max,
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
        build_validation_ms = 0.0
        build_pca_ms = 0.0
        build_table_add_ms = 0.0
        build_memory_error_count = 0
        rss_before_bytes = _get_process_rss_bytes()
        rss_peak_bytes = rss_before_bytes
        rss_after_pca_peak_bytes = 0
        rss_delta_peak_bytes = 0

        def _observe_rss(rss_bytes: int) -> None:
            nonlocal rss_peak_bytes, rss_delta_peak_bytes
            if rss_bytes <= 0:
                return
            rss_peak_bytes = max(rss_peak_bytes, rss_bytes)
            if rss_before_bytes > 0:
                rss_delta_peak_bytes = max(rss_delta_peak_bytes, max(rss_bytes - rss_before_bytes, 0))

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
                table_metrics = self.build_prompt_table(
                    prompt_id,
                    hidden_states,
                    data["tokens"],
                    data["rewards"],
                )
                build_validation_ms += table_metrics.validation_ms
                build_pca_ms += table_metrics.pca_ms
                build_table_add_ms += table_metrics.table_add_ms
                build_memory_error_count += table_metrics.memory_error_count
                rss_after_pca_peak_bytes = max(rss_after_pca_peak_bytes, table_metrics.rss_after_pca_bytes)
                _observe_rss(table_metrics.rss_before_pca_bytes)
                _observe_rss(table_metrics.rss_after_pca_bytes)
        hspec_record_store_metric("legacy_payload_count", legacy_payload_count)
        build_total_ms = float((time.perf_counter() - t0) * 1000.0)
        self._build_validation_ms += build_validation_ms
        self._build_pca_ms += build_pca_ms
        self._build_table_add_ms += build_table_add_ms
        self._build_total_ms += build_total_ms
        self._build_memory_error_count += build_memory_error_count
        self._build_rss_peak_bytes = max(self._build_rss_peak_bytes, rss_peak_bytes)
        self._build_rss_after_pca_peak_bytes = max(
            self._build_rss_after_pca_peak_bytes,
            rss_after_pca_peak_bytes,
        )
        self._build_rss_delta_peak_bytes = max(self._build_rss_delta_peak_bytes, rss_delta_peak_bytes)
        return self._build_result_metrics(
            t0=t0,
            prompt_count=prompt_count,
            desc_count=0,
            legacy_payload_count=legacy_payload_count,
            selected_desc_count=0,
            build_input_rows=0,
            build_selected_rows=0,
            build_loaded_raw_bytes=0,
            build_loaded_fp32_bytes=0,
            build_budget_drop_count=0,
            build_budget_drop_rows=0,
            build_budget_drop_raw_bytes=0,
            build_budget_drop_oversize_count=0,
            build_rss_cap_skip_count=0,
            build_memory_error_count=build_memory_error_count,
            build_validation_ms=build_validation_ms,
            build_materialize_ms=0.0,
            build_pca_ms=build_pca_ms,
            build_table_add_ms=build_table_add_ms,
            build_total_ms=build_total_ms,
            rss_before_bytes=rss_before_bytes,
            rss_after_materialize_peak_bytes=0,
            rss_after_pca_peak_bytes=rss_after_pca_peak_bytes,
            rss_peak_bytes=rss_peak_bytes,
            rss_delta_peak_bytes=rss_delta_peak_bytes,
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
        if isinstance(table, HSpecPromptTableDesc):
            try:
                data = materialize_prompt_table(table)
                pca_params = PromptPCAParams(
                    prompt_id=prompt_id,
                    mean=data["mean"],
                    components=data["components"],
                    n_samples=int(table.n_samples),
                )
                legacy = PromptTableData(
                    pca_params=pca_params,
                    max_entries=max(int(table.n_entries), 1),
                    initial_wnd=int(table.wnd_size),
                    max_wnd=int(table.max_wnd),
                    min_wnd=int(table.min_wnd),
                )
                legacy.keys = np.ascontiguousarray(data["keys"], dtype=np.float16)
                legacy.rollout_seqs = [
                    np.ascontiguousarray(seq, dtype=np.int32)
                    for seq in data["rollout_seqs"]
                ]
                legacy.entry_rollout_idx = np.ascontiguousarray(
                    data["entry_rollout_idx"], dtype=np.int32)
                legacy.entry_offset = np.ascontiguousarray(data["entry_offset"],
                                                           dtype=np.int32)
                legacy.rewards = np.ascontiguousarray(
                    data.get("rewards")
                    if data.get("rewards") is not None
                    else np.zeros((int(table.n_entries),), dtype=np.float32),
                    dtype=np.float32,
                )
                legacy.n_entries = int(table.n_entries)
                legacy.max_entries = int(table.n_entries)
                table = legacy
            except Exception:
                logger.debug("Failed to materialize descriptor table for debug query",
                             exc_info=True)
                return []
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
        if isinstance(table, HSpecPromptTableDesc):
            mean_arr = open_table_array(table.mean)
            comp_arr = open_table_array(table.components)
            try:
                return (
                    np.array(mean_arr, dtype=np.float32, copy=True),
                    np.array(comp_arr, dtype=np.float32, copy=True),
                )
            finally:
                _close_memmap(mean_arr)
                _close_memmap(comp_arr)
        return (table.pca_params.mean.copy(), table.pca_params.components.copy())

    def get_prompt_keys(self, prompt_id: str):
        """Return keys ndarray ``(n_entries, K)`` for prompt, or ``None``."""
        if prompt_id not in self._active:
            return None
        table = self._active[prompt_id]
        if isinstance(table, HSpecPromptTableDesc):
            keys_arr = open_table_array(table.keys)
            try:
                return np.array(keys_arr, copy=True)
            finally:
                _close_memmap(keys_arr)
        return table.get_keys_numpy()

    def get_prompt_table_data(self, prompt_id: str):
        """Return serialisable dict with full table data for proposer cache."""
        if prompt_id not in self._active:
            return None
        table = self._active[prompt_id]
        if isinstance(table, HSpecPromptTableDesc):
            hspec_record_store_metric("table_prefetch_legacy_array_count", 1)
            return materialize_prompt_table(table)
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
        self._table_writer = None
        self._building_version = None
        self._active_version = 0
        self._reset_metrics()
        clear_active_version_manifest(self.table_store_root)

    def exist(self, prompt_id: str) -> bool:
        return prompt_id in self._active

    def get_prompt_ids(self) -> List[str]:
        return list(self._active.keys())

    def num_prompts(self) -> int:
        return len(self._active)

    def total_entries(self) -> int:
        return sum(int(getattr(t, "n_entries", 0)) for t in self._active.values())

    # Double-buffer version management

    def swap(self):
        """Publish building tables as the active version at epoch boundary."""
        if not self._building:
            hspec_record_store_metric("table_swap_empty_count", 1)
            writer = self._table_writer
            if writer is not None:
                try:
                    writer.seal({
                        "step": "phase2_step4_empty_swap",
                        "status": "gc_deletable",
                    })
                except Exception:
                    logger.warning(
                        "Failed to seal empty HSpec table store writer; "
                        "keeping previous active table.",
                        exc_info=True,
                    )
            self._building = {}
            self._table_writer = None
            self._building_version = None
            self._maybe_gc_table_store_versions()
            self._reset_metrics()
            logger.info(
                "HSpec swap skipped empty building set: active_version=%d, "
                "prompts=%d, entries=%d",
                self._active_version,
                len(self._active),
                self.total_entries(),
            )
            return

        writer = self._table_writer
        if writer is None:
            has_descriptor = any(
                isinstance(table, HSpecPromptTableDesc)
                for table in self._building.values()
            )
            if has_descriptor:
                raise RuntimeError(
                    "HSpec descriptor building tables exist without a table "
                    "store writer; refusing to publish an inconsistent active "
                    "version."
                )
            self._active = dict(self._building)
            self._building = {}
            self._active_version += 1
            self._reset_metrics()
            logger.info(
                "HSpec legacy swap: active_version=%d, prompts=%d, entries=%d",
                self._active_version,
                len(self._active),
                self.total_entries(),
            )
            return

        building_version = int(
            self._building_version if self._building_version is not None
            else writer.version
        )
        new_active: Dict[str, HSpecPromptTableDesc] = {}
        for pid, table in self._building.items():
            if not isinstance(table, HSpecPromptTableDesc):
                raise TypeError(
                    "HSpec cannot publish mixed descriptor/legacy building "
                    f"tables; prompt_id={pid!r} type={type(table)!r}"
                )
            if int(table.version) != building_version:
                raise ValueError(
                    "HSpec prompt table descriptor version mismatch: "
                    f"prompt_id={pid!r} desc.version={table.version} "
                    f"building_version={building_version}"
                )
            if int(table.shard_id) != int(self.shard_id):
                raise ValueError(
                    "HSpec prompt table descriptor shard mismatch: "
                    f"prompt_id={pid!r} desc.shard_id={table.shard_id} "
                    f"actor.shard_id={self.shard_id}"
                )
            new_active[str(pid)] = table

        manifest = writer.seal({
            "step": "phase2_step4_swap",
            "active_publish_pending": True,
        })
        prompt_count = len(new_active)
        entry_count = sum(int(desc.n_entries) for desc in new_active.values())
        write_active_version_manifest(
            self.table_store_root,
            active_version=building_version,
            shard_id=self.shard_id,
            version_dir=manifest["version_dir"],
            manifest_path=manifest.get("manifest_path", writer.manifest_path),
            prompt_count=prompt_count,
            entry_count=entry_count,
        )

        self._active = new_active
        self._active_version = building_version
        self._building = {}
        self._table_writer = None
        self._building_version = None
        self._maybe_gc_table_store_versions()
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

    def get_active_table_desc_batch(
        self,
        prompt_ids: List[str],
    ) -> Tuple[int, Dict[str, Optional[HSpecPromptTableDesc]]]:
        """Batch fetch active table descriptors without materializing arrays."""
        result: Dict[str, Optional[HSpecPromptTableDesc]] = {}
        hit_count = 0
        for pid in prompt_ids:
            table = self._active.get(pid)
            if isinstance(table, HSpecPromptTableDesc):
                result[pid] = table
                hit_count += 1
            else:
                result[pid] = None
        hspec_record_store_metric("table_prefetch_descriptor_count", hit_count)
        return (self._active_version, result)

    def get_active_table_data_batch(
        self,
        prompt_ids: List[str],
    ) -> Tuple[int, Dict[str, Optional[Dict[str, Any]]]]:
        """Legacy array prefetch fallback for active tables.

        Descriptor mode must call ``get_active_table_desc_batch()`` to avoid
        Ray object-store table arrays.
        """
        result: Dict[str, Optional[Dict[str, Any]]] = {}
        for pid in prompt_ids:
            if pid not in self._active:
                result[pid] = None
                continue
            table = self._active[pid]
            if isinstance(table, HSpecPromptTableDesc):
                hspec_record_store_metric("table_prefetch_legacy_array_count", 1)
                result[pid] = materialize_prompt_table(table)
                continue
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
            "build_selected_desc_count": self._build_selected_desc_count,
            "build_input_rows": self._build_input_rows,
            "build_selected_rows": self._build_selected_rows,
            "build_loaded_raw_bytes": self._build_loaded_raw_bytes,
            "build_loaded_fp32_bytes": self._build_loaded_fp32_bytes,
            "build_budget_drop_count": self._build_budget_drop_count,
            "build_budget_drop_rows": self._build_budget_drop_rows,
            "build_budget_drop_raw_bytes": self._build_budget_drop_raw_bytes,
            "build_budget_drop_oversize_count": self._build_budget_drop_oversize_count,
            "build_rss_cap_skip_count": self._build_rss_cap_skip_count,
            "build_memory_error_count": self._build_memory_error_count,
            "build_validation_ms": self._build_validation_ms,
            "build_materialize_ms": self._build_materialize_ms,
            "build_pca_ms": self._build_pca_ms,
            "build_table_add_ms": self._build_table_add_ms,
            "build_pca_mean_ms": self._build_pca_mean_ms,
            "build_pca_basis_ms": self._build_pca_basis_ms,
            "build_projection_ms": self._build_projection_ms,
            "build_table_write_ms": self._build_table_write_ms,
            "build_processed_fp32_tile_bytes": self._build_processed_fp32_tile_bytes,
            "build_projection_tile_count": self._build_projection_tile_count,
            "build_pca_method_randomized_count": self._build_pca_method_randomized_count,
            "build_pca_method_covariance_count": self._build_pca_method_covariance_count,
            "build_pca_method_svd_reference_count": self._build_pca_method_svd_reference_count,
            "build_pca_method_fallback_count": self._build_pca_method_fallback_count,
            "build_pca_cov_bytes_max": self._build_pca_cov_bytes_max,
            "build_pca_randomized_rank_max": self._build_pca_randomized_rank_max,
            "build_total_ms": self._build_total_ms,
            "build_rss_peak_bytes": self._build_rss_peak_bytes,
            "build_rss_after_materialize_peak_bytes": self._build_rss_after_materialize_peak_bytes,
            "build_rss_after_pca_peak_bytes": self._build_rss_after_pca_peak_bytes,
            "build_rss_delta_peak_bytes": self._build_rss_delta_peak_bytes,
            "active_version": self._active_version,
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
        self._build_selected_desc_count = 0
        self._build_input_rows = 0
        self._build_selected_rows = 0
        self._build_loaded_raw_bytes = 0
        self._build_loaded_fp32_bytes = 0
        self._build_budget_drop_count = 0
        self._build_budget_drop_rows = 0
        self._build_budget_drop_raw_bytes = 0
        self._build_budget_drop_oversize_count = 0
        self._build_rss_cap_skip_count = 0
        self._build_memory_error_count = 0
        self._build_validation_ms = 0.0
        self._build_materialize_ms = 0.0
        self._build_pca_ms = 0.0
        self._build_table_add_ms = 0.0
        self._build_pca_mean_ms = 0.0
        self._build_pca_basis_ms = 0.0
        self._build_projection_ms = 0.0
        self._build_table_write_ms = 0.0
        self._build_processed_fp32_tile_bytes = 0
        self._build_projection_tile_count = 0
        self._build_pca_method_randomized_count = 0
        self._build_pca_method_covariance_count = 0
        self._build_pca_method_svd_reference_count = 0
        self._build_pca_method_fallback_count = 0
        self._build_pca_cov_bytes_max = 0
        self._build_pca_randomized_rank_max = 0
        self._build_total_ms = 0.0
        self._build_rss_peak_bytes = 0
        self._build_rss_after_materialize_peak_bytes = 0
        self._build_rss_after_pca_peak_bytes = 0
        self._build_rss_delta_peak_bytes = 0
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
            if isinstance(t, HSpecPromptTableDesc):
                mean_arr = open_table_array(t.mean)
                comp_arr = open_table_array(t.components)
                keys_arr = open_table_array(t.keys)
                entry_rollout_idx_arr = open_table_array(t.entry_rollout_idx)
                entry_offset_arr = open_table_array(t.entry_offset)
                token_buffer_arr = open_table_array(t.token_buffer)
                rollout_offset_arr = open_table_array(t.rollout_token_offset)
                rollout_len_arr = open_table_array(t.rollout_token_len)
                try:
                    key_sample = None
                    key_norm_sample = None
                    if t.n_entries > 0 and keys_arr.shape[0] > 0:
                        first_key = np.array(keys_arr[0], dtype=np.float32, copy=True)
                        key_sample = first_key[:8].tolist()
                        key_norm_sample = float(np.linalg.norm(first_key))
                    draft_sample = None
                    if t.n_entries > 0:
                        ridx = int(entry_rollout_idx_arr[0])
                        off = int(entry_offset_arr[0])
                        if 0 <= ridx < int(t.n_rollouts):
                            base = int(rollout_offset_arr[ridx])
                            length = int(rollout_len_arr[ridx])
                            draft_sample = np.array(
                                token_buffer_arr[base + off:base + min(length, off + min(5, t.wnd_size))],
                                dtype=np.int32,
                                copy=True,
                            ).tolist()
                    rollout_lens = np.array(rollout_len_arr, dtype=np.int32,
                                            copy=True).tolist()
                    info[label] = {
                        "storage": "descriptor",
                        "version": int(t.version),
                        "table_file": str(t.table_file),
                        "n_entries": int(t.n_entries),
                        "pca_method": str(t.pca_method),
                        "pca_mean_shape": list(mean_arr.shape),
                        "pca_components_shape": list(comp_arr.shape),
                        "pca_mean_norm": float(np.linalg.norm(mean_arr)),
                        "keys_shape": list(keys_arr.shape),
                        "key_sample_first8": key_sample,
                        "key_norm_sample": key_norm_sample,
                        "rollout_seqs_count": int(t.n_rollouts),
                        "rollout_seq_lens": rollout_lens,
                        "wnd_size": int(t.wnd_size),
                        "max_wnd": int(t.max_wnd),
                        "min_wnd": int(t.min_wnd),
                        "draft_sample_entry0": draft_sample,
                    }
                finally:
                    for arr in (
                        mean_arr,
                        comp_arr,
                        keys_arr,
                        entry_rollout_idx_arr,
                        entry_offset_arr,
                        token_buffer_arr,
                        rollout_offset_arr,
                        rollout_len_arr,
                    ):
                        _close_memmap(arr)
                continue
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
                "storage": "legacy_arrays",
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
            expected_build_max_prompt_rows=get_hspec_build_max_prompt_rows(),
            expected_build_max_prompt_raw_bytes=get_hspec_build_max_prompt_raw_bytes(),
            expected_build_max_prompt_descs=get_hspec_build_max_prompt_descs(),
            expected_build_max_rss_mb=get_hspec_build_max_rss_mb(),
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
        """Return max active table version across actors."""
        if not self.groups:
            return 0
        versions = ray.get([group.get_active_version.remote() for group in self.groups])
        return max((int(version) for version in versions), default=0)

    def prefetch_batch(self, prompt_ids: List[str]) -> Tuple[int, Dict[str, Optional[Any]]]:
        """Batch-fetch active table payloads for proposer cache.

        One Ray call per partition → minimal overhead.  Returns
        ``(latest_version, {prompt_id: payload | None})``.  The default
        payload is ``HSpecPromptTableDesc``; ``legacy_arrays`` mode returns
        the old materialized table-data dict.
        """
        from collections import defaultdict

        mode = get_hspec_table_prefetch_mode()
        partition_prompts: Dict[int, List[str]] = defaultdict(list)
        for pid in prompt_ids:
            partition_prompts[self._get_partition_id(pid)].append(pid)

        futures: Dict[int, ray.ObjectRef] = {}
        for part, pids in partition_prompts.items():
            if part < len(self.groups):
                if mode == "legacy_arrays":
                    futures[part] = self.groups[part].get_active_table_data_batch.remote(pids)
                else:
                    futures[part] = self.groups[part].get_active_table_desc_batch.remote(pids)

        result: Dict[str, Optional[Any]] = {}
        latest_version = -1
        for part, future in futures.items():
            version, batch_data = ray.get(future)
            latest_version = max(latest_version, version)
            result.update(batch_data)
        return latest_version, result

    def prefetch_batch_async(self, prompt_ids: List[str]) -> List[Tuple[ray.ObjectRef, List[str]]]:
        """Fire async prefetch – **non-blocking**, returns immediately.

        Returns ``[(ObjectRef, [prompt_ids]), ...]`` where each
        ObjectRef resolves to ``(active_version, {pid: payload | None})``.
        Descriptor mode returns ``HSpecPromptTableDesc`` payloads; explicit
        ``legacy_arrays`` mode returns the old materialized table dicts.
        """
        from collections import defaultdict

        mode = get_hspec_table_prefetch_mode()
        partition_prompts: Dict[int, List[str]] = defaultdict(list)
        for pid in prompt_ids:
            partition_prompts[self._get_partition_id(pid)].append(pid)

        result: List[Tuple[ray.ObjectRef, List[str]]] = []
        for part, pids in partition_prompts.items():
            if part < len(self.groups):
                if mode == "legacy_arrays":
                    future = self.groups[part].get_active_table_data_batch.remote(pids)
                else:
                    future = self.groups[part].get_active_table_desc_batch.remote(pids)
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
        max_agg_keys = {
            "build_rss_peak_bytes",
            "build_rss_after_materialize_peak_bytes",
            "build_rss_after_pca_peak_bytes",
            "build_rss_delta_peak_bytes",
            "build_pca_cov_bytes_max",
            "build_pca_randomized_rank_max",
            "active_version",
        }
        for metrics in metrics_list:
            for key, value in metrics.items():
                if key in max_agg_keys:
                    agg[key] = max(agg.get(key, 0.0), float(value))
                else:
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
            "hspec/build_selected_desc_count": agg.get("build_selected_desc_count", 0),
            "hspec/build_input_rows": agg.get("build_input_rows", 0),
            "hspec/build_selected_rows": agg.get("build_selected_rows", 0),
            "hspec/build_loaded_raw_bytes": agg.get("build_loaded_raw_bytes", 0),
            "hspec/build_loaded_fp32_bytes": agg.get("build_loaded_fp32_bytes", 0),
            "hspec/build_loaded_raw_mb": agg.get("build_loaded_raw_bytes", 0) / (1024 * 1024),
            "hspec/build_loaded_fp32_mb": agg.get("build_loaded_fp32_bytes", 0) / (1024 * 1024),
            "hspec/build_budget_drop_count": agg.get("build_budget_drop_count", 0),
            "hspec/build_budget_drop_rows": agg.get("build_budget_drop_rows", 0),
            "hspec/build_budget_drop_raw_bytes": agg.get("build_budget_drop_raw_bytes", 0),
            "hspec/build_budget_drop_raw_mb": agg.get("build_budget_drop_raw_bytes", 0) / (1024 * 1024),
            "hspec/build_budget_drop_oversize_count": agg.get("build_budget_drop_oversize_count", 0),
            "hspec/build_rss_cap_skip_count": agg.get("build_rss_cap_skip_count", 0),
            "hspec/build_memory_error_count": agg.get("build_memory_error_count", 0),
            "hspec/build_validation_ms": agg.get("build_validation_ms", 0),
            "hspec/build_materialize_ms": agg.get("build_materialize_ms", 0),
            "hspec/build_pca_ms": agg.get("build_pca_ms", 0),
            "hspec/build_table_add_ms": agg.get("build_table_add_ms", 0),
            "hspec/build_pca_mean_ms": agg.get("build_pca_mean_ms", 0),
            "hspec/build_pca_basis_ms": agg.get("build_pca_basis_ms", 0),
            "hspec/build_projection_ms": agg.get("build_projection_ms", 0),
            "hspec/build_table_write_ms": agg.get("build_table_write_ms", 0),
            "hspec/build_processed_fp32_tile_bytes": agg.get(
                "build_processed_fp32_tile_bytes", 0),
            "hspec/build_processed_fp32_tile_mb": agg.get(
                "build_processed_fp32_tile_bytes", 0) / (1024 * 1024),
            "hspec/build_projection_tile_count": agg.get("build_projection_tile_count", 0),
            "hspec/build_pca_method_randomized_count": agg.get(
                "build_pca_method_randomized_count", 0),
            "hspec/build_pca_method_covariance_count": agg.get(
                "build_pca_method_covariance_count", 0),
            "hspec/build_pca_method_svd_reference_count": agg.get(
                "build_pca_method_svd_reference_count", 0),
            "hspec/build_pca_method_fallback_count": agg.get(
                "build_pca_method_fallback_count", 0),
            "hspec/build_pca_cov_bytes": agg.get("build_pca_cov_bytes_max", 0),
            "hspec/build_pca_randomized_rank": agg.get(
                "build_pca_randomized_rank_max", 0),
            "hspec/build_total_ms": agg.get("build_total_ms", 0),
            "hspec/build_actor_rss_peak_mb": agg.get("build_rss_peak_bytes", 0) / (1024 * 1024),
            "hspec/build_actor_rss_after_materialize_mb_max": (
                agg.get("build_rss_after_materialize_peak_bytes", 0) / (1024 * 1024)
            ),
            "hspec/build_actor_rss_after_pca_mb_max": (
                agg.get("build_rss_after_pca_peak_bytes", 0) / (1024 * 1024)
            ),
            "hspec/build_actor_rss_delta_mb_max": agg.get("build_rss_delta_peak_bytes", 0) / (1024 * 1024),
            "hspec/active_version": agg.get("active_version", 0),
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
        result["hspec/table_store_descriptor_count"] = float(
            store_metrics.get("table_store_descriptor_count", 0))
        result["hspec/table_store_array_descriptor_count"] = float(
            store_metrics.get("table_store_array_descriptor_count", 0))
        result["hspec/table_store_reserved_bytes"] = float(
            store_metrics.get("table_store_reserved_bytes", 0))
        result["hspec/table_store_committed_prompts"] = float(
            store_metrics.get("table_store_committed_prompts", 0))
        result["hspec/table_store_manifest_write_error"] = float(
            store_metrics.get("table_store_manifest_write_error", 0))
        result["hspec/table_store_active_manifest_write_error"] = float(
            store_metrics.get("table_store_active_manifest_write_error", 0))
        result["hspec/table_store_fsync_count"] = float(
            store_metrics.get("table_store_fsync_count", 0))
        result["hspec/table_store_reader_load_error"] = float(
            store_metrics.get("table_store_reader_load_error", 0))
        result["hspec/table_store_materialize_count"] = float(
            store_metrics.get("table_store_materialize_count", 0))
        result["hspec/table_store_bytes_written"] = float(
            store_metrics.get("table_store_bytes_written", 0))
        result["hspec/table_store_prompt_count"] = float(
            store_metrics.get("table_store_prompt_count", 0))
        result["hspec/table_store_entry_count"] = float(
            store_metrics.get("table_store_entry_count", 0))
        result["hspec/table_store_version"] = float(
            store_metrics.get("table_store_version", 0))
        result["hspec/table_swap_empty_count"] = float(
            store_metrics.get("table_swap_empty_count", 0))
        result["hspec/table_store_gc_scanned_versions"] = float(
            store_metrics.get("table_store_gc_scanned_versions", 0))
        result["hspec/table_store_gc_retained_versions"] = float(
            store_metrics.get("table_store_gc_retained_versions", 0))
        result["hspec/table_store_gc_deleted_versions"] = float(
            store_metrics.get("table_store_gc_deleted_versions", 0))
        result["hspec/table_store_gc_delete_error"] = float(
            store_metrics.get("table_store_gc_delete_error", 0))
        result["hspec/table_store_active_manifest_clear_error"] = float(
            store_metrics.get("table_store_active_manifest_clear_error", 0))
        result["hspec/table_prefetch_descriptor_count"] = float(
            store_metrics.get("table_prefetch_descriptor_count", 0))
        result["hspec/table_prefetch_legacy_array_count"] = float(
            store_metrics.get("table_prefetch_legacy_array_count", 0))

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
            expected_build_max_prompt_rows=get_hspec_build_max_prompt_rows(),
            expected_build_max_prompt_raw_bytes=get_hspec_build_max_prompt_raw_bytes(),
            expected_build_max_prompt_descs=get_hspec_build_max_prompt_descs(),
            expected_build_max_rss_mb=get_hspec_build_max_rss_mb(),
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
        expected_build_max_prompt_rows=get_hspec_build_max_prompt_rows(),
        expected_build_max_prompt_raw_bytes=get_hspec_build_max_prompt_raw_bytes(),
        expected_build_max_prompt_descs=get_hspec_build_max_prompt_descs(),
        expected_build_max_rss_mb=get_hspec_build_max_rss_mb(),
    )
    hspec_record_store_metric("topology_actor_count", len(topologies))
    for item in topologies:
        logger.info(
            "HSpec build shard actor %d registered successfully: node_id=%s ray_node_id=%s "
            "build_actor_num_cpus=%s build_blas_threads=%s build_max_prompt_rows=%s "
            "build_max_prompt_raw_bytes=%s build_max_prompt_descs=%s build_max_rss_mb=%s",
            int(item["shard_id"]),
            item.get("logical_node_id"),
            item.get("ray_node_id"),
            item.get("build_actor_num_cpus"),
            item.get("build_blas_threads"),
            item.get("build_max_prompt_rows"),
            item.get("build_max_prompt_raw_bytes"),
            item.get("build_max_prompt_descs"),
            item.get("build_max_rss_mb"),
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
            expected_build_max_prompt_rows=get_hspec_build_max_prompt_rows(),
            expected_build_max_prompt_raw_bytes=get_hspec_build_max_prompt_raw_bytes(),
            expected_build_max_prompt_descs=get_hspec_build_max_prompt_descs(),
            expected_build_max_rss_mb=get_hspec_build_max_rss_mb(),
        )

    return GlobalHSpecTableGroup(
        similarity_threshold=similarity_threshold,
        max_entries_per_prompt=max_entries_per_prompt,
        n_components=n_components,
    )
