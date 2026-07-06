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
HSpec proposer: on-device hidden-state similarity speculative decoding.
"""

import logging
import os
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
try:
    import numba
    from numba import njit, prange
    from numba.typed import List as NumbaList
    _HSPEC_NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    numba = None
    njit = None
    prange = range
    NumbaList = None
    _HSPEC_NUMBA_AVAILABLE = False

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.spec_decode.hspec_table import GlobalHSpecTableGroup, get_hspec_tables
from vllm_ascend.spec_decode.hspec_table_store import (
    HSpecPromptTableDesc,
    estimate_prompt_table_desc_nbytes,
    open_array,
)
from vllm_ascend.spec_decode.hspec_utils import (
    hspec_profile_context_enabled,
    hspec_record_function,
    prompt_id_from_token_ids,
)
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("HSPEC_LOG_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "WARN")))

# Enable verbose HSpec debug logging when HSPEC_DEBUG is set.
HSPEC_DEBUG = os.getenv("HSPEC_DEBUG", "0") != "0"
# Per-step debug: only log one request per step to limit log volume.
HSPEC_DEBUG_REQ_IDX = int(os.getenv("HSPEC_DEBUG_REQ_IDX", "3"))

# Per-token generation breakdown timing (very verbose; use only for debugging).
# Enable with HSPEC_GEN=1. By default, trace the same request index as
# HSPEC_DEBUG_REQ_IDX; can override via HSPEC_GEN_REQ_IDX.
HSPEC_GEN = os.getenv("HSPEC_GEN", "0") != "0"
HSPEC_GEN_REQ_IDX = int(os.getenv("HSPEC_GEN_REQ_IDX", os.getenv("HSPEC_DEBUG_REQ_IDX", "3")))
HSPEC_GEN_MAX_CALLS = int(os.getenv("HSPEC_GEN_MAX_CALLS", "0"))
HSPEC_ADVAN_NGRAM = int(os.getenv("HSPEC_ADVAN_NGRAM", "3"))


def _get_env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(value), int(minimum))


def _tensor_nbytes(tensor: Any) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


if _HSPEC_NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _hspec_fill_batched_components_keys_numba(
        components_list,
        keys_list,
        components_t_batch_cpu,
        keys_batch_cpu,
        key_lengths_cpu,
    ):
        for row in prange(len(components_list)):
            comp_src = components_list[row]
            key_src = keys_list[row]
            hidden_dim = comp_src.shape[0]
            k_i = comp_src.shape[1]
            m_i = key_src.shape[0]

            for d in range(hidden_dim):
                for k in range(k_i):
                    components_t_batch_cpu[row, d, k] = comp_src[d, k]

            for m in range(m_i):
                for k in range(k_i):
                    keys_batch_cpu[row, m, k] = key_src[m, k]

            key_lengths_cpu[row] = m_i


def _hspec_make_numba_array_list(arrays: List[np.ndarray]):
    if not _HSPEC_NUMBA_AVAILABLE or NumbaList is None:
        return None
    typed_list = NumbaList()
    for arr in arrays:
        typed_list.append(arr)
    return typed_list


def _now_ns() -> int:
    # perf_counter_ns is monotonic and high resolution; safe for timings.
    return time.perf_counter_ns()


def _ns_to_ms(ns: int) -> float:
    return float(ns) / 1_000_000.0


def _close_hspec_memmap(array: np.ndarray | None) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


# Worker-local cached prompt table (on-device tensors + CPU refs)

class _CachedPromptTable:
    """Per-prompt cached table data with compact CPU refs.

    Per-prompt NPU tensors and prefix matrices are optional debug/A-B caches.
    The production path uses CPU refs plus a prebuilt batch table cache.
    """

    __slots__ = (
        "mean_cpu",
        "components_t_cpu",
        "keys_cpu",
        "mean",
        "components",
        "keys",
        "token_buffer",
        "rollout_token_offset",
        "rollout_token_len",
        "rollout_seqs",
        "entry_rollout_idx",
        "entry_offset",
        "draft_prefix_tokens",
        "draft_prefix_lens",
        "rollout_entry_starts",
        "rollout_entry_lens",
        "n_entries",
        "wnd_size",
        "max_wnd",
        "min_wnd",
        "entry_bias",
        "entry_hits",
        "entry_blend_horizon",
        "max_entry_bias",
    )

    def __init__(
        self,
        mean_cpu: np.ndarray,
        components_t_cpu: np.ndarray,
        keys_cpu: np.ndarray,
        mean: Optional[torch.Tensor],
        components: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        token_buffer: np.ndarray,
        rollout_token_offset: np.ndarray,
        rollout_token_len: np.ndarray,
        rollout_seqs: Optional[list],
        entry_rollout_idx: np.ndarray,
        entry_offset: np.ndarray,
        draft_prefix_tokens: Optional[np.ndarray],
        draft_prefix_lens: Optional[np.ndarray],
        rollout_entry_starts: np.ndarray,
        rollout_entry_lens: np.ndarray,
        n_entries: int,
        wnd_size: int = 8,
        max_wnd: int = 28,
        min_wnd: int = 2,
        entry_bias: np.ndarray | None = None,
        entry_hits: np.ndarray | None = None,
        entry_blend_horizon: int = 4,
        max_entry_bias: int = 8,
    ):
        self.mean_cpu = mean_cpu                      # (D,) float32, CPU
        self.components_t_cpu = components_t_cpu      # (D,K) float32, CPU
        self.keys_cpu = keys_cpu                      # (M,K) table dtype or float32, CPU
        self.mean = mean                              # optional (D,) device
        self.components = components                  # optional (K,D) device
        self.keys = keys                              # optional (M,K) device
        self.token_buffer = token_buffer              # flat int32 token store, CPU
        self.rollout_token_offset = rollout_token_offset  # (R,) int64, CPU
        self.rollout_token_len = rollout_token_len        # (R,) int32, CPU
        self.rollout_seqs = rollout_seqs              # optional legacy/debug CPU views
        self.entry_rollout_idx = entry_rollout_idx    # (M,) int32, CPU
        self.entry_offset = entry_offset              # (M,) int32, CPU
        self.draft_prefix_tokens = draft_prefix_tokens  # optional (M, W) int32
        self.draft_prefix_lens = draft_prefix_lens      # optional (M,) int32
        self.rollout_entry_starts = rollout_entry_starts  # (R,) int32, CPU
        self.rollout_entry_lens = rollout_entry_lens      # (R,) int32, CPU
        self.n_entries = n_entries
        self.wnd_size = wnd_size
        self.max_wnd = max_wnd
        self.min_wnd = min_wnd
        self.entry_bias = (
            entry_bias
            if entry_bias is not None
            else np.zeros((n_entries,), dtype=np.int8)
        )
        self.entry_hits = (
            entry_hits
            if entry_hits is not None
            else np.zeros((n_entries,), dtype=np.uint16)
        )
        self.entry_blend_horizon = max(int(entry_blend_horizon), 1)
        self.max_entry_bias = max(int(max_entry_bias), 0)

    def get_draft_tokens(self, entry_idx: int, max_tokens: int) -> List[int]:
        """O(1) draft lookup, using lazy token refs by default."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return []
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            return []
        if self.draft_prefix_tokens is not None and self.draft_prefix_lens is not None:
            take = min(max_tokens, int(self.draft_prefix_lens[entry_idx]))
            if take <= 0:
                return []
            return self.draft_prefix_tokens[entry_idx, :take].tolist()

        ridx = int(self.entry_rollout_idx[entry_idx])
        if ridx < 0 or ridx >= int(self.rollout_token_offset.shape[0]):
            return []
        off = int(self.entry_offset[entry_idx])
        base = int(self.rollout_token_offset[ridx])
        length = int(self.rollout_token_len[ridx])
        take = min(max_tokens, max(0, length - off))
        if take <= 0:
            return []
        return self.token_buffer[base + off:base + off + take].tolist()

    def get_rollout_tokens(self, rollout_idx: int) -> np.ndarray:
        ridx = int(rollout_idx)
        if ridx < 0 or ridx >= int(self.rollout_token_offset.shape[0]):
            return np.empty((0,), dtype=np.int32)
        base = int(self.rollout_token_offset[ridx])
        length = int(self.rollout_token_len[ridx])
        return self.token_buffer[base:base + length]

    def update_window(
        self,
        accept_length: int,
        drafted_len: int | None = None,
    ) -> int:
        """Congestion-control style adaptive window.

        The update is performed after verification and uses the actual draft
        length that was proposed for the request when available.
        """
        threshold = int(drafted_len) if drafted_len is not None else int(self.wnd_size)
        if threshold <= 0:
            threshold = int(self.wnd_size)
        if accept_length >= threshold:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)
        elif accept_length < 1:
            self.wnd_size = max(self.wnd_size // 2, self.min_wnd)
        return int(self.wnd_size)

    def get_entry_state(self, entry_idx: int) -> tuple[int, int]:
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return (0, 0)
        return (int(self.entry_bias[entry_idx]), int(self.entry_hits[entry_idx]))

    def get_effective_window(self, entry_idx: int) -> int:
        """Prompt baseline window plus blended entry-level bias."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return int(self.wnd_size)
        bias = int(self.entry_bias[entry_idx])
        hits = int(self.entry_hits[entry_idx])
        if bias == 0 or hits <= 0:
            return int(self.wnd_size)
        scale = min(hits, self.entry_blend_horizon)
        blended_bias = int((bias * scale) / self.entry_blend_horizon)
        eff_wnd = int(self.wnd_size) + blended_bias
        if eff_wnd < self.min_wnd:
            eff_wnd = int(self.min_wnd)
        elif eff_wnd > self.max_wnd:
            eff_wnd = int(self.max_wnd)
        return int(eff_wnd)

    def update_entry_bias_after_verification(
        self,
        entry_idx: int,
        accept_length: int,
        drafted_len: int,
    ) -> tuple[int, int]:
        """Update entry-local bias after true verification feedback."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return (0, 0)
        current_hits = int(self.entry_hits[entry_idx])
        if current_hits < np.iinfo(np.uint16).max:
            self.entry_hits[entry_idx] = current_hits + 1
        current_bias = int(self.entry_bias[entry_idx])
        delta = 0
        drafted_len = int(drafted_len)
        accept_length = int(accept_length)
        if drafted_len > 0:
            if accept_length >= drafted_len:
                delta = 2
            elif accept_length < 1:
                delta = -2
            elif accept_length + 1 < drafted_len:
                delta = -1
        if delta != 0 and self.max_entry_bias > 0:
            current_bias += delta
            if current_bias > self.max_entry_bias:
                current_bias = self.max_entry_bias
            elif current_bias < -self.max_entry_bias:
                current_bias = -self.max_entry_bias
            self.entry_bias[entry_idx] = current_bias
        return (int(self.entry_bias[entry_idx]), int(self.entry_hits[entry_idx]))


class _BatchedPromptTableCache:
    """Batch-aligned padded tensors for fast batched matching."""

    __slots__ = (
        "req_ids",
        "prompt_ids",
        "cache_generation",
        "batch_indices",
        "batch_idx_to_row",
        "cached_tables",
        "mean_batch",
        "components_t_batch",
        "keys_batch",
        "key_lengths",
        "invalid_key_mask",
        "ready_event",
        "estimated_npu_bytes",
    )

    def __init__(
        self,
        req_ids: tuple[str, ...],
        prompt_ids: tuple[str, ...],
        cache_generation: int,
        batch_indices: List[int],
        batch_idx_to_row: Dict[int, int],
        cached_tables: List["_CachedPromptTable"],
        mean_batch: torch.Tensor,
        components_t_batch: torch.Tensor,
        keys_batch: torch.Tensor,
        key_lengths: torch.Tensor,
        invalid_key_mask: torch.Tensor,
        ready_event: Optional[Any] = None,
        estimated_npu_bytes: int = 0,
    ):
        self.req_ids = req_ids
        self.prompt_ids = prompt_ids
        self.cache_generation = cache_generation
        self.batch_indices = batch_indices
        self.batch_idx_to_row = batch_idx_to_row
        self.cached_tables = cached_tables
        self.mean_batch = mean_batch
        self.components_t_batch = components_t_batch
        self.keys_batch = keys_batch
        self.key_lengths = key_lengths
        self.invalid_key_mask = invalid_key_mask
        self.ready_event = ready_event
        self.estimated_npu_bytes = int(estimated_npu_bytes)


# Helper function for detokenization (with internal debug when HSPEC_DEBUG=1)
def _detokenize_safe(tokenizer, token_ids) -> str:
    """Safely detokenize token_ids to text, returns '<decode_error>' on failure.

    token_ids may be list, numpy array, or contain numpy.int64/torch.Tensor
    elements; we normalize to list of Python ints so tokenizer.decode() works.
    """
    # if HSPEC_DEBUG:
    #     logger.info(
    #         "HSPEC DEBUG _detokenize_safe: tokenizer=%s, token_ids type=%s, len=%s",
    #         type(tokenizer).__name__ if tokenizer is not None else "None",
    #         type(token_ids).__name__ if token_ids is not None else "None",
    #         len(token_ids) if token_ids is not None else 0,
    #     )
    try:
        if tokenizer is None:
            if HSPEC_DEBUG:
                logger.info("HSPEC DEBUG _detokenize_safe: early return <no_tokenizer>")
            return "<no_tokenizer>"
        if token_ids is None:
            return "<empty>"
        # Flatten and coerce to Python ints (handles list, ndarray, and
        # elements that are numpy.int64 or torch.Tensor scalars)
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        ids = []
        for x in token_ids:
            if hasattr(x, "item"):
                ids.append(int(x.item()))
            else:
                ids.append(int(x))
        if not ids:
            return "<empty>"
        # if HSPEC_DEBUG:
        #     logger.info("HSPEC DEBUG _detokenize_safe: decode ok len(text)=%d", len(text))
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception as exc:
        n = len(token_ids) if token_ids is not None else 0
        return f"<decode_error: {n} tokens, {type(exc).__name__}: {exc}>"


# Cache for tokenizer loaded from name/path (model_config.tokenizer is str in vLLM)
_tokenizer_cache: Dict[str, Any] = {}


def _get_tokenizer_safe(runner) -> Optional[Any]:
    """Safely get tokenizer from runner/model/config.

    vLLM's model_config.tokenizer is often a string (model name/path), not an
    instance. When we get a str, we load via AutoTokenizer.from_pretrained(...)
    and cache by that path so decode works in debug.
    """
    global _tokenizer_cache
    if HSPEC_DEBUG:
        logger.info(
            "HSPEC DEBUG _get_tokenizer_safe: runner=%s, has model=%s, has vllm_config=%s, has tokenizer=%s",
            type(runner).__name__,
            hasattr(runner, "model"),
            hasattr(runner, "vllm_config"),
            hasattr(runner, "tokenizer"),
        )
    try:
        if hasattr(runner, "model") and runner.model is not None and hasattr(runner.model, "tokenizer"):
            tok = runner.model.tokenizer
            if not isinstance(tok, str):
                if HSPEC_DEBUG:
                        logger.info("HSPEC DEBUG _get_tokenizer_safe: got tokenizer from runner.model.tokenizer type=%s", type(tok).__name__)
                return tok
        if hasattr(runner, "vllm_config") and runner.vllm_config is not None:
            model_config = getattr(runner.vllm_config, "model_config", None)
            if model_config is not None and hasattr(model_config, "tokenizer"):
                tok = model_config.tokenizer
                if isinstance(tok, str):
                    if tok not in _tokenizer_cache:
                        try:
                            from transformers import AutoTokenizer
                            _tokenizer_cache[tok] = AutoTokenizer.from_pretrained(
                                tok, trust_remote_code=True)
                        except Exception:
                            return None
                    tok = _tokenizer_cache[tok]
                if HSPEC_DEBUG:
                    logger.info("HSPEC DEBUG _get_tokenizer_safe: got tokenizer from vllm_config.model_config.tokenizer type=%s", type(tok).__name__)
                return tok
        if hasattr(runner, "tokenizer"):
            tok = runner.tokenizer
            if not isinstance(tok, str):
                return tok
        return None
    except Exception:
        return None

# Main proposer

class HSpecProposer(Proposer):
    """HSpec proposer with worker-local cache and on-device query."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner,
    ):
        self.name = SpecDcodeType.HSPEC
        self.device = device
        self.runner = runner

        spec_config = vllm_config.speculative_config
        self.max_draft_tokens: int = spec_config.num_speculative_tokens
        self.similarity_threshold: float = getattr(spec_config, "hspec_similarity_threshold", 0.9)
        self.min_match_len: int = getattr(spec_config, "hspec_min_match_len", 1)
        self.n_components: int = getattr(spec_config, "hspec_n_components", 64)
        self.max_entries_per_prompt: int = getattr(spec_config, "hspec_max_entries_per_prompt", 10_000)

        self.hspec_tables: GlobalHSpecTableGroup = get_hspec_tables(
            similarity_threshold=self.similarity_threshold,
            n_components=self.n_components,
            max_entries_per_prompt=self.max_entries_per_prompt,
        )

        self._cache: OrderedDict[str, _CachedPromptTable] = OrderedDict()
        self._not_in_table: Set[str] = set()
        self._cache_version: int = -1
        self._max_cache_size: int = _get_env_int(
            "HSPEC_PROPOSER_CACHE_MAX_PROMPTS", 512, 0)
        self._cache_max_cpu_bytes: int = _get_env_int(
            "HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES", 0, 0)
        self._cache_max_npu_bytes: int = _get_env_int(
            "HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES", 0, 0)
        self._cache_max_entries: int = _get_env_int(
            "HSPEC_PROPOSER_CACHE_MAX_ENTRIES", 0, 0)
        self._max_ready_prefetch_materialize: int = _get_env_int(
            "HSPEC_MAX_READY_PREFETCH_MATERIALIZE", 0, 0)
        self._max_ready_prefetch_bytes: int = _get_env_int(
            "HSPEC_MAX_READY_PREFETCH_BYTES", 0, 0)
        self._batch_cache_prebuild: bool = (
            os.environ.get("HSPEC_PROPOSER_BATCH_CACHE_PREBUILD", "1") != "0"
        )
        self._allow_hot_batch_cache_build: bool = (
            os.environ.get("HSPEC_ALLOW_HOT_BATCH_CACHE_BUILD", "0") != "0"
        )
        self._batch_cache_max_npu_bytes: int = _get_env_int(
            "HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES", 0, 0)
        self._batch_cache_copy_stream: Optional[Any] = None
        self._batch_cache_copy_stream_disabled: bool = False
        self._prefix_cache_enabled: bool = (
            os.environ.get("HSPEC_PROPOSER_PREFIX_CACHE", "0") != "0"
        )
        self._store_per_prompt_npu: bool = (
            os.environ.get("HSPEC_PROPOSER_STORE_PER_PROMPT_NPU", "0") != "0"
        )
        self._keys_cpu_dtype_mode: str = os.environ.get(
            "HSPEC_PROPOSER_KEYS_CPU_DTYPE", "table").strip().lower()
        if self._keys_cpu_dtype_mode not in {"table", "float32"}:
            logger.warning(
                "Invalid HSPEC_PROPOSER_KEYS_CPU_DTYPE=%r; fallback to table",
                self._keys_cpu_dtype_mode,
            )
            self._keys_cpu_dtype_mode = "table"
        self._keys_device_dtype: torch.dtype = (
            torch.float16
            if os.environ.get(
                "HSPEC_PROPOSER_KEYS_DEVICE_DTYPE", "float32").strip().lower()
            in {"float16", "fp16", "half"}
            else torch.float32
        )
        self._cache_cpu_bytes: int = 0
        self._cache_npu_bytes: int = 0
        self._cache_entries: int = 0
        self._cache_prompt_bytes: Dict[str, tuple[int, int, int]] = {}
        self._cache_generation: int = 0
        self._hot_path_strict: bool = (
            os.environ.get("HSPEC_PROPOSER_HOT_PATH_STRICT", "1") != "0"
        )
        self._in_generate_token_ids: bool = False

        # Async prefetch state (never blocking in hot loop)
        # Each entry: (ray.ObjectRef, [prompt_ids]) where the future
        # resolves to (version, {pid: table_data | None}).
        self._pending_fetches: List[tuple] = []
        self._pending_pids: Set[str] = set()
        # Worker-local prompt baseline window priors. These survive cache
        # invalidation / epoch swap and are re-applied when the next version of
        # the same prompt is prefetched into the local cache.
        self._prompt_wnd_priors: Dict[str, int] = {}
        self._default_wnd_size: int = 8
        self._entry_blend_horizon: int = max(
            int(os.environ.get("HSPEC_ENTRY_BLEND_HORIZON", "4")),
            1,
        )
        self._entry_bias_cap: int = max(
            int(os.environ.get("HSPEC_ENTRY_BIAS_CAP", "8")),
            0,
        )
        # Cheap local-window cap driven by entry-position mismatch.
        self._abs_delta_cap_enabled: bool = (
            os.environ.get("HSPEC_ABS_DELTA_CAP", "1") != "0"
        )
        self._abs_delta_safe_threshold: int = max(
            int(os.environ.get("HSPEC_ABS_DELTA_SAFE_THRESHOLD", "2")),
            0,
        )
        self._abs_delta_mid_threshold: int = max(
            int(os.environ.get("HSPEC_ABS_DELTA_MID_THRESHOLD", "64")),
            self._abs_delta_safe_threshold,
        )
        self._abs_delta_mid_cap: int = max(
            int(os.environ.get("HSPEC_ABS_DELTA_MID_CAP", "8")),
            1,
        )
        self._abs_delta_far_cap: int = max(
            int(os.environ.get("HSPEC_ABS_DELTA_FAR_CAP", "4")),
            1,
        )

        # Accept-length tracking (adaptive window control)
        self._accept_lengths: Dict[str, int] = {}
        # req_id -> matched entry metadata for the *next* verification step.
        self._pending_verify_meta: Dict[str, Dict[str, Any]] = {}
        # req_id -> stable prompt_id cache to avoid repeated hashing of the
        # same prompt token ids across decode steps for a live request.
        self._req_prompt_ids: Dict[str, str] = {}
        # Batch-aligned prompt-id cache. prefetch_for_batch() runs before every
        # forward pass. Phase 3 Step 4 also uses this hook to prebuild the
        # padded NPU batch table cache before generate_token_ids().
        self._cached_batch_req_ids: tuple[str, ...] = ()
        self._cached_batch_prompt_ids: List[str] = []
        self._batched_table_cache: Optional[_BatchedPromptTableCache] = None
        self._use_numba_rebuild = (
            _HSPEC_NUMBA_AVAILABLE
            and os.environ.get("HSPEC_DISABLE_NUMBA_REBUILD", "0") == "0"
        )
        self._numba_rebuild_min_rows = int(os.environ.get("HSPEC_NUMBA_REBUILD_MIN_ROWS", "4"))
        self._numba_rebuild_min_elems = int(os.environ.get("HSPEC_NUMBA_REBUILD_MIN_ELEMS", "262144"))

        # Lightweight local metrics (for functional + perf validation).
        # These live in the vLLM worker process; we never RPC in the hot loop.
        self._stat_calls = 0
        self._stat_queries = 0
        self._stat_hits = 0
        self._stat_total_draft_len = 0
        self._stat_prefetch_fired = 0
        self._stat_prefetch_ready = 0
        self._stat_accept_sum = 0
        self._stat_accept_count = 0
        self._proposer_metric_deltas = defaultdict(float)
        self._proposer_metric_gauges: Dict[str, float] = {}
        self._last_log_t = time.time()
        self._log_every_calls = int(os.environ.get("HSPEC_LOG_EVERY_CALLS", "200"))
        self._log_every_s = float(os.environ.get("HSPEC_LOG_EVERY_S", "10"))

        # Low-frequency metrics reporting to table actors (so trainer-side
        # hspec_tables.compute_metrics() reflects worker-local online queries).
        self._report_every_calls = int(os.environ.get("HSPEC_REPORT_EVERY_CALLS", "200"))
        self._last_report_calls = 0
        self._reported_queries = 0
        self._reported_hits = 0
        self._reported_total_draft_len = 0
        self._prompt_access_pending = defaultdict(lambda: {"query": 0, "hit": 0})
        self._prompt_access_report_every_calls = max(
            int(os.environ.get("HSPEC_TABLE_ACCESS_REPORT_INTERVAL_STEPS", "16")),
            1,
        )
        self._last_prompt_access_report_calls = 0

        # Entry-position study buffers. These are flushed asynchronously to the
        # global HSpec table actors at low frequency.
        self._entry_pending_match_count = 0
        self._entry_pending_delta_sum = 0
        self._entry_pending_abs_delta_sum = 0
        self._entry_pending_verify_count = 0
        self._entry_pending_accept_count = 0
        self._entry_pending_accept_len_sum = 0
        self._entry_pending_abs_delta_verify = defaultdict(int)
        self._entry_pending_abs_delta_accept = defaultdict(int)
        self._entry_pending_abs_delta_accept_len_sum = defaultdict(int)

        logger.info(
            "HSpec proposer initialised: threshold=%.3f, max_draft=%d, cache_cap=%d, "
            "fully_batched_match=1, numba_rebuild=%s, entry_blend_horizon=%d, entry_bias_cap=%d, "
            "abs_delta_cap=%s safe<=%d mid<=%d mid_cap=%d far_cap=%d hot_path_strict=%s "
            "cache_max_cpu_bytes=%d cache_max_npu_bytes=%d cache_max_entries=%d "
            "max_ready_prefetch_materialize=%d max_ready_prefetch_bytes=%d "
            "batch_cache_prebuild=%s allow_hot_batch_cache_build=%s batch_cache_max_npu_bytes=%d "
            "prefix_cache=%s store_per_prompt_npu=%s keys_cpu_dtype=%s keys_device_dtype=%s",
            self.similarity_threshold,
            self.max_draft_tokens,
            self._max_cache_size,
            str(bool(self._use_numba_rebuild)),
            int(self._entry_blend_horizon),
            int(self._entry_bias_cap),
            str(bool(self._abs_delta_cap_enabled)),
            int(self._abs_delta_safe_threshold),
            int(self._abs_delta_mid_threshold),
            int(self._abs_delta_mid_cap),
            int(self._abs_delta_far_cap),
            str(bool(self._hot_path_strict)),
            int(self._cache_max_cpu_bytes),
            int(self._cache_max_npu_bytes),
            int(self._cache_max_entries),
            int(self._max_ready_prefetch_materialize),
            int(self._max_ready_prefetch_bytes),
            str(bool(self._batch_cache_prebuild)),
            str(bool(self._allow_hot_batch_cache_build)),
            int(self._batch_cache_max_npu_bytes),
            str(bool(self._prefix_cache_enabled)),
            str(bool(self._store_per_prompt_npu)),
            str(self._keys_cpu_dtype_mode),
            str(self._keys_device_dtype),
        )

        if self._use_numba_rebuild:
            try:
                self._warm_numba_rebuild_kernel()
            except Exception:
                logger.debug("HSpec: numba rebuild warmup failed", exc_info=True)

    # async prefetch

    def prefetch_for_batch(self, req_ids: List[str]) -> None:
        """Fire async prefetch for a batch of requests – **non-blocking**.

        Called by ``model_runner`` **before** the forward pass so the
        Ray futures have the entire forward-pass latency (10-100 ms) to
        resolve.  By the time ``generate_token_ids()`` is invoked the
        cache is typically warm already.

        If a prompt is not ready yet, ``generate_token_ids()`` simply
        returns ``draft=[]`` for that request (graceful degradation).
        """
        prompt_ids = self._get_prompt_ids_for_batch(req_ids)
        fetch_prompt_ids = [pid for pid in prompt_ids if pid]
        if not fetch_prompt_ids:
            return

        # Consume any futures that became ready since last call. This happens
        # before the model forward, so descriptor mmap/H2D work stays outside
        # the proposal hot path.
        self._poll_pending(materialize_ready=True,
                           priority_prompt_ids=fetch_prompt_ids)
        if self._batch_cache_prebuild:
            self._prepare_batched_table_cache(
                req_ids,
                prompt_ids,
                dtype=torch.float32,
                device=self.device,
                hot_path=False,
            )
        # Fire new async fetches for cache misses
        self._fire_prefetch_async(fetch_prompt_ids)
        self._maybe_report_metrics(force_proposer_metrics=True)

    def prefetch_prompt_token_ids_batch(
        self,
        prompt_token_ids_batch: List[List[int]],
    ) -> int:
        """Fire async prefetch for a known prompt-token batch.

        This is called before ``LLM.generate()`` starts. When ``max_num_seqs``
        splits one rollout batch into several scheduler waves, later-wave
        requests do not have live ``req_id`` state during the first wave. Their
        tables can still be warmed by stable prompt ids computed from the input
        token ids, avoiding several baseline decode steps after each wave
        enters the scheduler.
        """
        if not prompt_token_ids_batch:
            return 0

        prompt_ids: List[str] = []
        for token_ids in prompt_token_ids_batch:
            try:
                if token_ids:
                    prompt_ids.append(prompt_id_from_token_ids(token_ids))
            except Exception:
                logger.debug("HSpec: failed to build prompt_id for prefetch",
                             exc_info=True)
        if not prompt_ids:
            return 0

        self._poll_pending(materialize_ready=True,
                           priority_prompt_ids=prompt_ids)
        for pid in set(prompt_ids):
            self._not_in_table.discard(pid)
        before = len(self._pending_pids)
        self._fire_prefetch_async(prompt_ids, include_absent=True)
        self._maybe_report_metrics(force_proposer_metrics=True)
        return max(len(self._pending_pids) - before, 0)

    def prefetch_prompt_ids_batch(self, prompt_ids: List[str]) -> int:
        """Fire async prefetch for stable prompt ids.

        This is the lowest-overhead warmup path for rollout-level prefetch:
        the caller computes prompt ids once and the worker only schedules table
        actor fetches.
        """
        prompt_ids = [str(pid) for pid in prompt_ids if pid]
        if not prompt_ids:
            return 0

        self._poll_pending(materialize_ready=True,
                           priority_prompt_ids=prompt_ids)
        for pid in set(prompt_ids):
            self._not_in_table.discard(pid)
        before = len(self._pending_pids)
        self._fire_prefetch_async(prompt_ids, include_absent=True)
        self._maybe_report_metrics(force_proposer_metrics=True)
        return max(len(self._pending_pids) - before, 0)

    def _poll_pending(
        self,
        *,
        materialize_ready: bool = True,
        max_ready_refs: Optional[int] = None,
        priority_prompt_ids: Optional[List[str]] = None,
    ) -> None:
        """Non-blocking: consume any ready prefetch futures.

        Uses ``ray.wait(timeout=0)`` which returns immediately with
        whatever futures are already completed.
        """
        if not self._pending_fetches:
            return
        if not materialize_ready:
            return

        import ray as _ray

        cache_mutated = False
        all_futures = [f for f, _ in self._pending_fetches]
        ready_refs, _ = _ray.wait(all_futures, num_returns=len(all_futures), timeout=0)
        if not ready_refs:
            return
        ready_set_all = set(ready_refs)
        priority_set = {str(pid) for pid in (priority_prompt_ids or []) if pid}
        ready_entries = [
            (idx, future, pids)
            for idx, (future, pids) in enumerate(self._pending_fetches)
            if future in ready_set_all
        ]
        ready_entries.sort(
            key=lambda item: (
                0 if priority_set.intersection(str(pid) for pid in item[2]) else 1,
                item[0],
            )
        )
        ref_budget = int(max_ready_refs or self._max_ready_prefetch_materialize)
        if ref_budget > 0 and len(ready_entries) > ref_budget:
            self._record_proposer_metric(
                "prefetch_ready_ref_throttle_count",
                len(ready_entries) - ref_budget,
            )
            ready_entries = ready_entries[:ref_budget]
        ready_set = {future for _, future, _ in ready_entries}

        version_bumped = False
        materialized_bytes_this_call = 0
        still_pending: List[tuple] = []
        for future, pids in self._pending_fetches:
            if future not in ready_set:
                still_pending.append((future, pids))
                continue

            # Consume this ready future
            deferred_pids: List[str] = []
            try:
                version, table_data = _ray.get(future)
                self._stat_prefetch_ready += 1

                if version < self._cache_version:
                    # Stale data from a previous epoch – discard silently
                    pass
                else:
                    if version > self._cache_version:
                        # Epoch swap detected → invalidate old cache
                        self._clear_prompt_cache()
                        self._not_in_table.clear()
                        self._cache_version = version
                        version_bumped = True
                        cache_mutated = True

                    # Populate cache with fresh data.
                    for pid in pids:
                        data = table_data.get(pid)
                        if data is None:
                            self._not_in_table.add(pid)
                            self._record_proposer_metric("prefetch_absent_payload_count", 1)
                            cache_mutated = True
                            continue
                        estimated_bytes = self._estimate_pending_payload_bytes(data)
                        byte_budget = int(self._max_ready_prefetch_bytes)
                        if byte_budget > 0 and estimated_bytes > 0:
                            would_exceed = (
                                materialized_bytes_this_call
                                + int(estimated_bytes)
                                > byte_budget
                            )
                            if would_exceed and materialized_bytes_this_call > 0:
                                deferred_pids.append(pid)
                                self._record_proposer_metric(
                                    "prefetch_ready_bytes_throttle_count", 1)
                                self._record_proposer_metric(
                                    "prefetch_ready_bytes_deferred",
                                    int(estimated_bytes))
                                continue
                            if would_exceed and materialized_bytes_this_call == 0:
                                self._record_proposer_metric(
                                    "prefetch_ready_bytes_oversize_pass_count", 1)
                                self._record_proposer_metric(
                                    "prefetch_ready_bytes_oversize_pass_bytes",
                                    int(estimated_bytes))
                        try:
                            if isinstance(data, HSpecPromptTableDesc):
                                if int(data.version) > int(version):
                                    raise ValueError(
                                        "HSpec descriptor version mismatch: "
                                        f"prompt_id={pid!r} desc.version={data.version} "
                                        f"active_version={version}"
                                    )
                                self._record_proposer_metric(
                                    "prefetch_descriptor_payload_count", 1)
                                cached = self._build_cached_table_from_descriptor(
                                    data, prompt_id=pid)
                                self._record_proposer_metric(
                                    "descriptor_cache_build_count", 1)
                            elif isinstance(data, dict):
                                self._record_proposer_metric(
                                    "prefetch_legacy_payload_count", 1)
                                cached = self._build_cached_table(data, prompt_id=pid)
                                self._record_proposer_metric(
                                    "legacy_cache_build_count", 1)
                            else:
                                raise TypeError(
                                    f"Unsupported HSpec prefetch payload type: {type(data)!r}"
                                )
                            materialized_bytes_this_call += int(estimated_bytes)
                            if self._admit_cached_table(pid, cached):
                                self._not_in_table.discard(pid)
                            else:
                                self._not_in_table.add(pid)
                            cache_mutated = True
                        except Exception:
                            self._record_proposer_metric("cache_build_error_count", 1)
                            logger.debug(
                                "HSpec: failed to build proposer cache for prompt_id=%r",
                                pid,
                                exc_info=True,
                            )
                            self._not_in_table.add(pid)
                            cache_mutated = True
                    if deferred_pids:
                        still_pending.append((future, deferred_pids))
            except Exception:
                # On error mark prompts as absent to avoid infinite retry
                self._record_proposer_metric("cache_build_error_count", len(pids))
                for pid in pids:
                    self._not_in_table.add(pid)
                cache_mutated = True

            # Remove consumed pids from pending set
            for pid in pids:
                self._pending_pids.discard(pid)
            for pid in deferred_pids:
                self._pending_pids.add(pid)

        self._pending_fetches = still_pending

        # On epoch swap, abandon remaining (likely stale) pending
        # futures.  Their Ray ObjectRefs are GC'd harmlessly.  Fresh
        # fetches will be fired by the next _fire_prefetch_async() call.
        if version_bumped and self._pending_fetches:
            self._pending_fetches = []
            self._pending_pids.clear()

        if cache_mutated:
            self._cache_generation += 1
            self._batched_table_cache = None

    def _fire_prefetch_async(
        self,
        prompt_ids: List[str],
        include_absent: bool = False,
    ) -> None:
        """Fire async Ray futures for uncached prompts – **non-blocking**.

        Futures are appended to ``_pending_fetches`` and polled later
        by ``_poll_pending()``.  Prompts already cached, pending, or
        known-absent are skipped.
        """
        missing = [
            pid for pid in set(prompt_ids)
            if pid not in self._cache
            and (include_absent or pid not in self._not_in_table)
            and pid not in self._pending_pids
        ]
        if not missing:
            return

        try:
            new_futures = self.hspec_tables.prefetch_batch_async(missing)
            for future, pids in new_futures:
                self._pending_fetches.append((future, pids))
                self._pending_pids.update(pids)
            if new_futures:
                self._stat_prefetch_fired += len(new_futures)
        except Exception:
            logger.debug("HSpec: async prefetch fire failed", exc_info=True)

    def _get_or_create_prompt_id(self, req_id: str) -> str:
        """Return a stable prompt_id for a live request, caching by req_id."""
        cached = self._req_prompt_ids.get(req_id)
        if cached is not None:
            return cached

        req_state = self.runner.requests.get(req_id)
        if req_state is None:
            return ""

        prompt_id = prompt_id_from_token_ids(req_state.prompt_token_ids)
        self._req_prompt_ids[req_id] = prompt_id
        return prompt_id

    def _get_prompt_ids_for_batch(self, req_ids: List[str]) -> List[str]:
        """Return prompt_ids aligned to the current scheduled batch."""
        req_ids_tuple = tuple(str(req_id) for req_id in req_ids)
        if self._cached_batch_req_ids == req_ids_tuple:
            return self._cached_batch_prompt_ids

        prompt_ids = [self._get_or_create_prompt_id(req_id) for req_id in req_ids]
        self._cached_batch_req_ids = req_ids_tuple
        self._cached_batch_prompt_ids = prompt_ids
        return prompt_ids

    def _warm_numba_rebuild_kernel(self) -> None:
        """Compile the Numba rebuild kernel once outside the hot path."""
        if not self._use_numba_rebuild:
            return
        comp = np.zeros((2, 2), dtype=np.float32)
        keys = np.zeros((2, 2), dtype=np.float32)
        comp_list = _hspec_make_numba_array_list([comp])
        keys_list = _hspec_make_numba_array_list([keys])
        if comp_list is None or keys_list is None:
            return
        comp_out = np.zeros((1, 2, 2), dtype=np.float32)
        keys_out = np.zeros((1, 2, 2), dtype=np.float32)
        lens_out = np.empty((1,), dtype=np.int64)
        _hspec_fill_batched_components_keys_numba(
            comp_list,
            keys_list,
            comp_out,
            keys_out,
            lens_out,
        )

    def _get_or_build_batched_table_cache(
        self,
        req_ids: List[str],
        prompt_ids: List[str],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[_BatchedPromptTableCache]:
        """Compatibility wrapper for explicit debug fallback only.

        Production Step 4 flow prepares this cache from prefetch_for_batch()
        and generate_token_ids() consumes it with _get_prebuilt_batched_table_cache().
        """
        return self._prepare_batched_table_cache(
            req_ids,
            prompt_ids,
            dtype=dtype,
            device=device,
            hot_path=True,
        )

    def _batch_cache_matches(
        self,
        cached: Optional[_BatchedPromptTableCache],
        req_ids_tuple: tuple[str, ...],
        prompt_ids_tuple: tuple[str, ...],
    ) -> bool:
        return (
            cached is not None
            and cached.req_ids == req_ids_tuple
            and cached.prompt_ids == prompt_ids_tuple
            and cached.cache_generation == self._cache_generation
        )

    @staticmethod
    def _dtype_element_size(dtype: torch.dtype) -> int:
        try:
            return int(torch.empty((), dtype=dtype).element_size())
        except Exception:
            return 4

    def _estimate_batched_table_cache_nbytes(
        self,
        cached_tables: List[_CachedPromptTable],
        dtype: torch.dtype,
    ) -> int:
        if not cached_tables:
            return 0
        try:
            num_rows = len(cached_tables)
            k_max = max(int(cached.components_t_cpu.shape[1])
                        for cached in cached_tables)
            m_max = max(int(cached.n_entries) for cached in cached_tables)
            hidden_dim = int(cached_tables[0].mean_cpu.shape[0])
            model_elem = self._dtype_element_size(dtype)
            key_elem = self._dtype_element_size(self._keys_device_dtype)
            total = (
                num_rows * hidden_dim * model_elem
                + num_rows * hidden_dim * k_max * model_elem
                + num_rows * m_max * k_max * key_elem
                + num_rows * self._dtype_element_size(torch.long)
                + num_rows * m_max * self._dtype_element_size(torch.bool)
            )
            return int(total)
        except Exception:
            return 0

    def _get_batch_cache_stream_context(self) -> tuple[Any, bool]:
        if self._batch_cache_copy_stream_disabled:
            return nullcontext(), False
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            return nullcontext(), False
        try:
            stream_fn = getattr(npu_mod, "stream", None)
            stream_cls = getattr(npu_mod, "Stream", None)
            event_cls = getattr(npu_mod, "Event", None)
            if stream_fn is None or stream_cls is None or event_cls is None:
                self._batch_cache_copy_stream_disabled = True
                return nullcontext(), False
            if self._batch_cache_copy_stream is None:
                self._batch_cache_copy_stream = stream_cls()
            return stream_fn(self._batch_cache_copy_stream), True
        except Exception:
            self._batch_cache_copy_stream_disabled = True
            logger.debug("HSpec: batch cache copy stream unavailable", exc_info=True)
            return nullcontext(), False

    def _record_batch_cache_ready_event(self) -> Optional[Any]:
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            return None
        try:
            event_cls = getattr(npu_mod, "Event", None)
            if event_cls is None:
                return None
            event = event_cls()
            event.record()
            return event
        except Exception:
            logger.debug("HSpec: failed to record batch cache ready event",
                         exc_info=True)
            return None

    def _batch_cache_event_ready(self, cached: _BatchedPromptTableCache) -> bool:
        event = getattr(cached, "ready_event", None)
        if event is None:
            return True
        try:
            query = getattr(event, "query", None)
            if query is None:
                return True
            return bool(query())
        except Exception:
            logger.debug("HSpec: batch cache event query failed", exc_info=True)
            return True

    def _get_prebuilt_batched_table_cache(
        self,
        req_ids: List[str],
        prompt_ids: List[str],
    ) -> Optional[_BatchedPromptTableCache]:
        req_ids_tuple = tuple(str(req_id) for req_id in req_ids)
        prompt_ids_tuple = tuple(str(pid) for pid in prompt_ids)
        cached = self._batched_table_cache
        if not self._batch_cache_matches(cached, req_ids_tuple, prompt_ids_tuple):
            return None
        if cached is None:
            return None
        if not self._batch_cache_event_ready(cached):
            self._record_proposer_metric("batch_cache_event_not_ready_count", 1)
            return None
        return cached

    def _prepare_batched_table_cache(
        self,
        req_ids: List[str],
        prompt_ids: List[str],
        dtype: torch.dtype,
        device: torch.device,
        *,
        hot_path: bool = False,
    ) -> Optional[_BatchedPromptTableCache]:
        if hot_path and not self._allow_hot_batch_cache_build:
            self._record_proposer_metric("batch_cache_miss_no_hot_build", 1)
            return self._get_prebuilt_batched_table_cache(req_ids, prompt_ids)

        req_ids_tuple = tuple(str(req_id) for req_id in req_ids)
        prompt_ids_tuple = tuple(str(pid) for pid in prompt_ids)
        cached = self._batched_table_cache
        if self._batch_cache_matches(cached, req_ids_tuple, prompt_ids_tuple):
            if cached is not None and (not hot_path or self._batch_cache_event_ready(cached)):
                return cached

        batch_indices: List[int] = []
        cached_tables: List[_CachedPromptTable] = []
        with hspec_record_function("hspec/proposal/build_batch_indices_cached_tables"):
            for i, pid in enumerate(prompt_ids):
                prompt_table = self._cache.get(pid)
                if prompt_table is None or prompt_table.n_entries <= 0:
                    continue
                batch_indices.append(i)
                cached_tables.append(prompt_table)

        if not batch_indices:
            self._batched_table_cache = None
            return None

        estimated_npu_bytes = self._estimate_batched_table_cache_nbytes(
            cached_tables, dtype)
        if (self._batch_cache_max_npu_bytes > 0
                and estimated_npu_bytes > self._batch_cache_max_npu_bytes):
            self._batched_table_cache = None
            self._record_proposer_metric("batch_cache_budget_skip_count", 1)
            self._record_proposer_metric(
                "batch_cache_budget_skip_bytes", estimated_npu_bytes)
            self._record_proposer_gauge("batch_cache_npu_bytes", 0)
            return None

        build_t0 = _now_ns()
        ready_event: Optional[Any] = None
        if hot_path:
            mean_batch, components_t_batch, keys_batch, key_lengths, invalid_key_mask = (
                self._build_batched_table_tensors(cached_tables, dtype, device))
        else:
            stream_context, used_copy_stream = self._get_batch_cache_stream_context()
            with stream_context:
                mean_batch, components_t_batch, keys_batch, key_lengths, invalid_key_mask = (
                    self._build_batched_table_tensors(cached_tables, dtype, device))
                if used_copy_stream:
                    ready_event = self._record_batch_cache_ready_event()
            if used_copy_stream and ready_event is None:
                self._record_proposer_metric(
                    "batch_cache_event_record_error_count", 1)
                return None
        build_t1 = _now_ns()

        with hspec_record_function("hspec/proposal/build_batch_idx_to_row"):
            batch_idx_to_row = {batch_idx: row for row, batch_idx in enumerate(batch_indices)}

        with hspec_record_function("hspec/proposal/build_cached"):
            cached = _BatchedPromptTableCache(
                req_ids=req_ids_tuple,
                prompt_ids=prompt_ids_tuple,
                cache_generation=self._cache_generation,
                batch_indices=batch_indices,
                batch_idx_to_row=batch_idx_to_row,
                cached_tables=cached_tables,
                mean_batch=mean_batch,
                components_t_batch=components_t_batch,
                keys_batch=keys_batch,
                key_lengths=key_lengths,
                invalid_key_mask=invalid_key_mask,
                ready_event=ready_event,
                estimated_npu_bytes=estimated_npu_bytes,
            )
            self._batched_table_cache = cached
        build_ms = _ns_to_ms(build_t1 - build_t0)
        if hot_path:
            self._record_proposer_metric("batch_cache_hot_build_count", 1)
            self._record_proposer_metric("batch_cache_hot_build_ms", build_ms)
        else:
            self._record_proposer_metric("batch_cache_prebuild_count", 1)
            self._record_proposer_metric("batch_cache_prebuild_ms", build_ms)
            self._record_proposer_metric("batch_cache_h2d_submit_ms", build_ms)
        self._record_proposer_gauge("batch_cache_npu_bytes", estimated_npu_bytes)
        return cached

    def _maybe_log_metrics(self) -> None:
        """Best-effort periodic metrics log (no blocking, minimal overhead)."""
        self._stat_calls += 1
        now = time.time()
        if ((self._stat_calls % self._log_every_calls != 0)
                and ((now - self._last_log_t) < self._log_every_s)):
            return
        self._last_log_t = now

        q = max(int(self._stat_queries), 1)
        h = int(self._stat_hits)
        match_rate = float(h) / float(q)
        avg_draft = float(self._stat_total_draft_len) / float(max(h, 1))
        avg_accept = (
            float(self._stat_accept_sum) / float(self._stat_accept_count)
            if self._stat_accept_count > 0
            else 0.0
        )
        '''
        logger.info(
            "HSpec online metrics: queries=%d hits=%d match_rate=%.3f "
            "avg_draft_len=%.2f avg_accept_len=%.2f cache_size=%d "
            "pending=%d prefetch_fired=%d prefetch_ready=%d version=%d",
            int(self._stat_queries),
            h,
            match_rate,
            avg_draft,
            avg_accept,
            len(self._cache),
            len(self._pending_fetches),
            int(self._stat_prefetch_fired),
            int(self._stat_prefetch_ready),
            int(self._cache_version),
        )
        '''

        # Also report stats to the global table group at low frequency so the
        # trainer can observe match_rate/avg_draft_len even when queries are
        # executed locally on the worker.
        self._maybe_report_metrics()

    def _maybe_report_metrics(self, force_proposer_metrics: bool = False) -> None:
        """Non-blocking metrics reporting (fire-and-forget Ray RPC)."""
        proposer_metrics = dict(getattr(self, "_proposer_metric_deltas", {}))
        proposer_metrics.update(getattr(self, "_proposer_metric_gauges", {}))
        has_proposer_metrics = bool(proposer_metrics)
        prompt_access_pending = getattr(self, "_prompt_access_pending", {})
        has_prompt_access_pending = bool(prompt_access_pending)
        prompt_access_due = (
            has_prompt_access_pending
            and self._stat_calls - int(getattr(
                self, "_last_prompt_access_report_calls", 0)
            ) >= int(getattr(self, "_prompt_access_report_every_calls", 16))
        )
        if (
            not force_proposer_metrics
            and not prompt_access_due
            and self._stat_calls - self._last_report_calls < self._report_every_calls
        ):
            return
        if force_proposer_metrics and not has_proposer_metrics and not prompt_access_due:
            return
        self._last_report_calls = self._stat_calls

        dq = int(self._stat_queries - self._reported_queries)
        dh = int(self._stat_hits - self._reported_hits)
        ddl = int(self._stat_total_draft_len - self._reported_total_draft_len)
        has_entry_pending = (
            self._entry_pending_match_count > 0
            or self._entry_pending_verify_count > 0
            or self._entry_pending_accept_count > 0
            or bool(self._entry_pending_abs_delta_verify)
            or bool(self._entry_pending_abs_delta_accept)
            or bool(self._entry_pending_abs_delta_accept_len_sum)
        )
        if (
            dq <= 0
            and dh <= 0
            and ddl <= 0
            and not has_entry_pending
            and not has_proposer_metrics
            and not prompt_access_due
        ):
            return

        self._reported_queries = int(self._stat_queries)
        self._reported_hits = int(self._stat_hits)
        self._reported_total_draft_len = int(self._stat_total_draft_len)

        try:
            # Fire-and-forget, never block the hot loop.
            if hasattr(self.hspec_tables, "report_online_metrics_async"):
                self.hspec_tables.report_online_metrics_async(dq, dh, ddl)
        except Exception:
            # Swallow all errors; metrics must never affect decoding.
            pass

        if has_entry_pending:
            try:
                if hasattr(self.hspec_tables, "report_entry_metrics_async"):
                    self.hspec_tables.report_entry_metrics_async(
                        match_count=int(self._entry_pending_match_count),
                        delta_sum=int(self._entry_pending_delta_sum),
                        abs_delta_sum=int(self._entry_pending_abs_delta_sum),
                        verify_count=int(self._entry_pending_verify_count),
                        accept_count=int(self._entry_pending_accept_count),
                        accept_len_sum=int(self._entry_pending_accept_len_sum),
                        abs_delta_verify=dict(self._entry_pending_abs_delta_verify),
                        abs_delta_accept=dict(self._entry_pending_abs_delta_accept),
                        abs_delta_accept_len_sum=dict(self._entry_pending_abs_delta_accept_len_sum),
                    )
            except Exception:
                pass
            finally:
                self._entry_pending_match_count = 0
                self._entry_pending_delta_sum = 0
                self._entry_pending_abs_delta_sum = 0
                self._entry_pending_verify_count = 0
                self._entry_pending_accept_count = 0
                self._entry_pending_accept_len_sum = 0
                self._entry_pending_abs_delta_verify.clear()
                self._entry_pending_abs_delta_accept.clear()
                self._entry_pending_abs_delta_accept_len_sum.clear()

        if has_proposer_metrics:
            try:
                self._proposer_metric_deltas.clear()
                if hasattr(self.hspec_tables, "report_proposer_cache_metrics_async"):
                    self.hspec_tables.report_proposer_cache_metrics_async(proposer_metrics)
            except Exception:
                pass

        if prompt_access_due:
            prompt_stats: Dict[str, Dict[str, int]] = {}
            try:
                for prompt_id, stats in list(prompt_access_pending.items()):
                    if not isinstance(stats, dict):
                        continue
                    q = max(int(stats.get("query", 0) or 0), 0)
                    h = max(int(stats.get("hit", 0) or 0), 0)
                    if q <= 0 and h <= 0:
                        continue
                    prompt_stats[str(prompt_id)] = {"query": q, "hit": h}
                if prompt_stats and hasattr(
                        self.hspec_tables, "report_prompt_access_metrics_async"):
                    self.hspec_tables.report_prompt_access_metrics_async(prompt_stats)
                    prompt_access_pending.clear()
                    self._last_prompt_access_report_calls = int(self._stat_calls)
            except Exception:
                pass

    def _record_proposer_metric(self, key: str, value: float = 1.0) -> None:
        try:
            self._proposer_metric_deltas[str(key)] += float(value)
        except Exception:
            pass

    def _record_prompt_access(
        self,
        prompt_id: str,
        *,
        query: int = 0,
        hit: int = 0,
    ) -> None:
        try:
            if not prompt_id:
                return
            stats = self._prompt_access_pending[str(prompt_id)]
            stats["query"] = int(stats.get("query", 0)) + max(int(query), 0)
            stats["hit"] = int(stats.get("hit", 0)) + max(int(hit), 0)
        except Exception:
            pass

    def _record_proposer_gauge(self, key: str, value: float) -> None:
        try:
            self._proposer_metric_gauges[str(key)] = float(value)
        except Exception:
            pass

    def _update_cache_live_gauges(self) -> None:
        self._record_proposer_gauge("cache_live_cpu_bytes",
                                    float(self._cache_cpu_bytes))
        self._record_proposer_gauge("cache_live_npu_bytes",
                                    float(self._cache_npu_bytes))
        self._record_proposer_gauge("cache_live_entries",
                                    float(self._cache_entries))
        self._record_proposer_gauge("cache_live_prompts",
                                    float(len(self._cache)))

    def _estimate_cached_table_bytes(
        self,
        cached: _CachedPromptTable,
    ) -> tuple[int, int, int]:
        cpu_bytes = 0
        seen_arrays: Set[int] = set()

        def add_array(arr: Any) -> None:
            nonlocal cpu_bytes
            if not isinstance(arr, np.ndarray):
                return
            arr_id = id(arr)
            if arr_id in seen_arrays:
                return
            seen_arrays.add(arr_id)
            try:
                cpu_bytes += int(arr.nbytes)
            except Exception:
                pass

        for arr in (
            cached.mean_cpu,
            cached.components_t_cpu,
            cached.keys_cpu,
            cached.token_buffer,
            cached.rollout_token_offset,
            cached.rollout_token_len,
            cached.entry_rollout_idx,
            cached.entry_offset,
            cached.draft_prefix_tokens,
            cached.draft_prefix_lens,
            cached.rollout_entry_starts,
            cached.rollout_entry_lens,
            cached.entry_bias,
            cached.entry_hits,
        ):
            add_array(arr)
        for seq in cached.rollout_seqs or ():
            add_array(seq)

        npu_bytes = (
            _tensor_nbytes(cached.mean)
            + _tensor_nbytes(cached.components)
            + _tensor_nbytes(cached.keys)
        )
        entries = int(cached.n_entries)
        return int(cpu_bytes), int(npu_bytes), entries

    def _estimate_pending_payload_bytes(self, data: Any) -> int:
        if isinstance(data, HSpecPromptTableDesc):
            try:
                parts = estimate_prompt_table_desc_nbytes(data)
                total = int(parts.get("total", 0))
                if self._keys_cpu_dtype_mode == "float32":
                    key_count = 1
                    for dim in tuple(data.keys.shape):
                        key_count *= int(dim)
                    table_key_bytes = int(parts.get("keys", 0))
                    fp32_key_bytes = int(key_count) * np.dtype(np.float32).itemsize
                    total += max(0, fp32_key_bytes - table_key_bytes)
                return max(int(total), 0)
            except Exception:
                self._record_proposer_metric(
                    "prefetch_ready_bytes_estimate_error_count", 1)
                return 0
        if isinstance(data, dict):
            total = 0
            for value in data.values():
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, np.ndarray):
                            total += int(item.nbytes)
            return int(total)
        return 0

    def _remove_cached_prompt(
        self,
        pid: str,
        *,
        record_eviction: bool,
    ) -> bool:
        cached = self._cache.pop(pid, None)
        if cached is None:
            self._cache_prompt_bytes.pop(pid, None)
            return False
        cpu_bytes, npu_bytes, entries = self._cache_prompt_bytes.pop(
            pid, self._estimate_cached_table_bytes(cached))
        self._cache_cpu_bytes = max(0, int(self._cache_cpu_bytes) - int(cpu_bytes))
        self._cache_npu_bytes = max(0, int(self._cache_npu_bytes) - int(npu_bytes))
        self._cache_entries = max(0, int(self._cache_entries) - int(entries))
        if record_eviction:
            self._record_proposer_metric("cache_eviction_count", 1)
            self._record_proposer_metric("cache_eviction_cpu_bytes", cpu_bytes)
            self._record_proposer_metric("cache_eviction_npu_bytes", npu_bytes)
            self._record_proposer_metric("cache_eviction_entries", entries)
        self._update_cache_live_gauges()
        return True

    def _clear_prompt_cache(self) -> None:
        self._cache.clear()
        self._cache_prompt_bytes.clear()
        self._cache_cpu_bytes = 0
        self._cache_npu_bytes = 0
        self._cache_entries = 0
        self._batched_table_cache = None
        self._update_cache_live_gauges()

    def _cache_fits_after_add(
        self,
        cpu_bytes: int,
        npu_bytes: int,
        entries: int,
    ) -> bool:
        if self._max_cache_size > 0 and len(self._cache) + 1 > self._max_cache_size:
            return False
        if (self._cache_max_cpu_bytes > 0
                and self._cache_cpu_bytes + int(cpu_bytes) > self._cache_max_cpu_bytes):
            return False
        if (self._cache_max_npu_bytes > 0
                and self._cache_npu_bytes + int(npu_bytes) > self._cache_max_npu_bytes):
            return False
        if (self._cache_max_entries > 0
                and self._cache_entries + int(entries) > self._cache_max_entries):
            return False
        return True

    def _admit_cached_table(self, pid: str, cached: _CachedPromptTable) -> bool:
        cpu_bytes, npu_bytes, entries = self._estimate_cached_table_bytes(cached)
        oversize = (
            (self._cache_max_cpu_bytes > 0 and cpu_bytes > self._cache_max_cpu_bytes)
            or (self._cache_max_npu_bytes > 0 and npu_bytes > self._cache_max_npu_bytes)
            or (self._cache_max_entries > 0 and entries > self._cache_max_entries)
        )
        if oversize:
            self._record_proposer_metric("cache_budget_skip_count", 1)
            self._record_proposer_metric("cache_oversize_skip_count", 1)
            self._record_proposer_metric("cache_oversize_cpu_bytes", cpu_bytes)
            self._record_proposer_metric("cache_oversize_npu_bytes", npu_bytes)
            self._record_proposer_metric("cache_oversize_entries", entries)
            self._update_cache_live_gauges()
            return False

        if pid in self._cache:
            self._remove_cached_prompt(pid, record_eviction=False)

        while self._cache and not self._cache_fits_after_add(
                cpu_bytes, npu_bytes, entries):
            old_pid = next(iter(self._cache))
            self._remove_cached_prompt(old_pid, record_eviction=True)

        if not self._cache_fits_after_add(cpu_bytes, npu_bytes, entries):
            self._record_proposer_metric("cache_budget_skip_count", 1)
            self._update_cache_live_gauges()
            return False

        self._cache[pid] = cached
        self._cache.move_to_end(pid)
        self._cache_prompt_bytes[pid] = (
            int(cpu_bytes), int(npu_bytes), int(entries))
        self._cache_cpu_bytes += int(cpu_bytes)
        self._cache_npu_bytes += int(npu_bytes)
        self._cache_entries += int(entries)
        self._record_proposer_metric("cache_admit_count", 1)
        self._record_proposer_metric("cache_admit_cpu_bytes", cpu_bytes)
        self._record_proposer_metric("cache_admit_npu_bytes", npu_bytes)
        self._record_proposer_metric("cache_admit_entries", entries)
        self._update_cache_live_gauges()
        return True

    def _guard_no_hot_path_materialize(self, operation: str) -> bool:
        if not getattr(self, "_in_generate_token_ids", False):
            return True
        self._record_proposer_metric("hot_path_violation_count", 1)
        message = (
            "HSpec hot path attempted descriptor materialization or table H2D: "
            f"{operation}"
        )
        if getattr(self, "_hot_path_strict", True):
            raise RuntimeError(message)
        logger.debug(message)
        return False

    @staticmethod
    def _build_rollout_entry_spans(
        entry_rollout_idx: np.ndarray,
        n_entries: int,
        num_rollouts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build contiguous per-rollout entry spans for local-window matching."""
        starts = np.zeros((num_rollouts,), dtype=np.int32)
        lens = np.zeros((num_rollouts,), dtype=np.int32)
        if n_entries <= 0 or num_rollouts <= 0:
            return starts, lens

        counts = np.bincount(entry_rollout_idx[:n_entries], minlength=num_rollouts).astype(
            np.int32, copy=False)
        cursor = 0
        for ridx, count in enumerate(counts.tolist()):
            starts[ridx] = cursor
            lens[ridx] = count
            cursor += count
        return starts, lens

    @staticmethod
    def _clamp_wnd_size(wnd_size: int, min_wnd: int, max_wnd: int) -> int:
        return max(int(min_wnd), min(int(wnd_size), int(max_wnd)))

    def _copy_table_array_from_desc(self, array_desc, dtype=None) -> np.ndarray:
        self._guard_no_hot_path_materialize("copy_table_array_from_desc")
        mmap_arr = open_array(array_desc, mode="r")
        try:
            return np.ascontiguousarray(
                np.array(mmap_arr, dtype=dtype, copy=True))
        finally:
            _close_hspec_memmap(mmap_arr)

    @staticmethod
    def _build_draft_prefix_from_token_refs(
        token_buffer: np.ndarray,
        rollout_offsets: np.ndarray,
        rollout_lens: np.ndarray,
        entry_rollout_idx: np.ndarray,
        entry_offset: np.ndarray,
        n_entries: int,
        max_wnd: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_entries = int(n_entries)
        max_wnd = int(max_wnd)
        draft_prefix_tokens = np.zeros((n_entries, max_wnd), dtype=np.int32)
        draft_prefix_lens = np.zeros((n_entries,), dtype=np.int32)
        if n_entries <= 0 or max_wnd <= 0:
            return draft_prefix_tokens, draft_prefix_lens

        ridx = np.asarray(entry_rollout_idx[:n_entries], dtype=np.int64)
        off = np.asarray(entry_offset[:n_entries], dtype=np.int64)
        rollout_offsets = np.asarray(rollout_offsets, dtype=np.int64)
        rollout_lens = np.asarray(rollout_lens, dtype=np.int64)
        token_buffer = np.asarray(token_buffer, dtype=np.int32)
        if ridx.size and (int(ridx.min()) < 0 or int(ridx.max()) >= len(rollout_offsets)):
            raise ValueError(
                "HSpec descriptor entry_rollout_idx out of bounds: "
                f"min={int(ridx.min())} max={int(ridx.max())} n_rollouts={len(rollout_offsets)}"
            )
        if off.size and int(off.min()) < 0:
            raise ValueError("HSpec descriptor entry_offset must be non-negative")
        if len(rollout_offsets) != len(rollout_lens):
            raise ValueError("HSpec descriptor rollout offset/len shape mismatch")
        if len(rollout_offsets) > 0:
            rollout_end = rollout_offsets + rollout_lens
            if int(rollout_offsets.min()) < 0 or int(rollout_lens.min()) < 0:
                raise ValueError("HSpec descriptor rollout token refs must be non-negative")
            if int(rollout_end.max()) > int(token_buffer.shape[0]):
                raise ValueError(
                    "HSpec descriptor rollout token refs exceed token_buffer: "
                    f"end={int(rollout_end.max())} token_count={int(token_buffer.shape[0])}"
                )
        rollout_len_for_entry = rollout_lens[ridx]
        if np.any(off > rollout_len_for_entry):
            raise ValueError("HSpec descriptor entry_offset exceeds rollout length")

        starts = rollout_offsets[ridx] + off
        available = rollout_len_for_entry - off
        lens = np.minimum(max_wnd, np.maximum(available, 0)).astype(np.int32)
        draft_prefix_lens[:] = lens
        cols = np.arange(max_wnd, dtype=np.int64)
        max_index_elems = 1_000_000
        block_rows = max(1, max_index_elems // max(max_wnd, 1))
        for row_start in range(0, n_entries, block_rows):
            row_end = min(n_entries, row_start + block_rows)
            row_starts = starts[row_start:row_end]
            row_lens = draft_prefix_lens[row_start:row_end]
            idx = row_starts[:, None] + cols[None, :]
            mask = cols[None, :] < row_lens[:, None]
            if np.any(mask):
                draft_prefix_tokens[row_start:row_end][mask] = token_buffer[idx[mask]]
        return draft_prefix_tokens, draft_prefix_lens

    @staticmethod
    def _flatten_rollout_seqs(
        rollout_seqs: List[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = np.zeros((len(rollout_seqs),), dtype=np.int64)
        lens = np.zeros((len(rollout_seqs),), dtype=np.int32)
        total = 0
        seqs: List[np.ndarray] = []
        for idx, seq in enumerate(rollout_seqs):
            arr = np.ascontiguousarray(np.asarray(seq, dtype=np.int32))
            offsets[idx] = total
            lens[idx] = int(arr.shape[0])
            total += int(arr.shape[0])
            seqs.append(arr)
        if total <= 0:
            token_buffer = np.empty((0,), dtype=np.int32)
        else:
            token_buffer = np.empty((total,), dtype=np.int32)
            cursor = 0
            for arr in seqs:
                end = cursor + int(arr.shape[0])
                token_buffer[cursor:end] = arr
                cursor = end
        return token_buffer, offsets, lens

    def _finalize_cached_table(
        self,
        *,
        mean_np: np.ndarray,
        components_np: np.ndarray,
        keys_np: np.ndarray,
        token_buffer: np.ndarray,
        rollout_token_offset: np.ndarray,
        rollout_token_len: np.ndarray,
        entry_rollout_idx: np.ndarray,
        entry_offset: np.ndarray,
        n_entries: int,
        wnd_size: int,
        max_wnd: int,
        min_wnd: int,
        prompt_id: Optional[str],
        rollout_seqs: Optional[List[np.ndarray]] = None,
        draft_prefix_tokens: Optional[np.ndarray] = None,
        draft_prefix_lens: Optional[np.ndarray] = None,
        metric_prefix: Optional[str] = None,
    ) -> _CachedPromptTable:
        n_entries = int(n_entries)
        max_wnd = int(max_wnd)
        min_wnd = int(min_wnd)
        if n_entries <= 0:
            raise ValueError("HSpec cached table requires positive n_entries")

        mean_np = np.ascontiguousarray(mean_np, dtype=np.float32)
        components_np = np.ascontiguousarray(components_np, dtype=np.float32)
        if self._keys_cpu_dtype_mode == "float32":
            keys_np = np.ascontiguousarray(keys_np, dtype=np.float32)
        else:
            keys_np = np.ascontiguousarray(keys_np)
        components_t_cpu = np.ascontiguousarray(components_np.transpose(1, 0))
        token_buffer = np.ascontiguousarray(np.asarray(token_buffer, dtype=np.int32))
        rollout_token_offset = np.ascontiguousarray(
            np.asarray(rollout_token_offset, dtype=np.int64))
        rollout_token_len = np.ascontiguousarray(
            np.asarray(rollout_token_len, dtype=np.int32))
        entry_rollout_idx = np.ascontiguousarray(
            np.asarray(entry_rollout_idx[:n_entries], dtype=np.int32))
        entry_offset = np.ascontiguousarray(
            np.asarray(entry_offset[:n_entries], dtype=np.int32))
        n_rollouts = int(rollout_token_len.shape[0])
        if n_rollouts <= 0:
            raise ValueError("HSpec cached table requires at least one rollout")
        if entry_rollout_idx.shape[0] < n_entries or entry_offset.shape[0] < n_entries:
            raise ValueError("HSpec cached table entry arrays shorter than n_entries")
        if int(entry_rollout_idx.min()) < 0 or int(entry_rollout_idx.max()) >= n_rollouts:
            raise ValueError("HSpec cached table entry_rollout_idx out of bounds")
        if int(rollout_token_offset.min()) < 0 or int(rollout_token_len.min()) < 0:
            raise ValueError("HSpec cached table rollout token refs must be non-negative")
        rollout_end = rollout_token_offset + rollout_token_len.astype(np.int64)
        if int(rollout_end.max()) > int(token_buffer.shape[0]):
            raise ValueError("HSpec cached table rollout refs exceed token_buffer")
        rollout_len_for_entry = rollout_token_len[entry_rollout_idx]
        if np.any(entry_offset < 0) or np.any(entry_offset > rollout_len_for_entry):
            raise ValueError("HSpec cached table entry_offset exceeds rollout length")

        mean = None
        components = None
        keys = None
        if self._store_per_prompt_npu:
            self._guard_no_hot_path_materialize("finalize_cached_table_h2d")
            t0_h2d = _now_ns()
            mean = torch.from_numpy(mean_np).to(
                self.device, dtype=torch.float32, non_blocking=True)
            components = torch.from_numpy(components_np).to(
                self.device, dtype=torch.float32, non_blocking=True)
            keys = torch.from_numpy(keys_np).to(
                self.device, dtype=self._keys_device_dtype, non_blocking=True)
            h2d_ms = _ns_to_ms(_now_ns() - t0_h2d)
            if metric_prefix:
                self._record_proposer_metric(f"{metric_prefix}_h2d_submit_ms", h2d_ms)

        rollout_entry_starts, rollout_entry_lens = self._build_rollout_entry_spans(
            entry_rollout_idx, n_entries, n_rollouts)
        if prompt_id:
            prior = getattr(self, "_prompt_wnd_priors", {}).get(prompt_id)
            if prior is not None:
                wnd_size = int(prior)
        wnd_size = self._clamp_wnd_size(wnd_size, min_wnd, max_wnd)

        if self._prefix_cache_enabled:
            if draft_prefix_tokens is None or draft_prefix_lens is None:
                draft_prefix_tokens, draft_prefix_lens = self._build_draft_prefix_from_token_refs(
                    token_buffer,
                    rollout_token_offset,
                    rollout_token_len,
                    entry_rollout_idx,
                    entry_offset,
                    n_entries,
                    max_wnd,
                )
            else:
                draft_prefix_tokens = np.ascontiguousarray(
                    np.asarray(draft_prefix_tokens, dtype=np.int32))
                draft_prefix_lens = np.ascontiguousarray(
                    np.asarray(draft_prefix_lens, dtype=np.int32))
        else:
            draft_prefix_tokens = None
            draft_prefix_lens = None

        entry_bias = np.zeros((n_entries,), dtype=np.int8)
        entry_hits = np.zeros((n_entries,), dtype=np.uint16)
        return _CachedPromptTable(
            mean_cpu=mean_np,
            components_t_cpu=components_t_cpu,
            keys_cpu=keys_np,
            mean=mean,
            components=components,
            keys=keys,
            token_buffer=token_buffer,
            rollout_token_offset=rollout_token_offset,
            rollout_token_len=rollout_token_len,
            rollout_seqs=rollout_seqs,
            entry_rollout_idx=entry_rollout_idx,
            entry_offset=entry_offset,
            draft_prefix_tokens=draft_prefix_tokens,
            draft_prefix_lens=draft_prefix_lens,
            rollout_entry_starts=rollout_entry_starts,
            rollout_entry_lens=rollout_entry_lens,
            n_entries=n_entries,
            wnd_size=wnd_size,
            max_wnd=max_wnd,
            min_wnd=min_wnd,
            entry_bias=entry_bias,
            entry_hits=entry_hits,
            entry_blend_horizon=getattr(self, "_entry_blend_horizon", 4),
            max_entry_bias=min(
                getattr(self, "_entry_bias_cap", 8),
                max(0, max_wnd - min_wnd),
            ),
        )

    def _build_cached_table_from_descriptor(
        self,
        desc: HSpecPromptTableDesc,
        prompt_id: Optional[str] = None,
    ) -> _CachedPromptTable:
        """Build a worker-local cache from mmap table descriptors."""
        self._guard_no_hot_path_materialize(
            "build_cached_table_from_descriptor")
        if prompt_id and str(desc.prompt_id) != str(prompt_id):
            raise ValueError(
                f"HSpec descriptor prompt mismatch: desc={desc.prompt_id!r} "
                f"requested={prompt_id!r}"
            )
        n_entries = int(desc.n_entries)
        n_rollouts = int(desc.n_rollouts)
        hidden_dim = int(desc.hidden_dim)
        n_components = int(desc.n_components)
        if n_entries <= 0 or n_rollouts <= 0:
            raise ValueError("HSpec descriptor table is empty")
        if hidden_dim <= 0 or n_components <= 0:
            raise ValueError("HSpec descriptor PCA dimensions must be positive")

        t0_materialize = _now_ns()
        mean_np = self._copy_table_array_from_desc(desc.mean, np.float32)
        components_np = self._copy_table_array_from_desc(desc.components, np.float32)
        keys_dtype = np.float32 if self._keys_cpu_dtype_mode == "float32" else None
        keys_np = self._copy_table_array_from_desc(desc.keys, keys_dtype)
        token_buffer = self._copy_table_array_from_desc(desc.token_buffer, np.int32)
        rollout_offsets = self._copy_table_array_from_desc(
            desc.rollout_token_offset, np.int64)
        rollout_lens = self._copy_table_array_from_desc(desc.rollout_token_len, np.int32)
        entry_rollout_idx = self._copy_table_array_from_desc(
            desc.entry_rollout_idx, np.int32)
        entry_offset = self._copy_table_array_from_desc(desc.entry_offset, np.int32)
        t1_materialize = _now_ns()

        if mean_np.shape != (hidden_dim,):
            raise ValueError(f"HSpec descriptor mean shape mismatch: {mean_np.shape}")
        if components_np.shape != (n_components, hidden_dim):
            raise ValueError(
                f"HSpec descriptor components shape mismatch: {components_np.shape}"
            )
        if keys_np.shape != (n_entries, n_components):
            raise ValueError(f"HSpec descriptor keys shape mismatch: {keys_np.shape}")
        if entry_rollout_idx.shape[0] < n_entries or entry_offset.shape[0] < n_entries:
            raise ValueError("HSpec descriptor entry arrays shorter than n_entries")
        if rollout_offsets.shape[0] < n_rollouts or rollout_lens.shape[0] < n_rollouts:
            raise ValueError("HSpec descriptor rollout arrays shorter than n_rollouts")

        rollout_offsets = np.ascontiguousarray(rollout_offsets[:n_rollouts], dtype=np.int64)
        rollout_lens = np.ascontiguousarray(rollout_lens[:n_rollouts], dtype=np.int32)
        entry_rollout_idx = np.ascontiguousarray(
            entry_rollout_idx[:n_entries], dtype=np.int32)
        entry_offset = np.ascontiguousarray(entry_offset[:n_entries], dtype=np.int32)

        t0_prefix = _now_ns()
        draft_prefix_tokens = None
        draft_prefix_lens = None
        if self._prefix_cache_enabled:
            draft_prefix_tokens, draft_prefix_lens = self._build_draft_prefix_from_token_refs(
                token_buffer,
                rollout_offsets,
                rollout_lens,
                entry_rollout_idx,
                entry_offset,
                n_entries,
                int(desc.max_wnd),
            )
        t1_prefix = _now_ns()

        cached = self._finalize_cached_table(
            mean_np=mean_np,
            components_np=components_np,
            keys_np=keys_np,
            token_buffer=token_buffer,
            rollout_token_offset=rollout_offsets,
            rollout_token_len=rollout_lens,
            entry_rollout_idx=entry_rollout_idx,
            entry_offset=entry_offset,
            n_entries=n_entries,
            wnd_size=int(desc.wnd_size),
            max_wnd=int(desc.max_wnd),
            min_wnd=int(desc.min_wnd),
            prompt_id=str(prompt_id or desc.prompt_id),
            draft_prefix_tokens=draft_prefix_tokens,
            draft_prefix_lens=draft_prefix_lens,
            metric_prefix="descriptor",
        )
        self._record_proposer_metric(
            "descriptor_materialize_ms", _ns_to_ms(t1_materialize - t0_materialize))
        if self._prefix_cache_enabled:
            self._record_proposer_metric(
                "descriptor_prefix_ms", _ns_to_ms(t1_prefix - t0_prefix))
        self._record_proposer_metric(
            "descriptor_bytes",
            mean_np.nbytes
            + components_np.nbytes
            + keys_np.nbytes
            + token_buffer.nbytes
            + rollout_offsets.nbytes
            + rollout_lens.nbytes
            + entry_rollout_idx.nbytes
            + entry_offset.nbytes,
        )
        self._record_proposer_metric("descriptor_entries", n_entries)
        return cached

    def _build_cached_table(
        self,
        data: dict,
        prompt_id: Optional[str] = None,
    ) -> _CachedPromptTable:
        """Convert serialised table data dict → on-device cached table."""
        self._guard_no_hot_path_materialize("build_cached_table_legacy")
        # NOTE: Ray may deserialize numpy arrays as non-writable (read-only)
        # views; torch.from_numpy warns about undefined behaviour on write.
        # These tensors are read-only in our usage, but we still copy here to
        # silence warnings and keep behaviour well-defined. This is *prefetch*
        # (not in the hot loop).
        mean_np = np.array(data["mean"], dtype=np.float32, copy=True)
        components_np = np.array(data["components"], dtype=np.float32, copy=True)
        keys_dtype = np.float32 if self._keys_cpu_dtype_mode == "float32" else None
        keys_np = np.array(data["keys"], dtype=keys_dtype, copy=True)

        # Ensure rollout_seqs are numpy arrays on CPU
        rollout_seqs = []
        for s in data["rollout_seqs"]:
            if isinstance(s, np.ndarray):
                rollout_seqs.append(np.asarray(s, dtype=np.int32))
            else:
                rollout_seqs.append(np.asarray(s, dtype=np.int32))
        token_buffer, rollout_offsets, rollout_lens = self._flatten_rollout_seqs(
            rollout_seqs)

        entry_rollout_idx = np.asarray(data["entry_rollout_idx"], dtype=np.int32)
        n_entries = int(data["n_entries"])

        max_wnd = int(data.get("max_wnd", 28))
        min_wnd = int(data.get("min_wnd", 2))
        wnd_size = int(data.get("wnd_size", getattr(self, "_default_wnd_size", 8)))
        entry_offset = np.asarray(data["entry_offset"], dtype=np.int32)

        return self._finalize_cached_table(
            mean_np=mean_np,
            components_np=components_np,
            keys_np=keys_np,
            token_buffer=token_buffer,
            rollout_token_offset=rollout_offsets,
            rollout_token_len=rollout_lens,
            entry_rollout_idx=entry_rollout_idx,
            entry_offset=entry_offset,
            n_entries=n_entries,
            wnd_size=wnd_size,
            max_wnd=max_wnd,
            min_wnd=min_wnd,
            prompt_id=prompt_id,
        )

    @staticmethod
    def _window_base_pos(decoded_len: int) -> int:
        # After accepting `decoded_len` response tokens, the next query uses the
        # anchor at local key position `decoded_len - 1`.
        return max(int(decoded_len) - 1, 0)

    def _apply_abs_delta_cap(
        self,
        window: int,
        abs_delta: int,
        min_wnd: int,
    ) -> int:
        """Apply a cheap piecewise cap using matched-position distance.

        This runs only after the best entry has been selected, so it does not
        affect batched similarity matching.
        """
        wnd = int(window)
        if not self._abs_delta_cap_enabled:
            return wnd
        abs_delta = int(abs_delta)
        if abs_delta <= int(self._abs_delta_safe_threshold):
            return wnd
        if abs_delta <= int(self._abs_delta_mid_threshold):
            cap = max(int(min_wnd), min(int(self._abs_delta_mid_cap), int(self.max_draft_tokens)))
        else:
            cap = max(int(min_wnd), min(int(self._abs_delta_far_cap), int(self.max_draft_tokens)))
        return min(wnd, int(cap))

    def _build_batched_table_tensors(
        self,
        cached_tables: List[_CachedPromptTable],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack ragged per-prompt tables into padded batch tensors.

        Returns:
            mean_batch:         (B, D)
            components_t_batch: (B, D, K_max)  pre-transposed for projection
            keys_batch:         (B, M_max, K_max)
            key_lengths:        (B,)
            invalid_key_mask:   (B, M_max)
        """
        prof_enabled = hspec_profile_context_enabled()
        num_rows = len(cached_tables)
        if num_rows == 0:
            empty = torch.empty((0,), dtype=dtype, device=device)
            empty_long = torch.empty((0,), dtype=torch.long, device=device)
            empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
            return empty, empty, empty, empty_long, empty_bool

        k_max = max(int(cached.components_t_cpu.shape[1]) for cached in cached_tables)
        m_max = max(int(cached.n_entries) for cached in cached_tables)
        hidden_dim = int(cached_tables[0].mean_cpu.shape[0])
        # Rebuild on CPU to avoid many tiny NPU slice/copy kernels, then upload
        with (hspec_record_function("hspec/proposal/rebuild_on_cpu")
              if prof_enabled else nullcontext()):
            mean_batch_cpu = np.stack([cached.mean_cpu for cached in cached_tables], axis=0)
            components_t_batch_cpu = np.zeros((num_rows, hidden_dim, k_max), dtype=np.float32)
            keys_batch_cpu = np.zeros((num_rows, m_max, k_max), dtype=np.float32)
            key_lengths_cpu = np.empty((num_rows,), dtype=np.int64)

        with (hspec_record_function("hspec/proposal/rebuild_components_keys_mask")
              if prof_enabled else nullcontext()):
            total_elems = (num_rows * hidden_dim * k_max) + (num_rows * m_max * k_max)
            use_numba = (
                self._use_numba_rebuild
                and num_rows >= self._numba_rebuild_min_rows
                and total_elems >= self._numba_rebuild_min_elems
            )
            if use_numba:
                comp_list = _hspec_make_numba_array_list(
                    [cached.components_t_cpu for cached in cached_tables])
                keys_list = _hspec_make_numba_array_list(
                    [cached.keys_cpu[:cached.n_entries] for cached in cached_tables])
                if comp_list is not None and keys_list is not None:
                    _hspec_fill_batched_components_keys_numba(
                        comp_list,
                        keys_list,
                        components_t_batch_cpu,
                        keys_batch_cpu,
                        key_lengths_cpu,
                    )
                else:
                    use_numba = False
            if not use_numba:
                for row, cached in enumerate(cached_tables):
                    k_i = int(cached.components_t_cpu.shape[1])
                    m_i = int(cached.n_entries)
                    components_t_batch_cpu[row, :, :k_i] = cached.components_t_cpu[:, :k_i]
                    if m_i > 0:
                        keys_batch_cpu[row, :m_i, :k_i] = cached.keys_cpu[:m_i, :k_i]
                    key_lengths_cpu[row] = m_i

            invalid_key_mask_cpu = np.arange(m_max, dtype=np.int64)[None, :] >= key_lengths_cpu[:, None]

        with (hspec_record_function("hspec/proposal/convert_to_npu")
              if prof_enabled else nullcontext()):
            mean_batch = torch.from_numpy(mean_batch_cpu).to(device=device, dtype=dtype, non_blocking=True)
            components_t_batch = torch.from_numpy(components_t_batch_cpu).to(
                device=device, dtype=dtype, non_blocking=True)
            keys_batch = torch.from_numpy(keys_batch_cpu).to(
                device=device, dtype=self._keys_device_dtype, non_blocking=True)
            key_lengths = torch.from_numpy(key_lengths_cpu).to(
                device=device, dtype=torch.long, non_blocking=True)
            invalid_key_mask = torch.from_numpy(invalid_key_mask_cpu).to(
                device=device, dtype=torch.bool, non_blocking=True)

        return mean_batch, components_t_batch, keys_batch, key_lengths, invalid_key_mask

    def _match_projected_batch(
        self,
        z_batch: torch.Tensor,
        keys_batch: torch.Tensor,
        invalid_key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fully batched similarity matching with precomputed key padding mask."""
        if keys_batch.dtype != z_batch.dtype:
            z_batch = z_batch.to(dtype=keys_batch.dtype)
        sims = torch.bmm(keys_batch, z_batch.unsqueeze(-1)).squeeze(-1)
        if invalid_key_mask.numel() > 0:
            sims = sims.masked_fill(invalid_key_mask, torch.finfo(sims.dtype).min)
        return sims.max(dim=1)

    @staticmethod
    def _has_same_histo_ngram(
        current_tokens: List[int],
        matched_seq: np.ndarray,
        matched_pos: int,
    ) -> bool:
        """Whether HistoSpec could have matched using the same exact n-gram."""
        n = int(HSPEC_ADVAN_NGRAM)
        if n <= 0 or len(current_tokens) < n:
            return False

        hist_end = int(matched_pos) + 1
        if hist_end < n:
            return False

        current_ngram = [int(x) for x in current_tokens[-n:]]
        hist_ngram = matched_seq[hist_end - n:hist_end].tolist()
        return current_ngram == [int(x) for x in hist_ngram]

    # anchor hidden-state extraction

    @staticmethod
    def _compute_anchor_indices(
        sample_hidden_states: torch.Tensor,
        valid_sampled_token_ids: List[List[int]],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
    ) -> List[int]:
        """Compute per-request anchor indices on CPU.

        Small control-flow-heavy logic is cheaper on CPU than as many tiny NPU
        tensor ops. The caller can then gather all needed rows once with
        `index_select` on device.
        """
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return []

        max_hidden_rows = int(sample_hidden_states.shape[0])
        accept_lens = [len(sampled_ids) for sampled_ids in valid_sampled_token_ids]
        anchor_indices = [-1] * batch_size

        if spec_decode_metadata is None:
            for i, accept_len in enumerate(accept_lens):
                if accept_len > 0 and i < max_hidden_rows:
                    anchor_indices[i] = i
            return anchor_indices

        num_drafts = spec_decode_metadata.num_draft_tokens[:batch_size]
        bonus_indices = spec_decode_metadata.bonus_logits_indices[:batch_size].tolist()
        for i, accept_len in enumerate(accept_lens):
            if accept_len <= 0:
                continue
            bonus_idx = int(bonus_indices[i])
            n_draft = int(num_drafts[i])
            if n_draft == 0 or accept_len >= n_draft + 1:
                anchor_idx = bonus_idx
            else:
                anchor_idx = bonus_idx - n_draft + accept_len - 1
            if 0 <= anchor_idx < max_hidden_rows:
                anchor_indices[i] = anchor_idx
        return anchor_indices

    # main interface
    
    def generate_token_ids(
        self,
        valid_sampled_token_ids: List[List[int]],
        sampling_metadata: SamplingMetadata = None,
        scheduler_output: SchedulerOutput = None,
        spec_decode_metadata: SpecDecodeMetadata = None,
        positions: torch.Tensor = None,
        num_scheduled_tokens: int = 0,
        hidden_states: torch.Tensor = None,
        attn_metadata=None,
        aux_hidden_states: torch.Tensor = None,
    ) -> List[List[int]]:
        self._in_generate_token_ids = True
        try:
            return self._generate_token_ids_impl(
                valid_sampled_token_ids=valid_sampled_token_ids,
                sampling_metadata=sampling_metadata,
                scheduler_output=scheduler_output,
                spec_decode_metadata=spec_decode_metadata,
                positions=positions,
                num_scheduled_tokens=num_scheduled_tokens,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                aux_hidden_states=aux_hidden_states,
            )
        finally:
            self._in_generate_token_ids = False

    def _generate_token_ids_impl(
        self,
        valid_sampled_token_ids: List[List[int]],
        sampling_metadata: SamplingMetadata = None,
        scheduler_output: SchedulerOutput = None,
        spec_decode_metadata: SpecDecodeMetadata = None,
        positions: torch.Tensor = None,
        num_scheduled_tokens: int = 0,
        hidden_states: torch.Tensor = None,
        attn_metadata=None,
        aux_hidden_states: torch.Tensor = None,
    ) -> List[List[int]]:
        """Generate draft tokens via on-device hidden-state matching.

        Called by ``model_runner.propose_draft_token_ids()`` after the
        target model's forward pass.

        ``hidden_states`` should be **sample_hidden_states** (already
        indexed at logits positions) for correct per-request extraction.

        **Hot-loop invariant:** zero Ray / ZMQ / network calls in the
        steady state.  The only CPU work is a tiny scalar transfer
        ``(P, 2)`` and sub-µs numpy slices for draft tokens.
        """
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return []
        if hidden_states is None:
            return [[] for _ in range(batch_size)]
        input_batch = self.runner.input_batch

        gen_enabled = HSPEC_GEN
        gen_req_idx = HSPEC_GEN_REQ_IDX
        if gen_enabled and HSPEC_GEN_MAX_CALLS > 0 and getattr(self, "_stat_calls", 0) >= HSPEC_GEN_MAX_CALLS:
            gen_enabled = False
        prof_enabled = hspec_profile_context_enabled()

        # 1. Stable prompt_id + batch anchor hidden states
        req_ids = list(input_batch.req_ids[:batch_size])
        req_states = [self.runner.requests.get(req_id) for req_id in req_ids]

        t0_pid = _now_ns() if gen_enabled else 0
        with (hspec_record_function("hspec/proposal/prompt_id") if prof_enabled else nullcontext()):
            prompt_ids = self._get_prompt_ids_for_batch(req_ids)
        t1_pid = _now_ns() if gen_enabled else 0

        decoded_lens = [
            len(getattr(req_state, "output_token_ids", [])) if req_state is not None else 0
            for req_state in req_states
        ]

        t0_extract = _now_ns() if gen_enabled else 0
        with (hspec_record_function("hspec/proposal/extract_anchor_hs")
              if prof_enabled else nullcontext()):
            anchor_indices = self._compute_anchor_indices(hidden_states,
                                                          valid_sampled_token_ids,
                                                          spec_decode_metadata)
        t1_extract = _now_ns() if gen_enabled else 0

        trace_anchor = None
        if batch_size > 0:
            di = min(gen_req_idx if gen_enabled else HSPEC_DEBUG_REQ_IDX, batch_size - 1)
            anchor_idx = anchor_indices[di] if di < len(anchor_indices) else -1
            if 0 <= anchor_idx < hidden_states.shape[0]:
                trace_anchor = hidden_states[anchor_idx]

        if gen_enabled:
            self._hspec_gen_timing = {
                "prompt_id_ms": _ns_to_ms(t1_pid - t0_pid) if t0_pid else 0.0,
                "extract_anchor_hs_ms": _ns_to_ms(t1_extract - t0_extract) if t0_extract else 0.0,
                "anchor_total_ms": _ns_to_ms(t1_extract - t0_pid) if t0_pid else 0.0,
            }

        if HSPEC_DEBUG:
            # (1)(2) One prompt per step: only the chosen request
            # Log correlation fields (req_id, prompt_id, decoded_len, decoded_tokens)
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                req_id = req_ids[di]
                req_state = req_states[di]
                if req_state is None:
                    decoded_tokens = []
                    prompt_tokens = []
                else:
                    decoded_tokens = list(getattr(req_state, "output_token_ids", []))
                    prompt_tokens = list(getattr(req_state, "prompt_token_ids", []))
                decoded_len = len(decoded_tokens)
                # Correlation id: same (req_id, prompt_id, decoded_len) in STEP_BEGIN and CACHE_MATCH
                corr_id = f"req_id={req_id!r} prompt_id={prompt_ids[di]!r} decoded_len={decoded_len}"
                anchor_hs_summary = "anchor_hs=None"
                if trace_anchor is not None:
                    hs = trace_anchor
                    try:
                        norm = float(hs.float().norm().item())
                    except Exception:
                        norm = float("nan")
                    anchor_hs_summary = (
                        f"anchor_hs.shape={tuple(hs.shape)} norm={norm:.6f} device={hs.device}"
                    )
                # Detokenize prompt_tokens + decoded_tokens
                tokenizer = _get_tokenizer_safe(self.runner)
                prompt_decoded_text = _detokenize_safe(tokenizer, prompt_tokens + decoded_tokens)
                logger.info(
                    "------------------------------------------- generate_token_ids() STEP_BEGIN -------------------------------------------\n"
                    "HSPEC DEBUG STEP_BEGIN [req_idx=%d] %s | "
                    "decoded_tokens=%s | accepted_step_tokens=%s | "
                    "hidden_states.shape=%s dtype=%s device=%s | prompt_tokens=%s | "
                    "prompt+decoded_text=%r | %s",
                    di,
                    corr_id,
                    decoded_tokens,
                    valid_sampled_token_ids[di] if di < len(valid_sampled_token_ids) else [],
                    tuple(hidden_states.shape),
                    str(hidden_states.dtype),
                    hidden_states.device,
                    prompt_tokens,
                    prompt_decoded_text,
                    anchor_hs_summary,
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log prompt/hidden_state info")

        # 2. Keep the proposal hot path free of mmap/H2D work.
        # prefetch_for_batch() already consumes ready futures before forward.
        # This no-materialize poll is intentionally a cheap guard so a ready
        # descriptor never gets ray.get'ed and mmap'ed in generate_token_ids().
        t0_poll = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/poll_pending"):
            self._poll_pending(materialize_ready=False)
        t1_poll = _now_ns() if gen_enabled else 0

        t0_fire = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/fire_prefetch_async"):
            self._fire_prefetch_async(prompt_ids)
        t1_fire = _now_ns() if gen_enabled else 0

        # 3. On-device projection + similarity matching
        results: List[List[int]] = [[] for _ in range(batch_size)]
        trace_pending_j: Optional[int] = None
        trace_skip_reason: Optional[str] = None
        active_batch_indices: List[int] = []
        active_table_rows: List[int] = []
        active_cached_tables: List[_CachedPromptTable] = []
        active_base_positions: List[int] = []
        batch_table_cache = self._get_prebuilt_batched_table_cache(
            req_ids,
            prompt_ids,
        )
        if batch_table_cache is None:
            if self._allow_hot_batch_cache_build:
                batch_table_cache = self._get_or_build_batched_table_cache(
                    req_ids,
                    prompt_ids,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            else:
                self._record_proposer_metric("batch_cache_miss_no_hot_build", 1)
        if batch_table_cache is None and gen_enabled and batch_size > 0:
            trace_skip_reason = "batch_cache_not_prebuilt"

        with (hspec_record_function("hspec/proposal/get_active_batch_indices_and_rows", use_npu_stream=True)
              if prof_enabled else nullcontext()):
            if batch_table_cache is not None:
                for i in range(batch_size):
                    anchor_idx = anchor_indices[i] if i < len(anchor_indices) else -1
                    if anchor_idx < 0:
                        if gen_enabled and i == gen_req_idx:
                            trace_skip_reason = "anchor_none"
                        continue
                    row = batch_table_cache.batch_idx_to_row.get(i)
                    if row is None:
                        if gen_enabled and i == gen_req_idx:
                            trace_skip_reason = "prompt_not_cached"
                        continue
                    if len(valid_sampled_token_ids[i]) < self.min_match_len:
                        if gen_enabled and i == gen_req_idx:
                            trace_skip_reason = f"below_min_match_len:{len(valid_sampled_token_ids[i])}"
                        continue
                    active_batch_indices.append(i)
                    active_table_rows.append(row)

        with (hspec_record_function("hspec/proposal/get_active_cached_tables_and_position", use_npu_stream=True)
              if prof_enabled else nullcontext()):
            if batch_table_cache is not None and active_table_rows:
                active_cached_tables = [batch_table_cache.cached_tables[row] for row in active_table_rows]
                active_base_positions = [
                    self._window_base_pos(decoded_lens[batch_idx]) for batch_idx in active_batch_indices
                ]

        if active_batch_indices and batch_table_cache is not None:
            self._stat_queries += len(active_batch_indices)
            for batch_idx in active_batch_indices:
                if batch_idx < len(prompt_ids):
                    self._record_prompt_access(prompt_ids[batch_idx], query=1)
            t0_cast = _now_ns() if gen_enabled else 0
            with (hspec_record_function("hspec/proposal/anchor_gather", use_npu_stream=True)
                  if prof_enabled else nullcontext()):
                gather_idx = torch.tensor(
                    [anchor_indices[i] for i in active_batch_indices],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                active_anchor_hs = hidden_states.index_select(0, gather_idx).float()
                t1_cast = _now_ns() if gen_enabled else 0

            with (hspec_record_function("hspec/proposal/project", use_npu_stream=True)
                  if prof_enabled else nullcontext()):
                t0_proj = _now_ns() if gen_enabled else 0
                use_full_cached_batch = (
                    len(active_table_rows) == len(batch_table_cache.cached_tables)
                    and all(row == idx for idx, row in enumerate(active_table_rows))
                )
                if use_full_cached_batch:
                    mean_batch = batch_table_cache.mean_batch
                    components_t_batch = batch_table_cache.components_t_batch
                else:
                    table_rows = torch.tensor(active_table_rows,
                                              dtype=torch.long,
                                              device=hidden_states.device)
                    mean_batch = batch_table_cache.mean_batch.index_select(0, table_rows)
                    components_t_batch = batch_table_cache.components_t_batch.index_select(0, table_rows)
                z_batch = torch.bmm(
                    (active_anchor_hs - mean_batch).unsqueeze(1),
                    components_t_batch,
                ).squeeze(1)
                t1_proj = _now_ns() if gen_enabled else 0

            with (hspec_record_function("hspec/proposal/match", use_npu_stream=True)
                  if prof_enabled else nullcontext()):
                t0_sim = _now_ns() if gen_enabled else 0
                if use_full_cached_batch:
                    keys_batch = batch_table_cache.keys_batch
                    invalid_key_mask = batch_table_cache.invalid_key_mask
                else:
                    keys_batch = batch_table_cache.keys_batch.index_select(0, table_rows)
                    invalid_key_mask = batch_table_cache.invalid_key_mask.index_select(0, table_rows)
                best_sims, best_idxs = self._match_projected_batch(z_batch, keys_batch, invalid_key_mask)
                t1_sim = _now_ns() if gen_enabled else 0

            if gen_enabled:
                for row, batch_idx in enumerate(active_batch_indices):
                    if batch_idx == gen_req_idx:
                        trace_pending_j = row
                        break
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                td.update({
                    "cast_fp32_ms": _ns_to_ms(t1_cast - t0_cast) if t0_cast else 0.0,
                    "pca_project_ms": _ns_to_ms(t1_proj - t0_proj) if t0_proj else 0.0,
                    "similarity_ms": _ns_to_ms(t1_sim - t0_sim) if t0_sim else 0.0,
                })
                self._hspec_gen_timing = td
        else:
            best_sims = None
            best_idxs = None

        if HSPEC_DEBUG:
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                req_id = req_ids[di]
                req_state = req_states[di]
                decoded_tokens_at_match = []
                if req_state is not None:
                    decoded_tokens_at_match = list(getattr(req_state, "output_token_ids", []))
                decoded_len_at_match = len(decoded_tokens_at_match)
                corr_id = f"req_id={req_id!r} prompt_id={prompt_ids[di]!r} decoded_len={decoded_len_at_match}"

                # Pending list: only the chosen request if it is in pending
                pending_line = None
                cached_for_debug = None
                best_idx_val = -1
                sim_val = float("nan")
                if best_sims is not None and best_idxs is not None:
                    for row, batch_idx in enumerate(active_batch_indices):
                        if batch_idx != di:
                            continue
                        cached = active_cached_tables[row]
                        cached_for_debug = cached
                        try:
                            sim_val = float(best_sims[row].detach().float().item())
                        except Exception:
                            sim_val = float("nan")
                        try:
                            best_idx_val = int(best_idxs[row].detach().item())
                        except Exception:
                            best_idx_val = -1
                        pending_line = (
                            f"CACHE_MATCH [req_idx={di}] {corr_id} | "
                            f"decoded_tokens_at_match={decoded_tokens_at_match} | "
                            f"best_idx={best_idx_val} sim={sim_val:.6f} n_entries={cached.n_entries}"
                        )
                        break
                if pending_line is None:
                    pending_line = (
                        f"CACHE_MATCH [req_idx={di}] {corr_id} | "
                        f"decoded_tokens_at_match={decoded_tokens_at_match} | "
                        f"not_in_pending (pending_size={len(active_batch_indices)})"
                    )
                else:
                    # (1) Print cache table contents (all entries) with correlation to decoded state
                    if cached_for_debug is not None:
                        table_info_lines = []
                        n_entries = cached_for_debug.n_entries
                        table_info_lines.append(
                            f"cache_table {corr_id} | decoded_tokens_at_match={decoded_tokens_at_match} | "
                            f"n_entries={n_entries} wnd_size={cached_for_debug.wnd_size} "
                            f"mean.shape={tuple(cached_for_debug.mean_cpu.shape)} "
                            f"components_t.shape={tuple(cached_for_debug.components_t_cpu.shape)} "
                            f"keys_cpu.shape={tuple(cached_for_debug.keys_cpu.shape)} "
                            f"per_prompt_npu={cached_for_debug.keys is not None} | "
                            f"best_hit_entry={best_idx_val} → draft_tokens_for_next_step (see entry below)"
                        )
                        tokenizer = _get_tokenizer_safe(self.runner)
                        for entry_idx in range(n_entries):
                            ridx = int(cached_for_debug.entry_rollout_idx[entry_idx])
                            off = int(cached_for_debug.entry_offset[entry_idx])
                            draft_tokens = cached_for_debug.get_draft_tokens(
                                entry_idx, cached_for_debug.wnd_size)
                            # Detokenize rollout sequence
                            rollout_text = _detokenize_safe(tokenizer, draft_tokens)
                            # Get similarity for this entry (if keys available)
                            try:
                                # Use the projected anchor_hs to compute similarity
                                if trace_anchor is not None:
                                    hs_np = trace_anchor.detach().float().cpu().numpy()
                                    z_np = (
                                        (hs_np - cached_for_debug.mean_cpu)
                                        @ cached_for_debug.components_t_cpu
                                    )
                                    key_np = cached_for_debug.keys_cpu[entry_idx].astype(
                                        np.float32, copy=False)
                                    entry_sim = float(
                                        key_np.dot(z_np.astype(np.float32, copy=False)))
                                else:
                                    entry_sim = float("nan")
                            except Exception:
                                entry_sim = float("nan")
                            table_info_lines.append(
                                f"  entry[{entry_idx}] rollout_idx={ridx} offset={off} "
                                f"sim={entry_sim:.4f} | "
                                f"decoded_tokens_at_match={decoded_tokens_at_match} → draft_tokens={draft_tokens} "
                                f"rollout_text={rollout_text!r}"
                            )
                        pending_line += "\n" + "\n".join(table_info_lines)
                logger.info(
                    "------------------------------------------- generate_token_ids() CACHE_MATCH -------------------------------------------\n"
                    "HSPEC DEBUG %s",
                    pending_line,
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log pending list")

        if not active_batch_indices or best_sims is None or best_idxs is None:
            if gen_enabled and batch_size > 0:
                di = min(gen_req_idx, batch_size - 1)
                try:
                    req_id = req_ids[di]
                    req_state = req_states[di]
                    decoded_len = len(getattr(req_state, "output_token_ids", [])) if req_state is not None else -1
                    pid = prompt_ids[di] if di < len(prompt_ids) else ""
                    td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                    td.update({
                        "poll_pending_ms": _ns_to_ms(t1_poll - t0_poll) if t0_poll else 0.0,
                        "fire_prefetch_async_ms": _ns_to_ms(t1_fire - t0_fire) if t0_fire else 0.0,
                    })
                    self._hspec_gen_timing = td
                    logger.warning(
                        "HSPEC GEN proposer_breakdown [req_idx=%d] req_id=%s prompt_id=%s "
                        "decoded_len=%d status=no_pending skip_reason=%s timing_ms=%s",
                        int(di),
                        str(req_id),
                        str(pid),
                        int(decoded_len),
                        str(trace_skip_reason),
                        td,
                    )
                except Exception:
                    pass
            if HSPEC_DEBUG:
                logger.info("HSPEC DEBUG generate_token_ids(): no pending entries, all requests miss cache/anchor")
            return results

        # 4. Single device → host sync for the whole batch
        t0_stack = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/device_to_host_sync", use_npu_stream=True):
            t1_stack = _now_ns() if gen_enabled else 0
            t0_copy = _now_ns() if gen_enabled else 0
            sims_cpu = best_sims.cpu().numpy()
            idxs_cpu = best_idxs.cpu().numpy()
            t1_copy = _now_ns() if gen_enabled else 0

        # 5. Draft token retrieval (CPU-only, O(1) per request)
        hit_rows = np.flatnonzero(sims_cpu >= self.similarity_threshold)
        pending = []
        for j in hit_rows.tolist():
            i = active_batch_indices[j]
            req_id = req_ids[i]
            cached = active_cached_tables[j]
            base_pos = active_base_positions[j]
            matched_entry_idx = int(idxs_cpu[j])
            matched_pos = int(cached.entry_offset[matched_entry_idx]) - 1
            delta = int(matched_pos - int(base_pos))
            abs_delta = abs(delta)
            base_effective_wnd = min(
                cached.get_effective_window(matched_entry_idx),
                int(self.max_draft_tokens),
            )
            effective_wnd = self._apply_abs_delta_cap(
                window=base_effective_wnd,
                abs_delta=abs_delta,
                min_wnd=int(cached.min_wnd),
            )
            t0_retrieve = _now_ns() if (gen_enabled and i == gen_req_idx) else 0

            # Draft tokens from CPU cache (sub-µs numpy slice)
            prof_this_req = prof_enabled
            with (hspec_record_function("hspec/proposal/draft_retrieve")
                  if prof_this_req else nullcontext()):
                draft = cached.get_draft_tokens(matched_entry_idx, effective_wnd)
            results[i] = draft
            self._stat_hits += 1
            self._stat_total_draft_len += len(draft)
            if i < len(prompt_ids):
                self._record_prompt_access(prompt_ids[i], hit=1)
            pending.append((i, sims_cpu[j], idxs_cpu[j], cached, base_pos, effective_wnd))

            if draft:
                matched_rollout_idx = int(cached.entry_rollout_idx[matched_entry_idx])
                req_state = req_states[i]
                current_tokens = list(getattr(req_state, "output_token_ids", [])) if req_state is not None else []
                entry_bias, entry_hits = cached.get_entry_state(matched_entry_idx)
                histo_ngram_match = self._has_same_histo_ngram(
                    current_tokens,
                    cached.get_rollout_tokens(matched_rollout_idx),
                    matched_pos,
                )
                self._entry_pending_match_count += 1
                self._entry_pending_delta_sum += delta
                self._entry_pending_abs_delta_sum += abs_delta
                self._pending_verify_meta[req_id] = {
                    "prompt_id": prompt_ids[i],
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "base_pos": int(base_pos),
                    "matched_pos": matched_pos,
                    "matched_rollout_idx": matched_rollout_idx,
                    "histo_ngram_match": int(histo_ngram_match),
                    "matched_entry_idx": matched_entry_idx,
                    "drafted_len": int(len(draft)),
                    "wnd_size_at_match": int(cached.wnd_size),
                    "base_effective_wnd_at_match": int(base_effective_wnd),
                    "effective_wnd_at_match": int(effective_wnd),
                    "abs_delta_cap_applied": int(effective_wnd < base_effective_wnd),
                    "entry_bias_at_match": int(entry_bias),
                    "entry_hits_at_match": int(entry_hits),
                    "min_wnd": int(cached.min_wnd),
                    "max_wnd": int(cached.max_wnd),
                }

            t1_retrieve = _now_ns() if (gen_enabled and i == gen_req_idx) else 0
            if gen_enabled and i == gen_req_idx:
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                td.update({
                    "poll_pending_ms": _ns_to_ms(t1_poll - t0_poll) if t0_poll else 0.0,
                    "fire_prefetch_async_ms": _ns_to_ms(t1_fire - t0_fire) if t0_fire else 0.0,
                    "stack_scalars_ms": _ns_to_ms(t1_stack - t0_stack) if t0_stack else 0.0,
                    "device_to_host_ms": _ns_to_ms(t1_copy - t0_copy) if t0_copy else 0.0,
                    "draft_retrieve_ms": _ns_to_ms(t1_retrieve - t0_retrieve) if t0_retrieve else 0.0,
                })
                self._hspec_gen_timing = td

        self._maybe_log_metrics()

        # HSPEC_GEN: per-token breakdown log for one traced request only.
        # We intentionally log at WARNING to make sure it lands in the main
        # training output when log levels filter INFO.
        if gen_enabled and batch_size > 0:
            di = min(gen_req_idx, batch_size - 1)
            try:
                req_id = req_ids[di]
                req_state = req_states[di]
                decoded_len = len(getattr(req_state, "output_token_ids", [])) if req_state is not None else -1
                pid = prompt_ids[di] if di < len(prompt_ids) else ""
                anchor = trace_anchor if di == min(gen_req_idx, batch_size - 1) else None
                anchor_norm = None
                if anchor is not None:
                    try:
                        anchor_norm = float(anchor.float().norm().item())
                    except Exception:
                        anchor_norm = None

                # Similarity info (if this request is in pending)
                sim_val = None
                best_idx_val = None
                n_entries = None
                wnd_size = None
                prompt_wnd = None
                base_eff_wnd = None
                entry_bias = None
                entry_hits = None
                abs_delta = None
                if trace_pending_j is not None:
                    # Find the pending slot for this batch index
                    if trace_pending_j < len(pending) and pending[trace_pending_j][0] == di:
                        sim_val = float(sims_cpu[trace_pending_j])
                        best_idx_val = int(idxs_cpu[trace_pending_j])
                        n_entries = int(pending[trace_pending_j][3].n_entries)
                        prompt_wnd = int(pending[trace_pending_j][3].wnd_size)
                        wnd_size = int(pending[trace_pending_j][5])
                        entry_bias, entry_hits = pending[trace_pending_j][3].get_entry_state(best_idx_val)
                        meta = self._pending_verify_meta.get(req_id)
                        if meta is not None:
                            base_eff_wnd = int(meta.get("base_effective_wnd_at_match", wnd_size))
                            abs_delta = int(meta.get("abs_delta", 0))

                draft = results[di] if di < len(results) else []
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                logger.warning(
                    "HSPEC GEN proposer_breakdown [req_idx=%d] req_id=%s prompt_id=%s decoded_len=%d "
                    "anchor_norm=%s sim=%s best_idx=%s n_entries=%s prompt_wnd=%s eff_wnd=%s "
                    "base_eff_wnd=%s abs_delta=%s entry_bias=%s entry_hits=%s draft_len=%d "
                    "draft=%s timing_ms=%s",
                    int(di),
                    str(req_id),
                    str(pid),
                    int(decoded_len),
                    "None" if anchor_norm is None else f"{anchor_norm:.6f}",
                    "None" if sim_val is None else f"{sim_val:.6f}",
                    "None" if best_idx_val is None else str(best_idx_val),
                    "None" if n_entries is None else str(n_entries),
                    "None" if prompt_wnd is None else str(prompt_wnd),
                    "None" if wnd_size is None else str(wnd_size),
                    "None" if base_eff_wnd is None else str(base_eff_wnd),
                    "None" if abs_delta is None else str(abs_delta),
                    "None" if entry_bias is None else str(entry_bias),
                    "None" if entry_hits is None else str(entry_hits),
                    int(len(draft)),
                    list(draft),
                    td,
                )
            except Exception:
                pass

        if HSPEC_DEBUG:
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                # Matched + draft for chosen request only
                matched_line = None
                for j, (i, _, _, cached, _, eff_wnd) in enumerate(pending):
                    if i != di or sims_cpu[j] < self.similarity_threshold:
                        continue
                    entry_bias, entry_hits = cached.get_entry_state(int(idxs_cpu[j]))
                    meta = self._pending_verify_meta.get(req_ids[di], {})
                    matched_line = (
                        f"matched [req_idx={di}] prompt_id={prompt_ids[di]!r} "
                        f"sim={float(sims_cpu[j]):.4f} best_idx={int(idxs_cpu[j])} "
                        f"prompt_wnd={cached.wnd_size} eff_wnd={eff_wnd} "
                        f"base_eff_wnd={meta.get('base_effective_wnd_at_match', eff_wnd)} "
                        f"abs_delta={meta.get('abs_delta', 0)} "
                        f"entry_bias={entry_bias} entry_hits={entry_hits} "
                        f"draft_tokens={list(results[di])}"
                    )
                    break
                if matched_line is None:
                    matched_line = (
                        f"matched: req_idx={di} not above threshold "
                        f"(threshold={self.similarity_threshold:.4f}) or not in pending"
                    )
                logger.info(
                    "HSPEC DEBUG generate_token_ids() [req_idx=%d]: %s | final draft_token_ids=%s",
                    di,
                    matched_line,
                    results[di] if di < len(results) else [],
                )
                logger.info(
                    "------------------------------------------- generate_token_ids() end -------------------------------------------\n"
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log draft / match info")

        return results

    # bookkeeping

    def update_accept_lengths(
        self, req_ids: List[str], accept_lengths: List[int],
    ):
        """Record accepted lengths for stats/debugging.

        Window updates are intentionally deferred to post-verification.
        """
        for rid, al in zip(req_ids, accept_lengths):
            self._accept_lengths[rid] = al
            self._stat_accept_sum += int(al)
            self._stat_accept_count += 1

    def _update_prompt_window_after_verification(
        self,
        prompt_id: str,
        accept_length: int,
        drafted_len: int,
        wnd_size_at_match: int,
        min_wnd: int,
        max_wnd: int,
    ) -> None:
        cached = self._cache.get(prompt_id)
        if cached is not None:
            current_wnd = int(cached.wnd_size)
            min_wnd = int(cached.min_wnd)
            max_wnd = int(cached.max_wnd)
        else:
            current_wnd = int(
                self._prompt_wnd_priors.get(
                    prompt_id,
                    self._clamp_wnd_size(wnd_size_at_match, min_wnd, max_wnd),
                ))
            current_wnd = self._clamp_wnd_size(current_wnd, min_wnd, max_wnd)

        threshold = int(drafted_len) if int(drafted_len) > 0 else int(wnd_size_at_match)
        if threshold <= 0:
            threshold = current_wnd

        if int(accept_length) >= threshold:
            new_wnd = min(current_wnd + 1, int(max_wnd))
        elif int(accept_length) < 1:
            new_wnd = max(current_wnd // 2, int(min_wnd))
        else:
            new_wnd = current_wnd

        self._prompt_wnd_priors[prompt_id] = int(new_wnd)
        if cached is not None:
            cached.wnd_size = int(new_wnd)

    def _update_entry_state_after_verification(
        self,
        prompt_id: str,
        entry_idx: int,
        accept_length: int,
        drafted_len: int,
    ) -> None:
        cached = self._cache.get(prompt_id)
        if cached is None:
            return
        cached.update_entry_bias_after_verification(
            entry_idx=entry_idx,
            accept_length=int(accept_length),
            drafted_len=int(drafted_len),
        )

    def update_verification_outcomes(
        self,
        req_ids: List[str],
        accepted_prefix_lengths: List[int],
    ) -> tuple[int, int]:
        """Consume true draft-prefix acceptance outcomes for HSpec studies.

        ``accepted_prefix_lengths`` must be the verification-time
        prefix-match lengths between ``draft`` and ``out`` for each request.
        """
        accept_advan_count = 0
        reject_advan_count = 0
        for rid, accepted_prefix_len in zip(req_ids, accepted_prefix_lengths):
            meta = self._pending_verify_meta.pop(rid, None)
            if meta is None:
                continue
            prompt_id = str(meta.get("prompt_id", ""))
            abs_delta = int(meta["abs_delta"])
            apl = int(accepted_prefix_len)
            if prompt_id:
                self._update_prompt_window_after_verification(
                    prompt_id=prompt_id,
                    accept_length=apl,
                    drafted_len=int(meta.get("drafted_len", 0)),
                    wnd_size_at_match=int(meta.get("wnd_size_at_match", 0)),
                    min_wnd=int(meta.get("min_wnd", 2)),
                    max_wnd=int(meta.get("max_wnd", 28)),
                )
                self._update_entry_state_after_verification(
                    prompt_id=prompt_id,
                    entry_idx=int(meta.get("matched_entry_idx", -1)),
                    accept_length=apl,
                    drafted_len=int(meta.get("drafted_len", 0)),
                )
            self._entry_pending_verify_count += 1
            self._entry_pending_accept_len_sum += apl
            self._entry_pending_abs_delta_verify[abs_delta] += 1
            if apl >= 1:
                self._entry_pending_accept_count += 1
                self._entry_pending_abs_delta_accept[abs_delta] += 1
                self._entry_pending_abs_delta_accept_len_sum[abs_delta] += apl
                if not bool(meta.get("histo_ngram_match", 0)):
                    accept_advan_count += 1
            else:
                if not bool(meta.get("histo_ngram_match", 0)):
                    reject_advan_count += 1
        return (accept_advan_count, reject_advan_count)

    def update_entry_verification_outcomes(
        self,
        req_ids: List[str],
        accepted_prefix_lengths: List[int],
    ) -> None:
        """Backward-compatible wrapper for entry-only callers."""
        self.update_verification_outcomes(req_ids, accepted_prefix_lengths)

    def clear_request(self, req_id: str):
        """Clear per-request state on completion."""
        self._accept_lengths.pop(req_id, None)
        self._pending_verify_meta.pop(req_id, None)
        self._req_prompt_ids.pop(req_id, None)
        if req_id in self._cached_batch_req_ids:
            self._cached_batch_req_ids = ()
            self._cached_batch_prompt_ids = []
            self._batched_table_cache = None

    # interface stubs

    def load_model(self, model):
        pass  # HSpec reuses the target model's hidden states

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        use_cudagraphs: bool = False,
        is_profile: bool = False,
        **kwargs,
    ):
        # HSpec does not maintain a separate draft model. Proposal is built from
        # target-model hidden states collected after verification, so there is no
        # extra dummy forward / graph capture / logits warmup work to perform
        # here. Keep this as a strict no-op while accepting the evolving vLLM /
        # vllm-ascend drafter dummy_run keyword surface for compatibility.
        return None
