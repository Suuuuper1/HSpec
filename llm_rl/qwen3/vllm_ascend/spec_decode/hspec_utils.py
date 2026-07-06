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
Utility functions for HSpec (Hidden State based Speculative Decoding).

This module provides helper functions for hidden state collection and processing.
"""

from contextlib import contextmanager, nullcontext
import atexit
import logging
import hashlib
import os
import queue
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - fallback for older torch variants
    from torch.autograd.profiler import record_function as _record_function

logger = logging.getLogger(__name__)

_hspec_profile_local = threading.local()
_hspec_runtime_metric_lock = threading.Lock()
_hspec_runtime_metrics: Dict[str, int] = {
    "pinned_pool_miss": 0,
    "pinned_pageable_fallback": 0,
    "pinned_reserved_bytes": 0,
    "pinned_reserved_slots": 0,
    "pinned_checkout_count": 0,
    "pinned_reuse_count": 0,
    "pinned_alloc_count": 0,
    "pinned_miss_budget_bytes": 0,
    "pinned_miss_budget_slots": 0,
    "pinned_miss_alloc_error": 0,
    "pinned_miss_shape_too_large": 0,
    "copy_pending_tasks": 0,
    "copy_pending_tasks_max": 0,
    "copy_pending_rows": 0,
    "copy_pending_rows_max": 0,
    "copy_submitted_tasks": 0,
    "copy_submitted_rows": 0,
    "copy_finished_tasks": 0,
    "copy_finished_rows": 0,
    "copy_backpressure_drop": 0,
    "copy_backpressure_drop_rows": 0,
    "copy_backpressure_drop_reqs": 0,
    "collect_budget_drop": 0,
    "collect_budget_drop_bytes": 0,
    "collect_budget_drop_reqs": 0,
    "collect_budget_over_worker_bytes": 0,
    "collect_budget_over_epoch_bytes": 0,
    "backpressure_active": 0,
    "backpressure_collect_skip": 0,
    "pinned_fallback_ratio_skip": 0,
    "copy_worker_error": 0,
    "copy_submit_error": 0,
    "copy_worker_pair_write_error": 0,
    "copy_token_hidden_len_mismatch": 0,
    "flush_wait_ms_total": 0,
    "flush_wait_ms_max": 0,
    "flush_wait_count": 0,
}


def _get_env_int(name: str, default: int = 0, minimum: int = 0) -> int:
    value = os.getenv(name, str(default))
    try:
        return max(int(value), int(minimum))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%s", name, value)
        return max(int(default), int(minimum))


def _get_env_float(name: str, default: float = 0.0, minimum: float = 0.0) -> float:
    value = os.getenv(name, str(default))
    try:
        return max(float(value), float(minimum))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%s", name, value)
        return max(float(default), float(minimum))


def get_hspec_collect_max_bytes_per_worker() -> int:
    """Return the Phase-4 per-worker collect byte budget.

    Step 1 only exposes the contract; Step 2 wires it into collection gates.
    A value of 0 keeps Phase-3 behaviour.
    """
    return _get_env_int("HSPEC_COLLECT_MAX_BYTES_PER_WORKER", 0, 0)


def get_hspec_collect_skip_on_pinned_fallback_ratio() -> float:
    """Return the optional pageable-fallback ratio threshold.

    A value of 0 disables this soft backpressure trigger.
    """
    return _get_env_float("HSPEC_COLLECT_SKIP_ON_PINNED_FALLBACK_RATIO", 0.0, 0.0)


def get_hspec_phase4_metrics_every_steps() -> int:
    """Return the low-frequency Phase-4 metrics cadence."""
    return _get_env_int("HSPEC_PHASE4_METRICS_EVERY_STEPS", 1, 1)


def hspec_profile_build_cpu_enabled() -> bool:
    """Whether build actors may collect additional CPU profiling samples."""
    return os.getenv("HSPEC_PROFILE_BUILD_CPU", "0") != "0"


def _parse_profile_steps(value: str) -> set[int]:
    steps: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            steps.add(int(item))
        except ValueError:
            logger.warning("Ignoring invalid HSPEC_PROFILE_STEPS item: %s", item)
    return steps


def hspec_profile_enabled_for_step(global_step: Optional[int]) -> bool:
    if os.getenv("HSPEC_PROFILE", "0") == "0":
        return False
    if global_step is None:
        return False
    steps = _parse_profile_steps(os.getenv("HSPEC_PROFILE_STEPS", None))
    return int(global_step) in steps


def hspec_profile_req_idx() -> int:
    return -1


def hspec_profile_output_dir() -> str:
    return os.getenv("HSPEC_PROFILE_DIR", "/workspace/exp/hspec_npu_profile")


def hspec_profile_with_stack() -> bool:
    return os.getenv("HSPEC_PROFILE_WITH_STACK", "0") != "0"


def hspec_profile_memory() -> bool:
    return os.getenv("HSPEC_PROFILE_MEMORY", "0") != "0"


def hspec_profile_analyse_flag() -> bool:
    return os.getenv("HSPEC_PROFILE_ANALYSE", "1") != "0"


def hspec_profile_method() -> str:
    return os.getenv("HSPEC_PROFILE_METHOD", "mstx").strip().lower()


def hspec_profile_domain() -> str:
    return os.getenv("HSPEC_PROFILE_DOMAIN", "hspec")


def _hspec_metric_add(name: str, value: int = 1) -> None:
    with _hspec_runtime_metric_lock:
        _hspec_runtime_metrics[name] = _hspec_runtime_metrics.get(name, 0) + int(value)


def _hspec_metric_set(name: str, value: int) -> None:
    with _hspec_runtime_metric_lock:
        _hspec_runtime_metrics[name] = int(value)


def _hspec_metric_max(name: str, value: int) -> None:
    with _hspec_runtime_metric_lock:
        _hspec_runtime_metrics[name] = max(
            int(_hspec_runtime_metrics.get(name, 0)),
            int(value),
        )


def hspec_collect_runtime_metrics(reset: bool = True) -> Dict[str, int]:
    pool_snapshot: Dict[str, int] = {}
    pool = globals().get("_hspec_pinned_pool")
    if pool is not None and hasattr(pool, "snapshot"):
        try:
            pool_snapshot = pool.snapshot()
        except Exception:
            pool_snapshot = {}
    with _hspec_runtime_metric_lock:
        _hspec_runtime_metrics.update(pool_snapshot)
        metrics = dict(_hspec_runtime_metrics)
        if reset:
            for key in list(_hspec_runtime_metrics.keys()):
                _hspec_runtime_metrics[key] = int(pool_snapshot.get(key, 0))
    return metrics


def hspec_profile_context_enabled() -> bool:
    return bool(getattr(_hspec_profile_local, "enabled", False))


def hspec_profile_context_step() -> Optional[int]:
    return getattr(_hspec_profile_local, "step", None)


def hspec_profile_context_req_idx() -> int:
    return int(getattr(_hspec_profile_local, "req_idx", -1))


def hspec_set_profile_context(enabled: bool, step: Optional[int], req_idx: int) -> None:
    _hspec_profile_local.enabled = bool(enabled)
    _hspec_profile_local.step = step
    _hspec_profile_local.req_idx = int(req_idx)


def hspec_clear_profile_context() -> None:
    _hspec_profile_local.enabled = False
    _hspec_profile_local.step = None
    _hspec_profile_local.req_idx = -1


@contextmanager
def hspec_record_function(name: str, use_npu_stream: bool = False):
    if not hspec_profile_context_enabled():
        with nullcontext():
            yield
        return

    method = hspec_profile_method()
    if method == "mstx":
        import torch_npu

        stream = torch_npu.npu.current_stream() if use_npu_stream else None
        domain = hspec_profile_domain()
        try:
            range_id = torch_npu.npu.mstx.range_start(name, stream, domain=domain)
        except Exception:
            # Fallback to record_function only when mstx setup itself fails.
            pass
        else:
            try:
                yield
            finally:
                torch_npu.npu.mstx.range_end(range_id, domain=domain)
            return

    with _record_function(name):
        yield


def create_hspec_torch_npu_profiler(profile_dir: str):
    import torch_npu

    level_name = os.getenv(
        "HSPEC_PROFILE_LEVEL",
        "level_none" if hspec_profile_method() == "mstx" else "level1",
    ).lower()
    if level_name == "level0":
        level = torch_npu.profiler.ProfilerLevel.Level0
    elif level_name == "level1":
        level = torch_npu.profiler.ProfilerLevel.Level1
    elif level_name == "level2":
        level = torch_npu.profiler.ProfilerLevel.Level2
    elif level_name == "level_none":
        level = torch_npu.profiler.ProfilerLevel.Level_none
    else:
        raise ValueError(f"Unsupported HSPEC_PROFILE_LEVEL: {level_name}")

    use_mstx = hspec_profile_method() == "mstx"
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=(
            [torch_npu.profiler.ExportType.Db]
            if use_mstx
            else [torch_npu.profiler.ExportType.Text]
        ),
        profiler_level=level,
        mstx=use_mstx,
        mstx_domain_include=["default", hspec_profile_domain()] if use_mstx else [],
        mstx_domain_exclude=[],
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False if use_mstx else True,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=1,
            repeat=1,
            skip_first=0,
        ),
        with_stack=hspec_profile_with_stack(),
        profile_memory=hspec_profile_memory(),
        with_modules=False,
        record_shapes=False,
        with_flops=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profile_dir,
            analyse_flag=hspec_profile_analyse_flag(),
        ),
    )


def prompt_id_from_token_ids(prompt_token_ids: List[int]) -> str:
    """Compute a stable prompt_id from prompt token ids.
    """
    # Pack as little-endian uint32 stream to avoid ambiguity and reduce overhead.
    # Token ids are expected to be non-negative.
    buf = bytearray()
    for tid in prompt_token_ids:
        if tid < 0:
            raise ValueError(f"prompt_token_ids must be non-negative, got {tid}")
        buf += struct.pack("<I", int(tid))
    # blake2b is fast and available in stdlib; digest_size=8 gives 64-bit id.
    digest = hashlib.blake2b(buf, digest_size=8).hexdigest()
    return f"p{digest}"


def stable_partition_id(key: str, num_partitions: int) -> int:
    """Stable partitioner for strings across processes."""
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be > 0, got {num_partitions}")
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    hv = int.from_bytes(h, byteorder="little", signed=False)
    return hv % num_partitions


class HiddenStateCollector:
    """Collector for hidden states during model inference.
    
    This class helps collect and manage hidden states during generation,
    which are later used to build the HSpec query tables.
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 4096, device: str = "cpu"):
        """Initialize the hidden state collector.
        
        Args:
            hidden_dim: Dimension of hidden states.
            max_seq_len: Maximum sequence length to collect.
            device: Device for tensor operations.
        """
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Storage for collected hidden states
        self._hidden_states: Dict[str, List[np.ndarray]] = {}
        self._token_ids: Dict[str, List[int]] = {}
    
    def start_collection(self, req_id: str):
        """Start collecting hidden states for a request.
        
        Args:
            req_id: The request identifier.
        """
        self._hidden_states[req_id] = []
        self._token_ids[req_id] = []
    
    def collect(self, req_id: str, hidden_state: np.ndarray, token_id: int):
        """Collect a hidden state for a request.
        
        Args:
            req_id: The request identifier.
            hidden_state: The hidden state vector. Shape: (hidden_dim,)
            token_id: The corresponding token id.
        """
        if req_id not in self._hidden_states:
            self.start_collection(req_id)
        
        if len(self._hidden_states[req_id]) < self.max_seq_len:
            self._hidden_states[req_id].append(hidden_state)
            self._token_ids[req_id].append(token_id)
    
    def collect_batch(self, req_id: str, hidden_states: np.ndarray, token_ids: List[int]):
        """Collect a batch of hidden states for a request.
        
        Args:
            req_id: The request identifier.
            hidden_states: Hidden states array. Shape: (seq_len, hidden_dim)
            token_ids: List of token ids.
        """
        if req_id not in self._hidden_states:
            self.start_collection(req_id)

        for hs, tid in zip(hidden_states, token_ids):
            if len(self._hidden_states[req_id]) < self.max_seq_len:
                self._hidden_states[req_id].append(hs)
                self._token_ids[req_id].append(tid)

    def get_collected(self, req_id: str) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """Get collected hidden states and token ids for a request.
        
        Args:
            req_id: The request identifier.
        
        Returns:
            Tuple of (hidden_states array, token_ids list), or (None, None) if not found.
        """
        if req_id not in self._hidden_states:
            return None, None
        
        hidden_states = np.stack(self._hidden_states[req_id], axis=0)
        token_ids = self._token_ids[req_id]
        
        return hidden_states, token_ids
    
    def clear_request(self, req_id: str):
        """Clear collected data for a request.
        
        Args:
            req_id: The request identifier.
        """
        if req_id in self._hidden_states:
            del self._hidden_states[req_id]
        if req_id in self._token_ids:
            del self._token_ids[req_id]
    
    def clear_all(self):
        """Clear all collected data."""
        self._hidden_states.clear()
        self._token_ids.clear()


def extract_last_hidden_state(
    hidden_states: torch.Tensor,
    logits_indices: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Extract the last hidden state for each request.
    
    Args:
        hidden_states: Hidden states tensor. Shape: (total_tokens, hidden_dim)
        logits_indices: Indices of the last token for each request.
        device: Device for tensor operations.
    
    Returns:
        NumPy array of last hidden states. Shape: (num_requests, hidden_dim)
    """
    # Index hidden states at logits positions
    last_hidden_states = hidden_states[logits_indices]
    
    # Convert to numpy
    return last_hidden_states.cpu().numpy()


def compute_similarity(
    query: np.ndarray,
    keys: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute cosine similarity between query and keys.
    
    Args:
        query: Query vector. Shape: (hidden_dim,)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        normalize: Whether to normalize vectors before computing similarity.
    
    Returns:
        Similarity scores. Shape: (num_keys,)
    """
    if normalize:
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Normalize keys
        keys_norm = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_norm = np.maximum(keys_norm, 1e-8)
        keys = keys / keys_norm
    
    # Compute dot product (cosine similarity since vectors are normalized)
    similarities = np.dot(keys, query)
    
    return similarities


def batch_compute_similarity(
    queries: np.ndarray,
    keys: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Batch compute cosine similarity between queries and keys.
    
    Args:
        queries: Query matrix. Shape: (num_queries, hidden_dim)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        normalize: Whether to normalize vectors.
    
    Returns:
        Similarity matrix. Shape: (num_queries, num_keys)
    """
    if normalize:
        # Normalize queries
        queries_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_norm = np.maximum(queries_norm, 1e-8)
        queries = queries / queries_norm
        
        # Normalize keys
        keys_norm = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_norm = np.maximum(keys_norm, 1e-8)
        keys = keys / keys_norm
    
    # Compute similarity matrix
    similarities = np.dot(queries, keys.T)
    
    return similarities


def find_best_match(
    query: np.ndarray,
    keys: np.ndarray,
    threshold: float = 0.9,
    normalize: bool = True,
) -> Tuple[int, float]:
    """Find the best matching key for a query.
    
    Args:
        query: Query vector. Shape: (hidden_dim,)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        threshold: Minimum similarity threshold for a valid match.
        normalize: Whether to normalize vectors.
    
    Returns:
        Tuple of (best_index, best_similarity).
        Returns (-1, 0.0) if no match above threshold.
    """
    similarities = compute_similarity(query, keys, normalize)
    
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    
    if best_sim >= threshold:
        return int(best_idx), float(best_sim)
    return -1, 0.0


def prepare_hidden_states_for_storage(
    hidden_states: torch.Tensor,
    token_ids: List[int],
    pad_token_id: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Prepare hidden states for storage in HSpec tables.
    
    This function:
    1. Removes padding positions
    2. Converts to numpy
    3. Returns aligned hidden states and token ids
    
    Args:
        hidden_states: Hidden states tensor. Shape: (seq_len, hidden_dim)
        token_ids: List of token ids (may include padding).
        pad_token_id: The padding token id.
    
    Returns:
        Tuple of (hidden_states_np, valid_token_ids).
    """
    # Find valid (non-padding) positions
    valid_positions = [i for i, tid in enumerate(token_ids) if tid != pad_token_id]

    if not valid_positions:
        return np.array([]), []
    
    # Extract valid hidden states
    valid_indices = torch.tensor(valid_positions, dtype=torch.long)
    valid_hidden_states = hidden_states[valid_indices].cpu().numpy()
    valid_token_ids = [token_ids[i] for i in valid_positions]
    
    return valid_hidden_states, valid_token_ids


class PromptPCAParams:
    """Fitted PCA parameters for a single prompt.

    Instances are created during table-building (end of each epoch) and
    cached by the proposer for online query in the next epoch.

    Attributes:
        prompt_id:    Stable identifier produced by prompt_id_from_token_ids.
        mean:         Centroid μ, shape (D,), float32, C-contiguous.
        components:   Top-K principal components W, shape (K, D), float32,
                      C-contiguous.  Rows are the component directions.
        n_components: Actual K used (may be < requested when N or D < K).
        n_samples:    Number of token positions that were used for fitting.
    """

    __slots__ = ("prompt_id", "mean", "components", "n_components", "n_samples")

    def __init__(
        self,
        prompt_id: str,
        mean: np.ndarray,
        components: np.ndarray,
        n_samples: int,
    ):
        self.prompt_id = prompt_id
        self.mean = np.ascontiguousarray(mean, dtype=np.float32)         # (D,)
        self.components = np.ascontiguousarray(components, dtype=np.float32)  # (K, D)
        self.n_components = int(components.shape[0])
        self.n_samples = int(n_samples)

    # CPU / numpy projection  (used during table-building)
    def project(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states into the PCA space.

        Z = (H − μ) · W^T

        Args:
            hidden_states: (N, D) array of anchor hidden states, or a
                           single (D,) vector.

        Returns:
            (N, K) or (K,) projected array, dtype float32.
        """
        centered = hidden_states.astype(np.float32, copy=False) - self.mean
        projected = centered @ self.components.T  # (N, K) or (K,)
        return projected.astype(np.float32, copy=False)

    # Torch conversion  (used to cache params on NPU for decode queries)
    def to_torch(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Export (μ, W) as torch tensors on the given device.

        Returns:
            mean_t:       (D,)    tensor on *device*.
            components_t: (K, D)  tensor on *device*.

        Notes:
            - By default, we keep the original float32 numpy parameters
              (dtype=None).
            - For decode hot-loop performance, callers should pass a low
              precision dtype (e.g. torch.float16 / torch.bfloat16) and
              cache the returned tensors on device so that projection does
              not upcast hidden_states each step.
        """
        mean_t = torch.from_numpy(self.mean).to(device=device, dtype=dtype, non_blocking=True)
        components_t = torch.from_numpy(self.components).to(
            device=device, dtype=dtype, non_blocking=True)
        return mean_t, components_t

    def __repr__(self) -> str:
        hidden_dim = self.mean.shape[0]
        return (
            f"PromptPCAParams(prompt_id={self.prompt_id!r}, "
            f"D={hidden_dim}, K={self.n_components}, n_samples={self.n_samples})"
        )


def compute_pca(
    hidden_states: np.ndarray,
    n_components: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA parameters from anchor hidden states via economy SVD,
    and return exactly `n_components` principal components (padded with zeros
    if necessary).

    Given N token-level anchor hidden states of dimension D, compute the
    centroid μ and the top-K principal component directions W, where K is
    min(n_components, N, D). If K < n_components, the remaining components
    are set to zero vectors.
    
    Internally:
        1. μ = mean(H, axis=0)
        2. H_c = H − μ                         (centering)
        3. U, S, V^T = SVD(H_c, full=False)    (economy SVD)
        4. W = V^T[:K]                          (top-K rows)

    Args:
        hidden_states: (N, D) float array. N = total token positions,
                       D = model hidden dimension.
        n_components:  Desired number of principal components.

    Returns:
        mean:       (D,)        float32.
        components: (n_components, D)  float32. If the data rank is less than
                    n_components, the extra rows are zero vectors.

    Raises:
        ValueError: If input is not 2-D or has zero rows.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be 2-D (N, D), got shape {hidden_states.shape}")
    sample_count, hidden_dim = hidden_states.shape
    if sample_count == 0:
        raise ValueError("Cannot compute PCA on zero samples")

    # Upcast to float32 for numerical stability (no-copy if already f32).
    hs = hidden_states.astype(np.float32, copy=False)
    
    # 1. Centroid
    mean = hs.mean(axis=0)       # (D,)

    # 2. Center
    centered = hs - mean         # (N, D)

    # 3. Economy SVD: centered = U · diag(S) · V^T
    #    Vt has shape (min(N, D), D)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    # 4. Determine how many components we can actually take.
    #    Vt always has min(N, D) rows.
    max_components = min(sample_count, hidden_dim)
    effective_components = min(n_components, max_components)
    
    # 5. Take the top effective_components components.
    components = vt[:effective_components].astype(np.float32)

    # 6. Pad with zero rows if we have fewer than n_components.
    if components.shape[0] < n_components:
        padding = np.zeros((n_components - components.shape[0], hidden_dim),
                           dtype=np.float32)
        components = np.vstack([components, padding])

    # Ensure contiguous layout for later use (optional, but good practice).
    components = np.ascontiguousarray(components)

    return mean.astype(np.float32), components


def fit_pca_single_sequence(
    prompt_id: str,
    hidden_states: np.ndarray,
    n_components: int = 64,
) -> Tuple[PromptPCAParams, np.ndarray]:
    """Fit PCA on **one** rollout sequence and project  (PPO mode).

    This is used when each prompt produces exactly one rollout trajectory
    per epoch (the standard PPO setting).

    Args:
        prompt_id:     Stable prompt identifier.
        hidden_states: (L, D) anchor hidden states aligned 1-to-1 with the
                       response tokens y = [y_0, …, y_{L-1}].
        n_components:  Number of PCA dimensions K.

    Returns:
        pca_params: PromptPCAParams holding (μ, W) for this prompt.
        projected:  (L, K) projected key matrix  Z = (H − μ) W^T.

    Raises:
        ValueError: If hidden_states is not 2-D or is empty.
    """
    mean, components = compute_pca(hidden_states, n_components)
    params = PromptPCAParams(
        prompt_id=prompt_id,
        mean=mean,
        components=components,
        n_samples=hidden_states.shape[0],
    )
    projected = params.project(hidden_states)  # (L, K)
    return params, projected


def fit_pca_multi_sequence(
    prompt_id: str,
    hidden_states_list: List[np.ndarray],
    n_components: int = 64,
) -> Tuple[PromptPCAParams, List[np.ndarray]]:
    """Fit PCA on **multiple** rollout sequences and project each  (GRPO mode).

    When a prompt is rolled out N times in one epoch (e.g. GRPO with
    repeat-n), all N trajectories' anchor hidden states are *pooled*
    together to compute a single set of PCA parameters (μ, W).  Each
    trajectory is then projected independently to produce its own key
    matrix Z_i.

    Args:
        prompt_id:          Stable prompt identifier.
        hidden_states_list: List of (L_i, D) arrays, one per rollout
                            trajectory.  Every array must have the same
                            D (hidden dimension).
        n_components:       Number of PCA dimensions K.

    Returns:
        pca_params:     Shared PromptPCAParams for this prompt.
        projected_list: List of (L_i, K) projected key matrices, in the
                        same order as *hidden_states_list*.

    Raises:
        ValueError: If the list is empty, or arrays have inconsistent D.
    """
    if not hidden_states_list:
        raise ValueError("hidden_states_list must be non-empty")

    hidden_dim = hidden_states_list[0].shape[-1]
    for idx, hs in enumerate(hidden_states_list):
        if hs.ndim != 2:
            raise ValueError(
                f"hidden_states_list[{idx}] must be 2-D, got shape {hs.shape}")
        if hs.shape[1] != hidden_dim:
            raise ValueError(
                f"Inconsistent hidden dim: hidden_states_list[0] has D={hidden_dim}, "
                f"but hidden_states_list[{idx}] has D={hs.shape[1]}")

    # Concatenate all sequences for joint PCA fitting.
    all_hs = np.concatenate(hidden_states_list, axis=0)  # (Σ L_i, D)

    mean, components = compute_pca(all_hs, n_components)
    params = PromptPCAParams(
        prompt_id=prompt_id,
        mean=mean,
        components=components,
        n_samples=all_hs.shape[0],
    )

    # Project each sequence individually.
    projected_list = [params.project(hs) for hs in hidden_states_list]
    return params, projected_list


# On-device (NPU / GPU) projection for decode hot-loop

def project_hidden_states_torch(
    hidden_states: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    """Project hidden states on device via a single matmul  (zero CPU sync).

    This is the **hot-path** operation executed every decode step inside
    the proposer.  All three tensors must already reside on the *same*
    device (NPU / GPU).

        z = (h − μ) · W^T

    Args:
        hidden_states: (B, D) batch of anchor hidden states, or (D,)
                       for a single request.
        mean:          (D,)   PCA centroid μ.
        components:    (K, D) PCA components W.

    Returns:
        (B, K) or (K,) projected tensor, same dtype/device as inputs.
    """
    # IMPORTANT (performance):
    # - Do NOT unconditionally upcast to fp32 here; the caller should cache
    #   (mean, components) in the same dtype as hidden_states (fp16/bf16)
    #   before entering the decode hot-loop.
    #
    # We keep a small defensive cast in case a caller passes mismatched dtypes.
    target_dtype = hidden_states.dtype
    if mean.dtype != target_dtype:
        mean = mean.to(dtype=target_dtype)
    if components.dtype != target_dtype:
        components = components.to(dtype=target_dtype)

    centered = hidden_states - mean
    return torch.matmul(centered, components.t())


def batch_project_and_match_torch(
    hidden_states: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    keys: torch.Tensor,
    threshold: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project + cosine-similarity matching entirely on device.

    Combines projection and matching into a single fused pipeline so
    that the decode hot-loop never touches the CPU:

        1. z = (h − μ) · W^T               (projection)
        2. z_n = z / ‖z‖                    (L2-normalise)
        3. sim = z_n · keys^T               (dot-product similarity)
        4. best = argmax(sim)               (per-query best match)

    Args:
        hidden_states: (B, D)  batch of anchor hidden states.
        mean:          (D,)    PCA centroid μ.
        components:    (K, D)  PCA components W.
        keys:          (M, K)  stored keys, **already L2-normalised**.
        threshold:     Minimum cosine similarity for a valid match.

    Returns:
        best_indices:      (B,) long tensor.  Index into *keys* for the
                           best match; **-1** when no key exceeds the
                           threshold.
        best_similarities: (B,) float tensor.  The similarity score of
                           the best match (0.0 when no match).
    """
    # 1. Project  → (B, K)
    z = project_hidden_states_torch(hidden_states, mean, components)

    # 2. L2-normalise queries
    z_norm = torch.nn.functional.normalize(z, p=2, dim=-1)  # (B, K)

    # 3. Cosine similarity matrix  → (B, M)
    # Keep matmul in low precision for throughput. Ensure dtype alignment.
    if keys.dtype != z_norm.dtype:
        z_norm = z_norm.to(dtype=keys.dtype)
    sims = torch.matmul(z_norm, keys.t())

    # 4. Per-query best match
    best_sims, best_idxs = sims.max(dim=-1)  # (B,) each

    # 5. Mask out below-threshold matches
    no_match = best_sims < threshold
    best_idxs = best_idxs.clone()
    best_idxs[no_match] = -1
    best_sims = best_sims.clone()
    best_sims[no_match] = 0.0

    return best_idxs, best_sims


# Alignment validation

def validate_hidden_state_alignment(
    hidden_states: np.ndarray,
    response_tokens: List[int],
) -> bool:
    """Check that hidden states and response tokens are properly aligned.

    The HSpec design doc mandates ``len(H) == len(y)`` as a **hard**
    prerequisite before any data enters the query table.  Misaligned
    entries silently pollute the table and cause low match rates that
    are very difficult to diagnose.

    Args:
        hidden_states: (L, D) array of anchor hidden states.
        response_tokens: Response token id list y = [y_0, …, y_{L-1}].

    Returns:
        True if and only if the lengths match and shapes are valid.
    """
    if hidden_states.ndim != 2:
        logger.warning(
            "validate_hidden_state_alignment: expected 2-D hidden_states, got shape %s",
            hidden_states.shape,
        )
        return False
    aligned = hidden_states.shape[0] == len(response_tokens)
    if not aligned:
        logger.warning(
            "validate_hidden_state_alignment FAILED: len(hidden_states)=%d != "
            "len(response_tokens)=%d, this trajectory will be discarded.",
            hidden_states.shape[0],
            len(response_tokens),
        )
    return aligned


# Global Hidden State Store for HSpec Collection
#
# This module-level store allows the model_runner to accumulate
# anchor hidden states during generation, and the rollout code to
# retrieve them after generation completes.
#
# Compliance:
#   - No per-token .cpu() sync: tensors are cloned on-device and
#     transferred to CPU only once per request at flush time.
#   - Alignment guarantee: callers must ensure len(H) == len(y)
#     before feeding data into the HSpec table.
#
# Flow:
#   1. model_runner calls hspec_append_step_hs() each decode step.
#   2. After LLM.generate() returns, rollout calls
#      hspec_flush_and_get_all() which transfers device tensors to
#      CPU and returns the complete store.
#   3. The store is cleared for the next generation batch.

import threading as _threading

_hspec_store_lock = _threading.Lock()
_hspec_store_cond = _threading.Condition(_hspec_store_lock)

# Host-side accumulation buffers: req_id -> list of CPU tensors with shape
# (chunk_len, hidden_dim). Device-to-host copies are submitted asynchronously
# so hidden-state collection does not block the decode hot path.
_hspec_host_buffers: Dict[str, List[torch.Tensor]] = {}
# Token-side accumulation buffers: req_id -> list of accepted/generated token ids
_hspec_token_buffers: Dict[str, List[int]] = {}

# Whether collection is enabled (set by model_runner at init)
_hspec_collection_enabled: bool = False


class _HSpecAsyncCopyTask:

    __slots__ = (
        "req_slices",
        "token_slices",
        "cpu_tensor",
        "event",
        "device_tensor_ref",
        "pool_handle",
        "num_rows",
    )

    def __init__(
        self,
        req_slices: List[Tuple[str, int, int]],
        token_slices: Optional[Dict[str, List[int]]],
        cpu_tensor: torch.Tensor,
        event: Any,
        device_tensor_ref: Optional[torch.Tensor],
        pool_handle: Any = None,
        num_rows: int = 0,
    ) -> None:
        self.req_slices = req_slices
        self.token_slices = token_slices
        self.cpu_tensor = cpu_tensor
        self.event = event
        # Keep the source device tensor alive until the async copy completes.
        self.device_tensor_ref = device_tensor_ref
        self.pool_handle = pool_handle
        self.num_rows = int(num_rows)


class _HSpecAsyncAccumulateTask:

    __slots__ = (
        "req_ids",
        "req_slices",
        "flat_indices",
        "pending_req_ids",
        "token_slices",
        "num_rows",
        "sample_hidden_states",
        "valid_sampled_token_ids",
        "spec_decode_metadata",
        "accepted_prefix_lengths",
        "producer_event",
    )

    def __init__(
        self,
        req_ids: Tuple[str, ...],
        req_slices: List[Tuple[str, int, int]],
        flat_indices: List[int],
        pending_req_ids: List[str],
        token_slices: Optional[Dict[str, List[int]]],
        num_rows: int,
        sample_hidden_states: torch.Tensor,
        valid_sampled_token_ids: List[List[int]],
        spec_decode_metadata: Any,
        accepted_prefix_lengths: Optional[List[int]],
        producer_event: Any,
    ) -> None:
        self.req_ids = req_ids
        self.req_slices = req_slices
        self.flat_indices = flat_indices
        self.pending_req_ids = pending_req_ids
        self.token_slices = token_slices
        self.num_rows = int(num_rows)
        self.sample_hidden_states = sample_hidden_states
        self.valid_sampled_token_ids = valid_sampled_token_ids
        self.spec_decode_metadata = spec_decode_metadata
        self.accepted_prefix_lengths = accepted_prefix_lengths
        self.producer_event = producer_event


_hspec_copy_queue: "queue.SimpleQueue[_HSpecAsyncCopyTask | None]" = queue.SimpleQueue()
_hspec_copy_thread: Optional[_threading.Thread] = None
_hspec_accumulate_queue: "queue.SimpleQueue[_HSpecAsyncAccumulateTask | None]" = queue.SimpleQueue()
_hspec_accumulate_thread: Optional[_threading.Thread] = None
_hspec_async_transfer_streams: Dict[str, Any] = {}
_hspec_async_pending_total: int = 0
_hspec_async_pending_tasks: int = 0
_hspec_async_pending_rows: int = 0
_hspec_async_pending_by_req: Dict[str, int] = {}


class _HSpecPinnedPool:
    """Small fixed-size pinned host buffer pool for async D2H copies."""

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._free: Dict[Tuple[str, int, int], List[torch.Tensor]] = {}
        self._reserved_bytes = 0
        self._reserved_slots = 0
        self._max_bytes = int(
            os.getenv("HSPEC_PINNED_POOL_BYTES", str(256 * 1024 * 1024)))
        self._max_slots = int(os.getenv("HSPEC_PINNED_POOL_MAX_SLOTS", "64"))
        buckets = os.getenv(
            "HSPEC_PINNED_POOL_BUCKET_ROWS",
            "64,128,256,512,1024,2048,4096",
        )
        parsed = []
        for item in buckets.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = int(item)
                if value > 0:
                    parsed.append(value)
            except ValueError:
                logger.warning("Ignoring invalid HSPEC_PINNED_POOL_BUCKET_ROWS item: %s", item)
        self._bucket_rows = sorted(set(parsed)) or [64, 128, 256, 512, 1024, 2048, 4096]

    def _bucket_for_rows(self, rows: int) -> int:
        for bucket in self._bucket_rows:
            if rows <= bucket:
                return bucket
        # Keep large copies bounded to one exact-size allocation; they are not
        # rounded to avoid reserving a huge rarely reusable slot.
        return int(rows)

    def checkout(
        self,
        shape: torch.Size | Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[str, int, int], torch.Tensor]]]:
        rows = int(shape[0])
        hidden_dim = int(shape[1]) if len(shape) > 1 else 1
        bucket_rows = self._bucket_for_rows(rows)
        key = (str(dtype), hidden_dim, bucket_rows)
        _hspec_metric_add("pinned_checkout_count")

        with self._lock:
            free_list = self._free.get(key)
            if free_list:
                base = free_list.pop()
                _hspec_metric_add("pinned_reuse_count")
                return base[:rows, :hidden_dim], (key, base)

            bytes_needed = bucket_rows * hidden_dim * torch.empty((), dtype=dtype).element_size()
            if self._max_bytes > 0 and bytes_needed > self._max_bytes:
                _hspec_metric_add("pinned_miss_shape_too_large")
            over_bytes = (
                self._max_bytes > 0
                and self._reserved_bytes + bytes_needed > self._max_bytes
            )
            over_slots = self._max_slots > 0 and self._reserved_slots >= self._max_slots
            can_reserve = not over_bytes and not over_slots
            if can_reserve:
                try:
                    base = torch.empty(
                        (bucket_rows, hidden_dim),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self._reserved_bytes += bytes_needed
                    self._reserved_slots += 1
                    _hspec_metric_add("pinned_alloc_count")
                    _hspec_metric_set("pinned_reserved_bytes", self._reserved_bytes)
                    _hspec_metric_set("pinned_reserved_slots", self._reserved_slots)
                    return base[:rows, :hidden_dim], (key, base)
                except Exception:
                    logger.debug("HSpec pinned pool allocation failed; using pageable CPU buffer",
                                 exc_info=True)
                    _hspec_metric_add("pinned_pool_miss")
                    _hspec_metric_add("pinned_pageable_fallback")
                    _hspec_metric_add("pinned_miss_alloc_error")
                    return torch.empty(tuple(shape), dtype=dtype, device="cpu"), None
            else:
                if over_bytes:
                    _hspec_metric_add("pinned_miss_budget_bytes")
                if over_slots:
                    _hspec_metric_add("pinned_miss_budget_slots")

        # Budget exhausted or pin allocation failed: do not block decode.
        _hspec_metric_add("pinned_pool_miss")
        _hspec_metric_add("pinned_pageable_fallback")
        return torch.empty(tuple(shape), dtype=dtype, device="cpu"), None

    def release(self, handle: Optional[Tuple[Tuple[str, int, int], torch.Tensor]]) -> None:
        if handle is None:
            return
        key, base = handle
        with self._lock:
            self._free.setdefault(key, []).append(base)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "pinned_reserved_bytes": int(self._reserved_bytes),
                "pinned_reserved_slots": int(self._reserved_slots),
            }


_hspec_pinned_pool = _HSpecPinnedPool()


def _hspec_checkout_cpu_buffer(
    shape: torch.Size | Tuple[int, ...],
    dtype: torch.dtype,
    allow_pool: bool,
) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[str, int, int], torch.Tensor]]]:
    if allow_pool:
        return _hspec_pinned_pool.checkout(shape, dtype)
    try:
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True), None
    except Exception:
        _hspec_metric_add("pinned_pageable_fallback")
        return torch.empty(tuple(shape), dtype=dtype, device="cpu"), None


def _hspec_use_legacy_async_accumulate() -> bool:
    """Whether to use the old background-thread NPU accumulation path."""
    return os.getenv("HSPEC_ASYNC_HS_ACCUMULATE", "0") != "0"


def _hspec_use_async_copy_stream() -> bool:
    """Whether HSpec D2H copies may run on a separate NPU stream.

    The default path keeps NPU kernels on the caller's stream, then moves only
    the host copy to a transfer stream. This preserves HCCL/kernel ordering
    while hiding most D2H latency behind the next decode wave.
    """
    return os.getenv("HSPEC_ASYNC_HS_COPY_STREAM", "1") != "0"


def _get_hspec_copy_max_pending_tasks() -> int:
    try:
        return max(int(os.getenv("HSPEC_COPY_MAX_PENDING_TASKS", "0")), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_COPY_MAX_PENDING_TASKS")
        return 0


def _get_hspec_copy_max_pending_rows() -> int:
    try:
        return max(int(os.getenv("HSPEC_COPY_MAX_PENDING_ROWS", "0")), 0)
    except ValueError:
        logger.warning("Ignoring invalid HSPEC_COPY_MAX_PENDING_ROWS")
        return 0


def _hspec_drop_on_backpressure_enabled() -> bool:
    return os.getenv("HSPEC_DROP_ON_BACKPRESSURE", "1") != "0"


def hspec_should_collect_step(
    estimated_rows: int = 0,
    estimated_reqs: int = 0,
) -> bool:
    if not _hspec_collection_enabled:
        return False

    max_tasks = _get_hspec_copy_max_pending_tasks()
    max_rows = _get_hspec_copy_max_pending_rows()
    if max_tasks <= 0 and max_rows <= 0:
        return True

    rows = max(int(estimated_rows), 0)
    reqs = max(int(estimated_reqs), 0)
    with _hspec_store_cond:
        would_exceed_tasks = (
            max_tasks > 0 and _hspec_async_pending_tasks >= max_tasks
        )
        would_exceed_rows = (
            max_rows > 0 and _hspec_async_pending_rows + rows > max_rows
        )

    if not (would_exceed_tasks or would_exceed_rows):
        return True
    if _hspec_drop_on_backpressure_enabled():
        _hspec_metric_add("copy_backpressure_drop")
        _hspec_metric_add("copy_backpressure_drop_rows", rows)
        _hspec_metric_add("copy_backpressure_drop_reqs", reqs)
        return False
    return True


def _hspec_metadata_to_list(metadata: Any, cache_attr: str,
                            tensor_attr: str) -> List[int]:
    cached = getattr(metadata, cache_attr, None)
    if cached is not None:
        return [int(x) for x in cached]
    tensor = getattr(metadata, tensor_attr)
    return [int(x) for x in tensor.detach().cpu().tolist()]


def _hspec_compute_req_slices(
    req_ids: List[str],
    valid_sampled_token_ids: List[List[int]],
    spec_decode_metadata: Any = None,
    accepted_prefix_lengths: Optional[List[int]] = None,
) -> Tuple[List[Tuple[str, int, int]], List[int], List[str]]:
    req_to_row_indices: List[Tuple[str, List[int]]] = []
    n = min(len(req_ids), len(valid_sampled_token_ids))

    if spec_decode_metadata is None:
        for i in range(n):
            sampled_ids = valid_sampled_token_ids[i]
            if sampled_ids:
                req_to_row_indices.append((str(req_ids[i]), [i]))
    else:
        num_draft_list = spec_decode_metadata.num_draft_tokens
        bonus_indices = _hspec_metadata_to_list(
            spec_decode_metadata,
            "_hspec_bonus_logits_indices_cpu",
            "bonus_logits_indices",
        )
        target_logits_indices = _hspec_metadata_to_list(
            spec_decode_metadata,
            "_hspec_target_logits_indices_cpu",
            "target_logits_indices",
        )
        cu_num_draft_tokens = _hspec_metadata_to_list(
            spec_decode_metadata,
            "_hspec_cu_num_draft_tokens_cpu",
            "cu_num_draft_tokens",
        )

        for i in range(n):
            sampled_ids = valid_sampled_token_ids[i]
            if not sampled_ids:
                continue

            row_indices: List[int] = []
            if num_draft_list[i] == 0:
                row_indices.append(int(bonus_indices[i]))
                req_to_row_indices.append((str(req_ids[i]), row_indices))
                continue

            num_drafts = int(num_draft_list[i])
            accepted = (
                int(accepted_prefix_lengths[i])
                if accepted_prefix_lengths is not None and i < len(accepted_prefix_lengths)
                else 0
            )
            out_len = len(sampled_ids)

            start = int(cu_num_draft_tokens[i - 1]) if i > 0 else 0
            end = start + num_drafts
            local_target_rows = target_logits_indices[start:end]

            for j in range(min(accepted, num_drafts, out_len)):
                row_indices.append(int(local_target_rows[j]))

            if out_len > accepted:
                if accepted < num_drafts:
                    row_indices.append(int(local_target_rows[accepted]))
                else:
                    row_indices.append(int(bonus_indices[i]))

            if row_indices:
                req_to_row_indices.append((str(req_ids[i]), row_indices))

    flat_indices: List[int] = []
    req_slices: List[Tuple[str, int, int]] = []
    pending_req_ids: List[str] = []
    for req_id, row_indices in req_to_row_indices:
        if not row_indices:
            continue
        start = len(flat_indices)
        flat_indices.extend(row_indices)
        end = len(flat_indices)
        req_slices.append((req_id, start, end))
        pending_req_ids.append(req_id)

    return req_slices, flat_indices, pending_req_ids


def _hspec_contiguous_index_slice(
    indices: List[int],
) -> Optional[Tuple[int, int]]:
    if not indices:
        return None
    start = int(indices[0])
    if start < 0:
        return None
    for offset, index in enumerate(indices):
        if int(index) != start + offset:
            return None
    return start, len(indices)


def _hspec_compute_token_slices(
    req_ids: List[str],
    valid_sampled_token_ids: List[List[int]],
    req_slices: List[Tuple[str, int, int]],
) -> Optional[Dict[str, List[int]]]:
    req_to_tokens: Dict[str, List[int]] = {}
    n = min(len(req_ids), len(valid_sampled_token_ids))
    for i in range(n):
        req_to_tokens[str(req_ids[i])] = [
            int(x) for x in valid_sampled_token_ids[i]
        ]

    token_slices: Dict[str, List[int]] = {}
    for req_id, start, end in req_slices:
        expected = int(end) - int(start)
        if expected <= 0:
            continue
        tokens = req_to_tokens.get(str(req_id), [])
        if len(tokens) < expected:
            _hspec_metric_add("copy_token_hidden_len_mismatch")
            return None
        token_slices[str(req_id)] = tokens[:expected]
    return token_slices


def _hspec_validate_token_slices(
    req_slices: List[Tuple[str, int, int]],
    token_slices: Optional[Dict[str, List[int]]],
) -> bool:
    if token_slices is None:
        return True
    for req_id, start, end in req_slices:
        expected = int(end) - int(start)
        actual = len(token_slices.get(str(req_id), []))
        if actual != expected:
            _hspec_metric_add("copy_token_hidden_len_mismatch")
            return False
    return True


def _hspec_reserve_pending(req_ids: List[str], rows: int) -> None:
    global _hspec_async_pending_total
    global _hspec_async_pending_tasks
    global _hspec_async_pending_rows

    rows = max(int(rows), 0)
    with _hspec_store_cond:
        for req_id in req_ids:
            _hspec_async_pending_by_req[req_id] = (
                _hspec_async_pending_by_req.get(req_id, 0) + 1
            )
        _hspec_async_pending_total += len(req_ids)
        _hspec_async_pending_tasks += 1
        _hspec_async_pending_rows += rows
        _hspec_metric_set("copy_pending_tasks", _hspec_async_pending_tasks)
        _hspec_metric_set("copy_pending_rows", _hspec_async_pending_rows)
        _hspec_metric_max("copy_pending_tasks_max", _hspec_async_pending_tasks)
        _hspec_metric_max("copy_pending_rows_max", _hspec_async_pending_rows)


def _hspec_finish_pending(
    req_ids: List[str],
    rows: int = 0,
    task_count: int = 1,
) -> None:
    global _hspec_async_pending_total
    global _hspec_async_pending_tasks
    global _hspec_async_pending_rows

    with _hspec_store_cond:
        for req_id in req_ids:
            pending = _hspec_async_pending_by_req.get(req_id, 0) - 1
            if pending > 0:
                _hspec_async_pending_by_req[req_id] = pending
            else:
                _hspec_async_pending_by_req.pop(req_id, None)
        _hspec_async_pending_total = max(
            _hspec_async_pending_total - len(req_ids), 0)
        _hspec_async_pending_tasks = max(
            _hspec_async_pending_tasks - int(task_count), 0)
        _hspec_async_pending_rows = max(
            _hspec_async_pending_rows - max(int(rows), 0), 0)
        _hspec_metric_set("copy_pending_tasks", _hspec_async_pending_tasks)
        _hspec_metric_set("copy_pending_rows", _hspec_async_pending_rows)
        if task_count > 0:
            _hspec_metric_add("copy_finished_tasks", int(task_count))
        if rows > 0:
            _hspec_metric_add("copy_finished_rows", int(rows))
        _hspec_store_cond.notify_all()


def _hspec_copy_worker() -> None:
    while True:
        task = _hspec_copy_queue.get()
        if task is None:
            return
        task_req_ids = [req_id for req_id, _, _ in task.req_slices]
        try:
            if task.event is not None:
                task.event.synchronize()

            from vllm_ascend.spec_decode.hspec_store import (
                get_hspec_local_collector,
                hspec_legacy_dataproto_hs_enabled,
            )

            legacy_dataproto_hs = hspec_legacy_dataproto_hs_enabled()
            collector = get_hspec_local_collector()
            if legacy_dataproto_hs:
                with _hspec_store_cond:
                    for req_id, start, end in task.req_slices:
                        rows = task.cpu_tensor[start:end]
                        if req_id not in _hspec_host_buffers:
                            _hspec_host_buffers[req_id] = []
                        _hspec_host_buffers[req_id].append(rows)
            else:
                for req_id, start, end in task.req_slices:
                    rows = task.cpu_tensor[start:end]
                    tokens = (
                        task.token_slices.get(str(req_id), [])
                        if task.token_slices is not None else []
                    )
                    expected = int(end) - int(start)
                    if len(tokens) != expected:
                        _hspec_metric_add("copy_token_hidden_len_mismatch")
                        continue
                    try:
                        if hasattr(collector, "append_hidden_and_tokens"):
                            collector.append_hidden_and_tokens(req_id, rows, tokens)
                        else:
                            collector.append_hidden_rows(req_id, rows)
                            collector.extend_tokens(req_id, tokens)
                    except Exception:
                        _hspec_metric_add("copy_worker_pair_write_error")
                        logger.exception(
                            "HSpec async copy worker failed to write pair for req_id=%s",
                            req_id,
                        )
        except Exception:
            logger.exception("HSpec async copy worker failed")
            _hspec_metric_add("copy_worker_error")
        finally:
            _hspec_finish_pending(
                task_req_ids,
                rows=int(getattr(task, "num_rows", 0)),
                task_count=1,
            )
            _hspec_pinned_pool.release(task.pool_handle)


def _hspec_accumulate_worker() -> None:
    while True:
        task = _hspec_accumulate_queue.get()
        if task is None:
            return
        try:
            if task.producer_event is not None:
                task.producer_event.synchronize()

            req_slices = task.req_slices
            flat_indices = task.flat_indices
            pending_req_ids = task.pending_req_ids
            if not pending_req_ids:
                continue

            from vllm_ascend.spec_decode.hspec_store import hspec_legacy_dataproto_hs_enabled

            allow_pool = not hspec_legacy_dataproto_hs_enabled()
            device = task.sample_hidden_states.device
            gather_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
            transfer_stream = _hspec_get_transfer_stream(device)
            pool_handle = None
            with torch.npu.stream(transfer_stream) if device.type == "npu" else nullcontext():
                selected_rows = task.sample_hidden_states.index_select(0, gather_indices).detach()
                if device.type == "npu":
                    cpu_tensor, pool_handle = _hspec_checkout_cpu_buffer(
                        selected_rows.shape,
                        dtype=selected_rows.dtype,
                        allow_pool=allow_pool,
                    )
                    cpu_tensor.copy_(selected_rows, non_blocking=True)
                    import torch_npu  # type: ignore
                    copy_event = torch_npu.npu.Event()
                    copy_event.record(transfer_stream)
                else:
                    cpu_tensor = selected_rows.cpu()
                    copy_event = None

            _hspec_copy_queue.put(
                _HSpecAsyncCopyTask(
                    req_slices=req_slices,
                    token_slices=task.token_slices,
                    cpu_tensor=cpu_tensor,
                    event=copy_event,
                    device_tensor_ref=selected_rows if copy_event is not None else None,
                    pool_handle=pool_handle,
                    num_rows=task.num_rows,
                )
            )
        except Exception:
            logger.exception("HSpec async accumulate worker failed")
            _hspec_metric_add("copy_submit_error")
            _hspec_finish_pending(
                list(task.pending_req_ids),
                rows=int(getattr(task, "num_rows", 0)),
                task_count=1,
            )


def _hspec_ensure_async_worker(start_accumulate_worker: bool = False) -> None:
    global _hspec_copy_thread, _hspec_accumulate_thread
    if _hspec_copy_thread is None or not _hspec_copy_thread.is_alive():
        _hspec_copy_thread = _threading.Thread(
            target=_hspec_copy_worker,
            name="hspec_async_copy_worker",
            daemon=True,
        )
        _hspec_copy_thread.start()
    if not start_accumulate_worker:
        return
    if _hspec_accumulate_thread is not None and _hspec_accumulate_thread.is_alive():
        return
    _hspec_accumulate_thread = _threading.Thread(
        target=_hspec_accumulate_worker,
        name="hspec_async_accumulate_worker",
        daemon=True,
    )
    _hspec_accumulate_thread.start()


def _hspec_shutdown_async_worker() -> None:
    for q, thread in (
        (_hspec_accumulate_queue, _hspec_accumulate_thread),
        (_hspec_copy_queue, _hspec_copy_thread),
    ):
        if thread is None or not thread.is_alive():
            continue
        try:
            q.put(None)
            thread.join(timeout=1.0)
        except Exception:
            pass


atexit.register(_hspec_shutdown_async_worker)


def _hspec_get_transfer_stream(device: torch.device):
    key = str(device)
    stream = _hspec_async_transfer_streams.get(key)
    if stream is not None:
        return stream
    if device.type == "npu":
        import torch_npu  # type: ignore

        stream = torch_npu.npu.Stream(device=device)
    else:
        stream = None
    _hspec_async_transfer_streams[key] = stream
    return stream


def _hspec_wait_pending_for_req(req_id: str) -> None:
    req_id = str(req_id)
    with _hspec_store_cond:
        while _hspec_async_pending_by_req.get(req_id, 0) > 0:
            _hspec_store_cond.wait(timeout=0.05)


def _hspec_wait_pending_all() -> None:
    with _hspec_store_cond:
        while _hspec_async_pending_total > 0:
            _hspec_store_cond.wait(timeout=0.05)


def hspec_set_collection_enabled(enabled: bool):
    """Enable or disable hidden state collection globally."""
    global _hspec_collection_enabled
    _hspec_collection_enabled = enabled


def hspec_is_collection_enabled() -> bool:
    """Check if hidden state collection is enabled."""
    return _hspec_collection_enabled


def hspec_append_step_hs(req_id: str, hidden_state: torch.Tensor):
    """Append one anchor hidden state for a request.

    Called by model_runner at each decode step.  The *hidden_state*
    must already be ``.clone()``'d on the caller side so that it is
    safe from in-place overwrites by subsequent steps.

    The tensor stays on the device until :func:`hspec_flush_and_get_all`
    or :func:`hspec_pop_request` is called.  This avoids per-token
    device→host synchronisation.

    Args:
        req_id:       Internal vLLM request id.
        hidden_state: 1-D tensor of shape ``(hidden_dim,)`` **already
                      cloned** on the compute device.
    """
    if not _hspec_collection_enabled:
        return
    req_id = str(req_id)
    from vllm_ascend.spec_decode.hspec_store import (
        get_hspec_local_collector,
        hspec_legacy_dataproto_hs_enabled,
    )

    if not hspec_legacy_dataproto_hs_enabled():
        logger.warning(
            "hspec_append_step_hs() is disabled in descriptor mode; "
            "use hspec_submit_accumulate_task() to avoid per-token host sync."
        )
        return

    rows = hidden_state.unsqueeze(0).cpu()
    if hspec_legacy_dataproto_hs_enabled():
        with _hspec_store_lock:
            if req_id not in _hspec_host_buffers:
                _hspec_host_buffers[req_id] = []
            _hspec_host_buffers[req_id].append(rows)


def hspec_submit_accumulate_task(
    req_ids: List[str],
    sample_hidden_states: torch.Tensor,
    valid_sampled_token_ids: List[List[int]],
    spec_decode_metadata: Any = None,
    accepted_prefix_lengths: Optional[List[int]] = None,
) -> bool:
    """Submit HSpec hidden-state accumulation work.

    The normal path computes the CPU-side row mapping on the caller, enqueues
    the selected-row gather on the caller's stream, and performs the D2H copy
    asynchronously.  The legacy full background-NPU path is kept only behind
    ``HSPEC_ASYNC_HS_ACCUMULATE=1`` for debugging/experiments.
    """
    if not _hspec_collection_enabled:
        return False
    if sample_hidden_states is None or not req_ids:
        return False

    str_req_ids = [str(req_id) for req_id in req_ids]
    normalized_token_ids = [
        [int(x) for x in sampled_ids] for sampled_ids in valid_sampled_token_ids
    ]
    normalized_accepts = (
        [int(x) for x in accepted_prefix_lengths]
        if accepted_prefix_lengths is not None else None
    )

    req_slices, flat_indices, pending_req_ids = _hspec_compute_req_slices(
        str_req_ids,
        normalized_token_ids,
        spec_decode_metadata,
        normalized_accepts,
    )
    if not pending_req_ids:
        return False

    token_slices = _hspec_compute_token_slices(
        str_req_ids,
        normalized_token_ids,
        req_slices,
    )
    if token_slices is None:
        return False
    if not _hspec_validate_token_slices(req_slices, token_slices):
        return False

    num_rows = len(flat_indices)
    if not hspec_should_collect_step(num_rows, len(pending_req_ids)):
        return False

    legacy_async = _hspec_use_legacy_async_accumulate()
    _hspec_ensure_async_worker(start_accumulate_worker=legacy_async)
    _hspec_reserve_pending(pending_req_ids, num_rows)
    device = sample_hidden_states.device
    if not legacy_async:
        pool_handle = None
        try:
            from vllm_ascend.spec_decode.hspec_store import hspec_legacy_dataproto_hs_enabled

            allow_pool = not hspec_legacy_dataproto_hs_enabled()
            contiguous_slice = _hspec_contiguous_index_slice(flat_indices)
            if contiguous_slice is not None:
                start, length = contiguous_slice
                selected_rows = sample_hidden_states.narrow(0, start, length).detach()
            else:
                gather_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
                selected_rows = sample_hidden_states.index_select(0, gather_indices).detach()
            if device.type == "npu":
                import torch_npu  # type: ignore

                current_stream = torch_npu.npu.current_stream(device)
                if _hspec_use_async_copy_stream():
                    copy_stream = _hspec_get_transfer_stream(device)
                    copy_stream.wait_stream(current_stream)
                else:
                    copy_stream = current_stream
                with torch.npu.stream(copy_stream):
                    cpu_tensor, pool_handle = _hspec_checkout_cpu_buffer(
                        selected_rows.shape,
                        dtype=selected_rows.dtype,
                        allow_pool=allow_pool,
                    )
                    cpu_tensor.copy_(selected_rows, non_blocking=True)
                    copy_event = torch_npu.npu.Event()
                    copy_event.record(copy_stream)
            else:
                cpu_tensor = selected_rows.cpu()
                copy_event = None
                pool_handle = None

            _hspec_copy_queue.put(
                _HSpecAsyncCopyTask(
                    req_slices=req_slices,
                    token_slices=token_slices,
                    cpu_tensor=cpu_tensor,
                    event=copy_event,
                    device_tensor_ref=selected_rows if copy_event is not None else None,
                    pool_handle=pool_handle,
                    num_rows=num_rows,
                )
            )
            _hspec_metric_add("copy_submitted_tasks")
            _hspec_metric_add("copy_submitted_rows", num_rows)
            return True
        except Exception:
            logger.exception("HSpec main-thread async copy submit failed")
            _hspec_metric_add("copy_submit_error")
            _hspec_pinned_pool.release(pool_handle)
            _hspec_finish_pending(pending_req_ids, rows=num_rows, task_count=1)
            return False

    producer_event = None
    if device.type == "npu":
        import torch_npu  # type: ignore

        producer_event = torch_npu.npu.Event()
        producer_event.record(torch_npu.npu.current_stream(device))

    _hspec_accumulate_queue.put(
        _HSpecAsyncAccumulateTask(
            req_ids=tuple(str_req_ids),
            req_slices=req_slices,
            flat_indices=flat_indices,
            pending_req_ids=pending_req_ids,
            token_slices=token_slices,
            num_rows=num_rows,
            sample_hidden_states=sample_hidden_states,
            valid_sampled_token_ids=normalized_token_ids,
            spec_decode_metadata=spec_decode_metadata,
            accepted_prefix_lengths=normalized_accepts,
            producer_event=producer_event,
        )
    )
    _hspec_metric_add("copy_submitted_tasks")
    _hspec_metric_add("copy_submitted_rows", num_rows)
    return True


def hspec_extend_step_tokens(req_id: str, token_ids: List[int]) -> None:
    """Append the exact token ids whose anchor states were collected this step."""
    if not _hspec_collection_enabled:
        return
    if not token_ids:
        return
    req_id = str(req_id)
    from vllm_ascend.spec_decode.hspec_store import (
        get_hspec_local_collector,
        hspec_legacy_dataproto_hs_enabled,
    )

    if hspec_legacy_dataproto_hs_enabled():
        with _hspec_store_lock:
            if req_id not in _hspec_token_buffers:
                _hspec_token_buffers[req_id] = []
            _hspec_token_buffers[req_id].extend(int(t) for t in token_ids)
    else:
        get_hspec_local_collector().extend_tokens(req_id, token_ids)


def hspec_flush_and_get_descriptors(
    request_id_to_prompt_id: Optional[Dict[str, str]] = None,
    epoch: int = -1,
    global_step: int = -1,
) -> Dict[str, Any]:
    """Flush in-flight copies and return small trajectory descriptors only."""
    start_ns = time.perf_counter_ns()
    _hspec_wait_pending_all()
    wait_ms = int((time.perf_counter_ns() - start_ns) / 1_000_000)
    _hspec_metric_add("flush_wait_count")
    _hspec_metric_add("flush_wait_ms_total", wait_ms)
    _hspec_metric_max("flush_wait_ms_max", wait_ms)
    from vllm_ascend.spec_decode.hspec_store import get_hspec_local_collector

    return get_hspec_local_collector().flush_descriptors(
        request_id_to_prompt_id=request_id_to_prompt_id,
        epoch=epoch,
        global_step=global_step,
    )


def hspec_flush_and_get_all() -> Dict[str, Dict[str, Any]]:
    """Flush **all** device buffers to CPU and return the complete store.

    This is the main entry point called by the rollout code *after*
    ``LLM.generate()`` returns.  It performs one device→host transfer
    per request (a single ``torch.stack(...).cpu()``), converts to
    float16 numpy arrays, and clears the device buffers.

    Returns:
        Dictionary mapping ``req_id`` → ``{
            'hidden_states': np.ndarray[(seq_len, hidden_dim), float16],
            'token_ids': list[int],
        }``.
    """
    _hspec_wait_pending_all()
    with _hspec_store_lock:
        result: Dict[str, Dict[str, Any]] = {}
        all_req_ids = set(_hspec_host_buffers.keys()) | set(_hspec_token_buffers.keys())
        for req_id in all_req_ids:
            tensors = _hspec_host_buffers.get(req_id, [])
            token_ids = list(_hspec_token_buffers.get(req_id, []))
            cpu_array = None
            if tensors:
                stacked = torch.cat(tensors, dim=0)
                cpu_array = stacked.to(dtype=torch.float16).numpy()
            result[str(req_id)] = {
                "hidden_states": cpu_array,
                "token_ids": token_ids,
            }
        _hspec_host_buffers.clear()
        _hspec_token_buffers.clear()
        return result


def hspec_pop_request(req_id: str) -> Optional[Dict[str, Any]]:
    """Pop hidden states for a *single* request.

    Used by the output-processor when a request finishes (streaming
    scenario).  Falls back to the device buffer if not yet flushed.

    Returns:
        ``{'hidden_states': np.ndarray | None, 'token_ids': list[int]}``,
        or ``None`` if no data is stored for *req_id*.
    """
    req_id = str(req_id)
    _hspec_wait_pending_for_req(req_id)
    with _hspec_store_lock:
        has_hs = req_id in _hspec_host_buffers
        has_tok = req_id in _hspec_token_buffers
        if not has_hs and not has_tok:
            return None
        tensors = _hspec_host_buffers.pop(req_id, [])
        token_ids = list(_hspec_token_buffers.pop(req_id, []))
        cpu_array = None
        if tensors:
            stacked = torch.cat(tensors, dim=0)
            cpu_array = stacked.to(dtype=torch.float16).numpy()
        return {
            "hidden_states": cpu_array,
            "token_ids": token_ids,
        }


def hspec_clear_store():
    """Clear all stored hidden states (both device and CPU)."""
    _hspec_wait_pending_all()
    with _hspec_store_lock:
        for tensors in _hspec_host_buffers.values():
            tensors.clear()
        _hspec_host_buffers.clear()
        _hspec_token_buffers.clear()
    try:
        from vllm_ascend.spec_decode.hspec_store import get_hspec_local_collector

        get_hspec_local_collector().clear_batch()
    except Exception:
        logger.debug("HSpec descriptor collector clear failed", exc_info=True)


class HSpecConfig:
    """Configuration class for HSpec."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        num_speculative_tokens: int = 5,
        min_match_len: int = 1,
        max_entries_per_table: int = 10000,
        initial_window_size: int = 8,
        max_window_size: int = 28,
        min_window_size: int = 2,
    ):
        """Initialize HSpec configuration.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match.
            num_speculative_tokens: Maximum number of draft tokens to propose.
            min_match_len: Minimum sequence length before attempting matching.
            max_entries_per_table: Maximum entries per prompt's query table.
            initial_window_size: Initial prediction window size.
            max_window_size: Maximum window size.
            min_window_size: Minimum window size.
        """
        self.similarity_threshold = similarity_threshold
        self.num_speculative_tokens = num_speculative_tokens
        self.min_match_len = min_match_len
        self.max_entries_per_table = max_entries_per_table
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size

    @classmethod
    def from_speculative_config(cls, spec_config) -> "HSpecConfig":
        """Create HSpecConfig from vLLM speculative config.
        
        Args:
            spec_config: vLLM speculative configuration.
        
        Returns:
            HSpecConfig instance.
        """
        return cls(
            similarity_threshold=getattr(spec_config, "hspec_similarity_threshold", 0.9),
            num_speculative_tokens=spec_config.num_speculative_tokens,
            min_match_len=getattr(spec_config, "hspec_min_match_len", 1),
        )
