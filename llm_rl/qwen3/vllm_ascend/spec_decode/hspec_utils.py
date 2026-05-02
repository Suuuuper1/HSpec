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
import logging
import hashlib
import os
import struct
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - fallback for older torch variants
    from torch.autograd.profiler import record_function as _record_function

logger = logging.getLogger(__name__)

_hspec_profile_local = threading.local()


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
    steps = _parse_profile_steps(os.getenv("HSPEC_PROFILE_STEPS", "5,31"))
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
        try:
            import torch_npu

            stream = torch_npu.npu.current_stream() if use_npu_stream else None
            domain = hspec_profile_domain()
            range_id = torch_npu.npu.mstx.range_start(name, stream, domain=domain)
            try:
                yield
            finally:
                torch_npu.npu.mstx.range_end(range_id, domain=domain)
            return
        except Exception:
            # Fallback to record_function if mstx is unavailable.
            pass

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

# Device-side accumulation buffers:  req_id -> list of (hidden_dim,) tensors
_hspec_device_buffers: Dict[str, List[torch.Tensor]] = {}
# Token-side accumulation buffers: req_id -> list of accepted/generated token ids
_hspec_token_buffers: Dict[str, List[int]] = {}

# Whether collection is enabled (set by model_runner at init)
_hspec_collection_enabled: bool = False


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
    with _hspec_store_lock:
        if req_id not in _hspec_device_buffers:
            _hspec_device_buffers[req_id] = []
        _hspec_device_buffers[req_id].append(hidden_state)


def hspec_extend_step_tokens(req_id: str, token_ids: List[int]) -> None:
    """Append the exact token ids whose anchor states were collected this step."""
    if not _hspec_collection_enabled:
        return
    if not token_ids:
        return
    req_id = str(req_id)
    with _hspec_store_lock:
        if req_id not in _hspec_token_buffers:
            _hspec_token_buffers[req_id] = []
        _hspec_token_buffers[req_id].extend(int(t) for t in token_ids)


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
    with _hspec_store_lock:
        result: Dict[str, Dict[str, Any]] = {}
        all_req_ids = set(_hspec_device_buffers.keys()) | set(_hspec_token_buffers.keys())
        for req_id in all_req_ids:
            tensors = _hspec_device_buffers.get(req_id, [])
            token_ids = list(_hspec_token_buffers.get(req_id, []))
            cpu_array = None
            if tensors:
                # Stack all step tensors → (seq_len, hidden_dim)
                stacked = torch.stack(tensors)
                # Single device→host transfer per request
                cpu_array = stacked.to(dtype=torch.float16).cpu().numpy()
            result[str(req_id)] = {
                "hidden_states": cpu_array,
                "token_ids": token_ids,
            }
        _hspec_device_buffers.clear()
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
    with _hspec_store_lock:
        has_hs = req_id in _hspec_device_buffers
        has_tok = req_id in _hspec_token_buffers
        if not has_hs and not has_tok:
            return None
        tensors = _hspec_device_buffers.pop(req_id, [])
        token_ids = list(_hspec_token_buffers.pop(req_id, []))
        cpu_array = None
        if tensors:
            stacked = torch.stack(tensors)
            cpu_array = stacked.to(dtype=torch.float16).cpu().numpy()
        return {
            "hidden_states": cpu_array,
            "token_ids": token_ids,
        }


def hspec_clear_store():
    """Clear all stored hidden states (both device and CPU)."""
    with _hspec_store_lock:
        # Explicitly delete tensors to free device memory promptly
        for tensors in _hspec_device_buffers.values():
            tensors.clear()
        _hspec_device_buffers.clear()
        _hspec_token_buffers.clear()


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
