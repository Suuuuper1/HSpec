# Adapted from
# https://github.com/volcengine/verl/blob/v0.6.0/verl/models/mcore/util.py
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from functools import wraps
import json
import math
import os
import torch
from torch import distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from verl.models.mcore.util import preprocess_packed_seqs, postprocess_packed_seqs
from mindspeed.utils import set_actual_seq_len, get_actual_seq_len


from patches.verl.utils.hybrid_data_parallel.utils import set_batch_hdp_group, get_batch_hdp_group

_HDP_MAX_TOKEN_LEN = None


def set_hdp_max_token_len(max_token_len):
    global _HDP_MAX_TOKEN_LEN
    _HDP_MAX_TOKEN_LEN = max_token_len


def get_hdp_max_token_len():
    return _HDP_MAX_TOKEN_LEN


def _is_primary_hdp_logger():
    return (
        getattr(mpu, "get_data_parallel_rank", lambda: 0)() == 0
        and getattr(mpu, "get_context_parallel_rank", lambda: 0)() == 0
        and getattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)() == 0
        and getattr(mpu, "get_pipeline_model_parallel_rank", lambda: 0)() == 0
    )


def _maybe_dump_hdp_groups(seq_len_effective, batch_hdp_group, cp_size):
    out_dir = os.environ.get("VERL_HDP_GROUP_DIR")
    if not out_dir or not _is_primary_hdp_logger():
        return

    if not hasattr(generate_hdp_group_from_batch, "hdp_log_step"):
        generate_hdp_group_from_batch.hdp_log_step = 0
    generate_hdp_group_from_batch.hdp_log_step += 1
    step = generate_hdp_group_from_batch.hdp_log_step

    if isinstance(seq_len_effective, torch.Tensor):
        seq_len_list = seq_len_effective.detach().cpu().tolist()
    else:
        seq_len_list = list(seq_len_effective)

    if batch_hdp_group is None:
        groups = [list(range(cp_size)) for _ in seq_len_list]
    else:
        groups = batch_hdp_group

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "hdp_group.jsonl")

    lines = []
    for idx, seqlen in enumerate(seq_len_list):
        group = groups[idx]
        entry = {
            "step": int(step),
            "seq_idx": int(idx),
            "seq_len": int(seqlen),
            "hdp_size": int(len(group)),
            "group": [int(x) for x in group],
        }
        lines.append(json.dumps(entry, ensure_ascii=True))

    with open(filename, "a") as f:
        f.write("\n".join(lines) + "\n")


def generate_hdp_group_from_batch(
    atten_mask: torch.Tensor,
    max_token_len: float | None = None,
    rank_overload_scale: float = 1.18,
):
    """
    Generate HDP groups for micro-batch to optimize Ring Attention communication in CP.
    
    Args:
        atten_mask: Attention mask tensor of shape [micro_batch_size, sequence_length]
        max_token_len: Max token length budget used for micro-batch packing.
    
    Returns:
        List of lists containing rank indices assigned to each sequence
        Example: [[0,1], [2], [3]] means sequence 0 uses 2 ranks, sequences 1&2 use 1 rank each
    
    Example:
        For sequences [1024, 512, 512] with 4 ranks:
        - seq0: (1024/2048)*4 = 2.0 ranks → assigned to ranks [0,1]
        - seq1: (512/2048)*4 = 1.0 rank → assigned to rank [2]
        - seq2: (512/2048)*4 = 1.0 rank → assigned to rank [3]
        Result: [[0,1], [2], [3]]
    
    Benefits:
        - seq0: communicates between ranks 0 and 1 only (internal Ring Attention)
        - seq1, seq2: no cross-rank communication needed (completely local to rank 2 and rank 3 respectively)
        - Total: significantly reduces cross-rank communication compared to traditional CP

    Note:
        This grouping strategy significantly reduces communication overhead in Ring Attention by
        minimizing unnecessary cross-rank communication, especially for sequences that can be
        processed completely within a single rank.
    """

    if not hasattr(generate_hdp_group_from_batch, "has_run"):
        # First run hdp, use cp instead, for initializing the cp group.
        generate_hdp_group_from_batch.has_run = True
        set_batch_hdp_group(None)
        return
    cp_size = mpu.get_context_parallel_world_size()
    seq_len_effective = atten_mask.sum(dim=1)
    if max_token_len is None:
        max_token_len = get_hdp_max_token_len()
    if max_token_len is None:
        max_token_len = float(seq_len_effective.sum().item())

    max_token_len *= rank_overload_scale

    per_rank_cap = max_token_len / cp_size
    seq_len_list = seq_len_effective.tolist()
    ranks_per_seq = [max(1, math.ceil(seqlen / per_rank_cap)) for seqlen in seq_len_list]
    total_ranks = sum(ranks_per_seq)

    if total_ranks > cp_size:
        raise RuntimeError(
            f"HDP group generation failed: total ranks {total_ranks} exceeds cp_size {cp_size}. "
            f"Batch sequence lengths: {seq_len_list}"
        )
    if total_ranks < cp_size:
        max_idx = max(range(len(seq_len_list)), key=lambda i: seq_len_list[i])
        ranks_per_seq[max_idx] += cp_size - total_ranks

    start_rank = 0
    batch_hdp_group = []
    for ranks in ranks_per_seq:
        batch_hdp_group.append(list(range(start_rank, start_rank + ranks)))
        start_rank += ranks
    if start_rank != cp_size:
        raise RuntimeError(
            f"HDP group generation failed: allocated {start_rank} ranks, expected {cp_size}. "
            f"Batch sequence lengths: {seq_len_list}"
        )
    if len(batch_hdp_group[0]) == cp_size:
        # All rank in one group, use cp instead
        set_batch_hdp_group(None)
        _maybe_dump_hdp_groups(seq_len_effective, None, cp_size)
        return
    set_batch_hdp_group(batch_hdp_group)
    _maybe_dump_hdp_groups(seq_len_effective, batch_hdp_group, cp_size)
    return


def check_load_balance(batch_hdp_group, ranks_per_seq, max_load_imbalance_threshold=1.3):
    """
    Check load balance across ranks in HDP grouping
    
    Args:
        batch_hdp_group: HDP batch grouping information, indicating which ranks process each sequence
        ranks_per_seq: number of ranks per sequence, representing computational load
        cp_size: total number of ranks in model parallelism
        max_load_imbalance_threshold: maximum load imbalance threshold, default 1.3
    
    Returns:
        bool: True if all rank loads are within threshold, False otherwise
    """
    cp_size = mpu.get_context_parallel_world_size()
    rank_load = [0] * cp_size
    for seq_idx, ranks in enumerate(ranks_per_seq):
        for rank in batch_hdp_group[seq_idx]:
            rank_load[rank] += ranks / len(batch_hdp_group[seq_idx])

    if any(load > max_load_imbalance_threshold for load in rank_load):
        return False
    return True


def pack_frac(pairs, start_rank, batch_hdp_group):
    """
    Pack sequences with fractional rank requirements into HDP groups.

    Routes to appropriate packing strategy based on available ranks.
    """
    cp_size = mpu.get_context_parallel_world_size()
    
    if start_rank >= cp_size:
        return _pack_into_existing_groups(pairs, start_rank, batch_hdp_group)
    else:
        return _pack_into_new_ranks(pairs, start_rank, batch_hdp_group)


def _pack_into_existing_groups(pairs, start_rank, batch_hdp_group):
    # Create generator for non-empty HDP groups with their original indices
    non_empty_group_index = (
        (hdp_group.copy(), index) 
        for index, hdp_group in enumerate(batch_hdp_group)
        if hdp_group
    )
    # Cache for reusing HDP groups when we run out of unique groups
    cached_hdp_group = []
    cached_indices = []
    # Process each pair (sequence index, fractional value)
    for i, pair in enumerate(pairs):
        try:
            hdp_group, orig_idx = next(non_empty_group_index)
            cached_hdp_group.append(hdp_group)
            cached_indices.append(orig_idx)
        except StopIteration:
            if cached_hdp_group:
                cycle_idx = i % len(cached_hdp_group)
                hdp_group = cached_hdp_group[cycle_idx]
                orig_idx = cached_indices[cycle_idx]
            else:
                continue
        # Merge current sequence's group with the selected HDP group
        merged_hdp_group = list(set(batch_hdp_group[pair[0]] + hdp_group))
        batch_hdp_group[pair[0]] = batch_hdp_group[orig_idx] = merged_hdp_group
    return batch_hdp_group, start_rank


def _pack_into_new_ranks(pairs, start_rank, batch_hdp_group):
    cp_size = mpu.get_context_parallel_world_size()
    # Sort pairs by fractional value in descending order (largest fractions first)
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    groups = [[] for _ in range(cp_size - start_rank)]
    group_sums = [0.0 for _ in range(cp_size - start_rank)]
    # Greedy assignment: assign each sequence to the group with smallest current sum
    for pair in pairs:
        idx = group_sums.index(min(group_sums))
        groups[idx].append(pair[0])
        group_sums[idx] += pair[1]
    for group in groups:
        if not group:
            continue
        new_group = [start_rank]
        for seq_idx in group:
            new_group.extend(batch_hdp_group[seq_idx])
        new_group.sort()
        for seq_idx in group:
            batch_hdp_group[seq_idx] = new_group.copy()
        start_rank += 1
    return batch_hdp_group, start_rank


def pack_sequences_into_buckets(seqlen_list: list[int], max_token_len: int) -> list[list[int]]:
    """
    Pack sequences into buckets based on rank capacity (cp_size).

    Each sequence consumes ceil(seqlen / (max_token_len / cp_size)) ranks, and each bucket
    has a capacity of cp_size ranks. We greedily pack sequences without exceeding cp_size.
    """
    cp_size = mpu.get_context_parallel_world_size()
    per_rank_cap = max_token_len / cp_size
    ranks_per_seq = [max(1, math.ceil(seqlen / per_rank_cap)) for seqlen in seqlen_list]

    indexed = list(enumerate(ranks_per_seq))
    indexed.sort(key=lambda x: (-x[1], -seqlen_list[x[0]]))

    buckets = []
    bucket_loads = []

    for idx, rank_need in indexed:
        if rank_need > cp_size:
            raise RuntimeError(
                f"Sequence rank requirement {rank_need} exceeds cp_size {cp_size}. "
                f"max_token_len={max_token_len}, seqlen={seqlen_list[idx]}"
            )
        best_bucket = None
        best_remaining = None
        for b, load in enumerate(bucket_loads):
            if load + rank_need <= cp_size:
                remaining = cp_size - (load + rank_need)
                if best_remaining is None or remaining < best_remaining:
                    best_bucket = b
                    best_remaining = remaining
                    if remaining == 0:
                        break
        if best_bucket is None:
            buckets.append([idx])
            bucket_loads.append(rank_need)
        else:
            buckets[best_bucket].append(idx)
            bucket_loads[best_bucket] += rank_need

    return buckets


def find_group(batch_hdp_group, cur_rank):
    indices = []
    local_group = None

    for idx, group in enumerate(batch_hdp_group):
        if cur_rank in group:
            if local_group is None:
                local_group = group
            if local_group != group:
                raise ValueError(f"rank{cur_rank} found in inconsistent group: {group} vs {local_group}")
            indices.append(idx)
    return indices, local_group


def preprocess_packed_seqs_hdp(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True
) -> tuple[torch.Tensor, PackedSeqParams]:
    
    batch_hdp_group = get_batch_hdp_group()  
    if batch_hdp_group is None:
        return preprocess_packed_seqs(input_ids, attention_mask, pre_process)
    
    cp_rank = mpu.get_context_parallel_rank()
    indices, local_group = find_group(batch_hdp_group, cp_rank)
    batch_size = len(indices)
    hdp_size = len(local_group)
    hdp_rank = local_group.index(cp_rank)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    align_size = tp_size * hdp_size * 2

    attention_mask_in_batch = attention_mask[indices]
    input_ids_in_batch = input_ids[indices]
    seqlens_in_batch = attention_mask_in_batch.sum(dim=-1, dtype=torch.int32)

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int64, device=input_ids_in_batch.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int64, device=input_ids_in_batch.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids_in_batch.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // hdp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if hdp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx: start_idx + seqlen] = input_ids_in_batch[i, attention_mask_in_batch[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // hdp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // hdp_size
            # split to 2 chunks
            d = input_ids_in_batch[i, attention_mask_in_batch[i]]
            input_ids_rmpad[start_idx: start_idx + half_seqlen] = d[
                half_seqlen * hdp_rank: half_seqlen * (hdp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (hdp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * hdp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen: start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    set_actual_seq_len((cu_seqlens_padded // hdp_size))
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs_hdp(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)
    batch_hdp_group = get_batch_hdp_group()
    if batch_hdp_group is None:
        return postprocess_packed_seqs(output, packed_seq_params, attention_mask, batch_size, seq_len, 
                                       post_process=post_process)
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        # gather output shape
        local_seq_len = torch.tensor(output.size(1), device=output.device)
        seq_len_list = [torch.empty_like(local_seq_len) for _ in range(cp_size)]
        dist.all_gather(seq_len_list, local_seq_len, group=mpu.get_context_parallel_group())
        max_seq_len = max(seq_len_list)

        output_padded = torch.nn.functional.pad(
            output, (0, 0) * (output.dim() - 2) + (0, max_seq_len - local_seq_len),
            value=0
        )

        output_gather = [torch.empty_like(output_padded) for _ in range(cp_size)]

        dist.all_gather(output_gather, output_padded.detach(), group=mpu.get_context_parallel_group())

        output_list = []
        for i, seq_len in enumerate(seq_len_list):
            if seq_len < max_seq_len:
                output_list.append(output_gather[i][:, :seq_len, ...])
            else:
                output_list.append(output_gather[i])
        output_list[mpu.get_context_parallel_rank()] = output

        cu_seqlens_q_padded = torch.zeros(batch_size + 1, dtype=torch.int64, device='cuda')
        cu_seqlens_q_padded[:len(packed_seq_params.cu_seqlens_q_padded)] = packed_seq_params.cu_seqlens_q_padded
        cu_seqlens_list = [torch.zeros_like(cu_seqlens_q_padded) for _ in range(cp_size)]
        dist.all_gather(cu_seqlens_list, cu_seqlens_q_padded, group=mpu.get_context_parallel_group())

        seq_index = [0] * cp_size
    else:
        output_list = [output]

    for i in range(batch_size):
        hdp_group = batch_hdp_group[i]
        cu_padded_cpu = cu_seqlens_list[hdp_group[0]].tolist()
        cu_seq_index = seq_index[hdp_group[0]]
        hdp_size = len(hdp_group)
        if hdp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[cu_seq_index]
            output_new[i, attention_mask[i]] = output_list[hdp_group[0]][0][start_idx: start_idx + s]
            seq_index[hdp_group[0]] += 1
            continue

        s_len_padded_chunk = (cu_padded_cpu[cu_seq_index + 1] - cu_padded_cpu[cu_seq_index]) // hdp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * hdp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(hdp_size):
            rank_j = batch_hdp_group[i][j]
            o = output_list[rank_j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[cu_seq_index] // hdp_size
            o0, o1 = (
                o[packed_start_idx: packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen: packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen: (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen: s_len_padded - j * half_seqlen] = o1
        tmp = tmp.to(output_new.dtype)
        output_new[i, attention_mask[i]] = tmp[:s_len]
        seq_index[hdp_group[0]] += 1

    return output_new
