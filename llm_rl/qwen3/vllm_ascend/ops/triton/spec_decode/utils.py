# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

from vllm.triton_utils import tl, triton


@triton.jit
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)


@triton.jit
def compute_rejected_tokens_kernel(
    cu_num_draft_tokens_ptr,
    valid_sampled_tokens_count_ptr,
    num_rejected_tokens_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """Device-only form of rejected = draft + 1 - valid."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    safe_offsets = tl.where(mask, offsets, 0)
    cumulative = tl.load(cu_num_draft_tokens_ptr + safe_offsets)
    previous_offsets = tl.where(safe_offsets > 0, safe_offsets - 1, 0)
    previous = tl.load(
        cu_num_draft_tokens_ptr + previous_offsets,
        mask=mask & (safe_offsets > 0),
        other=0,
    )
    draft_count = cumulative - previous
    valid_count = tl.load(valid_sampled_tokens_count_ptr + safe_offsets)
    rejected = tl.where(draft_count > 0, draft_count + 1 - valid_count, 0)
    tl.store(num_rejected_tokens_ptr + safe_offsets, rejected, mask=mask)


@triton.jit
def sanitize_and_copy_context_kernel(
    target_positions_ptr,
    context_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    batch_size,
    block_size,
    null_block_id,
    PAD_SLOT_ID: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Copy target context and make rejected/null cache writes impossible."""
    request_index = tl.program_id(0)
    start = tl.load(query_start_loc_ptr + request_index)
    end = tl.load(query_start_loc_ptr + request_index + 1)
    rejected = tl.load(num_rejected_tokens_ptr + request_index)
    valid_end = end - rejected

    for block_start in tl.range(start, end, TILE_SIZE):
        offsets = block_start + tl.arange(0, TILE_SIZE)
        mask = offsets < end
        safe_offsets = tl.where(mask, offsets, start)
        positions = tl.load(target_positions_ptr + safe_offsets)
        slots = tl.load(context_slot_mapping_ptr + safe_offsets)
        invalid = offsets >= valid_end
        if HAS_NULL_BLOCK:
            physical_block = slots // block_size
            invalid = invalid | ((slots >= 0) & (physical_block == null_block_id))
        slots = tl.where(invalid, PAD_SLOT_ID, slots)
        tl.store(out_context_positions_ptr + safe_offsets, positions, mask=mask)
        tl.store(out_context_slot_mapping_ptr + safe_offsets, slots, mask=mask)


@triton.jit
def expand_parallel_queries_kernel(
    next_token_ids_ptr,
    target_positions_ptr,
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_slot_mapping_ptr,
    out_sample_indices_ptr,
    block_table_ptr,
    block_table_stride,
    max_blocks_per_req,
    query_start_loc_ptr,
    seq_lens_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    null_block_id,
    num_query_per_req,
    num_speculative_tokens,
    batch_size,
    PAD_SLOT_ID: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Expand DFlash/DSpark query ids, positions, physical slots and samples."""
    program_id = tl.program_id(0)
    total_queries = batch_size * num_query_per_req
    offsets = program_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < total_queries

    # Some Ascend compiler/runtime combinations form addresses before applying
    # vector masks. Keep every address valid, including inactive tail lanes.
    safe_offsets = tl.where(mask, offsets, 0)
    request_index = safe_offsets // num_query_per_req
    query_index = safe_offsets % num_query_per_req

    context_end = tl.load(query_start_loc_ptr + request_index + 1)
    rejected = tl.load(num_rejected_tokens_ptr + request_index)
    valid_context_end = context_end - rejected
    sequence_length = tl.load(seq_lens_ptr + request_index)
    effective_sequence_length = sequence_length - rejected
    last_position = tl.load(target_positions_ptr + valid_context_end - 1)
    tl.store(
        out_query_positions_ptr + safe_offsets,
        last_position + 1 + query_index,
        mask=mask,
    )

    linear_slot_position = effective_sequence_length + query_index
    block_index = linear_slot_position // block_size
    block_mask = mask & (block_index < max_blocks_per_req)
    safe_block_index = tl.where(block_mask, block_index, 0)
    block_id = tl.load(
        block_table_ptr
        + request_index * block_table_stride
        + safe_block_index
    ).to(tl.int64)
    slot = block_id * block_size + linear_slot_position % block_size
    invalid_slot = (~block_mask) | (block_id < 0)
    if HAS_NULL_BLOCK:
        invalid_slot = invalid_slot | (block_id == null_block_id)
    slot = tl.where(invalid_slot, PAD_SLOT_ID, slot)
    tl.store(out_query_slot_mapping_ptr + safe_offsets, slot, mask=mask)

    next_token = tl.load(next_token_ids_ptr + request_index)
    input_id = tl.where(
        query_index == 0, next_token, parallel_drafting_token_id
    )
    tl.store(out_input_ids_ptr + safe_offsets, input_id, mask=mask)

    if SAMPLE_FROM_ANCHOR:
        sample_output_index = (
            request_index * num_speculative_tokens + query_index
        )
        tl.store(
            out_sample_indices_ptr + sample_output_index,
            safe_offsets,
            mask=mask,
        )
    else:
        sample_mask = mask & (query_index > 0)
        sample_query_index = tl.where(query_index > 0, query_index - 1, 0)
        sample_output_index = (
            request_index * num_speculative_tokens + sample_query_index
        )
        tl.store(
            out_sample_indices_ptr + sample_output_index,
            safe_offsets,
            mask=sample_mask,
        )
