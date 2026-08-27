# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Framework-independent oracle for old-ABI parallel-block input geometry.

This module intentionally uses only Python data structures. Runtime code must
use device kernels; tests use this implementation as the independent semantic
reference for DFlash and both DSpark layouts.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

PAD_SLOT_ID = -1
ParallelBlockMethod = Literal["dflash", "dspark"]


class ParallelBlockLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class ParallelBlockGeometry:
    num_speculative_tokens: int
    num_queries: int
    lookahead_tokens: int
    additional_slots: int
    sample_offset: int
    sample_from_anchor: bool


@dataclass(frozen=True)
class ParallelBlockReferenceOutput:
    rejected_counts: tuple[int, ...]
    effective_context_ends: tuple[int, ...]
    effective_sequence_lengths: tuple[int, ...]
    context_positions: tuple[int, ...]
    context_slot_mapping: tuple[int, ...]
    query_input_ids: tuple[int, ...]
    query_positions: tuple[int, ...]
    query_slot_mapping: tuple[int, ...]
    sample_indices: tuple[int, ...]


def get_parallel_block_geometry(
    method: ParallelBlockMethod,
    num_speculative_tokens: int,
    *,
    sample_from_anchor: bool | None = None,
) -> ParallelBlockGeometry:
    if method not in ("dflash", "dspark"):
        raise ParallelBlockLayoutError(f"unsupported method: {method!r}")
    if not 1 <= num_speculative_tokens <= 15:
        raise ParallelBlockLayoutError(
            "num_speculative_tokens must be in the certified range [1, 15]"
        )

    if method == "dflash":
        if sample_from_anchor is True:
            raise ParallelBlockLayoutError("DFlash cannot sample from the anchor")
        anchor = False
    else:
        anchor = True if sample_from_anchor is None else sample_from_anchor

    num_queries = (
        num_speculative_tokens
        if method == "dspark" and anchor
        else num_speculative_tokens + 1
    )
    return ParallelBlockGeometry(
        num_speculative_tokens=num_speculative_tokens,
        num_queries=num_queries,
        lookahead_tokens=num_queries,
        additional_slots=num_queries - 1,
        sample_offset=0 if anchor else 1,
        sample_from_anchor=anchor,
    )


def compute_rejected_counts(
    cumulative_draft_counts: Sequence[int],
    valid_sampled_counts: Sequence[int],
) -> tuple[int, ...]:
    """Apply the old padded-verification correction without device state."""
    if len(cumulative_draft_counts) != len(valid_sampled_counts):
        raise ParallelBlockLayoutError(
            "cumulative_draft_counts and valid_sampled_counts must have equal length"
        )

    result: list[int] = []
    previous = 0
    for request_index, (cumulative, valid) in enumerate(
        zip(cumulative_draft_counts, valid_sampled_counts)
    ):
        draft_count = cumulative - previous
        previous = cumulative
        if draft_count < 0:
            raise ParallelBlockLayoutError("cumulative draft counts must be monotonic")
        if draft_count == 0:
            if valid < 0:
                raise ParallelBlockLayoutError("valid sampled count cannot be negative")
            result.append(0)
            continue

        rejected = draft_count + 1 - valid
        if rejected < 0 or rejected > draft_count:
            raise ParallelBlockLayoutError(
                f"request {request_index} has invalid valid_sampled_count={valid} "
                f"for draft_count={draft_count}"
            )
        result.append(rejected)
    return tuple(result)


def required_block_count(
    sequence_length: int,
    lookahead_tokens: int,
    block_size: int,
) -> int:
    """Blocks required by scheduler allocation for one scheduled query."""
    if sequence_length < 0 or lookahead_tokens < 0 or block_size <= 0:
        raise ParallelBlockLayoutError("invalid slot-reservation arguments")
    total_tokens = sequence_length + 1 + lookahead_tokens
    return max(1, (total_tokens + block_size - 1) // block_size)


def build_parallel_block_reference(
    *,
    method: ParallelBlockMethod,
    num_speculative_tokens: int,
    next_token_ids: Sequence[int],
    target_positions: Sequence[int],
    context_slot_mapping: Sequence[int],
    query_start_locs: Sequence[int],
    sequence_lengths: Sequence[int],
    block_table: Sequence[Sequence[int]],
    block_size: int,
    parallel_token_id: int,
    rejected_counts: Sequence[int] | None = None,
    sample_from_anchor: bool | None = None,
    null_block_id: int | None = None,
    max_model_len: int | None = None,
) -> ParallelBlockReferenceOutput:
    geometry = get_parallel_block_geometry(
        method,
        num_speculative_tokens,
        sample_from_anchor=sample_from_anchor,
    )
    batch_size = len(next_token_ids)
    if block_size <= 0:
        raise ParallelBlockLayoutError("block_size must be positive")
    if len(query_start_locs) != batch_size + 1 or query_start_locs[0] != 0:
        raise ParallelBlockLayoutError("query_start_locs must partition the context")
    if len(sequence_lengths) != batch_size or len(block_table) != batch_size:
        raise ParallelBlockLayoutError("per-request metadata length mismatch")
    if query_start_locs[-1] != len(target_positions):
        raise ParallelBlockLayoutError("query_start_locs[-1] must equal context size")
    if len(context_slot_mapping) != len(target_positions):
        raise ParallelBlockLayoutError("context position/slot lengths must match")

    if rejected_counts is None:
        rejected = (0,) * batch_size
    else:
        rejected = tuple(int(value) for value in rejected_counts)
        if len(rejected) != batch_size:
            raise ParallelBlockLayoutError("rejected_counts length mismatch")

    out_context_positions = tuple(int(value) for value in target_positions)
    out_context_slots = [int(value) for value in context_slot_mapping]
    query_ids: list[int] = []
    query_positions: list[int] = []
    query_slots: list[int] = []
    sample_indices: list[int] = []
    effective_ends: list[int] = []
    effective_lengths: list[int] = []

    for request_index in range(batch_size):
        start = query_start_locs[request_index]
        end = query_start_locs[request_index + 1]
        if start > end:
            raise ParallelBlockLayoutError("query_start_locs must be monotonic")
        num_rejected = rejected[request_index]
        context_length = end - start
        sequence_length = int(sequence_lengths[request_index])
        if num_rejected < 0 or num_rejected > context_length:
            raise ParallelBlockLayoutError("rejected suffix exceeds request context")
        if num_rejected > sequence_length:
            raise ParallelBlockLayoutError("rejected suffix exceeds sequence length")

        valid_end = end - num_rejected
        effective_length = sequence_length - num_rejected
        if valid_end <= start:
            raise ParallelBlockLayoutError(
                "parallel drafting requires at least one valid anchor context row"
            )
        if max_model_len is not None and (
            effective_length + geometry.num_queries > max_model_len
        ):
            raise ParallelBlockLayoutError(
                f"request {request_index} fixed-K query block exceeds max_model_len"
            )
        effective_ends.append(valid_end)
        effective_lengths.append(effective_length)

        for context_index in range(start, end):
            slot = out_context_slots[context_index]
            is_rejected = context_index >= valid_end
            is_null = (
                null_block_id is not None
                and slot >= 0
                and slot // block_size == null_block_id
            )
            if is_rejected or is_null:
                out_context_slots[context_index] = PAD_SLOT_ID

        last_position = int(target_positions[valid_end - 1])
        query_base = request_index * geometry.num_queries
        for query_index in range(geometry.num_queries):
            query_ids.append(
                int(next_token_ids[request_index])
                if query_index == 0
                else parallel_token_id
            )
            query_position = last_position + 1 + query_index
            if max_model_len is not None and query_position >= max_model_len:
                raise ParallelBlockLayoutError("query position exceeds max_model_len")
            query_positions.append(query_position)

            linear_slot_position = effective_length + query_index
            block_index = linear_slot_position // block_size
            if block_index >= len(block_table[request_index]):
                raise ParallelBlockLayoutError(
                    f"request {request_index} has no allocated block for "
                    f"linear position {linear_slot_position}"
                )
            block_id = int(block_table[request_index][block_index])
            if block_id < 0 or (
                null_block_id is not None and block_id == null_block_id
            ):
                query_slots.append(PAD_SLOT_ID)
            else:
                query_slots.append(
                    block_id * block_size + linear_slot_position % block_size
                )

        sample_indices.extend(
            query_base + geometry.sample_offset + offset
            for offset in range(num_speculative_tokens)
        )

    return ParallelBlockReferenceOutput(
        rejected_counts=rejected,
        effective_context_ends=tuple(effective_ends),
        effective_sequence_lengths=tuple(effective_lengths),
        context_positions=out_context_positions,
        context_slot_mapping=tuple(out_context_slots),
        query_input_ids=tuple(query_ids),
        query_positions=tuple(query_positions),
        query_slot_mapping=tuple(query_slots),
        sample_indices=tuple(sample_indices),
    )
