import random

import pytest

from vllm_ascend.spec_decode.parallel_block_reference import (
    PAD_SLOT_ID,
    ParallelBlockLayoutError,
    build_parallel_block_reference,
    compute_rejected_counts,
    get_parallel_block_geometry,
    required_block_count,
)


@pytest.mark.parametrize(
    "method,anchor,k,expected",
    [
        ("dflash", False, 1, (2, 2, 1, 1)),
        ("dflash", False, 15, (16, 16, 15, 1)),
        ("dspark", True, 7, (7, 7, 6, 0)),
        ("dspark", False, 7, (8, 8, 7, 1)),
    ],
)
def test_geometry_contract(method, anchor, k, expected):
    geometry = get_parallel_block_geometry(
        method, k, sample_from_anchor=anchor
    )
    assert (
        geometry.num_queries,
        geometry.lookahead_tokens,
        geometry.additional_slots,
        geometry.sample_offset,
    ) == expected


def test_rejected_count_formula_and_validation():
    assert compute_rejected_counts([0, 3, 10], [1, 2, 4]) == (0, 2, 4)
    assert compute_rejected_counts([1, 8, 23], [2, 1, 1]) == (0, 7, 15)
    with pytest.raises(ParallelBlockLayoutError, match="monotonic"):
        compute_rejected_counts([3, 2], [1, 1])
    with pytest.raises(ParallelBlockLayoutError, match="invalid"):
        compute_rejected_counts([3], [0])


def test_rejected_count_formula_exhaustive_certified_range():
    cumulative = []
    valid_counts = []
    expected = []
    total = 0
    for draft_count in range(16):
        counts = (0, 1) if draft_count == 0 else range(1, draft_count + 2)
        for valid_count in counts:
            total += draft_count
            cumulative.append(total)
            valid_counts.append(valid_count)
            expected.append(
                0 if draft_count == 0 else draft_count + 1 - valid_count
            )
    assert compute_rejected_counts(cumulative, valid_counts) == tuple(expected)


@pytest.mark.parametrize("sequence_length", [0, 1, 14, 15, 16, 17, 30, 31, 32])
@pytest.mark.parametrize("k", [1, 7, 15])
@pytest.mark.parametrize(
    "method,anchor", [("dflash", False), ("dspark", True), ("dspark", False)]
)
def test_slot_reservation_covers_every_query(
    sequence_length, k, method, anchor
):
    geometry = get_parallel_block_geometry(method, k, sample_from_anchor=anchor)
    blocks = required_block_count(
        sequence_length, geometry.lookahead_tokens, block_size=16
    )
    highest_query_slot = sequence_length + geometry.num_queries - 1
    assert blocks * 16 > highest_query_slot


@pytest.mark.parametrize(
    "method,anchor", [("dflash", False), ("dspark", True), ("dspark", False)]
)
@pytest.mark.parametrize("k", [1, 7, 15])
def test_reference_layout_rejection_noncontiguous_and_null(method, anchor, k):
    geometry = get_parallel_block_geometry(method, k, sample_from_anchor=anchor)
    context_lengths = (3, 18, 33)
    starts = [0]
    for length in context_lengths:
        starts.append(starts[-1] + length)
    positions = tuple(
        request * 64 + position * 2
        for request, length in enumerate(context_lengths)
        for position in range(length)
    )
    block_table = (
        (99, 3, 8, 4),
        (11, 2, 13, 5),
        (17, 6, 99, 9),
    )
    slots = tuple(
        block_table[request][position // 16] * 16 + position % 16
        for request, length in enumerate(context_lengths)
        for position in range(length)
    )
    rejected = (0, min(k, 7), min(k, 15))
    output = build_parallel_block_reference(
        method=method,
        num_speculative_tokens=k,
        next_token_ids=(101, 102, 103),
        target_positions=positions,
        context_slot_mapping=slots,
        query_start_locs=starts,
        sequence_lengths=context_lengths,
        block_table=block_table,
        block_size=16,
        parallel_token_id=777,
        rejected_counts=rejected,
        sample_from_anchor=anchor,
        null_block_id=99,
        max_model_len=256,
    )

    assert len(output.query_input_ids) == 3 * geometry.num_queries
    assert len(output.sample_indices) == 3 * k
    for request in range(3):
        base = request * geometry.num_queries
        assert output.query_input_ids[base] == 101 + request
        assert output.query_input_ids[base + 1 : base + geometry.num_queries] == (
            777,
        ) * (geometry.num_queries - 1)
        assert output.sample_indices[request * k : (request + 1) * k] == tuple(
            base + geometry.sample_offset + offset for offset in range(k)
        )
        end = starts[request + 1]
        for index in range(end - rejected[request], end):
            assert output.context_slot_mapping[index] == PAD_SLOT_ID

    # Request 0's valid context and query both resolve into the null block.
    assert output.context_slot_mapping[0] == PAD_SLOT_ID
    assert output.query_slot_mapping[0] == PAD_SLOT_ID


@pytest.mark.parametrize("batch_size", [1, 2, 7, 16])
@pytest.mark.parametrize("k", [1, 7, 15])
@pytest.mark.parametrize(
    "method,anchor", [("dflash", False), ("dspark", True), ("dspark", False)]
)
def test_reference_seeded_variable_batches(batch_size, k, method, anchor):
    rng = random.Random((batch_size << 16) + (k << 4) + int(anchor))
    geometry = get_parallel_block_geometry(method, k, sample_from_anchor=anchor)
    context_lengths = [rng.randint(1, 40) for _ in range(batch_size)]
    rejected = [rng.randint(0, min(k, length - 1)) for length in context_lengths]
    starts = [0]
    positions = []
    slots = []
    block_table = []
    for request_index, (length, num_rejected) in enumerate(
        zip(context_lengths, rejected)
    ):
        starts.append(starts[-1] + length)
        positions.extend(request_index * 64 + index * 2 for index in range(length))
        effective_length = length - num_rejected
        num_blocks = max(
            1,
            (max(length, effective_length + geometry.num_queries) + 15) // 16,
        )
        row = [100 + request_index * 64 + block * 3 for block in range(num_blocks)]
        block_table.append(row)
        slots.extend(row[index // 16] * 16 + index % 16 for index in range(length))

    output = build_parallel_block_reference(
        method=method,
        num_speculative_tokens=k,
        next_token_ids=[1000 + request for request in range(batch_size)],
        target_positions=positions,
        context_slot_mapping=slots,
        query_start_locs=starts,
        sequence_lengths=context_lengths,
        block_table=block_table,
        block_size=16,
        parallel_token_id=777,
        rejected_counts=rejected,
        sample_from_anchor=anchor,
        max_model_len=4096,
    )
    assert len(output.query_input_ids) == batch_size * geometry.num_queries
    assert len(output.query_slot_mapping) == batch_size * geometry.num_queries
    assert len(output.sample_indices) == batch_size * k
    assert output.effective_sequence_lengths == tuple(
        length - count for length, count in zip(context_lengths, rejected)
    )


def test_reference_max_length_and_missing_block_fail_closed():
    common = dict(
        method="dflash",
        num_speculative_tokens=7,
        next_token_ids=(1,),
        target_positions=tuple(range(15)),
        context_slot_mapping=tuple(range(15)),
        query_start_locs=(0, 15),
        sequence_lengths=(15,),
        block_size=16,
        parallel_token_id=99,
    )
    with pytest.raises(ParallelBlockLayoutError, match="max_model_len"):
        build_parallel_block_reference(
            **common, block_table=((2, 3),), max_model_len=22
        )
    with pytest.raises(ParallelBlockLayoutError, match="no allocated block"):
        build_parallel_block_reference(**common, block_table=((2,),))
