import inspect

import pytest
import torch

from vllm.triton_utils import triton
from vllm_ascend.ops.triton.spec_decode.utils import (
    compute_rejected_tokens_kernel,
    expand_parallel_queries_kernel,
    sanitize_and_copy_context_kernel,
)
from vllm_ascend.spec_decode.parallel_block_reference import (
    PAD_SLOT_ID,
    build_parallel_block_reference,
    compute_rejected_counts,
    get_parallel_block_geometry,
)


def _require_npu():
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            pytest.skip("torch-npu reports no available NPU")
        torch.empty(1, device="npu")
    except Exception as error:
        pytest.skip(f"NPU runtime unavailable: {type(error).__name__}: {error}")


def test_parallel_input_kernels_have_no_host_or_allocation_calls():
    source = "\n".join(
        inspect.getsource(kernel.fn)
        for kernel in (
            compute_rejected_tokens_kernel,
            sanitize_and_copy_context_kernel,
            expand_parallel_queries_kernel,
        )
    )
    for forbidden in (".item(", ".tolist(", "synchronize(", "torch.empty"):
        assert forbidden not in source


def test_compute_rejected_tokens_kernel_matches_cpu():
    _require_npu()
    cumulative_values = []
    valid_values = []
    total = 0
    for draft_count in range(16):
        counts = (0, 1) if draft_count == 0 else range(1, draft_count + 2)
        for valid_count in counts:
            total += draft_count
            cumulative_values.append(total)
            valid_values.append(valid_count)
    cumulative = torch.tensor(cumulative_values, dtype=torch.int32, device="npu")
    valid = torch.tensor(valid_values, dtype=torch.int32, device="npu")
    output = torch.empty_like(cumulative)
    compute_rejected_tokens_kernel[(1,)](
        cumulative, valid, output, len(cumulative_values), BLOCK_SIZE=256
    )
    torch.npu.synchronize()
    expected = compute_rejected_counts(cumulative_values, valid_values)
    assert tuple(output.cpu().tolist()) == expected


@pytest.mark.parametrize(
    "method,anchor,k,rejected",
    [
        ("dflash", False, 1, (0, 1, 0)),
        ("dflash", False, 7, (0, 3, 7)),
        ("dflash", False, 15, (0, 7, 15)),
        ("dspark", True, 1, (0, 1, 0)),
        ("dspark", True, 7, (0, 3, 7)),
        ("dspark", True, 15, (0, 7, 15)),
        ("dspark", False, 1, (0, 1, 0)),
        ("dspark", False, 7, (0, 3, 7)),
        ("dspark", False, 15, (0, 7, 15)),
    ],
)
def test_sanitize_and_expand_kernels_match_cpu(method, anchor, k, rejected):
    _require_npu()
    geometry = get_parallel_block_geometry(method, k, sample_from_anchor=anchor)
    context_lengths = (2, 20, 33)
    query_start_locs = [0]
    for length in context_lengths:
        query_start_locs.append(query_start_locs[-1] + length)
    positions = [
        request * 64 + position * 2
        for request, length in enumerate(context_lengths)
        for position in range(length)
    ]
    block_table = [
        [99, 3, 8, 4],
        [11, 2, 13, 5],
        [17, 6, 19, 9],
    ]
    slots = [
        block_table[request][position // 16] * 16 + position % 16
        for request, length in enumerate(context_lengths)
        for position in range(length)
    ]
    reference = build_parallel_block_reference(
        method=method,
        num_speculative_tokens=k,
        next_token_ids=(101, 102, 103),
        target_positions=positions,
        context_slot_mapping=slots,
        query_start_locs=query_start_locs,
        sequence_lengths=context_lengths,
        block_table=block_table,
        block_size=16,
        parallel_token_id=777,
        rejected_counts=rejected,
        sample_from_anchor=anchor,
        null_block_id=99,
        max_model_len=256,
    )

    device = "npu"
    positions_npu = torch.tensor(positions, dtype=torch.int32, device=device)
    slots_npu = torch.tensor(slots, dtype=torch.int32, device=device)
    starts_npu = torch.tensor(
        query_start_locs, dtype=torch.int32, device=device
    )
    rejected_npu = torch.tensor(rejected, dtype=torch.int32, device=device)
    sequence_lengths_npu = torch.tensor(
        context_lengths, dtype=torch.int32, device=device
    )
    next_tokens_npu = torch.tensor(
        [101, 102, 103], dtype=torch.int64, device=device
    )
    block_table_npu = torch.tensor(
        block_table, dtype=torch.int32, device=device
    )

    context_positions_guard = torch.full(
        (len(positions) + 8,), -31337, dtype=torch.int32, device=device
    )
    context_slots_guard = torch.full(
        (len(slots) + 8,), -31337, dtype=torch.int32, device=device
    )
    context_positions_out = context_positions_guard[: len(positions)]
    context_slots_out = context_slots_guard[: len(slots)]
    sanitize_and_copy_context_kernel[(3,)](
        positions_npu,
        slots_npu,
        context_positions_out,
        context_slots_out,
        starts_npu,
        rejected_npu,
        3,
        16,
        99,
        PAD_SLOT_ID=PAD_SLOT_ID,
        HAS_NULL_BLOCK=True,
        TILE_SIZE=64,
    )
    # Keep synchronization in the test only so a device fault is attributed to
    # the exact kernel that launched it. Production callers remain async.
    torch.npu.synchronize()

    query_count = 3 * geometry.num_queries
    query_ids_guard = torch.full(
        (query_count + 8,), -31337, dtype=torch.int64, device=device
    )
    query_positions_guard = torch.full(
        (query_count + 8,), -31337, dtype=torch.int32, device=device
    )
    query_slots_guard = torch.full(
        (query_count + 8,), -31337, dtype=torch.int32, device=device
    )
    sample_indices_guard = torch.full(
        (3 * k + 8,), -31337, dtype=torch.int32, device=device
    )
    query_ids_out = query_ids_guard[:query_count]
    query_positions_out = query_positions_guard[:query_count]
    query_slots_out = query_slots_guard[:query_count]
    sample_indices_out = sample_indices_guard[: 3 * k]
    grid = (max(1, triton.cdiv(query_count, 64)),)
    expand_parallel_queries_kernel[grid](
        next_tokens_npu,
        positions_npu,
        query_ids_out,
        query_positions_out,
        query_slots_out,
        sample_indices_out,
        block_table_npu,
        block_table_npu.stride(0),
        block_table_npu.shape[1],
        starts_npu,
        sequence_lengths_npu,
        rejected_npu,
        777,
        16,
        99,
        geometry.num_queries,
        k,
        3,
        PAD_SLOT_ID=PAD_SLOT_ID,
        HAS_NULL_BLOCK=True,
        SAMPLE_FROM_ANCHOR=anchor,
        TILE_SIZE=64,
    )
    torch.npu.synchronize()

    assert tuple(context_positions_out.cpu().tolist()) == reference.context_positions
    assert tuple(context_slots_out.cpu().tolist()) == reference.context_slot_mapping
    assert tuple(query_ids_out.cpu().tolist()) == reference.query_input_ids
    assert tuple(query_positions_out.cpu().tolist()) == reference.query_positions
    assert tuple(query_slots_out.cpu().tolist()) == reference.query_slot_mapping
    assert tuple(sample_indices_out.cpu().tolist()) == reference.sample_indices
    for guard in (
        context_positions_guard[len(positions) :],
        context_slots_guard[len(slots) :],
        query_ids_guard[query_count:],
        query_positions_guard[query_count:],
        query_slots_guard[query_count:],
        sample_indices_guard[3 * k :],
    ):
        assert torch.all(guard == -31337)
