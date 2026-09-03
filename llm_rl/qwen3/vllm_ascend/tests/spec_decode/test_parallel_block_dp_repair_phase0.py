from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.spec_decode.parallel_block_proposer as parallel_block
from vllm_ascend.attention.context_kv import PAD_SLOT_ID
from vllm_ascend.spec_decode.parallel_block_proposer import (
    DraftDPPlan,
    ParallelBlockProposer,
    validate_draft_dp_plan,
    validate_homogeneous_draft_capacities,
)


def _plan(
    *,
    method="dflash",
    batch_size=3,
    k=7,
    q=8,
    actual=24,
    execute=24,
    dp_size=1,
    dp_rank=0,
    counts=None,
    capacity=64,
):
    return validate_draft_dp_plan(
        method=method,
        batch_size=batch_size,
        num_speculative_tokens=k,
        num_queries=q,
        num_actual_tokens=actual,
        num_tokens=execute,
        dp_size=dp_size,
        dp_rank=dp_rank,
        num_tokens_across_dp=counts,
        max_query_tokens=capacity,
    )


@pytest.mark.parametrize(
    "kwargs,expected_padding",
    [
        ({}, 0),
        (
            {
                "dp_size": 2,
                "counts": torch.tensor([24, 24], dtype=torch.int32),
            },
            0,
        ),
        (
            {
                "batch_size": 1,
                "actual": 8,
                "execute": 24,
                "dp_size": 2,
                "counts": torch.tensor([24, 8], dtype=torch.int32),
            },
            16,
        ),
        (
            {
                "method": "dspark",
                "q": 7,
                "actual": 21,
                "execute": 24,
                "dp_size": 2,
                "counts": torch.tensor([24, 21], dtype=torch.int32),
            },
            3,
        ),
        (
            {
                "batch_size": 1,
                "actual": 8,
                "execute": 32,
                "dp_size": 4,
                "counts": torch.tensor([32, 8, 8, 8], dtype=torch.int32),
            },
            24,
        ),
    ],
)
def test_dp_plan_matrix(kwargs, expected_padding):
    plan = _plan(**kwargs)
    assert isinstance(plan, DraftDPPlan)
    assert plan.num_padding_tokens == expected_padding
    assert plan.num_tokens - plan.num_actual_tokens == expected_padding


@pytest.mark.parametrize(
    "method,k,q",
    [
        ("dflash", 7, 8),
        ("dspark", 7, 7),
        ("dspark", 7, 8),
    ],
)
def test_method_query_geometries_are_accepted(method, k, q):
    plan = _plan(method=method, k=k, q=q, actual=3 * q, execute=3 * q)
    assert plan.num_actual_tokens == 3 * q


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"execute": 23}, "0<=actual<=exec<=capacity"),
        ({"execute": 65}, "0<=actual<=exec<=capacity"),
        ({"actual": 23}, r"actual=B\*Q"),
        ({"dp_size": 2, "counts": None}, "counts tensor"),
        (
            {"dp_size": 2, "counts": torch.tensor([24], dtype=torch.int32)},
            r"shape \[2\]",
        ),
        (
            {
                "dp_size": 2,
                "counts": torch.tensor([23, 24], dtype=torch.int32),
            },
            "local count",
        ),
        (
            {
                "dp_size": 2,
                "counts": torch.tensor([24, 24], dtype=torch.int64),
            },
            "torch.int32",
        ),
        (
            {
                "dp_size": 2,
                "counts": torch.tensor([24, 24], dtype=torch.int32).reshape(2, 1),
            },
            r"shape \[2\]",
        ),
        (
            {
                "execute": 25,
                "counts": torch.tensor([25], dtype=torch.int32),
            },
            "counts=None",
        ),
    ],
)
def test_invalid_dp_plans_fail_fast_with_full_context(overrides, error):
    with pytest.raises((TypeError, ValueError), match=error) as exc_info:
        _plan(**overrides)
    message = str(exc_info.value)
    for field in (
        "method=dflash",
        "B=3",
        "K=7",
        "Q=8",
        "actual=",
        "exec=",
        "pad=",
        "DP size/rank=",
        "counts=",
        "capacity=64",
    ):
        assert field in message


def test_counts_must_be_cpu_without_reading_device_values():
    counts = torch.empty(2, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="remain on CPU"):
        _plan(dp_size=2, counts=counts)


def test_valid_plan_does_not_materialize_lazy_error_context(monkeypatch):
    def fail_if_called(**_):
        raise AssertionError("success path must not format counts")

    monkeypatch.setattr(parallel_block, "_format_dp_plan_context", fail_if_called)
    counts = torch.tensor([24, 24], dtype=torch.int32)
    plan = _plan(dp_size=2, counts=counts)
    assert plan.num_tokens_across_dp is counts


def test_proposer_validator_uses_engine_dp_rank_and_borrows_counts():
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.method = "dspark"
    proposer.num_speculative_tokens = 7
    proposer.num_queries = 7
    proposer.max_query_tokens = 64
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=1)
    )
    counts = torch.tensor([24, 21], dtype=torch.int32)

    plan = proposer._validate_dp_plan(
        batch_size=3,
        num_actual_tokens=21,
        num_tokens=21,
        num_tokens_across_dp=counts,
    )

    assert plan.num_padding_tokens == 0
    assert plan.num_tokens_across_dp is counts


def test_homogeneous_capacity_contract():
    validate_homogeneous_draft_capacities((64, 64), dp_size=2, local_capacity=64)
    with pytest.raises(ValueError, match="Heterogeneous"):
        validate_homogeneous_draft_capacities(
            (64, 32), dp_size=2, local_capacity=64
        )
    with pytest.raises(ValueError, match="one integer per rank"):
        validate_homogeneous_draft_capacities((64,), dp_size=2, local_capacity=64)


def _padding_proposer(capacity=40, mask_token_id=31):
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.method = "dflash"
    proposer.max_query_tokens = capacity
    proposer.model = SimpleNamespace(
        model=SimpleNamespace(mask_token_id=mask_token_id)
    )
    proposer._query_input_ids = torch.full((capacity,), 91, dtype=torch.int32)
    proposer._query_positions = torch.full((capacity,), 92, dtype=torch.int32)
    proposer._query_slots = torch.full((capacity,), 93, dtype=torch.int32)
    proposer._per_group_query_slot_mapping_buffers = {
        0: torch.full((capacity,), 94, dtype=torch.int32),
        1: torch.full((capacity,), 95, dtype=torch.int32),
    }
    return proposer


def test_padding_poison_preserves_prefix_and_sanitizes_arbitrary_tail():
    proposer = _padding_proposer()
    actual, execute = 21, 24
    snapshots = {
        name: value.clone()
        for name, value in (
            ("ids", proposer._query_input_ids),
            ("positions", proposer._query_positions),
            ("slots", proposer._query_slots),
        )
    }
    group_snapshots = {
        group: value.clone()
        for group, value in proposer._per_group_query_slot_mapping_buffers.items()
    }

    proposer._pad_draft_buffers(
        num_actual_tokens=actual,
        num_tokens=execute,
    )

    assert torch.equal(proposer._query_input_ids[:actual], snapshots["ids"][:actual])
    assert torch.equal(
        proposer._query_positions[:actual], snapshots["positions"][:actual]
    )
    assert torch.equal(proposer._query_slots[:actual], snapshots["slots"][:actual])
    assert torch.all(proposer._query_input_ids[actual:execute] == 31)
    assert torch.all(proposer._query_positions[actual:execute] == 0)
    assert torch.all(proposer._query_slots[actual:execute] == PAD_SLOT_ID)
    for group, value in proposer._per_group_query_slot_mapping_buffers.items():
        assert torch.equal(value[:actual], group_snapshots[group][:actual])
        assert torch.all(value[actual:execute] == PAD_SLOT_ID)


def test_zero_padding_returns_before_touching_buffers_or_model():
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.method = "dspark"
    proposer.max_query_tokens = 24
    proposer.model = Mock(side_effect=AssertionError("model must not be read"))
    proposer._query_input_ids = Mock(side_effect=AssertionError("must not touch ids"))
    proposer._query_positions = Mock(
        side_effect=AssertionError("must not touch positions")
    )
    proposer._query_slots = Mock(side_effect=AssertionError("must not touch slots"))

    proposer._pad_draft_buffers(num_actual_tokens=24, num_tokens=24)


def test_padding_precedes_metadata_build_and_preserves_logical_metadata_inputs():
    proposer = _padding_proposer()
    plan = DraftDPPlan(
        num_actual_tokens=8,
        num_tokens=11,
        num_padding_tokens=3,
        num_tokens_across_dp=torch.tensor([11, 8], dtype=torch.int32),
    )
    common = object()
    rejected = torch.zeros(1, dtype=torch.int32)
    observed = {}

    def build(common_arg, slot_mapping, rejected_arg):
        observed["tail"] = slot_mapping[8:11].clone()
        observed["common"] = common_arg
        observed["rejected"] = rejected_arg
        return "metadata"

    proposer.build_parallel_attention_metadata = build
    result = proposer._pad_and_build_parallel_attention_metadata(
        common, rejected, plan
    )

    assert result == "metadata"
    assert torch.all(observed["tail"] == PAD_SLOT_ID)
    assert observed["common"] is common
    assert observed["rejected"] is rejected


def test_padding_rejects_buffer_capacity_before_any_forward():
    proposer = _padding_proposer(capacity=24)
    ids_before = proposer._query_input_ids.clone()
    positions_before = proposer._query_positions.clone()
    proposer._query_slots = torch.empty(23, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="_query_slots capacity=23"):
        proposer._pad_draft_buffers(num_actual_tokens=8, num_tokens=24)
    assert torch.equal(proposer._query_input_ids, ids_before)
    assert torch.equal(proposer._query_positions, positions_before)


def test_production_dp_gate_remains_closed_in_phase0():
    source = __import__("inspect").getsource(ParallelBlockProposer.__init__)
    assert "data_parallel_size > 1" in source
    assert "raise NotImplementedError" in source
