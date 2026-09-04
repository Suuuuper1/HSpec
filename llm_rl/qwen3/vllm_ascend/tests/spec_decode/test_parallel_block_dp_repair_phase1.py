from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ascend_forward_context as ascend_context
import vllm_ascend.spec_decode.dflash_proposer as dflash_module
from vllm_ascend.attention.context_kv import PAD_SLOT_ID
from vllm_ascend.spec_decode.dflash_proposer import DFlashProposer
from vllm_ascend.spec_decode.dspark_proposer import DSparkProposer
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer
from vllm_ascend.spec_decode.parallel_draft_metrics import DraftDPObserver


def _parallel_config(dp_size=2, dp_rank=0):
    return SimpleNamespace(data_parallel_size=dp_size, data_parallel_rank=dp_rank)


def _coordination_proposer(*, q=8, k=7, execute=24, counts=None, trace=False):
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.method = "dflash"
    proposer.num_queries = q
    proposer.num_speculative_tokens = k
    proposer.max_query_tokens = 64
    proposer.vllm_config = SimpleNamespace(parallel_config=_parallel_config())
    proposer.runner = Mock()
    proposer.runner._sync_metadata_across_dp.return_value = (
        execute,
        counts
        if counts is not None
        else torch.tensor([execute, execute], dtype=torch.int32),
        False,
    )
    proposer._draft_dp_observer = DraftDPObserver(
        method="dflash",
        dp_size=2,
        dp_rank=0,
        draft_model_kind="dense",
        sample_every=1,
        trace_enabled=trace,
        trace_limit=8,
    )
    proposer._last_draft_dp_sequence = None
    return proposer


def test_coordinate_calls_only_old_runner_draft_sync_and_accounts_tokens():
    proposer = _coordination_proposer(execute=24)
    plan = proposer._coordinate_draft_dp(
        batch_size=1, num_actual_tokens=8, kind="real"
    )

    proposer.runner._sync_metadata_across_dp.assert_called_once_with(
        8, is_draft_model=True
    )
    assert (plan.num_actual_tokens, plan.num_tokens, plan.num_padding_tokens) == (
        8,
        24,
        16,
    )
    snapshot = proposer.draft_dp_observer.snapshot()
    assert snapshot["counters"] == {
        "draft_dp_real_calls": 1,
        "draft_dp_dummy_calls": 0,
        "draft_dp_profile_calls": 0,
        "draft_dp_sync_calls": 1,
        "draft_dp_sync_skipped_dense": 1,
        "draft_dp_actual_query_tokens": 8,
        "draft_dp_execution_query_tokens": 24,
        "draft_dp_padding_tokens": 16,
        "draft_dp_padding_steps": 1,
        "draft_dp_plan_failures": 0,
    }
    assert all(snapshot["invariants"].values())
    assert snapshot["host_sync_timer"]["sample_count"] == 1
    assert snapshot["device_synchronize_calls"] == 0


def test_coordinate_rejects_bad_runner_result_before_forward():
    proposer = _coordination_proposer(
        execute=7, counts=torch.tensor([7, 8], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="0<=actual<=exec<=capacity"):
        proposer._coordinate_draft_dp(
            batch_size=1, num_actual_tokens=8, kind="real"
        )
    snapshot = proposer.draft_dp_observer.snapshot()
    assert snapshot["counters"]["draft_dp_sync_calls"] == 1
    assert snapshot["counters"]["draft_dp_plan_failures"] == 1
    assert snapshot["counters"]["draft_dp_execution_query_tokens"] == 0


class _RealModel:
    def __init__(self, mask_token_id=127):
        self.model = SimpleNamespace(mask_token_id=mask_token_id)

    def combine_hidden_states(self, hidden):
        return hidden

    def compute_context_kv(self, hidden, positions):
        return hidden, hidden

    def get_context_kv_attention_layers(self):
        return []


@pytest.mark.parametrize(
    "method,k,q",
    [("dflash", 2, 3), ("dspark", 2, 2), ("dspark", 2, 3)],
)
def test_real_path_separates_execution_rows_from_logical_sampling(
    monkeypatch, method, k, q
):
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.method = method
    proposer.num_speculative_tokens = k
    proposer.num_queries = q
    proposer.max_batch_size = 4
    proposer.max_query_tokens = 12
    proposer._probabilistic = method == "dspark"
    proposer._last_draft_probs = None
    proposer._last_draft_dp_sequence = None
    proposer.vllm_config = SimpleNamespace(parallel_config=_parallel_config())
    actual = 2 * q
    execute = actual + 1
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=Mock(
            return_value=(
                execute,
                torch.tensor([execute, execute], dtype=torch.int32),
                False,
            )
        )
    )
    proposer.model = _RealModel()
    proposer._query_input_ids = torch.full((12,), 91, dtype=torch.int32)
    proposer._query_positions = torch.full((12,), 92, dtype=torch.int32)
    proposer._query_slots = torch.full((12,), 93, dtype=torch.int32)
    proposer._sample_indices = torch.arange(2 * k, dtype=torch.int32)
    proposer._context_positions = torch.arange(4, dtype=torch.int32)
    proposer._context_slots = torch.arange(4, dtype=torch.int32)
    proposer._num_rejected_tokens = torch.zeros(4, dtype=torch.int32)
    proposer._cannot_speculate = torch.zeros(4, dtype=torch.bool)
    proposer.attn_layer_names = ["draft.layer"]
    proposer._draft_dp_observer = DraftDPObserver(
        method=method,
        dp_size=2,
        dp_rank=0,
        draft_model_kind="dense",
        trace_enabled=True,
        trace_limit=4,
    )

    common = SimpleNamespace(num_reqs=2, sentinel="unchanged")
    sampling = SimpleNamespace(all_greedy=True)
    observed = {}

    def layout(*_args):
        proposer._query_input_ids[:actual].copy_(
            torch.arange(actual, dtype=torch.int32)
        )
        proposer._query_positions[:actual].copy_(
            torch.arange(actual, dtype=torch.int32) + 10
        )
        proposer._query_slots[:actual].copy_(
            torch.arange(actual, dtype=torch.int32) + 20
        )
        sample_indices = torch.arange(2 * k, dtype=torch.int32)
        return (
            proposer._context_positions[:4],
            proposer._context_slots[:4],
            proposer._query_input_ids[:actual],
            proposer._query_positions[:actual],
            proposer._query_slots[:actual],
            sample_indices,
        )

    def metadata_builder(common_arg, slot_mapping, rejected):
        observed["metadata_common"] = common_arg
        observed["metadata_slots"] = slot_mapping.clone()
        observed["rejected"] = rejected
        return "draft-metadata"

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        observed["context"] = (metadata, config, kwargs)
        yield

    def backbone(input_ids, positions, sample_indices, anchor, metadata):
        observed["backbone"] = {
            "ids": input_ids.clone(),
            "positions": positions.clone(),
            "sample_indices": sample_indices.clone(),
            "anchor": anchor.clone(),
            "sampling": metadata,
        }
        return torch.ones((2, k), dtype=torch.int64)

    proposer._build_parallel_layout = layout
    proposer.build_parallel_attention_metadata = metadata_builder
    proposer._run_parallel_backbone = backbone
    monkeypatch.setattr(dflash_module, "store_all_context_kv", lambda *_a, **_k: None)
    monkeypatch.setattr(dflash_module, "set_ascend_forward_context", forward_context)

    output = DFlashProposer._propose(
        proposer,
        target_token_ids=torch.zeros(4, dtype=torch.int32),
        target_positions=torch.arange(4, dtype=torch.int32),
        target_hidden_states=torch.zeros((4, 2)),
        next_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        last_token_indices=None,
        common_attn_metadata=common,
        sampling_metadata=sampling,
    )

    proposer.runner._sync_metadata_across_dp.assert_called_once_with(
        actual, is_draft_model=True
    )
    metadata, config, kwargs = observed["context"]
    assert metadata == {"draft.layer": "draft-metadata"}
    assert config is proposer.vllm_config
    assert kwargs["num_tokens"] == execute
    assert kwargs["num_actual_tokens"] == actual
    assert torch.equal(
        kwargs["num_tokens_across_dp"],
        torch.tensor([execute, execute], dtype=torch.int32),
    )
    assert observed["backbone"]["ids"].numel() == execute
    assert observed["backbone"]["positions"].numel() == execute
    assert observed["backbone"]["sample_indices"].numel() == 2 * k
    assert int(observed["backbone"]["sample_indices"].max()) < actual
    assert observed["backbone"]["sampling"] is sampling
    assert observed["backbone"]["ids"][-1] == 127
    assert observed["backbone"]["positions"][-1] == 0
    assert observed["metadata_slots"][-1] == PAD_SLOT_ID
    assert common.sentinel == "unchanged"
    assert tuple(output.shape) == (2, k)
    trace = proposer.draft_dp_observer.snapshot()["trace"]["rows"]
    assert trace[0]["context_enter"] is True
    assert trace[0]["context_exit"] is True and trace[0]["status"] == "PASS"


class _DummyModel:
    def __init__(self, mask_token_id=31):
        self.model = SimpleNamespace(mask_token_id=mask_token_id)
        self.combined_was_finite = False
        self.forward_inputs = []

    def combine_hidden_states(self, hidden):
        self.combined_was_finite = bool(torch.isfinite(hidden).all())
        return hidden

    def compute_context_kv(self, hidden, positions):
        assert hidden.shape[0] == positions.shape[0]
        return hidden, hidden

    def __call__(self, *, input_ids, positions):
        self.forward_inputs.append((input_ids.clone(), positions.clone()))
        return torch.zeros((input_ids.numel(), 4))


@pytest.mark.parametrize("is_profile,expected_context_actual", [(False, 0), (True, 8)])
def test_dummy_owns_counts_buffers_and_profile_mode(
    monkeypatch, is_profile, expected_context_actual
):
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.method = "dflash"
    proposer.num_speculative_tokens = 3
    proposer.num_queries = 4
    proposer.max_batch_size = 2
    proposer.max_query_tokens = 10
    proposer.max_num_tokens = 16
    proposer.vllm_config = SimpleNamespace(parallel_config=_parallel_config())
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=Mock(
            return_value=(
                10,
                torch.tensor([10, 10], dtype=torch.int32),
                False,
            )
        )
    )
    proposer.model = _DummyModel()
    proposer._query_input_ids = torch.full((10,), 99, dtype=torch.int32)
    proposer._query_positions = torch.full((10,), 98, dtype=torch.int32)
    proposer._query_slots = torch.full((10,), 97, dtype=torch.int32)
    proposer._per_group_query_slot_mapping_buffers = {
        0: torch.full((10,), 96, dtype=torch.int32)
    }
    proposer._parallel_dummy_sample_indices = torch.tensor(
        [1, 2, 3, 5, 6, 7], dtype=torch.int32
    )
    proposer._aux_hidden_buffer = torch.zeros((16, 4))
    proposer._get_positions = lambda count: torch.arange(count, dtype=torch.int32)
    proposer._draft_dp_observer = DraftDPObserver(
        method="dflash",
        dp_size=2,
        dp_rank=0,
        draft_model_kind="dense",
        trace_enabled=True,
        trace_limit=4,
    )
    proposer._last_draft_dp_sequence = None
    proposer._select_draft_tokens = Mock(return_value=torch.zeros((2, 3)))
    observed = {}

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        observed.update(kwargs)
        yield

    monkeypatch.setattr(ascend_context, "set_ascend_forward_context", forward_context)
    poison_target_counts = torch.tensor([-99], dtype=torch.int64)
    proposer.dummy_run(
        num_tokens=6,
        num_reqs=2,
        num_tokens_across_dp=poison_target_counts,
        is_profile=is_profile,
    )

    proposer.runner._sync_metadata_across_dp.assert_called_once_with(
        8, is_draft_model=True
    )
    assert observed["num_tokens"] == 10
    assert observed["num_actual_tokens"] == expected_context_actual
    assert observed["is_draft_model"] is True
    assert observed["in_profile_run"] is is_profile
    ids, positions = proposer.model.forward_inputs[0]
    assert torch.all(ids == 31)
    assert torch.all(positions == 0)
    assert torch.all(proposer._query_slots == PAD_SLOT_ID)
    assert torch.all(
        proposer._per_group_query_slot_mapping_buffers[0] == PAD_SLOT_ID
    )
    assert proposer.model.combined_was_finite
    proposer._select_draft_tokens.assert_called_once()
    assert proposer._select_draft_tokens.call_args.args[2] is None
    trace = proposer.draft_dp_observer.snapshot()["trace"]["rows"]
    assert trace[0]["kind"] == ("profile" if is_profile else "dummy")
    assert trace[0]["context_enter"] is True
    assert trace[0]["context_exit"] is True


class _SamplingModel:
    def __call__(self, *, input_ids, positions):
        rows = torch.arange(input_ids.numel() * 3, dtype=torch.float32)
        return rows.view(input_ids.numel(), 3)

    def compute_logits(self, hidden):
        return torch.stack(
            (hidden[:, 0], hidden[:, 1], hidden[:, 2], -hidden[:, 0]), dim=-1
        )

    compute_draft_logits = compute_logits

    def markov_embed_into(self, tokens, output):
        output[:, 0].copy_(tokens.remainder(2))
        output[:, 1].copy_(tokens.remainder(3))

    def markov_bias_into(self, embedding, output):
        output.zero_()
        output[:, :2].copy_(embedding)

    @staticmethod
    def map_draft_to_target(tokens):
        return tokens


@pytest.mark.parametrize(
    "proposer_type,q,sample_indices",
    [
        (DFlashProposer, 3, [0, 1, 3, 4]),
        (DSparkProposer, 2, [0, 1, 2, 3]),
        (DSparkProposer, 3, [1, 2, 4, 5]),
    ],
)
def test_probabilistic_tokens_q_and_rng_are_padding_invariant(
    proposer_type, q, sample_indices
):
    def run(execution_tokens):
        proposer = proposer_type.__new__(proposer_type)
        proposer.method = "dflash" if proposer_type is DFlashProposer else "dspark"
        proposer.num_queries = q
        proposer.num_speculative_tokens = 2
        proposer._probabilistic = True
        proposer._last_draft_probs = None
        proposer.model = _SamplingModel()
        if proposer_type is DSparkProposer:
            proposer._dspark_token_buffer = torch.empty((2, 3), dtype=torch.int64)
            proposer._dspark_embedding_buffer = torch.empty((2, 2))
            proposer._dspark_corrected_logits_buffer = torch.empty((2, 4))
        generators = {
            0: torch.Generator().manual_seed(1701),
            1: torch.Generator().manual_seed(2901),
        }
        sampling = SimpleNamespace(
            temperature=torch.tensor([0.7, 1.2]),
            all_greedy=False,
            all_random=True,
            generators=generators,
        )
        global_before = torch.random.get_rng_state().clone()
        tokens = proposer._run_parallel_backbone(
            torch.arange(execution_tokens, dtype=torch.int32),
            torch.arange(execution_tokens, dtype=torch.int32),
            torch.tensor(sample_indices, dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
            sampling,
        )
        global_after = torch.random.get_rng_state().clone()
        private_states = tuple(generator.get_state() for generator in generators.values())
        return (
            tokens.clone(),
            proposer.take_last_draft_probs().clone(),
            global_before,
            global_after,
            private_states,
        )

    actual_tokens = 2 * q
    no_padding = run(actual_tokens)
    uneven_padding = run(actual_tokens + 3)
    assert torch.equal(no_padding[0], uneven_padding[0])
    assert torch.equal(no_padding[1], uneven_padding[1])
    assert tuple(no_padding[1].shape) == (2, 2, 4)
    assert torch.equal(no_padding[2], no_padding[3])
    assert torch.equal(uneven_padding[2], uneven_padding[3])
    assert all(
        torch.equal(first, second)
        for first, second in zip(no_padding[4], uneven_padding[4])
    )


def test_dp_observer_trace_is_bounded_and_has_no_device_timing(monkeypatch):
    observer = DraftDPObserver(
        method="dspark",
        dp_size=2,
        dp_rank=1,
        draft_model_kind="dense",
        sample_every=2,
        trace_enabled=True,
        trace_limit=2,
    )
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.parallel_draft_metrics.time.perf_counter_ns",
        Mock(side_effect=range(0, 100_000, 1_000)),
    )
    for kind in ("real", "dummy", "profile"):
        sequence, start = observer.begin_sync(kind)
        observer.finish_sync(
            sequence=sequence,
            kind=kind,
            start_ns=start,
            batch_size=1,
            num_queries=7,
            num_actual_tokens=7,
            num_tokens=8,
            num_padding_tokens=1,
            success=True,
        )
        observer.finish_context(sequence, success=True)
    snapshot = observer.snapshot()
    assert snapshot["host_sync_timer"]["sample_count"] == 2
    assert len(snapshot["trace"]["rows"]) == 2
    assert snapshot["trace"]["dropped"] == 1
    assert snapshot["device_synchronize_calls"] == 0
    assert snapshot["new_npu_collectives"] == 0
    assert all(snapshot["invariants"].values())


def test_phase1_keeps_production_dp_gate_closed():
    source = __import__("inspect").getsource(ParallelBlockProposer.__init__)
    assert "data_parallel_size > 1" in source
    assert "raise NotImplementedError" in source
