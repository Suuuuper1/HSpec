import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.spec_decode.parallel_block_proposer as parallel_module
from vllm_ascend.spec_decode.dflash_proposer import DFlashProposer
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer


def test_aux_pack_uses_persistent_buffer_and_checkpoint_order():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.hidden_size = 4
    proposer.aux_hidden_state_layer_ids = (2, 10, 18)
    proposer._aux_hidden_buffer = torch.empty(8, 12)
    features = [
        torch.arange(32, dtype=torch.float32).view(8, 4) + offset
        for offset in (0, 100, 200)
    ]
    indices = torch.tensor([6, 1, 4], dtype=torch.int32)
    packed = proposer.pack_aux_hidden_states(features, indices, 3)
    expected = torch.cat([feature[indices] for feature in features], dim=-1)
    assert packed.data_ptr() == proposer._aux_hidden_buffer.data_ptr()
    assert torch.equal(packed, expected)


def test_aux_pack_rejects_width_and_index_count_mismatches():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.hidden_size = 4
    proposer.aux_hidden_state_layer_ids = (2,)
    proposer._aux_hidden_buffer = torch.empty(8, 4)
    with pytest.raises(ValueError, match=r"must be \[T,4\]"):
        proposer.pack_aux_hidden_states([torch.empty(8, 3)], None, 8)
    with pytest.raises(ValueError, match="contain num_tokens=3"):
        proposer.pack_aux_hidden_states(
            [torch.empty(8, 4)], torch.tensor([1, 2]), 3
        )


def test_parallel_metadata_clamps_incomplete_fixed_k_window():
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.num_queries = 4
    proposer._effective_seq_lens = torch.empty(2, dtype=torch.int32)
    proposer._can_speculate = torch.empty(2, dtype=torch.bool)
    proposer._cannot_speculate = torch.empty(2, dtype=torch.bool)
    proposer._query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    proposer._actual_query_lengths = ((4,), (4, 8))
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(max_model_len=16)
    )
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate")
    )
    common = SimpleNamespace(
        num_reqs=2,
        seq_lens=torch.tensor([12, 14], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 2, dtype=torch.int32),
    )
    metadata = proposer.build_parallel_attention_metadata(
        common,
        torch.full((8,), -1, dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
    )
    assert metadata.seq_lens.tolist() == [16, 1]
    assert proposer._can_speculate.tolist() == [True, False]


def test_parallel_metadata_propagates_uniform_draft_causality():
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.num_queries = 3
    proposer.draft_attn_causal = True
    persistent_mask = torch.ones(8, 8, dtype=torch.bool)
    proposer._draft_attn_mask = persistent_mask
    proposer._effective_seq_lens = torch.empty(1, dtype=torch.int32)
    proposer._can_speculate = torch.empty(1, dtype=torch.bool)
    proposer._cannot_speculate = torch.empty(1, dtype=torch.bool)
    proposer._query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    proposer._actual_query_lengths = ((3,),)
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(max_model_len=32)
    )
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate")
    )
    common = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 2, dtype=torch.int32),
    )
    metadata = proposer.build_parallel_attention_metadata(
        common,
        torch.full((3,), -1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    assert metadata.causal is True
    assert metadata.attn_mask is persistent_mask


def test_parallel_metadata_rejects_causal_mode_without_persistent_mask():
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.num_queries = 3
    proposer.draft_attn_causal = True
    proposer._draft_attn_mask = None
    proposer._effective_seq_lens = torch.empty(1, dtype=torch.int32)
    proposer._can_speculate = torch.empty(1, dtype=torch.bool)
    proposer._cannot_speculate = torch.empty(1, dtype=torch.bool)
    proposer._query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    proposer._actual_query_lengths = ((3,),)
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(max_model_len=32)
    )
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate")
    )
    common = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([10], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 2, dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="initialization-time NPU attention mask"):
        proposer.build_parallel_attention_metadata(
            common,
            torch.full((3,), -1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )


def test_old_and_parallel_padded_abis_remain_separate():
    old = inspect.signature(EagleProposer.prepare_inputs_padded).return_annotation
    new = inspect.signature(
        ParallelBlockProposer.prepare_parallel_inputs_padded
    ).return_annotation
    assert str(old).count("torch.Tensor") == 2
    assert str(new).count("torch.Tensor") == 3


def test_draft_attention_discovery_requires_registry_object_identity(monkeypatch):
    registry_layer = SimpleNamespace(layer_name="model.layers.36.self_attn.attn")
    detached_layer = SimpleNamespace(layer_name=registry_layer.layer_name)
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.model = SimpleNamespace(
        get_context_kv_attention_layers=lambda: (detached_layer,)
    )
    monkeypatch.setattr(
        parallel_module,
        "get_layers_from_vllm_config",
        lambda *_: {registry_layer.layer_name: registry_layer},
    )

    with pytest.raises(RuntimeError, match="registry identity mismatch"):
        proposer._discover_draft_attention_layers(set())

    proposer.model = SimpleNamespace(
        get_context_kv_attention_layers=lambda: (registry_layer,)
    )
    assert proposer._discover_draft_attention_layers(set()) == [
        registry_layer.layer_name
    ]


def test_phase1_hot_paths_have_no_host_sync_or_step_weight_concatenation():
    source = "\n".join(
        (
            inspect.getsource(DFlashProposer._propose),
            inspect.getsource(DFlashProposer._build_parallel_layout),
            inspect.getsource(ParallelBlockProposer.build_parallel_attention_metadata),
        )
    )
    for forbidden in (".item(", ".tolist(", "synchronize(", "torch.cat("):
        assert forbidden not in source
    assert source.count("_run_parallel_backbone(") == 1
    assert "validate_slot_range=False" in inspect.getsource(DFlashProposer._propose)


def test_rejected_count_launch_scales_past_one_tile():
    source = inspect.getsource(
        ParallelBlockProposer.prepare_parallel_inputs_padded
    )
    assert "triton.cdiv(num_reqs, block_size)" in source
    assert "compute_rejected_tokens_kernel[grid]" in source


def test_parallel_backbone_samples_exactly_k_rows_from_one_forward():
    class TinyDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, *, input_ids, positions):
            self.forward_calls += 1
            return torch.stack(
                (input_ids.float(), positions.float()), dim=-1
            )

        def compute_logits(self, hidden_states):
            logits = torch.full((hidden_states.shape[0], 16), -100.0)
            token_ids = hidden_states[:, 0].long() % 16
            logits.scatter_(1, token_ids[:, None], 100.0)
            return logits

    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.model = TinyDraft()
    proposer.num_speculative_tokens = 2
    input_ids = torch.tensor([9, 3, 4, 8, 6, 7], dtype=torch.int32)
    positions = torch.arange(6, dtype=torch.int32)
    # Q=K+1: skip each request's anchor row 0/3 and sample rows 1,2/4,5.
    sample_indices = torch.tensor([1, 2, 4, 5], dtype=torch.int32)
    tokens = proposer._run_parallel_backbone(
        input_ids, positions, sample_indices
    )
    assert proposer.model.forward_calls == 1
    assert tokens.dtype == torch.int64
    assert tokens.tolist() == [[3, 4], [6, 7]]


def test_parallel_dummy_run_profiles_aux_context_and_exact_greedy_head(monkeypatch):
    class ProfileDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.combine_rows = 0
            self.context_rows = 0
            self.forward_rows = 0
            self.logit_rows = 0

        def combine_hidden_states(self, hidden_states):
            self.combine_rows = hidden_states.shape[0]
            return hidden_states[:, :2]

        def compute_context_kv(self, hidden_states, positions):
            self.context_rows = hidden_states.shape[0]
            assert positions.shape[0] == hidden_states.shape[0]

        def forward(self, *, input_ids, positions):
            self.forward_rows = input_ids.shape[0]
            return torch.stack((input_ids.float(), positions.float()), dim=-1)

        def compute_logits(self, hidden_states):
            self.logit_rows = hidden_states.shape[0]
            return torch.zeros(hidden_states.shape[0], 8)

    import vllm_ascend.ascend_forward_context as forward_context

    monkeypatch.setattr(
        forward_context,
        "set_ascend_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.model = ProfileDraft()
    proposer.model.model = SimpleNamespace(mask_token_id=0)
    proposer.method = "dflash"
    proposer.max_num_tokens = 16
    proposer.max_batch_size = 3
    proposer.num_queries = 3
    proposer.num_speculative_tokens = 2
    proposer.max_query_tokens = 9
    proposer._aux_hidden_buffer = torch.zeros(16, 4)
    proposer._query_input_ids = torch.zeros(9, dtype=torch.int32)
    proposer._query_positions = torch.zeros(9, dtype=torch.int32)
    proposer._query_slots = torch.zeros(9, dtype=torch.int32)
    proposer._per_group_query_slot_mapping_buffers = {}
    proposer._parallel_dummy_positions = torch.zeros(9, dtype=torch.int32)
    proposer._parallel_dummy_sample_indices = torch.tensor(
        [1, 2, 4, 5, 7, 8], dtype=torch.int32
    )
    proposer._get_positions = lambda count: torch.arange(count, dtype=torch.int32)
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
    )
    proposer.runner = SimpleNamespace(
        _sync_metadata_across_dp=lambda tokens, **_: (tokens, None, False)
    )
    proposer._last_draft_dp_sequence = None

    proposer.dummy_run(num_tokens=7, num_reqs=2, is_profile=True)

    assert proposer.model.combine_rows == 7
    assert proposer.model.context_rows == 7
    assert proposer.model.forward_rows == 6
    assert proposer.model.logit_rows == 4
