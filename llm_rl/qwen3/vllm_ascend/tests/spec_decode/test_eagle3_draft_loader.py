# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.config import ModelConfig, ParallelConfig
from vllm.config.load import LoadConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.llama_eagle3 import \
    validate_eagle3_vocab_mapping
from vllm_ascend.spec_decode import eagle_proposer as eagle_module
from vllm_ascend.spec_decode.eagle_proposer import \
    EagleProposer, _with_eagle_draft_load_config


@dataclass(frozen=True)
class _VllmConfig:
    model_config: object
    load_config: object
    parallel_config: object
    speculative_config: object


def test_independent_draft_loader_preserves_target_identity():
    target_model = SimpleNamespace(num_hidden_layers=48)
    target_parallel = SimpleNamespace(tensor_parallel_size=4)
    target_loader = SimpleNamespace(load_format="dummy")
    draft_loader = SimpleNamespace(load_format="auto")
    draft_model = SimpleNamespace(num_hidden_layers=1)
    speculative = SimpleNamespace(
        draft_load_config=draft_loader,
        draft_model_config=draft_model,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    config = _VllmConfig(
        model_config=target_model,
        load_config=target_loader,
        parallel_config=target_parallel,
        speculative_config=speculative,
    )

    resolved = _with_eagle_draft_load_config(config)

    assert resolved is not config
    assert resolved.load_config is draft_loader
    assert resolved.model_config is target_model
    assert resolved.parallel_config is target_parallel
    assert config.load_config is target_loader


def test_default_eagle_loader_returns_original_config():
    speculative = SimpleNamespace(draft_load_config=None)
    config = _VllmConfig(object(), object(), object(), speculative)

    assert _with_eagle_draft_load_config(config) is config


def _write_config(path, architecture, model_type, num_hidden_layers):
    path.mkdir()
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 128,
    }
    if architecture == "LlamaForCausalLMEagle3":
        config["draft_vocab_size"] = 64
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_eagle3_allows_only_explicit_standard_rejection(tmp_path):
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    _write_config(target_path, "Qwen3ForCausalLM", "qwen3", 4)
    _write_config(draft_path, "LlamaForCausalLMEagle3", "llama", 1)
    target = ModelConfig(
        model=str(target_path),
        tokenizer=str(target_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=128,
    )
    common = {
        "method": "eagle3",
        "model": str(draft_path),
        "num_speculative_tokens": 3,
        "target_model_config": target,
        "target_parallel_config": ParallelConfig(),
        "draft_tensor_parallel_size": 1,
        "draft_load_config": LoadConfig(load_format="auto"),
    }

    spec = SpeculativeConfig(
        **common,
        rejection_sample_method="standard",
    )
    assert spec.rejection_sample_method == "standard"

    with pytest.raises(ValueError, match="must be 'standard'"):
        SpeculativeConfig(**common, rejection_sample_method="legacy")


def test_eagle3_reduced_vocab_mapping_validation():
    d2t = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    t2d = torch.tensor(
        [True, True, False, True, False, True], dtype=torch.bool
    )
    validate_eagle3_vocab_mapping(d2t, t2d, 4, 6)

    with pytest.raises(ValueError, match="uniquely"):
        validate_eagle3_vocab_mapping(
            torch.tensor([0, -1, 0, 0]), t2d, 4, 6
        )
    with pytest.raises(ValueError, match="membership"):
        validate_eagle3_vocab_mapping(
            d2t, torch.ones(6, dtype=torch.bool), 4, 6
        )
    with pytest.raises(ValueError, match="missing d2t"):
        validate_eagle3_vocab_mapping(None, None, 4, 6)


def test_padded_preparation_returns_rejected_counts_without_aliasing(monkeypatch):
    monkeypatch.setattr(eagle_module, "HAS_TRITON", False)
    proposer = EagleProposer.__new__(EagleProposer)
    proposer.arange = torch.arange(16, dtype=torch.int32)
    proposer.pcp_size = 1
    proposer.runner = SimpleNamespace(
        actual_seq_lengths_q=[],
        attn_state=None,
        decode_token_per_req=4,
    )
    common = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 7], dtype=torch.int32),
        seq_lens=torch.tensor([13, 27], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([13, 27], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([10, 23], dtype=torch.int32),
        num_actual_tokens=7,
        num_input_tokens=7,
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(7, dtype=torch.int32),
        positions=torch.arange(7, dtype=torch.int64),
    )
    spec_metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([2, 5], dtype=torch.int32)
    )
    rejected_out = torch.empty(2, dtype=torch.int32)

    metadata, token_indices, sample_indices, rejected = \
        proposer.prepare_inputs_padded(
            common,
            spec_metadata,
            torch.tensor([2, 1], dtype=torch.int32),
            num_rejected_tokens_out=rejected_out,
        )

    assert rejected.data_ptr() == rejected_out.data_ptr()
    assert rejected.tolist() == [1, 3]
    assert sample_indices.tolist() == [1, 3]
    assert token_indices.tolist() == list(range(7))
    assert metadata.seq_lens.data_ptr() != common.seq_lens.data_ptr()
    assert metadata.seq_lens_cpu.data_ptr() != common.seq_lens_cpu.data_ptr()
