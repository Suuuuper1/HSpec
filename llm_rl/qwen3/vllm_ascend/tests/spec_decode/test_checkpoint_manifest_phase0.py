import json

import pytest

from vllm.v1.spec_decode.checkpoint_manifest import (
    build_checkpoint_manifest,
    build_migration_manifest,
    compare_checkpoint_pair,
)


def _write_checkpoint(path, architecture, *, method=None, hidden_size=32):
    path.mkdir()
    config = {
        "architectures": [architecture],
        "model_type": "qwen3",
        "vocab_size": 128,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "dtype": "bfloat16",
        "num_hidden_layers": 4 if method is None else 2,
        "num_target_layers": 4 if method else None,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rope_theta": 10000,
        "rope_scaling": None,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "sliding_window": None,
    }
    if method == "dflash":
        config["layer_types"] = ["full_attention", "full_attention"]
        config["dflash_config"] = {
            "target_layer_ids": [0, 3],
            "mask_token_id": 127,
        }
    elif method == "dspark":
        config["layer_types"] = ["full_attention", "full_attention"]
        config["target_layer_ids"] = [0, 3]
        config["sample_from_anchor"] = True
        config["markov_rank"] = 8
    (path / "config.json").write_text(json.dumps(config))
    if method:
        safetensors = pytest.importorskip("safetensors.torch")
        import torch

        weights = {
            "fc.weight": torch.zeros((hidden_size, hidden_size * 2)),
            "hidden_norm.weight": torch.zeros(hidden_size),
            "norm.weight": torch.zeros(hidden_size),
        }
        layer_shapes = {
            "input_layernorm.weight": (hidden_size,),
            "post_attention_layernorm.weight": (hidden_size,),
            "self_attn.q_proj.weight": (32, hidden_size),
            "self_attn.k_proj.weight": (16, hidden_size),
            "self_attn.v_proj.weight": (16, hidden_size),
            "self_attn.o_proj.weight": (hidden_size, 32),
            "self_attn.q_norm.weight": (8,),
            "self_attn.k_norm.weight": (8,),
            "mlp.gate_proj.weight": (hidden_size * 2, hidden_size),
            "mlp.up_proj.weight": (hidden_size * 2, hidden_size),
            "mlp.down_proj.weight": (hidden_size, hidden_size * 2),
        }
        for layer in range(2):
            for suffix, shape in layer_shapes.items():
                weights[f"layers.{layer}.{suffix}"] = torch.zeros(shape)
        if method == "dspark":
            weights.update(
                {
                    "embed_tokens.weight": torch.zeros((128, hidden_size)),
                    "lm_head.weight": torch.zeros((128, hidden_size)),
                    "markov_head.markov_w1.weight": torch.zeros((128, 8)),
                    "markov_head.markov_w2.weight": torch.zeros((128, 8)),
                }
            )
        safetensors.save_file(weights, path / "model.safetensors")


def _write_tokenizer(path):
    (path / "tokenizer.json").write_text('{"model":{"vocab":{"a":0}}}')
    (path / "tokenizer_config.json").write_text("{}")


def test_manifest_pass_fail_and_missing_method(tmp_path):
    target_path = tmp_path / "target"
    dflash_path = tmp_path / "dflash"
    dspark_path = tmp_path / "dspark"
    _write_checkpoint(target_path, "Qwen3ForCausalLM")
    _write_checkpoint(dflash_path, "DFlashDraftModel", method="dflash")
    _write_checkpoint(dspark_path, "Qwen3DSparkModel", method="dspark")
    _write_tokenizer(target_path)

    manifest = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dflash": dflash_path, "dspark": dspark_path},
        pair_ids={"dflash": "target-r1", "dspark": "target-r1"},
        num_speculative_tokens=7,
        target_tp=4,
        draft_tp=1,
    )
    assert manifest["overall_status"] == "PASS"
    assert manifest["compatibility"]["dflash"]["status"] == "PASS"

    missing = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dflash": dflash_path},
        pair_ids={"dflash": "target-r1"},
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
    )
    assert missing["overall_status"] == "BLOCKED"
    assert missing["missing_methods"] == ["dspark"]

    target = build_checkpoint_manifest(
        target_path, role="target", tokenizer=target_path
    )
    bad_path = tmp_path / "bad"
    _write_checkpoint(
        bad_path, "DFlashDraftModel", method="dflash", hidden_size=64
    )
    bad = build_checkpoint_manifest(
        bad_path, role="draft", tokenizer=target_path, method="dflash"
    )
    result = compare_checkpoint_pair(
        target,
        bad,
        method="dflash",
        pair_id="target-r1",
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
    )
    assert result["status"] == "FAIL"
    assert any("hidden_size mismatch" in error for error in result["errors"])

    alternate_tokenizer = tmp_path / "alternate-tokenizer"
    alternate_tokenizer.mkdir()
    _write_tokenizer(alternate_tokenizer)
    (alternate_tokenizer / "tokenizer.json").write_text(
        '{"model":{"vocab":{"different":0}}}'
    )
    tokenizer_mismatch = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dflash": dflash_path, "dspark": dspark_path},
        draft_tokenizer_paths={"dflash": alternate_tokenizer},
        pair_ids={"dflash": "target-r1", "dspark": "target-r1"},
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
    )
    assert tokenizer_mismatch["compatibility"]["dflash"]["status"] == "FAIL"
    assert any(
        "tokenizer hash mismatch" in error
        for error in tokenizer_mismatch["compatibility"]["dflash"]["errors"]
    )


def test_manifest_normalizes_transformers_5_rope_parameters(tmp_path):
    target_path = tmp_path / "target"
    dspark_path = tmp_path / "dspark"
    _write_checkpoint(target_path, "Qwen3ForCausalLM")
    _write_checkpoint(dspark_path, "Qwen3DSparkModel", method="dspark")
    _write_tokenizer(target_path)

    config_path = dspark_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("rope_theta")
    config["rope_parameters"] = {
        "rope_theta": 10000,
        "rope_type": "default",
    }
    config_path.write_text(json.dumps(config))

    manifest = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dspark": dspark_path},
        pair_ids={"dspark": "published-qwen3-pair"},
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
    )
    assert manifest["compatibility"]["dspark"]["status"] == "PASS"
    assert manifest["drafts"]["dspark"]["rope_theta"] == 10000


def test_manifest_rejects_k_larger_than_checkpoint_block(tmp_path):
    target_path = tmp_path / "target"
    dflash_path = tmp_path / "dflash"
    _write_checkpoint(target_path, "Qwen3ForCausalLM")
    _write_checkpoint(dflash_path, "DFlashDraftModel", method="dflash")
    _write_tokenizer(target_path)

    config_path = dflash_path / "config.json"
    config = json.loads(config_path.read_text())
    config["block_size"] = 4
    config_path.write_text(json.dumps(config))

    manifest = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dflash": dflash_path},
        pair_ids={"dflash": "published-qwen3-pair"},
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
    )
    result = manifest["compatibility"]["dflash"]
    assert result["status"] == "FAIL"
    assert any("block_size" in error for error in result["errors"])


def test_manifest_understands_speculators_dflash_without_hiding_capabilities(
    tmp_path,
):
    target_path = tmp_path / "target"
    draft_path = tmp_path / "dflash-speculators"
    _write_checkpoint(target_path, "Qwen3MoeForCausalLM")
    _write_tokenizer(target_path)
    draft_path.mkdir()
    config = {
        "architectures": ["DFlashDraftModel"],
        "speculators_model_type": "dflash",
        "speculators_version": "test",
        "speculators_config": {
            "algorithm": "dflash",
            "proposal_methods": [{"speculative_tokens": 7}],
            "verifier": {"name_or_path": "Qwen/Qwen3-test"},
        },
        "aux_hidden_state_layer_ids": [1, 4],
        "block_size": 8,
        "draft_vocab_size": 16,
        "dtype": "bfloat16",
        "mask_token_id": 127,
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "model_type": "llama",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {"rope_theta": 10000},
            "rope_scaling": None,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "sliding_window": 16,
        },
    }
    (draft_path / "config.json").write_text(json.dumps(config))
    safetensors = pytest.importorskip("safetensors.torch")
    import torch

    safetensors.save_file(
        {
            "embed_tokens.weight": torch.zeros((128, 32)),
            "lm_head.weight": torch.zeros((16, 32)),
            "fc.weight": torch.zeros((32, 64)),
            "d2t": torch.zeros(16, dtype=torch.int64),
            "t2d": torch.tensor([True] * 16 + [False] * 112),
            "hidden_norm.weight": torch.zeros(32),
            "norm.weight": torch.zeros(32),
            **{
                f"layers.{layer}.{suffix}": torch.zeros(shape)
                for layer in range(2)
                for suffix, shape in {
                    "input_layernorm.weight": (32,),
                    "post_attention_layernorm.weight": (32,),
                    "self_attn.q_proj.weight": (32, 32),
                    "self_attn.k_proj.weight": (16, 32),
                    "self_attn.v_proj.weight": (16, 32),
                    "self_attn.o_proj.weight": (32, 32),
                    "self_attn.q_norm.weight": (8,),
                    "self_attn.k_norm.weight": (8,),
                    "mlp.gate_proj.weight": (64, 32),
                    "mlp.up_proj.weight": (64, 32),
                    "mlp.down_proj.weight": (32, 64),
                }.items()
            },
        },
        draft_path / "model.safetensors",
    )

    manifest = build_migration_manifest(
        target_path=target_path,
        tokenizer_path=target_path,
        draft_paths={"dflash": draft_path},
        pair_ids={"dflash": "published-qwen3-pair"},
        num_speculative_tokens=7,
        target_tp=1,
        draft_tp=1,
        required_methods=("dflash",),
    )
    draft = manifest["drafts"]["dflash"]
    result = manifest["compatibility"]["dflash"]
    assert manifest["missing_methods"] == []
    assert manifest["overall_status"] == "PASS"
    assert draft["schema"]["format"] == "speculators"
    assert draft["schema"]["source_transformer_model_type"] == "llama"
    assert draft["model_type"] == "qwen3"
    assert draft["hidden_size"] == 32
    assert draft["aux_layer_ids"] == [0, 3]
    assert draft["aux_concat_width"] == draft["draft_fc_input_width"] == 64
    assert draft["vocab_size"] == 128
    assert draft["draft_vocab_size"] == 16
    assert draft["layer_causality"] == [True, True]
    assert result["status"] == "PASS"
    joined = "\n".join(result["errors"])
    for false_diagnostic in (
        "hidden_size mismatch",
        "head_dim mismatch",
        "does not declare target aux layer ids",
        "FC input width",
    ):
        assert false_diagnostic not in joined
