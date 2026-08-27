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
    (path / "config.json").write_text(json.dumps(config))
    if method:
        safetensors = pytest.importorskip("safetensors.torch")
        import torch

        safetensors.save_file(
            {"fc.weight": torch.zeros((hidden_size, hidden_size * 2))},
            path / "model.safetensors",
        )


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
