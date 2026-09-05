from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _expand_draft_logits_to_target,
    _missing_packed_shards,
    _validate_dflash_attention_config,
    dflash_aux_capture_layer_ids,
)
from vllm_ascend.ops.triton.batch_invariant.rmsnorm import rms_norm_into
from vllm_ascend.ops.rotary_embedding import rotary_embedding_into


def _require_npu() -> None:
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            pytest.skip("torch-npu reports no available NPU")
        torch.empty(1, device="npu")
    except Exception as error:
        pytest.skip(f"NPU runtime unavailable: {type(error).__name__}: {error}")


def _draft_config(**dflash_overrides):
    dflash = {
        "mask_token_id": 127,
        "target_layer_ids": [1, 3],
        **dflash_overrides,
    }
    return SimpleNamespace(
        dflash_config=dflash,
        layer_types=["full_attention", "full_attention"],
        num_hidden_layers=2,
        sliding_window=None,
        vocab_size=128,
    )


def test_aux_capture_translates_hf_outputs_to_old_qwen_capture_points():
    assert dflash_aux_capture_layer_ids(_draft_config()) == (2, 4)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"future_attention_mode": "silent"}, "Unsupported algorithm-affecting"),
        ({"mask_token_id": 128}, "mask_token_id"),
        ({"target_layer_ids": [3, 1]}, "ordered set"),
        ({"use_aux_hidden_state": False}, "requires auxiliary"),
        ({"causal": True}, "non-causal"),
    ],
)
def test_algorithm_affecting_checkpoint_fields_fail_closed(override, error):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        _validate_dflash_attention_config(_draft_config(**override))


def test_uniform_causal_sliding_attention_is_the_only_second_surface():
    config = _draft_config(causal=True)
    config.layer_types = ["sliding_attention", "sliding_attention"]
    config.sliding_window = 32
    config.draft_vocab_size = 64
    _validate_dflash_attention_config(config)

    config.dflash_config["causal"] = False
    with pytest.raises(NotImplementedError, match="causal uniform sliding"):
        _validate_dflash_attention_config(config)


def test_reduced_logits_scatter_preserves_target_space_probability_semantics():
    logits = torch.tensor([[2.0, -1.0, 4.0]])
    target_ids = torch.tensor([1, 4, 6])
    expanded = _expand_draft_logits_to_target(logits, target_ids, 8)
    assert expanded.shape == (1, 8)
    assert torch.equal(expanded[:, target_ids], logits)
    assert torch.isneginf(expanded[:, [0, 2, 3, 5, 7]]).all()
    probabilities = expanded.softmax(dim=-1)
    assert torch.equal(probabilities[:, [0, 2, 3, 5, 7]], torch.zeros(1, 5))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))


def test_reduced_vocab_loader_validates_and_precomputes_mapping(monkeypatch):
    wrapper = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(vocab_size=8, hidden_size=3)
    wrapper.draft_vocab_size = 3
    wrapper.has_own_embed_tokens = True
    wrapper.has_own_lm_head = True
    wrapper.lm_head = nn.Embedding(3, 3)
    wrapper.draft_id_to_target_id = nn.Parameter(
        torch.zeros(3, dtype=torch.long), requires_grad=False
    )
    wrapper.register_buffer(
        "draft_target_ids", torch.arange(3, dtype=torch.long), persistent=False
    )
    model = SimpleNamespace(
        load_weights=lambda weights: list(weights),
        build_fused_context_kv_buffers=lambda: None,
        mask_token_id=None,
    )
    wrapper.model = model
    monkeypatch.setattr(wrapper, "_read_mask_embedding", lambda: None)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    membership = torch.tensor(
        [True, False, False, True, False, False, True, False]
    )
    wrapper.load_weights(
        [
            ("embed_tokens.weight", torch.ones(8, 3)),
            ("lm_head.weight", torch.arange(9).view(3, 3).float()),
            ("d2t", offsets),
            ("t2d", membership),
        ]
    )
    assert wrapper.draft_target_ids.tolist() == [0, 3, 6]
    assert torch.equal(wrapper.draft_id_to_target_id, offsets)

    bad_membership = membership.clone()
    bad_membership[7] = True
    with pytest.raises(ValueError, match="does not agree"):
        wrapper.load_weights(
            [
                ("embed_tokens.weight", torch.ones(8, 3)),
                ("lm_head.weight", torch.ones(3, 3)),
                ("d2t", offsets),
                ("t2d", bad_membership),
            ]
        )


def test_packed_weight_loader_requires_every_qkv_and_gate_up_shard():
    names = {
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "norm.weight",
    }
    missing = _missing_packed_shards(
        names,
        {
            "layers.0.self_attn.qkv_proj.weight": {"q", "k"},
            "layers.0.mlp.gate_up_proj.weight": {0, 1},
        },
    )
    assert missing == {
        "layers.0.self_attn.qkv_proj.weight": ["v"],
    }
    assert not _missing_packed_shards(
        names,
        {
            "layers.0.self_attn.qkv_proj.weight": {"q", "k", "v"},
            "layers.0.mlp.gate_up_proj.weight": {0, 1},
        },
    )


class _Projection(nn.Module):
    def __init__(self, hidden: int, q_size: int, kv_size: int, dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(q_size + 2 * kv_size, hidden, dtype=dtype)
        )
        self.bias = None


class _Norm(nn.Module):
    def __init__(self, hidden: int, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, dtype=dtype))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value.float() * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.float()).to(value.dtype)


class _Rotary(nn.Module):
    def __init__(self, head_dim: int, neox: bool):
        super().__init__()
        self.head_size = head_dim
        self.is_neox_style = neox
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        position = torch.arange(64, dtype=torch.float32)
        frequency = torch.outer(position, inv_freq)
        self.register_buffer("cos", frequency.cos(), persistent=False)
        self.register_buffer("sin", frequency.sin(), persistent=False)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((frequency.cos(), frequency.sin()), dim=-1),
            persistent=False,
        )

    def forward(self, positions, query, key=None):
        cos = self.cos[positions.long()].unsqueeze(1).to(query.dtype)
        sin = self.sin[positions.long()].unsqueeze(1).to(query.dtype)

        def rotate(value):
            shape = value.shape
            value = value.view(positions.numel(), -1, self.head_size)
            if self.is_neox_style:
                first, second = value.chunk(2, dim=-1)
                value = torch.cat(
                    (first * cos - second * sin, second * cos + first * sin),
                    -1,
                )
            else:
                first, second = value[..., ::2], value[..., 1::2]
                value = torch.stack(
                    (first * cos - second * sin, second * cos + first * sin),
                    dim=-1,
                ).flatten(-2)
            return value.reshape(shape)

        return rotate(query), rotate(key) if key is not None else None


class _SelfAttention(nn.Module):
    def __init__(self, *, hidden: int, neox: bool, dtype):
        super().__init__()
        self.q_size = hidden
        self.num_kv_heads = 2
        self.head_dim = 128
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = _Projection(hidden, self.q_size, self.kv_size, dtype)
        self.k_norm = _Norm(self.head_dim, dtype)
        self.rotary_emb = _Rotary(self.head_dim, neox)

    def set_rope_impl(self, implementation):
        self._rope_impl = implementation


class _Layer(nn.Module):
    def __init__(self, *, hidden: int, neox: bool, dtype):
        super().__init__()
        self.self_attn = _SelfAttention(hidden=hidden, neox=neox, dtype=dtype)


def _projection_model(*, neox: bool, dtype: torch.dtype) -> DFlashQwen3Model:
    hidden = 256
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=hidden)
    model.layers = nn.ModuleList(
        [_Layer(hidden=hidden, neox=neox, dtype=dtype) for _ in range(3)]
    )
    model.hidden_norm = _Norm(hidden, dtype)
    model._max_context_tokens = 17
    model._fused_context_kv_ready = False
    return model


@pytest.mark.parametrize("neox", [True, False])
def test_fused_context_kv_matches_per_layer_reference_and_reuses_buffers(neox):
    _require_npu()
    torch.manual_seed(20260828)
    dtype = torch.bfloat16
    model = _projection_model(neox=neox, dtype=dtype).to("npu")
    model.set_context_rms_norm_impl(rms_norm_into)
    model.set_context_rope_impl(rotary_embedding_into)
    model.build_fused_context_kv_buffers()
    context = torch.randn(11, 256, dtype=dtype, device="npu")
    positions = torch.tensor(
        [0, 1, 3, 4, 7, 8, 12, 13, 17, 21, 31],
        dtype=torch.int32,
        device="npu",
    )

    normed = model.hidden_norm(context)
    reference_k = []
    reference_v = []
    for layer in model.layers:
        attn = layer.self_attn
        kv = F.linear(normed, attn.qkv_proj.weight[attn.q_size :])
        key, value = kv.split(attn.kv_size, dim=-1)
        key = attn.k_norm(key.view(11, attn.num_kv_heads, attn.head_dim))
        key, _ = attn.rotary_emb(positions, key, key.clone())
        reference_k.append(key)
        reference_v.append(value.view(11, attn.num_kv_heads, attn.head_dim))

    key, value = model.compute_context_kv(context, positions)
    key_ptr = key.data_ptr()
    value_ptr = value.data_ptr()
    torch.npu.synchronize()
    torch.testing.assert_close(
        key, torch.stack(reference_k), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        value, torch.stack(reference_v), atol=5e-2, rtol=5e-2
    )

    key_again, value_again = model.compute_context_kv(context, positions)
    assert key_again.data_ptr() == key_ptr
    assert value_again.data_ptr() == value_ptr


def test_context_workspace_capacity_fails_before_projection():
    model = _projection_model(neox=True, dtype=torch.float32)
    model._max_context_tokens = 2
    model.build_fused_context_kv_buffers()
    with pytest.raises(ValueError, match="buffer capacity"):
        model.compute_context_kv(torch.randn(3, 256), torch.arange(3))


@pytest.mark.parametrize("neox", [True, False])
def test_explicit_npu_rope_handles_different_query_and_kv_widths(neox):
    _require_npu()
    torch.manual_seed(41)
    rotary = _Rotary(128, neox).to(device="npu", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 7, 13, 31], dtype=torch.int32, device="npu")
    query = torch.randn(5, 256, dtype=torch.bfloat16, device="npu")
    key = torch.randn(5, 128, dtype=torch.bfloat16, device="npu")
    expected_query, expected_key = rotary(positions, query, key)
    actual_query = query.clone()
    actual_key = key.clone()
    rotary_embedding_into(
        positions,
        actual_query,
        actual_key,
        rotary.head_size,
        rotary.cos_sin_cache,
        rotary.is_neox_style,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual_query, expected_query, atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        actual_key, expected_key, atol=5e-2, rtol=5e-2
    )


def test_independent_mask_embedding_loads_and_validates_token_id(tmp_path):
    checkpoint = tmp_path / "draft"
    checkpoint.mkdir()
    embedding = torch.arange(16, dtype=torch.bfloat16)
    torch.save(
        {"mask_token_id": 127, "embedding": embedding},
        checkpoint / "mask_embedding.pt",
    )
    wrapper = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(mask_token_id=127)
    wrapper.draft_model_config = SimpleNamespace(
        model=str(checkpoint), revision=None
    )
    assert torch.equal(wrapper._read_mask_embedding(), embedding)

    torch.save(
        {"mask_token_id": 126, "embedding": embedding},
        checkpoint / "mask_embedding.pt",
    )
    with pytest.raises(ValueError, match="does not match"):
        wrapper._read_mask_embedding()
