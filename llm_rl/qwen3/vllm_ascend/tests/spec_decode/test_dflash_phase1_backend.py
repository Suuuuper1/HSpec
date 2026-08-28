from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.context_kv import store_context_kv
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    rejection_sample,
    verifier_mode,
)


def _require_npu() -> None:
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            pytest.skip("torch-npu reports no available NPU")
        torch.empty(1, device="npu")
    except Exception as error:
        pytest.skip(f"NPU runtime unavailable: {type(error).__name__}: {error}")


def _bare_attention(cache: torch.Tensor) -> Attention:
    attention = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.layer_name = "model.layers.36.self_attn.attn"
    attention.kv_cache = [cache]
    return attention


def test_direct_context_store_changes_only_selected_slots():
    _require_npu()
    cache = torch.full(
        (2, 3, 128, 2, 4), -19.0, dtype=torch.float16, device="npu"
    )
    attention = _bare_attention(cache)
    slots = torch.tensor([0, 129, -1, 255], dtype=torch.int32, device="npu")
    key = torch.arange(32, dtype=torch.float16, device="npu").view(4, 2, 4)
    value = key + 100
    store_context_kv(attention, key, value, slots)
    torch.npu.synchronize()

    flat_key = cache[0].view(-1, 2, 4).cpu()
    flat_value = cache[1].view(-1, 2, 4).cpu()
    assert torch.equal(flat_key[0], key[0].cpu())
    assert torch.equal(flat_key[129], key[1].cpu())
    assert torch.equal(flat_key[255], key[3].cpu())
    assert torch.equal(flat_value[0], value[0].cpu())
    changed = (flat_key != -19).any(dim=-1).any(dim=-1).nonzero().flatten()
    assert changed.tolist() == [0, 129, 255]


@pytest.mark.parametrize(
    "key,slots,error",
    [
        (torch.zeros(2, 8), torch.zeros(2, dtype=torch.int32), r"\[T,nkv,d\]"),
        (torch.zeros(2, 2, 4), torch.zeros(2, dtype=torch.int64), "int32"),
    ],
)
def test_direct_context_store_validates_layout_before_device_operator(
    key, slots, error
):
    cache = torch.zeros(2, 1, 128, 2, 4)
    attention = _bare_attention(cache)
    with pytest.raises((TypeError, ValueError), match=error):
        store_context_kv(attention, key, key, slots)


@pytest.mark.parametrize("causal", [False, True])
def test_decoder_fia_selects_causality_from_metadata(monkeypatch, causal):
    import vllm_ascend.attention.attention_v1 as attention_module

    calls = []

    def fake_fia(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["query"]), None

    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(capturing=False),
    )
    monkeypatch.setattr(
        attention_module.torch_npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = None
    impl.attn_type = AttentionType.DECODER
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.scale = 0.5
    impl._get_fia_params = MethodType(
        lambda self, key, value, metadata: (key, value, 128, torch.zeros(1, 1), [5]),
        impl,
    )
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.ChunkedPrefill,
        actual_seq_lengths_q=(3,),
        attn_mask=torch.ones(4, 4, dtype=torch.bool),
        causal=causal,
    )
    query = torch.randn(3, 8)
    output = torch.empty(3, 2, 4)
    impl.forward_fused_infer_attention(
        query, torch.randn(5, 4), torch.randn(5, 4), metadata, output
    )

    assert len(calls) == 1
    assert calls[0]["sparse_mode"] == (3 if causal else 0)
    assert ("atten_mask" in calls[0]) is causal


def test_noncausal_fia_matches_multi_request_dense_reference(monkeypatch):
    import vllm_ascend.attention.attention_v1 as attention_module

    _require_npu()
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(capturing=False),
    )
    torch.manual_seed(20260828)
    num_heads = 4
    num_kv_heads = 2
    head_size = 16
    scale = head_size**-0.5
    cumulative_lengths = (3, 7)
    query = torch.randn(
        7, num_heads, head_size, dtype=torch.bfloat16, device="npu"
    )
    key = torch.randn(
        7, num_kv_heads, head_size, dtype=torch.bfloat16, device="npu"
    )
    value = torch.randn_like(key)

    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = None
    impl.attn_type = AttentionType.DECODER
    impl.num_heads = num_heads
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_size
    impl.scale = scale
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.PrefillNoCache,
        actual_seq_lengths_q=cumulative_lengths,
        causal=False,
    )
    output = torch.empty_like(query)
    impl.forward_fused_infer_attention(
        query, key, value, metadata, output
    )
    torch.npu.synchronize()

    reference = torch.empty_like(query, device="cpu", dtype=torch.float32)
    causal_reference = torch.empty_like(reference)
    start = 0
    for end in cumulative_lengths:
        q = query[start:end].float().cpu().transpose(0, 1)
        k = key[start:end].float().cpu().repeat_interleave(
            num_heads // num_kv_heads, dim=1
        ).transpose(0, 1)
        v = value[start:end].float().cpu().repeat_interleave(
            num_heads // num_kv_heads, dim=1
        ).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        dense = torch.matmul(scores.softmax(dim=-1), v).transpose(0, 1)
        causal_mask = torch.ones(
            end - start, end - start, dtype=torch.bool
        ).triu(diagonal=1)
        causal_scores = scores.masked_fill(causal_mask, float("-inf"))
        causal = torch.matmul(
            causal_scores.softmax(dim=-1), v
        ).transpose(0, 1)
        reference[start:end].copy_(dense)
        causal_reference[start:end].copy_(causal)
        start = end

    torch.testing.assert_close(
        output.float().cpu(), reference, atol=3e-2, rtol=3e-2
    )
    assert not torch.allclose(reference, causal_reference, atol=1e-2, rtol=1e-2)


def test_random_target_with_greedy_one_hot_proposal_preserves_distribution():
    _require_npu()
    init_device_properties_triton()
    torch.manual_seed(20260828)
    # The old Ascend rejection kernel launches one program per request.  Keep
    # each launch representative of an RL rollout micro-batch and accumulate
    # independent launches for statistical power instead of using an
    # unrealistic 20k-request grid.
    batch_size = 64
    num_rounds = 128
    target_distribution = torch.tensor(
        [0.05, 0.15, 0.50, 0.20, 0.10],
        dtype=torch.float32,
        device="npu",
    )
    target_probs = target_distribution.repeat(batch_size, 1)
    draft_tokens = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="npu"
    )
    cumulative = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device="npu"
    )
    empty_float = torch.empty(0, dtype=torch.float32, device="npu")
    sampling = SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device="npu"),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=empty_float,
        presence_penalties=empty_float,
        repetition_penalties=empty_float,
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )
    observed = torch.zeros(5, dtype=torch.int64)
    for _ in range(num_rounds):
        bonus = torch.multinomial(
            target_distribution, batch_size, replacement=True
        ).view(batch_size, 1).to(torch.int32)
        output = rejection_sample(
            draft_tokens,
            [1] * batch_size,
            1,
            cumulative,
            None,
            target_probs,
            bonus,
            sampling,
        )
        observed += torch.bincount(
            output[:, 0].cpu().long(), minlength=5
        )
    torch.npu.synchronize()
    observed = observed.float() / (batch_size * num_rounds)
    total_variation = 0.5 * torch.abs(observed - target_distribution.cpu()).sum()
    assert total_variation.item() < 0.025


@pytest.mark.parametrize(
    "mode,expected_kernel",
    [("standard", "standard"), ("block", "block"), ("legacy", "block")],
)
def test_verifier_mode_selects_kernel_explicitly(
    monkeypatch, mode, expected_kernel
):
    import vllm_ascend.sample.rejection_sampler as rejection_module

    calls = []

    class FakeKernel:
        def __init__(self, label):
            self.label = label

        def __getitem__(self, grid):
            del grid

            def launch(*args, **kwargs):
                del args, kwargs
                calls.append(self.label)

            return launch

    monkeypatch.setattr(rejection_module, "HAS_TRITON", True)
    monkeypatch.setattr(rejection_module, "cal_grid_and_block_size", lambda _: (1, 1))
    monkeypatch.setattr(
        rejection_module,
        "generate_uniform_probs",
        lambda num_tokens, *args: torch.full((num_tokens,), 0.5),
    )
    monkeypatch.setattr(
        rejection_module,
        "sample_recovered_tokens",
        lambda *args, **kwargs: torch.zeros(7, dtype=torch.int32),
    )
    monkeypatch.setattr(
        rejection_module, "rejection_random_sample_kernel", FakeKernel("standard")
    )
    monkeypatch.setattr(
        rejection_module,
        "rejection_random_sample_block_verify_kernel",
        FakeKernel("block"),
    )
    sampling = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(1),
        generators={},
    )
    with verifier_mode(mode):
        rejection_sample(
            torch.zeros(7, dtype=torch.int32),
            [7],
            7,
            torch.tensor([7], dtype=torch.int32),
            None,
            torch.full((7, 5), 0.2),
            torch.zeros(1, 1, dtype=torch.int32),
            sampling,
        )
    assert calls == [expected_kernel]


def test_ascend_rejection_sampler_binds_mode_to_instance(monkeypatch):
    observed = []

    def fake_forward(self, *args, **kwargs):
        del self, args, kwargs
        import vllm_ascend.sample.rejection_sampler as rejection_module

        observed.append(rejection_module._VERIFIER_MODE.get())
        return "sampled"

    monkeypatch.setattr(RejectionSampler, "forward", fake_forward)
    sampler = AscendRejectionSampler.__new__(AscendRejectionSampler)
    nn.Module.__init__(sampler)
    sampler.verifier_mode = "standard"
    assert sampler() == "sampled"
    assert observed == ["standard"]
