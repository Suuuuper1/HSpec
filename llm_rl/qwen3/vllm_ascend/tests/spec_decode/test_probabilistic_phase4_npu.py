from types import SimpleNamespace

import pytest
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import rejection_sample, verifier_mode
from vllm_ascend.spec_decode.dspark_proposer import dspark_markov_probabilistic
from vllm_ascend.spec_decode.probabilistic import (
    DraftSamplingWorkspace,
    DraftProbabilityCache,
    sample_draft_logits,
)


def _require_npu() -> None:
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            pytest.skip("torch-npu reports no available NPU")
        torch.empty(1, device="npu")
    except Exception as error:
        pytest.skip(f"NPU runtime unavailable: {type(error).__name__}: {error}")


def _sampling(batch_size: int) -> SamplingMetadata:
    empty = torch.empty(0, dtype=torch.float32, device="npu")
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device="npu"),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=empty,
        presence_penalties=empty,
        repetition_penalties=empty,
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )


def _cpu_standard_reference(
    target: torch.Tensor,
    draft: torch.Tensor,
    samples: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    draft_ids = torch.multinomial(
        draft, samples, replacement=True, generator=generator
    )
    accept_probability = torch.minimum(
        torch.ones(samples), target[draft_ids] / draft[draft_ids]
    )
    accepted = torch.rand(samples, generator=generator) < accept_probability
    recovered_distribution = torch.clamp(target - draft, min=0)
    recovered_distribution /= recovered_distribution.sum()
    recovered = torch.multinomial(
        recovered_distribution, samples, replacement=True, generator=generator
    )
    output = torch.where(accepted, draft_ids, recovered)
    return torch.bincount(output, minlength=target.numel())


def test_npu_probabilistic_proposal_and_standard_rejection_match_cpu_reference():
    _require_npu()
    init_device_properties_triton()
    torch.manual_seed(20260830)
    batch_size, rounds = 64, 128
    target_cpu = torch.tensor([0.05, 0.15, 0.50, 0.20, 0.10])
    draft_cpu = torch.tensor([0.55, 0.10, 0.15, 0.15, 0.05])
    target = target_cpu.to("npu")
    draft_logits = draft_cpu.log().to("npu")
    target_probs = target.repeat(batch_size, 1)
    cumulative = torch.arange(1, batch_size + 1, dtype=torch.int32, device="npu")
    sampling = _sampling(batch_size)
    observed = torch.zeros(target.numel(), dtype=torch.int64)
    for _ in range(rounds):
        proposal, q = sample_draft_logits(
            draft_logits.repeat(batch_size, 1),
            sampling.temperature,
            rows_per_request=1,
            all_greedy=False,
            all_random=True,
        )
        assert q is not None
        torch.testing.assert_close(
            q[0].cpu(), draft_cpu, atol=2e-6, rtol=2e-6
        )
        with verifier_mode("standard"):
            output = rejection_sample(
                proposal.to(torch.int32),
                [1] * batch_size,
                1,
                cumulative,
                q,
                target_probs,
                torch.zeros(batch_size, 1, dtype=torch.int32, device="npu"),
                sampling,
            )
        observed += torch.bincount(output[:, 0].cpu().long(), minlength=5)
    torch.npu.synchronize()

    samples = batch_size * rounds
    reference = _cpu_standard_reference(target_cpu, draft_cpu, samples, 20260831)
    npu_frequency = observed.float() / samples
    cpu_frequency = reference.float() / samples
    npu_tv = 0.5 * torch.abs(npu_frequency - target_cpu).sum()
    cpu_tv = 0.5 * torch.abs(cpu_frequency - target_cpu).sum()
    cross_tv = 0.5 * torch.abs(npu_frequency - cpu_frequency).sum()
    assert npu_tv.item() < 0.03
    assert cpu_tv.item() < 0.03
    assert cross_tv.item() < 0.04


def test_npu_mixed_temperature_stores_one_hot_and_random_q():
    _require_npu()
    logits = torch.tensor(
        [[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        device="npu",
    )
    tokens, q = sample_draft_logits(
        logits.clone(),
        torch.tensor([0.0, 0.8], device="npu"),
        rows_per_request=2,
        all_greedy=False,
        all_random=False,
    )
    assert q is not None
    assert tokens[:2].cpu().tolist() == [1, 0]
    assert q[:2].cpu().tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    torch.testing.assert_close(
        q[2:].cpu(), torch.softmax(logits[2:].cpu() / 0.8, dim=-1)
    )
    cache = DraftProbabilityCache(True, 2, 3)
    cache.publish(q.view(2, 2, 3).contiguous(), ["greedy", "random"])
    aligned = cache.consume(
        ["random", "greedy"], [1, 2], require_probabilities=True
    )
    assert aligned is not None and aligned.shape == (3, 3)
    assert cache.snapshot()["validation_sync_count"] == 1


def test_npu_persistent_sampling_workspace_is_exact_and_address_stable():
    _require_npu()
    batch_size, rows_per_request, vocab_size = 3, 7, 257
    rows = batch_size * rows_per_request
    logits = torch.randn(rows, vocab_size, dtype=torch.bfloat16, device="npu")
    temperatures = torch.tensor([0.0, 0.8, 1.2], device="npu")
    workspace = DraftSamplingWorkspace(rows, vocab_size, device=torch.device("npu"))

    torch.npu.manual_seed(20260830)
    expected_tokens, expected_q = sample_draft_logits(
        logits.clone(),
        temperatures,
        rows_per_request=rows_per_request,
        all_greedy=False,
        all_random=False,
    )
    torch.npu.manual_seed(20260830)
    actual_tokens, actual_q = sample_draft_logits(
        logits.clone(),
        temperatures,
        rows_per_request=rows_per_request,
        all_greedy=False,
        all_random=False,
        workspace=workspace,
    )
    torch.npu.synchronize()
    assert actual_q is not None and expected_q is not None
    assert torch.equal(actual_tokens.cpu(), expected_tokens.cpu())
    assert torch.equal(actual_q.cpu(), expected_q.cpu())
    pointers = (actual_q.data_ptr(), workspace.race.data_ptr())

    _, second_q = sample_draft_logits(
        logits.clone(),
        temperatures,
        rows_per_request=rows_per_request,
        all_greedy=False,
        all_random=False,
        workspace=workspace,
    )
    assert second_q is not None
    assert (second_q.data_ptr(), workspace.race.data_ptr()) == pointers


class _NPUFixedDSpark(torch.nn.Module):
    def __init__(self, base_logits: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
        super().__init__()
        self.base_logits = base_logits
        self.w1 = w1
        self.w2 = w2

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[0] == self.base_logits.shape[0]
        return self.base_logits.clone()

    def markov_embed_into(self, token_ids: torch.Tensor, output: torch.Tensor):
        return torch.index_select(self.w1, 0, token_ids, out=output)

    def markov_bias_into(self, embedding: torch.Tensor, output: torch.Tensor):
        return torch.mm(embedding, self.w2.t(), out=output)

    @staticmethod
    def map_draft_to_target(token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids


def test_npu_dspark_probability_rows_follow_sampled_markov_chain():
    _require_npu()
    torch.manual_seed(20260832)
    batch, k, vocab, rank = 2, 3, 7, 3
    base = torch.randn(batch, k, vocab, device="npu")
    w1 = torch.randn(vocab, rank, device="npu")
    w2 = torch.randn(vocab, rank, device="npu")
    anchors = torch.tensor([1, 5], dtype=torch.int64, device="npu")
    temperatures = torch.tensor([0.7, 1.2], device="npu")
    model = _NPUFixedDSpark(base.flatten(0, 1), w1, w2)
    tokens, q = dspark_markov_probabilistic(
        model,
        torch.zeros(batch * k, 2, device="npu"),
        anchors,
        k,
        torch.empty(batch, k + 1, dtype=torch.int64, device="npu"),
        torch.empty(batch, rank, device="npu"),
        torch.empty(batch, vocab, device="npu"),
        SimpleNamespace(
            temperature=temperatures,
            all_greedy=False,
            all_random=True,
            generators={},
        ),
    )
    assert q is not None
    previous = anchors
    for position in range(k):
        expected = torch.softmax(
            (base[:, position] + w1[previous] @ w2.t())
            / temperatures.unsqueeze(-1),
            dim=-1,
            dtype=torch.float32,
        )
        torch.testing.assert_close(q[:, position], expected, atol=2e-5, rtol=2e-5)
        previous = tokens[:, position]
