from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.spec_decode.dspark_proposer import dspark_markov_probabilistic
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer
from vllm_ascend.spec_decode.probabilistic import (
    DraftProbabilityCache,
    estimate_draft_probability_bytes,
    sample_draft_logits,
)


def _metadata(
    temperatures: torch.Tensor,
    *,
    all_greedy: bool = False,
    all_random: bool = True,
    generators: dict[int, torch.Generator] | None = None,
):
    return SimpleNamespace(
        temperature=temperatures,
        all_greedy=all_greedy,
        all_random=all_random,
        generators=generators or {},
    )


def test_probability_memory_estimate_is_method_specific_and_exact():
    dflash = estimate_draft_probability_bytes(4, 7, 11, method="dflash")
    dspark = estimate_draft_probability_bytes(4, 7, 11, method="dspark")
    assert dflash == {
        "probability_bytes": 4 * 7 * 11 * 4,
        "row_bytes": 4 * 11 * 4,
        "scratch_bytes": 4 * 7 * 11 * 4,
        "peak_bytes": 2 * 4 * 7 * 11 * 4,
    }
    assert dspark["scratch_bytes"] == 2 * 4 * 11 * 4
    assert dspark["peak_bytes"] == dspark["probability_bytes"] + 2 * 4 * 11 * 4
    with pytest.raises(ValueError, match="unsupported"):
        estimate_draft_probability_bytes(1, 1, 1, method="unknown")


def test_fixed_logits_q_temperature_expansion_and_sampling_frequency():
    trials = 40_000
    base = torch.tensor([[0.2, 1.1, -0.7]], dtype=torch.float32)
    logits = base.expand(trials, -1).clone()
    temperatures = torch.full((trials,), 0.8)
    torch.manual_seed(20260830)
    tokens, q = sample_draft_logits(
        logits,
        temperatures,
        rows_per_request=1,
        all_greedy=False,
        all_random=True,
    )
    expected = torch.softmax(base[0] / 0.8, dim=-1)
    assert q is not None and q.dtype == torch.float32 and q.is_contiguous()
    assert torch.allclose(q[0], expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(q.sum(dim=-1), torch.ones(trials), atol=1e-6)
    observed = torch.bincount(tokens, minlength=3).float() / trials
    assert torch.max(torch.abs(observed - expected)) < 0.012


def test_request_major_k_rows_and_mixed_greedy_random_q_are_exact():
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 4.0],
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
        ]
    )
    original = logits.clone()
    tokens, q = sample_draft_logits(
        logits,
        torch.tensor([0.0, 0.5]),
        rows_per_request=3,
        all_greedy=False,
        all_random=False,
    )
    assert q is not None
    greedy = original[:3].argmax(dim=-1)
    assert torch.equal(tokens[:3], greedy)
    assert torch.equal(q[:3].argmax(dim=-1), greedy)
    assert torch.equal(q[:3].sum(dim=-1), torch.ones(3))
    assert torch.all((q[:3] == 0).sum(dim=-1) == 2)
    assert torch.allclose(q[3:], torch.softmax(original[3:] / 0.5, dim=-1))


def test_all_greedy_probabilistic_batch_emits_no_q():
    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0]])
    tokens, q = sample_draft_logits(
        logits,
        torch.zeros(2),
        rows_per_request=1,
        all_greedy=True,
        all_random=False,
    )
    assert tokens.tolist() == [1, 0]
    assert q is None


def test_private_generators_are_reproducible_without_global_rng_consumption():
    logits = torch.randn(6, 5)

    def run():
        generators = {
            0: torch.Generator().manual_seed(17),
            1: torch.Generator().manual_seed(29),
        }
        before = torch.random.get_rng_state().clone()
        tokens, q = sample_draft_logits(
            logits.clone(),
            torch.tensor([0.7, 1.2]),
            rows_per_request=3,
            all_greedy=False,
            all_random=True,
            generators=generators,
        )
        after = torch.random.get_rng_state()
        return tokens, q, before, after

    torch.manual_seed(1234)
    first = run()
    torch.manual_seed(1234)
    second = run()
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], first[3])
    assert torch.equal(second[2], second[3])


def _probabilities(batch: int, k: int, vocab: int) -> torch.Tensor:
    values = torch.arange(1, batch * k * vocab + 1, dtype=torch.float32)
    values = values.view(batch, k, vocab)
    return values / values.sum(dim=-1, keepdim=True)


def _inject_negative_probability(q: torch.Tensor) -> torch.Tensor:
    first = q[..., 0].clone()
    q[..., 0] = -0.1
    q[..., 1].add_(first + 0.1)
    return q


def test_probability_cache_fast_path_reorder_finish_refill_and_variable_k():
    cache = DraftProbabilityCache(True, 3, 5)
    q = _probabilities(3, 3, 5)
    cache.publish(q, ["a", "b", "c"])
    fast = cache.consume(["a", "b", "c"], [3, 3, 3], require_probabilities=True)
    assert fast is not None and fast.data_ptr() == q.data_ptr()
    assert cache.current_bytes == 0

    cache.publish(q, ["old", "keep", "finish"])
    aligned = cache.consume(
        ["keep", "new", "old", "finish"],
        [2, 0, 1, 0],
        require_probabilities=True,
    )
    assert aligned is not None
    assert torch.equal(aligned, torch.cat((q[1, :2], q[0, :1]), dim=0))
    assert aligned.is_contiguous()
    assert cache.snapshot()["consume_count"] == 2


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda q: q[:, :, :-1], "shape mismatch"),
        (lambda q: q.double(), "float32"),
        (
            lambda q: q.transpose(1, 2).contiguous().transpose(1, 2),
            "contiguous",
        ),
        (lambda q: q.fill_(float("nan")), "non-finite"),
        (_inject_negative_probability, "negative"),
        (lambda q: q.mul_(0.5), "non-normalized"),
    ],
)
def test_probability_cache_publish_validation_is_fail_closed(mutation, error):
    cache = DraftProbabilityCache(True, 2, 3)
    with pytest.raises((TypeError, ValueError), match=error):
        cache.publish(mutation(_probabilities(2, 2, 3)), ["a", "b"])


def test_probability_cache_missing_duplicate_stale_and_greedy_fail_closed():
    cache = DraftProbabilityCache(True, 2, 3)
    with pytest.raises(RuntimeError, match="missing"):
        cache.consume(["a"], [1], require_probabilities=True)

    with pytest.raises(ValueError, match="unique"):
        cache.publish(_probabilities(2, 2, 3), ["a", "a"])

    cache.publish(_probabilities(1, 2, 3), ["a"])
    with pytest.raises(ValueError, match="duplicated"):
        cache.consume(["a", "a"], [1, 1], require_probabilities=True)
    assert cache.current_bytes == 0

    cache.publish(_probabilities(1, 2, 3), ["a"])
    with pytest.raises(RuntimeError, match="request 'b'"):
        cache.consume(["b"], [1], require_probabilities=True)
    assert cache.current_bytes == 0

    cache.publish(_probabilities(1, 2, 3), ["a"])
    with pytest.raises(RuntimeError, match="stale"):
        cache.consume(["a"], [1], require_probabilities=False)
    assert cache.current_bytes == 0


class _FixedHead:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[0] == self.logits.shape[0]
        return self.logits.clone()


def test_dflash_select_preserves_b_k_v_probabilities():
    batch, k, vocab = 2, 3, 5
    logits = torch.randn(batch * k, vocab)
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.model = _FixedHead(logits)
    proposer.num_speculative_tokens = k
    proposer._probabilistic = True
    proposer._last_draft_probs = None
    tokens = proposer._select_draft_tokens(
        torch.zeros(batch * k, 4),
        None,
        _metadata(torch.tensor([0.6, 1.1])),
    )
    q = proposer.take_last_draft_probs()
    assert tokens.shape == (batch, k)
    assert q is not None and q.shape == (batch, k, vocab)
    assert torch.allclose(q[0], torch.softmax(logits[:k] / 0.6, dim=-1))
    assert torch.allclose(q[1], torch.softmax(logits[k:] / 1.1, dim=-1))
    assert proposer.take_last_draft_probs() is None


class _FixedDSpark(torch.nn.Module):
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


def test_dspark_probabilistic_q_uses_actual_sequential_token_feedback():
    torch.manual_seed(73)
    batch, k, vocab, rank = 2, 4, 7, 3
    base = torch.randn(batch, k, vocab)
    w1 = torch.randn(vocab, rank)
    w2 = torch.randn(vocab, rank)
    anchors = torch.tensor([1, 5])
    model = _FixedDSpark(base.flatten(0, 1), w1, w2)
    tokens, q = dspark_markov_probabilistic(
        model,
        torch.zeros(batch * k, 2),
        anchors,
        k,
        torch.empty(batch, k + 1, dtype=torch.int64),
        torch.empty(batch, rank),
        torch.empty(batch, vocab),
        _metadata(torch.tensor([0.7, 1.3])),
    )
    assert q is not None and q.shape == (batch, k, vocab)
    previous = anchors
    for position in range(k):
        corrected = base[:, position] + w1[previous] @ w2.t()
        expected = torch.softmax(
            corrected / torch.tensor([0.7, 1.3]).unsqueeze(-1), dim=-1
        )
        assert torch.allclose(q[:, position], expected, atol=1e-6, rtol=1e-5)
        previous = tokens[:, position]
