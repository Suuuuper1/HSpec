import importlib
import inspect

import pytest
import torch

from vllm_ascend.spec_decode.dflash_proposer import DFlashProposer
from vllm_ascend.spec_decode.dspark_proposer import (
    DSparkProposer,
    dspark_markov_greedy,
)
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer


class _FixedDSpark(torch.nn.Module):
    def __init__(self, base_logits, w1, w2):
        super().__init__()
        self.base_logits = base_logits
        self.w1 = w1
        self.w2 = w2
        self.forward_calls = 0
        self.confidence_calls = 0

    def forward(self, *, input_ids, positions):
        self.forward_calls += 1
        return torch.stack((input_ids.float(), positions.float()), dim=-1)

    def compute_draft_logits(self, hidden_states):
        assert hidden_states.shape[0] == self.base_logits.shape[0]
        return self.base_logits.clone()

    def markov_embed_into(self, token_ids, output):
        torch.index_select(self.w1, 0, token_ids, out=output)
        return output

    def markov_bias_into(self, embedding, output):
        torch.mm(embedding, self.w2.t(), out=output)
        return output

    def map_draft_to_target(self, token_ids):
        return token_ids

    def compute_confidence(self, *_):
        self.confidence_calls += 1
        raise AssertionError("confidence must not enter Phase-2 fixed-K runtime")


def _buffers(batch, k, rank, vocab, fill=0.0):
    return (
        torch.full((batch, k + 1), 12345, dtype=torch.int64),
        torch.full((batch, rank), fill, dtype=torch.float32),
        torch.full((batch, vocab), fill, dtype=torch.float32),
    )


def _reference(base_logits, anchors, w1, w2):
    batch, k, _ = base_logits.shape
    output = torch.empty(batch, k, dtype=torch.int64)
    previous = anchors
    for position in range(k):
        logits = base_logits[:, position] + w1[previous] @ w2.t()
        previous = logits.argmax(dim=-1)
        output[:, position] = previous
    return output


@pytest.mark.parametrize("k", [1, 7, 15])
def test_markov_greedy_matches_fixed_reference_for_batched_boundaries(k):
    torch.manual_seed(20260829 + k)
    batch, vocab, rank = 3, 11, 4
    base = torch.randn(batch, k, vocab)
    w1 = torch.randn(vocab, rank)
    w2 = torch.randn(vocab, rank)
    anchors = torch.tensor([0, 4, 9], dtype=torch.int64)
    model = _FixedDSpark(base.flatten(0, 1), w1, w2)
    buffers = _buffers(batch, k, rank, vocab, fill=-9876.0)
    actual = dspark_markov_greedy(
        model, torch.zeros(batch * k, 2), anchors, k, *buffers
    )
    assert torch.equal(actual, _reference(base, anchors, w1, w2))
    assert model.confidence_calls == 0


def test_markov_loop_feeds_selected_token_not_anchor_at_every_position():
    vocab = rank = 6
    k = 4
    base = torch.zeros(2, k, vocab)
    w1 = torch.eye(vocab)
    w2 = torch.zeros(vocab, rank)
    for previous in range(vocab):
        w2[(previous + 1) % vocab, previous] = 10.0
    anchors = torch.tensor([0, 3], dtype=torch.int64)
    model = _FixedDSpark(base.flatten(0, 1), w1, w2)
    actual = dspark_markov_greedy(
        model,
        torch.zeros(2 * k, 2),
        anchors,
        k,
        *_buffers(2, k, rank, vocab),
    )
    assert actual.tolist() == [[1, 2, 3, 4], [4, 5, 0, 1]]
    assert actual.tolist() != [[1, 1, 1, 1], [4, 4, 4, 4]]


def test_markov_buffers_isolate_requests_and_have_no_cross_call_bias_residue():
    torch.manual_seed(91)
    batch, k, vocab, rank = 2, 3, 7, 3
    w1 = torch.randn(vocab, rank)
    w2 = torch.randn(vocab, rank)
    anchors = torch.tensor([1, 5])
    buffers = _buffers(batch, k, rank, vocab, fill=1e9)
    for base in (torch.randn(batch, k, vocab), torch.randn(batch, k, vocab) * 3):
        model = _FixedDSpark(base.flatten(0, 1), w1, w2)
        actual = dspark_markov_greedy(
            model, torch.zeros(batch * k, 2), anchors, k, *buffers
        ).clone()
        assert torch.equal(actual, _reference(base, anchors, w1, w2))


def test_dspark_parallel_backbone_executes_transformer_once_and_returns_b_by_k():
    batch, k, vocab, rank = 2, 3, 8, 2
    model = _FixedDSpark(
        torch.zeros(batch * k, vocab),
        torch.randn(vocab, rank),
        torch.randn(vocab, rank),
    )
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = model
    proposer.num_speculative_tokens = k
    (
        proposer._dspark_token_buffer,
        proposer._dspark_embedding_buffer,
        proposer._dspark_corrected_logits_buffer,
    ) = _buffers(batch, k, rank, vocab)
    tokens = proposer._run_parallel_backbone(
        torch.tensor([4, 9, 9, 5, 9, 9]),
        torch.arange(batch * k),
        torch.arange(batch * k),
        torch.tensor([4, 5]),
    )
    assert model.forward_calls == 1
    assert tokens.shape == (batch, k)
    assert tokens.dtype == torch.int64


def test_dspark_layout_is_config_driven_and_dflash_default_is_unchanged():
    source = inspect.getsource(DFlashProposer._build_parallel_layout)
    assert "self.speculative_config.sample_from_anchor is True" in source
    assert "SAMPLE_FROM_ANCHOR=False" not in source
    assert ParallelBlockProposer._select_draft_tokens is not (
        DSparkProposer._select_draft_tokens
    )


def test_factory_routes_dspark_to_dedicated_proposer(monkeypatch):
    spec_decode = importlib.import_module("vllm_ascend.spec_decode")
    marker = object()

    def construct(vllm_config, device, runner):
        assert (vllm_config, device, runner) == ("config", "npu", "runner")
        return marker

    monkeypatch.setattr(spec_decode, "DSparkProposer", construct)
    assert spec_decode.get_spec_decode_method(
        "dspark", "config", "npu", "runner"
    ) is marker


def test_phase2_hot_path_has_no_host_sync_confidence_or_collectives():
    source = "\n".join(
        (
            inspect.getsource(DSparkProposer._select_draft_tokens),
            inspect.getsource(dspark_markov_greedy),
        )
    )
    for forbidden in (
        ".item(",
        ".tolist(",
        "synchronize(",
        "compute_confidence(",
        "all_reduce(",
        "all_gather(",
        "torch.stack(",
    ):
        assert forbidden not in source
    assert source.count("compute_draft_logits(") == 1
