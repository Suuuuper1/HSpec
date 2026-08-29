import pytest
import torch

from vllm_ascend.spec_decode.dspark_proposer import dspark_markov_greedy


def _require_npu() -> None:
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            pytest.skip("torch-npu reports no available NPU")
        torch.empty(1, device="npu")
    except Exception as error:
        pytest.skip(f"NPU runtime unavailable: {type(error).__name__}: {error}")


class _NpuMarkovModel:
    def __init__(self, base_logits, w1, w2):
        self.base_logits = base_logits
        self.w1 = w1
        self.w2 = w2

    def compute_draft_logits(self, hidden_states):
        del hidden_states
        return self.base_logits.clone()

    def markov_embed_into(self, token_ids, output):
        torch.index_select(self.w1, 0, token_ids, out=output)

    def markov_bias_into(self, embedding, output):
        torch.mm(embedding, self.w2.t(), out=output)

    def map_draft_to_target(self, token_ids):
        return token_ids


@pytest.mark.parametrize("k", [1, 7, 15])
def test_npu_markov_feedback_boundaries_and_persistent_buffers(k):
    _require_npu()
    batch = 3
    vocab = rank = 16
    dtype = torch.bfloat16
    base = torch.zeros(batch * k, vocab, dtype=dtype, device="npu")
    w1 = torch.eye(vocab, dtype=dtype, device="npu")
    w2 = torch.zeros(vocab, rank, dtype=dtype, device="npu")
    for previous in range(vocab):
        w2[(previous + 1) % vocab, previous] = 16
    anchors = torch.tensor([0, 5, 13], dtype=torch.int64, device="npu")
    model = _NpuMarkovModel(base, w1, w2)
    tokens = torch.full(
        (batch, k + 1), 777, dtype=torch.int64, device="npu"
    )
    embeddings = torch.full(
        (batch, rank), -123, dtype=dtype, device="npu"
    )
    corrected = torch.full(
        (batch, vocab), -123, dtype=dtype, device="npu"
    )
    pointers = (
        tokens.data_ptr(),
        embeddings.data_ptr(),
        corrected.data_ptr(),
    )

    output = dspark_markov_greedy(
        model,
        torch.zeros(batch * k, 2, dtype=dtype, device="npu"),
        anchors,
        k,
        tokens,
        embeddings,
        corrected,
    )
    torch.npu.synchronize()
    expected = [
        [(int(anchor) + step + 1) % vocab for step in range(k)]
        for anchor in anchors.cpu()
    ]
    assert output.cpu().tolist() == expected
    assert pointers == (
        tokens.data_ptr(),
        embeddings.data_ptr(),
        corrected.data_ptr(),
    )

    # A second call with a different anchor proves that the full [B,V] scratch
    # is overwritten and no bias from the first proposal survives.
    second_anchors = torch.tensor([2, 7, 10], dtype=torch.int64, device="npu")
    second = dspark_markov_greedy(
        model,
        torch.zeros(batch * k, 2, dtype=dtype, device="npu"),
        second_anchors,
        k,
        tokens,
        embeddings,
        corrected,
    )
    torch.npu.synchronize()
    second_expected = [
        [(int(anchor) + step + 1) % vocab for step in range(k)]
        for anchor in second_anchors.cpu()
    ]
    assert second.cpu().tolist() == second_expected
