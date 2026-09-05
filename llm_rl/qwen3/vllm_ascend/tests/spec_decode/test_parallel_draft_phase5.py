from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.spec_decode.parallel_draft_metrics as metrics_module
from vllm_ascend.spec_decode.parallel_draft_metrics import (
    ParallelDraftMetrics,
    phase5_capability_manifest,
)
from vllm_ascend.spec_decode.probabilistic import (
    DraftSamplingWorkspace,
    sample_draft_logits,
)


class _FakeEvent:
    clock = 0

    def __init__(self, enable_timing: bool = False):
        assert enable_timing
        self.value = 0

    def record(self):
        type(self).clock += 1
        self.value = type(self).clock

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return float(other.value - self.value)


def test_sampled_metrics_are_disabled_by_default_and_bounded(monkeypatch):
    monkeypatch.setattr(metrics_module, "Event", _FakeEvent)
    disabled = ParallelDraftMetrics(
        method="dflash", enabled=False, sample_every=1, flush_every=1
    )
    disabled.begin_step(batch_size=2, context_tokens=8, k=7)
    with disabled.timer("spec/draft_backbone_ms"):
        pass
    disabled.end_step()
    snapshot = disabled.snapshot()
    assert snapshot["proposal_steps"] == 1
    assert snapshot["sampled_steps"] == 0
    assert snapshot["timers"] == {}

    observed = ParallelDraftMetrics(
        method="dspark", enabled=True, sample_every=2, flush_every=2
    )
    for ordinal in range(4):
        observed.begin_step(batch_size=ordinal + 1, context_tokens=8, k=7)
        with observed.timer("spec/dspark_markov_ms"):
            pass
        observed.add_counter("spec/draft_tokens", 7)
        observed.add_device_counter(
            "spec/draft_skipped_max_len", torch.tensor(ordinal % 2)
        )
        observed.end_step()
    snapshot = observed.snapshot()
    assert snapshot["proposal_steps"] == 4
    assert snapshot["sampled_steps"] == 2
    assert snapshot["flush_count"] == 1
    assert snapshot["timers"]["spec/dspark_markov_ms"]["count"] == 2
    assert snapshot["counters"]["spec/draft_tokens"] == 14
    assert snapshot["counters"]["spec/draft_skipped_max_len"] == 0
    assert snapshot["transfers"] == {
        "new_hotpath_h2d_bytes": 0,
        "profile_d2h_scalar_count": 2,
    }
    assert len(snapshot["shape_samples"]) == 2


def test_persistent_sampling_workspace_preserves_q_and_addresses():
    rows, vocab = 6, 11
    workspace = DraftSamplingWorkspace(rows, vocab, device=torch.device("cpu"))
    q_pointer = workspace.probabilities.data_ptr()
    race_pointer = workspace.race.data_ptr()
    base = torch.randn(rows, vocab)
    temperatures = torch.tensor([0.7, 1.3])

    torch.manual_seed(19)
    expected_tokens, expected_q = sample_draft_logits(
        base.clone(), temperatures, rows_per_request=3,
        all_greedy=False, all_random=True,
    )
    torch.manual_seed(19)
    actual_tokens, actual_q = sample_draft_logits(
        base.clone(), temperatures, rows_per_request=3,
        all_greedy=False, all_random=True, workspace=workspace,
    )
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(actual_q, expected_q)
    assert actual_q is not None and actual_q.data_ptr() == q_pointer
    assert workspace.race.data_ptr() == race_pointer

    _, second_q = sample_draft_logits(
        base.clone(), temperatures, rows_per_request=3,
        all_greedy=False, all_random=True, workspace=workspace,
    )
    assert second_q is not None and second_q.data_ptr() == q_pointer
    assert workspace.race.data_ptr() == race_pointer


def test_capability_manifest_separates_stable_and_experimental_paths():
    config = SimpleNamespace(
        method="dflash",
        draft_sample_method="probabilistic",
        rejection_sample_method="standard",
        num_speculative_tokens=7,
        parallel_draft_profile_enabled=True,
        target_model_config=SimpleNamespace(get_vocab_size=lambda: 151936),
        draft_model_config=SimpleNamespace(
            get_vocab_size=lambda: 151936,
            hf_text_config=SimpleNamespace(draft_vocab_size=32000),
        ),
    )
    manifest = phase5_capability_manifest(config)
    assert manifest["stable_baseline"] == {
        "draft_execution": "eager",
        "proposal": "probabilistic",
        "verification": "standard",
        "fixed_k": 7,
        "full_vocab": True,
        "vocabulary": {
            "internal_draft_vocab_size": 32000,
            "output_vocab_size": 151936,
            "target_vocab_size": 151936,
            "target_probability_vocab": True,
            "mapped_reduced_vocab": True,
            "mode": "mapped_reduced",
        },
    }
    assert manifest["enabled"]["persistent_sampling_workspace"] is False
    assert manifest["evaluated_and_rolled_back"]["persistent_sampling_workspace"]
    assert all(manifest["unsupported_fail_closed"].values())


def test_workspace_rejects_capacity_overrun():
    workspace = DraftSamplingWorkspace(2, 3, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="exceed"):
        workspace.rows(3)
