import hashlib
import json
from pathlib import Path

import numpy as np

from vllm_ascend.spec_decode.hspec_selector_online import (
    HSpecS14Config,
    continuation_candidate_slot,
    make_verified_continuation,
    rollout_posterior_signal,
    saturating_rollout_feedback_update,
    select_s14_candidate,
)
from vllm_ascend.spec_decode.hspec_s14_trace import (
    HSpecS14RequestSampler,
    HSpecS14TraceRecorder,
)
from vllm_ascend.spec_decode.hspec_selector_survival import (
    HSPEC_SURVIVAL_ACTIONS,
    score_utility_candidates_one_prompt_python,
    score_utility_one_prompt_python,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _sha(path)


def test_continuation_requires_exact_emitted_segment_and_existing_topk():
    rollout = np.asarray([10, 11, 12, 13, 14], dtype=np.int32)
    state = make_verified_continuation(
        prompt_id="p", table_version=3, cache_generation=7,
        rollout_idx=0, matched_pos=0, emitted_token_ids=[11, 12],
        rollout_tokens=rollout,
    )
    assert state is not None and state.expected_matched_pos == 2
    assert state.source_matched_pos == 0
    assert state.emitted_token_ids == (11, 12)
    assert make_verified_continuation(
        prompt_id="p", table_version=3, cache_generation=7,
        rollout_idx=0, matched_pos=0, emitted_token_ids=[11, 99],
        rollout_tokens=rollout,
    ) is None
    indices = np.asarray([2, 1], dtype=np.int64)
    entry_rollout = np.asarray([0, 0, 0], dtype=np.int32)
    entry_offset = np.asarray([1, 2, 3], dtype=np.int32)
    assert continuation_candidate_slot(
        state, prompt_id="p", table_version=3, cache_generation=7,
        candidate_indices=indices, entry_rollout_idx=entry_rollout,
        entry_offset=entry_offset,
    ) == 0
    assert continuation_candidate_slot(
        state, prompt_id="p", table_version=3, cache_generation=8,
        candidate_indices=indices, entry_rollout_idx=entry_rollout,
        entry_offset=entry_offset,
    ) == -1
    assert continuation_candidate_slot(
        state, prompt_id="other", table_version=3, cache_generation=7,
        candidate_indices=indices, entry_rollout_idx=entry_rollout,
        entry_offset=entry_offset,
    ) == -1


def test_rollout_feedback_is_independent_shrunk_and_saturating():
    trials = np.zeros(2, dtype=np.uint16)
    first = np.zeros(2, dtype=np.uint16)
    accepted = np.zeros(2, dtype=np.uint32)
    drafted = np.zeros(2, dtype=np.uint32)
    signal0, p10, pt0, reliability0 = rollout_posterior_signal(0, 0, 0, 0)
    assert (signal0, p10, pt0, reliability0) == (0.0, 0.5, 0.5, 0.0)
    assert saturating_rollout_feedback_update(
        1, 3, 4, trials, first, accepted, drafted
    )
    signal1, p11, pt1, reliability1 = rollout_posterior_signal(
        trials[1], first[1], accepted[1], drafted[1]
    )
    assert trials.tolist() == [0, 1]
    assert first.tolist() == [0, 1]
    assert accepted.tolist() == [0, 3]
    assert drafted.tolist() == [0, 4]
    assert 0.0 < reliability1 < 1.0
    assert p11 > p10 and pt1 > pt0 and signal1 > signal0


def test_s14_rerank_cannot_revive_abstained_or_below_threshold_candidate():
    utilities = np.asarray([0.20, 0.19, -0.2], dtype=np.float64)
    indices = np.asarray([0, 1, 2], dtype=np.int64)
    rollouts = np.asarray([0, 1, 2], dtype=np.int32)
    trials = np.asarray([0, 8, 8], dtype=np.uint16)
    first = np.asarray([0, 8, 8], dtype=np.uint16)
    accepted = np.asarray([0, 32, 32], dtype=np.uint32)
    drafted = np.asarray([0, 32, 32], dtype=np.uint32)
    slot, adjusted, _ = select_s14_candidate(
        base_slot=0, utilities=utilities, candidate_indices=indices,
        entry_rollout_idx=rollouts, continuation_slot=1,
        rollout_trials=trials, rollout_first_accepts=first,
        rollout_accept_sum=accepted, rollout_draft_sum=drafted,
        utility_threshold=0.0, mode="joint", continuation_weight=0.1,
        posterior_weight=1.0,
    )
    assert slot == 1
    assert np.isneginf(adjusted[2])
    abstained, _, _ = select_s14_candidate(
        base_slot=-1, utilities=utilities, candidate_indices=indices,
        entry_rollout_idx=rollouts, continuation_slot=1,
        rollout_trials=trials, rollout_first_accepts=first,
        rollout_accept_sum=accepted, rollout_draft_sum=drafted,
        utility_threshold=0.0, mode="joint", continuation_weight=100.0,
        posterior_weight=100.0,
    )
    assert abstained == -1


def test_candidate_scorer_preserves_frozen_p3_winner():
    scores = np.linspace(1.0, 0.93, 8, dtype=np.float32)
    indices = np.arange(8, dtype=np.int64)
    entry_rollout = np.arange(8, dtype=np.int32)
    entry_offset = np.ones(8, dtype=np.int32)
    token_buffer = np.arange(32, dtype=np.int32)
    rollout_offsets = np.arange(0, 32, 4, dtype=np.int64)
    rollout_lens = np.full(8, 4, dtype=np.int32)
    key_norms = np.ones(8, dtype=np.float32)
    current = np.asarray([0], dtype=np.int32)
    mean = np.zeros(18, dtype=np.float64)
    scale = np.ones(18, dtype=np.float64)
    theta = np.linspace(-0.1, 0.1, 18, dtype=np.float64)
    depth_bias = -np.arange(15, dtype=np.float64) * 0.1
    actions = np.asarray(HSPEC_SURVIVAL_ACTIONS, dtype=np.int16)
    costs = np.asarray([1, 1.05, 1.10, 1.20, 1.40, 1.70], dtype=np.float64)
    frozen = score_utility_one_prompt_python(
        scores, indices, entry_rollout, entry_offset, token_buffer,
        rollout_offsets, rollout_lens, key_norms, current, 1.0, 0, 1, 8,
        mean, scale, theta, depth_bias, actions, costs, 1.0, -1e30,
    )
    candidate = score_utility_candidates_one_prompt_python(
        scores, indices, entry_rollout, entry_offset, token_buffer,
        rollout_offsets, rollout_lens, key_norms, current, 1.0, 0, 1, 8,
        mean, scale, theta, depth_bias, actions, costs, 1.0,
    )
    slot = int(np.argmax(candidate[1]))
    assert frozen[0] == slot
    assert frozen[1] == int(candidate[0][slot])
    assert np.isclose(frozen[2], candidate[1][slot])
    assert np.isclose(frozen[3], candidate[2][slot])


def test_s14_config_is_hash_bound_and_fail_closed(tmp_path):
    s13 = tmp_path / "s13.json"
    s13_hash = _write(s13, {
        "schema_version": "hspec.s13.fastpath-robust-throughput-gate.v1",
        "status": "PASS",
        "decision": "S13_PASS_PATCH3A_FASTPATH_ADJUDICATED_ELIGIBLE_FOR_S14",
        "checks": {"sealed": True},
    })
    policy = tmp_path / "policy.json"
    policy_hash = _write(policy, {
        "schema_version": "hspec.s14.online-policy.v1",
        "candidate_scope": "existing_top8_only", "forced_candidate": False,
        "s13_authority_gate_sha256": s13_hash,
        "continuation_weight": 0.1, "posterior_weight": 0.2,
        "posterior_min_trials": 4,
    })
    replay = tmp_path / "replay.json"
    replay_hash = _write(replay, {
        "schema_version": "hspec.s14.sequential-gate.v1", "status": "PASS",
        "decision": "READY_FOR_S14_ONLINE_AB", "policy_sha256": policy_hash,
        "s13_authority_gate_sha256": s13_hash, "checks": {"parity": True},
    })
    env = {
        "HSPEC_S14_MODE": "joint",
        "HSPEC_S14_S13_GATE_PATH": str(s13),
        "HSPEC_S14_S13_GATE_SHA256": s13_hash,
        "HSPEC_S14_POLICY_PATH": str(policy),
        "HSPEC_S14_POLICY_SHA256": policy_hash,
        "HSPEC_S14_REPLAY_GATE_PATH": str(replay),
        "HSPEC_S14_REPLAY_GATE_SHA256": replay_hash,
        "HSPEC_S14_EXECUTION_LEVEL": "functional",
    }
    config = HSpecS14Config.from_environment(env)
    assert config.enabled and config.executes and config.mode == "joint"
    functional = tmp_path / "functional.json"
    functional_hash = _write(functional, {
        "schema_version": "hspec.s14.functional-gate.v1",
        "status": "PASS", "decision": "READY_FOR_S14_30B_AB",
        "policy_sha256": policy_hash, "replay_gate_sha256": replay_hash,
        "checks": {"functional": True},
    })
    env.update({
        "HSPEC_S14_EXECUTION_LEVEL": "performance",
        "HSPEC_S14_EXECUTION_GATE_PATH": str(functional),
        "HSPEC_S14_EXECUTION_GATE_SHA256": functional_hash,
    })
    performance = HSpecS14Config.from_environment(env)
    assert performance.enabled and performance.execution_level == "performance"
    env["HSPEC_S14_POLICY_SHA256"] = "0" * 64
    failed = HSpecS14Config.from_environment(env)
    assert not failed.enabled and "mismatch" in str(failed.fallback_reason)


def test_s14_trace_recorder_flushes_without_loss(tmp_path):
    recorder = HSpecS14TraceRecorder(tmp_path)
    assert recorder.record_many([{
        "event": "selection", "query_id": "q", "request_id": "r",
    }])
    recorder.flush("unit_test_flush")
    recorder.close("unit_test_close")
    traces = list(tmp_path.glob("*.jsonl"))
    statuses = list(tmp_path.glob("*.status.json"))
    assert len(traces) == len(statuses) == 1
    event = json.loads(traces[0].read_text(encoding="utf-8"))
    status = json.loads(statuses[0].read_text(encoding="utf-8"))
    assert event["schema_version"] == "hspec.s14.sequential-trace.v1"
    assert event["producer_sequence"] == 0
    assert status["enqueued_records"] == status["written_records"] == 1
    assert status["dropped_records"] == status["write_errors"] == 0
    assert status["closed"] and status["quiescent"]


def test_s14_request_sampler_preserves_trajectories_and_prompt_coverage():
    sampler = HSpecS14RequestSampler(prompt_every=1, request_every=32)
    for prompt_index in range(64):
        prompt = f"prompt-{prompt_index}"
        assert sampler.sampled(prompt, "first-request")
        first_decision = sampler.sampled(prompt, "later-request")
        assert sampler.sampled(prompt, "later-request") == first_decision

    dense = HSpecS14RequestSampler(prompt_every=1, request_every=1)
    assert dense.sampled("prompt", "request-a")
    assert dense.sampled("prompt", "request-b")
