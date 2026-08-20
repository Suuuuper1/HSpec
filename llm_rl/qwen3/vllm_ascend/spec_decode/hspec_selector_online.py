# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""S14 request-continuation and rollout-feedback policy.

The module is intentionally worker-local and dependency-free.  Artifact and
gate files are read only during proposer construction; the decode path uses
small numeric arrays and request-local dictionaries only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HSPEC_S14_MODES = frozenset({
    "off", "observe", "continuation", "posterior", "joint",
})
HSPEC_S14_POLICY_SCHEMA = "hspec.s14.online-policy.v1"
HSPEC_S14_REPLAY_GATE_SCHEMA = "hspec.s14.sequential-gate.v1"
HSPEC_S14_FUNCTIONAL_GATE_SCHEMA = "hspec.s14.functional-gate.v1"
HSPEC_S13_AUTHORITY_SCHEMA = "hspec.s13.fastpath-robust-throughput-gate.v1"
HSPEC_S13_AUTHORITY_DECISION = (
    "S13_PASS_PATCH3A_FASTPATH_ADJUDICATED_ELIGIBLE_FOR_S14"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_file(
    source: Mapping[str, str], path_name: str, hash_name: str
) -> tuple[Path, str, dict[str, Any]]:
    path = Path(str(source.get(path_name, "")).strip()).expanduser()
    if not str(path) or not path.is_file():
        raise ValueError(f"{path_name} is not a file")
    actual_hash = _sha256(path)
    expected_hash = str(source.get(hash_name, "")).strip().lower()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(f"{hash_name} mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path_name} must contain a JSON object")
    return path.resolve(), actual_hash, payload


@dataclass(frozen=True)
class HSpecS14Config:
    mode: str = "off"
    enabled: bool = False
    executes: bool = False
    continuation_weight: float = 0.0
    posterior_weight: float = 0.0
    posterior_min_trials: int = 4
    policy_path: str = ""
    policy_sha256: str = ""
    replay_gate_path: str = ""
    replay_gate_sha256: str = ""
    s13_gate_path: str = ""
    s13_gate_sha256: str = ""
    execution_level: str = ""
    execution_gate_path: str = ""
    execution_gate_sha256: str = ""
    fallback_reason: str | None = None

    @classmethod
    def disabled(cls, reason: str | None = None) -> "HSpecS14Config":
        return cls(fallback_reason=reason)

    @property
    def uses_continuation(self) -> bool:
        return self.mode in {"observe", "continuation", "joint"}

    @property
    def uses_posterior(self) -> bool:
        return self.mode in {"observe", "posterior", "joint"}

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "HSpecS14Config":
        source = os.environ if env is None else env
        mode = str(source.get("HSPEC_S14_MODE", "off")).strip().lower()
        if mode == "off":
            return cls.disabled()
        if mode not in HSPEC_S14_MODES:
            return cls.disabled(f"unsupported HSPEC_S14_MODE={mode!r}")
        try:
            s13_path, s13_hash, s13_gate = _validated_file(
                source, "HSPEC_S14_S13_GATE_PATH", "HSPEC_S14_S13_GATE_SHA256"
            )
            checks = s13_gate.get("checks", {})
            if not (
                s13_gate.get("schema_version") == HSPEC_S13_AUTHORITY_SCHEMA
                and s13_gate.get("status") == "PASS"
                and s13_gate.get("decision") == HSPEC_S13_AUTHORITY_DECISION
                and isinstance(checks, dict)
                and checks
                and all(bool(value) for value in checks.values())
            ):
                raise ValueError("S13 authority gate does not authorize S14")

            if mode == "observe":
                return cls(
                    mode=mode,
                    enabled=True,
                    executes=False,
                    s13_gate_path=str(s13_path),
                    s13_gate_sha256=s13_hash,
                )

            policy_path, policy_hash, policy = _validated_file(
                source, "HSPEC_S14_POLICY_PATH", "HSPEC_S14_POLICY_SHA256"
            )
            replay_path, replay_hash, replay_gate = _validated_file(
                source,
                "HSPEC_S14_REPLAY_GATE_PATH",
                "HSPEC_S14_REPLAY_GATE_SHA256",
            )
            if not (
                policy.get("schema_version") == HSPEC_S14_POLICY_SCHEMA
                and policy.get("candidate_scope") == "existing_top8_only"
                and policy.get("forced_candidate") is False
                and policy.get("s13_authority_gate_sha256") == s13_hash
            ):
                raise ValueError("invalid or unbound S14 policy")
            replay_checks = replay_gate.get("checks", {})
            if not (
                replay_gate.get("schema_version") == HSPEC_S14_REPLAY_GATE_SCHEMA
                and replay_gate.get("status") == "PASS"
                and replay_gate.get("decision") == "READY_FOR_S14_ONLINE_AB"
                and replay_gate.get("policy_sha256") == policy_hash
                and replay_gate.get("s13_authority_gate_sha256") == s13_hash
                and isinstance(replay_checks, dict)
                and replay_checks
                and all(bool(value) for value in replay_checks.values())
            ):
                raise ValueError("sequential replay gate does not authorize policy")

            continuation_weight = float(policy["continuation_weight"])
            posterior_weight = float(policy["posterior_weight"])
            min_trials = int(policy.get("posterior_min_trials", 4))
            if not (
                math.isfinite(continuation_weight)
                and continuation_weight >= 0.0
                and math.isfinite(posterior_weight)
                and posterior_weight >= 0.0
                and min_trials == 4
            ):
                raise ValueError("invalid S14 policy weights")

            level = str(source.get("HSPEC_S14_EXECUTION_LEVEL", "functional"))
            level = level.strip().lower()
            execution_path = ""
            execution_hash = ""
            if level not in {"functional", "performance"}:
                raise ValueError("S14 execution level must be functional/performance")
            if level == "performance":
                path, execution_hash, gate = _validated_file(
                    source,
                    "HSPEC_S14_EXECUTION_GATE_PATH",
                    "HSPEC_S14_EXECUTION_GATE_SHA256",
                )
                gate_checks = gate.get("checks", {})
                if not (
                    gate.get("schema_version") == HSPEC_S14_FUNCTIONAL_GATE_SCHEMA
                    and gate.get("status") == "PASS"
                    and gate.get("decision") == "READY_FOR_S14_30B_AB"
                    and gate.get("policy_sha256") == policy_hash
                    and gate.get("replay_gate_sha256") == replay_hash
                    and isinstance(gate_checks, dict)
                    and gate_checks
                    and all(bool(value) for value in gate_checks.values())
                ):
                    raise ValueError("functional gate does not authorize 30B S14")
                execution_path = str(path)

            return cls(
                mode=mode,
                enabled=True,
                executes=True,
                continuation_weight=continuation_weight,
                posterior_weight=posterior_weight,
                posterior_min_trials=min_trials,
                policy_path=str(policy_path),
                policy_sha256=policy_hash,
                replay_gate_path=str(replay_path),
                replay_gate_sha256=replay_hash,
                s13_gate_path=str(s13_path),
                s13_gate_sha256=s13_hash,
                execution_level=level,
                execution_gate_path=execution_path,
                execution_gate_sha256=execution_hash,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return cls.disabled(str(exc))


@dataclass(frozen=True)
class HSpecContinuationState:
    prompt_id: str
    table_version: int
    cache_generation: int
    rollout_idx: int
    source_matched_pos: int
    expected_matched_pos: int
    drafted_len: int
    accepted_len: int
    emitted_token_ids: tuple[int, ...]


def make_verified_continuation(
    *,
    prompt_id: str,
    table_version: int,
    cache_generation: int,
    rollout_idx: int,
    matched_pos: int,
    emitted_token_ids: Sequence[int],
    rollout_tokens: np.ndarray,
    drafted_len: int = 0,
    accepted_len: int = 0,
) -> HSpecContinuationState | None:
    """Return continuation state only after an exact emitted-segment match."""
    emitted = tuple(int(token) for token in emitted_token_ids)
    start = int(matched_pos) + 1
    end = start + len(emitted)
    if not emitted or start < 0 or end > int(rollout_tokens.shape[0]):
        return None
    for offset, token in enumerate(emitted):
        if int(rollout_tokens[start + offset]) != token:
            return None
    return HSpecContinuationState(
        prompt_id=str(prompt_id),
        table_version=int(table_version),
        cache_generation=int(cache_generation),
        rollout_idx=int(rollout_idx),
        source_matched_pos=int(matched_pos),
        expected_matched_pos=int(matched_pos) + len(emitted),
        drafted_len=max(int(drafted_len), 0),
        accepted_len=min(
            max(int(accepted_len), 0), max(int(drafted_len), 0)
        ),
        emitted_token_ids=emitted,
    )


def continuation_candidate_slot(
    state: HSpecContinuationState | None,
    *,
    prompt_id: str,
    table_version: int,
    cache_generation: int,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    entry_offset: np.ndarray,
) -> int:
    """Find the exact continuation position in the existing top-k only."""
    if state is None or (
        state.prompt_id != str(prompt_id)
        or state.table_version != int(table_version)
        or state.cache_generation != int(cache_generation)
    ):
        return -1
    for slot, raw_idx in enumerate(candidate_indices):
        idx = int(raw_idx)
        if idx < 0 or idx >= int(entry_rollout_idx.shape[0]):
            continue
        if (
            int(entry_rollout_idx[idx]) == state.rollout_idx
            and int(entry_offset[idx]) - 1 == state.expected_matched_pos
        ):
            return int(slot)
    return -1


def rollout_posterior_signal(
    trials: int,
    first_accepts: int,
    accepted_sum: int,
    drafted_sum: int,
    *,
    min_trials: int = 4,
) -> tuple[float, float, float, float]:
    """Return centered Beta(2,2) feedback and its small-sample reliability."""
    trials_i = max(int(trials), 0)
    first_i = min(max(int(first_accepts), 0), trials_i)
    drafted_i = max(int(drafted_sum), 0)
    accepted_i = min(max(int(accepted_sum), 0), drafted_i)
    p_first = (2.0 + float(first_i)) / (4.0 + float(trials_i))
    p_token = (2.0 + float(accepted_i)) / (4.0 + float(drafted_i))
    reliability = min(float(trials_i) / float(max(int(min_trials), 1)), 1.0)
    signal = reliability * (0.5 * (p_first - 0.5) + 0.5 * (p_token - 0.5))
    return signal, p_first, p_token, reliability


def select_s14_candidate(
    *,
    base_slot: int,
    utilities: np.ndarray,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    continuation_slot: int,
    rollout_trials: np.ndarray,
    rollout_first_accepts: np.ndarray,
    rollout_accept_sum: np.ndarray,
    rollout_draft_sum: np.ndarray,
    utility_threshold: float,
    mode: str,
    continuation_weight: float,
    posterior_weight: float,
    posterior_min_trials: int = 4,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Constrained top-8 rerank; never changes P3 abstention eligibility."""
    width = int(candidate_indices.shape[0])
    adjusted = np.full((width,), -math.inf, dtype=np.float64)
    posterior = np.zeros((width,), dtype=np.float64)
    if base_slot < 0 or base_slot >= width:
        return int(base_slot), adjusted, posterior
    uses_continuation = mode in {"observe", "continuation", "joint"}
    uses_posterior = mode in {"observe", "posterior", "joint"}
    best_slot = int(base_slot)
    best_score = -math.inf
    for slot in range(width):
        utility = float(utilities[slot])
        if not math.isfinite(utility) or utility <= float(utility_threshold):
            continue
        idx = int(candidate_indices[slot])
        if idx < 0 or idx >= int(entry_rollout_idx.shape[0]):
            continue
        rollout = int(entry_rollout_idx[idx])
        if rollout < 0 or rollout >= int(rollout_trials.shape[0]):
            continue
        signal, _, _, _ = rollout_posterior_signal(
            int(rollout_trials[rollout]),
            int(rollout_first_accepts[rollout]),
            int(rollout_accept_sum[rollout]),
            int(rollout_draft_sum[rollout]),
            min_trials=posterior_min_trials,
        )
        posterior[slot] = signal
        score = utility
        if uses_continuation and slot == int(continuation_slot):
            score += float(continuation_weight)
        if uses_posterior:
            score += float(posterior_weight) * signal
        adjusted[slot] = score
        if score > best_score:
            best_score = score
            best_slot = slot
    return int(best_slot), adjusted, posterior


def saturating_rollout_feedback_update(
    rollout_idx: int,
    accepted_len: int,
    drafted_len: int,
    trials: np.ndarray,
    first_accepts: np.ndarray,
    accept_sum: np.ndarray,
    draft_sum: np.ndarray,
) -> bool:
    """Apply one true-verification update to independent rollout aggregates."""
    rollout = int(rollout_idx)
    if rollout < 0 or rollout >= int(trials.shape[0]) or int(drafted_len) <= 0:
        return False
    accepted = min(max(int(accepted_len), 0), int(drafted_len))
    drafted = max(int(drafted_len), 0)
    trials[rollout] = min(int(trials[rollout]) + 1, np.iinfo(trials.dtype).max)
    if accepted > 0:
        first_accepts[rollout] = min(
            int(first_accepts[rollout]) + 1, np.iinfo(first_accepts.dtype).max
        )
    accept_sum[rollout] = min(
        int(accept_sum[rollout]) + accepted, np.iinfo(accept_sum.dtype).max
    )
    draft_sum[rollout] = min(
        int(draft_sum[rollout]) + drafted, np.iinfo(draft_sum.dtype).max
    )
    return True
