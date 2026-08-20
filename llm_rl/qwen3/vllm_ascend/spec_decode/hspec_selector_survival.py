# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Frozen S13 Patch 3A utility/survival selector.

The model and its independent promotion gate are read once when a worker is
constructed.  The decode path receives only contiguous numeric arrays and has
no filesystem, JSON, sklearn, or environment dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from numba import njit
    from numba.typed import List as NumbaList

    HSPEC_SURVIVAL_NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional on analysis hosts
    njit = None
    NumbaList = None
    HSPEC_SURVIVAL_NUMBA_AVAILABLE = False


HSPEC_SURVIVAL_MODE = "topk_utility"
HSPEC_SURVIVAL_SCHEMA = "hspec.survival-model.v1"
HSPEC_SURVIVAL_GATE_SCHEMA = "hspec.s12-to-s13.independent-gate.v1"
HSPEC_S13_ENTRY_GATE_SCHEMA = "hspec.s13.entry-gate.v1"
HSPEC_SURVIVAL_FEATURES = (
    "similarity_closeness",
    "suffix_cap_8",
    "cosine_score",
    "top1_top2_margin",
    "cosine_robust_z",
    "log_query_norm",
    "log_key_norm",
    "exact_position",
    "negative_log_abs_delta",
    "signed_forward",
    "retrieval_rank",
    "log_decoded_len",
    "remaining_fraction_15",
    "log_table_entries",
    "first_token_density",
    "prefix2_density",
    "prefix4_density",
    "same_rollout_density",
)
HSPEC_SURVIVAL_ACTIONS = (0, 1, 2, 4, 8, 15)
HSPEC_SURVIVAL_TOPK = 8
HSPEC_SURVIVAL_MAX_DEPTH = 15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean, got {value!r}")


def _finite_array(
    artifact: Mapping[str, Any], name: str, size: int
) -> np.ndarray:
    value = np.ascontiguousarray(artifact[name], dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class HSpecSurvivalConfig:
    """Init-only S13 configuration and validated numeric artifact."""

    enabled: bool = False
    shadow: bool = True
    allow_execute: bool = False
    model_path: str = ""
    model_sha256: str = ""
    gate_path: str = ""
    gate_sha256: str = ""
    execution_gate_path: str = ""
    execution_gate_sha256: str = ""
    execution_level: str = ""
    model_version: str = ""
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    theta: np.ndarray | None = None
    depth_bias: np.ndarray | None = None
    actions: np.ndarray | None = None
    costs_ms: np.ndarray | None = None
    temperature: float = 1.0
    utility_threshold: float = 0.0
    selector_extra_p95_ms: float = 0.0
    fallback_reason: str | None = None

    @property
    def executes_utility(self) -> bool:
        return bool(self.enabled and not self.shadow and self.allow_execute)

    @classmethod
    def disabled(cls, reason: str | None = None) -> "HSpecSurvivalConfig":
        return cls(fallback_reason=reason)

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "HSpecSurvivalConfig":
        source = os.environ if env is None else env
        mode = str(source.get("HSPEC_SELECT_MODE", "hardmax")).strip().lower()
        if mode != HSPEC_SURVIVAL_MODE:
            return cls.disabled()
        try:
            shadow = _parse_bool(source.get("HSPEC_SELECT_SHADOW", "1"))
            allow_execute = _parse_bool(
                source.get("HSPEC_SELECT_ALLOW_EXECUTE", "0")
            )
            if not shadow and not allow_execute:
                raise ValueError(
                    "topk_utility execute requires HSPEC_SELECT_ALLOW_EXECUTE=1"
                )
            model_path = Path(
                str(source.get("HSPEC_SELECT_MODEL_PATH", "")).strip()
            ).expanduser()
            gate_path = Path(
                str(source.get("HSPEC_SELECT_PROMOTION_GATE_PATH", "")).strip()
            ).expanduser()
            if not str(model_path) or not model_path.is_file():
                raise ValueError("HSPEC_SELECT_MODEL_PATH is not a file")
            if not str(gate_path) or not gate_path.is_file():
                raise ValueError("HSPEC_SELECT_PROMOTION_GATE_PATH is not a file")

            actual_model_hash = _sha256(model_path)
            expected_model_hash = str(
                source.get("HSPEC_SELECT_MODEL_SHA256", "")
            ).strip().lower()
            if not expected_model_hash or actual_model_hash != expected_model_hash:
                raise ValueError("survival model SHA-256 mismatch")
            actual_gate_hash = _sha256(gate_path)
            expected_gate_hash = str(
                source.get("HSPEC_SELECT_PROMOTION_GATE_SHA256", "")
            ).strip().lower()
            if not expected_gate_hash or actual_gate_hash != expected_gate_hash:
                raise ValueError("promotion gate SHA-256 mismatch")

            artifact = json.loads(model_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if artifact.get("schema_version") != HSPEC_SURVIVAL_SCHEMA:
                raise ValueError("unsupported survival model schema")
            if gate.get("schema_version") not in {
                HSPEC_SURVIVAL_GATE_SCHEMA,
                HSPEC_S13_ENTRY_GATE_SCHEMA,
            }:
                raise ValueError("unsupported promotion gate schema")
            gate_checks = gate.get("checks", {})
            if not (
                gate.get("status") == "PASS"
                and gate.get("decision") == "READY_FOR_S13_PATCH3A"
                and gate.get("ready_for_s13") is True
                and gate.get("promotion_allowed") is True
                and gate.get("candidate_status") == "SHADOW_CANDIDATE"
                and isinstance(gate_checks, dict)
                and bool(gate_checks)
                and all(bool(value) for value in gate_checks.values())
            ):
                raise ValueError("promotion gate does not authorize S13 Patch 3A")
            if gate.get("schema_version") == HSPEC_S13_ENTRY_GATE_SCHEMA and not (
                gate.get("independent_gate_status") == "PASS"
                and gate.get("independent_gate_decision")
                == "READY_FOR_S13_PATCH3A"
                and len(str(gate.get("independent_gate_sha256", ""))) == 64
            ):
                raise ValueError("S13 entry gate lacks the independent PASS binding")
            if str(gate.get("candidate_model_sha256", "")) != actual_model_hash:
                raise ValueError("promotion gate is bound to a different model")

            execution_gate_path = ""
            execution_gate_hash = ""
            execution_level = ""
            if not shadow:
                execution_level = str(
                    source.get("HSPEC_SELECT_EXECUTION_LEVEL", "")
                ).strip().lower()
                expected_execution = {
                    "functional": (
                        "hspec.s13.functional-shadow-gate.v2",
                        "READY_FOR_1P5B_EXECUTION_SMOKE",
                    ),
                    "performance": (
                        "hspec.s13.target-shadow-gate.v2",
                        "READY_FOR_30B_ONLINE_AB",
                    ),
                }.get(execution_level)
                if expected_execution is None:
                    raise ValueError(
                        "execute requires functional/performance execution level"
                    )
                execution_path = Path(str(
                    source.get("HSPEC_SELECT_EXECUTION_GATE_PATH", "")
                ).strip()).expanduser()
                if not str(execution_path) or not execution_path.is_file():
                    raise ValueError("execution gate is not a file")
                execution_gate_hash = _sha256(execution_path)
                expected_execution_hash = str(
                    source.get("HSPEC_SELECT_EXECUTION_GATE_SHA256", "")
                ).strip().lower()
                if (
                    not expected_execution_hash
                    or execution_gate_hash != expected_execution_hash
                ):
                    raise ValueError("execution gate SHA-256 mismatch")
                execution_gate = json.loads(
                    execution_path.read_text(encoding="utf-8")
                )
                if not (
                    execution_gate.get("schema_version") == expected_execution[0]
                    and execution_gate.get("status") == "PASS"
                    and execution_gate.get("decision") == expected_execution[1]
                    and isinstance(execution_gate.get("checks"), dict)
                    and bool(execution_gate["checks"])
                    and all(bool(value) for value in execution_gate["checks"].values())
                    and execution_gate.get("model_sha256") == actual_model_hash
                    and execution_gate.get("entry_gate_sha256") == actual_gate_hash
                ):
                    raise ValueError(
                        "execution gate does not authorize this S13 online level"
                    )
                execution_gate_path = str(execution_path.resolve())
            if tuple(artifact.get("feature_names", ())) != HSPEC_SURVIVAL_FEATURES:
                raise ValueError("survival feature order differs from frozen S13 order")
            if artifact.get("candidate_scope") != "full_top8":
                raise ValueError("S13 Patch 3A requires full_top8 candidate scope")
            if tuple(int(value) for value in artifact.get("length_actions", ())) != (
                HSPEC_SURVIVAL_ACTIONS
            ):
                raise ValueError("survival action set differs from the frozen action set")
            if artifact.get("quality_schema_used") is not False:
                raise ValueError("S13 Patch 3A forbids schema-v2 quality features")

            model_version = str(artifact.get("model_version", ""))
            expected_version = str(
                source.get("HSPEC_SELECT_MODEL_VERSION", model_version)
            ).strip()
            if not model_version or expected_version != model_version:
                raise ValueError("survival model version mismatch")
            mean = _finite_array(artifact, "feature_mean", len(HSPEC_SURVIVAL_FEATURES))
            scale = _finite_array(artifact, "feature_scale", len(HSPEC_SURVIVAL_FEATURES))
            if np.any(scale <= 0.0):
                raise ValueError("feature_scale must be positive")
            theta = _finite_array(artifact, "theta", len(HSPEC_SURVIVAL_FEATURES))
            bias = _finite_array(artifact, "depth_bias", HSPEC_SURVIVAL_MAX_DEPTH)
            if np.any(bias[1:] > bias[:-1] + 1.0e-12):
                raise ValueError("depth_bias must be monotone non-increasing")
            temperature = float(artifact["temperature"])
            threshold = float(artifact["utility_threshold"])
            selector_extra = float(artifact["selector_extra_p95_ms"])
            if not (
                math.isfinite(temperature)
                and temperature > 0.0
                and math.isfinite(threshold)
                and threshold >= 0.0
                and math.isfinite(selector_extra)
                and selector_extra >= 0.0
            ):
                raise ValueError("invalid temperature/threshold/selector cost")
            action_values = np.asarray(HSPEC_SURVIVAL_ACTIONS, dtype=np.int16)
            costs_map = artifact.get("costs_ms", {})
            costs = np.asarray(
                [float(costs_map[str(action)]) for action in HSPEC_SURVIVAL_ACTIONS],
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(costs))
                or costs[0] <= 0.0
                or np.any(np.diff(costs) < 0.0)
            ):
                raise ValueError(
                    "action costs require a positive baseline and monotone values"
                )
            action_values.setflags(write=False)
            costs.setflags(write=False)
            return cls(
                enabled=True,
                shadow=shadow,
                allow_execute=allow_execute,
                model_path=str(model_path.resolve()),
                model_sha256=actual_model_hash,
                gate_path=str(gate_path.resolve()),
                gate_sha256=actual_gate_hash,
                execution_gate_path=execution_gate_path,
                execution_gate_sha256=execution_gate_hash,
                execution_level=execution_level,
                model_version=model_version,
                feature_mean=mean,
                feature_scale=scale,
                theta=theta,
                depth_bias=bias,
                actions=action_values,
                costs_ms=costs,
                temperature=temperature,
                utility_threshold=threshold,
                selector_extra_p95_ms=selector_extra,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return cls.disabled(str(exc))


def _extract_features_python_impl(
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    entry_offset: np.ndarray,
    token_buffer: np.ndarray,
    rollout_offsets: np.ndarray,
    rollout_lens: np.ndarray,
    key_norms: np.ndarray,
    current_tokens: np.ndarray,
    query_norm: float,
    base_pos: int,
    decoded_len: int,
    n_entries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract the exact frozen 18-feature full-top8 block."""
    width = int(candidate_scores.shape[0])
    features = np.empty((width, len(HSPEC_SURVIVAL_FEATURES)), dtype=np.float64)
    suffixes = np.zeros((width,), dtype=np.int16)
    abs_deltas = np.zeros((width,), dtype=np.int32)
    remaining = np.zeros((width,), dtype=np.int32)
    if width != HSPEC_SURVIVAL_TOPK or candidate_indices.shape[0] != width:
        return features, suffixes, abs_deltas, remaining, 1
    if n_entries < width or rollout_offsets.shape[0] != rollout_lens.shape[0]:
        return features, suffixes, abs_deltas, remaining, 2
    if not math.isfinite(float(query_norm)) or float(query_norm) < 0.0:
        return features, suffixes, abs_deltas, remaining, 5

    scores = np.empty((width,), dtype=np.float64)
    rollouts = np.empty((width,), dtype=np.int32)
    offsets = np.empty((width,), dtype=np.int32)
    prefix_lens = np.empty((width,), dtype=np.int16)
    prefix_tokens = np.zeros((width, 4), dtype=np.int32)
    for slot in range(width):
        idx = int(candidate_indices[slot])
        score = float(candidate_scores[slot])
        if (
            idx < 0
            or idx >= int(n_entries)
            or idx >= int(entry_rollout_idx.shape[0])
            or idx >= int(entry_offset.shape[0])
            or idx >= int(key_norms.shape[0])
            or not math.isfinite(score)
        ):
            return features, suffixes, abs_deltas, remaining, 2
        rollout = int(entry_rollout_idx[idx])
        if rollout < 0 or rollout >= int(rollout_lens.shape[0]):
            return features, suffixes, abs_deltas, remaining, 3
        offset = int(entry_offset[idx])
        rollout_len = int(rollout_lens[rollout])
        base = int(rollout_offsets[rollout])
        remain = rollout_len - offset
        key_norm = float(key_norms[idx])
        if (
            offset <= 0
            or remain <= 0
            or base < 0
            or rollout_len < 0
            or base + rollout_len > int(token_buffer.shape[0])
        ):
            return features, suffixes, abs_deltas, remaining, 4
        if not math.isfinite(key_norm) or key_norm < 0.0:
            return features, suffixes, abs_deltas, remaining, 5
        scores[slot] = score
        rollouts[slot] = rollout
        offsets[slot] = offset
        remaining[slot] = remain
        take = min(remain, 4)
        prefix_lens[slot] = take
        for depth in range(take):
            prefix_tokens[slot, depth] = int(token_buffer[base + offset + depth])

    sorted_scores = scores.copy()
    for left in range(1, width):
        value = sorted_scores[left]
        cursor = left - 1
        while cursor >= 0 and sorted_scores[cursor] > value:
            sorted_scores[cursor + 1] = sorted_scores[cursor]
            cursor -= 1
        sorted_scores[cursor + 1] = value
    median = 0.5 * (sorted_scores[width // 2 - 1] + sorted_scores[width // 2])
    deviations = np.empty((width,), dtype=np.float64)
    for slot in range(width):
        deviations[slot] = abs(scores[slot] - median)
    for left in range(1, width):
        value = deviations[left]
        cursor = left - 1
        while cursor >= 0 and deviations[cursor] > value:
            deviations[cursor + 1] = deviations[cursor]
            cursor -= 1
        deviations[cursor + 1] = value
    mad = 0.5 * (deviations[width // 2 - 1] + deviations[width // 2])
    mad = max(mad, 1.0e-6)
    denominator = max(abs(scores[0]), 1.0e-12)
    margin = max(scores[0] - scores[1], 0.0)
    current_end = int(current_tokens.shape[0]) - 1

    for slot in range(width):
        idx = int(candidate_indices[slot])
        rollout = int(rollouts[slot])
        offset = int(offsets[slot])
        base = int(rollout_offsets[rollout])
        suffix = 0
        suffix_limit = min(8, current_end + 1, offset)
        while suffix < suffix_limit:
            if int(current_tokens[current_end - suffix]) != int(
                token_buffer[base + offset - 1 - suffix]
            ):
                break
            suffix += 1
        suffixes[slot] = suffix
        signed_delta = int(offset - 1 - int(base_pos))
        abs_delta = abs(signed_delta)
        abs_deltas[slot] = abs_delta
        relative_drop = max((scores[0] - scores[slot]) / denominator, 0.0)
        robust_z = (scores[slot] - median) / mad
        robust_z = min(max(robust_z, -12.0), 12.0)

        density1 = 0
        density2 = 0
        density4 = 0
        rollout_density = 0
        for other in range(width):
            if int(rollouts[other]) == rollout:
                rollout_density += 1
            if prefix_lens[slot] >= 1 and prefix_lens[other] >= 1:
                if prefix_tokens[slot, 0] == prefix_tokens[other, 0]:
                    density1 += 1
                    if prefix_lens[slot] >= 2 and prefix_lens[other] >= 2:
                        if prefix_tokens[slot, 1] == prefix_tokens[other, 1]:
                            density2 += 1
                            if prefix_lens[slot] >= 4 and prefix_lens[other] >= 4:
                                if (
                                    prefix_tokens[slot, 2] == prefix_tokens[other, 2]
                                    and prefix_tokens[slot, 3] == prefix_tokens[other, 3]
                                ):
                                    density4 += 1

        features[slot, 0] = -relative_drop
        features[slot, 1] = min(float(suffix), 8.0) / 8.0
        features[slot, 2] = scores[slot]
        features[slot, 3] = margin
        features[slot, 4] = robust_z
        features[slot, 5] = math.log1p(max(float(query_norm), 0.0))
        features[slot, 6] = math.log1p(max(float(key_norms[idx]), 0.0))
        features[slot, 7] = 1.0 if abs_delta == 0 else 0.0
        features[slot, 8] = -math.log1p(abs_delta)
        features[slot, 9] = 1.0 if signed_delta >= 0 else 0.0
        features[slot, 10] = float(slot) / float(max(width - 1, 1))
        features[slot, 11] = math.log1p(max(float(decoded_len), 0.0))
        features[slot, 12] = min(float(remaining[slot]), 15.0) / 15.0
        features[slot, 13] = math.log1p(max(float(n_entries), 0.0))
        features[slot, 14] = float(density1) / float(width)
        features[slot, 15] = float(density2) / float(width)
        features[slot, 16] = float(density4) / float(width)
        features[slot, 17] = float(rollout_density) / float(width)
    return features, suffixes, abs_deltas, remaining, 0


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


if HSPEC_SURVIVAL_NUMBA_AVAILABLE:
    extract_utility_features_numba = njit(cache=True, nogil=True)(
        _extract_features_python_impl
    )
    _sigmoid_for_scorer = njit(cache=True, nogil=True)(_sigmoid)
else:  # pragma: no cover
    extract_utility_features_numba = None
    _sigmoid_for_scorer = _sigmoid


def _extract_features_for_scorer(*args):
    return _extract_features_python_impl(*args)


if HSPEC_SURVIVAL_NUMBA_AVAILABLE:
    _extract_features_for_scorer = extract_utility_features_numba


def _score_utility_one_prompt_python_impl(
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    entry_offset: np.ndarray,
    token_buffer: np.ndarray,
    rollout_offsets: np.ndarray,
    rollout_lens: np.ndarray,
    key_norms: np.ndarray,
    current_tokens: np.ndarray,
    query_norm: float,
    base_pos: int,
    decoded_len: int,
    n_entries: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    theta: np.ndarray,
    depth_bias: np.ndarray,
    actions: np.ndarray,
    costs_ms: np.ndarray,
    temperature: float,
    utility_threshold: float,
) -> tuple[int, int, float, float, int, int, int, int]:
    """Return slot, action, utility, P1, suffix, abs-delta, remaining, status.

    Status zero is a valid decision (including abstain). Non-zero means that
    the row is outside the frozen full-top8 contract and must execute R1.
    """
    features, suffixes, abs_deltas, remaining, status = _extract_features_for_scorer(
        candidate_scores,
        candidate_indices,
        entry_rollout_idx,
        entry_offset,
        token_buffer,
        rollout_offsets,
        rollout_lens,
        key_norms,
        current_tokens,
        query_norm,
        base_pos,
        decoded_len,
        n_entries,
    )
    if status != 0:
        return -2, 0, -math.inf, 0.0, 0, 0, 0, status
    base_cost = float(costs_ms[0])
    best_slot = -1
    best_action = 0
    best_utility = -math.inf
    best_p1 = 0.0
    width = int(candidate_scores.shape[0])
    for slot in range(width):
        base_logit = 0.0
        for feature in range(features.shape[1]):
            standardized = (
                float(features[slot, feature]) - float(feature_mean[feature])
            ) / float(feature_scale[feature])
            standardized = min(max(standardized, -12.0), 12.0)
            base_logit += standardized * float(theta[feature])
        survival = np.empty((depth_bias.shape[0],), dtype=np.float64)
        usable_depth = min(int(remaining[slot]), int(depth_bias.shape[0]))
        for depth in range(int(depth_bias.shape[0])):
            if depth < usable_depth:
                survival[depth] = _sigmoid_for_scorer(
                    (base_logit + float(depth_bias[depth])) / float(temperature)
                )
            else:
                survival[depth] = 0.0
        for action_index in range(1, int(actions.shape[0])):
            action = int(actions[action_index])
            usable = min(action, int(remaining[slot]), int(depth_bias.shape[0]))
            expected = 0.0
            for depth in range(usable):
                expected += survival[depth]
            penalty = (float(costs_ms[action_index]) - base_cost) / base_cost
            utility = expected - penalty
            if utility > best_utility:
                best_slot = slot
                best_action = action
                best_utility = utility
                best_p1 = float(survival[0]) if usable_depth > 0 else 0.0
    if best_utility <= float(utility_threshold):
        return -1, 0, best_utility, best_p1, 0, 0, 0, 0
    return (
        best_slot,
        best_action,
        best_utility,
        best_p1,
        int(suffixes[best_slot]),
        int(abs_deltas[best_slot]),
        int(remaining[best_slot]),
        0,
    )


extract_utility_features_python = _extract_features_python_impl
score_utility_one_prompt_python = _score_utility_one_prompt_python_impl


def _score_utility_candidates_one_prompt_python_impl(
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    entry_offset: np.ndarray,
    token_buffer: np.ndarray,
    rollout_offsets: np.ndarray,
    rollout_lens: np.ndarray,
    key_norms: np.ndarray,
    current_tokens: np.ndarray,
    query_norm: float,
    base_pos: int,
    decoded_len: int,
    n_entries: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    theta: np.ndarray,
    depth_bias: np.ndarray,
    actions: np.ndarray,
    costs_ms: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Score every frozen top-8 candidate for S14 constrained reranking.

    This is deliberately separate from the promoted S13 selector.  S13 keeps
    its exact single-winner ABI and behavior; S14 may inspect these per-slot
    values only after S13 has produced a valid, non-abstaining decision.
    """
    features, suffixes, abs_deltas, remaining, status = _extract_features_for_scorer(
        candidate_scores,
        candidate_indices,
        entry_rollout_idx,
        entry_offset,
        token_buffer,
        rollout_offsets,
        rollout_lens,
        key_norms,
        current_tokens,
        query_norm,
        base_pos,
        decoded_len,
        n_entries,
    )
    width = int(candidate_scores.shape[0])
    selected_actions = np.zeros((width,), dtype=np.int16)
    utilities = np.full((width,), -math.inf, dtype=np.float64)
    probabilities1 = np.zeros((width,), dtype=np.float64)
    if status != 0:
        return selected_actions, utilities, probabilities1, suffixes, remaining, status

    base_cost = float(costs_ms[0])
    for slot in range(width):
        base_logit = 0.0
        for feature in range(features.shape[1]):
            standardized = (
                float(features[slot, feature]) - float(feature_mean[feature])
            ) / float(feature_scale[feature])
            standardized = min(max(standardized, -12.0), 12.0)
            base_logit += standardized * float(theta[feature])
        usable_depth = min(int(remaining[slot]), int(depth_bias.shape[0]))
        survival = np.zeros((depth_bias.shape[0],), dtype=np.float64)
        for depth in range(usable_depth):
            survival[depth] = _sigmoid_for_scorer(
                (base_logit + float(depth_bias[depth])) / float(temperature)
            )
        probabilities1[slot] = survival[0] if usable_depth > 0 else 0.0
        best_utility = -math.inf
        best_action = 0
        for action_index in range(1, int(actions.shape[0])):
            action = int(actions[action_index])
            usable = min(action, usable_depth)
            expected = 0.0
            for depth in range(usable):
                expected += survival[depth]
            penalty = (float(costs_ms[action_index]) - base_cost) / base_cost
            utility = expected - penalty
            if utility > best_utility:
                best_utility = utility
                best_action = action
        selected_actions[slot] = best_action
        utilities[slot] = best_utility
    return selected_actions, utilities, probabilities1, suffixes, remaining, 0


score_utility_candidates_one_prompt_python = (
    _score_utility_candidates_one_prompt_python_impl
)

if HSPEC_SURVIVAL_NUMBA_AVAILABLE:
    score_utility_one_prompt_numba = njit(cache=True, nogil=True, fastmath=False)(
        _score_utility_one_prompt_python_impl
    )
    score_utility_candidates_one_prompt_numba = njit(
        cache=True, nogil=True, fastmath=False
    )(_score_utility_candidates_one_prompt_python_impl)
else:  # pragma: no cover
    extract_utility_features_numba = None
    score_utility_one_prompt_numba = None
    score_utility_candidates_one_prompt_numba = None


def _score_utility_batch_numba_impl(
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    table_rows: np.ndarray,
    entry_rollout_idx_list,
    entry_offset_list,
    token_buffer_list,
    rollout_offsets_list,
    rollout_lens_list,
    key_norms_list,
    current_tails: np.ndarray,
    current_tail_lens: np.ndarray,
    query_norms: np.ndarray,
    base_positions: np.ndarray,
    decoded_lens: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    theta: np.ndarray,
    depth_bias: np.ndarray,
    actions: np.ndarray,
    costs_ms: np.ndarray,
    temperature: float,
    utility_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Score one selection window with one dispatcher crossing.

    Prompt tables have different extents, so their compact CPU arrays are
    supplied as typed lists built with the prefetch batch cache. Only the
    bounded response tail is copied into the dense per-window input.
    """
    rows = int(candidate_scores.shape[0])
    slots = np.empty((rows,), dtype=np.int16)
    selected_actions = np.empty((rows,), dtype=np.int16)
    utilities = np.empty((rows,), dtype=np.float64)
    probabilities1 = np.empty((rows,), dtype=np.float64)
    suffixes = np.empty((rows,), dtype=np.int16)
    abs_deltas = np.empty((rows,), dtype=np.int32)
    remaining = np.empty((rows,), dtype=np.int32)
    statuses = np.empty((rows,), dtype=np.int16)
    tail_width = int(current_tails.shape[1])
    for row in range(rows):
        table_row = int(table_rows[row])
        tail_len = int(current_tail_lens[row])
        if tail_len < 0:
            tail_len = 0
        elif tail_len > tail_width:
            tail_len = tail_width
        result = score_utility_one_prompt_numba(
            candidate_scores[row],
            candidate_indices[row],
            entry_rollout_idx_list[table_row],
            entry_offset_list[table_row],
            token_buffer_list[table_row],
            rollout_offsets_list[table_row],
            rollout_lens_list[table_row],
            key_norms_list[table_row],
            current_tails[row, tail_width - tail_len :],
            float(query_norms[row]),
            int(base_positions[row]),
            int(decoded_lens[row]),
            int(entry_rollout_idx_list[table_row].shape[0]),
            feature_mean,
            feature_scale,
            theta,
            depth_bias,
            actions,
            costs_ms,
            float(temperature),
            float(utility_threshold),
        )
        slots[row] = int(result[0])
        selected_actions[row] = int(result[1])
        utilities[row] = float(result[2])
        probabilities1[row] = float(result[3])
        suffixes[row] = int(result[4])
        abs_deltas[row] = int(result[5])
        remaining[row] = int(result[6])
        statuses[row] = int(result[7])
    return (
        slots,
        selected_actions,
        utilities,
        probabilities1,
        suffixes,
        abs_deltas,
        remaining,
        statuses,
    )


if HSPEC_SURVIVAL_NUMBA_AVAILABLE:
    score_utility_batch_numba = njit(cache=True, nogil=True, fastmath=False)(
        _score_utility_batch_numba_impl
    )
else:  # pragma: no cover
    score_utility_batch_numba = None


def warm_survival_selector() -> None:
    """Compile the production signature before the first decode call."""
    if score_utility_one_prompt_numba is None:
        return
    scores = np.linspace(1.0, 0.99, 8, dtype=np.float32)
    indices = np.arange(8, dtype=np.int64)
    rollout_idx = np.arange(8, dtype=np.int32)
    entry_offset = np.ones((8,), dtype=np.int32)
    token_buffer = np.arange(24, dtype=np.int32)
    rollout_offsets = np.arange(0, 24, 3, dtype=np.int64)
    rollout_lens = np.full((8,), 3, dtype=np.int32)
    key_norms = np.ones((8,), dtype=np.float32)
    current = np.asarray([0], dtype=np.int32)
    mean = np.zeros((18,), dtype=np.float64)
    scale = np.ones((18,), dtype=np.float64)
    theta = np.zeros((18,), dtype=np.float64)
    bias = -np.arange(15, dtype=np.float64)
    actions = np.asarray(HSPEC_SURVIVAL_ACTIONS, dtype=np.int16)
    costs = np.arange(6, dtype=np.float64) + 1.0
    score_utility_one_prompt_numba(
        scores, indices, rollout_idx, entry_offset, token_buffer,
        rollout_offsets, rollout_lens, key_norms, current, 1.0, 0, 1, 8,
        mean, scale, theta, bias, actions, costs, 1.0, 0.0,
    )
    score_utility_candidates_one_prompt_numba(
        scores, indices, rollout_idx, entry_offset, token_buffer,
        rollout_offsets, rollout_lens, key_norms, current, 1.0, 0, 1, 8,
        mean, scale, theta, bias, actions, costs, 1.0,
    )
    if score_utility_batch_numba is not None and NumbaList is not None:
        def typed(values):
            result = NumbaList()
            result.append(values)
            return result

        score_utility_batch_numba(
            scores[None, :],
            indices[None, :],
            np.zeros((1,), dtype=np.int32),
            typed(rollout_idx),
            typed(entry_offset),
            typed(token_buffer),
            typed(rollout_offsets),
            typed(rollout_lens),
            typed(key_norms),
            current[None, :],
            np.ones((1,), dtype=np.int16),
            np.ones((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.int32),
            np.ones((1,), dtype=np.int32),
            mean,
            scale,
            theta,
            bias,
            actions,
            costs,
            1.0,
            0.0,
        )
