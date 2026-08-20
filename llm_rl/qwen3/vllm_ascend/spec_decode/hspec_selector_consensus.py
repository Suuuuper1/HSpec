# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fixed-width prefix consensus for HSpec S15.

The implementation deliberately uses only primitive contiguous arrays and
linear scans over K<=16.  It never constructs a trie or a token hash map in
the decode path.  Every emitted C2 prefix retains at least one source
candidate, which makes cross-trajectory token splicing impossible.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    from numba.extending import register_jitable

    HSPEC_CONSENSUS_NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    njit = None
    HSPEC_CONSENSUS_NUMBA_AVAILABLE = False


HSPEC_CONSENSUS_MODE_C1 = 1
HSPEC_CONSENSUS_MODE_C2 = 2
HSPEC_CONSENSUS_WEIGHT_UTILITY = 0
HSPEC_CONSENSUS_WEIGHT_P1 = 1
HSPEC_CONSENSUS_STOP_VALUE_END = 0
HSPEC_CONSENSUS_STOP_DISAGREEMENT = 1
HSPEC_CONSENSUS_STOP_INVALID = 2
HSPEC_CONSENSUS_STOP_C1_FOLLOW = 3


def _candidate_weights_python_impl(
    utilities: np.ndarray,
    probabilities1: np.ndarray,
    candidate_rollouts: np.ndarray,
    candidate_lens: np.ndarray,
    utility_threshold: float,
    weight_source: int,
    temperature: float,
    rollout_normalize: bool,
) -> np.ndarray:
    width = int(utilities.shape[0])
    weights = np.zeros((width,), dtype=np.float64)
    max_utility = -math.inf
    for slot in range(width):
        utility = float(utilities[slot])
        if (
            int(candidate_lens[slot]) > 0
            and math.isfinite(utility)
            and utility > float(utility_threshold)
            and utility > max_utility
        ):
            max_utility = utility
    if not math.isfinite(max_utility):
        return weights
    safe_temperature = max(float(temperature), 1e-6)
    for slot in range(width):
        utility = float(utilities[slot])
        if (
            int(candidate_lens[slot]) <= 0
            or not math.isfinite(utility)
            or utility <= float(utility_threshold)
        ):
            continue
        if int(weight_source) == HSPEC_CONSENSUS_WEIGHT_P1:
            weights[slot] = max(float(probabilities1[slot]), 0.0)
        else:
            log_weight = (utility - max_utility) / safe_temperature
            weights[slot] = math.exp(max(log_weight, -40.0))
    if rollout_normalize:
        for slot in range(width):
            if weights[slot] <= 0.0:
                continue
            multiplicity = 0
            rollout = int(candidate_rollouts[slot])
            for other in range(width):
                if weights[other] > 0.0 and int(candidate_rollouts[other]) == rollout:
                    multiplicity += 1
            if multiplicity > 1:
                weights[slot] /= float(multiplicity)
    return weights


def _prefix_consensus_python_impl(
    utilities: np.ndarray,
    probabilities1: np.ndarray,
    candidate_rollouts: np.ndarray,
    candidate_tokens: np.ndarray,
    candidate_lens: np.ndarray,
    utility_threshold: float,
    weight_source: int,
    temperature: float,
    rollout_normalize: bool,
    mode: int,
    minimum_confidence: float,
    max_output_tokens: int,
) -> tuple[
    np.ndarray,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return draft, length, source slot, stop code and depth evidence."""
    width = int(candidate_tokens.shape[0])
    depth_limit = min(
        max(int(max_output_tokens), 0), int(candidate_tokens.shape[1])
    )
    output = np.zeros((int(candidate_tokens.shape[1]),), dtype=np.int32)
    confidences = np.zeros((int(candidate_tokens.shape[1]),), dtype=np.float64)
    alive_masses = np.zeros((int(candidate_tokens.shape[1]),), dtype=np.float64)
    alive_counts = np.zeros((int(candidate_tokens.shape[1]),), dtype=np.int16)
    weights = _candidate_weights_python_impl(
        utilities,
        probabilities1,
        candidate_rollouts,
        candidate_lens,
        utility_threshold,
        weight_source,
        temperature,
        rollout_normalize,
    )
    total_weight = 0.0
    for slot in range(width):
        total_weight += float(weights[slot])
    if depth_limit <= 0 or total_weight <= 0.0:
        return (
            output, 0, -1, HSPEC_CONSENSUS_STOP_INVALID,
            confidences, alive_masses, alive_counts,
        )

    alive = np.zeros((width,), dtype=np.uint8)
    for slot in range(width):
        alive[slot] = 1 if weights[slot] > 0.0 else 0

    if int(mode) == HSPEC_CONSENSUS_MODE_C1:
        unique_tokens = np.zeros((width,), dtype=np.int32)
        token_masses = np.zeros((width,), dtype=np.float64)
        unique_count = 0
        for slot in range(width):
            if alive[slot] == 0 or int(candidate_lens[slot]) <= 0:
                continue
            token = int(candidate_tokens[slot, 0])
            token_index = -1
            for index in range(unique_count):
                if int(unique_tokens[index]) == token:
                    token_index = index
                    break
            if token_index < 0:
                token_index = unique_count
                unique_tokens[unique_count] = token
                unique_count += 1
            token_masses[token_index] += float(weights[slot])
        if unique_count <= 0:
            return (
                output, 0, -1, HSPEC_CONSENSUS_STOP_INVALID,
                confidences, alive_masses, alive_counts,
            )
        best_token_index = 0
        for index in range(1, unique_count):
            if token_masses[index] > token_masses[best_token_index]:
                best_token_index = index
        best_token = int(unique_tokens[best_token_index])
        best_slot = -1
        best_utility = -math.inf
        for slot in range(width):
            if (
                alive[slot] != 0
                and int(candidate_lens[slot]) > 0
                and int(candidate_tokens[slot, 0]) == best_token
                and float(utilities[slot]) > best_utility
            ):
                best_slot = slot
                best_utility = float(utilities[slot])
        if best_slot < 0:
            return (
                output, 0, -1, HSPEC_CONSENSUS_STOP_INVALID,
                confidences, alive_masses, alive_counts,
            )
        output_len = min(int(candidate_lens[best_slot]), depth_limit)
        for depth in range(output_len):
            output[depth] = int(candidate_tokens[best_slot, depth])
        confidences[0] = float(token_masses[best_token_index]) / total_weight
        alive_masses[0] = total_weight
        alive_counts[0] = sum(alive)
        return (
            output, output_len, best_slot, HSPEC_CONSENSUS_STOP_C1_FOLLOW,
            confidences, alive_masses, alive_counts,
        )

    output_len = 0
    stop_code = HSPEC_CONSENSUS_STOP_VALUE_END
    for depth in range(depth_limit):
        unique_tokens = np.zeros((width,), dtype=np.int32)
        token_masses = np.zeros((width,), dtype=np.float64)
        unique_count = 0
        total_alive_mass = 0.0
        count_alive = 0
        for slot in range(width):
            if alive[slot] == 0 or int(candidate_lens[slot]) <= depth:
                continue
            count_alive += 1
            weight = float(weights[slot])
            total_alive_mass += weight
            token = int(candidate_tokens[slot, depth])
            token_index = -1
            for index in range(unique_count):
                if int(unique_tokens[index]) == token:
                    token_index = index
                    break
            if token_index < 0:
                token_index = unique_count
                unique_tokens[unique_count] = token
                unique_count += 1
            token_masses[token_index] += weight
        if unique_count <= 0 or total_alive_mass <= 0.0:
            break
        best_token_index = 0
        for index in range(1, unique_count):
            if token_masses[index] > token_masses[best_token_index]:
                best_token_index = index
        confidence = float(token_masses[best_token_index]) / total_alive_mass
        confidences[depth] = confidence
        alive_masses[depth] = total_alive_mass
        alive_counts[depth] = count_alive
        if confidence < float(minimum_confidence):
            stop_code = HSPEC_CONSENSUS_STOP_DISAGREEMENT
            break
        best_token = int(unique_tokens[best_token_index])
        output[depth] = best_token
        output_len = depth + 1
        for slot in range(width):
            if (
                alive[slot] != 0
                and (
                    int(candidate_lens[slot]) <= depth
                    or int(candidate_tokens[slot, depth]) != best_token
                )
            ):
                alive[slot] = 0

    source_slot = -1
    for slot in range(width):
        if alive[slot] != 0:
            source_slot = slot
            break
    if output_len > 0 and source_slot < 0:
        return (
            output, 0, -1, HSPEC_CONSENSUS_STOP_INVALID,
            confidences, alive_masses, alive_counts,
        )
    return (
        output, output_len, source_slot, stop_code,
        confidences, alive_masses, alive_counts,
    )


candidate_weights_python = _candidate_weights_python_impl
prefix_consensus_python = _prefix_consensus_python_impl

if HSPEC_CONSENSUS_NUMBA_AVAILABLE:
    # The prefix kernel calls this helper from nopython mode.  Register the
    # original function itself so Python and Numba share one implementation.
    register_jitable(_candidate_weights_python_impl)
    candidate_weights_numba = njit(cache=True, nogil=True, fastmath=False)(
        _candidate_weights_python_impl
    )
    prefix_consensus_numba = njit(cache=True, nogil=True, fastmath=False)(
        _prefix_consensus_python_impl
    )
else:  # pragma: no cover
    candidate_weights_numba = None
    prefix_consensus_numba = None


def warm_consensus_selector() -> None:
    if prefix_consensus_numba is None:
        return
    prefix_consensus_numba(
        np.linspace(1.0, 0.3, 8, dtype=np.float64),
        np.linspace(0.9, 0.6, 8, dtype=np.float64),
        np.arange(8, dtype=np.int16),
        np.asarray([
            [1, 2, 3, 4], [1, 2, 5, 6], [1, 7, 8, 9], [2, 3, 4, 5],
            [2, 3, 6, 7], [2, 8, 9, 0], [3, 4, 5, 6], [3, 7, 8, 9],
        ], dtype=np.int32),
        np.full((8,), 4, dtype=np.int16),
        0.0,
        HSPEC_CONSENSUS_WEIGHT_UTILITY,
        1.0,
        True,
        HSPEC_CONSENSUS_MODE_C2,
        0.5,
        4,
    )
