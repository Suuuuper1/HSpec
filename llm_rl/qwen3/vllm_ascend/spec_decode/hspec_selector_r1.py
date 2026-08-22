# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Frozen S6 R1 configuration and low-overhead CPU candidate reranking."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

try:
    from numba import njit

    HSPEC_R1_NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional on analysis hosts
    njit = None
    HSPEC_R1_NUMBA_AVAILABLE = False


_VALID_MODES = frozenset({"hardmax", "topk_position"})
_VALID_SIM_MODES = frozenset({"raw", "cosine"})
_VALID_POSITION_MODES = frozenset({"none", "exact", "log"})
_VALID_D2H = frozenset({"two_cpu", "pinned_two_async"})


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean, got {value!r}")


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(env.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class HSpecR1Config:
    """Worker-local selector configuration, parsed exactly once at init."""

    # S18 production default selected by the final S17 30B gate.  Explicit
    # hardmax remains the fail-closed and operator rollback configuration.
    mode: str = "topk_position"
    topk: int = 8
    sim_mode: str = "cosine"
    relative_radius: float = 0.0001
    suffix_cap: int = 8
    position_mode: str = "none"
    shadow: bool = False
    sample_log_rate: float = 0.0
    relative_weight: float = 7.719409849724556
    suffix_weight: float = 0.6382322349890022
    position_exact_weight: float = 0.0
    position_log_weight: float = 0.0
    utility_threshold: float = -1.0e30
    d2h_strategy: str = "pinned_two_async"
    fallback_reason: str | None = None

    @property
    def computes_topk(self) -> bool:
        return self.mode == "topk_position"

    @property
    def executes_topk(self) -> bool:
        return self.computes_topk and not self.shadow

    @property
    def position_mode_code(self) -> int:
        return {"none": 0, "exact": 1, "log": 2}[self.position_mode]

    def hardmax_fallback(self, reason: str) -> "HSpecR1Config":
        return replace(
            self,
            mode="hardmax",
            topk=1,
            sim_mode="raw",
            shadow=False,
            fallback_reason=str(reason),
        )

    @classmethod
    def from_environment(
        cls, env: Mapping[str, str] | None = None
    ) -> "HSpecR1Config":
        source = os.environ if env is None else env
        try:
            mode = str(
                source.get("HSPEC_SELECT_MODE", "topk_position")
            ).strip().lower()
            if mode not in _VALID_MODES:
                raise ValueError(f"unsupported HSPEC_SELECT_MODE={mode!r}")
            topk = _parse_int(source, "HSPEC_SELECT_TOPK", 8)
            sim_mode = str(
                source.get("HSPEC_SELECT_SIM_MODE", "cosine")
            ).strip().lower()
            if sim_mode not in _VALID_SIM_MODES:
                raise ValueError(f"unsupported HSPEC_SELECT_SIM_MODE={sim_mode!r}")
            relative_radius = _parse_float(
                source, "HSPEC_SELECT_RELATIVE_RADIUS", 0.0001
            )
            suffix_cap = _parse_int(source, "HSPEC_SELECT_SUFFIX_CAP", 8)
            position_mode = str(
                source.get("HSPEC_SELECT_POSITION_MODE", "none")
            ).strip().lower()
            if position_mode not in _VALID_POSITION_MODES:
                raise ValueError(
                    f"unsupported HSPEC_SELECT_POSITION_MODE={position_mode!r}"
                )
            shadow = _parse_bool(source.get("HSPEC_SELECT_SHADOW", "0"))
            sample_log_rate = _parse_float(
                source, "HSPEC_SELECT_SAMPLE_LOG_RATE", 0.0
            )
            relative_weight = _parse_float(
                source, "HSPEC_SELECT_RELATIVE_WEIGHT", 7.719409849724556
            )
            suffix_weight = _parse_float(
                source, "HSPEC_SELECT_SUFFIX_WEIGHT", 0.6382322349890022
            )
            exact_weight = _parse_float(
                source, "HSPEC_SELECT_POSITION_EXACT_WEIGHT", 0.0
            )
            log_weight = _parse_float(
                source, "HSPEC_SELECT_POSITION_LOG_WEIGHT", 0.0
            )
            utility_threshold = _parse_float(
                source, "HSPEC_SELECT_UTILITY_THRESHOLD", -1.0e30
            )
            d2h_strategy = str(
                source.get("HSPEC_SELECT_D2H_STRATEGY", "pinned_two_async")
            ).strip().lower()
            if d2h_strategy not in _VALID_D2H:
                raise ValueError(
                    f"unsupported HSPEC_SELECT_D2H_STRATEGY={d2h_strategy!r}"
                )
            if not 1 <= topk <= 16:
                raise ValueError("HSPEC_SELECT_TOPK must be in [1,16]")
            if relative_radius < 0.0:
                raise ValueError("HSPEC_SELECT_RELATIVE_RADIUS must be >= 0")
            if not 0 <= suffix_cap <= 64:
                raise ValueError("HSPEC_SELECT_SUFFIX_CAP must be in [0,64]")
            if not 0.0 <= sample_log_rate <= 1.0:
                raise ValueError("HSPEC_SELECT_SAMPLE_LOG_RATE must be in [0,1]")
            if mode == "hardmax":
                return cls(
                    mode="hardmax",
                    topk=1,
                    sim_mode="raw",
                    relative_radius=relative_radius,
                    suffix_cap=suffix_cap,
                    position_mode="none",
                    shadow=False,
                    sample_log_rate=sample_log_rate,
                    d2h_strategy=d2h_strategy,
                )
            if topk <= 1:
                raise ValueError("topk_position requires HSPEC_SELECT_TOPK > 1")
            if sim_mode != "cosine":
                raise ValueError("frozen R1 requires HSPEC_SELECT_SIM_MODE=cosine")
            return cls(
                mode=mode,
                topk=topk,
                sim_mode=sim_mode,
                relative_radius=relative_radius,
                suffix_cap=suffix_cap,
                position_mode=position_mode,
                shadow=shadow,
                sample_log_rate=sample_log_rate,
                relative_weight=relative_weight,
                suffix_weight=suffix_weight,
                position_exact_weight=exact_weight,
                position_log_weight=log_weight,
                utility_threshold=utility_threshold,
                d2h_strategy=d2h_strategy,
            )
        except ValueError as exc:
            # Invalid release/custom configuration must never inherit the R1
            # dataclass default accidentally.  This is the decode-safe P0
            # rollback required by the S18 merge contract.
            return cls(
                mode="hardmax",
                topk=1,
                sim_mode="raw",
                position_mode="none",
                shadow=False,
                fallback_reason=str(exc),
            )


def _rerank_one_prompt_python_impl(
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    entry_rollout_idx: np.ndarray,
    entry_offset: np.ndarray,
    token_buffer: np.ndarray,
    rollout_offsets: np.ndarray,
    rollout_lens: np.ndarray,
    current_tokens: np.ndarray,
    base_pos: int,
    n_entries: int,
    relative_radius: float,
    suffix_cap: int,
    relative_weight: float,
    suffix_weight: float,
    position_mode: int,
    position_exact_weight: float,
    position_log_weight: float,
    utility_threshold: float,
) -> tuple[int, int, float, int, int]:
    """Reference implementation shared by tests and the Numba fallback."""
    best_slot = -1
    best_suffix = 0
    best_utility = -math.inf
    best_abs_delta = 0
    best_remaining = 0
    top_score = -math.inf
    for slot in range(candidate_scores.shape[0]):
        idx = int(candidate_indices[slot])
        score = float(candidate_scores[slot])
        if 0 <= idx < int(n_entries) and math.isfinite(score):
            top_score = max(top_score, score)
    if not math.isfinite(top_score):
        return best_slot, best_suffix, best_utility, best_abs_delta, best_remaining
    denominator = max(abs(top_score), 1.0e-12)
    current_end = int(current_tokens.shape[0]) - 1
    for slot in range(candidate_scores.shape[0]):
        idx = int(candidate_indices[slot])
        score = float(candidate_scores[slot])
        if idx < 0 or idx >= int(n_entries) or not math.isfinite(score):
            continue
        relative_drop = max((top_score - score) / denominator, 0.0)
        if relative_drop > float(relative_radius):
            continue
        rollout_idx = int(entry_rollout_idx[idx])
        if rollout_idx < 0 or rollout_idx >= int(rollout_lens.shape[0]):
            continue
        offset = int(entry_offset[idx])
        rollout_len = int(rollout_lens[rollout_idx])
        remaining = rollout_len - offset
        if offset <= 0 or remaining <= 0:
            continue
        history_end = offset - 1
        base = int(rollout_offsets[rollout_idx])
        suffix = 0
        limit = min(int(suffix_cap), current_end + 1, history_end + 1)
        while suffix < limit:
            if int(current_tokens[current_end - suffix]) != int(
                token_buffer[base + history_end - suffix]
            ):
                break
            suffix += 1
        abs_delta = abs(history_end - int(base_pos))
        utility = (
            -float(relative_weight) * relative_drop
            + float(suffix_weight) * suffix
        )
        if position_mode == 1 and abs_delta == 0:
            utility += float(position_exact_weight)
        if position_mode == 2:
            utility -= float(position_log_weight) * math.log1p(abs_delta)
        if utility > best_utility:
            best_slot = slot
            best_suffix = suffix
            best_utility = utility
            best_abs_delta = abs_delta
            best_remaining = remaining
    if best_utility < float(utility_threshold):
        return -1, 0, best_utility, 0, 0
    return best_slot, best_suffix, best_utility, best_abs_delta, best_remaining


def rerank_one_prompt_python(*args):
    return _rerank_one_prompt_python_impl(*args)


if HSPEC_R1_NUMBA_AVAILABLE:
    rerank_one_prompt_numba = njit(cache=True, nogil=True, fastmath=True)(
        _rerank_one_prompt_python_impl
    )
else:  # pragma: no cover
    rerank_one_prompt_numba = None


def warm_r1_reranker() -> None:
    """Compile the selected sequential kernel before the first rollout."""
    if rerank_one_prompt_numba is None:
        raise RuntimeError("Numba is unavailable")
    scores = np.asarray([1.0, 0.99995], dtype=np.float32)
    indices = np.asarray([0, 1], dtype=np.int64)
    entry_rollout = np.asarray([0, 0], dtype=np.int32)
    entry_offset = np.asarray([1, 2], dtype=np.int32)
    tokens = np.asarray([10, 11, 12], dtype=np.int32)
    rollout_offsets = np.asarray([0], dtype=np.int64)
    rollout_lens = np.asarray([3], dtype=np.int32)
    current = np.asarray([10, 11], dtype=np.int32)
    rerank_one_prompt_numba(
        scores,
        indices,
        entry_rollout,
        entry_offset,
        tokens,
        rollout_offsets,
        rollout_lens,
        current,
        1,
        2,
        0.0001,
        8,
        7.719409849724556,
        0.6382322349890022,
        0,
        0.0,
        0.0,
        -1.0e30,
    )
