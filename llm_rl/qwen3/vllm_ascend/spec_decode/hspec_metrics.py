# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bounded, dependency-free metrics for the HSpec selector hot path."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, Mapping, Optional


_MAX_ACCEPT_HISTOGRAM_BIN = 64
_STOP_REASONS = frozenset({
    "score_gate",
    "empty_value",
    "batch_resource_cap",
    "abs_delta_cap",
    "adaptive_window",
    "value_end",
    "max_draft_tokens",
    "utility_abstain",
    "utility_length",
    "other",
})
_TIMING_FIELDS = frozenset({
    "project_submit_ms",
    "match_submit_ms",
    "d2h_sync_ms",
    "cpu_retrieve_ms",
    "total_ms",
    "r1_cpu_rerank_ms",
    "utility_cpu_score_ms",
    "utility_tail_pack_ms",
    "utility_batch_kernel_ms",
})
R1_RERANK_HISTOGRAM_US_BOUNDS = (
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    1000,
    1250,
    1500,
    2000,
    4000,
    8000,
)
_R1_ONLINE_ADDITIVE_KEYS = frozenset({
    "select_r1_execution_batches",
    "select_r1_execution_queries",
    "select_r1_changed_entry_count",
    "select_r1_rank_one_based_sum",
    "select_r1_suffix_sum",
    "select_r1_cpu_rerank_samples",
    "select_r1_cpu_rerank_us_overflow",
    "selector_config_fallback_count",
    "selector_runtime_fallback_count",
    "selector_runtime_query_fallback_count",
    "selector_d2h_fallback_count",
}) | frozenset(
    f"select_r1_cpu_rerank_us_le_{bound}"
    for bound in R1_RERANK_HISTOGRAM_US_BOUNDS
)
UTILITY_SCORE_HISTOGRAM_US_BOUNDS = (
    25,
    50,
    100,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2000,
    4000,
    8000,
)
_UTILITY_ONLINE_ADDITIVE_KEYS = frozenset({
    "selector_utility_model_fallback_count",
    "selector_utility_runtime_fallback_count",
    "selector_utility_query_fallback_count",
    "selector_utility_width_fallback_count",
    "selector_utility_invalid_row_fallback_count",
    "selector_utility_batch_fallback_count",
    "select_utility_batch_kernel_batches",
    "select_utility_batch_kernel_queries",
    "select_utility_lazy_r1_fallback_queries",
    "select_utility_r1_compare_queries",
    "select_utility_cpu_score_samples",
    "select_utility_cpu_score_us_overflow",
}) | frozenset(
    f"select_utility_cpu_score_us_le_{bound}"
    for bound in UTILITY_SCORE_HISTOGRAM_US_BOUNDS
) | frozenset(
    f"select_utility_{state}_{suffix}"
    for state in ("shadow", "execution")
    for suffix in (
        "batches",
        "queries",
        "proposed_count",
        "abstained_count",
        "changed_vs_r1_count",
        "action_sum",
    )
)
_BASE_ADDITIVE_KEYS = frozenset({
    "select_metric_windows",
    "select_eligible_queries",
    "select_proposed_requests",
    "select_abstained_requests",
    "select_drafted_tokens",
    "select_verified_requests",
    "select_first_token_accepts",
    "select_accepted_tokens",
    "select_emitted_tokens",
    "select_zero_accept_requests",
    "select_canceled_requests",
    "select_drafted_length_mismatch_count",
    # HSpec-only resource controller. Raw = emitted + truncated is audited at
    # the trainer interval boundary; emitted is the actual downstream draft
    # population and therefore must equal select_drafted_tokens.
    "select_draft_budget_batches",
    "select_draft_budget_hit_batches",
    "select_draft_budget_raw_tokens",
    "select_draft_budget_emitted_tokens",
    "select_draft_budget_truncated_tokens",
    "select_draft_budget_limited_requests",
    "select_top1_top2_margin_sum",
    "select_top1_top2_margin_count",
    "select_active_table_version_sum",
    "select_active_table_version_sq_sum",
    # S2 baseline-audit funnel. These are request counts from one strictly
    # nested population, so every downstream value must be <= its parent.
    "select_funnel_decode_requests",
    "select_funnel_active_table_requests",
    "select_funnel_prompt_id_ready_requests",
    "select_funnel_prompt_table_ready_requests",
    "select_funnel_batch_cache_ready_requests",
    "select_funnel_anchor_ready_requests",
    "select_funnel_eligible_queries",
})
SELECTOR_ADDITIVE_METRIC_KEYS = frozenset(
    set(_BASE_ADDITIVE_KEYS)
    | set(_R1_ONLINE_ADDITIVE_KEYS)
    | set(_UTILITY_ONLINE_ADDITIVE_KEYS)
    | {f"select_stop_{reason}_count" for reason in _STOP_REASONS}
    | {f"select_{name}" for name in _TIMING_FIELDS}
    | {f"select_accept_len_{length}_count" for length in range(_MAX_ACCEPT_HISTOGRAM_BIN + 1)}
    | {"select_accept_len_overflow_count"}
)


def hspec_r1_rerank_histogram_key(milliseconds: float) -> str:
    """Map one batch-level R1 CPU rerank duration to one fixed bucket."""
    microseconds = max(float(milliseconds) * 1000.0, 0.0)
    for bound in R1_RERANK_HISTOGRAM_US_BOUNDS:
        if microseconds <= float(bound):
            return f"select_r1_cpu_rerank_us_le_{bound}"
    return "select_r1_cpu_rerank_us_overflow"


def hspec_utility_score_histogram_key(milliseconds: float) -> str:
    """Map one batch-level utility score duration to one fixed bucket."""
    microseconds = max(float(milliseconds) * 1000.0, 0.0)
    for bound in UTILITY_SCORE_HISTOGRAM_US_BOUNDS:
        if microseconds <= float(bound):
            return f"select_utility_cpu_score_us_le_{bound}"
    return "select_utility_cpu_score_us_overflow"


def is_selector_additive_metric(key: str) -> bool:
    """Return whether ``key`` belongs to Patch 0's fixed metric keyspace."""
    return str(key) in SELECTOR_ADDITIVE_METRIC_KEYS


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def derive_selector_metrics(counters: Mapping[str, float]) -> Dict[str, float]:
    """Derive ratios from one already-aligned primitive-counter interval."""
    eligible = float(counters.get("select_eligible_queries", 0.0))
    proposed = float(counters.get("select_proposed_requests", 0.0))
    drafted = float(counters.get("select_drafted_tokens", 0.0))
    verified = float(counters.get("select_verified_requests", 0.0))
    first_accepts = float(counters.get("select_first_token_accepts", 0.0))
    accepted = float(counters.get("select_accepted_tokens", 0.0))
    emitted = float(counters.get("select_emitted_tokens", 0.0))
    zero_accepts = float(counters.get("select_zero_accept_requests", 0.0))
    margin_sum = float(counters.get("select_top1_top2_margin_sum", 0.0))
    margin_count = float(counters.get("select_top1_top2_margin_count", 0.0))
    windows = float(counters.get("select_metric_windows", 0.0))
    version_sum = float(counters.get("select_active_table_version_sum", 0.0))
    version_sq_sum = float(
        counters.get("select_active_table_version_sq_sum", 0.0)
    )
    version_mean = _safe_ratio(version_sum, windows)
    funnel_decode = float(counters.get("select_funnel_decode_requests", 0.0))
    funnel_active_table = float(
        counters.get("select_funnel_active_table_requests", 0.0)
    )
    funnel_prompt_id = float(
        counters.get("select_funnel_prompt_id_ready_requests", 0.0)
    )
    funnel_prompt_table = float(
        counters.get("select_funnel_prompt_table_ready_requests", 0.0)
    )
    funnel_batch_cache = float(
        counters.get("select_funnel_batch_cache_ready_requests", 0.0)
    )
    funnel_anchor = float(
        counters.get("select_funnel_anchor_ready_requests", 0.0)
    )
    funnel_eligible = float(
        counters.get("select_funnel_eligible_queries", 0.0)
    )

    result = {
        "select_proposal_coverage": _safe_ratio(proposed, eligible),
        "select_match_rate": _safe_ratio(first_accepts, verified),
        # This includes zero-accept verifications and is the unambiguous Patch 0 mean.
        "select_avg_accept_len": _safe_ratio(accepted, verified),
        # Kept separately to compare with the legacy hspec/avg_accept_length curve.
        "select_avg_accept_length_accepted_only": _safe_ratio(accepted, first_accepts),
        "select_accepted_tokens_per_query": _safe_ratio(accepted, eligible),
        "select_accept_efficiency": _safe_ratio(accepted, drafted),
        "select_zero_accept_rate": _safe_ratio(zero_accepts, verified),
        "select_avg_draft_len": _safe_ratio(drafted, proposed),
        "select_avg_emitted_tokens": _safe_ratio(emitted, verified),
        "select_top1_top2_margin_mean": _safe_ratio(margin_sum, margin_count),
        "select_active_table_version_mean": version_mean,
        "select_active_table_version_variance": max(
            _safe_ratio(version_sq_sum, windows) - version_mean * version_mean,
            0.0,
        ),
        "select_funnel_active_table_ratio": _safe_ratio(
            funnel_active_table, funnel_decode
        ),
        "select_funnel_prompt_id_ready_ratio": _safe_ratio(
            funnel_prompt_id, funnel_active_table
        ),
        "select_funnel_prompt_table_ready_ratio": _safe_ratio(
            funnel_prompt_table, funnel_prompt_id
        ),
        "select_funnel_batch_cache_ready_ratio": _safe_ratio(
            funnel_batch_cache, funnel_prompt_table
        ),
        "select_funnel_anchor_ready_ratio": _safe_ratio(
            funnel_anchor, funnel_batch_cache
        ),
        "select_funnel_eligible_after_anchor_ratio": _safe_ratio(
            funnel_eligible, funnel_anchor
        ),
        "select_funnel_end_to_end_eligible_ratio": _safe_ratio(
            funnel_eligible, funnel_decode
        ),
    }
    for field_name in _TIMING_FIELDS:
        value = float(counters.get(f"select_{field_name}", 0.0))
        result[f"select_{field_name}_per_window"] = _safe_ratio(value, windows)
    return result


@dataclass
class _SelectionWindow:
    window_id: int
    eligible_queries: int
    active_table_version: int
    proposed_requests: int = 0
    drafted_tokens: int = 0
    verified_requests: int = 0
    first_token_accepts: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    zero_accept_requests: int = 0
    canceled_requests: int = 0
    drafted_length_mismatch_count: int = 0
    proposal_finalized: bool = False
    margin_sum: float = 0.0
    margin_count: int = 0
    stop_reasons: Dict[str, int] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    accept_histogram: Dict[int, int] = field(default_factory=dict)
    accept_histogram_overflow: int = 0

    @property
    def terminal_requests(self) -> int:
        return int(self.verified_requests) + int(self.canceled_requests)

    @property
    def is_closed(self) -> bool:
        return self.proposal_finalized and self.terminal_requests >= self.proposed_requests


class HSpecSelectionMetricTracker:
    """Join proposal and verification counters locally before async reporting.

    The tracker never performs I/O and stores at most one tiny record per
    outstanding proposer invocation. A window closes only after every emitted
    proposal is either verified or explicitly canceled.
    """

    def __init__(self, max_draft_tokens: int):
        self._next_window_id = 1
        self._windows: Dict[int, _SelectionWindow] = {}
        self._histogram_max = min(max(int(max_draft_tokens), 0), _MAX_ACCEPT_HISTOGRAM_BIN)

    @property
    def pending_window_count(self) -> int:
        return len(self._windows)

    def begin_window(self, eligible_queries: int, active_table_version: int = 0) -> int:
        window_id = int(self._next_window_id)
        self._next_window_id += 1
        self._windows[window_id] = _SelectionWindow(
            window_id=window_id,
            eligible_queries=max(int(eligible_queries), 0),
            active_table_version=int(active_table_version),
        )
        return window_id

    def finalize_proposals(
        self,
        window_id: int,
        *,
        proposed_requests: int,
        drafted_tokens: int,
        stop_reasons: Optional[Mapping[str, int]] = None,
        margin_sum: float = 0.0,
        margin_count: int = 0,
        timings: Optional[Mapping[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        window = self._windows.get(int(window_id))
        if window is None or window.proposal_finalized:
            return None
        proposed = max(int(proposed_requests), 0)
        window.proposed_requests = min(proposed, window.eligible_queries)
        window.drafted_tokens = max(int(drafted_tokens), 0)
        window.margin_sum = float(margin_sum) if isfinite(float(margin_sum)) else 0.0
        window.margin_count = max(int(margin_count), 0)
        for reason, count in (stop_reasons or {}).items():
            normalized = str(reason) if str(reason) in _STOP_REASONS else "other"
            window.stop_reasons[normalized] = (
                window.stop_reasons.get(normalized, 0) + max(int(count), 0)
            )
        for name, value in (timings or {}).items():
            if str(name) not in _TIMING_FIELDS:
                continue
            numeric = float(value)
            if isfinite(numeric) and numeric >= 0.0:
                window.timings[str(name)] = numeric
        window.proposal_finalized = True
        return self._close_if_ready(window)

    def record_verification(
        self,
        window_id: int,
        *,
        accepted_prefix_len: int,
        emitted_tokens: int = 0,
        drafted_length_mismatch: bool = False,
    ) -> Optional[Dict[str, float]]:
        window = self._windows.get(int(window_id))
        if window is None or not window.proposal_finalized:
            return None
        if window.terminal_requests >= window.proposed_requests:
            return None
        accepted = max(int(accepted_prefix_len), 0)
        window.verified_requests += 1
        window.accepted_tokens += accepted
        window.emitted_tokens += max(int(emitted_tokens), 0)
        if accepted >= 1:
            window.first_token_accepts += 1
        else:
            window.zero_accept_requests += 1
        if drafted_length_mismatch:
            window.drafted_length_mismatch_count += 1
        if accepted <= self._histogram_max:
            window.accept_histogram[accepted] = window.accept_histogram.get(accepted, 0) + 1
        else:
            window.accept_histogram_overflow += 1
        return self._close_if_ready(window)

    def record_cancellation(self, window_id: int) -> Optional[Dict[str, float]]:
        window = self._windows.get(int(window_id))
        if window is None or not window.proposal_finalized:
            return None
        if window.terminal_requests >= window.proposed_requests:
            return None
        window.canceled_requests += 1
        return self._close_if_ready(window)

    def _close_if_ready(self, window: _SelectionWindow) -> Optional[Dict[str, float]]:
        if not window.is_closed:
            return None
        counters: Dict[str, float] = {
            "select_metric_windows": 1,
            "select_eligible_queries": window.eligible_queries,
            "select_proposed_requests": window.proposed_requests,
            "select_abstained_requests": max(
                window.eligible_queries - window.proposed_requests, 0
            ),
            "select_drafted_tokens": window.drafted_tokens,
            "select_verified_requests": window.verified_requests,
            "select_first_token_accepts": window.first_token_accepts,
            "select_accepted_tokens": window.accepted_tokens,
            "select_emitted_tokens": window.emitted_tokens,
            "select_zero_accept_requests": window.zero_accept_requests,
            "select_canceled_requests": window.canceled_requests,
            "select_drafted_length_mismatch_count": window.drafted_length_mismatch_count,
            "select_top1_top2_margin_sum": window.margin_sum,
            "select_top1_top2_margin_count": window.margin_count,
            "select_active_table_version_sum": window.active_table_version,
            "select_active_table_version_sq_sum": (
                window.active_table_version * window.active_table_version
            ),
        }
        for reason in _STOP_REASONS:
            counters[f"select_stop_{reason}_count"] = window.stop_reasons.get(reason, 0)
        for name in _TIMING_FIELDS:
            counters[f"select_{name}"] = window.timings.get(name, 0.0)
        for length, count in window.accept_histogram.items():
            counters[f"select_accept_len_{length}_count"] = count
        counters["select_accept_len_overflow_count"] = window.accept_histogram_overflow
        self._windows.pop(window.window_id, None)
        return counters


def hspec_abs_delta_bucket(abs_delta: int) -> str:
    """Map an exact position delta to a fixed, low-cardinality bucket."""
    value = max(int(abs_delta), 0)
    if value == 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 8:
        return "3_8"
    if value <= 32:
        return "9_32"
    if value <= 64:
        return "33_64"
    if value <= 256:
        return "65_256"
    return "gt_256"


class HSpecSelectorMetricStore:
    """Actor-side interval/cumulative store with atomic read-and-reset semantics."""

    def __init__(self):
        self._interval: Dict[str, float] = {}
        self._cumulative: Dict[str, float] = {}
        self._window_id = 0

    def record(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            key = str(key)
            if not is_selector_additive_metric(key):
                continue
            if not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not isfinite(numeric):
                continue
            self._interval[key] = self._interval.get(key, 0.0) + numeric
            self._cumulative[key] = self._cumulative.get(key, 0.0) + numeric

    def snapshot_and_reset(self) -> tuple[int, Dict[str, float], Dict[str, float]]:
        interval = dict(self._interval)
        if interval:
            self._window_id += 1
        self._interval.clear()
        return self._window_id, interval, dict(self._cumulative)
