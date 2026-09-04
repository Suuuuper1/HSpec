# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sampled, bounded observability for old-ABI parallel draft proposers."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch_npu.npu.streams import Event
from vllm.logger import logger


_SCHEMA = "dflash-dspark.phase5-draft-metrics.v1"
_MAX_PERCENTILE_SAMPLES = 4096
_MAX_SHAPES = 256
_DP_SCHEMA = "dflash-dspark.dp-repair-draft-metrics.v1"
_DP_CAPABILITY_SCHEMA = "dflash-dspark.dp-repair-capability.v2"


class DraftDPObserver:
    """Always-on integer accounting with sampled host-only sync timing.

    The correctness counters deliberately do not depend on Phase-5 device
    profiling. They add no device event, transfer, synchronization, or
    per-token logging. Detailed trace rows are opt-in and bounded in memory.
    """

    _KINDS = ("real", "dummy", "profile")

    def __init__(
        self,
        *,
        method: str,
        dp_size: int,
        dp_rank: int,
        draft_model_kind: str,
        sample_every: int = 64,
        trace_enabled: bool | None = None,
        trace_limit: int | None = None,
    ) -> None:
        if not method:
            raise ValueError("Draft DP observer method must be non-empty")
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError(
                f"Invalid draft DP observer topology: size/rank={dp_size}/{dp_rank}"
            )
        if draft_model_kind not in {"dense", "moe"}:
            raise ValueError(
                f"Invalid draft model kind {draft_model_kind!r}; expected dense or moe"
            )
        if sample_every <= 0:
            raise ValueError("Draft DP sync sample interval must be positive")
        if trace_enabled is None:
            trace_enabled = os.getenv("VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE", "0") != "0"
        if trace_limit is None:
            trace_limit = int(
                os.getenv("VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE_LIMIT", "256")
            )
        if trace_limit <= 0:
            raise ValueError("Draft DP trace limit must be positive")

        self.method = method
        self.dp_size = int(dp_size)
        self.dp_rank = int(dp_rank)
        self.draft_model_kind = draft_model_kind
        self.sync_mode = (
            "local_fast_path" if draft_model_kind == "dense" else "cpu_group_max_pad"
        )
        self.sample_every = int(sample_every)
        self.trace_enabled = bool(trace_enabled)
        self.trace_limit = int(trace_limit)
        self._sequence = 0
        self._counters = {
            "draft_dp_real_calls": 0,
            "draft_dp_dummy_calls": 0,
            "draft_dp_profile_calls": 0,
            "draft_dp_sync_calls": 0,
            "draft_dp_sync_skipped_dense": 0,
            "draft_dp_actual_query_tokens": 0,
            "draft_dp_execution_query_tokens": 0,
            "draft_dp_padding_tokens": 0,
            "draft_dp_padding_steps": 0,
            "draft_dp_plan_failures": 0,
        }
        self._sync_samples_ms: list[float] = []
        self._sync_sample_overflow = 0
        self._trace: list[dict[str, Any]] = []
        self._trace_dropped = 0

    def begin_sync(self, kind: str) -> tuple[int, int | None]:
        if kind not in self._KINDS:
            raise ValueError(f"Unknown draft DP invocation kind {kind!r}")
        sequence = self._sequence
        self._sequence += 1
        self._counters[f"draft_dp_{kind}_calls"] += 1
        self._counters["draft_dp_sync_calls"] += 1
        if self.dp_size > 1 and self.draft_model_kind == "dense":
            self._counters["draft_dp_sync_skipped_dense"] += 1
        start_ns = (
            time.perf_counter_ns() if sequence % self.sample_every == 0 else None
        )
        return sequence, start_ns

    def finish_sync(
        self,
        *,
        sequence: int,
        kind: str,
        start_ns: int | None,
        batch_size: int,
        num_queries: int,
        num_actual_tokens: int,
        num_tokens: int | None,
        num_padding_tokens: int | None,
        success: bool,
    ) -> None:
        if start_ns is not None:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            if len(self._sync_samples_ms) < _MAX_PERCENTILE_SAMPLES:
                self._sync_samples_ms.append(duration_ms)
            else:
                self._sync_sample_overflow += 1
        if not success:
            self._counters["draft_dp_plan_failures"] += 1
        else:
            assert num_tokens is not None and num_padding_tokens is not None
            if num_tokens != num_actual_tokens + num_padding_tokens:
                raise AssertionError("Draft DP observer token conservation failed")
            self._counters["draft_dp_actual_query_tokens"] += num_actual_tokens
            self._counters["draft_dp_execution_query_tokens"] += num_tokens
            self._counters["draft_dp_padding_tokens"] += num_padding_tokens
            self._counters["draft_dp_padding_steps"] += int(num_padding_tokens > 0)

        if not self.trace_enabled:
            return
        row = {
            "schema_version": "dflash-dspark.dp-repair-trace-row.v1",
            "method": self.method,
            "dp_rank": self.dp_rank,
            "proposal_sequence": int(sequence),
            "kind": kind,
            "batch_size": int(batch_size),
            "num_queries": int(num_queries),
            "num_actual_tokens": int(num_actual_tokens),
            "num_tokens": num_tokens,
            "num_padding_tokens": num_padding_tokens,
            "sync_enter": True,
            "sync_exit": bool(success),
            "context_enter": False,
            "context_exit": False,
            "status": "SYNCED" if success else "FAILED",
        }
        if len(self._trace) < self.trace_limit:
            self._trace.append(row)
        else:
            self._trace_dropped += 1

    def begin_context(self, sequence: int) -> None:
        if not self.trace_enabled:
            return
        for row in reversed(self._trace):
            if row["proposal_sequence"] == sequence:
                row["context_enter"] = True
                row["status"] = "CONTEXT_ENTERED"
                return

    def finish_context(self, sequence: int, *, success: bool) -> None:
        if not self.trace_enabled:
            return
        for row in reversed(self._trace):
            if row["proposal_sequence"] == sequence:
                row["context_exit"] = bool(success)
                row["status"] = "PASS" if success else "FAILED"
                return

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = int(math.ceil(percentile * len(ordered))) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    def snapshot(self) -> dict[str, Any]:
        samples = self._sync_samples_ms
        return {
            "schema_version": _DP_SCHEMA,
            "method": self.method,
            "draft_dp_size": self.dp_size,
            "draft_dp_rank": self.dp_rank,
            "draft_model_kind": self.draft_model_kind,
            "draft_dp_sync_mode": self.sync_mode,
            "counters": dict(self._counters),
            "invariants": {
                "sync_call_parity": self._counters["draft_dp_sync_calls"]
                == self._counters["draft_dp_real_calls"]
                + self._counters["draft_dp_dummy_calls"]
                + self._counters["draft_dp_profile_calls"],
                "token_conservation": self._counters[
                    "draft_dp_execution_query_tokens"
                ]
                == self._counters["draft_dp_actual_query_tokens"]
                + self._counters["draft_dp_padding_tokens"],
            },
            "host_sync_timer": {
                "sample_every": self.sample_every,
                "sample_count": len(samples),
                "sample_overflow": self._sync_sample_overflow,
                "p50_ms": self._percentile(samples, 0.50),
                "p95_ms": self._percentile(samples, 0.95),
                "max_ms": max(samples) if samples else None,
            },
            "trace": {
                "enabled": self.trace_enabled,
                "limit": self.trace_limit,
                "dropped": self._trace_dropped,
                "rows": [dict(row) for row in self._trace],
            },
            "device_synchronize_calls": 0,
            "new_npu_collectives": 0,
        }


@dataclass
class _TimerAccumulator:
    count: int = 0
    total_ms: float = 0.0
    minimum_ms: float = math.inf
    maximum_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)
    _replace_index: int = 0

    def add(self, duration_ms: float) -> None:
        value = float(duration_ms)
        self.count += 1
        self.total_ms += value
        self.minimum_ms = min(self.minimum_ms, value)
        self.maximum_ms = max(self.maximum_ms, value)
        if len(self.samples_ms) < _MAX_PERCENTILE_SAMPLES:
            self.samples_ms.append(value)
        else:
            self.samples_ms[self._replace_index] = value
            self._replace_index = (self._replace_index + 1) % _MAX_PERCENTILE_SAMPLES

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = int(math.ceil(percentile * len(ordered))) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / self.count if self.count else None,
            "min_ms": self.minimum_ms if self.count else None,
            "max_ms": self.maximum_ms if self.count else None,
            "p50_ms": self._percentile(self.samples_ms, 0.50),
            "p90_ms": self._percentile(self.samples_ms, 0.90),
            "p99_ms": self._percentile(self.samples_ms, 0.99),
            "percentile_sample_count": len(self.samples_ms),
        }


@dataclass
class _PendingStep:
    ordinal: int
    shape: tuple[int, int, int]
    host_start_ns: int
    timers: list[tuple[str, Event, Event]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    device_counters: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    completion_event: Event | None = None
    host_end_ns: int = 0


class ParallelDraftMetrics:
    """Collect device timers without synchronizing the normal decode path.

    Disabled mode is a few Python branches. Enabled mode records only every
    ``sample_every`` proposal and synchronizes once per ``flush_every`` sampled
    proposals. Percentile storage and the B/T/K histogram are both bounded.
    """

    def __init__(
        self,
        *,
        method: str,
        enabled: bool,
        sample_every: int,
        flush_every: int,
    ) -> None:
        if sample_every <= 0 or flush_every <= 0:
            raise ValueError("Phase-5 profiling intervals must be positive")
        self.method = method
        self.enabled = bool(enabled)
        self.sample_every = int(sample_every)
        self.flush_every = int(flush_every)
        self._step_ordinal = 0
        self._sampled_steps = 0
        self._flush_count = 0
        self._active: _PendingStep | None = None
        self._pending: list[_PendingStep] = []
        self._timers: dict[str, _TimerAccumulator] = defaultdict(_TimerAccumulator)
        self._counters: dict[str, int] = defaultdict(int)
        self._shapes: dict[str, int] = {}
        self._shape_overflow = 0
        self._profile_d2h_scalar_count = 0

    @property
    def sampling_active(self) -> bool:
        return self._active is not None

    def begin_step(self, *, batch_size: int, context_tokens: int, k: int) -> None:
        if self._active is not None:
            raise RuntimeError("parallel draft metric step was not closed")
        ordinal = self._step_ordinal
        self._step_ordinal += 1
        if not self.enabled or ordinal % self.sample_every != 0:
            return
        self._active = _PendingStep(
            ordinal=ordinal,
            shape=(int(batch_size), int(context_tokens), int(k)),
            host_start_ns=time.perf_counter_ns(),
        )

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        active = self._active
        if active is None:
            yield
            return
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        start.record()
        try:
            with torch.profiler.record_function(name):
                yield
        finally:
            end.record()
            active.timers.append((str(name), start, end))

    def add_counter(self, name: str, value: int) -> None:
        if self._active is not None:
            self._active.counters[str(name)] = (
                self._active.counters.get(str(name), 0) + int(value)
            )

    def add_device_counter(self, name: str, value: torch.Tensor) -> None:
        if self._active is None:
            return
        if value.numel() != 1:
            raise ValueError("parallel draft device counters must be scalar tensors")
        self._active.device_counters.append((str(name), value))

    def end_step(self) -> None:
        active = self._active
        if active is None:
            return
        active.host_end_ns = time.perf_counter_ns()
        active.completion_event = Event(enable_timing=True)
        active.completion_event.record()
        self._pending.append(active)
        self._active = None
        self._sampled_steps += 1
        if len(self._pending) >= self.flush_every:
            self.flush()

    def abort_step(self) -> None:
        self._active = None

    def flush(self) -> dict[str, Any] | None:
        if not self._pending:
            return None
        final_events = [
            step.completion_event
            for step in self._pending
            if step.completion_event is not None
        ]
        if final_events:
            # All proposer events are recorded on the current compute stream.
            # Synchronizing the final event covers every earlier sampled event.
            final_events[-1].synchronize()
        for step in self._pending:
            host_ms = (step.host_end_ns - step.host_start_ns) / 1_000_000.0
            self._timers["spec/draft_host_enqueue_ms"].add(host_ms)
            for name, start, end in step.timers:
                self._timers[name].add(start.elapsed_time(end))
            for name, value in step.counters.items():
                self._counters[name] += value
            for name, value in step.device_counters:
                self._counters[name] += int(value)
                self._profile_d2h_scalar_count += 1
            shape_key = f"B={step.shape[0]},T={step.shape[1]},K={step.shape[2]}"
            if shape_key in self._shapes:
                self._shapes[shape_key] += 1
            elif len(self._shapes) < _MAX_SHAPES:
                self._shapes[shape_key] = 1
            else:
                self._shape_overflow += 1
        self._pending.clear()
        self._flush_count += 1
        payload = self.snapshot()
        logger.info("PHASE5_DRAFT_METRICS %s", json.dumps(payload, sort_keys=True))
        return payload

    def snapshot(self) -> dict[str, Any]:
        peak_allocated = None
        peak_reserved = None
        if self.enabled and hasattr(torch, "npu"):
            try:
                peak_allocated = int(torch.npu.max_memory_allocated())
                peak_reserved = int(torch.npu.max_memory_reserved())
            except Exception:
                # Device memory counters are ancillary; timer evidence remains
                # valid and the benchmark analyzer reports their absence.
                pass
        return {
            "schema_version": _SCHEMA,
            "method": self.method,
            "enabled": self.enabled,
            "sample_every": self.sample_every,
            "flush_every": self.flush_every,
            "proposal_steps": self._step_ordinal,
            "sampled_steps": self._sampled_steps,
            "flush_count": self._flush_count,
            "timers": {
                name: accumulator.snapshot()
                for name, accumulator in sorted(self._timers.items())
            },
            "counters": dict(sorted(self._counters.items())),
            "transfers": {
                "new_hotpath_h2d_bytes": 0,
                "profile_d2h_scalar_count": self._profile_d2h_scalar_count,
            },
            "shape_samples": dict(sorted(self._shapes.items())),
            "shape_overflow": self._shape_overflow,
            "npu_peak_allocated_bytes": peak_allocated,
            "npu_peak_reserved_bytes": peak_reserved,
        }


def phase5_capability_manifest(
    speculative_config: Any,
    *,
    vllm_config: Any | None = None,
    draft_model_kind: str = "dense",
    certification_state: str = "phase1_production_gate_closed",
) -> dict[str, Any]:
    """Return the runtime support surface consumed by the R5 analyzer."""
    manifest = {
        "schema_version": "dflash-dspark.phase5-capability.v1",
        "method": speculative_config.method,
        "stable_baseline": {
            "draft_execution": "eager",
            "proposal": speculative_config.draft_sample_method,
            "verification": speculative_config.rejection_sample_method,
            "fixed_k": int(speculative_config.num_speculative_tokens),
            "full_vocab": True,
        },
        "enabled": {
            "sampled_observability": bool(
                speculative_config.parallel_draft_profile_enabled
            ),
            "persistent_sampling_workspace": False,
            "fused_context_kv_projection": True,
        },
        "evaluated_and_rolled_back": {
            "persistent_sampling_workspace": True,
        },
        "unsupported_fail_closed": {
            "draft_full_graph": True,
            "incremental_context_kv": True,
            "prefix_sharing": True,
            "dspark_draft_topk": True,
            "dynamic_k_or_confidence": True,
        },
    }
    # Keep the v1 schema/fields accepted by the frozen Phase-5 analyzer while
    # making new logs self-describing. DP certification consumes the strict
    # repair manifest below, never this compatibility record alone.
    if vllm_config is not None:
        parallel = vllm_config.parallel_config
        manifest["dp_extension"] = {
            "effective_vllm_dp_size": int(parallel.data_parallel_size),
            "draft_model_kind": draft_model_kind,
            "draft_dp_sync_mode": (
                "local_fast_path"
                if draft_model_kind == "dense"
                else "cpu_group_max_pad"
            ),
            "certification_state": certification_state,
        }
    return manifest


def dp_repair_capability_manifest(
    speculative_config: Any,
    vllm_config: Any,
    *,
    draft_model_kind: str,
    requested_vllm_dp_size: int | None = None,
    production_dp_gt1_enabled: bool = False,
) -> dict[str, Any]:
    """Strict per-rank capability record for the DP repair analyzer."""

    parallel = vllm_config.parallel_config
    effective_dp_size = int(parallel.data_parallel_size)
    requested_dp_size = (
        effective_dp_size
        if requested_vllm_dp_size is None
        else int(requested_vllm_dp_size)
    )
    manifest = {
        "schema_version": (
            _DP_CAPABILITY_SCHEMA
            if production_dp_gt1_enabled
            else "dflash-dspark.dp-repair-capability.v1"
        ),
        "method": speculative_config.method,
        "effective_vllm_dp_size": effective_dp_size,
        "effective_vllm_dp_rank": int(parallel.data_parallel_rank),
        "draft_model_kind": draft_model_kind,
        "draft_dp_sync_mode": (
            "local_fast_path"
            if draft_model_kind == "dense"
            else "cpu_group_max_pad"
        ),
        "production_dp_gt1_enabled": bool(production_dp_gt1_enabled),
        "certified_candidate": {
            "draft_execution": "eager",
            "proposal": speculative_config.draft_sample_method,
            "verification": speculative_config.rejection_sample_method,
            "fixed_k": int(speculative_config.num_speculative_tokens),
            "full_vocab": True,
            "draft_tensor_parallel_size": int(
                speculative_config.draft_parallel_config.tensor_parallel_size
            ),
        },
        "unsupported_fail_closed": {
            "draft_moe_or_ep": True,
            "draft_tensor_parallel_gt1": True,
            "pipeline_or_context_parallel": True,
            "async_scheduling": True,
            "draft_graph": True,
            "incremental_or_prefix_kv": True,
            "dynamic_k_or_confidence": True,
            "topk_or_reduced_vocab": True,
        },
    }
    if production_dp_gt1_enabled:
        manifest["requested_vllm_dp_size"] = requested_dp_size
        manifest["certification_state"] = "phase2_candidate_r1_r2_required"
    return manifest
