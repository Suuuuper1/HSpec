# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sampled, bounded observability for old-ABI parallel draft proposers."""

from __future__ import annotations

import json
import math
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


def phase5_capability_manifest(speculative_config: Any) -> dict[str, Any]:
    """Return the runtime support surface consumed by the R5 analyzer."""
    return {
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
