# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bounded evidence recorder for the Phase-3 Verl speculative lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def module_storage_keys(module: torch.nn.Module | None) -> set[tuple[int, int]]:
    if module is None:
        return set()
    return {_storage_key(parameter) for parameter in module.parameters()}


def _sample_indices(numel: int, count: int) -> list[int]:
    if numel <= count:
        return list(range(numel))
    if count == 1:
        return [numel // 2]
    return sorted({round(index * (numel - 1) / (count - 1)) for index in range(count)})


def _unravel(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates: list[int] = []
    for size in reversed(shape):
        coordinates.append(index % size)
        index //= size
    return tuple(reversed(coordinates))


@torch.no_grad()
def sampled_parameter_checksum(
    module: torch.nn.Module | None,
    *,
    samples_per_parameter: int,
    excluded_storage_keys: set[tuple[int, int]] | None = None,
    max_parameters: int = 64,
) -> dict[str, Any] | None:
    """Hash a bounded, deterministic sample without flattening large tensors."""
    if module is None:
        return None
    excluded = excluded_storage_keys or set()
    parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.numel() and _storage_key(parameter) not in excluded
    ]
    if not parameters:
        return {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "parameter_count": 0,
            "sampled_parameter_count": 0,
            "sample_count": 0,
            "excluded_parameter_count": sum(
                1
                for parameter in module.parameters()
                if _storage_key(parameter) in excluded
            ),
        }
    chosen_positions = _sample_indices(len(parameters), min(max_parameters, len(parameters)))
    chosen = [parameters[index] for index in chosen_positions]
    values: list[torch.Tensor] = []
    metadata: list[str] = []
    for name, parameter in chosen:
        shape = tuple(parameter.shape)
        indices = _sample_indices(parameter.numel(), samples_per_parameter)
        metadata.append(f"{name}|{shape}|{parameter.dtype}|{indices}")
        values.extend(parameter[_unravel(index, shape)] for index in indices)
    # One bounded D2H synchronization per checksum, independent of model size.
    sample = torch.stack(values).to(dtype=torch.float32, device="cpu")
    digest = hashlib.sha256()
    digest.update("\n".join(metadata).encode("utf-8"))
    digest.update(sample.numpy().tobytes())
    return {
        "sha256": digest.hexdigest(),
        "parameter_count": len(parameters),
        "sampled_parameter_count": len(chosen),
        "sample_count": sample.numel(),
        "excluded_parameter_count": sum(
            1
            for parameter in module.parameters()
            if _storage_key(parameter) in excluded
        ),
    }


def _driver_worker(inference_engine):
    return inference_engine.llm_engine.model_executor.driver_worker.worker


def _models(inference_engine):
    runner = _driver_worker(inference_engine).model_runner
    target = runner.get_model()
    drafter = getattr(runner, "drafter", None)
    draft = getattr(drafter, "model", None)
    return runner, target, drafter, draft


def _allocator_summary() -> dict[str, Any]:
    try:
        from vllm_ascend.device_allocator.camem import CaMemAllocator

        allocator = CaMemAllocator.get_instance()
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    by_tag: dict[str, dict[str, int]] = {}
    pointer_rows: list[str] = []
    for pointer, data in allocator.pointer_to_data.items():
        size = int(data.handle[1])
        row = by_tag.setdefault(data.tag, {"allocations": 0, "bytes": 0, "backed_bytes": 0})
        row["allocations"] += 1
        row["bytes"] += size
        if data.cpu_backup_tensor is not None:
            row["backed_bytes"] += int(
                data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size()
            )
        pointer_rows.append(f"{data.tag}:{pointer}:{size}")
    kv_rows = sorted(row for row in pointer_rows if row.startswith("kv_cache:"))
    return {
        "available": True,
        "by_tag": by_tag,
        "kv_pointer_sha256": hashlib.sha256("\n".join(kv_rows).encode()).hexdigest(),
        "kv_allocation_count": len(kv_rows),
    }


def _rng_summary() -> dict[str, Any]:
    try:
        state = torch.npu.get_rng_state().cpu()
    except Exception as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "available": True,
        "bytes": int(state.numel() * state.element_size()),
        "sha256": hashlib.sha256(state.numpy().tobytes()).hexdigest(),
    }


def _draft_method_state(method: str | None, drafter: Any) -> dict[str, Any] | None:
    if method != "dspark" or drafter is None:
        return None
    names = (
        "_dspark_token_buffer",
        "_dspark_embedding_buffer",
        "_dspark_corrected_logits_buffer",
    )
    shapes = {
        name: list(getattr(drafter, name).shape)
        for name in names
        if isinstance(getattr(drafter, name, None), torch.Tensor)
    }
    return {
        "anchor_required": True,
        "markov_feedback": "sequential",
        "workspace_shapes": shapes,
    }


class SpeculativeLifecycleAudit:
    """Fail-fast state-machine checker and append-only JSONL recorder."""

    def __init__(self, config, resolved) -> None:
        self.enabled = bool(config.get("speculative_lifecycle_audit", False))
        self.strict = bool(config.get("speculative_lifecycle_strict", True))
        self.method = resolved.method
        self.manifest = dict(resolved.manifest)
        self.samples = int(config.get("speculative_lifecycle_samples_per_parameter", 8))
        self.sequence = 0
        self.target_updates = 0
        self.rollouts = 0
        self._draft_reference: str | None = None
        self._target_sync_reference: str | None = None
        self._kv_pointer_reference: str | None = None
        self._terminal_request_ids: set[str] = set()
        self.path: Path | None = None
        if not self.enabled:
            return
        output = os.environ.get("VERL_SPECULATIVE_LIFECYCLE_DIR")
        if not output:
            raise RuntimeError(
                "speculative_lifecycle_audit requires "
                "VERL_SPECULATIVE_LIFECYCLE_DIR"
            )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"lifecycle_rank{rank:05d}_pid{os.getpid()}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"refusing to append to existing lifecycle audit {self.path}")
        self._write({"event": "manifest", "manifest": self.manifest, "rank": rank})

    def _write(self, record: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        value = {
            "schema_version": "dflash-dspark.phase3-lifecycle.v1",
            "sequence": self.sequence,
            "time_ns": time.time_ns(),
            "method": self.method,
            **record,
        }
        self.sequence += 1
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _snapshot(
        self,
        inference_engine,
        event: str,
        *,
        checksum_weights: bool = True,
        flush_draft_metrics: bool = False,
        **extra,
    ) -> dict[str, Any]:
        runner, target, drafter, draft = _models(inference_engine)
        if checksum_weights:
            target_storage = module_storage_keys(target)
            target_checksum = sampled_parameter_checksum(
                target, samples_per_parameter=self.samples
            )
            draft_checksum = sampled_parameter_checksum(
                draft,
                samples_per_parameter=self.samples,
                excluded_storage_keys=target_storage,
            )
        else:
            target_checksum = None
            draft_checksum = None
        input_batch = getattr(runner, "input_batch", None)
        runner_request_ids = sorted(
            str(req_id) for req_id in getattr(runner, "requests", {})
        )
        raw_input_batch_request_ids = getattr(input_batch, "req_ids", None)
        request_identities_available = raw_input_batch_request_ids is not None
        input_batch_request_ids = (
            sorted(str(req_id) for req_id in raw_input_batch_request_ids if req_id is not None)
            if request_identities_available
            else []
        )
        llm_engine = inference_engine.llm_engine
        target_model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        compilation = getattr(runner, "compilation_config", None) or getattr(
            getattr(runner, "vllm_config", None), "compilation_config", None
        )
        cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
        probability_cache = (
            runner.draft_probability_cache_snapshot()
            if hasattr(runner, "draft_probability_cache_snapshot")
            else None
        )
        draft_observability = None
        if drafter is not None:
            if flush_draft_metrics and hasattr(
                drafter, "flush_observability_metrics"
            ):
                draft_observability = drafter.flush_observability_metrics()
                # The sampled profiler may have auto-flushed exactly at its
                # cadence boundary. Preserve its cumulative snapshot in the
                # lifecycle record even when that final explicit flush had no
                # pending device events and therefore returned phase5=None.
                if (
                    isinstance(draft_observability, dict)
                    and draft_observability.get("phase5") is None
                    and getattr(drafter, "phase5_metrics", None) is not None
                ):
                    draft_observability = {
                        **draft_observability,
                        "phase5": drafter.phase5_metrics.snapshot(),
                    }
            else:
                observer = getattr(drafter, "draft_dp_observer", None)
                phase5 = getattr(drafter, "phase5_metrics", None)
                draft_observability = {
                    "dp": observer.snapshot() if observer is not None else None,
                    "phase5": phase5.snapshot() if phase5 is not None else None,
                }
        qualification = getattr(drafter, "_dp_qualification", None)
        draft_capability = None
        if qualification is not None:
            draft_capability = {
                name: getattr(qualification, name)
                for name in (
                    "method",
                    "proposal",
                    "requested_dp_size",
                    "effective_dp_size",
                    "effective_dp_rank",
                    "draft_model_kind",
                    "draft_dp_sync_mode",
                )
            }
        record = {
            "event": event,
            "target_updates": self.target_updates,
            "rollouts": self.rollouts,
            "target_checksum": target_checksum,
            "draft_checksum": draft_checksum,
            "draft_checkpoint_load_count": getattr(
                drafter, "_checkpoint_load_count", None
            ),
            "drafter_type": type(drafter).__name__ if drafter is not None else None,
            "target_enforce_eager": getattr(target_model_config, "enforce_eager", None),
            "target_cudagraph_mode": getattr(cudagraph_mode, "name", str(cudagraph_mode)),
            "draft_uses_graph": getattr(drafter, "use_cuda_graph", None),
            "unfinished_requests": int(llm_engine.get_num_unfinished_requests()),
            "runner_request_count": len(runner_request_ids),
            "runner_request_ids": runner_request_ids,
            "input_batch_num_reqs": int(getattr(input_batch, "num_reqs", 0)),
            "input_batch_request_ids": input_batch_request_ids,
            "request_identities_available": request_identities_available,
            "hspec_collection_enabled": bool(getattr(runner, "_hspec_collect", False)),
            "draft_probability_cache": probability_cache,
            "draft_observability": draft_observability,
            "draft_capability": draft_capability,
            "rng_state": _rng_summary(),
            "draft_method_state": _draft_method_state(self.method, drafter),
            "allocator": _allocator_summary(),
            **extra,
        }
        return record

    @staticmethod
    def _digest(record: dict[str, Any], name: str) -> str | None:
        value = record.get(name)
        return value.get("sha256") if isinstance(value, dict) else None

    def _request_ids(self, record: dict[str, Any], event: str) -> set[str]:
        if not self.strict:
            return set(record.get("runner_request_ids", []))
        if record["unfinished_requests"] != 0:
            raise RuntimeError(
                f"Phase-3 {event} has unfinished engine requests: "
                f"{record['unfinished_requests']}"
            )
        if not record.get("request_identities_available", False):
            if record["runner_request_count"] or record["input_batch_num_reqs"]:
                raise RuntimeError(
                    f"Phase-3 {event} cannot audit non-empty request identities"
                )
            return set()

        runner_ids = set(record["runner_request_ids"])
        input_batch_ids = set(record["input_batch_request_ids"])
        if record["runner_request_count"] != len(runner_ids):
            raise RuntimeError(f"Phase-3 {event} has duplicate runner request IDs")
        if record["input_batch_num_reqs"] != len(input_batch_ids):
            raise RuntimeError(
                f"Phase-3 {event} input-batch count does not match its request IDs"
            )
        if runner_ids != input_batch_ids:
            raise RuntimeError(
                f"Phase-3 {event} runner/input-batch request IDs diverged"
            )
        return runner_ids

    def _require_quiescent_terminal_cache(
        self, record: dict[str, Any], event: str
    ) -> None:
        """Accept old-vLLM's terminal cache, but never partial or active state.

        V1 reports the final outputs before the model runner receives the next
        ``finished_req_ids`` update. Consequently ``LLM.generate()`` can return
        with a quiescent persistent batch containing only the just-finished
        request IDs. The next scheduler update must retire all of those IDs.
        """
        request_ids = self._request_ids(record, event)
        if self.strict and request_ids not in (set(), self._terminal_request_ids):
            raise RuntimeError(
                f"Phase-3 {event} contains request IDs outside the previous "
                "terminal cache"
            )

    def _check_parallel_draft(self, record: dict[str, Any]) -> None:
        if self.method not in {"dflash", "dspark"}:
            return
        if self.strict and record["draft_checkpoint_load_count"] != 1:
            raise RuntimeError(
                "Phase-3 draft must be loaded exactly once; got "
                f"{record['draft_checkpoint_load_count']!r}"
            )
        if self.strict and record["hspec_collection_enabled"]:
            raise RuntimeError(f"HSpec collection leaked into {self.method}")
        probability_cache = record.get("draft_probability_cache")
        if (
            self.strict
            and self.manifest.get("draft_sample_method") == "probabilistic"
            and (
                not isinstance(probability_cache, dict)
                or not probability_cache.get("enabled", False)
                or probability_cache.get("current_bytes") != 0
                or probability_cache.get("cached_request_count") != 0
            )
        ):
            raise RuntimeError(
                "Phase-4 probabilistic q cache is unavailable or non-empty at "
                f"lifecycle event {record['event']}"
            )
        digest = self._digest(record, "draft_checksum")
        if digest is None:
            return
        if self._draft_reference is None:
            self._draft_reference = digest
        elif self.strict and digest != self._draft_reference:
            raise RuntimeError(
                f"{self.method} immutable draft checksum changed across the RL lifecycle"
            )

    def after_load(self, inference_engine) -> None:
        if not self.enabled:
            return
        record = self._snapshot(inference_engine, "after_load")
        self._check_parallel_draft(record)
        self._require_quiescent_terminal_cache(record, "after_load")
        self._write(record)

    def after_wake(self, inference_engine, tags: list[str]) -> None:
        if not self.enabled:
            return
        event = "after_wake_weights" if "weights" in tags else "after_wake_kv"
        record = self._snapshot(
            inference_engine,
            event,
            checksum_weights=event == "after_wake_weights",
            tags=list(tags),
        )
        self._check_parallel_draft(record)
        self._require_quiescent_terminal_cache(record, event)
        if event == "after_wake_kv":
            allocator = record["allocator"]
            signature = allocator.get("kv_pointer_sha256") if allocator.get("available") else None
            if self._kv_pointer_reference is None:
                self._kv_pointer_reference = signature
            elif self.strict and signature != self._kv_pointer_reference:
                raise RuntimeError("KV virtual allocation identity changed across wake cycles")
        self._write(record)

    def after_target_update(self, inference_engine) -> None:
        if not self.enabled:
            return
        self.target_updates += 1
        record = self._snapshot(inference_engine, "after_target_update")
        self._check_parallel_draft(record)
        self._require_quiescent_terminal_cache(record, "after_target_update")
        digest = self._digest(record, "target_checksum")
        if self.strict and self._target_sync_reference == digest:
            raise RuntimeError("target checksum did not change after an actor update")
        self._target_sync_reference = digest
        self._write(record)

    def before_rollout(self, inference_engine) -> None:
        if not self.enabled:
            return
        if self.strict and self.target_updates != self.rollouts + 1:
            raise RuntimeError(
                "Phase-3 lifecycle must synchronize target weights exactly once "
                "before each rollout"
            )
        record = self._snapshot(
            inference_engine, "before_rollout", checksum_weights=False
        )
        self._check_parallel_draft(record)
        self._require_quiescent_terminal_cache(record, "before_rollout")
        self._write(record)

    def after_rollout(self, inference_engine) -> None:
        if not self.enabled:
            return
        record = self._snapshot(
            inference_engine,
            "after_rollout",
            flush_draft_metrics=self.method in {"dflash", "dspark"},
        )
        self._check_parallel_draft(record)
        request_ids = self._request_ids(record, "after_rollout")
        leaked = request_ids & self._terminal_request_ids
        if self.strict and leaked:
            raise RuntimeError(
                "Phase-3 scheduler did not retire previous terminal request IDs: "
                f"{sorted(leaked)}"
            )
        self._terminal_request_ids = request_ids
        self._write(record)
        self.rollouts += 1

    def after_dp_repair_prelude(
        self, inference_engine, prelude: dict[str, Any]
    ) -> None:
        """Adopt the old-V1 terminal cache left by the qualification wave.

        The prelude bypasses the RL rollout counter and actor update state
        machine. Old V1 keeps the just-finished request IDs until the next
        scheduler update, so record them as the prior terminal generation;
        ``after_rollout`` will require the formal wave to retire them.
        """
        if not self.enabled:
            return
        record = self._snapshot(
            inference_engine,
            "after_dp_repair_prelude",
            flush_draft_metrics=True,
            prelude=prelude,
        )
        self._check_parallel_draft(record)
        request_ids = self._request_ids(record, "after_dp_repair_prelude")
        leaked = request_ids & self._terminal_request_ids
        if self.strict and leaked:
            raise RuntimeError(
                "Phase-3 prelude reused an earlier terminal request ID: "
                f"{sorted(leaked)}"
            )
        self._terminal_request_ids = request_ids
        self._write(record)

    def after_sleep(self, *, reset_prefix_cache_succeeded: bool) -> None:
        if not self.enabled:
            return
        allocator = _allocator_summary()
        weights = allocator.get("by_tag", {}).get("weights", {})
        kv_cache = allocator.get("by_tag", {}).get("kv_cache", {})
        record = {
            "event": "after_sleep",
            "target_updates": self.target_updates,
            "rollouts": self.rollouts,
            "reset_prefix_cache_succeeded": bool(reset_prefix_cache_succeeded),
            "allocator": allocator,
        }
        if self.strict and not reset_prefix_cache_succeeded:
            raise RuntimeError("prefix/request cache reset failed before sleep")
        if self.strict and self.method in {"dflash", "dspark"}:
            if weights.get("bytes", 0) and weights.get("backed_bytes", 0) != weights.get("bytes", 0):
                raise RuntimeError("sleep did not back all weight-tag allocations")
            if kv_cache.get("backed_bytes", 0) != 0:
                raise RuntimeError("KV cache was copied to CPU instead of discarded")
        self._write(record)
