# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-shot, same-engine DP qualification wave for repair Phase 3."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


_SCHEMA = "dflash-dspark.dp-repair.phase3-prelude-record.v1"
_MANIFEST_SCHEMA = "dflash-dspark.dp-repair.phase3-prelude-manifest.v1"


def _one_worker_state(values: list[dict[str, Any]]) -> dict[str, Any]:
    if len(values) != 1 or not isinstance(values[0], dict):
        raise RuntimeError(
            "Phase-3 TP1 prelude requires exactly one local worker state; "
            f"got {len(values)}"
        )
    return values[0]


def _counter_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    before_values = (
        ((before.get("draft_observability") or {}).get("dp") or {}).get(
            "counters"
        )
        or {}
    )
    after_values = (
        ((after.get("draft_observability") or {}).get("dp") or {}).get(
            "counters"
        )
        or {}
    )
    keys = sorted(set(before_values) | set(after_values))
    return {
        key: int(after_values.get(key, 0)) - int(before_values.get(key, 0))
        for key in keys
    }


def _cache_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    before_cache = before.get("draft_probability_cache") or {}
    after_cache = after.get("draft_probability_cache") or {}
    fields = (
        "generation",
        "publish_count",
        "consume_count",
        "discard_count",
        "validation_sync_count",
    )
    return {
        field: int(after_cache.get(field, 0)) - int(before_cache.get(field, 0))
        for field in fields
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prelude evidence: {path}")
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _build_prelude_sampling_params(
    base_sampling_params: Any,
    *,
    max_tokens: int,
    seed: int,
) -> Any:
    """Build an isolated request compatible with the old speculative ABI."""
    sampling = copy.deepcopy(base_sampling_params)
    sampling.max_tokens = max_tokens
    # ignore_eos already makes max_tokens the exact decode bound.  Keeping
    # min_tokens at zero avoids the old V1 speculative-input restriction.
    sampling.min_tokens = 0
    sampling.min_p = 0.0
    sampling.logit_bias = None
    sampling.ignore_eos = True
    sampling.n = 1
    sampling.seed = seed
    return sampling


def _validate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    by_group: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        topology = record.get("topology") or {}
        group = tuple(int(value) for value in topology.get("vllm_dp_group_ranks", []))
        by_group.setdefault(group, []).append(record)

    groups: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        dp_size = int((rows[0].get("topology") or {}).get("effective_vllm_dp_size", 0))
        real_rows = [row for row in rows if row.get("submitted_requests") == 1]
        idle_rows = [row for row in rows if row.get("submitted_requests") == 0]
        sync_deltas = [
            int((row.get("counter_delta") or {}).get("draft_dp_sync_calls", 0))
            for row in rows
        ]
        group_errors: list[str] = []
        if len(group) != dp_size or len(rows) != dp_size:
            group_errors.append("group cardinality does not equal effective DP size")
        if len(real_rows) != 1 or len(idle_rows) != max(dp_size - 1, 0):
            group_errors.append("group does not contain exactly one real rank")
        if not sync_deltas or min(sync_deltas) <= 0 or len(set(sync_deltas)) != 1:
            group_errors.append("draft sync sequence counts are not positive and aligned")
        for row in real_rows:
            counters = row.get("counter_delta") or {}
            cache = row.get("probability_cache_delta") or {}
            if row.get("completed_requests") != 1 or int(
                row.get("output_token_count", 0)
            ) <= 0:
                group_errors.append("active rank did not complete its request")
            if int(counters.get("draft_dp_real_calls", 0)) <= 0:
                group_errors.append("active rank recorded no real proposal")
            if int(cache.get("publish_count", 0)) <= 0:
                group_errors.append("active rank published no probabilistic q")
            if int(cache.get("consume_count", 0)) <= 0:
                group_errors.append("active rank consumed no probabilistic q")
        for row in idle_rows:
            counters = row.get("counter_delta") or {}
            cache = row.get("probability_cache_delta") or {}
            if row.get("completed_requests") != 0 or row.get("output_token_count") != 0:
                group_errors.append("idle rank unexpectedly produced output")
            if int(counters.get("draft_dp_dummy_calls", 0)) <= 0:
                group_errors.append("idle rank recorded no runtime dummy proposal")
            if int(cache.get("publish_count", 0)) != 0 or int(
                cache.get("consume_count", 0)
            ) != 0:
                group_errors.append("idle rank published or consumed probabilistic q")
            if row.get("rng_before_sha256") != row.get("rng_after_sha256"):
                group_errors.append("idle runtime dummy advanced the NPU RNG")
        for row in rows:
            counters = row.get("counter_delta") or {}
            sync = int(counters.get("draft_dp_sync_calls", -1))
            calls = sum(
                int(counters.get(name, 0))
                for name in (
                    "draft_dp_real_calls",
                    "draft_dp_dummy_calls",
                    "draft_dp_profile_calls",
                )
            )
            if sync != calls:
                group_errors.append("sync call parity failed")
            if int(counters.get("draft_dp_execution_query_tokens", -1)) != int(
                counters.get("draft_dp_actual_query_tokens", 0)
            ) + int(counters.get("draft_dp_padding_tokens", 0)):
                group_errors.append("query token conservation failed")
            if int(counters.get("draft_dp_plan_failures", 0)) != 0:
                group_errors.append("draft DP plan failure was observed")
            if int(counters.get("draft_dp_sync_skipped_dense", -1)) != sync:
                group_errors.append("dense draft did not skip every DP collective")
            if not row.get("rng_restored", False):
                group_errors.append("prelude RNG state was not restored")
        if any(not row.get("cleanup_passed", False) for row in rows):
            group_errors.append("one or more ranks failed prelude cleanup")
        groups.append(
            {
                "ranks": list(group),
                "status": "PASS" if not group_errors else "FAIL",
                "errors": group_errors,
                "sync_call_deltas": sync_deltas,
            }
        )
        errors.extend(f"group {group}: {error}" for error in group_errors)
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rank_count": len(records),
        "groups": groups,
        "records": records,
    }


@torch.no_grad()
def run_dp_repair_phase3_prelude(rollout: Any, prompts: Any) -> dict[str, Any]:
    """Run one real request per DP group while all peer ranks stay idle."""
    resolved = rollout._resolved_speculation
    topology = dict(getattr(rollout, "_vllm_dp_topology_record", {}) or {})
    method = resolved.method
    if method not in {"dflash", "dspark"}:
        raise RuntimeError(f"Phase-3 prelude requires DFlash/DSpark, got {method!r}")
    if resolved.manifest.get("draft_sample_method") != "probabilistic":
        raise RuntimeError("Phase-3 prelude requires probabilistic proposal")
    if int(topology.get("effective_vllm_dp_size", 0)) <= 1:
        raise RuntimeError("Phase-3 prelude requires model-internal DP>1")
    if topology.get("draft_model_kind") != "dense":
        raise RuntimeError("Phase-3 prelude requires a replicated dense draft")

    engine = rollout.inference_engine
    before = _one_worker_state(
        engine.llm_engine.collective_rpc(
            "get_parallel_draft_worker_state",
            args=("pre_decode", True, True),
        )
    )
    dp_rank = int(topology["effective_vllm_dp_rank"])
    input_ids = prompts.batch["input_ids"]
    attention_mask = prompts.batch.get("attention_mask")
    local_prompts: list[dict[str, list[int]]] = []
    if dp_rank == 0:
        if input_ids.shape[0] == 0:
            raise RuntimeError("active prelude rank received no source prompt")
        if attention_mask is None:
            token_ids = input_ids[0].tolist()
        else:
            token_ids = input_ids[0][attention_mask[0].bool()].tolist()
        if not token_ids:
            raise RuntimeError("active prelude prompt is empty after padding removal")
        local_prompts = [{"prompt_token_ids": [int(value) for value in token_ids]}]

    max_tokens = int(os.environ.get("VERL_DP_REPAIR_PHASE3_PRELUDE_TOKENS", "2"))
    if not 1 <= max_tokens <= 8:
        raise ValueError("VERL_DP_REPAIR_PHASE3_PRELUDE_TOKENS must be in [1, 8]")
    sampling = _build_prelude_sampling_params(
        rollout.sampling_params,
        max_tokens=max_tokens,
        seed=int(os.environ.get("VERL_DP_REPAIR_PHASE3_SEED", "20260829")),
    )

    rng_before = torch.npu.get_rng_state().cpu()
    rng_before_sha256 = hashlib.sha256(rng_before.numpy().tobytes()).hexdigest()

    started_ns = time.perf_counter_ns()
    outputs = engine.generate(
        prompts=local_prompts,
        sampling_params=sampling,
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    rng_after = torch.npu.get_rng_state().cpu()
    rng_after_sha256 = hashlib.sha256(rng_after.numpy().tobytes()).hexdigest()
    torch.npu.set_rng_state(rng_before)
    rng_restored = torch.equal(torch.npu.get_rng_state().cpu(), rng_before)
    engine.llm_engine.collective_rpc("clear_draft_probability_cache", args=())
    reset_prefix_cache_succeeded = bool(engine.reset_prefix_cache())
    after = _one_worker_state(
        engine.llm_engine.collective_rpc(
            "get_parallel_draft_worker_state",
            args=("post_decode", False, True),
        )
    )
    counter_delta = _counter_delta(before, after)
    cache_delta = _cache_delta(before, after)
    after_cache = after.get("draft_probability_cache") or {}
    cleanup_passed = (
        reset_prefix_cache_succeeded
        and int(after_cache.get("current_bytes", -1)) == 0
        and int(after_cache.get("cached_request_count", -1)) == 0
    )
    output_token_ids = [
        int(token)
        for output in outputs
        for completion in output.outputs
        for token in completion.token_ids
    ]
    model_config = engine.llm_engine.model_config
    vocab_size = int(model_config.get_vocab_size())
    output_tokens_valid = all(0 <= token < vocab_size for token in output_token_ids)
    cleanup_passed = cleanup_passed and rng_restored and output_tokens_valid
    record = {
        "schema_version": _SCHEMA,
        "status": "PASS" if cleanup_passed else "FAIL",
        "method": method,
        "global_rank": int(torch.distributed.get_rank()),
        "worker_pid": os.getpid(),
        "topology": topology,
        "submitted_requests": len(local_prompts),
        "completed_requests": len(outputs),
        "output_token_count": len(output_token_ids),
        "output_token_preview": output_token_ids[:8],
        "output_tokens_valid": output_tokens_valid,
        "target_vocab_size": vocab_size,
        "elapsed_ms": elapsed_ms,
        "before": before,
        "after": after,
        "counter_delta": counter_delta,
        "probability_cache_delta": cache_delta,
        "reset_prefix_cache_succeeded": reset_prefix_cache_succeeded,
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "rng_restored": rng_restored,
        "cleanup_passed": cleanup_passed,
    }
    records: list[dict[str, Any] | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(records, record)
    manifest = _validate_group([row for row in records if row is not None])

    evidence_dir_value = os.environ.get("VERL_DP_REPAIR_PHASE3_PRELUDE_DIR")
    if not evidence_dir_value:
        raise RuntimeError("Phase-3 prelude evidence directory is not configured")
    evidence_dir = Path(evidence_dir_value)
    _atomic_write(
        evidence_dir / f"prelude_rank{record['global_rank']:05d}_pid{os.getpid()}.json",
        record,
    )
    if record["global_rank"] == 0:
        _atomic_write(evidence_dir / "prelude_manifest.json", manifest)

    lifecycle_record = {
        key: record[key]
        for key in (
            "schema_version",
            "status",
            "global_rank",
            "submitted_requests",
            "completed_requests",
            "output_token_count",
            "output_tokens_valid",
            "counter_delta",
            "probability_cache_delta",
            "reset_prefix_cache_succeeded",
            "rng_before_sha256",
            "rng_after_sha256",
            "rng_restored",
            "cleanup_passed",
        )
    }
    rollout._lifecycle_audit.after_dp_repair_prelude(engine, lifecycle_record)
    if manifest["status"] != "PASS":
        raise RuntimeError(
            "Phase-3 same-engine prelude failed: " + "; ".join(manifest["errors"])
        )
    return record
