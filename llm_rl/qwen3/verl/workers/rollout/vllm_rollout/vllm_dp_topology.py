"""Pure ownership and validation helpers for model-internal vLLM DP."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


VLLM_DP_SIZE_ENV = "VLLM_DP_SIZE"
VLLM_DP_SOURCE_ENV = "VERL_VLLM_DP_SIZE_SOURCE"
_VALID_ENV_SOURCES = {"user-env", "worker-derived", "yaml"}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from error
    if parsed <= 0 or str(value).strip() != str(parsed):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


@dataclass(frozen=True, slots=True)
class ResolvedVllmDP:
    size: int
    source: str
    yaml_value: int | None
    environment_value: int | None


def resolve_vllm_data_parallel_size(
    yaml_value: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedVllmDP:
    """Resolve YAML/env exactly once and reject ambiguous ownership."""

    env = os.environ if environ is None else environ
    configured = (
        None
        if yaml_value is None
        else _positive_int(yaml_value, name="rollout.vllm_data_parallel_size")
    )
    raw_environment = env.get(VLLM_DP_SIZE_ENV)
    environment = (
        None
        if raw_environment is None
        else _positive_int(raw_environment, name=VLLM_DP_SIZE_ENV)
    )
    if configured is not None and environment is not None and configured != environment:
        raise ValueError(
            "Conflicting model-internal vLLM DP configuration: "
            f"rollout.vllm_data_parallel_size={configured}, "
            f"{VLLM_DP_SIZE_ENV}={environment}. Set exactly one source or make "
            "both values equal."
        )

    if configured is not None:
        return ResolvedVllmDP(configured, "yaml", configured, environment)
    if environment is not None:
        source = env.get(VLLM_DP_SOURCE_ENV, "user-env")
        if source not in _VALID_ENV_SOURCES:
            raise ValueError(
                f"Invalid {VLLM_DP_SOURCE_ENV}={source!r}; expected one of "
                f"{sorted(_VALID_ENV_SOURCES)}"
            )
        return ResolvedVllmDP(environment, source, None, environment)
    return ResolvedVllmDP(1, "default", None, None)


def apply_vllm_dp_environment(
    resolved: ResolvedVllmDP,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Publish the unique result for old vLLM's environment-backed config."""

    env = os.environ if environ is None else environ
    env[VLLM_DP_SIZE_ENV] = str(resolved.size)
    env[VLLM_DP_SOURCE_ENV] = resolved.source


@dataclass(frozen=True, slots=True)
class VllmDPLayout:
    world_size: int
    external_dp_size: int
    vllm_dp_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    rollout_data_parallel_size: int
    dp_groups: tuple[tuple[int, ...], ...]


def validate_vllm_dp_layout(
    *,
    world_size: Any,
    vllm_dp_size: Any,
    tensor_parallel_size: Any,
    pipeline_parallel_size: Any = 1,
    prefill_context_parallel_size: Any = 1,
    rollout_data_parallel_size: Any = 1,
    require_rollout_dispatch_one: bool = True,
) -> VllmDPLayout:
    """Validate ``ExternalDP x DP x PP x PCP x TP`` and enumerate DP groups."""

    world = _positive_int(world_size, name="distributed world size")
    dp = _positive_int(vllm_dp_size, name="model-internal vLLM DP size")
    tp = _positive_int(tensor_parallel_size, name="vLLM tensor parallel size")
    pp = _positive_int(pipeline_parallel_size, name="vLLM pipeline parallel size")
    pcp = _positive_int(
        prefill_context_parallel_size, name="vLLM prefill context parallel size"
    )
    rollout_dp = _positive_int(
        rollout_data_parallel_size, name="rollout.data_parallel_size"
    )
    if require_rollout_dispatch_one and rollout_dp != 1:
        raise ValueError(
            "rollout.data_parallel_size is Verl request/output dispatch DP and "
            "must remain 1 for the synchronous DFlash/DSpark path; configure "
            "rollout.vllm_data_parallel_size (or VLLM_DP_SIZE) for model-internal DP"
        )
    model_parallel = dp * pp * pcp * tp
    if world % model_parallel != 0:
        raise ValueError(
            "Distributed world size is not divisible by the model-internal vLLM "
            f"layout: world={world}, DP={dp}, PP={pp}, PCP={pcp}, TP={tp}, "
            f"product={model_parallel}"
        )
    external = world // model_parallel

    groups: list[tuple[int, ...]] = []
    for external_rank in range(external):
        external_base = external_rank * model_parallel
        for pp_rank in range(pp):
            for pcp_rank in range(pcp):
                for tp_rank in range(tp):
                    group = tuple(
                        external_base
                        + (((dp_rank * pp + pp_rank) * pcp + pcp_rank) * tp)
                        + tp_rank
                        for dp_rank in range(dp)
                    )
                    groups.append(group)
    flattened = [rank for group in groups for rank in group]
    if sorted(flattened) != list(range(world)):
        raise AssertionError("Predicted vLLM DP groups are not a world partition")
    return VllmDPLayout(
        world_size=world,
        external_dp_size=external,
        vllm_dp_size=dp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        rollout_data_parallel_size=rollout_dp,
        dp_groups=tuple(groups),
    )


def predicted_dp_group(layout: VllmDPLayout, global_rank: int) -> tuple[int, ...]:
    matches = tuple(group for group in layout.dp_groups if global_rank in group)
    if len(matches) != 1:
        raise ValueError(
            f"Global rank {global_rank} belongs to {len(matches)} predicted DP groups"
        )
    return matches[0]


def build_topology_record(
    *,
    resolved: ResolvedVllmDP,
    layout: VllmDPLayout,
    global_rank: int,
    actual_dp_size: int | None = None,
    actual_dp_rank: int | None = None,
    actual_dp_group_ranks: Sequence[int] | None = None,
    method: str | None = None,
    draft_model_kind: str | None = None,
) -> dict[str, Any]:
    expected_group = predicted_dp_group(layout, global_rank)
    expected_rank = expected_group.index(global_rank)
    if actual_dp_size is not None and int(actual_dp_size) != resolved.size:
        raise RuntimeError(
            f"Engine vLLM DP size {actual_dp_size} != requested {resolved.size}"
        )
    if actual_dp_rank is not None and int(actual_dp_rank) != expected_rank:
        raise RuntimeError(
            f"Engine vLLM DP rank {actual_dp_rank} != predicted {expected_rank} "
            f"for global rank {global_rank}"
        )
    if actual_dp_group_ranks is not None:
        observed_group = tuple(int(rank) for rank in actual_dp_group_ranks)
        if observed_group != expected_group:
            raise RuntimeError(
                f"Engine DP group {observed_group} != predicted {expected_group} "
                f"for global rank {global_rank}"
            )
    effective_size = resolved.size if actual_dp_size is None else int(actual_dp_size)
    effective_rank = expected_rank if actual_dp_rank is None else int(actual_dp_rank)
    effective_group = (
        expected_group
        if actual_dp_group_ranks is None
        else tuple(int(rank) for rank in actual_dp_group_ranks)
    )
    parallel_block = method in {"dflash", "dspark"}
    if draft_model_kind is not None and draft_model_kind not in {"dense", "moe"}:
        raise ValueError(
            f"Invalid draft model kind {draft_model_kind!r}; expected dense or moe"
        )
    if not parallel_block and draft_model_kind is not None:
        raise ValueError("draft_model_kind is only valid for DFlash/DSpark topology")
    sync_mode = None
    if draft_model_kind == "dense":
        sync_mode = "local_fast_path"
    elif draft_model_kind == "moe":
        sync_mode = "cpu_group_max_pad"
    return {
        "schema_version": "dflash-dspark.dp-repair-topology-record.v1",
        "global_rank": int(global_rank),
        "world_size": layout.world_size,
        "requested_vllm_dp_size": resolved.size,
        "vllm_dp_source": resolved.source,
        "effective_vllm_dp_size": effective_size,
        "effective_vllm_dp_rank": effective_rank,
        "vllm_dp_group_ranks": list(effective_group),
        "external_dp_size": layout.external_dp_size,
        "rollout_data_parallel_size": layout.rollout_data_parallel_size,
        "tensor_parallel_size": layout.tensor_parallel_size,
        "pipeline_parallel_size": layout.pipeline_parallel_size,
        "prefill_context_parallel_size": layout.prefill_context_parallel_size,
        "method": method,
        "draft_model_kind": draft_model_kind,
        "draft_dp_sync_mode": sync_mode,
        "engine_reflected": actual_dp_size is not None,
    }


def build_topology_manifest(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strictly validate a complete set of per-rank records."""

    if not records:
        raise ValueError("Topology manifest requires at least one rank record")
    declared_world_sizes = {int(record["world_size"]) for record in records}
    if len(declared_world_sizes) != 1:
        raise ValueError(
            "Topology field world_size differs across ranks: "
            f"{sorted(declared_world_sizes)}"
        )
    world_size = declared_world_sizes.pop()
    ranks = [int(record["global_rank"]) for record in records]
    if len(records) != world_size or sorted(ranks) != list(range(world_size)):
        raise ValueError(
            f"Topology records must contain each rank once: world={world_size}, ranks={ranks}"
        )
    fields = (
        "requested_vllm_dp_size",
        "vllm_dp_source",
        "effective_vllm_dp_size",
        "external_dp_size",
        "rollout_data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "method",
        "draft_model_kind",
        "draft_dp_sync_mode",
    )
    for field in fields:
        values = {json.dumps(record.get(field), sort_keys=True) for record in records}
        if len(values) != 1:
            raise ValueError(f"Topology field {field} differs across ranks: {values}")
    dp_size = int(records[0]["effective_vllm_dp_size"])
    groups = {tuple(int(rank) for rank in record["vllm_dp_group_ranks"]) for record in records}
    if any(len(group) != dp_size or len(set(group)) != dp_size for group in groups):
        raise ValueError(f"Topology has malformed DP groups: {sorted(groups)}")
    expected_group_count = world_size // dp_size
    flattened_groups = [rank for group in groups for rank in group]
    if (
        world_size % dp_size != 0
        or len(groups) != expected_group_count
        or sorted(flattened_groups) != list(range(world_size))
    ):
        raise ValueError(
            "Topology DP groups must form an equal, disjoint world partition: "
            f"world={world_size}, DP={dp_size}, groups={sorted(groups)}"
        )
    for record in records:
        group = tuple(int(rank) for rank in record["vllm_dp_group_ranks"])
        rank = int(record["global_rank"])
        dp_rank = int(record["effective_vllm_dp_rank"])
        if not 0 <= dp_rank < dp_size or rank not in group or group[dp_rank] != rank:
            raise ValueError(
                f"Topology rank/group mismatch: rank={rank}, dp_rank={dp_rank}, group={group}"
            )
    return {
        "schema_version": "dflash-dspark.dp-repair-topology-manifest.v1",
        "status": "PASS",
        "world_size": world_size,
        "rank_count": len(records),
        "dp_group_count": len(groups),
        "dp_groups": [list(group) for group in sorted(groups)],
        "requested_vllm_dp_size": int(records[0]["requested_vllm_dp_size"]),
        "vllm_dp_source": records[0]["vllm_dp_source"],
        "effective_vllm_dp_size": dp_size,
        "rollout_data_parallel_size": int(
            records[0]["rollout_data_parallel_size"]
        ),
        "tensor_parallel_size": int(records[0]["tensor_parallel_size"]),
        "pipeline_parallel_size": int(records[0]["pipeline_parallel_size"]),
        "prefill_context_parallel_size": int(
            records[0]["prefill_context_parallel_size"]
        ),
        "method": records[0].get("method"),
        "draft_model_kind": records[0].get("draft_model_kind"),
        "draft_dp_sync_mode": records[0].get("draft_dp_sync_mode"),
        "all_engine_reflected": all(
            bool(record.get("engine_reflected")) for record in records
        ),
    }
