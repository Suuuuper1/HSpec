# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure Ray accelerator-capacity checks shared by launchers and trainers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _capacity(resources: Mapping[str, Any]) -> float:
    """Return the schedulable accelerator resource used by RayResourcePool.

    Ascend deployments may expose NPU directly or alias it to GPU because the
    existing Verl pool requests ``use_gpu=True``.  Preserve that precedence and
    never add GPU and NPU, which can be two names for the same physical device.
    """
    key = "GPU" if "GPU" in resources else "NPU"
    value = resources.get(key, 0)
    try:
        capacity = float(value)
    except (TypeError, ValueError):
        return 0.0
    return capacity if math.isfinite(capacity) and capacity > 0 else 0.0


def node_accelerator_capacities(
    resources_per_node: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Normalize Ray's per-node resource mapping without double counting."""
    return {
        str(node): _capacity(resources)
        for node, resources in resources_per_node.items()
        if _capacity(resources) > 0
    }


def evaluate_resource_pool_capacity(
    resource_pool_spec: Mapping[str, Sequence[int | float]],
    resources_per_node: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate total capacity and node-local bundle placement.

    A pool entry such as ``[16]`` is one node-local bundle.  It cannot be
    satisfied by two nodes with eight accelerators each even though the totals
    match.  Multiple bundles may share a larger node when their sum fits.
    """
    errors: list[str] = []
    required_bundles: list[dict[str, Any]] = []
    normalized_spec: dict[str, list[float]] = {}
    for pool, bundles in resource_pool_spec.items():
        normalized: list[float] = []
        for index, raw in enumerate(bundles):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value) or value <= 0:
                errors.append(
                    f"resource pool {pool!r} bundle {index} must be positive, got {raw!r}"
                )
                continue
            normalized.append(value)
            required_bundles.append(
                {"pool": str(pool), "index": index, "accelerators": value}
            )
        normalized_spec[str(pool)] = normalized

    available = node_accelerator_capacities(resources_per_node)
    remaining = dict(available)
    placements: list[dict[str, Any]] = []
    unplaced: list[dict[str, Any]] = []
    for bundle in sorted(
        required_bundles, key=lambda row: row["accelerators"], reverse=True
    ):
        candidates = [
            (capacity, node)
            for node, capacity in remaining.items()
            if capacity >= bundle["accelerators"]
        ]
        if not candidates:
            unplaced.append(bundle)
            continue
        _, node = min(candidates)
        remaining[node] -= bundle["accelerators"]
        placements.append({**bundle, "node": node})

    required_total = sum(row["accelerators"] for row in required_bundles)
    available_total = sum(available.values())
    if available_total < required_total:
        errors.append(
            f"total available accelerators {available_total:g} is less than "
            f"total required {required_total:g}"
        )
    if unplaced:
        errors.append(
            "node-local accelerator bundles cannot be placed; a bundle cannot "
            "be split across Ray nodes"
        )

    return {
        "status": "PASS" if not errors and not unplaced else "FAIL",
        "resource_pool_spec": normalized_spec,
        "required_total": required_total,
        "available_by_node": available,
        "available_total": available_total,
        "placements": placements,
        "unplaced_bundles": unplaced,
        "remaining_by_node": remaining,
        "errors": errors,
    }


def format_resource_capacity_error(assessment: Mapping[str, Any]) -> str:
    """Build one actionable error without method-specific terminology."""
    return (
        "Ray accelerator topology cannot satisfy the requested resource pools: "
        f"required={assessment.get('resource_pool_spec')}, "
        f"required_total={assessment.get('required_total')}, "
        f"available_by_node={assessment.get('available_by_node')}, "
        f"available_total={assessment.get('available_total')}, "
        f"unplaced={assessment.get('unplaced_bundles')}. "
        "Each process_on_nodes entry is node-local and cannot be split across "
        "nodes. Allocate the declared topology or change the experiment contract "
        "and all comparison arms together; do not fake Ray resources or silently "
        "shrink one speculative-decoding mode."
    )
