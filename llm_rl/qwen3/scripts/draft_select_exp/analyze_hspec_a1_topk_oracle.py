#!/usr/bin/env python3
"""Inventory and exact top-k oracle ceiling analysis for HSpec A1.

This tool intentionally avoids importing the whole training/runtime stack.
It loads only the local descriptor/table helper modules and can operate in
environments where ``torch`` is unavailable, because the read-side helpers
used here do not execute torch code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict, OrderedDict
from dataclasses import asdict
from pathlib import Path
import sys
import types
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD = 0.85
DEFAULT_DRAFT_HORIZON = 15


def _parse_step_spec(value: str) -> list[int]:
    steps: set[int] = set()
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if end_i < start_i:
                raise ValueError(f"invalid step range: {item!r}")
            steps.update(range(start_i, end_i + 1))
        else:
            steps.add(int(item))
    if not steps:
        raise ValueError("no target steps provided")
    return sorted(steps)


def _read_env_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_records(path_base: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Write parquet when available, always emit jsonl as a portable fallback."""
    _ensure_dir(path_base.parent)
    outputs: dict[str, str] = {}

    jsonl_path = path_base.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    outputs["jsonl"] = str(jsonl_path)

    if pd is not None:
        parquet_path = path_base.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        outputs["parquet"] = str(parquet_path)
    else:  # pragma: no cover - optional dependency
        csv_path = path_base.with_suffix(".csv")
        _write_csv(csv_path, rows)
        outputs["csv"] = str(csv_path)
    return outputs


def _install_lightweight_package_stubs(repo_root: Path) -> None:
    pkg = sys.modules.get("vllm_ascend")
    if pkg is None:
        pkg = types.ModuleType("vllm_ascend")
        pkg.__path__ = [str(repo_root / "vllm_ascend")]  # type: ignore[attr-defined]
        sys.modules["vllm_ascend"] = pkg

    spec_pkg = sys.modules.get("vllm_ascend.spec_decode")
    if spec_pkg is None:
        spec_pkg = types.ModuleType("vllm_ascend.spec_decode")
        spec_pkg.__path__ = [str(repo_root / "vllm_ascend/spec_decode")]  # type: ignore[attr-defined]
        sys.modules["vllm_ascend.spec_decode"] = spec_pkg

    torch_available = False
    try:
        torch_available = importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        torch_available = "torch" in sys.modules and getattr(sys.modules.get("torch"), "__spec__", None) is not None
    if not torch_available and "torch" not in sys.modules:
        def _noop_record_function(*args, **kwargs):
            class _NoOpContext:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _NoOpContext()

        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object  # type: ignore[attr-defined]
        torch_stub.dtype = object  # type: ignore[attr-defined]
        torch_stub.Size = tuple  # type: ignore[attr-defined]
        torch_stub.device = lambda value="cpu": str(value)  # type: ignore[attr-defined]
        torch_stub.float16 = "float16"  # type: ignore[attr-defined]
        torch_stub.float32 = "float32"  # type: ignore[attr-defined]
        torch_stub.long = "long"  # type: ignore[attr-defined]
        profiler_mod = types.ModuleType("torch.profiler")
        profiler_mod.record_function = _noop_record_function  # type: ignore[attr-defined]
        autograd_mod = types.ModuleType("torch.autograd")
        autograd_profiler_mod = types.ModuleType("torch.autograd.profiler")
        autograd_profiler_mod.record_function = _noop_record_function  # type: ignore[attr-defined]
        autograd_mod.profiler = autograd_profiler_mod  # type: ignore[attr-defined]
        torch_stub.profiler = profiler_mod  # type: ignore[attr-defined]
        torch_stub.autograd = autograd_mod  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub
        sys.modules["torch.profiler"] = profiler_mod
        sys.modules["torch.autograd"] = autograd_mod
        sys.modules["torch.autograd.profiler"] = autograd_profiler_mod


def _load_local_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_hspec_modules(repo_root: Path) -> tuple[Any, Any, Any]:
    _install_lightweight_package_stubs(repo_root)
    hspec_utils_mod = _load_local_module(
        "vllm_ascend.spec_decode.hspec_utils",
        repo_root / "vllm_ascend/spec_decode/hspec_utils.py",
    )
    hspec_store_mod = _load_local_module(
        "vllm_ascend.spec_decode.hspec_store",
        repo_root / "vllm_ascend/spec_decode/hspec_store.py",
    )
    hspec_table_store_mod = _load_local_module(
        "vllm_ascend.spec_decode.hspec_table_store",
        repo_root / "vllm_ascend/spec_decode/hspec_table_store.py",
    )
    return hspec_store_mod, hspec_table_store_mod, hspec_utils_mod


def _scan_descriptors(
    hspec_store_dir: Path,
    hspec_store_mod: Any,
    *,
    target_steps: set[int] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not hspec_store_dir.exists():
        raise FileNotFoundError(str(hspec_store_dir))

    for desc_manifest_path in sorted(hspec_store_dir.rglob("desc.jsonl")):
        segment_dir = desc_manifest_path.parent
        segment_manifest_path = segment_dir / "segment.json"
        segment_manifest: dict[str, Any] = {}
        if segment_manifest_path.exists():
            with open(segment_manifest_path, "r", encoding="utf-8") as f:
                segment_manifest = json.load(f)
        segment_status = str(segment_manifest.get("status", ""))

        with open(desc_manifest_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                desc = hspec_store_mod.coerce_hspec_desc(json.loads(line))
                step = int(desc.global_step)
                if target_steps is not None and step not in target_steps:
                    continue
                results.append(
                    {
                        "desc": desc,
                        "epoch": int(desc.epoch),
                        "global_step": step,
                        "prompt_id": str(desc.prompt_id),
                        "request_id": str(desc.request_id),
                        "external_request_id": str(getattr(desc, "external_request_id", "") or ""),
                        "shard_id": int(desc.shard_id),
                        "worker_rank": int(desc.worker_rank),
                        "tp_group_id": int(desc.tp_group_id),
                        "length": int(desc.length),
                        "hidden_dim": int(desc.hidden_dim),
                        "hs_path": str(desc.hs_path),
                        "token_path": str(desc.token_path),
                        "segment_dir": str(segment_dir),
                        "desc_manifest_path": str(desc_manifest_path),
                        "desc_manifest_line": int(line_no),
                        "segment_status": segment_status,
                        "segment_manifest_path": str(segment_manifest_path),
                        "segment_manifest": segment_manifest,
                    }
                )
    return results


def _scan_table_versions(
    hspec_table_store_dir: Path,
    hspec_table_store_mod: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not hspec_table_store_dir.exists():
        raise FileNotFoundError(str(hspec_table_store_dir))

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for shard_dir in sorted(hspec_table_store_dir.glob("shard_*")):
        if not shard_dir.is_dir():
            continue
        try:
            shard_id = int(shard_dir.name.split("_", 1)[1])
        except Exception:
            warnings.append(f"skip unparsable shard dir: {shard_dir}")
            continue

        for version_dir in sorted(shard_dir.glob("version_*")):
            if not version_dir.is_dir():
                continue
            try:
                version = int(version_dir.name.split("_", 1)[1])
            except Exception:
                warnings.append(f"skip unparsable version dir: {version_dir}")
                continue

            manifest_path = version_dir / "manifest.json"
            if not manifest_path.exists():
                warnings.append(f"missing manifest.json: {version_dir}")
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            active_index_path = version_dir / "active_prompt_index.jsonl"
            prompt_index_path = version_dir / "prompt_index.jsonl"
            active_records: dict[str, Any] = {}
            active_epoch_values: set[int] = set()
            if active_index_path.exists():
                active_records = hspec_table_store_mod.read_active_prompt_index(active_index_path)
                active_epoch_values = {
                    int(record.active_epoch) for record in active_records.values()
                }
                if len(active_epoch_values) != 1:
                    errors.append(
                        "version has non-unique active_epoch values: "
                        f"shard={shard_id} version={version} values={sorted(active_epoch_values)}"
                    )

            active_epoch = sorted(active_epoch_values)[0] if len(active_epoch_values) == 1 else None
            prompt_count = (
                len(active_records)
                if active_records
                else int(manifest.get("prompt_count", 0))
            )
            entry_count = (
                sum(int(record.desc.n_entries) for record in active_records.values())
                if active_records
                else int(manifest.get("entry_count", 0))
            )

            records.append(
                {
                    "shard_id": shard_id,
                    "version": version,
                    "active_epoch": active_epoch,
                    "prompt_count": int(prompt_count),
                    "entry_count": int(entry_count),
                    "status": str(manifest.get("status", "")),
                    "version_dir": str(version_dir),
                    "manifest_path": str(manifest_path),
                    "active_prompt_index_path": str(active_index_path) if active_index_path.exists() else "",
                    "prompt_index_path": str(prompt_index_path) if prompt_index_path.exists() else "",
                    "table_file": str(manifest.get("table_file", version_dir / "table.bin")),
                    "table_file_size": int(manifest.get("table_file_size", 0)),
                    "has_active_prompt_index": bool(active_index_path.exists()),
                }
            )

    by_shard_epoch: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        active_epoch = record.get("active_epoch")
        if active_epoch is None:
            continue
        shard_id = int(record["shard_id"])
        active_epoch_i = int(active_epoch)
        if active_epoch_i in by_shard_epoch[shard_id]:
            prev = by_shard_epoch[shard_id][active_epoch_i]
            errors.append(
                "duplicate version for shard/active_epoch: "
                f"shard={shard_id} active_epoch={active_epoch_i} "
                f"versions={[prev['version'], record['version']]}"
            )
        else:
            by_shard_epoch[shard_id][active_epoch_i] = record

    report = {
        "errors": errors,
        "warnings": warnings,
        "num_records": len(records),
    }
    return records, report


def _build_step_inventory(
    descriptors: list[dict[str, Any]],
    hspec_store_mod: Any,
    version_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_shard_epoch: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in version_records:
        active_epoch = record.get("active_epoch")
        if active_epoch is None:
            continue
        by_shard_epoch[int(record["shard_id"])][int(active_epoch)] = record

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in descriptors:
        grouped[int(item["global_step"])].append(item)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for step in sorted(grouped):
        items = grouped[step]
        epochs = sorted({int(item["epoch"]) for item in items})
        epoch_value = epochs[0] if len(epochs) == 1 else None
        if len(epochs) != 1:
            errors.append(f"step has mixed epochs: step={step} epochs={epochs}")

        prompt_ids = {str(item["prompt_id"]) for item in items if str(item["prompt_id"])}
        segment_dirs = {str(item["segment_dir"]) for item in items}
        shard_ids = sorted({int(item["shard_id"]) for item in items})
        empty_prompt_id_count = sum(1 for item in items if not str(item["prompt_id"]))

        raw_bytes_total = 0
        missing_files = 0
        for item in items:
            desc = item["desc"]
            try:
                raw_bytes_total += int(hspec_store_mod.estimate_hspec_trajectory_bytes(desc))
            except Exception:
                warnings.append(
                    f"failed to estimate raw bytes: step={step} request_id={item['request_id']}"
                )
            if not Path(item["hs_path"]).exists() or not Path(item["token_path"]).exists():
                missing_files += 1

        analysis_ready = False
        ready_versions: dict[str, int] = {}
        missing_shards: list[int] = []
        if epoch_value is not None and int(epoch_value) >= 1:
            target_active_epoch = int(epoch_value) - 1
            analysis_ready = True
            for shard_id in shard_ids:
                version_record = by_shard_epoch.get(int(shard_id), {}).get(target_active_epoch)
                if version_record is None:
                    analysis_ready = False
                    missing_shards.append(int(shard_id))
                else:
                    ready_versions[str(shard_id)] = int(version_record["version"])

        rows.append(
            {
                "epoch": epoch_value,
                "global_step": int(step),
                "desc_count": len(items),
                "prompt_count": len(prompt_ids),
                "token_count_total": int(sum(int(item["length"]) for item in items)),
                "raw_bytes_total": int(raw_bytes_total),
                "segment_count": len(segment_dirs),
                "missing_segment_files": int(missing_files),
                "empty_prompt_id_count": int(empty_prompt_id_count),
                "empty_prompt_id_ratio": (
                    float(empty_prompt_id_count) / float(len(items))
                    if items else 0.0
                ),
                "query_epoch_usable": bool(epoch_value is not None and int(epoch_value) >= 1),
                "analysis_ready": bool(analysis_ready),
                "missing_table_mapping_shards": missing_shards,
                "ready_versions_by_shard": ready_versions,
                "shard_ids": shard_ids,
            }
        )

    report = {
        "errors": errors,
        "warnings": warnings,
        "num_steps": len(rows),
    }
    return rows, report


def _version_index(records: Iterable[dict[str, Any]]) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        active_epoch = record.get("active_epoch")
        if active_epoch is None:
            continue
        result[int(record["shard_id"])][int(active_epoch)] = record
    return result


def _parse_inventory_command(args: argparse.Namespace) -> int:
    manifest = _read_env_manifest(Path(args.run_manifest) if args.run_manifest else None)
    hspec_store_dir_value = args.hspec_store_dir or manifest.get("HSPEC_STORE_DIR", "")
    hspec_table_store_dir_value = args.hspec_table_store_dir or manifest.get("HSPEC_TABLE_STORE_DIR", "")
    if not hspec_store_dir_value:
        raise ValueError("hspec store dir is required")
    if not hspec_table_store_dir_value:
        raise ValueError("hspec table store dir is required")
    hspec_store_dir = Path(hspec_store_dir_value).expanduser()
    hspec_table_store_dir = Path(hspec_table_store_dir_value).expanduser()

    hspec_store_mod, hspec_table_store_mod, _ = _load_hspec_modules(REPO_ROOT)
    descriptors = _scan_descriptors(hspec_store_dir, hspec_store_mod)
    version_records, version_report = _scan_table_versions(hspec_table_store_dir, hspec_table_store_mod)
    step_rows, step_report = _build_step_inventory(descriptors, hspec_store_mod, version_records)

    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    descriptor_rows = [
        {
            "epoch": int(item["epoch"]),
            "global_step": int(item["global_step"]),
            "prompt_id": str(item["prompt_id"]),
            "request_id": str(item["request_id"]),
            "external_request_id": str(item["external_request_id"]),
            "shard_id": int(item["shard_id"]),
            "worker_rank": int(item["worker_rank"]),
            "tp_group_id": int(item["tp_group_id"]),
            "length": int(item["length"]),
            "hidden_dim": int(item["hidden_dim"]),
            "hs_path": str(item["hs_path"]),
            "token_path": str(item["token_path"]),
            "segment_dir": str(item["segment_dir"]),
            "segment_status": str(item["segment_status"]),
            "desc_manifest_path": str(item["desc_manifest_path"]),
            "desc_manifest_line": int(item["desc_manifest_line"]),
        }
        for item in descriptors
    ]

    _write_json(out_dir / "step_inventory.json", step_rows)
    _write_csv(out_dir / "step_inventory.csv", step_rows)
    _write_json(out_dir / "table_version_catalog.json", version_records)
    _write_json(
        out_dir / "integrity_report.json",
        {
            "descriptor_count": len(descriptors),
            "version_record_count": len(version_records),
            "step_inventory_count": len(step_rows),
            "step_report": step_report,
            "version_report": version_report,
        },
    )
    _write_records(out_dir / "descriptor_inventory", descriptor_rows)

    print(f"Inventory written to {out_dir}")
    return 0


def _load_target_descriptors(
    hspec_store_dir: Path,
    hspec_store_mod: Any,
    target_steps: set[int],
) -> list[dict[str, Any]]:
    return _scan_descriptors(hspec_store_dir, hspec_store_mod, target_steps=target_steps)


def _dedupe_tp_fanout_for_step(
    items: list[dict[str, Any]],
    hspec_utils_mod: Any,
) -> list[dict[str, Any]]:
    """Keep one descriptor per (tp_group_id, external_request_id) within a step."""
    kept: dict[tuple[int, str], dict[str, Any]] = {}
    for item in items:
        external_request_id = str(
            item.get("external_request_id")
            or hspec_utils_mod.hspec_external_request_id(str(item["request_id"]))
        )
        key = (int(item["tp_group_id"]), external_request_id)
        prev = kept.get(key)
        if prev is None:
            kept[key] = item
            continue
        current_order = (int(item["worker_rank"]), str(item["request_id"]))
        prev_order = (int(prev["worker_rank"]), str(prev["request_id"]))
        if current_order < prev_order:
            kept[key] = item
    return sorted(
        kept.values(),
        key=lambda item: (
            int(item["tp_group_id"]),
            str(item.get("external_request_id") or ""),
            int(item["worker_rank"]),
            str(item["request_id"]),
        ),
    )


def _infer_num_shards(version_records: list[dict[str, Any]]) -> int:
    shard_ids = sorted({int(record["shard_id"]) for record in version_records})
    if not shard_ids:
        raise RuntimeError("failed to infer num_shards from empty table version catalog")
    return len(shard_ids)


def _load_version_catalog_from_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"version catalog must be a list: {path}")
    return data


def _load_prompt_active_index(
    version_record: dict[str, Any],
    hspec_table_store_mod: Any,
    cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (int(version_record["shard_id"]), int(version_record["version"]))
    cached = cache.get(key)
    if cached is not None:
        return cached

    active_index_path = Path(str(version_record.get("active_prompt_index_path", "")))
    if not active_index_path.exists():
        raise FileNotFoundError(
            f"missing active_prompt_index for shard={key[0]} version={key[1]}: {active_index_path}"
        )
    active_records = hspec_table_store_mod.read_active_prompt_index(active_index_path)
    prompt_descs = {
        str(prompt_id): record.desc for prompt_id, record in active_records.items()
    }
    cache[key] = prompt_descs
    return prompt_descs


class _PromptTableCache:
    def __init__(self, max_prompts: int = 64):
        self.max_prompts = max(int(max_prompts), 1)
        self._cache: OrderedDict[tuple[int, int, str], dict[str, Any]] = OrderedDict()

    def get_or_load(
        self,
        shard_id: int,
        version: int,
        prompt_id: str,
        loader,
    ) -> dict[str, Any]:
        key = (int(shard_id), int(version), str(prompt_id))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        value = loader()
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_prompts:
            self._cache.popitem(last=False)
        return value


def _prefix_match_len(a: np.ndarray | list[int], b: np.ndarray | list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def _extract_draft(table_data: dict[str, Any], entry_idx: int, horizon: int) -> np.ndarray:
    ridx = int(table_data["entry_rollout_idx"][entry_idx])
    off = int(table_data["entry_offset"][entry_idx])
    seq = table_data["rollout_seqs"][ridx]
    end = min(off + int(horizon), int(len(seq)))
    return np.ascontiguousarray(seq[off:end], dtype=np.int32)


def _exact_topk_blocked(
    z_queries: np.ndarray,
    keys: np.ndarray,
    *,
    topk: int,
    query_block_size: int,
    key_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if z_queries.ndim != 2:
        raise ValueError(f"z_queries must be 2-D, got {z_queries.shape}")
    if keys.ndim != 2:
        raise ValueError(f"keys must be 2-D, got {keys.shape}")
    if z_queries.shape[1] != keys.shape[1]:
        raise ValueError(
            f"query/key dim mismatch: queries={z_queries.shape} keys={keys.shape}"
        )
    if keys.shape[0] <= 0:
        raise ValueError("keys must have at least one entry")

    topk_eff = min(max(int(topk), 1), int(keys.shape[0]))
    num_queries = int(z_queries.shape[0])
    result_vals = np.full((num_queries, topk_eff), -np.inf, dtype=np.float32)
    result_idxs = np.full((num_queries, topk_eff), -1, dtype=np.int32)

    keys_f32 = np.ascontiguousarray(keys, dtype=np.float32)
    z_f32 = np.ascontiguousarray(z_queries, dtype=np.float32)

    for q_start in range(0, num_queries, max(int(query_block_size), 1)):
        q_stop = min(q_start + max(int(query_block_size), 1), num_queries)
        q_block = z_f32[q_start:q_stop]
        block_size = q_stop - q_start

        running_vals = np.full((block_size, topk_eff), -np.inf, dtype=np.float32)
        running_idxs = np.full((block_size, topk_eff), -1, dtype=np.int32)

        for k_start in range(0, keys_f32.shape[0], max(int(key_block_size), 1)):
            k_stop = min(k_start + max(int(key_block_size), 1), keys_f32.shape[0])
            key_block = keys_f32[k_start:k_stop]
            sims = q_block @ key_block.T

            if key_block.shape[0] > topk_eff:
                part = np.argpartition(-sims, kth=topk_eff - 1, axis=1)[:, :topk_eff]
                block_vals = np.take_along_axis(sims, part, axis=1)
                order = np.argsort(-block_vals, axis=1, kind="stable")
                block_local_idxs = np.take_along_axis(part, order, axis=1)
                block_vals = np.take_along_axis(block_vals, order, axis=1)
            else:
                block_local_idxs = np.argsort(-sims, axis=1, kind="stable")[:, :topk_eff]
                block_vals = np.take_along_axis(sims, block_local_idxs, axis=1)

            block_global_idxs = block_local_idxs.astype(np.int32) + int(k_start)
            merged_vals = np.concatenate([running_vals, block_vals.astype(np.float32)], axis=1)
            merged_idxs = np.concatenate([running_idxs, block_global_idxs.astype(np.int32)], axis=1)

            if merged_vals.shape[1] > topk_eff:
                part = np.argpartition(-merged_vals, kth=topk_eff - 1, axis=1)[:, :topk_eff]
                candidate_vals = np.take_along_axis(merged_vals, part, axis=1)
                candidate_idxs = np.take_along_axis(merged_idxs, part, axis=1)
                order = np.argsort(-candidate_vals, axis=1, kind="stable")
                running_vals = np.take_along_axis(candidate_vals, order, axis=1)
                running_idxs = np.take_along_axis(candidate_idxs, order, axis=1)
            else:
                order = np.argsort(-merged_vals, axis=1, kind="stable")
                running_vals = np.take_along_axis(merged_vals, order, axis=1)[:, :topk_eff]
                running_idxs = np.take_along_axis(merged_idxs, order, axis=1)[:, :topk_eff]

        result_vals[q_start:q_stop] = running_vals
        result_idxs[q_start:q_stop] = running_idxs

    return result_vals, result_idxs


def _target_slice(tokens: np.ndarray, token_idx: int, horizon: int) -> np.ndarray:
    start = int(token_idx) + 1
    end = min(start + int(horizon), int(tokens.shape[0]))
    return np.ascontiguousarray(tokens[start:end], dtype=np.int32)


def _summarize_subset(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_rows:
        return {
            "count": 0,
            "top1_mean_accept_len": 0.0,
            "oracle_mean_accept_len": 0.0,
            "mean_gain": 0.0,
            "win_rate": 0.0,
            "winner_rank_histogram": {},
        }
    top1_vals = [float(row["top1_accept_len"]) for row in query_rows]
    oracle_vals = [float(row["oracle_topk_accept_len"]) for row in query_rows]
    gains = [float(row["oracle_gain"]) for row in query_rows]
    wins = [1.0 if float(row["oracle_gain"]) > 0.0 else 0.0 for row in query_rows]
    hist: dict[str, int] = defaultdict(int)
    for row in query_rows:
        hist[str(int(row["oracle_best_rank"]))] += 1
    return {
        "count": len(query_rows),
        "top1_mean_accept_len": float(np.mean(top1_vals)) if top1_vals else 0.0,
        "oracle_mean_accept_len": float(np.mean(oracle_vals)) if oracle_vals else 0.0,
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "winner_rank_histogram": dict(sorted(hist.items(), key=lambda item: int(item[0]))),
    }


def _analyze_prompt_group(
    *,
    step: int,
    epoch: int,
    prompt_id: str,
    shard_id: int,
    version_record: dict[str, Any],
    desc_items: list[dict[str, Any]],
    table_desc: Any | None,
    table_cache: _PromptTableCache,
    prompt_desc_cache: dict[tuple[int, int], dict[str, Any]],
    hspec_store_mod: Any,
    hspec_table_store_mod: Any,
    topk: int,
    horizon: int,
    threshold: float,
    query_block_size: int,
    key_block_size: int,
    query_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    step_counters: dict[str, int],
) -> None:
    trajectories: list[dict[str, Any]] = []
    total_query_points = 0

    for item in desc_items:
        desc = item["desc"]
        hs, tokens = hspec_store_mod.materialize_hspec_trajectory(desc)
        hs_np = np.ascontiguousarray(np.asarray(hs, dtype=np.float32))
        tok_np = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
        if hs_np.shape[0] != tok_np.shape[0]:
            raise RuntimeError(
                f"trajectory token/hidden mismatch: request_id={desc.request_id} "
                f"hs_rows={hs_np.shape[0]} token_len={tok_np.shape[0]}"
            )
        query_count = max(int(tok_np.shape[0]) - 1, 0)
        total_query_points += query_count
        trajectories.append(
            {
                "desc": desc,
                "hs": hs_np,
                "tokens": tok_np,
                "query_count": query_count,
            }
        )

    if table_desc is None:
        for traj in trajectories:
            tokens = traj["tokens"]
            desc = traj["desc"]
            for token_idx in range(traj["query_count"]):
                query_records.append(
                    {
                        "epoch": int(epoch),
                        "global_step": int(step),
                        "prompt_id": str(prompt_id),
                        "request_id": str(desc.request_id),
                        "shard_id": int(shard_id),
                        "query_token_idx": int(token_idx),
                        "query_token_id": int(tokens[token_idx]),
                        "response_len": int(tokens.shape[0]),
                        "table_present": False,
                        "table_version": int(version_record["version"]),
                        "table_active_epoch": int(version_record["active_epoch"]),
                        "table_entry_count": 0,
                        "top1_sim": None,
                        "top2_sim": None,
                        "top1_margin": None,
                        "top1_accept_len": None,
                        "oracle_topk_accept_len": None,
                        "oracle_gain": None,
                        "oracle_best_rank": -1,
                        "oracle_best_entry_idx": -1,
                        "eligible_by_threshold": False,
                        "query_target_len": int(min(horizon, tokens.shape[0] - token_idx - 1)),
                    }
                )
        step_counters["num_query_points_total"] += total_query_points
        step_counters["num_query_points_table_miss"] += total_query_points
        return

    table_data = table_cache.get_or_load(
        int(shard_id),
        int(version_record["version"]),
        str(prompt_id),
        loader=lambda: hspec_table_store_mod.materialize_prompt_table(table_desc),
    )

    mean = np.ascontiguousarray(np.asarray(table_data["mean"], dtype=np.float32))
    components = np.ascontiguousarray(np.asarray(table_data["components"], dtype=np.float32))
    keys = np.ascontiguousarray(np.asarray(table_data["keys"], dtype=np.float32))
    rewards = table_data.get("rewards")
    if rewards is not None:
        rewards = np.ascontiguousarray(np.asarray(rewards, dtype=np.float32))

    all_queries: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    for traj in trajectories:
        hs_np = traj["hs"]
        tokens = traj["tokens"]
        desc = traj["desc"]
        for token_idx in range(traj["query_count"]):
            all_queries.append(hs_np[token_idx])
            all_meta.append(
                {
                    "request_id": str(desc.request_id),
                    "external_request_id": str(
                        getattr(desc, "external_request_id", "")
                        or hspec_utils_mod.hspec_external_request_id(str(desc.request_id))
                    ),
                    "tp_group_id": int(getattr(desc, "tp_group_id", 0)),
                    "query_token_idx": int(token_idx),
                    "query_token_id": int(tokens[token_idx]),
                    "response_len": int(tokens.shape[0]),
                    "target": _target_slice(tokens, token_idx, horizon),
                }
            )

    if not all_queries:
        return

    q_matrix = np.ascontiguousarray(np.stack(all_queries, axis=0), dtype=np.float32)
    z_queries = np.ascontiguousarray((q_matrix - mean) @ components.T, dtype=np.float32)
    topk_vals, topk_idxs = _exact_topk_blocked(
        z_queries,
        keys,
        topk=topk,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )

    step_counters["num_query_points_total"] += len(all_meta)
    step_counters["num_query_points_with_table"] += len(all_meta)

    for q_idx, meta in enumerate(all_meta):
        target = meta["target"]
        sims = topk_vals[q_idx]
        idxs = topk_idxs[q_idx]

        top1_sim = float(sims[0])
        top2_sim = float(sims[1]) if sims.shape[0] > 1 else None
        top1_margin = top1_sim - top2_sim if top2_sim is not None else None
        eligible = bool(top1_sim >= threshold)

        top1_accept_len: int | None = None
        oracle_best_rank = -1
        oracle_best_entry_idx = -1
        oracle_best_accept = -1
        oracle_best_sim = -math.inf
        candidate_start = len(candidate_records)

        for rank_idx, (sim_value, entry_idx) in enumerate(zip(sims.tolist(), idxs.tolist()), start=1):
            entry_i = int(entry_idx)
            if entry_i < 0:
                continue
            draft = _extract_draft(table_data, entry_i, horizon)
            accept_len = _prefix_match_len(draft, target)
            entry_reward = float(rewards[entry_i]) if rewards is not None and entry_i < len(rewards) else math.nan
            entry_rollout_idx = int(table_data["entry_rollout_idx"][entry_i])
            entry_offset = int(table_data["entry_offset"][entry_i])
            candidate_records.append(
                {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "prompt_id": str(prompt_id),
                    "request_id": str(meta["request_id"]),
                    "external_request_id": str(meta["external_request_id"]),
                    "shard_id": int(shard_id),
                    "table_version": int(version_record["version"]),
                    "table_active_epoch": int(version_record["active_epoch"]),
                    "tp_group_id": int(meta["tp_group_id"]),
                    "query_token_idx": int(meta["query_token_idx"]),
                    "candidate_rank_by_sim": int(rank_idx),
                    "candidate_entry_idx": entry_i,
                    "candidate_sim": float(sim_value),
                    "candidate_accept_len": int(accept_len),
                    "candidate_reward": None if math.isnan(entry_reward) else entry_reward,
                    "candidate_rollout_idx": entry_rollout_idx,
                    "candidate_entry_offset": entry_offset,
                    "candidate_tail_len": int(len(table_data["rollout_seqs"][entry_rollout_idx]) - entry_offset),
                    "is_top1": bool(rank_idx == 1),
                    "is_oracle_best": False,
                }
            )
            if rank_idx == 1:
                top1_accept_len = int(accept_len)
            if (
                int(accept_len) > oracle_best_accept
                or (
                    int(accept_len) == oracle_best_accept
                    and float(sim_value) > oracle_best_sim
                )
            ):
                oracle_best_accept = int(accept_len)
                oracle_best_sim = float(sim_value)
                oracle_best_rank = int(rank_idx)
                oracle_best_entry_idx = entry_i

        if top1_accept_len is None:
            raise RuntimeError(
                f"top-1 accept length missing for step={step} prompt={prompt_id} query={q_idx}"
            )

        for cand_idx in range(candidate_start, len(candidate_records)):
            cand = candidate_records[cand_idx]
            if (
                int(cand["global_step"]) == int(step)
                and str(cand["prompt_id"]) == str(prompt_id)
                and str(cand["request_id"]) == str(meta["request_id"])
                and int(cand["query_token_idx"]) == int(meta["query_token_idx"])
                and int(cand["candidate_entry_idx"]) == int(oracle_best_entry_idx)
                and int(cand["candidate_rank_by_sim"]) == int(oracle_best_rank)
            ):
                cand["is_oracle_best"] = True
                break

        step_counters["num_query_points_usable"] += 1
        if eligible:
            step_counters["num_query_points_spec_eligible"] += 1

        query_records.append(
            {
                "epoch": int(epoch),
                "global_step": int(step),
                "prompt_id": str(prompt_id),
                    "request_id": str(meta["request_id"]),
                    "external_request_id": str(meta["external_request_id"]),
                    "shard_id": int(shard_id),
                    "tp_group_id": int(meta["tp_group_id"]),
                    "query_token_idx": int(meta["query_token_idx"]),
                "query_token_id": int(meta["query_token_id"]),
                "response_len": int(meta["response_len"]),
                "table_present": True,
                "table_version": int(version_record["version"]),
                "table_active_epoch": int(version_record["active_epoch"]),
                "table_entry_count": int(keys.shape[0]),
                "top1_sim": top1_sim,
                "top2_sim": top2_sim,
                "top1_margin": top1_margin,
                "top1_accept_len": int(top1_accept_len),
                "oracle_topk_accept_len": int(oracle_best_accept),
                "oracle_gain": int(oracle_best_accept - top1_accept_len),
                "oracle_best_rank": int(oracle_best_rank),
                "oracle_best_entry_idx": int(oracle_best_entry_idx),
                "eligible_by_threshold": eligible,
                "query_target_len": int(len(target)),
            }
        )


def _analyze_command(args: argparse.Namespace) -> int:
    manifest = _read_env_manifest(Path(args.run_manifest) if args.run_manifest else None)
    hspec_store_dir_value = args.hspec_store_dir or manifest.get("HSPEC_STORE_DIR", "")
    hspec_table_store_dir_value = args.hspec_table_store_dir or manifest.get("HSPEC_TABLE_STORE_DIR", "")
    if not hspec_store_dir_value:
        raise ValueError("hspec store dir is required")
    if not hspec_table_store_dir_value:
        raise ValueError("hspec table store dir is required")

    target_steps = set(_parse_step_spec(args.target_steps))
    threshold = float(
        args.similarity_threshold
        if args.similarity_threshold is not None
        else manifest.get("A1_HSPEC_SIMILARITY_THRESHOLD", DEFAULT_THRESHOLD)
    )
    horizon = int(
        args.draft_horizon
        if args.draft_horizon is not None
        else manifest.get("A1_HSPEC_NUM_SPECULATIVE_TOKENS", DEFAULT_DRAFT_HORIZON)
    )

    hspec_store_mod, hspec_table_store_mod, hspec_utils_mod = _load_hspec_modules(REPO_ROOT)
    descriptors = _load_target_descriptors(
        Path(hspec_store_dir_value).expanduser(),
        hspec_store_mod,
        target_steps,
    )
    version_records = (
        _load_version_catalog_from_json(Path(args.table_version_catalog))
        if args.table_version_catalog
        else _scan_table_versions(Path(hspec_table_store_dir_value).expanduser(), hspec_table_store_mod)[0]
    )
    version_map = _version_index(version_records)
    num_shards = _infer_num_shards(version_records)

    grouped_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in descriptors:
        grouped_steps[int(item["global_step"])].append(item)

    missing_steps = sorted(step for step in target_steps if step not in grouped_steps)
    if missing_steps:
        raise RuntimeError(f"target steps not found in raw descriptors: {missing_steps}")

    query_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    prompt_desc_cache: dict[tuple[int, int], dict[str, Any]] = {}
    table_cache = _PromptTableCache(max_prompts=int(args.table_cache_prompts))

    for step in sorted(target_steps):
        items = grouped_steps[step]
        epochs = sorted({int(item["epoch"]) for item in items})
        if len(epochs) != 1:
            raise RuntimeError(f"step has mixed epochs: step={step} epochs={epochs}")
        epoch = int(epochs[0])
        if epoch < 1:
            raise RuntimeError(f"step={step} belongs to epoch 0 and is not eligible for A1")
        empty_prompt_ids = sum(1 for item in items if not str(item["prompt_id"]))
        if empty_prompt_ids:
            raise RuntimeError(
                f"step={step}: {empty_prompt_ids}/{len(items)} descriptors have empty prompt_id. "
                "Raw-store seal mapping is broken; refusing silent skip. "
                "See HSpec_reaserch_doc/optim_draft_select/exp/hspec_a1_prompt_id_disk_fix_guide.md"
            )

        deduped_items = _dedupe_tp_fanout_for_step(items, hspec_utils_mod)
        for item in deduped_items:
            prompt_id = str(item["prompt_id"])
            expected_shard = int(hspec_utils_mod.stable_partition_id(prompt_id, num_shards))
            if int(item["shard_id"]) != expected_shard:
                raise RuntimeError(
                    "descriptor shard mismatch after prompt_id repair: "
                    f"step={step} request_id={item['request_id']} prompt_id={prompt_id} "
                    f"desc.shard_id={item['shard_id']} expected_shard={expected_shard} "
                    f"num_shards={num_shards}"
                )

        target_active_epoch = epoch - 1
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in deduped_items:
            prompt_id = str(item["prompt_id"])
            by_prompt[prompt_id].append(item)

        step_counters = {
            "num_desc_scanned": len(items),
            "num_desc_after_tp_dedupe": len(deduped_items),
            "num_desc_analyzed": len(deduped_items),
            "num_prompts_analyzed": len(by_prompt),
            "num_query_points_total": 0,
            "num_query_points_usable": 0,
            "num_query_points_with_table": 0,
            "num_query_points_table_miss": 0,
            "num_query_points_spec_eligible": 0,
        }

        for prompt_id, prompt_items in sorted(by_prompt.items()):
            shard_ids = sorted({int(item["shard_id"]) for item in prompt_items})
            if len(shard_ids) != 1:
                raise RuntimeError(
                    f"prompt spans multiple shards in one step: step={step} prompt_id={prompt_id} shards={shard_ids}"
                )
            shard_id = int(shard_ids[0])
            version_record = version_map.get(shard_id, {}).get(target_active_epoch)
            if version_record is None:
                raise RuntimeError(
                    "missing table version mapping for target step: "
                    f"step={step} epoch={epoch} shard={shard_id} target_active_epoch={target_active_epoch}"
                )
            prompt_descs = _load_prompt_active_index(version_record, hspec_table_store_mod, prompt_desc_cache)
            table_desc = prompt_descs.get(prompt_id)
            _analyze_prompt_group(
                step=step,
                epoch=epoch,
                prompt_id=prompt_id,
                shard_id=shard_id,
                version_record=version_record,
                desc_items=prompt_items,
                table_desc=table_desc,
                table_cache=table_cache,
                prompt_desc_cache=prompt_desc_cache,
                hspec_store_mod=hspec_store_mod,
                hspec_table_store_mod=hspec_table_store_mod,
                topk=int(args.topk),
                horizon=horizon,
                threshold=threshold,
                query_block_size=int(args.query_block_size),
                key_block_size=int(args.key_block_size),
                query_records=query_records,
                candidate_records=candidate_records,
                step_counters=step_counters,
            )

        step_query_rows = [row for row in query_records if int(row["global_step"]) == step]
        usable_rows = [row for row in step_query_rows if bool(row["table_present"])]
        eligible_rows = [row for row in usable_rows if bool(row["eligible_by_threshold"])]
        step_rows.append(
            {
                "epoch": epoch,
                "global_step": step,
                **step_counters,
                "top1_mean_accept_len_all": _summarize_subset(usable_rows)["top1_mean_accept_len"],
                "oracle_mean_accept_len_all": _summarize_subset(usable_rows)["oracle_mean_accept_len"],
                "oracle_mean_gain_all": _summarize_subset(usable_rows)["mean_gain"],
                "oracle_win_rate_all": _summarize_subset(usable_rows)["win_rate"],
                "top1_mean_accept_len_spec_eligible": _summarize_subset(eligible_rows)["top1_mean_accept_len"],
                "oracle_mean_accept_len_spec_eligible": _summarize_subset(eligible_rows)["oracle_mean_accept_len"],
                "oracle_mean_gain_spec_eligible": _summarize_subset(eligible_rows)["mean_gain"],
                "oracle_win_rate_spec_eligible": _summarize_subset(eligible_rows)["win_rate"],
            }
        )

    usable_query_rows = [row for row in query_records if bool(row["table_present"])]
    eligible_query_rows = [row for row in usable_query_rows if bool(row["eligible_by_threshold"])]
    summary = {
        "num_steps_analyzed": len(target_steps),
        "num_desc_scanned": int(sum(int(row["num_desc_scanned"]) for row in step_rows)),
        "num_desc_after_tp_dedupe": int(
            sum(int(row["num_desc_after_tp_dedupe"]) for row in step_rows)
        ),
        "num_desc_analyzed": int(sum(int(row["num_desc_analyzed"]) for row in step_rows)),
        "num_query_points_total": int(len(query_records)),
        "num_query_points_usable": int(len(usable_query_rows)),
        "num_query_points_with_table": int(len(usable_query_rows)),
        "num_query_points_spec_eligible": int(len(eligible_query_rows)),
        "analysis_config": {
            "topk": int(args.topk),
            "draft_horizon": int(horizon),
            "similarity_threshold": float(threshold),
            "query_block_size": int(args.query_block_size),
            "key_block_size": int(args.key_block_size),
            "table_cache_prompts": int(args.table_cache_prompts),
            "target_steps": sorted(target_steps),
        },
        "all_query_points": _summarize_subset(usable_query_rows),
        "spec_eligible_query_points": _summarize_subset(eligible_query_rows),
    }

    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "step_summary.csv", step_rows)
    query_outputs = _write_records(out_dir / "query_records", query_records)
    candidate_outputs = _write_records(out_dir / "candidate_records", candidate_records)
    _write_json(
        out_dir / "artifact_paths.json",
        {
            "query_records": query_outputs,
            "candidate_records": candidate_outputs,
        },
    )

    print(f"A1 analysis written to {out_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="scan descriptor and table-store inventories")
    inventory.add_argument("--run-manifest", type=str, default=None, help="optional .env manifest from A1 collect run")
    inventory.add_argument("--hspec-store-dir", type=str, default=None, help="descriptor raw store root")
    inventory.add_argument("--hspec-table-store-dir", type=str, default=None, help="table store root")
    inventory.add_argument("--out-dir", type=str, required=True, help="directory to write inventory artifacts")

    analyze = subparsers.add_parser("analyze", help="run exact top-k oracle ceiling analysis")
    analyze.add_argument("--run-manifest", type=str, default=None, help="optional .env manifest from A1 collect run")
    analyze.add_argument("--hspec-store-dir", type=str, default=None, help="descriptor raw store root")
    analyze.add_argument("--hspec-table-store-dir", type=str, default=None, help="table store root")
    analyze.add_argument("--table-version-catalog", type=str, default=None, help="optional JSON catalog from inventory")
    analyze.add_argument("--target-steps", type=str, required=True, help="comma list or ranges, e.g. 12,13,20-25")
    analyze.add_argument("--topk", type=int, default=8, help="exact top-k to compute")
    analyze.add_argument("--draft-horizon", type=int, default=None, help="draft horizon H; defaults from run manifest or 15")
    analyze.add_argument("--similarity-threshold", type=float, default=None, help="eligibility threshold; defaults from run manifest or 0.85")
    analyze.add_argument("--query-block-size", type=int, default=256, help="number of query rows per blocked similarity pass")
    analyze.add_argument("--key-block-size", type=int, default=8192, help="number of keys per blocked similarity pass")
    analyze.add_argument("--table-cache-prompts", type=int, default=64, help="max materialized prompt tables to keep in the LRU")
    analyze.add_argument("--out-dir", type=str, required=True, help="directory to write analysis artifacts")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "inventory":
        return _parse_inventory_command(args)
    if args.command == "analyze":
        return _analyze_command(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
