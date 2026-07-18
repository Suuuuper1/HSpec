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

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:  # pragma: no cover - optional dependency
    import torch
except Exception:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD = 0.85
DEFAULT_DRAFT_HORIZON = 15
DEFAULT_COMPUTE_BACKEND = "auto"
DEFAULT_TORCH_DEVICE = "auto"


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


def _parse_positive_int_spec(value: str | int) -> list[int]:
    values: set[int] = set()
    if isinstance(value, int):
        if int(value) <= 0:
            raise ValueError(f"topk must be > 0, got {value}")
        return [int(value)]
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if start_i <= 0 or end_i <= 0:
                raise ValueError(f"topk range must be positive: {item!r}")
            if end_i < start_i:
                raise ValueError(f"invalid topk range: {item!r}")
            values.update(range(start_i, end_i + 1))
        else:
            current = int(item)
            if current <= 0:
                raise ValueError(f"topk values must be > 0, got {current}")
            values.add(current)
    if not values:
        raise ValueError("no topk values provided")
    return sorted(values)


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


def _resolve_compute_backend(
    backend: str,
    torch_device: str,
) -> tuple[str, str | None]:
    backend_norm = str(backend).strip().lower()
    device_norm = str(torch_device).strip().lower()
    if backend_norm not in {"auto", "numpy", "torch"}:
        raise ValueError(f"unsupported compute backend: {backend!r}")
    if device_norm not in {"auto", "cpu", "cuda", "npu"}:
        raise ValueError(f"unsupported torch device: {torch_device!r}")

    if backend_norm == "numpy":
        return ("numpy", None)

    torch_mod = torch
    if torch_mod is None:
        if backend_norm == "torch":
            raise RuntimeError("torch backend requested but torch is not importable")
        return ("numpy", None)

    chosen_device = "cpu"
    if device_norm == "auto":
        has_npu = bool(hasattr(torch_mod, "npu") and getattr(torch_mod.npu, "is_available", lambda: False)())
        has_cuda = bool(getattr(torch_mod.cuda, "is_available", lambda: False)())
        if has_npu:
            chosen_device = "npu"
        elif has_cuda:
            chosen_device = "cuda"
        else:
            chosen_device = "cpu"
    elif device_norm == "cuda":
        if not bool(getattr(torch_mod.cuda, "is_available", lambda: False)()):
            raise RuntimeError("torch cuda device requested but not available")
        chosen_device = "cuda"
    elif device_norm == "npu":
        if not bool(hasattr(torch_mod, "npu") and getattr(torch_mod.npu, "is_available", lambda: False)()):
            raise RuntimeError("torch npu device requested but not available")
        chosen_device = "npu"
    else:
        chosen_device = "cpu"

    return ("torch", chosen_device)


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


def _prepare_table_data(table_data: dict[str, Any]) -> dict[str, Any]:
    if "_rollout_seq_lens" not in table_data:
        table_data["_rollout_seq_lens"] = np.ascontiguousarray(
            [len(seq) for seq in table_data["rollout_seqs"]],
            dtype=np.int32,
        )
    if "_entry_rollout_idx_i32" not in table_data:
        table_data["_entry_rollout_idx_i32"] = np.ascontiguousarray(
            np.asarray(table_data["entry_rollout_idx"], dtype=np.int32)
        )
    if "_entry_offset_i32" not in table_data:
        table_data["_entry_offset_i32"] = np.ascontiguousarray(
            np.asarray(table_data["entry_offset"], dtype=np.int32)
        )
    return table_data


def _extract_draft_batch(
    table_data: dict[str, Any],
    entry_indices: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table_data = _prepare_table_data(table_data)
    entry_idx_arr = np.ascontiguousarray(np.asarray(entry_indices, dtype=np.int32))
    num_entries = int(entry_idx_arr.shape[0])
    horizon_i = max(int(horizon), 0)
    draft_matrix = np.full((num_entries, horizon_i), -1, dtype=np.int32)
    draft_lens = np.zeros((num_entries,), dtype=np.int32)
    rollout_idx = table_data["_entry_rollout_idx_i32"][entry_idx_arr]
    entry_offsets = table_data["_entry_offset_i32"][entry_idx_arr]
    rollout_lens = table_data["_rollout_seq_lens"][rollout_idx]
    tail_lens = np.maximum(rollout_lens - entry_offsets, 0).astype(np.int32, copy=False)
    for row, (ridx, off, tail_len) in enumerate(zip(rollout_idx.tolist(), entry_offsets.tolist(), tail_lens.tolist(), strict=True)):
        take = min(int(tail_len), horizon_i)
        if take <= 0:
            continue
        seq = table_data["rollout_seqs"][int(ridx)]
        draft_matrix[row, :take] = np.asarray(seq[int(off):int(off) + take], dtype=np.int32)
        draft_lens[row] = int(take)
    return draft_matrix, draft_lens, rollout_idx, entry_offsets, tail_lens


def _prefix_match_lens_batch(
    draft_matrix: np.ndarray,
    draft_lens: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    draft_mat = np.ascontiguousarray(np.asarray(draft_matrix, dtype=np.int32))
    lens = np.ascontiguousarray(np.asarray(draft_lens, dtype=np.int32))
    target_np = np.ascontiguousarray(np.asarray(target, dtype=np.int32))
    if draft_mat.ndim != 2:
        raise ValueError(f"draft_matrix must be 2-D, got {draft_mat.shape}")
    if lens.ndim != 1 or lens.shape[0] != draft_mat.shape[0]:
        raise ValueError(
            f"draft_lens shape mismatch: matrix={draft_mat.shape} lens={lens.shape}"
        )
    if draft_mat.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    horizon = int(draft_mat.shape[1])
    target_len = min(int(target_np.shape[0]), horizon)
    if target_len <= 0:
        return np.zeros((draft_mat.shape[0],), dtype=np.int32)
    target_pad = np.full((horizon,), -2, dtype=np.int32)
    target_pad[:target_len] = target_np[:target_len]
    compare_len = np.minimum(lens, target_len)
    positions = np.arange(horizon, dtype=np.int32)[None, :]
    valid_mask = positions < compare_len[:, None]
    mismatch = valid_mask & (draft_mat != target_pad[None, :])
    has_mismatch = mismatch.any(axis=1)
    first_mismatch = mismatch.argmax(axis=1).astype(np.int32, copy=False)
    return np.where(has_mismatch, first_mismatch, compare_len).astype(np.int32, copy=False)


def _exact_topk_blocked_numpy(
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


def _exact_topk_blocked_torch(
    z_queries: np.ndarray,
    keys: np.ndarray,
    *,
    topk: int,
    query_block_size: int,
    key_block_size: int,
    torch_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("torch backend selected but torch is unavailable")
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

    device = torch.device(str(torch_device))
    keys_t = torch.as_tensor(np.ascontiguousarray(keys, dtype=np.float32), dtype=torch.float32, device=device)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)

    with torch.inference_mode():
        for q_start in range(0, num_queries, max(int(query_block_size), 1)):
            q_stop = min(q_start + max(int(query_block_size), 1), num_queries)
            q_block_np = np.ascontiguousarray(z_queries[q_start:q_stop], dtype=np.float32)
            q_block = torch.as_tensor(q_block_np, dtype=torch.float32, device=device)
            block_size = q_stop - q_start

            running_vals = torch.full((block_size, topk_eff), neg_inf, dtype=torch.float32, device=device)
            running_idxs = torch.full((block_size, topk_eff), -1, dtype=torch.int64, device=device)

            for k_start in range(0, int(keys_t.shape[0]), max(int(key_block_size), 1)):
                k_stop = min(k_start + max(int(key_block_size), 1), int(keys_t.shape[0]))
                key_block = keys_t[k_start:k_stop]
                sims = q_block @ key_block.transpose(0, 1)
                block_k = min(topk_eff, int(key_block.shape[0]))
                block_vals, block_local_idxs = torch.topk(sims, k=block_k, dim=1, largest=True, sorted=True)
                block_global_idxs = block_local_idxs.to(dtype=torch.int64) + int(k_start)
                merged_vals = torch.cat([running_vals, block_vals], dim=1)
                merged_idxs = torch.cat([running_idxs, block_global_idxs], dim=1)
                running_vals, select = torch.topk(
                    merged_vals,
                    k=topk_eff,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                running_idxs = torch.gather(merged_idxs, 1, select)

            result_vals[q_start:q_stop] = running_vals.cpu().numpy().astype(np.float32, copy=False)
            result_idxs[q_start:q_stop] = running_idxs.cpu().numpy().astype(np.int32, copy=False)

    return result_vals, result_idxs


def _exact_topk_blocked(
    z_queries: np.ndarray,
    keys: np.ndarray,
    *,
    topk: int,
    query_block_size: int,
    key_block_size: int,
    compute_backend: str = DEFAULT_COMPUTE_BACKEND,
    torch_device: str = DEFAULT_TORCH_DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    backend_mode, resolved_device = _resolve_compute_backend(compute_backend, torch_device)
    if backend_mode == "torch":
        return _exact_topk_blocked_torch(
            z_queries,
            keys,
            topk=topk,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
            torch_device=str(resolved_device),
        )
    return _exact_topk_blocked_numpy(
        z_queries,
        keys,
        topk=topk,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )


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


def _group_query_rows_by_topk(query_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[int(row["topk_requested"])].append(row)
    return grouped


def _build_topk_sweep_summary(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_query_rows_by_topk(query_rows)
    return {
        str(topk): _summarize_subset(rows)
        for topk, rows in sorted(grouped.items())
    }


def _write_topk_plot_a1(out_dir: Path, summary_by_topk: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if plt is None:  # pragma: no cover - optional dependency
        outputs["warning"] = "matplotlib is unavailable; plot was skipped"
        return outputs

    topk_values = sorted(int(key) for key in summary_by_topk.keys())
    if not topk_values:
        outputs["warning"] = "no topk values to plot"
        return outputs

    def _series(universe: str, metric: str) -> list[float]:
        return [
            float(summary_by_topk[str(topk)][universe][metric])
            for topk in topk_values
        ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    universe_specs = [
        ("all_query_points", "All Query Points", axes[0]),
        ("spec_eligible_query_points", "Spec Eligible", axes[1]),
    ]
    for universe_key, title, row_axes in universe_specs:
        row_axes[0].plot(topk_values, _series(universe_key, "top1_mean_accept_len"), marker="o", label="top1_mean_accept")
        row_axes[0].plot(topk_values, _series(universe_key, "oracle_mean_accept_len"), marker="o", label="oracle_mean_accept")
        row_axes[0].set_title(f"{title}: Accept Length")
        row_axes[0].set_xlabel("topk")
        row_axes[0].set_ylabel("accept len")
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend()

        row_axes[1].plot(topk_values, _series(universe_key, "mean_gain"), marker="o", label="mean_gain")
        row_axes[1].plot(topk_values, _series(universe_key, "win_rate"), marker="o", label="win_rate")
        row_axes[1].set_title(f"{title}: Gain / Win Rate")
        row_axes[1].set_xlabel("topk")
        row_axes[1].grid(True, alpha=0.3)
        row_axes[1].legend()

    plot_path = out_dir / "topk_sweep_a1.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    outputs["png"] = str(plot_path)
    return outputs


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
    topk_values: list[int],
    horizon: int,
    threshold: float,
    query_block_size: int,
    key_block_size: int,
    compute_backend: str,
    torch_device: str,
    query_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    step_counters: dict[str, int],
) -> None:
    max_topk = max(int(value) for value in topk_values)
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
                external_request_id = str(
                    getattr(desc, "external_request_id", "")
                    or hspec_store_mod.hspec_external_request_id(str(desc.request_id))
                )
                for topk_requested in topk_values:
                    query_records.append(
                        {
                            "epoch": int(epoch),
                            "global_step": int(step),
                            "topk_requested": int(topk_requested),
                            "prompt_id": str(prompt_id),
                            "request_id": str(desc.request_id),
                            "external_request_id": external_request_id,
                            "shard_id": int(shard_id),
                            "tp_group_id": int(getattr(desc, "tp_group_id", 0)),
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
                            "analysis_universe_main": False,
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
    table_data = _prepare_table_data(table_data)

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
        topk=max_topk,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
        compute_backend=compute_backend,
        torch_device=torch_device,
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
        candidate_start = len(candidate_records)
        valid_mask = np.asarray(idxs >= 0, dtype=np.bool_)
        candidate_entry_indices = np.ascontiguousarray(idxs[valid_mask], dtype=np.int32)
        candidate_sims = np.ascontiguousarray(sims[valid_mask], dtype=np.float32)

        if candidate_entry_indices.size == 0:
            raise RuntimeError(
                f"no valid candidates for step={step} prompt={prompt_id} query={q_idx}"
            )

        draft_matrix, draft_lens, rollout_idx_arr, entry_offsets_arr, tail_lens_arr = _extract_draft_batch(
            table_data,
            candidate_entry_indices,
            horizon,
        )
        candidate_accept_lens = _prefix_match_lens_batch(draft_matrix, draft_lens, target)
        prefix_best_accept = np.empty((candidate_entry_indices.shape[0],), dtype=np.int32)
        prefix_best_rank = np.empty((candidate_entry_indices.shape[0],), dtype=np.int32)
        prefix_best_entry_idx = np.empty((candidate_entry_indices.shape[0],), dtype=np.int32)
        running_best_accept = -1
        running_best_rank = -1
        running_best_entry = -1
        running_best_sim = -math.inf

        for rank_idx, entry_i in enumerate(candidate_entry_indices.tolist(), start=1):
            accept_len = int(candidate_accept_lens[rank_idx - 1])
            sim_value = float(candidate_sims[rank_idx - 1])
            entry_reward = (
                float(rewards[entry_i])
                if rewards is not None and entry_i < len(rewards)
                else math.nan
            )
            candidate_records.append(
                {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "topk_max_requested": int(max_topk),
                    "prompt_id": str(prompt_id),
                    "request_id": str(meta["request_id"]),
                    "external_request_id": str(meta["external_request_id"]),
                    "shard_id": int(shard_id),
                    "table_version": int(version_record["version"]),
                    "table_active_epoch": int(version_record["active_epoch"]),
                    "tp_group_id": int(meta["tp_group_id"]),
                    "query_token_idx": int(meta["query_token_idx"]),
                    "candidate_rank_by_sim": int(rank_idx),
                    "candidate_entry_idx": int(entry_i),
                    "candidate_sim": sim_value,
                    "candidate_accept_len": accept_len,
                    "candidate_reward": None if math.isnan(entry_reward) else float(entry_reward),
                    "candidate_rollout_idx": int(rollout_idx_arr[rank_idx - 1]),
                    "candidate_entry_offset": int(entry_offsets_arr[rank_idx - 1]),
                    "candidate_tail_len": int(tail_lens_arr[rank_idx - 1]),
                    "is_top1": bool(rank_idx == 1),
                    "is_oracle_best": False,
                }
            )
            if rank_idx == 1:
                top1_accept_len = accept_len
            if (
                accept_len > running_best_accept
                or (accept_len == running_best_accept and sim_value > running_best_sim)
            ):
                running_best_accept = accept_len
                running_best_rank = rank_idx
                running_best_entry = int(entry_i)
                running_best_sim = sim_value
            prefix_best_accept[rank_idx - 1] = int(running_best_accept)
            prefix_best_rank[rank_idx - 1] = int(running_best_rank)
            prefix_best_entry_idx[rank_idx - 1] = int(running_best_entry)

        if top1_accept_len is None:
            raise RuntimeError(
                f"top-1 accept length missing for step={step} prompt={prompt_id} query={q_idx}"
            )

        step_counters["num_query_points_usable"] += 1
        if eligible:
            step_counters["num_query_points_spec_eligible"] += 1

        for topk_requested in topk_values:
            eff_k = min(int(topk_requested), int(candidate_entry_indices.shape[0]))
            eff_idx = eff_k - 1
            oracle_best_accept = int(prefix_best_accept[eff_idx])
            oracle_best_rank = int(prefix_best_rank[eff_idx])
            oracle_best_entry_idx = int(prefix_best_entry_idx[eff_idx])
            if int(topk_requested) == int(max_topk):
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
            query_row = {
                "epoch": int(epoch),
                "global_step": int(step),
                "topk_requested": int(topk_requested),
                "topk_effective": int(eff_k),
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
                "top1_sim": float(top1_sim),
                "top2_sim": float(top2_sim) if top2_sim is not None else None,
                "top1_margin": float(top1_margin) if top1_margin is not None else None,
                "top1_accept_len": int(top1_accept_len),
                "oracle_topk_accept_len": int(oracle_best_accept),
                "oracle_gain": int(oracle_best_accept - top1_accept_len),
                "oracle_best_rank": int(oracle_best_rank),
                "oracle_best_entry_idx": int(oracle_best_entry_idx),
                "eligible_by_threshold": bool(eligible),
                "analysis_universe_main": bool(eligible),
                "query_target_len": int(len(target)),
            }
            query_records.append(query_row)


def _analyze_command(args: argparse.Namespace) -> int:
    manifest = _read_env_manifest(Path(args.run_manifest) if args.run_manifest else None)
    hspec_store_dir_value = args.hspec_store_dir or manifest.get("HSPEC_STORE_DIR", "")
    hspec_table_store_dir_value = args.hspec_table_store_dir or manifest.get("HSPEC_TABLE_STORE_DIR", "")
    if not hspec_store_dir_value:
        raise ValueError("hspec store dir is required")
    if not hspec_table_store_dir_value:
        raise ValueError("hspec table store dir is required")

    target_steps = set(_parse_step_spec(args.target_steps))
    topk_values = _parse_positive_int_spec(args.topk)
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
                topk_values=topk_values,
                horizon=horizon,
                threshold=threshold,
                query_block_size=int(args.query_block_size),
                key_block_size=int(args.key_block_size),
                compute_backend=str(args.compute_backend),
                torch_device=str(args.torch_device),
                query_records=query_records,
                candidate_records=candidate_records,
                step_counters=step_counters,
            )

        step_query_rows = [row for row in query_records if int(row["global_step"]) == step]
        for topk_requested in topk_values:
            step_topk_rows = [
                row for row in step_query_rows
                if int(row["topk_requested"]) == int(topk_requested)
            ]
            usable_rows = [row for row in step_topk_rows if bool(row["table_present"])]
            eligible_rows = [row for row in usable_rows if bool(row["eligible_by_threshold"])]
            step_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "topk_requested": int(topk_requested),
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
    summary_by_topk: dict[str, Any] = {}
    for topk_requested in topk_values:
        topk_all_rows = [
            row for row in usable_query_rows
            if int(row["topk_requested"]) == int(topk_requested)
        ]
        topk_eligible_rows = [
            row for row in eligible_query_rows
            if int(row["topk_requested"]) == int(topk_requested)
        ]
        summary_by_topk[str(int(topk_requested))] = {
            "all_query_points": _summarize_subset(topk_all_rows),
            "spec_eligible_query_points": _summarize_subset(topk_eligible_rows),
        }

    unique_step_rows = [
        row for row in step_rows
        if int(row["topk_requested"]) == int(topk_values[0])
    ]

    summary = {
        "topk_values": topk_values,
        "num_steps_analyzed": len(target_steps),
        "num_desc_scanned": int(sum(int(row["num_desc_scanned"]) for row in unique_step_rows)),
        "num_desc_after_tp_dedupe": int(sum(int(row["num_desc_after_tp_dedupe"]) for row in unique_step_rows)),
        "num_desc_analyzed": int(sum(int(row["num_desc_analyzed"]) for row in unique_step_rows)),
        "num_query_points_total": int(sum(int(row["num_query_points_total"]) for row in unique_step_rows)),
        "num_query_points_usable": int(sum(int(row["num_query_points_usable"]) for row in unique_step_rows)),
        "num_query_points_with_table": int(sum(int(row["num_query_points_with_table"]) for row in unique_step_rows)),
        "num_query_points_spec_eligible": int(sum(int(row["num_query_points_spec_eligible"]) for row in unique_step_rows)),
        "analysis_config": {
            "topk_values": topk_values,
            "draft_horizon": int(horizon),
            "similarity_threshold": float(threshold),
            "query_block_size": int(args.query_block_size),
            "key_block_size": int(args.key_block_size),
            "table_cache_prompts": int(args.table_cache_prompts),
            "compute_backend": str(args.compute_backend),
            "torch_device": str(args.torch_device),
            "target_steps": sorted(target_steps),
        },
        "by_topk": summary_by_topk,
    }

    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)
    _write_json(out_dir / "summary_by_topk.json", summary)
    _write_csv(out_dir / "step_summary_by_topk.csv", step_rows)
    query_outputs = _write_records(out_dir / "query_records_by_topk", query_records)
    candidate_outputs = _write_records(out_dir / "candidate_records_max_topk", candidate_records)
    plot_outputs = _write_topk_plot_a1(out_dir, summary_by_topk)
    _write_json(
        out_dir / "artifact_paths.json",
        {
            "query_records_by_topk": query_outputs,
            "candidate_records_max_topk": candidate_outputs,
            "plots": plot_outputs,
        },
    )

    if len(topk_values) == 1:
        topk_key = str(int(topk_values[0]))
        legacy_summary = {
            "num_steps_analyzed": int(summary["num_steps_analyzed"]),
            "num_desc_scanned": int(summary["num_desc_scanned"]),
            "num_desc_after_tp_dedupe": int(summary["num_desc_after_tp_dedupe"]),
            "num_desc_analyzed": int(summary["num_desc_analyzed"]),
            "num_query_points_total": int(summary["num_query_points_total"]),
            "num_query_points_usable": int(summary["num_query_points_usable"]),
            "num_query_points_with_table": int(summary["num_query_points_with_table"]),
            "num_query_points_spec_eligible": int(summary["num_query_points_spec_eligible"]),
            "analysis_config": summary["analysis_config"],
            "all_query_points": summary_by_topk[topk_key]["all_query_points"],
            "spec_eligible_query_points": summary_by_topk[topk_key]["spec_eligible_query_points"],
        }
        _write_json(out_dir / "summary.json", legacy_summary)
        _write_csv(out_dir / "step_summary.csv", [row for row in step_rows if int(row["topk_requested"]) == int(topk_values[0])])
        _write_records(
            out_dir / "query_records",
            [row for row in query_records if int(row["topk_requested"]) == int(topk_values[0])],
        )
        _write_records(out_dir / "candidate_records", candidate_records)

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
    analyze.add_argument("--topk", type=str, default="8", help="one or more positive integers, e.g. 8 or 2,4,8")
    analyze.add_argument("--draft-horizon", type=int, default=None, help="draft horizon H; defaults from run manifest or 15")
    analyze.add_argument("--similarity-threshold", type=float, default=None, help="eligibility threshold; defaults from run manifest or 0.85")
    analyze.add_argument("--query-block-size", type=int, default=256, help="number of query rows per blocked similarity pass")
    analyze.add_argument("--key-block-size", type=int, default=8192, help="number of keys per blocked similarity pass")
    analyze.add_argument("--table-cache-prompts", type=int, default=64, help="max materialized prompt tables to keep in the LRU")
    analyze.add_argument(
        "--compute-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default=DEFAULT_COMPUTE_BACKEND,
        help="backend for blocked exact top-k retrieval",
    )
    analyze.add_argument(
        "--torch-device",
        type=str,
        choices=("auto", "cpu", "cuda", "npu"),
        default=DEFAULT_TORCH_DEVICE,
        help="device used when compute-backend=torch or auto resolves to torch",
    )
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
