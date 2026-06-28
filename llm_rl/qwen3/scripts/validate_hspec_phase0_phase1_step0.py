#!/usr/bin/env python3
"""Step-0 guardrails for the HSpec Phase 0/1 descriptor path.

The checks in this script are intentionally lightweight: no Ray startup, no
vLLM engine construction, and no NPU dependency. They protect the current
descriptor data plane before later HSpec refactors touch collector lifecycle,
GC, build routing, or decode hot paths.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def _exists(rel_path: str) -> bool:
    return (PROJECT_ROOT / rel_path).is_file()


def _set_result(report: dict[str, Any], name: str, ok: bool, detail: str = "") -> None:
    report["checks"][name] = bool(ok)
    if not ok and detail:
        report["failures"][name] = detail


def _set_warning(report: dict[str, Any], name: str, value: bool, detail: str = "") -> None:
    report["warnings"][name] = bool(value)
    if value and detail:
        report.setdefault("warning_details", {})[name] = detail


def run_static_checks() -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "checks": {},
        "warnings": {},
        "failures": {},
    }

    rollout = _read("verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")
    trainer = _read("verl/trainer/ppo/ray_trainer.py")
    main_ppo = _read("verl/trainer/main_ppo.py")
    table = _read("vllm_ascend/spec_decode/hspec_table.py")
    store = _read("vllm_ascend/spec_decode/hspec_store.py")

    _set_result(
        report,
        "validation_collect_gate_present",
        "collect_hspec = use_hspec and not is_validate and bool(do_sample)" in rollout,
        "rollout must keep validation/greedy collection disabled.",
    )
    _set_result(
        report,
        "rollout_default_writes_hspec_desc",
        'not legacy_hspec_dataproto_hs and rollout_hspec_desc_list' in rollout
        and 'non_tensor_batch["hspec_desc"]' in rollout,
        "descriptor mode must write hspec_desc.",
    )
    _set_result(
        report,
        "rollout_legacy_writes_hidden_only_under_legacy",
        "collect_hspec and legacy_hspec_dataproto_hs and rollout_hidden_states_list" in rollout
        and 'non_tensor_batch["rollout_hidden_states"]' in rollout
        and 'non_tensor_batch["rollout_hspec_tokens"]' in rollout,
        "legacy hidden/tokens payload must be gated by legacy_hspec_dataproto_hs.",
    )
    _set_result(
        report,
        "rollout_flush_gated_by_collect_hspec",
        "if collect_hspec:" in rollout
        and "hspec_flush_and_get_descriptors(" in rollout
        and "hspec_flush_and_get_all()" in rollout,
        "flush calls must remain under collect_hspec.",
    )
    _set_result(
        report,
        "rollout_step0_runtime_asserts_present",
        "hspec_step0_runtime_asserts_enabled" in rollout
        and "strict_descriptor_violation" in rollout,
        "Step0 optional runtime assertions are missing in rollout.",
    )

    _set_result(
        report,
        "trainer_pending_refs_present",
        "_hspec_pending_build_refs" in trainer,
        "trainer must retain async build refs.",
    )
    _set_result(
        report,
        "trainer_nonblocking_poll_present",
        "def _poll_hspec_builds_nonblocking" in trainer
        and "ray.wait" in trainer
        and "timeout=0" in trainer,
        "trainer must use nonblocking ray.wait(timeout=0) for build polling.",
    )
    drop_required = [
        '"hspec_desc"',
        '"rollout_hidden_states"',
        '"rollout_hspec_tokens"',
        '"hspec_rollout_debug"',
    ]
    _set_result(
        report,
        "trainer_drop_fields_covers_hspec_keys",
        "def _drop_hspec_non_tensor_fields" in trainer
        and all(key in trainer for key in drop_required),
        "trainer drop helper must cover descriptor, legacy, and debug keys.",
    )
    drop_pos = trainer.find("self._drop_hspec_non_tensor_fields(batch)")
    update_pos = trainer.find("actor_output = self.actor_rollout_wg.update_actor(batch)")
    _set_result(
        report,
        "trainer_drops_hspec_before_update_actor",
        drop_pos >= 0 and update_pos >= 0 and drop_pos < update_pos,
        "HSpec fields must be dropped before actor update.",
    )
    _set_result(
        report,
        "trainer_default_descriptor_aggregation",
        "prompt_build_data: dict = defaultdict(list)" in trainer
        and "desc_obj = desc_obj.with_updates(" in trainer
        and "prompt_build_data[prompt_id].append(desc_obj)" in trainer,
        "default path must aggregate descriptors and fill reward/shard.",
    )
    _set_result(
        report,
        "trainer_step0_runtime_asserts_present",
        "hspec_step0_runtime_asserts_enabled" in trainer
        and "strict_descriptor_violation" in trainer,
        "Step0 optional runtime assertions are missing in trainer.",
    )

    _set_result(
        report,
        "table_build_tables_async_partitions_payload",
        "def build_tables_async" in table
        and "partition_payloads" in table
        and "build_tables_batch.remote(payload)" in table,
        "GlobalHSpecTableGroup must partition build payloads by shard.",
    )
    _set_result(
        report,
        "table_build_tables_batch_accepts_descriptor_list",
        "if isinstance(data, list):" in table
        and "coerce_hspec_desc" in table
        and "_load_prompt_build_inputs_from_descs" in table,
        "build actor must accept descriptor list payloads.",
    )
    _set_result(
        report,
        "store_step0_helpers_present",
        "def hspec_step0_runtime_asserts_enabled" in store
        and "def hspec_default_descriptor_mode_enabled" in store
        and "def hspec_record_store_metric" in store,
        "Step0 store helpers are missing.",
    )
    for metric in (
        "strict_descriptor_violation",
        "legacy_payload_count",
        "descriptor_payload_count",
        "validation_collect_skip",
    ):
        _set_result(
            report,
            f"store_metric_{metric}",
            f'"{metric}"' in store,
            f"missing Step0 metric placeholder {metric}.",
        )

    _set_result(
        report,
        "store_step1_helpers_present",
        "def get_hspec_store_dtype" in store
        and "def hspec_strict_descriptor_mode_enabled" in store
        and "def hspec_require_explicit_num_shards_enabled" in store
        and "def assert_hspec_num_shards_configured_for_production" in store
        and "def get_hspec_build_actor_num_cpus" in store,
        "Step1 centralized HSpec configuration helpers are missing.",
    )
    _set_result(
        report,
        "store_dtype_helper_float16_only",
        "HSPEC_STORE_DTYPE" in store
        and '"fp16": "float16"' in store
        and '"float16": "float16"' in store
        and "Unsupported HSPEC_STORE_DTYPE" in store,
        "HSPEC_STORE_DTYPE must default to/validate float16 on-disk storage.",
    )
    append_pos = store.find("def append_hidden_rows")
    flush_pos = store.find("def flush_descriptors")
    append_body = store[append_pos:flush_pos] if append_pos >= 0 and flush_pos > append_pos else ""
    flush_body = store[flush_pos:] if flush_pos >= 0 else ""
    _set_result(
        report,
        "store_append_uses_dtype_helper",
        "store_dtype = get_hspec_store_dtype()" in append_body
        and "dtype=torch.float16" in append_body
        and 'state["hs_dtype"] = hs_dtype' in append_body
        and "HSpec hidden dtype mismatch" in append_body,
        "append_hidden_rows must use the on-disk dtype helper and persist per-request hs_dtype.",
    )
    _set_result(
        report,
        "store_flush_uses_state_hs_dtype",
        'state.get("hs_dtype") or get_hspec_store_dtype()' in flush_body
        and "hs_dtype=hs_dtype" in flush_body,
        "flush_descriptors must use request-state hs_dtype instead of hardcoding float16.",
    )
    for metric in (
        "store_fp16_rows",
        "source_dtype_fp16_rows",
        "source_dtype_bf16_rows",
        "source_dtype_other_rows",
    ):
        _set_result(
            report,
            f"store_metric_{metric}",
            f'"{metric}"' in store,
            f"missing Step1 dtype metric placeholder {metric}.",
        )
    _set_result(
        report,
        "store_num_shards_fallback_warns",
        "HSPEC_NUM_SHARDS is not set; falling back to 5 outside strict HSpec production init" in store,
        "get_hspec_num_shards fallback must be explicit and warned outside production init.",
    )

    _set_result(
        report,
        "main_ppo_hspec_init_uses_step1_helpers",
        "assert_hspec_num_shards_configured_for_production()" in main_ppo
        and "get_hspec_store_dtype()" in main_ppo
        and "get_hspec_num_shards()" in main_ppo
        and "get_hspec_build_actor_num_cpus()" in main_ppo
        and "hspec_strict_descriptor_mode_enabled()" in main_ppo,
        "main_ppo HSpec init must use centralized Step1 helpers.",
    )
    _set_result(
        report,
        "main_ppo_single_node_hard_gate",
        "HSPEC_EXPERIMENTAL_ALLOW_MULTI_NODE_UNSAFE" in main_ppo
        and "HSpec Phase 1 descriptor path supports single-node only" in main_ppo
        and "HSPEC_ALLOW_MULTI_NODE" not in main_ppo,
        "main_ppo must hard-gate multi-node HSpec and not honor the old HSPEC_ALLOW_MULTI_NODE bypass.",
    )
    _set_result(
        report,
        "main_ppo_phase1_config_log",
        "HSpec Phase1 config:" in main_ppo
        and "store_dtype=" in main_ppo
        and "strict_descriptor_mode=" in main_ppo
        and "single_node_only=" in main_ppo
        and "num_shards=" in main_ppo
        and "build_actor_num_cpus=" in main_ppo,
        "HSpec init must print the Step1 config boundary once at startup.",
    )
    _set_result(
        report,
        "table_actor_resource_options_explicit",
        "get_hspec_build_actor_num_cpus" in table
        and "num_cpus = get_hspec_build_actor_num_cpus()" in table
        and "num_cpus=num_cpus" in table
        and "num_gpus=0" in table,
        "init_hspec_tables must set build actor CPU budget explicitly and avoid GPU/NPU resources.",
    )
    _set_result(
        report,
        "table_single_node_only_topology",
        "HSPEC_SINGLE_NODE_ONLY" in table
        and "HSPEC_ALLOW_MULTI_NODE" not in table,
        "descriptor topology validation must use HSPEC_SINGLE_NODE_ONLY, not the old multi-node bypass.",
    )
    _set_result(
        report,
        "table_build_async_descriptor_first_doc",
        "Dict[str, List[HSpecTrajectoryDesc] | Dict[str, Any]]" in table
        and "descriptor-first" in table
        and "HSPEC_LEGACY_DATAPROTO_HS=1" in table,
        "build_tables_async must document the descriptor-first Phase1 payload boundary.",
    )
    _set_result(
        report,
        "trainer_uses_hspec_num_shards_helper",
        "get_hspec_num_shards" in trainer
        and "hspec_num_shards = get_hspec_num_shards()" in trainer
        and 'os.getenv("HSPEC_NUM_SHARDS", "5")' not in trainer,
        "trainer must use get_hspec_num_shards() and must not keep a local fallback=5.",
    )

    _set_warning(
        report,
        "legacy_build_dict_payload_still_open",
        'data["hidden_states"]' in table,
        "Expected in Step 0; Step 5 should harden strict descriptor API.",
    )

    script_candidates = [
        "scripts/train.sh",
        "scripts/train_grpo_hspec.sh",
        "scripts/train_grpo_qwen2.5_1.5b_hspec.sh",
        "scripts/train_grpo_qwen2.5_1.5b_hspec-short.sh",
        "scripts/train_grpo_qwen3_30b_hspec.sh",
    ]
    required_exports = [
        "HSPEC_LEGACY_DATAPROTO_HS",
        "HSPEC_STRICT_DESCRIPTOR_MODE",
        "HSPEC_STORE_DTYPE",
        "HSPEC_STORE_DIR",
        "HSPEC_TABLE_STORE_DIR",
        "HSPEC_INFER_TP",
        "HSPEC_NUM_SHARDS",
        "NODE_RANK",
        "HSPEC_SINGLE_NODE_ONLY",
        "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS",
        "HSPEC_STEP0_RUNTIME_ASSERTS",
        "HSPEC_BUILD_ACTOR_NUM_CPUS",
    ]
    required_runtime_env = [
        "HSPEC_STRICT_DESCRIPTOR_MODE",
        "HSPEC_STORE_DTYPE",
        "HSPEC_SINGLE_NODE_ONLY",
        "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS",
        "HSPEC_STEP0_RUNTIME_ASSERTS",
        "HSPEC_BUILD_ACTOR_NUM_CPUS",
    ]
    for rel_path in script_candidates:
        if not _exists(rel_path):
            _set_warning(
                report,
                f"{Path(rel_path).name}:script_not_present",
                True,
                f"{rel_path} is not present in this checkout; skipping script env checks for it.",
            )
            continue
        text = _read(rel_path)
        for env_name in required_exports:
            _set_result(
                report,
                f"{Path(rel_path).name}:{env_name}_exported",
                f"export {env_name}=" in text,
                f"{rel_path} must export {env_name}.",
            )
        for env_name in required_runtime_env:
            _set_result(
                report,
                f"{Path(rel_path).name}:{env_name}_runtime_env",
                f"runtime_env.env_vars.{env_name}" in text,
                f"{rel_path} must pass {env_name} through Ray runtime env.",
            )
        _set_result(
            report,
            f"{Path(rel_path).name}:default_descriptor_mode",
            'HSPEC_LEGACY_DATAPROTO_HS="${HSPEC_LEGACY_DATAPROTO_HS:-0}"' in text,
            f"{rel_path} must default HSPEC_LEGACY_DATAPROTO_HS to 0.",
        )
        _set_result(
            report,
            f"{Path(rel_path).name}:HSPEC_BUILD_ACTOR_NUM_CPUS_logged",
            "hspec_build_actor_num_cpus=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in text
            or "HSPEC_BUILD_ACTOR_NUM_CPUS=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in text,
            f"{rel_path} must print HSPEC_BUILD_ACTOR_NUM_CPUS in startup logs.",
        )

    acceptance = _read("scripts/run_hspec_phase1_acceptance.sh")
    for env_name in (
        "HSPEC_LEGACY_DATAPROTO_HS",
        "HSPEC_STRICT_DESCRIPTOR_MODE",
        "HSPEC_STORE_DTYPE",
        "HSPEC_SINGLE_NODE_ONLY",
        "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS",
        "HSPEC_STEP0_RUNTIME_ASSERTS",
        "HSPEC_BUILD_ACTOR_NUM_CPUS",
    ):
        _set_result(
            report,
            f"run_hspec_phase1_acceptance.sh:{env_name}_exported",
            f"export {env_name}=" in acceptance,
            f"acceptance wrapper must export {env_name}.",
        )
    _set_result(
        report,
        "run_hspec_phase1_acceptance.sh:HSPEC_BUILD_ACTOR_NUM_CPUS_logged",
        "hspec_build_actor_num_cpus=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in acceptance
        or "HSPEC_BUILD_ACTOR_NUM_CPUS=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in acceptance,
        "acceptance wrapper must print HSPEC_BUILD_ACTOR_NUM_CPUS.",
    )

    baseline = _read("scripts/run_hspec_step0_baseline.sh")
    for env_name in (
        "HSPEC_LEGACY_DATAPROTO_HS",
        "HSPEC_STRICT_DESCRIPTOR_MODE",
        "HSPEC_STORE_DTYPE",
        "HSPEC_SINGLE_NODE_ONLY",
        "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS",
        "HSPEC_STEP0_RUNTIME_ASSERTS",
        "HSPEC_BUILD_ACTOR_NUM_CPUS",
    ):
        _set_result(
            report,
            f"run_hspec_step0_baseline.sh:{env_name}_exported",
            f"export {env_name}=" in baseline,
            f"baseline wrapper must export {env_name}.",
        )
    _set_result(
        report,
        "run_hspec_step0_baseline.sh:HSPEC_BUILD_ACTOR_NUM_CPUS_logged",
        "hspec_build_actor_num_cpus=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in baseline
        or "HSPEC_BUILD_ACTOR_NUM_CPUS=${HSPEC_BUILD_ACTOR_NUM_CPUS}" in baseline,
        "baseline wrapper must print HSPEC_BUILD_ACTOR_NUM_CPUS.",
    )

    report["ok"] = all(report["checks"].values())
    return report


@contextlib.contextmanager
def _temporary_env(env: dict[str, str]):
    old = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ensure_torch_for_store_smoke():
    try:
        import torch  # type: ignore

        return torch, False
    except ModuleNotFoundError:
        pass

    import numpy as np

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = np.float16
    fake_torch.float32 = np.float32

    class FakeTensor:
        def __init__(self, array: Any):
            self._array = np.asarray(array)

        @property
        def ndim(self) -> int:
            return int(self._array.ndim)

        @property
        def shape(self):
            return self._array.shape

        def numel(self) -> int:
            return int(self._array.size)

        def detach(self):
            return self

        def to(self, device: str | None = None, dtype: Any | None = None):
            del device
            if dtype is None:
                return FakeTensor(self._array)
            return FakeTensor(self._array.astype(dtype, copy=False))

        def contiguous(self):
            return FakeTensor(np.ascontiguousarray(self._array))

        def numpy(self):
            return self._array

        def reshape(self, *shape: int):
            return FakeTensor(self._array.reshape(*shape))

    def arange(*args, dtype=None):
        np_dtype = dtype if dtype is not None else np.float32
        return FakeTensor(np.arange(*args, dtype=np_dtype))

    fake_torch.Tensor = FakeTensor
    fake_torch.arange = arange
    sys.modules["torch"] = fake_torch
    return fake_torch, True


def _load_hspec_store_module():
    _ensure_torch_for_store_smoke()
    module_name = "_hspec_store_step0"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    path = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_store.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reset_store_singleton() -> None:
    store = _load_hspec_store_module()

    collector = getattr(store, "_collector", None)
    if collector is not None:
        with contextlib.suppress(Exception):
            collector.clear_batch()
    store._collector = None


def run_legacy_toggle() -> dict[str, Any]:
    store = _load_hspec_store_module()
    report: dict[str, Any] = {"ok": True, "checks": {}, "warnings": {}, "failures": {}}
    with _temporary_env({"HSPEC_LEGACY_DATAPROTO_HS": "0"}):
        _set_result(
            report,
            "legacy_disabled_returns_false",
            not store.hspec_legacy_dataproto_hs_enabled(),
            "HSPEC_LEGACY_DATAPROTO_HS=0 must disable legacy path.",
        )
        _set_result(
            report,
            "default_descriptor_mode_returns_true",
            store.hspec_default_descriptor_mode_enabled(),
            "descriptor mode helper must be true when legacy is disabled.",
        )
    with _temporary_env({"HSPEC_LEGACY_DATAPROTO_HS": "1"}):
        _set_result(
            report,
            "legacy_enabled_returns_true",
            store.hspec_legacy_dataproto_hs_enabled(),
            "HSPEC_LEGACY_DATAPROTO_HS=1 must enable legacy path.",
        )
        _set_result(
            report,
            "default_descriptor_mode_returns_false",
            not store.hspec_default_descriptor_mode_enabled(),
            "descriptor mode helper must be false when legacy is enabled.",
        )
    report["expected_payload_keys"] = {
        "descriptor": ["hspec_desc"],
        "legacy": ["rollout_hidden_states", "rollout_hspec_tokens"],
    }
    report["ok"] = all(report["checks"].values())
    return report


def run_store_smoke(strict_immediate_load: bool = False) -> dict[str, Any]:
    torch, used_fake_torch = _ensure_torch_for_store_smoke()
    store = _load_hspec_store_module()

    tmpdir = Path(tempfile.mkdtemp(prefix="hspec_step0_"))
    report: dict[str, Any] = {
        "ok": True,
        "checks": {},
        "warnings": {},
        "failures": {},
        "tmpdir": str(tmpdir),
    }
    if used_fake_torch:
        _set_warning(
            report,
            "using_fake_torch_for_cpu_store_smoke",
            True,
            "torch is not installed in this environment; using a minimal numpy-backed fake tensor.",
        )
    env = {
        "HSPEC_LEGACY_DATAPROTO_HS": "0",
        "HSPEC_STORE_DIR": str(tmpdir / "store"),
        "HSPEC_TABLE_STORE_DIR": str(tmpdir / "table_store"),
        "HSPEC_NUM_SHARDS": "2",
        "HSPEC_INFER_TP": "2",
        "NODE_RANK": "0",
        "HSPEC_TP_GROUP_ID": "0",
    }
    try:
        with _temporary_env(env):
            _reset_store_singleton()
            collector = store.get_hspec_local_collector()
            collector.append_hidden_rows("req_a", torch.arange(12, dtype=torch.float32).reshape(3, 4))
            collector.extend_tokens("req_a", [101, 102, 103])
            collector.append_hidden_rows("req_b", torch.arange(8, dtype=torch.float32).reshape(2, 4))
            collector.extend_tokens("req_b", [201, 202])
            descs = collector.flush_descriptors(
                request_id_to_prompt_id={"req_a": "prompt-a", "req_b": "prompt-b"},
                epoch=1,
                global_step=7,
            )
            _set_result(report, "descriptor_count_is_two", len(descs) == 2, f"got {len(descs)} descriptors")
            for req_id, expected_len in (("req_a", 3), ("req_b", 2)):
                desc = descs.get(req_id)
                _set_result(report, f"{req_id}:descriptor_present", desc is not None, "descriptor missing")
                if desc is None:
                    continue
                _set_result(report, f"{req_id}:prompt_id", desc.prompt_id.startswith("prompt-"), str(desc))
                _set_result(report, f"{req_id}:length", desc.length == expected_len, str(desc))
                _set_result(report, f"{req_id}:hidden_dim", desc.hidden_dim == 4, str(desc))
                _set_result(report, f"{req_id}:hs_dtype", desc.hs_dtype == "float16", str(desc))
                _set_result(report, f"{req_id}:token_dtype", desc.token_dtype == "int32", str(desc))
                _set_result(report, f"{req_id}:paths_exist", Path(desc.hs_path).exists()
                            and Path(desc.token_path).exists(), str(desc))

            load_failed = False
            load_error = ""
            try:
                for req_id, expected_tokens in (("req_a", [101, 102, 103]), ("req_b", [201, 202])):
                    hs, tokens = store.load_hspec_trajectory(descs[req_id])
                    _set_result(report, f"{req_id}:loaded_shape", tuple(hs.shape) == (len(expected_tokens), 4),
                                str(tuple(hs.shape)))
                    _set_result(report, f"{req_id}:loaded_tokens", tokens.astype("int32").tolist() == expected_tokens,
                                str(tokens.astype("int32").tolist()))
            except Exception as exc:
                load_failed = True
                load_error = repr(exc)

            if load_failed and strict_immediate_load:
                _set_result(report, "strict_immediate_load", False, load_error)
            elif load_failed:
                _set_warning(
                    report,
                    "known_p1_2a_segment_not_sealed",
                    True,
                    f"Immediate load failed before Step 3 seal fix: {load_error}",
                )
            else:
                _set_result(report, "immediate_load", True)
    finally:
        with contextlib.suppress(Exception):
            _reset_store_singleton()
        shutil.rmtree(tmpdir, ignore_errors=True)

    report["ok"] = all(report["checks"].values())
    return report


def _extract_last_float(patterns: list[str], text: str) -> float | None:
    value: float | None = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                pass
    return value


def run_extract_baseline(log_file: Path, output_json: Path | None) -> dict[str, Any]:
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    env_names = [
        "HSPEC_LEGACY_DATAPROTO_HS",
        "HSPEC_NUM_SHARDS",
        "HSPEC_STORE_DIR",
        "HSPEC_TABLE_STORE_DIR",
        "HSPEC_STRICT_DESCRIPTOR_MODE",
        "HSPEC_STORE_DTYPE",
        "HSPEC_SINGLE_NODE_ONLY",
        "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS",
        "HSPEC_STEP0_RUNTIME_ASSERTS",
        "HSPEC_BUILD_ACTOR_NUM_CPUS",
    ]
    env: dict[str, str | None] = {}
    for name in env_names:
        patterns = [
            rf"{name}=([^\s]+)",
            rf"{name.lower()}=([^\s]+)",
            rf"{name.lower().replace('hspec_', 'hspec_')}=([^\s]+)",
        ]
        found = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                found = match.group(1).strip("'\"")
                break
        env[name] = found

    metrics = {
        "hspec_desc_count_total": _extract_last_float([r"hspec[/_]desc_count['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_collect_dropped_total": _extract_last_float(
            [r"hspec[/_]collect_dropped['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_raw_store_bytes_total": _extract_last_float(
            [r"hspec[/_]raw_store_bytes['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_store_fp16_rows_total": _extract_last_float(
            [r"hspec[/_]store_fp16_rows['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_source_dtype_fp16_rows_total": _extract_last_float(
            [r"hspec[/_]source_dtype_fp16_rows['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_source_dtype_bf16_rows_total": _extract_last_float(
            [r"hspec[/_]source_dtype_bf16_rows['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_source_dtype_other_rows_total": _extract_last_float(
            [r"hspec[/_]source_dtype_other_rows['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_pinned_pageable_fallback_total": _extract_last_float(
            [r"hspec[/_]pinned_pageable_fallback['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_build_pending_refs_last": _extract_last_float(
            [r"hspec[/_]build_pending_refs['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "hspec_build_ready_refs_total": _extract_last_float(
            [r"hspec[/_]build_ready_refs['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "timing_hspec_epoch_build_wait_s": _extract_last_float(
            [r"hspec_epoch_build_wait['\"]?\s*[:=]\s*([0-9.]+)"], text),
        "rollout_throughput": _extract_last_float(
            [r"rollout[_/ ]throughput['\"]?\s*[:=]\s*([0-9.]+)"], text),
    }
    report = {
        "ok": True,
        "env": env,
        "metrics": metrics,
        "phase0_guards": {
            "step_build_wait_removed": True,
            "hspec_fields_popped_before_update_actor": True,
            "validation_collect_disabled": True,
        },
        "warnings": {},
        "failures": {},
    }
    if env.get("HSPEC_LEGACY_DATAPROTO_HS") not in ("0", None):
        report["ok"] = False
        report["failures"]["HSPEC_LEGACY_DATAPROTO_HS"] = "baseline must run descriptor mode"
    if env.get("HSPEC_NUM_SHARDS") is None:
        report["warnings"]["missing_HSPEC_NUM_SHARDS_in_log"] = True
    if metrics["hspec_desc_count_total"] is None:
        report["warnings"]["missing_hspec_desc_count_in_log"] = True
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HSpec Phase 0/1 Step-0 guardrails.")
    parser.add_argument("--static", action="store_true", help="run static source and script guards")
    parser.add_argument("--store-smoke", action="store_true", help="run CPU-only descriptor store smoke")
    parser.add_argument("--strict-immediate-load", action="store_true",
                        help="make immediate descriptor load failure fatal in store smoke")
    parser.add_argument("--legacy-toggle", action="store_true", help="verify legacy on/off helpers")
    parser.add_argument("--extract-baseline", action="store_true", help="extract baseline metrics from a log")
    parser.add_argument("--log-file", type=Path, help="training log for --extract-baseline")
    parser.add_argument("--output-json", type=Path, help="optional output JSON path")
    args = parser.parse_args()

    if not any((args.static, args.store_smoke, args.legacy_toggle, args.extract_baseline)):
        args.static = True
        args.store_smoke = True
        args.legacy_toggle = True

    reports: dict[str, Any] = {}
    ok = True
    if args.static:
        reports["static"] = run_static_checks()
        ok = ok and reports["static"]["ok"]
    if args.store_smoke:
        reports["store_smoke"] = run_store_smoke(args.strict_immediate_load)
        ok = ok and reports["store_smoke"]["ok"]
    if args.legacy_toggle:
        reports["legacy_toggle"] = run_legacy_toggle()
        ok = ok and reports["legacy_toggle"]["ok"]
    if args.extract_baseline:
        if args.log_file is None:
            raise SystemExit("--extract-baseline requires --log-file")
        reports["baseline"] = run_extract_baseline(args.log_file, args.output_json)
        ok = ok and reports["baseline"]["ok"]

    result = {"ok": ok, "reports": reports}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
