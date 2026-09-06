from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.vllm_rollout.cudagraph_config import (
    resolve_vllm_cudagraph_kwargs,
)
from verl.workers.rollout.vllm_rollout.speculative_config import (
    resolve_rollout_speculation,
)


PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
BASELINE = SCRIPTS / "run_qwen3_baseline_eager_tq2.sh"
MATRIX = {
    "run_qwen3_dflash_eager_tq2.sh": ("dflash", "eager", "NONE", "True"),
    "run_qwen3_dflash_piecewise_tq2.sh": ("dflash", "piecewise", "PIECEWISE", "False"),
    "run_qwen3_dflash_full_graph_decode_tq2.sh": (
        "dflash",
        "full_graph_decode",
        "FULL_DECODE_ONLY",
        "False",
    ),
    "run_qwen3_dspark_eager_tq2.sh": ("dspark", "eager", "NONE", "True"),
    "run_qwen3_dspark_piecewise_tq2.sh": ("dspark", "piecewise", "PIECEWISE", "False"),
    "run_qwen3_dspark_full_graph_decode_tq2.sh": (
        "dspark",
        "full_graph_decode",
        "FULL_DECODE_ONLY",
        "False",
    ),
}
ALLOWED_COMMAND_DIFFERENCES = {
    "actor_rollout_ref.rollout.mode",
    "actor_rollout_ref.rollout.pipeline_model_parallel_size",
    "actor_rollout_ref.rollout.data_parallel_size",
    "actor_rollout_ref.rollout.vllm_data_parallel_size",
    "actor_rollout_ref.rollout.enforce_eager",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.speculative_method",
    "actor_rollout_ref.rollout.speculative_model",
    "actor_rollout_ref.rollout.num_speculative_tokens",
    "actor_rollout_ref.rollout.draft_tensor_parallel_size",
    "actor_rollout_ref.rollout.draft_sample_method",
    "actor_rollout_ref.rollout.draft_probability_max_memory_mb",
    "actor_rollout_ref.rollout.draft_load_format",
    "actor_rollout_ref.rollout.rejection_sample_method",
    "actor_rollout_ref.rollout.speculative_enforce_eager",
    "actor_rollout_ref.rollout.parallel_draft_profile_enabled",
    "actor_rollout_ref.rollout.parallel_draft_allow_target_eager_experiment",
    "actor_rollout_ref.rollout.cudagraph_mode",
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DP_SIZE",
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_GRAPH_MODE",
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE",
    "+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_IF_BASE_PORT",
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_CUDAGRAPH_MODE",
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_STRICT_FULL_GRAPH",
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "+trainer.rollout_data_dir",
    "+trainer.rollout_length_dir",
}


def dry_run(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VLLM_", "VERL_"))
    }
    completed = subprocess.run(
        ["bash", str(path), "dry-run"],
        cwd=PROJECT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    command = shlex.split(lines[-1])
    assert command[:3] == ["python3", "-m", "verl.trainer.main_ppo"]
    args = {}
    for token in command[3:]:
        if "=" in token:
            key, value = token.split("=", 1)
            assert key not in args, f"duplicate Hydra key: {key}"
            args[key] = value
    headers = {}
    for line in lines[:-1]:
        if "=" in line:
            key, value = line.split("=", 1)
            headers[key] = value
    return headers, args


@pytest.fixture(scope="module")
def baseline_args():
    return dry_run(BASELINE)[1]


@pytest.mark.parametrize("script", sorted(MATRIX))
def test_each_entrypoint_has_exact_method_and_graph_contract(script, baseline_args):
    method, label, graph_mode, target_eager = MATRIX[script]
    headers, args = dry_run(SCRIPTS / script)
    assert headers["baseline_script"] == str(BASELINE)
    assert headers["method"] == method
    assert headers["target_mode_label"] == label
    assert headers["framework_cudagraph_mode"] == graph_mode
    assert headers["target_enforce_eager"] == target_eager
    assert headers["draft_enforce_eager"] == "True"
    assert headers["vllm_dp"] == "16"
    assert headers["rollout_dispatch_dp"] == "1"
    assert headers["gpu_memory_utilization"] == "0.72"
    assert headers["baseline_gpu_memory_utilization"] == "0.83"
    assert args["actor_rollout_ref.rollout.speculative_method"] == method
    assert args["actor_rollout_ref.rollout.enforce_eager"] == target_eager
    assert args["actor_rollout_ref.rollout.speculative_enforce_eager"] == "True"
    assert args["actor_rollout_ref.rollout.parallel_draft_profile_enabled"] == "False"
    if graph_mode == "NONE":
        assert "actor_rollout_ref.rollout.cudagraph_mode" not in args
        assert args["actor_rollout_ref.rollout.parallel_draft_allow_target_eager_experiment"] == "True"
    else:
        assert args["actor_rollout_ref.rollout.cudagraph_mode"] == graph_mode

    baseline_common = {
        key: value
        for key, value in baseline_args.items()
        if key not in ALLOWED_COMMAND_DIFFERENCES
    }
    candidate_common = {
        key: value
        for key, value in args.items()
        if key not in ALLOWED_COMMAND_DIFFERENCES
    }
    assert candidate_common == baseline_common


@pytest.mark.parametrize("script", sorted(MATRIX))
def test_resolved_old_abi_config_keeps_target_and_draft_modes_separate(script):
    method, _, graph_mode, target_eager = MATRIX[script]
    config = RolloutConfig(
        name="vllm",
        mode="sync",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        data_parallel_size=1,
        vllm_data_parallel_size=16,
        enforce_eager=target_eager == "True",
        cudagraph_mode=None if graph_mode == "NONE" else graph_mode,
        enable_prefix_caching=False,
        free_cache_engine=True,
        speculative_method=method,
        speculative_model=f"/models/{method}",
        num_speculative_tokens=7,
        draft_tensor_parallel_size=1,
        draft_sample_method="probabilistic",
        draft_probability_max_memory_mb=512,
        draft_load_format="auto",
        rejection_sample_method="standard",
        speculative_enforce_eager=True,
        parallel_draft_allow_target_eager_experiment=graph_mode == "NONE",
    )
    resolved = resolve_rollout_speculation(config)
    assert resolved.method == method
    assert resolved.speculative_config["enforce_eager"] is True
    kwargs = resolve_vllm_cudagraph_kwargs(
        config,
        resolved.engine_kwargs,
        {
            "VERL_VLLM_CUDAGRAPH_MODE": "" if graph_mode == "NONE" else graph_mode,
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV": "1" if graph_mode == "FULL_DECODE_ONLY" else "0",
        },
    )
    compilation = kwargs.get("compilation_config")
    if graph_mode == "NONE":
        assert compilation is None
    else:
        assert compilation["cudagraph_mode"] == graph_mode


def test_six_wrappers_are_thin_and_common_runtime_has_post_run_mode_guards():
    assert set(MATRIX) == {
        path.name for path in SCRIPTS.glob("run_qwen3_*_*_tq2.sh")
        if "baseline" not in path.name
    }
    for script, (method, label, _, _) in MATRIX.items():
        text = (SCRIPTS / script).read_text(encoding="utf-8")
        assert f"COMPARE_METHOD={method}" in text
        assert f"COMPARE_TARGET_MODE={label}" in text
        assert "qwen3_parallel_draft_compare_common.bash" in text
    common = (SCRIPTS / "qwen3_parallel_draft_compare_common.bash").read_text()
    assert "PIECEWISE compilation enabled on NPU" in common
    assert "FULL_DECODE_ONLY compilation enabled on NPU" in common
    assert "Compilation disabled, using eager mode by default" in common
    assert "parallel_draft_profile_enabled=False" in common
    assert "comparison_manifest.txt" in common
    assert "baseline_sha256=" in common
    assert "entry_sha256=" in common
    assert "common_sha256=" in common
