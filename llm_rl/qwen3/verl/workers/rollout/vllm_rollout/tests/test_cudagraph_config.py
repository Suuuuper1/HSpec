import json

import pytest

from verl.workers.rollout.vllm_rollout.cudagraph_config import (
    resolve_vllm_cudagraph_kwargs,
    serialize_vllm_cli_value,
)


def _config(**overrides):
    config = {
        "enforce_eager": False,
        "cudagraph_mode": None,
        "cudagraph_capture_sizes": None,
    }
    config.update(overrides)
    return config


def test_unspecified_graph_config_preserves_vllm_defaults():
    assert resolve_vllm_cudagraph_kwargs(_config(), {"disable_log_stats": True}, {}) == {
        "disable_log_stats": True
    }


def test_capture_sizes_only_preserves_piecewise_compatibility():
    kwargs = resolve_vllm_cudagraph_kwargs(
        _config(cudagraph_capture_sizes=[1, 2, 4]), {}, {}
    )
    assert kwargs["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4],
    }


def test_full_decode_mode_and_sizes_merge_with_engine_compilation_config():
    kwargs = resolve_vllm_cudagraph_kwargs(
        _config(
            cudagraph_mode="full_decode_only",
            cudagraph_capture_sizes=[1, 16, 32],
        ),
        {"compilation_config": {"cudagraph_num_of_warmups": 2}},
        {},
    )
    assert kwargs["compilation_config"] == {
        "cudagraph_num_of_warmups": 2,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 16, 32],
    }


def test_environment_mode_is_compatibility_fallback():
    kwargs = resolve_vllm_cudagraph_kwargs(
        _config(), {}, {"VERL_VLLM_CUDAGRAPH_MODE": "CUDAGraphMode.FULL_DECODE_ONLY"}
    )
    assert kwargs["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"


@pytest.mark.parametrize(
    ("config", "engine_kwargs", "environment"),
    [
        (
            _config(cudagraph_mode="FULL_DECODE_ONLY"),
            {},
            {"VERL_VLLM_CUDAGRAPH_MODE": "PIECEWISE"},
        ),
        (
            _config(cudagraph_mode="FULL_DECODE_ONLY"),
            {"compilation_config": {"cudagraph_mode": "PIECEWISE"}},
            {},
        ),
        (
            _config(cudagraph_capture_sizes=[1, 2]),
            {"compilation_config": {"cudagraph_capture_sizes": [1, 4]}},
            {},
        ),
    ],
)
def test_conflicting_graph_sources_fail_closed(config, engine_kwargs, environment):
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_vllm_cudagraph_kwargs(config, engine_kwargs, environment)


def test_full_graph_rejects_eager_execution():
    with pytest.raises(ValueError, match="enforce_eager=False"):
        resolve_vllm_cudagraph_kwargs(
            _config(enforce_eager=True, cudagraph_mode="FULL_DECODE_ONLY"), {}, {}
        )


@pytest.mark.parametrize("mode", ["FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
def test_full_graph_moe_contract_rejects_missing_aiv(mode):
    with pytest.raises(RuntimeError, match="HCCL_OP_EXPANSION_MODE=AIV is required"):
        resolve_vllm_cudagraph_kwargs(
            _config(cudagraph_mode=mode),
            {},
            {"VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV": "1"},
        )


def test_full_graph_moe_contract_accepts_aiv():
    kwargs = resolve_vllm_cudagraph_kwargs(
        _config(cudagraph_mode="FULL_DECODE_ONLY"),
        {},
        {
            "VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
        },
    )
    assert kwargs["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"


def test_invalid_full_graph_moe_require_aiv_flag_fails_closed():
    with pytest.raises(ValueError, match="must be a boolean flag"):
        resolve_vllm_cudagraph_kwargs(
            _config(cudagraph_mode="FULL_DECODE_ONLY"),
            {},
            {"VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV": "sometimes"},
        )


@pytest.mark.parametrize("sizes", [[2, 1], [1, 1], [], [0, 1]])
def test_invalid_capture_sizes_fail_closed(sizes):
    with pytest.raises(ValueError):
        resolve_vllm_cudagraph_kwargs(
            _config(cudagraph_capture_sizes=sizes), {}, {}
        )


def test_async_cli_uses_json_for_compilation_config():
    value = serialize_vllm_cli_value(
        "compilation_config",
        {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 16]},
    )
    assert json.loads(value) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 16],
    }
