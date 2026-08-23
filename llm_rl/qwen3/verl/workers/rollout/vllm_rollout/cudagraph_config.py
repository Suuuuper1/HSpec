# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve verl rollout graph settings into vLLM's CompilationConfig."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_CUDAGRAPH_MODES = frozenset(
    {
        "NONE",
        "PIECEWISE",
        "FULL",
        "FULL_DECODE_ONLY",
        "FULL_AND_PIECEWISE",
    }
)
_MODE_ENV = "VERL_VLLM_CUDAGRAPH_MODE"
_FULL_GRAPH_MOE_SAFE_ENV = "VLLM_ASCEND_FULL_GRAPH_MOE_COMM_SAFE"


def _normalize_mode(value: Any, source: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    mode = str(value).strip().upper()
    if mode.startswith("CUDAGRAPHMODE."):
        mode = mode.split(".", 1)[1]
    if mode not in _CUDAGRAPH_MODES:
        valid = ", ".join(sorted(_CUDAGRAPH_MODES))
        raise ValueError(f"Invalid cudagraph mode from {source}: {value!r}. Valid modes: {valid}")
    return mode


def _normalize_sizes(value: Any, source: str) -> list[int] | None:
    if value is None:
        return None
    try:
        sizes = [int(size) for size in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a list of positive integers, got {value!r}") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"{source} must be a non-empty list of positive integers, got {value!r}")
    if sizes != sorted(set(sizes)):
        raise ValueError(f"{source} must be strictly increasing without duplicates, got {value!r}")
    return sizes


def _env_flag(env: Mapping[str, str], name: str) -> bool:
    value = str(env.get(name, "0")).strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean flag, got {env.get(name)!r}")


def resolve_vllm_cudagraph_kwargs(
    config: Any,
    engine_kwargs: Mapping[str, Any] | None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge rollout graph settings into a copied vLLM engine kwargs dict.

    ``rollout.cudagraph_mode`` is the primary interface. The environment
    variable is retained as a compatibility fallback for existing launchers.
    Explicitly conflicting sources are rejected so an experiment cannot be
    silently executed with a different graph mode or capture-size set.
    """

    env = os.environ if environ is None else environ
    resolved = dict(engine_kwargs or {})

    raw_compilation_config = resolved.get("compilation_config")
    if raw_compilation_config is None:
        compilation_config: dict[str, Any] = {}
    elif isinstance(raw_compilation_config, Mapping):
        compilation_config = dict(raw_compilation_config)
    else:
        raise TypeError(
            "engine_kwargs.vllm.compilation_config must be a mapping when "
            "rollout cudagraph settings are used"
        )

    config_mode = _normalize_mode(config.get("cudagraph_mode"), "rollout.cudagraph_mode")
    env_mode = _normalize_mode(env.get(_MODE_ENV), _MODE_ENV)
    if config_mode is not None and env_mode is not None and config_mode != env_mode:
        raise ValueError(
            f"Conflicting cudagraph modes: rollout.cudagraph_mode={config_mode} "
            f"but {_MODE_ENV}={env_mode}"
        )
    requested_mode = config_mode or env_mode

    engine_mode = _normalize_mode(
        compilation_config.get("cudagraph_mode"),
        "engine_kwargs.vllm.compilation_config.cudagraph_mode",
    )
    if requested_mode is not None and engine_mode is not None and requested_mode != engine_mode:
        raise ValueError(
            "Conflicting cudagraph modes between rollout configuration and "
            "engine_kwargs.vllm.compilation_config"
        )
    effective_mode = requested_mode or engine_mode

    # On the 30B MoE/EP rollout path, decode selects MC2. Capturing MC2 while
    # HCCL expands collectives as AIV can produce a graph that captures
    # successfully but deadlocks at its first replay. The experiment launcher
    # removes HCCL_OP_EXPANSION_MODE before process creation; retain this check
    # in every Ray worker so an inherited or externally injected AIV setting
    # cannot silently reintroduce the unsafe combination.
    if (
        effective_mode in {"FULL", "FULL_DECODE_ONLY"}
        and _env_flag(env, _FULL_GRAPH_MOE_SAFE_ENV)
        and str(env.get("HCCL_OP_EXPANSION_MODE", "")).strip().upper() == "AIV"
    ):
        raise RuntimeError(
            "Unsafe full-graph MoE communication configuration: "
            "HCCL_OP_EXPANSION_MODE=AIV can deadlock MC2 collectives during "
            "ACLGraph replay. Unset HCCL_OP_EXPANSION_MODE before launching "
            "Ray workers."
        )

    requested_sizes = _normalize_sizes(
        config.get("cudagraph_capture_sizes"), "rollout.cudagraph_capture_sizes"
    )
    engine_sizes = _normalize_sizes(
        compilation_config.get("cudagraph_capture_sizes"),
        "engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes",
    )
    if requested_sizes is not None and engine_sizes is not None and requested_sizes != engine_sizes:
        raise ValueError(
            "Conflicting cudagraph capture sizes between rollout configuration and "
            "engine_kwargs.vllm.compilation_config"
        )
    effective_sizes = requested_sizes or engine_sizes

    enforce_eager = bool(config.get("enforce_eager", False))
    if effective_mode == "NONE" and effective_sizes is not None:
        raise ValueError("cudagraph_capture_sizes cannot be set when cudagraph_mode=NONE")
    if enforce_eager and effective_mode not in (None, "NONE"):
        raise ValueError(f"cudagraph_mode={effective_mode} requires rollout.enforce_eager=False")

    # Preserve verl 0.6 behavior for capture-size-only configurations.
    if effective_mode is None and effective_sizes is not None and not enforce_eager:
        effective_mode = "PIECEWISE"

    if enforce_eager and effective_mode is None:
        # Historical behavior ignored rollout capture sizes in eager mode.
        return resolved

    if effective_mode is not None:
        compilation_config["cudagraph_mode"] = effective_mode
    if effective_sizes is not None:
        compilation_config["cudagraph_capture_sizes"] = effective_sizes

    if compilation_config:
        resolved["compilation_config"] = compilation_config
    return resolved


def serialize_vllm_cli_value(key: str, value: Any) -> Any:
    """Serialize structured vLLM CLI values with JSON, not Python repr."""

    if key == "compilation_config" and isinstance(value, Mapping):
        import json

        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return value
