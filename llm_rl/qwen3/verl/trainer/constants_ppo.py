# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os
from collections.abc import Mapping
from typing import Any

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env


def normalize_ray_runtime_env(runtime_env: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Hydra scalars at Ray's string-only environment ABI.

    Hydra preserves unquoted CLI values as Python scalars, while Ray requires
    every runtime environment variable key and value to be a string.  Numeric
    and boolean scalars have an unambiguous shell representation; structured
    values and null remain configuration errors.
    """
    if not isinstance(runtime_env, Mapping):
        raise TypeError(
            "Ray runtime_env must be a mapping, "
            f"got {type(runtime_env).__name__}"
        )

    normalized = dict(runtime_env)
    raw_env_vars = normalized.get("env_vars")
    if raw_env_vars is None:
        return normalized
    if not isinstance(raw_env_vars, Mapping):
        raise TypeError(
            "Ray runtime_env.env_vars must be a mapping, "
            f"got {type(raw_env_vars).__name__}"
        )

    env_vars: dict[str, str] = {}
    for key, value in raw_env_vars.items():
        if not isinstance(key, str):
            raise TypeError(
                "Ray runtime_env.env_vars keys must be strings, "
                f"got {key!r} ({type(key).__name__})"
            )
        if isinstance(value, str):
            env_vars[key] = value
        elif isinstance(value, bool):
            env_vars[key] = "1" if value else "0"
        elif isinstance(value, (int, float)):
            env_vars[key] = str(value)
        else:
            raise TypeError(
                "Ray runtime_env.env_vars values must be string-compatible "
                f"scalars; {key} has {type(value).__name__}"
            )
    normalized["env_vars"] = env_vars
    return normalized
