# SPDX-License-Identifier: Apache-2.0
"""Phase-0 registry skeleton for the old-ABI Qwen3 DSpark model."""

from typing import NoReturn

from vllm.config import VllmConfig

from .qwen3 import Qwen3ForCausalLM


class Qwen3DSparkForCausalLM(Qwen3ForCausalLM):
    phase0_placeholder = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> NoReturn:
        del vllm_config, prefix
        raise NotImplementedError(
            "Qwen3 DSpark model execution is introduced in migration Phase 2; "
            "Phase 0 supports config/registry resolution only."
        )
