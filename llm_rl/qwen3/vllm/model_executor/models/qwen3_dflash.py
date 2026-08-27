# SPDX-License-Identifier: Apache-2.0
"""Phase-0 registry skeleton for the old-ABI Qwen3 DFlash model.

The real model implementation is a Phase-1 deliverable. Inheriting the Qwen3
interfaces lets registry inspection validate the checkpoint architecture, while
the constructor fails closed so Phase 0 can never run ordinary Qwen3 semantics
under a DFlash name.
"""

from typing import NoReturn

from vllm.config import VllmConfig

from .qwen3 import Qwen3ForCausalLM


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    phase0_placeholder = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> NoReturn:
        del vllm_config, prefix
        raise NotImplementedError(
            "Qwen3 DFlash model execution is introduced in migration Phase 1; "
            "Phase 0 supports config/registry resolution only."
        )
