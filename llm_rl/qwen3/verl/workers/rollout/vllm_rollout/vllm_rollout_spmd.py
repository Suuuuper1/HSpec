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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import json
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config.lora import LoRAConfig
from vllm.lora.request import LoRARequest
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.spec_decode.hspec_utils import (
    create_hspec_torch_npu_profiler,
    hspec_begin_decode_step_profile_session,
    hspec_clear_profile_context,
    hspec_end_decode_step_profile_session,
    hspec_profile_decode_step_sampling_enabled,
    hspec_profile_enabled_for_step,
    hspec_profile_output_dir,
    hspec_record_function,
    hspec_set_profile_context,
    prompt_id_from_token_ids,
    stable_partition_id,
)

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.cudagraph_config import (
    resolve_vllm_cudagraph_kwargs,
)
from verl.workers.rollout.vllm_rollout.speculative_config import (
    resolve_rollout_speculation,
)
from verl.workers.rollout.vllm_rollout.speculative_lifecycle import (
    SpeculativeLifecycleAudit,
)
from verl.workers.rollout.vllm_rollout.vllm_dp_topology import (
    apply_vllm_dp_environment,
    build_topology_manifest,
    build_topology_record,
    resolve_vllm_data_parallel_size,
    validate_vllm_dp_layout,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ignore redundant logs
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

# Print resolved speculative config once per process.
_PRINTED_VLLM_SPEC_CONFIG = False
_PRINTED_VLLM_COMPILATION_CONFIG = False
_PRINTED_HSPEC_PHASE3_RUNTIME = False
_HSPEC_ALIGN_DEBUG = os.getenv("HSPEC_ALIGN_DEBUG", "0") != "0"
_HSPEC_ALIGN_DEBUG_PREVIEW = int(os.getenv("HSPEC_ALIGN_DEBUG_PREVIEW", "8"))


def _maybe_log_hspec_phase3_runtime(
    *,
    use_hspec: bool,
    collect_hspec: bool,
    is_validate: bool,
    rank: int,
) -> None:
    """Log Phase 3 runtime semantics once per rollout worker process."""
    if not use_hspec:
        return
    global _PRINTED_HSPEC_PHASE3_RUNTIME
    if _PRINTED_HSPEC_PHASE3_RUNTIME:
        return
    _PRINTED_HSPEC_PHASE3_RUNTIME = True
    if int(rank) != 0:
        return
    logger.warning(
        "HSpec Phase 3 rollout runtime: "
        "table_prefetch_mode=%s allow_legacy_table_prefetch=%s enable_zmq_query=%s "
        "full_batch_prefetch=%s proposer_cache_max_cpu_bytes=%s proposer_cache_max_npu_bytes=%s "
        "proposer_batch_cache_prebuild=%s allow_hot_batch_cache_build=%s "
        "proposer_batch_cache_max_npu_bytes=%s proposer_prefix_cache=%s "
        "proposer_store_per_prompt_npu=%s validation=%s collect_hspec=%s "
        "validation_policy=query_prefetch_enabled_collection_disabled",
        os.getenv("HSPEC_TABLE_PREFETCH_MODE", "descriptor"),
        os.getenv("HSPEC_ALLOW_LEGACY_TABLE_PREFETCH", "0"),
        os.getenv("HSPEC_ENABLE_ZMQ_QUERY", "0"),
        os.getenv("HSPEC_FULL_BATCH_PREFETCH", "1"),
        os.getenv("HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES", "0"),
        os.getenv("HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES", "0"),
        os.getenv("HSPEC_PROPOSER_BATCH_CACHE_PREBUILD", "1"),
        os.getenv("HSPEC_ALLOW_HOT_BATCH_CACHE_BUILD", "0"),
        os.getenv("HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES", "0"),
        os.getenv("HSPEC_PROPOSER_PREFIX_CACHE", "0"),
        os.getenv("HSPEC_PROPOSER_STORE_PER_PROMPT_NPU", "0"),
        str(bool(is_validate)),
        str(bool(collect_hspec)),
    )


def _get_model_hidden_size(model_runner: Any) -> int | None:
    model_config = getattr(getattr(model_runner, "vllm_config", None), "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hidden_size = getattr(hf_text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(model_config, "hf_config", None), "hidden_size", None)
    return int(hidden_size) if hidden_size is not None else None


def _replace_parameter_view(model: torch.nn.Module, name: str, data: torch.Tensor) -> None:
    parts = name.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    param_name = parts[-1]
    old_param = getattr(parent, param_name)
    new_param = torch.nn.Parameter(data, requires_grad=False)
    if hasattr(old_param, "weight_loader"):
        new_param.weight_loader = old_param.weight_loader
    setattr(parent, param_name, new_param)


def _prepare_ascend_moe_weights_for_reload(model: torch.nn.Module, hidden_size: int | None) -> int:
    """Put Ascend MoE weights in the layout expected by vLLM's weight_loader.

    vllm-ascend keeps unquantized MoE weights in runtime layout after initial
    model loading. Runtime actor weight sync must write through a transposed
    view of the same storage so captured ACL graphs keep seeing updated
    weights at the original addresses.
    """
    if hidden_size is None:
        return 0

    converted = 0
    for name, param in list(model.named_parameters()):
        if param.ndim != 3:
            continue
        if name.endswith("w2_weight") and param.shape[2] == hidden_size:
            _replace_parameter_view(model, name, param.data.transpose(1, 2))
            converted += 1
        elif name.endswith("w13_weight") and param.shape[1] == hidden_size:
            _replace_parameter_view(model, name, param.data.transpose(1, 2))
            converted += 1
    return converted


def _restore_ascend_moe_weights_after_reload(model: torch.nn.Module, hidden_size: int | None) -> int:
    """Restore Ascend MoE runtime layout without allocating new weight storage."""
    if hidden_size is None:
        return 0

    converted = 0
    for name, param in list(model.named_parameters()):
        if param.ndim != 3:
            continue
        if name.endswith("w2_weight") and param.shape[1] == hidden_size:
            _replace_parameter_view(model, name, param.data.transpose(1, 2))
            converted += 1
        elif name.endswith("w13_weight") and param.shape[2] == hidden_size:
            _replace_parameter_view(model, name, param.data.transpose(1, 2))
            converted += 1
    return converted


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _build_hspec_store_index(hs_store: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key: dict[str, Any] = {}
    by_prefix: dict[str, Any] = {}
    for key, payload in hs_store.items():
        key_str = str(key)
        by_key[key_str] = payload
        if "-" in key_str:
            by_prefix.setdefault(key_str.rsplit("-", 1)[0] + "-", payload)
    return by_key, by_prefix


def _lookup_hspec_store_payload(
    hs_store_index: tuple[dict[str, Any], dict[str, Any]],
    request_id: Any,
    sample_id: int,
    num_samples: int,
) -> Any:
    by_key, by_prefix = hs_store_index
    req_id = str(request_id)

    payload = by_key.get(req_id)
    if payload is not None:
        return payload

    prefixes: list[str] = []
    if num_samples > 1:
        prefixes.append(f"{int(sample_id)}_{req_id}-")
    prefixes.append(f"{req_id}-")
    for prefix in prefixes:
        payload = by_prefix.get(prefix)
        if payload is not None:
            return payload
    return None


def _hspec_prefetch_rollout_prompts(
    inference_engine: LLM,
    vllm_inputs: list[dict[str, Any]],
    *,
    max_num_seqs: int,
) -> None:
    """Warm worker-local HSpec proposer caches before scheduling starts.

    When ``max_num_seqs`` is below the logical rollout batch size, vLLM admits
    requests in several scheduler waves. Waiting until each wave reaches the
    model runner to prefetch its table causes baseline decode steps. This hook
    starts non-blocking table prefetches for the full rollout batch up front.
    """
    if os.getenv("HSPEC_FULL_BATCH_PREFETCH", "1") == "0":
        return

    waves_ahead = max(int(os.getenv("HSPEC_PREFETCH_WAVES_AHEAD", "1")), 1)
    hard_cap = max(int(os.getenv("HSPEC_PREFETCH_MAX_PROMPTS", "0")), 0)
    prefetch_limit = max(int(max_num_seqs), 1) * waves_ahead
    if hard_cap > 0:
        prefetch_limit = min(prefetch_limit, hard_cap)

    prompt_ids: list[str] = []
    seen_prompt_ids: set[str] = set()
    for input_data in vllm_inputs[:prefetch_limit]:
        prompt_token_ids = input_data.get("prompt_token_ids", [])
        if not prompt_token_ids:
            continue
        prompt_id = prompt_id_from_token_ids(list(prompt_token_ids))
        if not prompt_id or prompt_id in seen_prompt_ids:
            continue
        seen_prompt_ids.add(prompt_id)
        prompt_ids.append(prompt_id)
    if not prompt_ids:
        return

    try:
        inference_engine.llm_engine.collective_rpc(
            "hspec_prefetch_prompt_ids_batch",
            args=(prompt_ids,),
        )
    except Exception:
        logger.debug("HSpec rollout prompt prefetch failed", exc_info=True)


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        distributed_world_size = torch.distributed.get_world_size()
        assert tensor_parallel_size <= distributed_world_size, (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        self._resolved_speculation = resolve_rollout_speculation(config)
        method = self._resolved_speculation.method
        parallel_block_enabled = method in {"dflash", "dspark"}
        self._resolved_vllm_dp = resolve_vllm_data_parallel_size(
            config.get("vllm_data_parallel_size", None)
        )
        apply_vllm_dp_environment(self._resolved_vllm_dp)
        self._vllm_dp_layout = validate_vllm_dp_layout(
            world_size=distributed_world_size,
            vllm_dp_size=self._resolved_vllm_dp.size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=config.get("pipeline_model_parallel_size", 1),
            prefill_context_parallel_size=1,
            rollout_data_parallel_size=config.get("data_parallel_size", 1),
            require_rollout_dispatch_one=parallel_block_enabled,
        )
        global_rank = torch.distributed.get_rank()
        if parallel_block_enabled:
            predicted_record = build_topology_record(
                resolved=self._resolved_vllm_dp,
                layout=self._vllm_dp_layout,
                global_rank=global_rank,
                method=method,
            )
            logger.warning(
                "DP_REPAIR_TOPOLOGY_RECORD_PRE_ENGINE %s",
                json.dumps(predicted_record, sort_keys=True),
            )
            if global_rank == 0:
                predicted_records = [
                    build_topology_record(
                        resolved=self._resolved_vllm_dp,
                        layout=self._vllm_dp_layout,
                        global_rank=rank,
                        method=method,
                    )
                    for rank in range(distributed_world_size)
                ]
                logger.warning(
                    "DP_REPAIR_TOPOLOGY_MANIFEST %s",
                    json.dumps(
                        build_topology_manifest(predicted_records), sort_keys=True
                    ),
                )

        # The old external-launcher integration must construct model-internal
        # DP groups before LLM initialization when DP is active.
        if self._resolved_vllm_dp.size > 1:
            from r1_ascend.vllm_parallel_state import init_parallel_state

            init_parallel_state(
                tensor_parallel_size,
                expected_data_parallel_size=self._resolved_vllm_dp.size,
                pipeline_parallel_size=config.get("pipeline_model_parallel_size", 1),
                prefill_context_parallel_size=1,
            )

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        if load_format == "dummy":
            logger.info(
                "vLLM rollout initializes with load_format='dummy'. This is expected for "
                "hybrid-engine training only if actor weights are synchronized before generation."
            )

        engine_kwargs = dict(self._resolved_speculation.engine_kwargs)
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        global _PRINTED_VLLM_SPEC_CONFIG
        if not _PRINTED_VLLM_SPEC_CONFIG:
            _PRINTED_VLLM_SPEC_CONFIG = True
            logger.warning(
                "Resolved Verl speculative manifest: %s",
                self._resolved_speculation.manifest,
            )

        engine_kwargs = resolve_vllm_cudagraph_kwargs(config, engine_kwargs)
        resolved_compilation_config = engine_kwargs.get("compilation_config")
        if not config.enforce_eager and resolved_compilation_config:
            torch._dynamo.config.log_compilation_metrics = False

        global _PRINTED_VLLM_COMPILATION_CONFIG
        if not _PRINTED_VLLM_COMPILATION_CONFIG:
            _PRINTED_VLLM_COMPILATION_CONFIG = True
            logger.warning("Resolved vLLM compilation_config: %s", resolved_compilation_config)
            if resolved_compilation_config and str(
                resolved_compilation_config.get("cudagraph_mode", "")
            ).upper() in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}:
                hccl_expansion = os.environ.get("HCCL_OP_EXPANSION_MODE")
                logger.warning(
                    "Resolved full-graph communication contract: "
                    "HCCL_OP_EXPANSION_MODE=%s moe_comm_require_aiv=%s",
                    hccl_expansion if hccl_expansion else "<unset>",
                    os.environ.get("VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV", "0"),
                )

        self.dynamic_eplb = int(os.environ.get("VLLM_ENABLE_EPLB", "0")) == 1
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            enable_expert_parallel=int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")),
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            async_scheduling=False,
            additional_config={
                "ascend_scheduler_config": {
                    "enabled": True,
                    "enable_chunked_prefill": bool(config.enable_chunked_prefill),
                },
                "refresh": True,
                "dynamic_eplb": self.dynamic_eplb,
                "num_iterations_eplb_update": 400,  # gather stable workload over 400 iterations
                "gate_eplb": True,
                "num_wait_worker_iterations": 30,  # wait for 30 iterations to complete the EPLB calculation
                "npugraph_ex_config": {
                    "enable": True,
                    "enable_static_kernel": eval(os.environ.get("NPUGRAPH_EX_ENABLE_STATIC_KERNEL", "False"))
                }
            },
            **self.lora_kwargs,
            **engine_kwargs,
        )
        model_runner = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        )
        if parallel_block_enabled:
            engine_parallel = model_runner.vllm_config.parallel_config
            from vllm.distributed import get_dp_group

            dp_group = get_dp_group()
            reflected_record = build_topology_record(
                resolved=self._resolved_vllm_dp,
                layout=self._vllm_dp_layout,
                global_rank=global_rank,
                actual_dp_size=engine_parallel.data_parallel_size,
                actual_dp_rank=engine_parallel.data_parallel_rank,
                actual_dp_group_ranks=dp_group.ranks,
                method=method,
                draft_model_kind=(
                    "moe"
                    if model_runner.speculative_config.draft_model_config.is_moe
                    else "dense"
                ),
            )
            self._vllm_dp_topology_record = reflected_record
            logger.warning(
                "DP_REPAIR_TOPOLOGY_RECORD %s",
                json.dumps(reflected_record, sort_keys=True),
            )
            reflected_records: list[dict[str, Any] | None] = [
                None for _ in range(distributed_world_size)
            ]
            torch.distributed.all_gather_object(reflected_records, reflected_record)
            reflected_manifest = build_topology_manifest(
                [record for record in reflected_records if record is not None]
            )
            self._vllm_dp_topology_manifest = reflected_manifest
            if global_rank == 0:
                logger.warning(
                    "DP_REPAIR_TOPOLOGY_MANIFEST_ENGINE %s",
                    json.dumps(reflected_manifest, sort_keys=True),
                )
        # vLLM may expose logits for padded vocabulary rows from tensor
        # parallel output heads. Those rows are not tokenizer tokens and must
        # not be sampled; otherwise rollout text can degenerate into invalid
        # or nonsensical token streams.
        _monkey_patch_compute_logits(
            model_runner.get_model(),
            len(tokenizer),
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        # Patch: unset logprobs if speculative_config is enabled.
        if "speculative_config" in engine_kwargs:
            logger.warning("The 'logprobs' parameter is incompatible with Speculative Decoding and has been disabled.")
            del kwargs["logprobs"]

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k not in ("seed", "n"):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self._hspec_align_debug = _HSPEC_ALIGN_DEBUG
        self._hspec_align_debug_preview = _HSPEC_ALIGN_DEBUG_PREVIEW
        self._lifecycle_audit = SpeculativeLifecycleAudit(
            config, self._resolved_speculation
        )
        self._lifecycle_audit.after_load(self.inference_engine)

        self.eplb_end()

    def eplb_start(self):
        # Restart the EPLB process before switching from training to inference.
        if self.dynamic_eplb:
            model = self.model_runner.get_model()
            model.clear_all_moe_loads()
            model.reset_all_expert_map_and_log2phy()
            self.model_runner.eplb_adaptor.__init__(model)
            self.model_runner.eplb_loader.__init__()
            self.model_runner.eplb_process.__init__(shared_dict=self.model_runner.shared_dict, policy_type=1, enable_d2d=True)
            ascend_config = get_ascend_config()
            self.model_runner.process = self.model_runner.eplb_process._launch_process()
            self.model_runner.eplb_updator.__init__(ascend_config, self.model_runner.eplb_loader, self.model_runner.eplb_process,
                                                    self.model_runner.process)
            self.model_runner.eplb_updator.get_init_expert_map()
            self.model_runner.eplb_updator.compute_and_set_moe_load()

    def eplb_end(self):
        # Shut down the EPLB service and release memory when switching from inference to training.
        if self.dynamic_eplb:
            self.model_runner.eplb_updator.adaptor.release_memory()
            self.model_runner.eplb_updator.shutdown()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        global_step = prompts.meta_info.get("global_steps")
        hspec_epoch = prompts.meta_info.get("hspec_epoch", -1)
        use_hspec = self._resolved_speculation.uses_hspec
        is_validate = prompts.meta_info.get("validate", False)
        do_sample = prompts.meta_info.get("do_sample", True)
        collect_hspec = use_hspec and not is_validate and bool(do_sample)
        profile_this_step = hspec_profile_enabled_for_step(global_step)
        profiler = None
        decode_step_profile_session = False
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _maybe_log_hspec_phase3_runtime(
            use_hspec=bool(use_hspec),
            collect_hspec=bool(collect_hspec),
            is_validate=bool(is_validate),
            rank=int(rank),
        )

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # used for history tree and prompt grouping in trainer
        _vllm_inputs_list = [input_data["prompt_token_ids"] for input_data in vllm_inputs]
        _vllm_inputs_arr = np.empty((len(_vllm_inputs_list),), dtype=object)
        _vllm_inputs_arr[:] = _vllm_inputs_list
        non_tensor_batch["vllm_inputs"] = _vllm_inputs_arr

        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        if use_hspec:
            try:
                self.inference_engine.llm_engine.collective_rpc(
                    "hspec_set_collection_enabled",
                    args=(bool(collect_hspec),),
                )
                self.inference_engine.llm_engine.collective_rpc(
                    "hspec_set_collection_context",
                    args=(int(hspec_epoch) if hspec_epoch is not None else -1,),
                )
            except Exception:
                logger.debug("HSpec collection mode update failed", exc_info=True)

        if collect_hspec:
            from vllm_ascend.spec_decode.hspec_utils import (
                hspec_clear_store,
                hspec_flush_and_get_all,
                hspec_flush_and_get_descriptors,
            )
            from vllm_ascend.spec_decode.hspec_store import (
                get_hspec_num_shards,
                hspec_legacy_dataproto_hs_enabled,
            )
            hspec_clear_store()
            legacy_hspec_dataproto_hs = hspec_legacy_dataproto_hs_enabled()
        else:
            legacy_hspec_dataproto_hs = False
        if profile_this_step:
            profile_dir = os.path.join(
                hspec_profile_output_dir(),
                f"step_{int(global_step)}",
            )
            os.makedirs(profile_dir, exist_ok=True)
            if hspec_profile_decode_step_sampling_enabled():
                hspec_begin_decode_step_profile_session(
                    step=int(global_step),
                    req_idx=-1,
                    profile_dir=profile_dir,
                )
                decode_step_profile_session = True
            else:
                profiler = create_hspec_torch_npu_profiler(profile_dir)
                hspec_set_profile_context(
                    enabled=True,
                    step=int(global_step),
                    req_idx=-1,
                )
                profiler.start()
                try:
                    profiler.add_metadata_json(
                        "hspec_profile_context",
                        (
                            f'{{"global_step": {int(global_step)}, '
                            f'"req_scope": "all_requests", '
                            f'"rank": {int(rank)}, '
                            f'"mode": "{os.getenv("HSPEC_PROFILE_METHOD", "mstx")}"}}'
                        ),
                    )
                except Exception:
                    pass

        try:
            with self.update_sampling_params(**kwargs):
                if use_hspec:
                    try:
                        self.inference_engine.llm_engine.collective_rpc(
                            "hspec_begin_prefetch_window",
                            args=(),
                        )
                    except Exception:
                        logger.debug("HSpec prefetch window reset failed", exc_info=True)
                    with hspec_record_function("hspec/rollout/full_batch_prefetch"):
                        _hspec_prefetch_rollout_prompts(
                            self.inference_engine,
                            vllm_inputs,
                            max_num_seqs=int(self.config.max_num_seqs),
                        )
                self._lifecycle_audit.before_rollout(self.inference_engine)
                if self._lifecycle_audit.enabled:
                    logger.warning(
                        "PHASE3_ROLLOUT_BEGIN method=%s round=%d",
                        self._resolved_speculation.method,
                        self._lifecycle_audit.rollouts,
                    )
                with hspec_record_function("hspec/rollout/engine_generate", use_npu_stream=True):
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=True,
                    )
                if use_hspec:
                    with hspec_record_function("hspec/rollout/round_finalize"):
                        self.inference_engine.llm_engine.collective_rpc(
                            "hspec_finalize_rollout_round",
                            args=(),
                        )
                if (
                    self._resolved_speculation.method in {"dflash", "dspark"}
                    and self._resolved_speculation.manifest.get("draft_sample_method")
                    == "probabilistic"
                ):
                    self.inference_engine.llm_engine.collective_rpc(
                        "clear_draft_probability_cache",
                        args=(),
                    )
                self._lifecycle_audit.after_rollout(self.inference_engine)
                if self._lifecycle_audit.enabled:
                    logger.warning(
                        "PHASE3_ROLLOUT_END method=%s round=%d",
                        self._resolved_speculation.method,
                        self._lifecycle_audit.rollouts - 1,
                    )

                hs_store: dict = {}
                hs_store_index: tuple[dict[str, Any], dict[str, Any]] = ({}, {})
                hspec_desc_store: dict = {}
                hspec_desc_store_index: tuple[dict[str, Any], dict[str, Any]] = ({}, {})
                if collect_hspec:
                    request_id_to_prompt_id = {}
                    for output_idx, output in enumerate(outputs):
                        if output_idx < len(vllm_inputs):
                            prompt_tokens = list(vllm_inputs[output_idx].get("prompt_token_ids", []))
                        else:
                            prompt_tokens = []
                        prompt_id = prompt_id_from_token_ids(prompt_tokens) if prompt_tokens else ""
                        request_id_to_prompt_id[str(output.request_id)] = prompt_id
                    with hspec_record_function("hspec/rollout/hidden_state_flush", use_npu_stream=True):
                        if legacy_hspec_dataproto_hs:
                            hs_store = hspec_flush_and_get_all()
                            hs_store_index = _build_hspec_store_index(hs_store)
                        else:
                            hspec_desc_store = hspec_flush_and_get_descriptors(
                                request_id_to_prompt_id=request_id_to_prompt_id,
                                epoch=int(hspec_epoch) if hspec_epoch is not None else -1,
                                global_step=int(global_step) if global_step is not None else -1,
                            )
                            hspec_desc_store_index = _build_hspec_store_index(hspec_desc_store)

                response = []
                rollout_log_probs = []
                rollout_hidden_states_list: list = []
                rollout_hspec_token_ids_list: list = []
                rollout_hspec_desc_list: list = []
                rollout_debug_list: list = []
                with hspec_record_function("hspec/rollout/output_collect"):
                    for output_idx, output in enumerate(outputs):
                        output_prompt_id = ""
                        if output_idx < len(vllm_inputs):
                            prompt_tokens = list(vllm_inputs[output_idx].get("prompt_token_ids", []))
                            output_prompt_id = prompt_id_from_token_ids(prompt_tokens) if prompt_tokens else ""
                        for sample_id in range(len(output.outputs)):
                            response_ids = output.outputs[sample_id].token_ids
                            response.append(response_ids)
                            if self.config.calculate_log_probs:
                                curr_log_prob = []
                                for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                    curr_log_prob.append(logprob[response_ids[i]].logprob)
                                rollout_log_probs.append(curr_log_prob)
                            if collect_hspec:
                                hs_source = "completion"
                                hs = getattr(output.outputs[sample_id], "hidden_states", None)
                                hspec_token_ids = getattr(output.outputs[sample_id], "hspec_token_ids", None)
                                hspec_desc = None
                                if legacy_hspec_dataproto_hs and hs is None:
                                    hs_source = "store"
                                    payload = _lookup_hspec_store_payload(
                                        hs_store_index,
                                        output.request_id,
                                        sample_id,
                                        len(output.outputs),
                                    )
                                    if payload is not None:
                                        hs = payload.get("hidden_states")
                                        if hspec_token_ids is None:
                                            hspec_token_ids = payload.get("token_ids")
                                    else:
                                        hs = None
                                if not legacy_hspec_dataproto_hs:
                                    hs_source = "descriptor"
                                    hspec_desc = _lookup_hspec_store_payload(
                                        hspec_desc_store_index,
                                        output.request_id,
                                        sample_id,
                                        len(output.outputs),
                                    )
                                    if hspec_desc is None:
                                        hs_source = "none"
                                    elif output_prompt_id:
                                        try:
                                            shard_id = stable_partition_id(
                                                output_prompt_id,
                                                get_hspec_num_shards(),
                                            )
                                            existing_prompt_id = ""
                                            if hasattr(hspec_desc, "prompt_id"):
                                                existing_prompt_id = str(
                                                    getattr(hspec_desc, "prompt_id", "") or ""
                                                )
                                            elif isinstance(hspec_desc, dict):
                                                existing_prompt_id = str(
                                                    hspec_desc.get("prompt_id", "") or ""
                                                )
                                            if (
                                                existing_prompt_id
                                                and existing_prompt_id != output_prompt_id
                                            ):
                                                try:
                                                    from vllm_ascend.spec_decode.hspec_store import (
                                                        hspec_record_store_metric,
                                                    )

                                                    hspec_record_store_metric("prompt_id_conflict", 1)
                                                except Exception:
                                                    logger.debug(
                                                        "Failed to record HSpec prompt_id conflict metric",
                                                        exc_info=True,
                                                    )
                                                logger.warning(
                                                    "HSpec descriptor prompt_id conflict in rollout output collect: "
                                                    "request_id=%s existing_prompt_id=%s output_prompt_id=%s",
                                                    output.request_id,
                                                    existing_prompt_id,
                                                    output_prompt_id,
                                                )
                                            if hasattr(hspec_desc, "with_updates"):
                                                hspec_desc = hspec_desc.with_updates(
                                                    prompt_id=output_prompt_id,
                                                    shard_id=shard_id,
                                                )
                                            elif isinstance(hspec_desc, dict):
                                                hspec_desc = dict(hspec_desc)
                                                hspec_desc["prompt_id"] = output_prompt_id
                                                hspec_desc["shard_id"] = shard_id
                                        except Exception:
                                            pass
                                    hs = None
                                if hs is None:
                                    if legacy_hspec_dataproto_hs:
                                        hs_source = "none"
                                rollout_hidden_states_list.append(hs)
                                rollout_hspec_token_ids_list.append(hspec_token_ids)
                                rollout_hspec_desc_list.append(hspec_desc)
                                if self._hspec_align_debug:
                                    raw_pad_idx = -1
                                    try:
                                        raw_pad_idx = response_ids.index(self.pad_token_id)
                                    except ValueError:
                                        pass
                                    hs_len = -1
                                    if hasattr(hs, "shape") and getattr(hs, "ndim", None) == 2:
                                        hs_len = int(hs.shape[0])
                                    elif hspec_desc is not None and hasattr(hspec_desc, "length"):
                                        hs_len = int(hspec_desc.length)
                                    elif isinstance(hspec_desc, dict) and "length" in hspec_desc:
                                        hs_len = int(hspec_desc["length"])
                                    preview = int(self._hspec_align_debug_preview)
                                    rollout_debug_list.append(
                                        {
                                            "request_id": str(output.request_id),
                                            "sample_id": int(sample_id),
                                            "raw_response_len": int(len(response_ids)),
                                            "raw_response_first_pad_index": int(raw_pad_idx),
                                            "hs_len": int(hs_len),
                                            "hspec_token_len": int(len(hspec_token_ids))
                                            if hspec_token_ids is not None else -1,
                                            "hspec_desc_present": bool(hspec_desc is not None),
                                            "hs_source": hs_source,
                                            "response_head": list(response_ids[:preview]),
                                            "response_tail": list(response_ids[-preview:]) if response_ids else [],
                                        }
                                    )

                with hspec_record_function("hspec/rollout/pad_concat", use_npu_stream=True):
                    response = pad_2d_list_to_length(
                        response, self.pad_token_id, max_length=self.config.response_length
                    ).to(idx.device)
                    if self.config.calculate_log_probs:
                        rollout_log_probs = pad_2d_list_to_length(
                            rollout_log_probs, -1, max_length=self.config.response_length
                        ).to(idx.device)
                        rollout_log_probs = rollout_log_probs.to(torch.float32)

                    seq = torch.cat([idx, response], dim=-1)
        finally:
            if decode_step_profile_session:
                hspec_end_decode_step_profile_session()
            if profiler is not None:
                try:
                    torch.npu.synchronize()
                except Exception:
                    pass
                try:
                    profiler.step()
                except Exception:
                    pass
                profiler.stop()
            hspec_clear_profile_context()

        with hspec_record_function("hspec/rollout/metadata_pack", use_npu_stream=True):
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                    batch_size, position_ids.size(1), -1
                )

            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )
            if self.config.calculate_log_probs:
                batch["rollout_log_probs"] = rollout_log_probs

            if collect_hspec and legacy_hspec_dataproto_hs and rollout_hidden_states_list:
                _hs_list = list(rollout_hidden_states_list)
                _hs_arr = np.empty((len(_hs_list),), dtype=object)
                _hs_arr[:] = _hs_list
                non_tensor_batch["rollout_hidden_states"] = _hs_arr

                _tok_list = list(rollout_hspec_token_ids_list)
                _tok_arr = np.empty((len(_tok_list),), dtype=object)
                _tok_arr[:] = _tok_list
                non_tensor_batch["rollout_hspec_tokens"] = _tok_arr
                try:
                    from vllm_ascend.spec_decode.hspec_store import hspec_record_store_metric

                    hspec_record_store_metric("legacy_payload_count", len(_hs_list))
                except Exception:
                    logger.debug("Failed to record HSpec legacy payload metric", exc_info=True)

            if collect_hspec and not legacy_hspec_dataproto_hs and rollout_hspec_desc_list:
                _desc_list = list(rollout_hspec_desc_list)
                _desc_arr = np.empty((len(_desc_list),), dtype=object)
                _desc_arr[:] = _desc_list
                non_tensor_batch["hspec_desc"] = _desc_arr
                try:
                    from vllm_ascend.spec_decode.hspec_store import hspec_record_store_metric

                    hspec_record_store_metric("descriptor_payload_count", len(_desc_list))
                except Exception:
                    logger.debug("Failed to record HSpec descriptor payload metric", exc_info=True)

            if collect_hspec and self._hspec_align_debug and rollout_debug_list:
                _dbg_arr = np.empty((len(rollout_debug_list),), dtype=object)
                _dbg_arr[:] = rollout_debug_list
                non_tensor_batch["hspec_rollout_debug"] = _dbg_arr

            if use_hspec and not collect_hspec:
                try:
                    from vllm_ascend.spec_decode.hspec_store import hspec_record_store_metric

                    hspec_record_store_metric("validation_collect_skip", batch_size)
                except Exception:
                    logger.debug("Failed to record HSpec collect-skip metric", exc_info=True)

            try:
                from vllm_ascend.spec_decode.hspec_store import (
                    hspec_record_store_metric,
                    hspec_strict_descriptor_mode_enabled,
                    hspec_step0_runtime_asserts_enabled,
                )

                if hspec_strict_descriptor_mode_enabled():
                    forbidden = ("rollout_hidden_states", "rollout_hspec_tokens")
                    present = [key for key in forbidden if key in non_tensor_batch]
                    if present:
                        hspec_record_store_metric("strict_descriptor_violation", len(present))
                        raise RuntimeError(
                            "HSpec strict descriptor mode forbids legacy rollout "
                            f"payload keys: {present}"
                        )
                if hspec_step0_runtime_asserts_enabled():
                    if not legacy_hspec_dataproto_hs:
                        forbidden = ("rollout_hidden_states", "rollout_hspec_tokens")
                        present = [key for key in forbidden if key in non_tensor_batch]
                        if present:
                            hspec_record_store_metric("strict_descriptor_violation", len(present))
                            raise RuntimeError(
                                "HSpec Step0 invariant failed: default descriptor path "
                                f"emitted legacy payload keys {present}"
                            )
                    if use_hspec and not collect_hspec:
                        forbidden = ("hspec_desc", "rollout_hidden_states", "rollout_hspec_tokens")
                        present = [key for key in forbidden if key in non_tensor_batch]
                        if present:
                            hspec_record_store_metric("strict_descriptor_violation", len(present))
                            raise RuntimeError(
                                "HSpec Step0 invariant failed: collection-disabled rollout "
                                f"emitted HSpec payload keys {present}"
                            )
            except RuntimeError:
                raise
            except Exception:
                logger.debug("Failed to run HSpec Step0 runtime assertions", exc_info=True)

        meta_info = {}
        if use_hspec:
            try:
                from vllm_ascend.spec_decode.hspec_store import collect_hspec_store_metrics
                from vllm_ascend.spec_decode.hspec_utils import hspec_collect_runtime_metrics

                store_metrics = collect_hspec_store_metrics(reset=True)
                runtime_metrics = hspec_collect_runtime_metrics(reset=True)
                meta_info["metrics"] = {
                    "hspec/raw_store_bytes": float(store_metrics.get("raw_store_bytes", 0)),
                    "hspec/desc_count": float(store_metrics.get("desc_count", 0)),
                    "hspec/desc_multiextent_count": float(
                        store_metrics.get("desc_multiextent_count", 0)),
                    "hspec/desc_extent_total": float(
                        store_metrics.get("desc_extent_total", 0)),
                    "hspec/desc_extent_max": float(
                        store_metrics.get("desc_extent_max", 0)),
                    "hspec/collect_dropped": float(store_metrics.get("collect_dropped", 0)),
                    "hspec/collect_dropped_empty": float(
                        store_metrics.get("collect_dropped_empty", 0)),
                    "hspec/collect_dropped_invalid_dim": float(
                        store_metrics.get("collect_dropped_invalid_dim", 0)),
                    "hspec/collect_dropped_missing_offset": float(
                        store_metrics.get("collect_dropped_missing_offset", 0)),
                    "hspec/collect_dropped_align_mismatch": float(
                        store_metrics.get("collect_dropped_align_mismatch", 0)),
                    "hspec/collect_dropped_unpaired_extent": float(
                        store_metrics.get("collect_dropped_unpaired_extent", 0)),
                    "hspec/strict_descriptor_violation": float(
                        store_metrics.get("strict_descriptor_violation", 0)),
                    "hspec/legacy_payload_count": float(store_metrics.get("legacy_payload_count", 0)),
                    "hspec/descriptor_payload_count": float(
                        store_metrics.get("descriptor_payload_count", 0)),
                    "hspec/validation_collect_skip": float(
                        store_metrics.get("validation_collect_skip", 0)),
                    "hspec/segment_sealed": float(store_metrics.get("segment_sealed", 0)),
                    "hspec/segment_rotated": float(store_metrics.get("segment_rotated", 0)),
                    "hspec/segment_manifest_write_error": float(
                        store_metrics.get("segment_manifest_write_error", 0)),
                    "hspec/raw_store_budget_gc_skipped": float(
                        store_metrics.get("raw_store_budget_gc_skipped", 0)),
                    "hspec/raw_store_epoch_bytes": float(
                        store_metrics.get("raw_store_epoch_bytes", 0)),
                    "hspec/raw_store_epoch_budget_bytes": float(
                        store_metrics.get("raw_store_epoch_budget_bytes", 0)),
                    "hspec/raw_store_collect_budget_blocked": float(
                        store_metrics.get("raw_store_collect_budget_blocked", 0)),
                    "hspec/raw_store_collect_budget_unblocked": float(
                        store_metrics.get("raw_store_collect_budget_unblocked", 0)),
                    "hspec/raw_store_collect_drop_bytes": float(
                        store_metrics.get("raw_store_collect_drop_bytes", 0)),
                    "hspec/raw_store_budget_active": float(
                        store_metrics.get("raw_store_budget_active", 0)),
                    "hspec/collect_dropped_budget_worker_bytes": float(
                        store_metrics.get("collect_dropped_budget_worker_bytes", 0)),
                    "hspec/collect_dropped_budget_epoch_bytes": float(
                        store_metrics.get("collect_dropped_budget_epoch_bytes", 0)),
                    "hspec/collect_dropped_raw_store_over_budget": float(
                        store_metrics.get("collect_dropped_raw_store_over_budget", 0)),
                    "hspec/unsafe_descriptor_cleanup_suppressed": float(
                        store_metrics.get("unsafe_descriptor_cleanup_suppressed", 0)),
                    "hspec/pinned_pool_miss": float(runtime_metrics.get("pinned_pool_miss", 0)),
                    "hspec/pinned_pageable_fallback": float(runtime_metrics.get("pinned_pageable_fallback", 0)),
                    "hspec/pinned_reserved_bytes": float(
                        runtime_metrics.get("pinned_reserved_bytes", 0)),
                    "hspec/pinned_reserved_slots": float(
                        runtime_metrics.get("pinned_reserved_slots", 0)),
                    "hspec/pinned_checkout_count": float(
                        runtime_metrics.get("pinned_checkout_count", 0)),
                    "hspec/pinned_reuse_count": float(
                        runtime_metrics.get("pinned_reuse_count", 0)),
                    "hspec/pinned_alloc_count": float(
                        runtime_metrics.get("pinned_alloc_count", 0)),
                    "hspec/pinned_miss_budget_bytes": float(
                        runtime_metrics.get("pinned_miss_budget_bytes", 0)),
                    "hspec/pinned_miss_budget_slots": float(
                        runtime_metrics.get("pinned_miss_budget_slots", 0)),
                    "hspec/pinned_miss_alloc_error": float(
                        runtime_metrics.get("pinned_miss_alloc_error", 0)),
                    "hspec/pinned_miss_shape_too_large": float(
                        runtime_metrics.get("pinned_miss_shape_too_large", 0)),
                    "hspec/copy_pending_tasks_max": float(
                        runtime_metrics.get("copy_pending_tasks_max", 0)),
                    "hspec/copy_pending_rows_max": float(
                        runtime_metrics.get("copy_pending_rows_max", 0)),
                    "hspec/copy_submitted_tasks": float(
                        runtime_metrics.get("copy_submitted_tasks", 0)),
                    "hspec/copy_submitted_rows": float(
                        runtime_metrics.get("copy_submitted_rows", 0)),
                    "hspec/copy_backpressure_drop": float(
                        runtime_metrics.get("copy_backpressure_drop", 0)),
                    "hspec/copy_backpressure_drop_rows": float(
                        runtime_metrics.get("copy_backpressure_drop_rows", 0)),
                    "hspec/copy_backpressure_drop_reqs": float(
                        runtime_metrics.get("copy_backpressure_drop_reqs", 0)),
                    "hspec/collect_budget_drop": float(
                        runtime_metrics.get("collect_budget_drop", 0)),
                    "hspec/collect_budget_drop_bytes": float(
                        runtime_metrics.get("collect_budget_drop_bytes", 0)),
                    "hspec/collect_budget_drop_reqs": float(
                        runtime_metrics.get("collect_budget_drop_reqs", 0)),
                    "hspec/collect_budget_over_worker_bytes": float(
                        runtime_metrics.get("collect_budget_over_worker_bytes", 0)),
                    "hspec/collect_budget_over_epoch_bytes": float(
                        runtime_metrics.get("collect_budget_over_epoch_bytes", 0)),
                    "hspec/backpressure_active": float(
                        runtime_metrics.get("backpressure_active", 0)),
                    "hspec/backpressure_collect_skip": float(
                        runtime_metrics.get("backpressure_collect_skip", 0)),
                    "hspec/pinned_fallback_ratio_skip": float(
                        runtime_metrics.get("pinned_fallback_ratio_skip", 0)),
                    "hspec/copy_worker_error": float(
                        runtime_metrics.get("copy_worker_error", 0)),
                    "hspec/copy_submit_error": float(
                        runtime_metrics.get("copy_submit_error", 0)),
                    "hspec/copy_worker_pair_write_error": float(
                        runtime_metrics.get("copy_worker_pair_write_error", 0)),
                    "hspec/copy_token_hidden_len_mismatch": float(
                        runtime_metrics.get("copy_token_hidden_len_mismatch", 0)),
                    "hspec/flush_wait_ms_total": float(
                        runtime_metrics.get("flush_wait_ms_total", 0)),
                    "hspec/flush_wait_ms_max": float(
                        runtime_metrics.get("flush_wait_ms_max", 0)),
                    "hspec/flush_wait_count": float(
                        runtime_metrics.get("flush_wait_count", 0)),
                }
            except Exception:
                logger.debug("Failed to collect HSpec rollout metrics", exc_info=True)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        observe_wake = bool(
            self.config.get("parallel_draft_profile_enabled", False)
        )
        wake_start_ns = time.perf_counter_ns() if observe_wake else 0
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()
        if observe_wake:
            # Ray keeps this Verl worker logger at WARNING in the certified
            # runtime; INFO would make the required evidence unreachable.
            logger.warning(
                "PHASE5_WAKE_METRIC metric=%s method=%s tags=%s value_ms=%.6f",
                "spec/draft_weight_wake_ms" if "weights" in tags else "spec/draft_kv_wake_ms",
                self._resolved_speculation.method,
                ",".join(sorted(tags)),
                (time.perf_counter_ns() - wake_start_ns) / 1_000_000.0,
            )
        self._lifecycle_audit.after_wake(self.inference_engine, list(tags))

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        reset_succeeded = bool(self.inference_engine.reset_prefix_cache())

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)
        self._lifecycle_audit.after_sleep(
            reset_prefix_cache_succeeded=reset_succeeded
        )

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            model = model_runner.get_model()
            patch_vllm_moe_model_weight_loader(model)
            loaded_params = model.load_weights(weights)

            model_config = model_runner.vllm_config.model_config
            device_config = model_runner.vllm_config.device_config
            load_config = model_runner.vllm_config.load_config
            load_device = (
                device_config.device if load_config.device is None else load_config.device
            )
            target_device = torch.device(load_device)
            process_weights_after_loading(model, model_config, target_device)

            loaded_count = len(loaded_params) if loaded_params is not None else -1
            logger.warning(
                "vLLM rollout loaded %s parameters from actor weights",
                loaded_count,
            )
            if loaded_count == 0:
                raise RuntimeError(
                    "vLLM rollout weight synchronization loaded 0 parameters. "
                    "Generation would continue with dummy/random weights."
                )
        self._lifecycle_audit.after_target_update(self.inference_engine)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": model_config.lora_rank} if model_config.lora_rank > 0 else {}
        )

        # https://github.com/vllm-project/vllm/issues/25171
        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            observe_wake = bool(
                self.config.get("parallel_draft_profile_enabled", False)
            )
            wake_start_ns = time.perf_counter_ns() if observe_wake else 0
            self.inference_engine.wake_up(tags=tags)
            if observe_wake:
                # Keep the async path under the same worker evidence contract.
                logger.warning(
                    "PHASE5_WAKE_METRIC metric=%s method=%s tags=%s value_ms=%.6f",
                    "spec/draft_weight_wake_ms" if "weights" in tags else "spec/draft_kv_wake_ms",
                    getattr(getattr(self, "_resolved_speculation", None), "method", None),
                    ",".join(sorted(tags)),
                    (time.perf_counter_ns() - wake_start_ns) / 1_000_000.0,
                )

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
