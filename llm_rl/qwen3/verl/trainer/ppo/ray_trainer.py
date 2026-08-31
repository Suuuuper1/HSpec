# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import psutil
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.ray_resource import (
    evaluate_resource_pool_capacity,
    format_resource_capacity_error,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


def _maybe_get_ray_object_store_used_mb() -> float:
    try:
        import ray._private.internal_api as internal_api

        summary = internal_api.memory_summary(stats_only=True)
        for line in str(summary).splitlines():
            if "Plasma memory usage" in line or "Object Store memory usage" in line:
                numbers = [token for token in line.replace(",", " ").split() if token.replace(".", "", 1).isdigit()]
                if numbers:
                    return float(numbers[0])
    except Exception:
        pass
    return -1.0


def _hspec_get_env_int(name: str, default: int = 0, minimum: int = 0) -> int:
    value = os.getenv(name, str(default))
    try:
        return max(int(value), int(minimum))
    except (TypeError, ValueError):
        return max(int(default), int(minimum))


def _hspec_get_env_float(name: str, default: float = 0.0, minimum: float = 0.0) -> float:
    value = os.getenv(name, str(default))
    try:
        return max(float(value), float(minimum))
    except (TypeError, ValueError):
        return max(float(default), float(minimum))


def _hspec_build_max_pending_epochs() -> int:
    return _hspec_get_env_int("HSPEC_BUILD_MAX_PENDING_EPOCHS", 0, 0)


def _hspec_build_queue_max_lag_s() -> float:
    return _hspec_get_env_float("HSPEC_BUILD_QUEUE_MAX_LAG_S", 0.0, 0.0)


def _hspec_epoch_build_barrier_timeout_s() -> float:
    return _hspec_get_env_float("HSPEC_EPOCH_BUILD_BARRIER_TIMEOUT_S", 0.0, 0.0)


def _hspec_swap_partial_on_timeout_enabled() -> bool:
    return os.getenv("HSPEC_SWAP_PARTIAL_ON_TIMEOUT", "0") != "0"


def _hspec_build_timeout_discard_unfinished_enabled() -> bool:
    return os.getenv("HSPEC_BUILD_TIMEOUT_DISCARD_UNFINISHED", "1") != "0"


def _hspec_phase4_metrics_every_steps() -> int:
    try:
        from vllm_ascend.spec_decode.hspec_utils import get_hspec_phase4_metrics_every_steps

        return get_hspec_phase4_metrics_every_steps()
    except Exception:
        return _hspec_get_env_int("HSPEC_PHASE4_METRICS_EVERY_STEPS", 1, 1)


_HSPEC_PHASE4_GAUGE_CACHE_KEYS = (
    "hspec/active_version",
    "hspec/num_prompts",
    "hspec/total_entries",
    "hspec/table_active_bytes",
    "hspec/table_active_mb",
    "hspec/table_active_prompts",
    "hspec/table_active_entries",
    "hspec/table_store_referenced_versions",
    "hspec/proposer_cache_live_cpu_bytes",
    "hspec/proposer_cache_live_cpu_mb",
    "hspec/proposer_cache_live_npu_bytes",
    "hspec/proposer_cache_live_npu_mb",
    "hspec/proposer_cache_live_entries",
    "hspec/proposer_cache_live_prompts",
)


def _extract_and_sum_hspec_rollout_metrics(meta_info: dict | None) -> dict[str, float]:
    if not isinstance(meta_info, dict):
        return {}
    metrics_obj = meta_info.get("metrics")
    if not isinstance(metrics_obj, dict):
        return {}

    sum_keys = (
        "hspec/raw_store_bytes",
        "hspec/desc_count",
        "hspec/desc_multiextent_count",
        "hspec/desc_extent_total",
        "hspec/collect_dropped",
        "hspec/collect_dropped_align_mismatch",
        "hspec/collect_dropped_unpaired_extent",
        "hspec/pinned_pool_miss",
        "hspec/pinned_pageable_fallback",
        "hspec/pinned_reserved_bytes",
        "hspec/pinned_reserved_slots",
        "hspec/pinned_checkout_count",
        "hspec/pinned_reuse_count",
        "hspec/pinned_alloc_count",
        "hspec/pinned_miss_budget_bytes",
        "hspec/pinned_miss_budget_slots",
        "hspec/pinned_miss_alloc_error",
        "hspec/pinned_miss_shape_too_large",
        "hspec/copy_submitted_tasks",
        "hspec/copy_submitted_rows",
        "hspec/copy_backpressure_drop",
        "hspec/copy_backpressure_drop_rows",
        "hspec/copy_backpressure_drop_reqs",
        "hspec/collect_budget_drop",
        "hspec/collect_budget_drop_bytes",
        "hspec/collect_budget_drop_reqs",
        "hspec/collect_budget_over_worker_bytes",
        "hspec/collect_budget_over_epoch_bytes",
        "hspec/backpressure_active",
        "hspec/backpressure_collect_skip",
        "hspec/pinned_fallback_ratio_skip",
        "hspec/raw_store_epoch_bytes",
        "hspec/raw_store_epoch_budget_bytes",
        "hspec/raw_store_collect_budget_blocked",
        "hspec/raw_store_collect_budget_unblocked",
        "hspec/raw_store_collect_drop_bytes",
        "hspec/raw_store_budget_active",
        "hspec/collect_dropped_budget_worker_bytes",
        "hspec/collect_dropped_budget_epoch_bytes",
        "hspec/collect_dropped_raw_store_over_budget",
        "hspec/copy_worker_error",
        "hspec/copy_submit_error",
        "hspec/copy_worker_pair_write_error",
        "hspec/copy_token_hidden_len_mismatch",
        "hspec/flush_wait_ms_total",
        "hspec/flush_wait_count",
    )
    max_keys = (
        "hspec/desc_extent_max",
        "hspec/copy_pending_tasks_max",
        "hspec/copy_pending_rows_max",
        "hspec/flush_wait_ms_max",
    )
    result: dict[str, float] = {}
    for key in sum_keys:
        value = metrics_obj.pop(key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            result[key] = float(sum(value))
        else:
            result[key] = float(value)
    for key in max_keys:
        value = metrics_obj.pop(key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            result[key] = float(max(value)) if value else 0.0
        else:
            result[key] = float(value)
    checkout_count = result.get("hspec/pinned_checkout_count", 0.0)
    if checkout_count > 0:
        result["hspec/pinned_pageable_fallback_ratio"] = (
            result.get("hspec/pinned_pageable_fallback", 0.0) / checkout_count
        )
        result["hspec/pinned_pool_miss_ratio"] = (
            result.get("hspec/pinned_pool_miss", 0.0) / checkout_count
        )
    flush_wait_count = result.get("hspec/flush_wait_count", 0.0)
    if flush_wait_count > 0:
        result["hspec/flush_wait_ms_avg"] = (
            result.get("hspec/flush_wait_ms_total", 0.0) / flush_wait_count
        )
    if not metrics_obj:
        meta_info.pop("metrics", None)
    return result


@dataclass
class HSpecPendingBuild:
    epoch: int
    ref: ray.ObjectRef
    shard_id: int
    segments: frozenset[object] = field(default_factory=frozenset)
    prompt_ids: tuple[str, ...] = field(default_factory=tuple)
    submitted_time_ns: int = 0
    deadline_ns: int = 0
    legacy: bool = False
    done: bool = False
    timed_out: bool = False
    result_metrics: dict[str, float] = field(default_factory=dict)
    result_payload: dict = field(default_factory=dict)


@dataclass
class HSpecEpochBuildBarrierResult:
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    completed_prompt_ids: tuple[str, ...] = field(default_factory=tuple)
    timed_out_prompt_ids: tuple[str, ...] = field(default_factory=tuple)
    ready_shard_ids: tuple[int, ...] = field(default_factory=tuple)
    timed_out_shard_ids: tuple[int, ...] = field(default_factory=tuple)
    timed_out: bool = False


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        # Check before creating any placement group, then let the worker launch
        # path check again so a capacity race remains fail-closed.
        self.validate_resource_available()
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def validate_resource_available(self) -> dict:
        """Check total capacity and node-local Ray bundle placement."""
        node_available_resources = ray._private.state.available_resources_per_node()
        assessment = evaluate_resource_pool_capacity(
            self.resource_pool_spec, node_available_resources
        )
        if assessment["status"] != "PASS":
            raise ValueError(format_resource_capacity_error(assessment))
        print(
            "Ray accelerator topology validation PASS: "
            + json.dumps(assessment, sort_keys=True, separators=(",", ":"))
        )
        return assessment

    def _check_resource_available(self):
        """Compatibility alias for callers using the previous private hook."""
        return self.validate_resource_available()


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._hspec_dump_enabled = os.getenv("HSPEC_DUMP", "0") != "0"
        self._hspec_dump_root = os.getenv("HSPEC_DUMP_DIR", "/workspace/exp/hspec_dump")
        self._hspec_dump_epoch_meta_written: set[int] = set()
        self._hspec_dump_tables_written: dict[int, set[str]] = defaultdict(set)
        self._hspec_align_debug = os.getenv("HSPEC_ALIGN_DEBUG", "0") != "0"
        self._hspec_align_debug_max_logs = int(os.getenv("HSPEC_ALIGN_DEBUG_MAX_LOGS", "24"))
        # HSpec build records retain segment keys until epoch-level GC.
        self._hspec_pending_build_refs: list[HSpecPendingBuild] = []
        self._hspec_last_epoch_build_barrier = HSpecEpochBuildBarrierResult(epoch=-1)
        self._hspec_phase4_metrics_every_steps = _hspec_phase4_metrics_every_steps()
        self._hspec_last_phase4_metrics_step = -1
        self._hspec_last_phase4_metrics_wall_ns = 0
        self._hspec_phase4_metrics_sample_count = 0
        self._hspec_phase4_metrics_skip_cadence_count = 0
        self._hspec_phase4_metrics_skip_inflight_count = 0
        self._hspec_phase4_metrics_error_count = 0
        self._hspec_cached_phase4_gauges: dict[str, float] = {}
        self._hspec_force_phase4_metrics_next = True

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]
        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        train_multiple = train_batch_size
        if train_multiple > 0:
            train_len = len(self.train_dataset)
            aligned_train_len = train_len - train_len % train_multiple
            if 0 < aligned_train_len < train_len:
                if hasattr(self.train_dataset, "dataframe"):
                    self.train_dataset.dataframe = self.train_dataset.dataframe.select(range(aligned_train_len))
                    if train_sampler is not None:
                        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
                    print(
                        f"[DataAlign] floor train dataset to batch multiple: "
                        f"{train_len} -> {aligned_train_len}"
                    )
                else:
                    print(
                        f"[DataAlign] skip train dataset flooring for dataset type "
                        f"{type(self.train_dataset).__name__}"
                    )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    @staticmethod
    def _drop_hspec_non_tensor_fields(batch: DataProto | None) -> None:
        if batch is None or batch.non_tensor_batch is None:
            return
        for key in (
            "hspec_desc",
            "rollout_hidden_states",
            "rollout_hspec_tokens",
            "hspec_rollout_debug",
        ):
            batch.non_tensor_batch.pop(key, None)

    def _hspec_has_inflight_builds(self) -> bool:
        return any(
            not record.done
            for record in getattr(self, "_hspec_pending_build_refs", [])
        )

    def _hspec_phase4_metrics_due(self, *, force: bool = False) -> tuple[bool, str]:
        if force:
            return True, "force"
        every = max(int(getattr(self, "_hspec_phase4_metrics_every_steps", 1)), 1)
        step = int(getattr(self, "global_steps", 0))
        last = int(getattr(self, "_hspec_last_phase4_metrics_step", -1))
        if last < 0:
            return True, "first"
        if every <= 1 or step - last >= every:
            return True, "cadence"
        return False, "cadence_skip"

    def _emit_cached_hspec_phase4_gauges(self, metrics: dict) -> None:
        cached = getattr(self, "_hspec_cached_phase4_gauges", {})
        if not cached:
            metrics["hspec/phase4_metrics_cached_gauge_count"] = 0.0
            metrics["hspec/phase4_metrics_cached_age_steps"] = -1.0
            return
        step = int(getattr(self, "global_steps", 0))
        last = int(getattr(self, "_hspec_last_phase4_metrics_step", -1))
        for key, value in cached.items():
            short_key = key[len("hspec/"):] if key.startswith("hspec/") else key
            metrics[f"hspec/phase4_cached/{short_key}"] = float(value)
        metrics["hspec/phase4_metrics_cached_gauge_count"] = float(len(cached))
        metrics["hspec/phase4_metrics_cached_age_steps"] = float(max(step - last, 0))

    def _update_cached_hspec_phase4_gauges(self, table_metrics: dict) -> None:
        cached: dict[str, float] = {}
        for key in _HSPEC_PHASE4_GAUGE_CACHE_KEYS:
            value = table_metrics.get(key)
            if isinstance(value, (int, float)):
                cached[key] = float(value)
        self._hspec_cached_phase4_gauges = cached

    def _maybe_collect_hspec_phase4_metrics(self, metrics: dict, *, force: bool = False) -> bool:
        due, reason = self._hspec_phase4_metrics_due(force=force)
        every = max(int(getattr(self, "_hspec_phase4_metrics_every_steps", 1)), 1)
        metrics["hspec/phase4_metrics_every_steps"] = float(every)
        metrics["hspec/phase4_metrics_due"] = 1.0 if due else 0.0
        metrics["hspec/phase4_metrics_sampled"] = 0.0
        metrics["hspec/phase4_metrics_compute_ms"] = 0.0
        metrics["hspec/phase4_metrics_interval_steps"] = 0.0
        metrics["hspec/phase4_metrics_skip_cadence"] = 0.0
        metrics["hspec/phase4_metrics_skip_inflight"] = 0.0
        metrics["hspec/phase4_metrics_reason_first"] = 1.0 if reason == "first" else 0.0
        metrics["hspec/phase4_metrics_reason_cadence"] = 1.0 if reason == "cadence" else 0.0
        metrics["hspec/phase4_metrics_reason_force"] = 1.0 if reason == "force" else 0.0
        metrics["hspec/phase4_metrics_sample_count"] = float(
            getattr(self, "_hspec_phase4_metrics_sample_count", 0)
        )

        if self._hspec_has_inflight_builds():
            if due:
                self._hspec_phase4_metrics_skip_inflight_count += 1
                metrics["hspec/phase4_metrics_skip_inflight"] = 1.0
            metrics["hspec/phase4_metrics_skip_inflight_count"] = float(
                self._hspec_phase4_metrics_skip_inflight_count
            )
            metrics["hspec/phase4_metrics_skip_cadence_count"] = float(
                self._hspec_phase4_metrics_skip_cadence_count
            )
            self._emit_cached_hspec_phase4_gauges(metrics)
            return False

        if not due:
            self._hspec_phase4_metrics_skip_cadence_count += 1
            metrics["hspec/phase4_metrics_skip_cadence"] = 1.0
            metrics["hspec/phase4_metrics_skip_cadence_count"] = float(
                self._hspec_phase4_metrics_skip_cadence_count
            )
            metrics["hspec/phase4_metrics_skip_inflight_count"] = float(
                self._hspec_phase4_metrics_skip_inflight_count
            )
            self._emit_cached_hspec_phase4_gauges(metrics)
            return False

        t0 = time.perf_counter_ns()
        try:
            table_metrics = self.hspec_tables.compute_metrics()
        except Exception:
            self._hspec_phase4_metrics_error_count += 1
            metrics["hspec/phase4_metrics_error_count"] = float(
                self._hspec_phase4_metrics_error_count
            )
            self._emit_cached_hspec_phase4_gauges(metrics)
            return False
        elapsed_ms = float((time.perf_counter_ns() - t0) / 1_000_000.0)
        step = int(getattr(self, "global_steps", 0))
        last = int(getattr(self, "_hspec_last_phase4_metrics_step", -1))
        interval_steps = 1 if last < 0 else max(step - last, 1)
        metrics.update(table_metrics)
        metrics["hspec/phase4_metrics_sampled"] = 1.0
        metrics["hspec/phase4_metrics_compute_ms"] = elapsed_ms
        metrics["hspec/phase4_metrics_interval_steps"] = float(interval_steps)
        self._hspec_last_phase4_metrics_step = step
        self._hspec_last_phase4_metrics_wall_ns = time.time_ns()
        self._hspec_phase4_metrics_sample_count += 1
        metrics["hspec/phase4_metrics_sample_count"] = float(
            self._hspec_phase4_metrics_sample_count
        )
        metrics["hspec/phase4_metrics_skip_cadence_count"] = float(
            self._hspec_phase4_metrics_skip_cadence_count
        )
        metrics["hspec/phase4_metrics_skip_inflight_count"] = float(
            self._hspec_phase4_metrics_skip_inflight_count
        )
        metrics["hspec/phase4_metrics_error_count"] = float(
            self._hspec_phase4_metrics_error_count
        )
        self._update_cached_hspec_phase4_gauges(table_metrics)
        self._emit_cached_hspec_phase4_gauges(metrics)
        return True

    @staticmethod
    def _coerce_hspec_result_metrics(result: object) -> dict[str, float]:
        if not isinstance(result, dict):
            return {}
        coerced: dict[str, float] = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                coerced[str(key)] = float(value)
        return coerced

    @staticmethod
    def _merge_hspec_build_result_metrics(metrics: dict | None, result: object) -> None:
        if metrics is None or not isinstance(result, dict):
            return
        additive_keys = (
            "prompt_count",
            "desc_count",
            "selected_desc_count",
            "legacy_payload_count",
            "build_count_delta",
            "discard_count_delta",
            "build_total_ms",
            "build_validation_ms",
            "build_materialize_ms",
            "build_pca_ms",
            "build_table_add_ms",
            "build_input_rows",
            "build_selected_rows",
            "build_loaded_raw_bytes",
            "build_loaded_fp32_bytes",
            "build_budget_drop_count",
            "build_budget_drop_rows",
            "build_budget_drop_raw_bytes",
            "build_budget_drop_oversize_count",
            "build_rss_cap_skip_count",
            "build_memory_error_count",
            "build_pca_mean_processed_fp32_tile_bytes",
            "build_pca_basis_processed_fp32_tile_bytes",
            "build_pca_reference_processed_fp32_tile_bytes",
            "build_projection_processed_fp32_tile_bytes",
            "build_processed_fp32_tile_bytes",
            "build_projection_tile_count",
            "build_queue_lag_ms_total",
            "build_queue_lag_count",
            "build_queue_reject_count",
            "build_queue_reject_descs",
            "build_queue_reject_bytes",
            "completed_prompt_count",
            "unfinished_prompt_count",
            "build_deadline_hit_count",
            "build_timeout_discard",
            "build_timeout_unfinished_prompts",
            "build_cpu_profile_count",
            "build_cpu_profile_ms",
            "build_cpu_profile_dump_ms",
            "build_cpu_profile_error_count",
            "build_cpu_profile_skipped_count",
            "build_cpu_profile_batches",
        )
        for key in additive_keys:
            value = result.get(key)
            if isinstance(value, (int, float)):
                metric_key = f"hspec/build_result_{key}"
                metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value)
                if key.endswith("_bytes"):
                    mb_key = f"hspec/build_result_{key[:-6]}_mb"
                    metrics[mb_key] = metrics.get(mb_key, 0.0) + float(value) / (1024 * 1024)
        max_keys = (
            "build_actor_rss_mb",
            "build_actor_rss_before_mb",
            "build_actor_rss_after_materialize_mb_max",
            "build_actor_rss_after_pca_mb_max",
            "build_actor_rss_peak_mb",
            "build_actor_rss_delta_mb_max",
            "build_queue_pending_descs",
            "build_queue_pending_bytes",
            "build_queue_lag_ms_max",
            "build_cpu_profile_enabled",
            "build_cpu_profile_top_cumtime_ms_max",
        )
        for key in max_keys:
            value = result.get(key)
            if isinstance(value, (int, float)) and float(value) >= 0:
                metric_key = f"hspec/build_result_{key}_max"
                metrics[metric_key] = max(float(metrics.get(metric_key, 0.0)), float(value))
                if key == "build_actor_rss_mb":
                    metrics["hspec/build_actor_rss_mb_max"] = max(
                        float(metrics.get("hspec/build_actor_rss_mb_max", 0.0)),
                        float(value),
                    )
        lag_count = metrics.get("hspec/build_result_build_queue_lag_count", 0.0)
        if lag_count:
            metrics["hspec/build_result_build_queue_lag_ms_avg"] = (
                metrics.get("hspec/build_result_build_queue_lag_ms_total", 0.0) / lag_count
            )

    @staticmethod
    def _hspec_result_prompt_ids(result: object, key: str) -> tuple[str, ...]:
        if not isinstance(result, dict):
            return ()
        value = result.get(key)
        if not isinstance(value, (list, tuple, set)):
            return ()
        return tuple(str(prompt_id) for prompt_id in value if str(prompt_id))

    def _hspec_record_completed_prompt_ids(self, record: HSpecPendingBuild) -> tuple[str, ...]:
        result = getattr(record, "result_payload", {}) or {}
        completed = self._hspec_result_prompt_ids(result, "completed_prompt_ids")
        if completed:
            return completed
        unfinished = set(self._hspec_result_prompt_ids(result, "unfinished_prompt_ids"))
        return tuple(str(prompt_id) for prompt_id in record.prompt_ids if str(prompt_id) not in unfinished)

    def _hspec_record_unfinished_prompt_ids(self, record: HSpecPendingBuild) -> tuple[str, ...]:
        result = getattr(record, "result_payload", {}) or {}
        unfinished = self._hspec_result_prompt_ids(result, "unfinished_prompt_ids")
        if unfinished:
            return unfinished
        if bool(record.timed_out) and not bool(record.done):
            return tuple(str(prompt_id) for prompt_id in record.prompt_ids if str(prompt_id))
        return ()

    def _discard_hspec_building_for_records(
        self,
        records: list[HSpecPendingBuild],
        *,
        epoch: int,
        metrics: dict | None = None,
        reason: str = "build_timeout_discard",
    ) -> None:
        if not records or not hasattr(self, "hspec_tables"):
            return
        shard_ids = sorted({int(record.shard_id) for record in records})
        try:
            discard_metrics = self.hspec_tables.discard_building_for_shards(
                shard_ids,
                epoch=int(epoch),
                reason=reason,
            )
        except Exception:
            if metrics is not None:
                metrics["hspec/build_timeout_discard_error"] = (
                    metrics.get("hspec/build_timeout_discard_error", 0.0) + 1.0
                )
            print(
                "HSpec: failed to discard timed-out building tables; "
                "continuing training with previous active tables."
            )
            return
        if metrics is not None and isinstance(discard_metrics, dict):
            for key, value in discard_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = metrics.get(key, 0.0) + float(value)

    @staticmethod
    def _hspec_pending_segment_count(records: list[HSpecPendingBuild]) -> int:
        return len({segment for record in records for segment in record.segments})

    @staticmethod
    def _hspec_segments_from_prompt_build_data(prompt_build_data: dict) -> set[object]:
        segments: set[object] = set()
        if not isinstance(prompt_build_data, dict):
            return segments
        try:
            from vllm_ascend.spec_decode.hspec_store import (
                coerce_hspec_desc,
                hspec_segment_key_from_desc,
            )
        except Exception:
            return segments
        for data in prompt_build_data.values():
            if not isinstance(data, list):
                continue
            for item in data:
                if item is None:
                    continue
                try:
                    segments.add(hspec_segment_key_from_desc(coerce_hspec_desc(item)))
                except Exception:
                    continue
        return segments

    def _mark_hspec_segments_gc_deletable(
        self,
        segments: set[object],
        *,
        epoch: int,
        reason: str,
        timing_raw: dict | None = None,
    ) -> None:
        if not segments:
            return
        from vllm_ascend.spec_decode.hspec_store import (
            delete_hspec_segment,
            hspec_raw_store_gc_after_epoch_enabled,
            hspec_record_store_metric,
            update_hspec_segment_manifest_status,
        )

        deleted = 0

        def _mark_and_maybe_delete() -> None:
            nonlocal deleted
            for segment in sorted(segments, key=lambda item: str(getattr(item, "segment_dir", item))):
                try:
                    segment_dir = getattr(segment, "segment_dir", segment)
                    update_hspec_segment_manifest_status(
                        segment_dir,
                        "gc_deletable",
                        extra={
                            "epoch": int(epoch),
                            "gc_reason": str(reason),
                            "build_status": str(reason),
                        },
                    )
                    if hspec_raw_store_gc_after_epoch_enabled():
                        if delete_hspec_segment(segment, caller_confirmed_safe=False):
                            deleted += 1
                except Exception:
                    hspec_record_store_metric("raw_store_epoch_gc_error", 1)
                    print(
                        "HSpec: failed to mark/delete skipped raw segment "
                        f"for reason={reason}; continuing training."
                    )
                    continue

        if timing_raw is None:
            _mark_and_maybe_delete()
        else:
            with marked_timer("hspec_backpressure_raw_store_gc", timing_raw, color="teal"):
                _mark_and_maybe_delete()

        hspec_record_store_metric("raw_store_epoch_gc_segments", len(segments))
        if hspec_raw_store_gc_after_epoch_enabled():
            hspec_record_store_metric("raw_store_epoch_gc_deleted", deleted)
        else:
            hspec_record_store_metric("raw_store_epoch_gc_skipped", len(segments))

    def _wait_hspec_epoch_builds(self, epoch: int, timing_raw: dict | None = None) -> dict[str, float]:
        records: list[HSpecPendingBuild] = getattr(self, "_hspec_pending_build_refs", [])
        if not records:
            self._hspec_last_epoch_build_barrier = HSpecEpochBuildBarrierResult(epoch=int(epoch))
            return {}

        epoch_records = [record for record in records if record.epoch == epoch]
        if not epoch_records:
            unresolved_timed_out = [
                record for record in records
                if bool(record.timed_out) and not bool(record.done)
            ]
            if unresolved_timed_out:
                timed_out_prompt_ids = sorted({
                    str(prompt_id)
                    for record in unresolved_timed_out
                    for prompt_id in record.prompt_ids
                    if str(prompt_id)
                })
                metrics = {
                    "hspec/epoch_build_barrier_timeout_count": 1.0,
                    "hspec/epoch_build_barrier_wait_ms": 0.0,
                    "hspec/build_timeout_unfinished_prompts": float(len(timed_out_prompt_ids)),
                    "hspec/build_timeout_discard": (
                        float(len(timed_out_prompt_ids))
                        if _hspec_build_timeout_discard_unfinished_enabled()
                        else 0.0
                    ),
                    "hspec/partial_swap_count": 0.0,
                    "hspec/partial_swap_completed_prompts": 0.0,
                    "hspec/partial_swap_reused_old_prompts": 0.0,
                }
                self._hspec_last_epoch_build_barrier = HSpecEpochBuildBarrierResult(
                    epoch=int(epoch),
                    metrics=dict(metrics),
                    timed_out_prompt_ids=tuple(timed_out_prompt_ids),
                    timed_out_shard_ids=tuple(sorted({
                        int(record.shard_id) for record in unresolved_timed_out
                    })),
                    timed_out=True,
                )
                return metrics
            self._hspec_last_epoch_build_barrier = HSpecEpochBuildBarrierResult(epoch=int(epoch))
            return {}

        build_result_metrics: dict[str, float] = {}
        timeout_s = _hspec_epoch_build_barrier_timeout_s()
        wait_started = time.perf_counter()

        def _mark_done(record: HSpecPendingBuild, result: object) -> None:
            record.done = True
            record.result_payload = result if isinstance(result, dict) else {}
            record.result_metrics = self._coerce_hspec_result_metrics(result)
            self._merge_hspec_build_result_metrics(build_result_metrics, result)

        def _wait_for_epoch() -> None:
            pending = [record for record in epoch_records if not record.done]
            if not pending:
                return
            if timeout_s <= 0:
                for record in pending:
                    _mark_done(record, ray.get(record.ref))
                return
            ref_to_record = {record.ref: record for record in pending}
            ready_refs, _ = ray.wait(
                list(ref_to_record.keys()),
                num_returns=len(ref_to_record),
                timeout=float(timeout_s),
            )
            for ref in ready_refs:
                _mark_done(ref_to_record[ref], ray.get(ref))

        if timing_raw is None:
            _wait_for_epoch()
        else:
            with marked_timer("hspec_epoch_build_wait", timing_raw, color="teal"):
                _wait_for_epoch()

        wait_ms = float((time.perf_counter() - wait_started) * 1000.0)
        pending_after_wait = [record for record in epoch_records if not record.done]
        for record in pending_after_wait:
            record.timed_out = timeout_s > 0

        completed_prompt_ids: list[str] = []
        timed_out_prompt_ids: list[str] = []
        done_records = [record for record in epoch_records if record.done]
        for record in done_records:
            completed_prompt_ids.extend(self._hspec_record_completed_prompt_ids(record))
            timed_out_prompt_ids.extend(self._hspec_record_unfinished_prompt_ids(record))
        for record in pending_after_wait:
            timed_out_prompt_ids.extend(str(prompt_id) for prompt_id in record.prompt_ids if str(prompt_id))

        completed_prompt_ids = sorted(set(completed_prompt_ids))
        timed_out_prompt_ids = sorted(set(timed_out_prompt_ids))
        timed_out = bool(timed_out_prompt_ids or pending_after_wait)
        ready_shard_ids = tuple(sorted({int(record.shard_id) for record in done_records}))
        timed_out_shard_ids = tuple(sorted({int(record.shard_id) for record in pending_after_wait}))

        build_result_metrics["hspec/epoch_build_barrier_wait_ms"] = wait_ms
        build_result_metrics["hspec/epoch_build_barrier_timeout_count"] = 1.0 if timed_out else 0.0
        build_result_metrics["hspec/build_timeout_unfinished_prompts"] = float(len(timed_out_prompt_ids))
        build_result_metrics["hspec/build_timeout_discard"] = (
            float(len(timed_out_prompt_ids))
            if timed_out and _hspec_build_timeout_discard_unfinished_enabled()
            else 0.0
        )
        build_result_metrics.setdefault("hspec/partial_swap_count", 0.0)
        build_result_metrics.setdefault("hspec/partial_swap_completed_prompts", 0.0)
        build_result_metrics.setdefault("hspec/partial_swap_reused_old_prompts", 0.0)

        self._hspec_last_epoch_build_barrier = HSpecEpochBuildBarrierResult(
            epoch=int(epoch),
            metrics=dict(build_result_metrics),
            completed_prompt_ids=tuple(completed_prompt_ids),
            timed_out_prompt_ids=tuple(timed_out_prompt_ids),
            ready_shard_ids=ready_shard_ids,
            timed_out_shard_ids=timed_out_shard_ids,
            timed_out=timed_out,
        )

        segments = {
            segment
            for record in done_records
            for segment in record.segments
        }
        from vllm_ascend.spec_decode.hspec_store import (
            delete_hspec_segment,
            hspec_raw_store_gc_after_epoch_enabled,
            hspec_record_store_metric,
            update_hspec_segment_manifest_status,
        )

        should_gc_done_segments = (
            bool(segments)
            and (not timed_out or _hspec_build_timeout_discard_unfinished_enabled())
        )
        if should_gc_done_segments and hspec_raw_store_gc_after_epoch_enabled():
            deleted = 0

            def _gc_segments() -> None:
                nonlocal deleted
                for segment in sorted(segments, key=lambda item: str(getattr(item, "segment_dir", item))):
                    try:
                        if hasattr(segment, "segment_dir"):
                            update_hspec_segment_manifest_status(
                                segment.segment_dir,
                                "epoch_build_done",
                                extra={"epoch": int(epoch), "gc_reason": "epoch_build_success"},
                            )
                        if delete_hspec_segment(
                            segment,
                            caller_confirmed_safe=True,
                        ):
                            deleted += 1
                    except Exception:
                        hspec_record_store_metric("raw_store_epoch_gc_error", 1)
                        raise

            if timing_raw is None:
                _gc_segments()
            else:
                with marked_timer("hspec_epoch_raw_store_gc", timing_raw, color="teal"):
                    _gc_segments()
            hspec_record_store_metric("raw_store_epoch_gc_segments", len(segments))
            hspec_record_store_metric("raw_store_epoch_gc_deleted", deleted)
        elif segments:
            hspec_record_store_metric("raw_store_epoch_gc_skipped", len(segments))

        self._hspec_pending_build_refs = [
            record
            for record in records
            # Phase 0/1 static compatibility: the old full-wait predicate was
            # record.epoch != epoch. Step 5 keeps timed-out unfinished records
            # until their actor result is ready and safely discarded.
            if not (record.epoch == epoch and record.done)
        ]
        return build_result_metrics

    def _publish_hspec_epoch_tables_after_barrier(self, epoch: int) -> dict[str, float]:
        barrier = getattr(
            self,
            "_hspec_last_epoch_build_barrier",
            HSpecEpochBuildBarrierResult(epoch=int(epoch)),
        )
        if not bool(barrier.timed_out):
            print(f"HSpec: swap at epoch={epoch} (promote building -> active)")
            metrics = self.hspec_tables.swap(epoch=epoch)
            self._hspec_force_phase4_metrics_next = True
            return metrics

        if _hspec_swap_partial_on_timeout_enabled():
            completed_prompt_ids = list(barrier.completed_prompt_ids)
            if completed_prompt_ids:
                print(
                    "HSpec: partial swap after build timeout "
                    f"epoch={epoch} completed_prompts={len(completed_prompt_ids)} "
                    f"timed_out_prompts={len(barrier.timed_out_prompt_ids)}"
                )
                metrics = self.hspec_tables.swap(
                    epoch=epoch,
                    partial=True,
                    completed_prompt_ids=completed_prompt_ids,
                    timed_out_prompt_ids=list(barrier.timed_out_prompt_ids),
                )
                self._hspec_force_phase4_metrics_next = True
                return metrics
            print(
                "HSpec: partial swap skipped after build timeout "
                f"epoch={epoch}; no completed prompts, keeping previous active."
            )
            return {
                "hspec/partial_swap_count": 0.0,
                "hspec/partial_swap_completed_prompts": 0.0,
                "hspec/partial_swap_reused_old_prompts": 0.0,
            }

        print(
            "HSpec: build timeout without partial swap "
            f"epoch={epoch}; discarding ready building tables and keeping previous active."
        )
        metrics: dict[str, float] = {
            "hspec/partial_swap_count": 0.0,
            "hspec/partial_swap_completed_prompts": 0.0,
            "hspec/partial_swap_reused_old_prompts": 0.0,
        }
        ready_shard_ids = list(barrier.ready_shard_ids)
        if ready_shard_ids:
            self._discard_hspec_building_for_records(
                [
                    HSpecPendingBuild(
                        epoch=int(epoch),
                        ref=None,
                        shard_id=int(shard_id),
                    )
                    for shard_id in ready_shard_ids
                ],
                epoch=int(epoch),
                metrics=metrics,
                reason="build_timeout_partial_disabled_discard",
            )
            self._hspec_force_phase4_metrics_next = True
        return metrics

    def _poll_hspec_builds_nonblocking(self, metrics: dict | None = None) -> None:
        records: list[HSpecPendingBuild] = getattr(self, "_hspec_pending_build_refs", [])
        if not records:
            if metrics is not None:
                metrics["hspec/build_pending_refs"] = 0
                metrics["hspec/build_ready_refs"] = 0
                metrics["hspec/build_done_refs"] = 0
                metrics["hspec/build_pending_records"] = 0
                metrics["hspec/build_pending_segments"] = 0
                metrics["hspec/build_pending_epochs"] = 0
                metrics["hspec/build_pending_epoch_backpressure"] = 0
                metrics["hspec/build_submission_skipped_pending_epochs"] = 0
                metrics["hspec/build_submission_skipped_timeout_cleanup"] = 0
                metrics["hspec/epoch_build_barrier_timeout_count"] = 0
                metrics["hspec/epoch_build_barrier_wait_ms"] = 0
                metrics["hspec/build_timeout_discard"] = 0
                metrics["hspec/build_timeout_unfinished_prompts"] = 0
                metrics["hspec/partial_swap_count"] = 0
                metrics["hspec/partial_swap_completed_prompts"] = 0
                metrics["hspec/partial_swap_reused_old_prompts"] = 0
            return

        active_records = [record for record in records if not record.done]
        ready_refs = []
        timed_out_ready_records: list[HSpecPendingBuild] = []
        if active_records:
            ref_to_record = {record.ref: record for record in active_records}
            ready_refs, _ = ray.wait(
                list(ref_to_record.keys()),
                num_returns=len(ref_to_record),
                timeout=0,
            )
            for ref in ready_refs:
                record = ref_to_record[ref]
                result = ray.get(ref)
                record.done = True
                record.result_payload = result if isinstance(result, dict) else {}
                record.result_metrics = self._coerce_hspec_result_metrics(result)
                self._merge_hspec_build_result_metrics(metrics, result)
                if record.timed_out:
                    timed_out_ready_records.append(record)

        if timed_out_ready_records:
            self._discard_hspec_building_for_records(
                timed_out_ready_records,
                epoch=int(timed_out_ready_records[0].epoch),
                metrics=metrics,
                reason="build_timeout_late_result_discard",
            )
            if _hspec_build_timeout_discard_unfinished_enabled():
                timed_out_segments = {
                    segment
                    for record in timed_out_ready_records
                    for segment in record.segments
                }
                self._mark_hspec_segments_gc_deletable(
                    timed_out_segments,
                    epoch=int(timed_out_ready_records[0].epoch),
                    reason="build_timeout_late_result_discard",
                )
            self._hspec_pending_build_refs = [
                record
                for record in records
                if not (record.timed_out and record.done)
            ]
            records = self._hspec_pending_build_refs

        if metrics is not None:
            metrics["hspec/build_pending_refs"] = sum(not record.done for record in records)
            metrics["hspec/build_ready_refs"] = len(ready_refs)
            metrics["hspec/build_done_refs"] = sum(record.done for record in records)
            metrics["hspec/build_pending_records"] = len(records)
            metrics["hspec/build_pending_segments"] = self._hspec_pending_segment_count(records)
            metrics["hspec/build_pending_epochs"] = len({
                int(record.epoch) for record in records if not record.done
            })
            metrics.setdefault("hspec/build_pending_epoch_backpressure", 0)
            metrics.setdefault("hspec/build_submission_skipped_pending_epochs", 0)
            metrics.setdefault("hspec/build_submission_skipped_timeout_cleanup", 0)
            metrics.setdefault("hspec/epoch_build_barrier_timeout_count", 0)
            metrics.setdefault("hspec/epoch_build_barrier_wait_ms", 0)
            metrics.setdefault("hspec/build_timeout_discard", 0)
            metrics.setdefault("hspec/build_timeout_unfinished_prompts", 0)
            metrics.setdefault("hspec/partial_swap_count", 0)
            metrics.setdefault("hspec/partial_swap_completed_prompts", 0)
            metrics.setdefault("hspec/partial_swap_reused_old_prompts", 0)

    @staticmethod
    def _hspec_dump_object_array(items):
        arr = np.empty((len(items),), dtype=object)
        arr[:] = items
        return arr

    def _hspec_dump_epoch_dir(self, epoch: int) -> str:
        return os.path.join(self._hspec_dump_root, f"epoch_{int(epoch):04d}")

    def _maybe_write_hspec_dump_epoch_meta(
        self,
        epoch: int,
        active_table_version: int,
    ) -> None:
        if not self._hspec_dump_enabled:
            return
        if epoch in self._hspec_dump_epoch_meta_written:
            return

        epoch_dir = self._hspec_dump_epoch_dir(epoch)
        os.makedirs(epoch_dir, exist_ok=True)
        meta = {
            "epoch": int(epoch),
            "global_step_at_first_dump": int(self.global_steps),
            "active_table_version": int(active_table_version),
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "hspec_dump_root": self._hspec_dump_root,
            "hspec_similarity_threshold": float(
                self.config.actor_rollout_ref.rollout.get("hspec_similarity_threshold", 0.9)
            ),
        }
        with open(os.path.join(epoch_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._hspec_dump_epoch_meta_written.add(epoch)

    def _dump_hspec_rollouts_and_tables(
        self,
        epoch: int,
        prompt_build_data: dict,
    ) -> None:
        """Dump current epoch rollouts and the active query table snapshot."""
        if not self._hspec_dump_enabled or not prompt_build_data:
            return

        prompt_ids = list(prompt_build_data.keys())
        active_table_version, active_table_data = self.hspec_tables.prefetch_batch(prompt_ids)
        self._maybe_write_hspec_dump_epoch_meta(epoch, active_table_version)

        epoch_dir = self._hspec_dump_epoch_dir(epoch)
        step_rollout_dir = os.path.join(epoch_dir, "rollouts", f"step_{int(self.global_steps):08d}")
        table_dir = os.path.join(epoch_dir, "tables")
        os.makedirs(step_rollout_dir, exist_ok=True)
        os.makedirs(table_dir, exist_ok=True)

        written_tables = self._hspec_dump_tables_written[epoch]

        for prompt_id, data in prompt_build_data.items():
            prompt_token_ids = data.get("prompt_token_ids")
            table_data = active_table_data.get(prompt_id)
            table_present = table_data is not None

            hidden_states_list = []
            projected_hidden_states_list = []
            projection_available = []

            mean = None
            components = None
            if table_present:
                mean = np.ascontiguousarray(table_data["mean"], dtype=np.float32)
                components = np.ascontiguousarray(table_data["components"], dtype=np.float32)

            for hs in data["hidden_states"]:
                hs_np = np.ascontiguousarray(np.asarray(hs))
                hidden_states_list.append(hs_np)
                if table_present:
                    hs_f32 = hs_np.astype(np.float32, copy=False)
                    proj = np.ascontiguousarray((hs_f32 - mean) @ components.T, dtype=np.float32)
                    projected_hidden_states_list.append(proj)
                    projection_available.append(True)
                else:
                    projected_hidden_states_list.append(None)
                    projection_available.append(False)

            rollout_path = os.path.join(step_rollout_dir, f"{prompt_id}.npz")
            np.savez_compressed(
                rollout_path,
                prompt_id=np.asarray(prompt_id),
                prompt_token_ids=np.asarray(prompt_token_ids if prompt_token_ids is not None else [], dtype=np.int32),
                epoch=np.asarray(int(epoch), dtype=np.int32),
                global_step=np.asarray(int(self.global_steps), dtype=np.int32),
                active_table_version=np.asarray(int(active_table_version), dtype=np.int32),
                table_present=np.asarray(bool(table_present), dtype=np.bool_),
                rewards=np.asarray(data["rewards"], dtype=np.float32),
                response_tokens=self._hspec_dump_object_array(
                    [np.ascontiguousarray(np.asarray(tok, dtype=np.int32)) for tok in data["tokens"]]
                ),
                hidden_states=self._hspec_dump_object_array(hidden_states_list),
                projected_hidden_states=self._hspec_dump_object_array(projected_hidden_states_list),
                projection_available=np.asarray(projection_available, dtype=np.bool_),
            )

            if table_present and prompt_id not in written_tables:
                table_path = os.path.join(table_dir, f"{prompt_id}.npz")
                np.savez_compressed(
                    table_path,
                    prompt_id=np.asarray(prompt_id),
                    prompt_token_ids=np.asarray(prompt_token_ids if prompt_token_ids is not None else [], dtype=np.int32),
                    epoch=np.asarray(int(epoch), dtype=np.int32),
                    active_table_version=np.asarray(int(active_table_version), dtype=np.int32),
                    hspec_similarity_threshold=np.asarray(
                        float(self.config.actor_rollout_ref.rollout.get("hspec_similarity_threshold", 0.9)),
                        dtype=np.float32,
                    ),
                    mean=np.ascontiguousarray(table_data["mean"], dtype=np.float32),
                    components=np.ascontiguousarray(table_data["components"], dtype=np.float32),
                    keys=np.ascontiguousarray(table_data["keys"]),
                    rollout_seqs=self._hspec_dump_object_array(
                        [np.ascontiguousarray(np.asarray(seq, dtype=np.int32)) for seq in table_data["rollout_seqs"]]
                    ),
                    entry_rollout_idx=np.ascontiguousarray(table_data["entry_rollout_idx"], dtype=np.int32),
                    entry_offset=np.ascontiguousarray(table_data["entry_offset"], dtype=np.int32),
                    n_entries=np.asarray(int(table_data["n_entries"]), dtype=np.int32),
                    wnd_size=np.asarray(int(table_data["wnd_size"]), dtype=np.int32),
                )
                written_tables.add(prompt_id)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            _extract_and_sum_hspec_rollout_metrics(test_output_gen_batch_padded.meta_info)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        from verl.workers.config.rollout import resolve_rollout_speculative_method

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        use_hspec_decode = (
            resolve_rollout_speculative_method(
                self.config.actor_rollout_ref.rollout
            )
            == "hspec"
        )
        if use_hspec_decode and self.async_rollout_mode:
            raise NotImplementedError(
                "HSpec rollout collection is only implemented for sync vLLM rollout in the current migration."
            )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        if use_hspec_decode:
            from vllm_ascend.spec_decode.hspec_table import get_hspec_tables

            similarity_threshold = self.config.actor_rollout_ref.rollout.get(
                "hspec_similarity_threshold", 0.9
            )
            hspec_n_components = self.config.actor_rollout_ref.rollout.get(
                "hspec_n_components", 64
            )
            hspec_max_entries = self.config.actor_rollout_ref.rollout.get(
                "hspec_max_entries_per_prompt", 10000
            )
            self.hspec_tables = get_hspec_tables(
                similarity_threshold=similarity_threshold,
                n_components=hspec_n_components,
                max_entries_per_prompt=hspec_max_entries,
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["hspec_epoch"] = epoch

                data_rebalance = self.config.actor_rollout_ref.rollout.data_rebalance if hasattr(
                    self.config.actor_rollout_ref.rollout, 'data_rebalance') else True
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n,
                                             interleave=not data_rebalance)
                if data_rebalance:
                    interleave_indices = torch.arange(gen_batch.batch.batch_size[0]).view(
                        -1, batch.batch.batch_size[0]).transpose(1, 0).reshape(-1)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        if data_rebalance:
                            gen_batch_output.reorder(interleave_indices)
                        
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        metrics.update(_extract_and_sum_hspec_rollout_metrics(gen_batch_output.meta_info))

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            
                            if data_rebalance:
                                gen_baseline_output.reorder(interleave_indices)
                            metrics.update(_extract_and_sum_hspec_rollout_metrics(gen_baseline_output.meta_info))

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if self.config.actor_rollout_ref.actor.recompute_old_log_prob:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                        else:
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if use_hspec_decode:
                        with marked_timer("update_hspec_tables", timing_raw, color="teal"):
                            self._poll_hspec_builds_nonblocking(metrics)
                            force_hspec_phase4_metrics = bool(
                                getattr(self, "_hspec_force_phase4_metrics_next", False)
                            )
                            sampled_hspec_phase4_metrics = self._maybe_collect_hspec_phase4_metrics(
                                metrics,
                                force=force_hspec_phase4_metrics,
                            )
                            self._hspec_force_phase4_metrics_next = bool(
                                force_hspec_phase4_metrics
                                and not sampled_hspec_phase4_metrics
                                and self._hspec_has_inflight_builds()
                            )
                            from vllm_ascend.spec_decode.hspec_store import (
                                coerce_hspec_desc,
                                get_hspec_num_shards,
                                hspec_legacy_dataproto_hs_enabled,
                                hspec_record_store_metric,
                                hspec_strict_descriptor_mode_enabled,
                                hspec_step0_runtime_asserts_enabled,
                            )
                            from vllm_ascend.spec_decode.hspec_utils import (
                                prompt_id_from_token_ids,
                                stable_partition_id,
                            )

                            hspec_num_shards = get_hspec_num_shards()
                            legacy_hspec_dataproto_hs = hspec_legacy_dataproto_hs_enabled()
                            strict_descriptor_mode = hspec_strict_descriptor_mode_enabled()
                            step0_runtime_asserts = hspec_step0_runtime_asserts_enabled()
                            if strict_descriptor_mode:
                                forbidden = ("rollout_hidden_states", "rollout_hspec_tokens")
                                present = [key for key in forbidden if key in batch.non_tensor_batch]
                                if present:
                                    hspec_record_store_metric("strict_descriptor_violation", len(present))
                                    raise RuntimeError(
                                        "HSpec strict descriptor mode forbids legacy trainer "
                                        f"payload keys in descriptor mode: {present}"
                                    )
                            if legacy_hspec_dataproto_hs:
                                prompt_build_data: dict = defaultdict(
                                    lambda: {
                                        "hidden_states": [],
                                        "tokens": [],
                                        "rewards": [],
                                        "prompt_token_ids": None,
                                    }
                                )
                            else:
                                prompt_build_data: dict = defaultdict(list)
                            _hspec_skip = 0
                            _hspec_none_count = 0
                            _hspec_empty_resp_count = 0
                            _hspec_align_fail_count = 0
                            _hspec_align_fail_trainer_len = 0
                            _hspec_align_fail_upstream_hs = 0
                            _hspec_align_fail_both = 0
                            _hspec_align_fail_unknown = 0
                            _hspec_align_debug_logged = 0

                            for i in range(len(batch)):
                                batch_item = batch[i]
                                hs = None
                                desc_obj = None
                                desc_len = None
                                hspec_desc = batch_item.non_tensor_batch.get("hspec_desc")
                                if legacy_hspec_dataproto_hs:
                                    hs = batch_item.non_tensor_batch.get("rollout_hidden_states")
                                    if hs is None:
                                        _hspec_skip += 1
                                        _hspec_none_count += 1
                                        continue
                                else:
                                    if hspec_desc is None:
                                        _hspec_skip += 1
                                        _hspec_none_count += 1
                                        continue
                                    try:
                                        desc_obj = coerce_hspec_desc(hspec_desc)
                                        desc_len = int(desc_obj.length)
                                    except Exception:
                                        _hspec_skip += 1
                                        _hspec_align_fail_count += 1
                                        _hspec_align_fail_unknown += 1
                                        continue

                                if not legacy_hspec_dataproto_hs:
                                    # Descriptor/token file is the source of truth
                                    # for table-building alignment. Do not infer
                                    # length from padded responses because
                                    # pad_token_id may equal eos_token_id.
                                    response = [0] * desc_len
                                    response_before_trim = list(response)
                                else:
                                    hspec_tokens = batch_item.non_tensor_batch.get("rollout_hspec_tokens")
                                    if hspec_tokens is not None:
                                        response = [int(x) for x in list(hspec_tokens)]
                                        response_before_trim = list(response)
                                    else:
                                        response = batch_item.batch["responses"].cpu().numpy().tolist()
                                        response_before_trim = list(response)
                                        try:
                                            pad_idx = response.index(self.tokenizer.pad_token_id)
                                            response = response[:pad_idx]
                                        except ValueError:
                                            pass

                                if len(response) == 0:
                                    _hspec_skip += 1
                                    _hspec_empty_resp_count += 1
                                    continue

                                if legacy_hspec_dataproto_hs:
                                    align_mismatch = (
                                        hasattr(hs, "shape")
                                        and hs.ndim == 2
                                        and hs.shape[0] != len(response)
                                    )
                                else:
                                    # Descriptor mode has already made
                                    # response length equal to desc.length.
                                    # The builder validates the mmap-backed
                                    # hidden rows and token file again before
                                    # admitting the trajectory.
                                    align_mismatch = False
                                if align_mismatch:
                                    _hspec_skip += 1
                                    _hspec_align_fail_count += 1
                                    debug_meta = batch_item.non_tensor_batch.get("hspec_rollout_debug")
                                    raw_response_len = None
                                    raw_first_pad_idx = None
                                    hs_len_debug = None
                                    hspec_token_len_debug = None
                                    hs_source = None
                                    response_head = None
                                    response_tail = None
                                    if isinstance(debug_meta, dict):
                                        raw_response_len = debug_meta.get("raw_response_len")
                                        raw_first_pad_idx = debug_meta.get("raw_response_first_pad_index")
                                        hs_len_debug = debug_meta.get("hs_len")
                                        hspec_token_len_debug = debug_meta.get("hspec_token_len")
                                        hs_source = debug_meta.get("hs_source")
                                        response_head = debug_meta.get("response_head")
                                        response_tail = debug_meta.get("response_tail")
                                    trainer_len_mismatch = (
                                        raw_response_len is not None and int(raw_response_len) != len(response)
                                    )
                                    upstream_hs_mismatch = (
                                        hspec_token_len_debug is not None
                                        and hs_len_debug is not None
                                        and int(hspec_token_len_debug) >= 0
                                        and int(hs_len_debug) >= 0
                                        and int(hs_len_debug) != int(hspec_token_len_debug)
                                    )
                                    if not legacy_hspec_dataproto_hs and desc_len is not None:
                                        upstream_hs_mismatch = int(desc_len) != len(response)
                                    if trainer_len_mismatch and upstream_hs_mismatch:
                                        _hspec_align_fail_both += 1
                                        align_reason = "both"
                                    elif trainer_len_mismatch:
                                        _hspec_align_fail_trainer_len += 1
                                        align_reason = "trainer_trim"
                                    elif upstream_hs_mismatch:
                                        _hspec_align_fail_upstream_hs += 1
                                        align_reason = "upstream_hs"
                                    else:
                                        _hspec_align_fail_unknown += 1
                                        align_reason = "unknown"

                                    if (
                                        self._hspec_align_debug
                                        and _hspec_align_debug_logged < self._hspec_align_debug_max_logs
                                    ):
                                        _hspec_align_debug_logged += 1
                                        try:
                                            prompt_token_ids = batch_item.non_tensor_batch["vllm_inputs"]
                                            prompt_id_dbg = prompt_id_from_token_ids(prompt_token_ids)
                                        except Exception:
                                            prompt_id_dbg = "<prompt_id_error>"
                                        print(
                                            "HSPEC ALIGN DEBUG: "
                                            f"epoch={epoch} step={self.global_steps} item={i} "
                                            f"reason={align_reason} prompt_id={prompt_id_dbg} "
                                            f"trainer_trimmed_len={len(response)} "
                                            f"padded_response_len={len(response_before_trim)} "
                                            f"hs_len={int(hs.shape[0]) if hasattr(hs, 'shape') else desc_len} "
                                            f"raw_response_len={raw_response_len} "
                                            f"hspec_token_len={hspec_token_len_debug} "
                                            f"raw_first_pad_idx={raw_first_pad_idx} "
                                            f"hs_source={hs_source} "
                                            f"response_head={response_head} "
                                            f"response_tail={response_tail}"
                                        )
                                    continue

                                prompt_token_ids = batch_item.non_tensor_batch["vllm_inputs"]
                                prompt_id = prompt_id_from_token_ids(prompt_token_ids)
                                reward = batch_item.batch["token_level_scores"].sum().item()

                                if legacy_hspec_dataproto_hs:
                                    prompt_build_data[prompt_id]["hidden_states"].append(hs)
                                    prompt_build_data[prompt_id]["tokens"].append(response)
                                    prompt_build_data[prompt_id]["rewards"].append(reward)
                                    if prompt_build_data[prompt_id]["prompt_token_ids"] is None:
                                        prompt_build_data[prompt_id]["prompt_token_ids"] = list(prompt_token_ids)
                                else:
                                    existing_prompt_id = str(getattr(desc_obj, "prompt_id", "") or "")
                                    if not existing_prompt_id:
                                        hspec_record_store_metric("prompt_id_empty_from_rollout", 1)
                                        if strict_descriptor_mode:
                                            raise RuntimeError(
                                                "HSpec descriptor path received an empty prompt_id "
                                                f"before trainer build aggregation: step={self.global_steps} "
                                                f"epoch={epoch} request_id={getattr(desc_obj, 'request_id', '<unknown>')!r}"
                                            )
                                    elif existing_prompt_id != prompt_id:
                                        hspec_record_store_metric("prompt_id_conflict", 1)
                                        if strict_descriptor_mode:
                                            raise RuntimeError(
                                                "HSpec descriptor prompt_id conflict before trainer build "
                                                f"aggregation: step={self.global_steps} epoch={epoch} "
                                                f"request_id={getattr(desc_obj, 'request_id', '<unknown>')!r} "
                                                f"desc.prompt_id={existing_prompt_id!r} prompt_id_from_vllm_inputs={prompt_id!r}"
                                            )
                                    desc_obj = desc_obj.with_updates(
                                        prompt_id=prompt_id,
                                        shard_id=stable_partition_id(
                                            prompt_id,
                                            hspec_num_shards,
                                        ),
                                        reward=float(reward),
                                    )
                                    prompt_build_data[prompt_id].append(desc_obj)

                            if _hspec_skip > 0:
                                print(
                                    f"HSpec: skipped {_hspec_skip} samples "
                                    f"(hs_none={_hspec_none_count}, "
                                    f"empty_resp={_hspec_empty_resp_count}, "
                                    f"align_fail={_hspec_align_fail_count}, "
                                    f"align_fail_trainer_trim={_hspec_align_fail_trainer_len}, "
                                    f"align_fail_upstream_hs={_hspec_align_fail_upstream_hs}, "
                                    f"align_fail_both={_hspec_align_fail_both}, "
                                    f"align_fail_unknown={_hspec_align_fail_unknown})"
                                )

                            if (
                                prompt_build_data
                                and self._hspec_dump_enabled
                                and legacy_hspec_dataproto_hs
                            ):
                                with marked_timer("hspec_dump", timing_raw, color="teal"):
                                    self._dump_hspec_rollouts_and_tables(
                                        epoch,
                                        dict(prompt_build_data),
                                    )
                            if prompt_build_data:
                                pending_build_records = [
                                    record
                                    for record in getattr(self, "_hspec_pending_build_refs", [])
                                    if not record.done
                                ]
                                unresolved_timeout = any(
                                    bool(record.timed_out)
                                    for record in pending_build_records
                                )
                                pending_epochs = {
                                    int(record.epoch)
                                    for record in pending_build_records
                                }
                                max_pending_epochs = _hspec_build_max_pending_epochs()
                                would_pending_epochs = set(pending_epochs)
                                would_pending_epochs.add(int(epoch))
                                skip_hspec_build = (
                                    unresolved_timeout
                                    or (
                                        max_pending_epochs > 0
                                        and len(would_pending_epochs) > max_pending_epochs
                                    )
                                )
                                if skip_hspec_build:
                                    metrics["hspec/build_pending_epoch_backpressure"] = (
                                        metrics.get("hspec/build_pending_epoch_backpressure", 0.0) + 1.0
                                    )
                                    metrics["hspec/build_submission_skipped_pending_epochs"] = (
                                        metrics.get("hspec/build_submission_skipped_pending_epochs", 0.0) + 1.0
                                    )
                                    if unresolved_timeout:
                                        metrics["hspec/build_submission_skipped_timeout_cleanup"] = (
                                            metrics.get(
                                                "hspec/build_submission_skipped_timeout_cleanup",
                                                0.0,
                                            ) + 1.0
                                        )
                                    if not legacy_hspec_dataproto_hs:
                                        skipped_segments = self._hspec_segments_from_prompt_build_data(
                                            dict(prompt_build_data)
                                        )
                                        self._mark_hspec_segments_gc_deletable(
                                            skipped_segments,
                                            epoch=int(epoch),
                                            reason="build_skipped_backpressure",
                                            timing_raw=timing_raw,
                                        )
                                    metrics["hspec/build_submitted_refs"] = 0
                                    metrics["hspec/build_submitted_segments"] = 0
                                else:
                                    barrier_timeout_s = _hspec_epoch_build_barrier_timeout_s()
                                    submitted_time_ns = time.time_ns()
                                    deadline_ns = (
                                        submitted_time_ns + int(barrier_timeout_s * 1_000_000_000)
                                        if barrier_timeout_s > 0 else 0
                                    )
                                    if legacy_hspec_dataproto_hs:
                                        ray_hspec_tasks = self.hspec_tables.build_tables_async_legacy(
                                            dict(prompt_build_data),
                                            epoch=epoch,
                                            submitted_time_ns=submitted_time_ns,
                                            deadline_ns=deadline_ns,
                                        )
                                    else:
                                        ray_hspec_tasks = self.hspec_tables.build_tables_async(
                                            dict(prompt_build_data),
                                            epoch=epoch,
                                            submitted_time_ns=submitted_time_ns,
                                            deadline_ns=deadline_ns,
                                        )
                                    pending_records = []
                                    for submission in ray_hspec_tasks:
                                        submission_submitted_time_ns = int(
                                            getattr(submission, "submitted_time_ns", 0) or time.time_ns()
                                        )
                                        submission_deadline_ns = int(
                                            getattr(submission, "deadline_ns", 0) or deadline_ns
                                        )
                                        pending_records.append(
                                            HSpecPendingBuild(
                                                epoch=epoch,
                                                ref=submission.ref,
                                                shard_id=int(submission.shard_id),
                                                segments=submission.segments,
                                                prompt_ids=submission.prompt_ids,
                                                submitted_time_ns=submission_submitted_time_ns,
                                                deadline_ns=submission_deadline_ns,
                                                legacy=bool(getattr(submission, "legacy", False)),
                                            )
                                        )
                                    self._hspec_pending_build_refs.extend(pending_records)
                                    metrics["hspec/build_submitted_refs"] = len(ray_hspec_tasks)
                                    metrics["hspec/build_submitted_segments"] = len(
                                        {segment for submission in ray_hspec_tasks for segment in submission.segments}
                                    )
                            self._drop_hspec_non_tensor_fields(batch)
                            if step0_runtime_asserts:
                                forbidden_after_drop = (
                                    "hspec_desc",
                                    "rollout_hidden_states",
                                    "rollout_hspec_tokens",
                                    "hspec_rollout_debug",
                                )
                                present = [
                                    key for key in forbidden_after_drop
                                    if key in batch.non_tensor_batch
                                ]
                                if present:
                                    hspec_record_store_metric(
                                        "strict_descriptor_violation",
                                        len(present),
                                    )
                                    raise RuntimeError(
                                        "HSpec Step0 invariant failed: HSpec non-tensor "
                                        f"fields survived cleanup before actor update: {present}"
                                    )

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics["hspec/driver_rss_gb"] = psutil.Process().memory_info().rss / (1024**3)
                metrics["hspec/ray_object_store_used_mb"] = _maybe_get_ray_object_store_used_mb()
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if use_hspec_decode:
                        barrier_metrics = self._wait_hspec_epoch_builds(epoch, timing_raw)
                        swap_metrics = self._publish_hspec_epoch_tables_after_barrier(epoch)
                        if swap_metrics:
                            barrier_metrics.update(swap_metrics)
                        if barrier_metrics:
                            barrier_metrics.update({
                                "training/global_step": self.global_steps,
                                "training/epoch": epoch,
                            })
                            logger.log(data=barrier_metrics, step=self.global_steps)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

            if use_hspec_decode:
                barrier_metrics = self._wait_hspec_epoch_builds(epoch, timing_raw)
                swap_metrics = self._publish_hspec_epoch_tables_after_barrier(epoch)
                if swap_metrics:
                    barrier_metrics.update(swap_metrics)
                if barrier_metrics:
                    barrier_metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    })
                    logger.log(data=barrier_metrics, step=self.global_steps)
