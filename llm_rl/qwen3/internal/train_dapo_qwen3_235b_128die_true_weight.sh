# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# Model and dataset
HOME=$(pwd)
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
MODEL_PATH=${MODEL_PATH:-"${HOME}/Qwen3-235B-A22B-hf"}
DISTCP_PATH="./your_sharded_weights" # modify it to path of sharded model weights
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/dapo_math/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/dapo_math/aime-2024.parquet"}

# configs
NODES=8
GPU_MEMORY_UTILIZATION=0.85
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=34816
MAX_NUM_SEQS=64
CUDAGRAPH_SIZES='[8,16,32,64]'

INFER_TP=4
INFER_DP=$((NODES * 16 / INFER_TP))

TRAIN_TP=4
TRAIN_PP=4
TRAIN_CP=4
TRAIN_EP=$((NODES * 16 / TRAIN_PP))

TRAIN_BATCH_SIZE=128
GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE))
MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / TRAIN_CP))

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 8))
overlong_penalty_factor=0.1

# Rollout Importance Sampling parameters
rollout_is=True
rollout_is_threshold=2.0
rollout_is_threshold_lower=null # No lower bound
rollout_is_level=token  # token-level
rollout_is_mode=truncate    # truncate mode
rollout_is_veto_threshold=null  # No veto

loss_agg_mode="token-mean"


python3 -m verl.trainer.main_dapo --config-path="${CONFIG_DIR}" \
    --config-name='dapo_megatron_trainer.yaml'\
    data.prompt_key=prompt  \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc  \
    algorithm.filter_groups.max_num_gen_batches=10  \
    algorithm.kl_ctrl.kl_coef=0.0   \
    algorithm.rollout_is=${rollout_is}  \
    algorithm.rollout_is_threshold=${rollout_is_threshold}  \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower}  \
    algorithm.rollout_is_level=${rollout_is_level}  \
    algorithm.rollout_is_mode=${rollout_is_mode}    \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold}    \
    actor_rollout_ref.actor.use_kl_loss=False    \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}    \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}  \
    actor_rollout_ref.actor.clip_ratio_c=10.0   \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}  \
    actor_rollout_ref.actor.optim.clip_grad=10000   \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP}  \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1  \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True  \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.load_weight=True  \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True  \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${DISTCP_PATH} \
    actor_rollout_ref.actor.use_dynamic_bsz=True    \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}  \
    actor_rollout_ref.actor.recompute_old_log_prob=True    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=${CUDAGRAPH_SIZES} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False   \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}   \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.9   \
    actor_rollout_ref.rollout.top_k=50  \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False  \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${DISTCP_PATH} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_dapo_example' \
    trainer.experiment_name='qwen3_235b_dapo_verl_true_weights' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=${NODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers="[[23],[24],[24],[23]]" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=23 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=23 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${TRAIN_CP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_cp_send_recv_overlap=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo='megatron_cp_algo' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_ring_attention_update=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.cp_window_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_zero_memory='level1' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gemm_gradient_accumulation_fusion=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=sam \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=3 $@
