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

# export ASCEND_LAUNCH_BLOCKING=1

# Model and dataset
HOME=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_API_LIB="${CUSTOM_OPP_PATH}/op_api/lib"

if [ -d "${CUSTOM_OPP_PATH}" ]; then
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH}"
fi

if [ -d "${CUSTOM_OP_API_LIB}" ]; then
    export LD_LIBRARY_PATH="${CUSTOM_OP_API_LIB}:${LD_LIBRARY_PATH}"
fi

export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_HISTORY_TREE_DEBUG=0
export VLLM_HISTORY_TREE_MAX_SPEC_REQS=64
export VLLM_DYNAMIC_RL_ENABLE_EAGLE=${VLLM_DYNAMIC_RL_ENABLE_EAGLE:-1}
export VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ=${VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ:-8}
export VLLM_DYNAMIC_RL_EAGLE_PROBE_MAX_BSZ=${VLLM_DYNAMIC_RL_EAGLE_PROBE_MAX_BSZ:-2}
export VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE=${VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE:-64}
export VLLM_DYNAMIC_RL_POLICY=${VLLM_DYNAMIC_RL_POLICY:-timing_guard}
export VLLM_DYNAMIC_RL_HISTORY_WARMUP_RECORDS=${VLLM_DYNAMIC_RL_HISTORY_WARMUP_RECORDS:-1}
export VLLM_DYNAMIC_RL_COMPARE_AFTER_STEPS=${VLLM_DYNAMIC_RL_COMPARE_AFTER_STEPS:-64}
export VLLM_DYNAMIC_RL_EAGLE_PROBE_STEPS=${VLLM_DYNAMIC_RL_EAGLE_PROBE_STEPS:-1}
export VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_STEPS=${VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_STEPS:-8192}
export VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_GROWTH=${VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_GROWTH:-4.0}
export VLLM_DYNAMIC_RL_EAGLE_ENFORCE_EAGER=${VLLM_DYNAMIC_RL_EAGLE_ENFORCE_EAGER:-1}
export VLLM_DYNAMIC_RL_EAGLE_COLD_START_GUARD=${VLLM_DYNAMIC_RL_EAGLE_COLD_START_GUARD:-1}
export VLLM_ASCEND_DEBUG_TREE_DRAFT=0
export VLLM_ASCEND_DEBUG_TREE_LAYOUT=0
export VLLM_ASCEND_DEBUG_SPEC_COMPARE=0
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_SPEC_TIMING=1
export VLLM_ASCEND_SPEC_TIMING_LOG_EVERY=${VLLM_ASCEND_SPEC_TIMING_LOG_EVERY:-200}
export VLLM_ASCEND_SPEC_TIMING_FIRST_N=${VLLM_ASCEND_SPEC_TIMING_FIRST_N:-5}
export VLLM_DYNAMIC_RL_DEBUG=${VLLM_DYNAMIC_RL_DEBUG:-1}
# export VLLM_DYNAMIC_RL_DEBUG_FIRST_N=32

CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen3-30B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}
DISTCP_PATH=${DISTCP_PATH:-"/home/data/Qwen3-30B-A3B_megatron"}

# configs
NODES=1
GPU_MEMORY_UTILIZATION=0.87
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=16384 # 16384
MAX_NUM_SEQS=64
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.9}

INFER_TP=4
INFER_DP=$((NODES * 16 / INFER_TP))

TRAIN_BATCH_SIZE=64
GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE))
ROLLOUT_LOG_PATH=${VLLM_DYNAMIC_RL_LOG_PATH:-outputs/rl/test.txt}
ROLLOUT_LENGTH_DIR=${ROLLOUT_LENGTH_DIR:-outputs/rl/rollout_length}

OUT="${OUT:-/workspace/cann-recipes-train/llm_rl/qwen3/output/train_grpo_hspec-30b.txt}"

mkdir -p "$(dirname "${ROLLOUT_LOG_PATH}")" "${ROLLOUT_LENGTH_DIR}"
{
    echo
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') dynamic_rl run ====="
    echo "log_path=${ROLLOUT_LOG_PATH}"
    echo "rollout_length_dir=${ROLLOUT_LENGTH_DIR}"
    echo "spec_method=dynamic_rl"
    echo "VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ=${VLLM_DYNAMIC_RL_EAGLE_MAX_BSZ}"
    echo "VLLM_DYNAMIC_RL_EAGLE_PROBE_MAX_BSZ=${VLLM_DYNAMIC_RL_EAGLE_PROBE_MAX_BSZ}"
    echo "VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE=${VLLM_DYNAMIC_RL_HISTORY_UPPER_BSZ_EXCLUSIVE}"
    echo "VLLM_DYNAMIC_RL_POLICY=${VLLM_DYNAMIC_RL_POLICY}"
    echo "VLLM_DYNAMIC_RL_HISTORY_WARMUP_RECORDS=${VLLM_DYNAMIC_RL_HISTORY_WARMUP_RECORDS}"
    echo "VLLM_DYNAMIC_RL_COMPARE_AFTER_STEPS=${VLLM_DYNAMIC_RL_COMPARE_AFTER_STEPS}"
    echo "VLLM_DYNAMIC_RL_EAGLE_PROBE_STEPS=${VLLM_DYNAMIC_RL_EAGLE_PROBE_STEPS}"
    echo "VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_STEPS=${VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_STEPS}"
    echo "VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_GROWTH=${VLLM_DYNAMIC_RL_EAGLE_COOLDOWN_GROWTH}"
    echo "VLLM_DYNAMIC_RL_EAGLE_ENFORCE_EAGER=${VLLM_DYNAMIC_RL_EAGLE_ENFORCE_EAGER}"
    echo "VLLM_DYNAMIC_RL_EAGLE_COLD_START_GUARD=${VLLM_DYNAMIC_RL_EAGLE_COLD_START_GUARD}"
    echo "VLLM_ASCEND_SPEC_TIMING_LOG_EVERY=${VLLM_ASCEND_SPEC_TIMING_LOG_EVERY}"
    echo "VLLM_ASCEND_SPEC_TIMING_FIRST_N=${VLLM_ASCEND_SPEC_TIMING_FIRST_N}"
} >> "${ROLLOUT_LOG_PATH}"


python3 -m verl.trainer.main_ppo  --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.dataset_fraction=0.004\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[4,8,16,24,32,48,64]' \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example_deepscaler' \
    trainer.experiment_name='qwen3_30b_verl_true_weights' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=${NODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=5 \
    +trainer.rollout_length_dir="${ROLLOUT_LENGTH_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[11],[13],[13],[11]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    > "${OUT}" 2>&1 "$@"

    # actor_rollout_ref.rollout.spec_method='dynamic_rl' \
    # actor_rollout_ref.rollout.eagle3_draft_model='/home/data/Qwen3-30B-moe-eagle3' \
    # actor_rollout_ref.rollout.spec_num_speculative_tokens=4 \
    # actor_rollout_ref.rollout.speculative_token_tree="'[(0,),(0,0),(0,0,0),(0,0,0,0)]'" \
