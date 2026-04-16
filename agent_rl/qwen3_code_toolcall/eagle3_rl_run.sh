# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
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
# Note: This script uses standard verl training arguments common to all
# verl-based RL recipes. Any similarity to other verl training scripts
# reflects the shared framework interface, not code copying.


set -x

export PYTHONUNBUFFERED=1
export VLLM_ASCEND_ENABLE_NZ=0

# ── PATHS ────────────────────────────────────────────────────────────
DRAFT_MODEL_PATH=/path/models/AngelSlim--Qwen3-1.7B_eagle3
ASCEND_HOME_TOOLKIT=/path/Ascend/
MODEL_PATH=/path/models/Qwen3-1.7B
DATA_PATH=/path/data


# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_PROMPT_LENGTH=2048
RES_LENGTH=16384
ROLLOUT_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
TRAIN_TEMPERATURE=0.9
GROUP_SIZE=8
MAX_TURNS=16
AGENT_NUM_WORKERS=8
NNODES=1
SP=1

# ── Speculative Decoding ───────────────────────────────────────────────────────
NUM_SPECULATIVE_TOKENS=2

# ── Model Paths ────────────────────────────────────────────────────────────────
TOOL_CONFIG="$PWD/tool_config/scalebox_tool_config.yaml"

# ── Naming ─────────────────────────────────────────────────────────────────────
PROJECT_NAME="verl_sandbox_code_specrl"
EXP_NAME_BASE=1.5B_L$(($RES_LENGTH / 1024))k
MODEL_NAME=$(basename $MODEL_PATH)
EXP_NAME=${EXP_NAME_BASE}-${MODEL_NAME}-bs${ROLLOUT_BATCH_SIZE}-minibs${PPO_MINI_BATCH_SIZE}-gs${GROUP_SIZE}-temp${TRAIN_TEMPERATURE}-${NNODES}nodes

# ── Ascend Environment ─────────────────────────────────────────────────────────
source $ASCEND_HOME_TOOLKIT/cann-8.5.0/set_env.sh
source $ASCEND_HOME_TOOLKIT/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MAX_TOKEN_LEN=$(((RES_LENGTH + MAX_PROMPT_LENGTH) / SP))

# ── Training ───────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=naive \
    \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/validation.parquet \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RES_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.tool_config_path=$TOOL_CONFIG \
    \
    custom_reward_function.path=scalebox.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.sandbox_fusion_url='http://localhost:8080/common_evaluate_batch' \
    +custom_reward_function.reward_kwargs.return_dict=True \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.load_format=auto \
    \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm={speculative_config: {method: eagle3, num_speculative_tokens: '"$NUM_SPECULATIVE_TOKENS"', model: '"$DRAFT_MODEL_PATH"'}}' \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.device=npu \
    | tee ./coderl_spec_$(date +%Y%m%d_%H%M%S).log \
    $@
