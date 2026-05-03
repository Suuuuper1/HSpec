#!/bin/bash
#
# HSpec GRPO validation script adapted for the current CANN qwen3 platform.
# Goal: preserve the original train_grpo_hspec.sh training/debug/profile knobs
# as much as possible, while making only the minimal compatibility changes
# required by the new ray_start_npu.sh + current vllm/vllm-ascend/verl stack.
#
# Compatibility updates relative to the old standalone script:
# 1. Connect to the Ray cluster launched by ray_start_npu.sh via address=auto.
# 2. Keep rollout strictly sync; current HSpec migration only supports sync vLLM rollout.
# 3. Force vllm_ascend speculative batch threshold off via ENV_SCRIPT
#    (VLLM_SPECULATIVE_BATCH_SIZE_THRE=-1), so HSpec is never silently disabled.
# 4. Keep the original model / dataset / HSpec debug-trace-profile settings.
#
set -euo pipefail

# Paths and data: keep the original HSpec validation choices.
HOME=$(pwd)
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen2.5-1.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/workspace/cann-recipes-train/llm_rl/qwen3/dataset/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"/workspace/cann-recipes-train/llm_rl/qwen3/dataset/gsm8k/test.parquet"}

# Output log file
OUT=${OUT:-"/workspace/output/train_grpo_hspec.txt"}
mkdir -p "$(dirname "${OUT}")"

# Keep the original data-size switching behavior for dump mode.
if [ "${HSPEC_DUMP:-0}" = "0" ]; then
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-40}"
    ROLLOUT_N="${ROLLOUT_N:-5}"
else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
    ROLLOUT_N="${ROLLOUT_N:-5}"
    TRAIN_FILE="${TRAIN_FILE:-"/workspace/cann-recipes-train/llm_rl/qwen3/dataset/gsm8k/train.parquet"}"
    TEST_FILE="${TEST_FILE:-"/workspace/cann-recipes-train/llm_rl/qwen3/dataset/gsm8k/train.parquet"}"
fi

set -x

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.num_cpus=128 \
    +ray_kwargs.ray_init.address=auto \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG='"'"${HSPEC_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TRACE='"'"${HSPEC_TRACE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP='"'"${HSPEC_DUMP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP_DIR='"'"${HSPEC_DUMP_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DISABLE_NUMBA_REBUILD='"'"${HSPEC_DISABLE_NUMBA_REBUILD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ROWS='"'"${HSPEC_NUMBA_REBUILD_MIN_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ELEMS='"'"${HSPEC_NUMBA_REBUILD_MIN_ELEMS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG='"'"${HSPEC_ALIGN_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG_MAX_LOGS='"'"${HSPEC_ALIGN_DEBUG_MAX_LOGS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG_PREVIEW='"'"${HSPEC_ALIGN_DEBUG_PREVIEW}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ENTRY='"'"${HSPEC_ENTRY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.MATCH_WND='"'"${MATCH_WND}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ADVAN_NGRAM='"'"${HSPEC_ADVAN_NGRAM}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN='"'"${HSPEC_GEN}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN_REQ_IDX='"'"${HSPEC_GEN_REQ_IDX}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN_MAX_CALLS='"'"${HSPEC_GEN_MAX_CALLS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE='"'"${HSPEC_PROFILE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_STEPS='"'"${HSPEC_PROFILE_STEPS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_METHOD='"'"${HSPEC_PROFILE_METHOD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_DIR='"'"${HSPEC_PROFILE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_LEVEL='"'"${HSPEC_PROFILE_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_ANALYSE='"'"${HSPEC_PROFILE_ANALYSE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_WITH_STACK='"'"${HSPEC_PROFILE_WITH_STACK}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_MEMORY='"'"${HSPEC_PROFILE_MEMORY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LOG_LEVEL='"'"${HSPEC_LOG_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL='"'"${VERL_LOGGING_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='"'"INFO"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_SPECULATIVE_BATCH_SIZE_THRE='"'"${VLLM_SPECULATIVE_BATCH_SIZE_THRE}"'"' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.temperature=0.3 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.use_hspec_decode="${USE_HSPEC_DECODE}" \
    actor_rollout_ref.rollout.hspec_num_speculative_tokens=5 \
    actor_rollout_ref.rollout.hspec_similarity_threshold=0.85 \
    actor_rollout_ref.rollout.hspec_min_match_len=1 \
    actor_rollout_ref.rollout.hspec_n_components="${PCA_COMPONENTS}" \
    actor_rollout_ref.rollout.hspec_max_entries_per_prompt=10000 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.device=npu \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_gsm8k_hspec_validate' \
    trainer.experiment_name='qwen_hspec_validate_small' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=5 \
    > "${OUT}" 2>&1 "$@"

echo "OK: training finished. See log at ${OUT}"
