#!/usr/bin/env bash

set -euo pipefail

# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_LAUNCH_BLOCKING=1

HOME=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_API_LIB="${CUSTOM_OPP_PATH}/op_api/lib"

if [ -d "${CUSTOM_OPP_PATH}" ]; then
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
fi

if [ -d "${CUSTOM_OP_API_LIB}" ]; then
    export LD_LIBRARY_PATH="${CUSTOM_OP_API_LIB}:${LD_LIBRARY_PATH:-}"
fi

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

# HSpec system data-plane switches. 
export HSPEC_LEGACY_DATAPROTO_HS="${HSPEC_LEGACY_DATAPROTO_HS:-0}"
export HSPEC_STORE_DIR="${HSPEC_STORE_DIR:-/tmp/hspec_store}"
export HSPEC_NUM_SHARDS="${HSPEC_NUM_SHARDS:-5}"


# HSpec debug/tracing/profile switches.
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export HSPEC_DEBUG="${HSPEC_DEBUG:-0}"
export HSPEC_TRACE="${HSPEC_TRACE:-0}"
export HSPEC_DUMP="${HSPEC_DUMP:-0}"
export HSPEC_PROFILE="${HSPEC_PROFILE:-0}" 
export HSPEC_DUMP_DIR="${HSPEC_DUMP_DIR:-/workspace/exp/hspec_dump-rollout_1024}"

# Core HSpec knobs.
export PCA_COMPONENTS="${PCA_COMPONENTS:-64}"
export HSPEC_DISABLE_NUMBA_REBUILD="${HSPEC_DISABLE_NUMBA_REBUILD:-0}"
export HSPEC_NUMBA_REBUILD_MIN_ROWS="${HSPEC_NUMBA_REBUILD_MIN_ROWS:-0}"
export HSPEC_NUMBA_REBUILD_MIN_ELEMS="${HSPEC_NUMBA_REBUILD_MIN_ELEMS:-0}"
export HSPEC_ALIGN_DEBUG="${HSPEC_ALIGN_DEBUG:-0}"
export HSPEC_ALIGN_DEBUG_MAX_LOGS="${HSPEC_ALIGN_DEBUG_MAX_LOGS:-256}"
export HSPEC_ALIGN_DEBUG_PREVIEW="${HSPEC_ALIGN_DEBUG_PREVIEW:-8}"
export HSPEC_ENTRY="${HSPEC_ENTRY:-0}"
export MATCH_WND="${MATCH_WND:-16}"
export HSPEC_ADVAN_NGRAM="${HSPEC_ADVAN_NGRAM:-1}"
export HSPEC_ASYNC_HS_ACCUMULATE="${HSPEC_ASYNC_HS_ACCUMULATE:-0}"
export HSPEC_ASYNC_HS_COPY_STREAM="${HSPEC_ASYNC_HS_COPY_STREAM:-1}"
export HSPEC_FULL_BATCH_PREFETCH="${HSPEC_FULL_BATCH_PREFETCH:-1}"

# Per-step HSpec breakdown / profiler controls.
export HSPEC_GEN="${HSPEC_GEN:-0}"
export HSPEC_GEN_REQ_IDX="${HSPEC_GEN_REQ_IDX:-0}"
export HSPEC_GEN_MAX_CALLS="${HSPEC_GEN_MAX_CALLS:-0}"
export HSPEC_PROFILE_STEPS="${HSPEC_PROFILE_STEPS:-5,31,63,91}"
export HSPEC_PROFILE_DIR="${HSPEC_PROFILE_DIR:-/home/xy/hspec_profile_batch-update_prefill_spec_on}"
export HSPEC_PROFILE_METHOD="${HSPEC_PROFILE_METHOD:-mstx}"
export HSPEC_PROFILE_LEVEL="${HSPEC_PROFILE_LEVEL:-level_none}"
export HSPEC_PROFILE_ANALYSE="${HSPEC_PROFILE_ANALYSE:-1}"
export HSPEC_PROFILE_WITH_STACK="${HSPEC_PROFILE_WITH_STACK:-0}"
export HSPEC_PROFILE_MEMORY="${HSPEC_PROFILE_MEMORY:-0}"

# rollout strictly sync; HSpec path assumes sync vLLM rollout.
export USE_HSPEC_DECODE="${USE_HSPEC_DECODE:-1}"
export VLLM_SPECULATIVE_BATCH_SIZE_THRE="${VLLM_SPECULATIVE_BATCH_SIZE_THRE:--1}"

# Logging behavior.
export HSPEC_LOG_EVERY_CALLS="${HSPEC_LOG_EVERY_CALLS:-50}"
export HSPEC_LOG_EVERY_S="${HSPEC_LOG_EVERY_S:-5}"
export HSPEC_LOG_LEVEL="${HSPEC_LOG_LEVEL:-INFO}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# model / dataset defaults.
export MODEL_PATH="${MODEL_PATH:-/home/data/Qwen2.5-1.5B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-/home/xy/gsm8k/train.parquet}"
export TEST_FILE="${TEST_FILE:-/home/xy/gsm8k/test.parquet}"

# dump-mode behavior for batch sizing.
if [ "${HSPEC_DUMP}" = "0" ]; then
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-40}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    export ROLLOUT_N="${ROLLOUT_N:-5}"
else
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
    export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    export ROLLOUT_N="${ROLLOUT_N:-5}"
fi

# Log/output path conventions.
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/rl}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
export OUT="${OUT:-/workspace/cann-recipes-train/llm_rl/qwen3/output/train_grpo_hspec-1.5b-64.txt}"
mkdir -p "${LOG_DIR}" "$(dirname "${OUT}")"

{
    echo
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') hspec single run ====="
    echo "project_root=${PROJECT_ROOT}"
    echo "model_path=${MODEL_PATH}"
    echo "train_file=${TRAIN_FILE}"
    echo "test_file=${TEST_FILE}"
    echo "log_path=${OUT}"
    echo "use_hspec_decode=${USE_HSPEC_DECODE}"
    echo "hspec_profile=${HSPEC_PROFILE}"
    echo "hspec_dump=${HSPEC_DUMP}"
    echo "pca_components=${PCA_COMPONENTS}"
    echo "vllm_spec_batch_threshold=${VLLM_SPECULATIVE_BATCH_SIZE_THRE}"
} >> "${OUT}"

set -x

python -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.num_cpus=128 \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG='"'"${HSPEC_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_REQS='"2"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_SAMPLES='"4"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_VALUES='"8"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='"'"${VLLM_LOGGING_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LEGACY_DATAPROTO_HS='"'"${HSPEC_LEGACY_DATAPROTO_HS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_DIR='"'"${HSPEC_STORE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUM_SHARDS='"'"$
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TRACE='"'"${HSPEC_TRACE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP='"'"${HSPEC_DUMP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP_DIR='"'"${HSPEC_DUMP_DIR}"'"' \
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
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DISABLE_NUMBA_REBUILD='"'"${HSPEC_DISABLE_NUMBA_REBUILD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ROWS='"'"${HSPEC_NUMBA_REBUILD_MIN_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ELEMS='"'"${HSPEC_NUMBA_REBUILD_MIN_ELEMS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG='"'"${HSPEC_ALIGN_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG_MAX_LOGS='"'"${HSPEC_ALIGN_DEBUG_MAX_LOGS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALIGN_DEBUG_PREVIEW='"'"${HSPEC_ALIGN_DEBUG_PREVIEW}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ENTRY='"'"${HSPEC_ENTRY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.MATCH_WND='"'"${MATCH_WND}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ADVAN_NGRAM='"'"${HSPEC_ADVAN_NGRAM}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ASYNC_HS_ACCUMULATE='"'"${HSPEC_ASYNC_HS_ACCUMULATE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ASYNC_HS_COPY_STREAM='"'"${HSPEC_ASYNC_HS_COPY_STREAM}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_FULL_BATCH_PREFETCH='"'"${HSPEC_FULL_BATCH_PREFETCH}"'"' \
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
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
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
    actor_rollout_ref.rollout.hspec_num_speculative_tokens=15 \
    actor_rollout_ref.rollout.hspec_similarity_threshold=0.85 \
    actor_rollout_ref.rollout.hspec_min_match_len=1 \
    actor_rollout_ref.rollout.hspec_n_components="${PCA_COMPONENTS}" \
    actor_rollout_ref.rollout.hspec_max_entries_per_prompt=10000 \
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
