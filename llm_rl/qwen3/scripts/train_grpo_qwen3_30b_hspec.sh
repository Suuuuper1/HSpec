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
export HSPEC_STORE_DIR="${HSPEC_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_store}"
export HSPEC_TABLE_STORE_DIR="${HSPEC_TABLE_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_table_store}"

# capture graph
# export VERL_VLLM_CUDAGRAPH_MODE="${VERL_VLLM_CUDAGRAPH_MODE:-FULL}"

# HSpec debug / tracing / profiling switches.
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export HSPEC_DEBUG="${HSPEC_DEBUG:-0}"
export HSPEC_TRACE="${HSPEC_TRACE:-0}"
export HSPEC_DUMP="${HSPEC_DUMP:-0}"
export HSPEC_PROFILE="${HSPEC_PROFILE:-1}"
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

# Per-step HSpec breakdown / profiler controls.
export HSPEC_GEN="${HSPEC_GEN:-0}"
export HSPEC_GEN_REQ_IDX="${HSPEC_GEN_REQ_IDX:-0}"
export HSPEC_GEN_MAX_CALLS="${HSPEC_GEN_MAX_CALLS:-0}"
export HSPEC_PROFILE_STEPS="${HSPEC_PROFILE_STEPS:-1}"
export HSPEC_PROFILE_DIR="${HSPEC_PROFILE_DIR:-/home/sharedata/xy_profile/hspec_profile_new-30b}"
export HSPEC_PROFILE_METHOD="${HSPEC_PROFILE_METHOD:-mstx}"
export HSPEC_PROFILE_LEVEL="${HSPEC_PROFILE_LEVEL:-level_none}"
export HSPEC_PROFILE_ANALYSE="${HSPEC_PROFILE_ANALYSE:-1}"
export HSPEC_PROFILE_WITH_STACK="${HSPEC_PROFILE_WITH_STACK:-0}"
export HSPEC_PROFILE_MEMORY="${HSPEC_PROFILE_MEMORY:-0}"

export USE_HSPEC_DECODE="${USE_HSPEC_DECODE:-1}"
export VLLM_SPECULATIVE_BATCH_SIZE_THRE="${VLLM_SPECULATIVE_BATCH_SIZE_THRE:--1}"

# Logging behavior.
export HSPEC_LOG_EVERY_CALLS="${HSPEC_LOG_EVERY_CALLS:-50}"
export HSPEC_LOG_EVERY_S="${HSPEC_LOG_EVERY_S:-5}"
export HSPEC_LOG_LEVEL="${HSPEC_LOG_LEVEL:-INFO}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# model / dataset defaults.
CONFIG_DIR="${CONFIG_DIR:-${HOME}/verl/trainer/config}"
MODEL_PATH="${MODEL_PATH:-/home/data/Qwen3-30B-A3B}"
TRAIN_FILE="${TRAIN_FILE:-/data/deepscaler/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/deepscaler/test.parquet}"
DISTCP_PATH="${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}"

NODES="${NODES:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
INFER_TP="${INFER_TP:-4}"
export HSPEC_INFER_TP="${HSPEC_INFER_TP:-${INFER_TP}}"
export HSPEC_NUM_SHARDS="${HSPEC_NUM_SHARDS:-${HSPEC_INFER_TP}}"
export NODE_RANK="${NODE_RANK:-0}"
export HSPEC_TP_GROUP_ID="${HSPEC_TP_GROUP_ID:-}"

# dump-mode behavior for batch sizes and rollout count.
if [ "${HSPEC_DUMP}" = "0" ]; then
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/rl}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
ROLL_LEN_ROOT="${ROLL_LEN_ROOT:-${OUTPUT_ROOT}/rollout_length}"
TB_ROOT="${TB_ROOT:-${OUTPUT_ROOT}/tensorboard}"
RUN_NAME="${RUN_NAME:-qwen3_30b_hspec_single}"
ROLLOUT_LENGTH_DIR="${ROLLOUT_LENGTH_DIR:-${ROLL_LEN_ROOT}/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TB_ROOT}/${RUN_NAME}}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-${LOG_DIR}/${RUN_NAME}.log}"
OUT="${OUT:-/workspace/cann-recipes-train/llm_rl/qwen3/output/train_grpo_hspec-30b.txt}"

mkdir -p "${LOG_DIR}" "${ROLL_LEN_ROOT}" "${TB_ROOT}" "${ROLLOUT_LENGTH_DIR}" "$(dirname "${OUT}")"

{
    echo
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') hspec single run ====="
    echo "project_root=${PROJECT_ROOT}"
    echo "config_dir=${CONFIG_DIR}"
    echo "model_path=${MODEL_PATH}"
    echo "train_file=${TRAIN_FILE}"
    echo "test_file=${TEST_FILE}"
    echo "out=${OUT}"
    echo "rollout_log_path=${ROLLOUT_LOG_PATH}"
    echo "rollout_length_dir=${ROLLOUT_LENGTH_DIR}"
    echo "tensorboard_dir=${TENSORBOARD_DIR}"
    echo "use_hspec_decode=${USE_HSPEC_DECODE}"
    echo "hspec_profile=${HSPEC_PROFILE}"
    echo "hspec_dump=${HSPEC_DUMP}"
    echo "pca_components=${PCA_COMPONENTS}"
    echo "vllm_spec_batch_threshold=${VLLM_SPECULATIVE_BATCH_SIZE_THRE}"
} >> "${OUT}"

set -x

env \
    ROLLOUT_LENGTH_DIR="${ROLLOUT_LENGTH_DIR}" \
    TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.use_hspec_decode="${USE_HSPEC_DECODE}" \
    actor_rollout_ref.rollout.hspec_num_speculative_tokens=15 \
    actor_rollout_ref.rollout.hspec_similarity_threshold=0.85 \
    actor_rollout_ref.rollout.hspec_min_match_len=1 \
    actor_rollout_ref.rollout.hspec_n_components="${PCA_COMPONENTS}" \
    actor_rollout_ref.rollout.hspec_max_entries_per_prompt=160000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_gsm8k_hspec_validate' \
    trainer.experiment_name='qwen_hspec_validate_small' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes="${NODES}" \
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
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG='"'"${HSPEC_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_REQS='"2"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_SAMPLES='"4"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_VALUES='"8"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='"'"${VLLM_LOGGING_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TRACE='"'"${HSPEC_TRACE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP='"'"${HSPEC_DUMP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DUMP_DIR='"'"${HSPEC_DUMP_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LEGACY_DATAPROTO_HS='"'"${HSPEC_LEGACY_DATAPROTO_HS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_DIR='"'"${HSPEC_STORE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_DIR='"'"${HSPEC_TABLE_STORE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUM_SHARDS='"'"${HSPEC_NUM_SHARDS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_INFER_TP='"'"${HSPEC_INFER_TP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.NODE_RANK='"'"${NODE_RANK}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TP_GROUP_ID='"'"${HSPEC_TP_GROUP_ID}"'"' \
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
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_SPECULATIVE_BATCH_SIZE_THRE='"'"${VLLM_SPECULATIVE_BATCH_SIZE_THRE}"'"' \
    > "${OUT}" 2>&1 "$@"

echo "OK: training finished. See log at ${OUT}"
