#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <dflash|dspark> <arm-output-dir>" >&2
    exit 2
fi

METHOD=$1
ARM_DIR=$(realpath -m "$2")
if [[ "${METHOD}" != "dflash" && "${METHOD}" != "dspark" ]]; then
    echo "Phase 3 certifies only dflash or dspark" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TARGET_MODEL=${PHASE3_TARGET_MODEL:-/home/data/Qwen3-30B-A3B}
ACTOR_DIST_CHECKPOINT=${PHASE3_ACTOR_DIST_CHECKPOINT:-/home/data/Qwen3-30B-A3B_megatron}
TRAIN_FILE=${PHASE3_TRAIN_FILE:-/data/deepscaler/train.parquet}
VALIDATION_FILE=${PHASE3_VALIDATION_FILE:-/data/deepscaler/test.parquet}
if [[ "${METHOD}" == "dflash" ]]; then
    DRAFT_MODEL=${PHASE3_DFLASH_MODEL:?PHASE3_DFLASH_MODEL is required}
else
    DRAFT_MODEL=${PHASE3_DSPARK_MODEL:?PHASE3_DSPARK_MODEL is required}
fi

# These are certification constants, not tuning knobs. The two arms differ
# only in method, checkpoint and isolated output paths.
readonly DP_SIZE=16
readonly SEED=20260829
readonly SPECULATIVE_K=7
readonly TRAINING_STEPS=2
readonly PROBABILITY_BUDGET_MB=512
readonly GPU_MEMORY_UTILIZATION=0.72
readonly TRACE_LIMIT=256
readonly HCCL_BUFFER_MB=800
readonly HCCL_HOST_PORT_COUNT=32
if [[ "${METHOD}" == "dflash" ]]; then
    readonly EXPECTED_HCCL_IF_BASE_PORT=63000
else
    readonly EXPECTED_HCCL_IF_BASE_PORT=63032
fi

if [[ ${VLLM_DP_SIZE:-${DP_SIZE}} != "${DP_SIZE}" ]]; then
    echo "VLLM_DP_SIZE must equal frozen Phase-3 DP=${DP_SIZE}" >&2
    exit 2
fi
if [[ ${HCCL_BUFFSIZE:-} != "${HCCL_BUFFER_MB}" ]]; then
    echo "HCCL_BUFFSIZE must equal frozen Phase-3 value ${HCCL_BUFFER_MB} MB" >&2
    exit 2
fi
if [[ ${HCCL_IF_BASE_PORT:-} != "${EXPECTED_HCCL_IF_BASE_PORT}" ]]; then
    echo "HCCL_IF_BASE_PORT must equal frozen Phase-3 ${METHOD} value ${EXPECTED_HCCL_IF_BASE_PORT}" >&2
    exit 2
fi
IFS=',' read -r -a DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES:-}"
if [[ ${#DEVICES[@]} -ne ${DP_SIZE} ]]; then
    echo "ASCEND_RT_VISIBLE_DEVICES must name exactly ${DP_SIZE} devices" >&2
    exit 2
fi

LIFECYCLE_DIR=${ARM_DIR}/lifecycle
PRELUDE_DIR=${ARM_DIR}/prelude
mkdir -p "${ARM_DIR}" "${LIFECYCLE_DIR}" "${PRELUDE_DIR}"
if find "${LIFECYCLE_DIR}" "${PRELUDE_DIR}" -type f -print -quit | grep -q .; then
    echo "refusing to reuse non-empty Phase-3 evidence directories" >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_DP_SIZE=${DP_SIZE}
export VLLM_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_STREAM='ext://sys.stdout'
export VLLM_LOGGING_CONFIG_PATH="${PROJECT_ROOT}/HSpec_research_doc/repair/DFlash_DSpark_migrate/phase3/vllm_logging_config.json"
unset VLLM_LOG_STATS_INTERVAL
export RAY_DEDUP_LOGS=0
export USE_HSPEC_DECODE=0
export HSPEC_PROFILE=0
export VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE=1
export VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE_LIMIT=${TRACE_LIMIT}
export HSPEC_S7_ENGINE_TIMING=1
export HSPEC_S7_OBSERVER_SCHEMA=v2
export HSPEC_S7_OBSERVER_SAMPLE_EVERY=256
export VERL_SPECULATIVE_LIFECYCLE_DIR=${LIFECYCLE_DIR}
export VERL_DP_REPAIR_PHASE3_PRELUDE=1
export VERL_DP_REPAIR_PHASE3_PRELUDE_DIR=${PRELUDE_DIR}
export VERL_DP_REPAIR_PHASE3_PRELUDE_TOKENS=2
export VERL_DP_REPAIR_PHASE3_SEED=${SEED}
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_STRICT_FULL_GRAPH=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=${HCCL_BUFFER_MB}
export HCCL_IF_BASE_PORT=${EXPECTED_HCCL_IF_BASE_PORT}
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

# Inputs consumed by the existing, production-shaped 30B Megatron launcher.
export RUN_NAME="dp_repair_phase3_${METHOD}"
export MODEL_PATH=${TARGET_MODEL}
export DISTCP_PATH=${ACTOR_DIST_CHECKPOINT}
export TRAIN_FILE VALIDATION_FILE
export TEST_FILE=${VALIDATION_FILE}
export OUT=${ARM_DIR}/train.log
export OUTPUT_ROOT=${ARM_DIR}/outputs
export NODES=1
export TRAIN_GPUS_PER_NODE=16
export TRAIN_TP=4
export TRAIN_PP=4
export TRAIN_CP=1
export TRAIN_EP=4
export TRAIN_ETP=1
export REF_TP=1
export REF_PP=1
export REF_CP=1
export REF_EP=1
export REF_ETP=1
export INFER_TP=1
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_N=16
export DATASET_FRACTION=0.005
export TOTAL_EPOCHS=1
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=16384
export MAX_NUM_SEQS=32
export ROLLOUT_TEMPERATURE=0.9
export ROLLOUT_TOP_K=50
export ROLLOUT_TOP_P=0.9
export ROLLOUT_ENABLE_PREFIX_CACHING=False
export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
export GPU_MEMORY_UTILIZATION

echo "DP_REPAIR_PHASE3_ARM_BEGIN method=${METHOD} dp=${DP_SIZE} steps=${TRAINING_STEPS} proposal=probabilistic k=${SPECULATIVE_K} seed=${SEED}"

bash "${SCRIPT_DIR}/train_grpo_qwen3_30b_hspec.sh" \
    actor_rollout_ref.rollout.seed=${SEED} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.vllm_data_parallel_size=${DP_SIZE} \
    actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.speculative_method=${METHOD} \
    actor_rollout_ref.rollout.speculative_model="${DRAFT_MODEL}" \
    actor_rollout_ref.rollout.num_speculative_tokens=${SPECULATIVE_K} \
    actor_rollout_ref.rollout.draft_tensor_parallel_size=1 \
    actor_rollout_ref.rollout.draft_sample_method=probabilistic \
    actor_rollout_ref.rollout.draft_probability_max_memory_mb=${PROBABILITY_BUDGET_MB} \
    actor_rollout_ref.rollout.draft_load_format=auto \
    actor_rollout_ref.rollout.rejection_sample_method=standard \
    actor_rollout_ref.rollout.speculative_enforce_eager=True \
    actor_rollout_ref.rollout.parallel_draft_profile_enabled=True \
    actor_rollout_ref.rollout.parallel_draft_profile_sample_every=64 \
    actor_rollout_ref.rollout.parallel_draft_profile_flush_every=4 \
    actor_rollout_ref.rollout.speculative_lifecycle_audit=True \
    actor_rollout_ref.rollout.speculative_lifecycle_strict=True \
    actor_rollout_ref.rollout.speculative_lifecycle_samples_per_parameter=8 \
    trainer.logger=console \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${TRAINING_STEPS} \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DP_SIZE="'${DP_SIZE}'" \
    ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CONFIGURE_LOGGING="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_STREAM=ext://sys.stdout \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_CONFIG_PATH="'${VLLM_LOGGING_CONFIG_PATH}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_PARALLEL_DRAFT_DP_TRACE_LIMIT="'${TRACE_LIMIT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SPECULATIVE_LIFECYCLE_DIR="'${LIFECYCLE_DIR}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DP_REPAIR_PHASE3_PRELUDE="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DP_REPAIR_PHASE3_PRELUDE_DIR="'${PRELUDE_DIR}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DP_REPAIR_PHASE3_PRELUDE_TOKENS="'2'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DP_REPAIR_PHASE3_SEED="'${SEED}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE="'${HCCL_BUFFER_MB}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_IF_BASE_PORT="'${EXPECTED_HCCL_IF_BASE_PORT}'"

echo "DP_REPAIR_PHASE3_ARM_END method=${METHOD} status=PASS"
