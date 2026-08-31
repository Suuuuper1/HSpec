#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <baseline|hspec|dflash|dspark> <arm-output-dir>" >&2
    exit 2
fi

ARM="$1"
ARM_DIR="$2"
case "${ARM}" in
    baseline) ARM_CODE="b" ;;
    dflash) ARM_CODE="f" ;;
    dspark) ARM_CODE="s" ;;
    hspec) ARM_CODE="h" ;;
    *) echo "unsupported Phase-3 arm: ${ARM}" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
MODEL_PATH="${PHASE3_TARGET_MODEL:-/home/data/Qwen3-8B}"
DFLASH_MODEL="${PHASE3_DFLASH_MODEL:-/home/data/Qwen3-8B-dflash}"
DSPARK_MODEL="${PHASE3_DSPARK_MODEL:-/home/data/Qwen3-8B-dspark}"
TRAIN_FILE="${PHASE3_TRAIN_FILE:-/home/xy/gsm8k/train.parquet}"
VAL_FILE="${PHASE3_VAL_FILE:-/home/xy/gsm8k/test.parquet}"
TOTAL_STEPS="${PHASE3_TOTAL_STEPS:-3}"
ACTOR_LR="${PHASE3_ACTOR_LR:-5e-7}"
LIFECYCLE_AUDIT="${PHASE3_LIFECYCLE_AUDIT:-true}"
EXPERIMENT_SEED="${PHASE3_SEED:-20260829}"
SPECULATIVE_K="${PHASE3_NUM_SPECULATIVE_TOKENS:-7}"
DRAFT_SAMPLE_METHOD="${PHASE3_DRAFT_SAMPLE_METHOD:-greedy}"
DRAFT_PROBABILITY_MAX_MEMORY_MB="${PHASE3_DRAFT_PROBABILITY_MAX_MEMORY_MB:-2048}"
PARALLEL_DRAFT_PROFILE_ENABLED="${PHASE3_PARALLEL_DRAFT_PROFILE_ENABLED:-false}"
PARALLEL_DRAFT_PROFILE_SAMPLE_EVERY="${PHASE3_PARALLEL_DRAFT_PROFILE_SAMPLE_EVERY:-64}"
PARALLEL_DRAFT_PROFILE_FLUSH_EVERY="${PHASE3_PARALLEL_DRAFT_PROFILE_FLUSH_EVERY:-4}"
LIFECYCLE_DIR="${ARM_DIR}/lifecycle"
RAY_TMPDIR="${PHASE3_RAY_TMPDIR:-/dev/shm/p3r3/manual/${ARM_CODE}}"
RAY_TMPDIR_MAX_BYTES="${PHASE3_RAY_TMPDIR_MAX_BYTES:-31}"
RAY_TMPDIR_BYTES=$(LC_ALL=C printf '%s' "${RAY_TMPDIR}" | wc -c)
if ! [[ "${RAY_TMPDIR_MAX_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "PHASE3_RAY_TMPDIR_MAX_BYTES must be a non-negative integer" >&2
    exit 2
fi
if [[ "${DRAFT_SAMPLE_METHOD}" != "greedy" && "${DRAFT_SAMPLE_METHOD}" != "probabilistic" ]]; then
    echo "PHASE3_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
    exit 2
fi
if ! [[ "${DRAFT_PROBABILITY_MAX_MEMORY_MB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE3_DRAFT_PROBABILITY_MAX_MEMORY_MB must be a positive integer" >&2
    exit 2
fi
if ! [[ "${EXPERIMENT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "PHASE3_SEED must be a non-negative integer" >&2
    exit 2
fi
if [[ "${LIFECYCLE_AUDIT}" != "true" && "${LIFECYCLE_AUDIT}" != "false" ]]; then
    echo "PHASE3_LIFECYCLE_AUDIT must be true or false" >&2
    exit 2
fi
if ! [[ "${SPECULATIVE_K}" =~ ^([1-9]|1[0-5])$ ]]; then
    echo "PHASE3_NUM_SPECULATIVE_TOKENS must be in [1, 15]" >&2
    exit 2
fi
if [[ "${PARALLEL_DRAFT_PROFILE_ENABLED}" != "true" && "${PARALLEL_DRAFT_PROFILE_ENABLED}" != "false" ]]; then
    echo "PHASE3_PARALLEL_DRAFT_PROFILE_ENABLED must be true or false" >&2
    exit 2
fi
for PHASE3_PROFILE_INTERVAL in "${PARALLEL_DRAFT_PROFILE_SAMPLE_EVERY}" "${PARALLEL_DRAFT_PROFILE_FLUSH_EVERY}"; do
    if ! [[ "${PHASE3_PROFILE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Phase-5 draft profiling intervals must be positive integers" >&2
        exit 2
    fi
done
unset PHASE3_PROFILE_INTERVAL
if [[ "${ARM}" != "dflash" && "${ARM}" != "dspark" && "${PARALLEL_DRAFT_PROFILE_ENABLED}" == "true" ]]; then
    echo "parallel draft profiling is only valid for dflash/dspark arms" >&2
    exit 2
fi
if (( RAY_TMPDIR_BYTES > RAY_TMPDIR_MAX_BYTES )); then
    echo "Phase-3 RAY_TMPDIR is ${RAY_TMPDIR_BYTES} bytes, exceeding the "\
         "${RAY_TMPDIR_MAX_BYTES}-byte Ray AF_UNIX-safe budget: ${RAY_TMPDIR}" >&2
    exit 2
fi

mkdir -p "${ARM_DIR}" "${LIFECYCLE_DIR}" "${RAY_TMPDIR}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
# R3 consumes vLLM's native INFO-level draft-load and speculative counters.
# Force the frozen values here and in Ray runtime_env so Verl's WARN default,
# or a stale caller environment, cannot silently disable the evidence path.
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOG_STATS_INTERVAL=0.1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_STREAM='ext://sys.stdout'
unset VLLM_LOGGING_CONFIG_PATH
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_STRICT_FULL_GRAPH=1
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR
export VERL_SPECULATIVE_LIFECYCLE_DIR="${LIFECYCLE_DIR}"

CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
if [[ -d "${CUSTOM_OPP_PATH}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="${CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
fi

METHOD="null"
DRAFT_MODEL="null"
# Phase-3 is a frozen certification launcher.  Remove caller leftovers before
# selecting an arm so neural/baseline jobs cannot inherit HSpec behavior and
# the HSpec job uses only the explicitly declared configuration below.
for PHASE3_ENV_NAME in "${!HSPEC_@}"; do
    unset "${PHASE3_ENV_NAME}"
done
unset PHASE3_ENV_NAME
RAY_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOG_STATS_INTERVAL='0.1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CONFIGURE_LOGGING='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_STREAM=ext://sys.stdout"
)
if [[ "${ARM}" == "dflash" ]]; then
    METHOD="dflash"
    DRAFT_MODEL="${DFLASH_MODEL}"
elif [[ "${ARM}" == "dspark" ]]; then
    METHOD="dspark"
    DRAFT_MODEL="${DSPARK_MODEL}"
elif [[ "${ARM}" == "hspec" ]]; then
    METHOD="hspec"
    export HSPEC_RUN_UID="phase3_${ARM}_$(date -u '+%Y%m%dT%H%M%SZ')_$$"
    export HSPEC_STORE_DIR="${ARM_DIR}/hspec/raw"
    export HSPEC_TABLE_STORE_DIR="${ARM_DIR}/hspec/table"
    export HSPEC_INFER_TP=1
    export HSPEC_NUM_SHARDS=1
    export HSPEC_SINGLE_NODE_ONLY=1
    export HSPEC_TOPOLOGY_STRICT=1
    export HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS=1
    export HSPEC_LEGACY_DATAPROTO_HS=0
    export HSPEC_STRICT_DESCRIPTOR_MODE=1
    export HSPEC_STRICT_PROMPT_ID_ON_SEAL=1
    export HSPEC_TABLE_PREFETCH_MODE=descriptor
    export HSPEC_FULL_BATCH_PREFETCH=1
    export HSPEC_ASYNC_HS_COPY_STREAM=1
    export HSPEC_PINNED_POOL_BYTES=268435456
    # Keep fp16 table storage, but materialize proposer keys as fp32 before
    # Numba packing, matching the production HSpec launchers.
    export HSPEC_TABLE_KEYS_DTYPE=float16
    export HSPEC_PROPOSER_KEYS_CPU_DTYPE=float32
    export HSPEC_PROPOSER_KEYS_DEVICE_DTYPE=float32
    export HSPEC_DISABLE_NUMBA_REBUILD=0
    export HSPEC_NUMBA_REBUILD_MIN_ROWS=0
    export HSPEC_NUMBA_REBUILD_MIN_ELEMS=0
    export HSPEC_LOG_LEVEL=INFO
    export HSPEC_CLEAN_STORE_ON_START=1
    export HSPEC_CLEAN_RAW_STORE_ON_START=1
    export HSPEC_CLEAN_TABLE_STORE_ON_START=1
    mkdir -p "${HSPEC_STORE_DIR}" "${HSPEC_TABLE_STORE_DIR}"
    RAY_ENV_ARGS+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RUN_UID=${HSPEC_RUN_UID}"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_DIR=${HSPEC_STORE_DIR}"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_DIR=${HSPEC_TABLE_STORE_DIR}"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_INFER_TP='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUM_SHARDS='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_SINGLE_NODE_ONLY='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TOPOLOGY_STRICT='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LEGACY_DATAPROTO_HS='0'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STRICT_DESCRIPTOR_MODE='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STRICT_PROMPT_ID_ON_SEAL='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_PREFETCH_MODE=descriptor"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_FULL_BATCH_PREFETCH='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ASYNC_HS_COPY_STREAM='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PINNED_POOL_BYTES='268435456'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_KEYS_DTYPE=float16"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_KEYS_CPU_DTYPE=float32"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_KEYS_DEVICE_DTYPE=float32"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DISABLE_NUMBA_REBUILD='0'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ROWS='0'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUMBA_REBUILD_MIN_ELEMS='0'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LOG_LEVEL=INFO"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_STORE_ON_START='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_RAW_STORE_ON_START='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_TABLE_STORE_ON_START='1'"
    )
else
    # Neural and baseline arms must not create or consume HSpec stores.  All
    # caller-provided HSPEC_* values were removed before arm selection.
    :
fi

echo "PHASE3_ARM_BEGIN arm=${ARM} method=${METHOD} steps=${TOTAL_STEPS} seed=${EXPERIMENT_SEED} k=${SPECULATIVE_K} draft_sample_method=${DRAFT_SAMPLE_METHOD} parallel_draft_profile=${PARALLEL_DRAFT_PROFILE_ENABLED}"
echo "PHASE3_RAY_TMPDIR path=${RAY_TMPDIR} bytes=${RAY_TMPDIR_BYTES} max_bytes=${RAY_TMPDIR_MAX_BYTES}"
echo "PHASE3_OBSERVABILITY VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL} VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING} VLLM_LOGGING_CONFIG_PATH=${VLLM_LOGGING_CONFIG_PATH:-none} VLLM_LOGGING_STREAM=stdout ray_runtime_override=INFO"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    +data.seed="${EXPERIMENT_SEED}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.seed="${EXPERIMENT_SEED}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.72 \
    actor_rollout_ref.rollout.max_num_batched_tokens=640 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.use_hspec_decode=False \
    actor_rollout_ref.rollout.speculative_method="${METHOD}" \
    actor_rollout_ref.rollout.speculative_model="${DRAFT_MODEL}" \
    actor_rollout_ref.rollout.num_speculative_tokens="${SPECULATIVE_K}" \
    actor_rollout_ref.rollout.draft_tensor_parallel_size=1 \
    actor_rollout_ref.rollout.draft_sample_method="${DRAFT_SAMPLE_METHOD}" \
    actor_rollout_ref.rollout.draft_probability_max_memory_mb="${DRAFT_PROBABILITY_MAX_MEMORY_MB}" \
    actor_rollout_ref.rollout.draft_load_format=auto \
    actor_rollout_ref.rollout.rejection_sample_method=standard \
    actor_rollout_ref.rollout.speculative_enforce_eager=True \
    actor_rollout_ref.rollout.parallel_draft_profile_enabled="${PARALLEL_DRAFT_PROFILE_ENABLED}" \
    actor_rollout_ref.rollout.parallel_draft_profile_sample_every="${PARALLEL_DRAFT_PROFILE_SAMPLE_EVERY}" \
    actor_rollout_ref.rollout.parallel_draft_profile_flush_every="${PARALLEL_DRAFT_PROFILE_FLUSH_EVERY}" \
    actor_rollout_ref.rollout.speculative_lifecycle_audit="${LIFECYCLE_AUDIT}" \
    actor_rollout_ref.rollout.speculative_lifecycle_strict=True \
    actor_rollout_ref.rollout.speculative_lifecycle_samples_per_parameter=8 \
    actor_rollout_ref.rollout.hspec_similarity_threshold=0.85 \
    actor_rollout_ref.rollout.hspec_min_match_len=1 \
    actor_rollout_ref.rollout.hspec_n_components=64 \
    actor_rollout_ref.rollout.hspec_max_entries_per_prompt=10000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.device=npu \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=dflash_dspark_phase3 \
    trainer.experiment_name="qwen3_8b_${ARM}_s${EXPERIMENT_SEED}_k${SPECULATIVE_K}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.shutdown_on_exit=True \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SPECULATIVE_LIFECYCLE_DIR="${LIFECYCLE_DIR}" \
    "${RAY_ENV_ARGS[@]}"

echo "PHASE3_ARM_END arm=${ARM} method=${METHOD} steps=${TOTAL_STEPS} seed=${EXPERIMENT_SEED} k=${SPECULATIVE_K} draft_sample_method=${DRAFT_SAMPLE_METHOD} parallel_draft_profile=${PARALLEL_DRAFT_PROFILE_ENABLED}"
