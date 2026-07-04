#!/usr/bin/env bash

set -euo pipefail

# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_LAUNCH_BLOCKING=1

HOME=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_API_LIB="${CUSTOM_OPP_PATH}/op_api/lib"
RUN_NAME="${RUN_NAME:-qwen3_30b_hspec_single}"
HSPEC_RUN_NAME="${HSPEC_RUN_NAME:-${RUN_NAME}}"
source "${SCRIPT_DIR}/hspec_store_lifecycle.sh"

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
export HSPEC_STRICT_DESCRIPTOR_MODE="${HSPEC_STRICT_DESCRIPTOR_MODE:-1}"
export HSPEC_STORE_DTYPE="${HSPEC_STORE_DTYPE:-float16}"
hspec_configure_store_lifecycle unique
export HSPEC_SINGLE_NODE_ONLY="${HSPEC_SINGLE_NODE_ONLY:-1}"
export HSPEC_TOPOLOGY_STRICT="${HSPEC_TOPOLOGY_STRICT:-1}"
export HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS="${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS:-1}"
export HSPEC_STEP0_RUNTIME_ASSERTS="${HSPEC_STEP0_RUNTIME_ASSERTS:-0}"
export HSPEC_BUILD_ACTOR_NUM_CPUS="${HSPEC_BUILD_ACTOR_NUM_CPUS:-1}"
export HSPEC_BUILD_BLAS_THREADS="${HSPEC_BUILD_BLAS_THREADS:-1}"
export HSPEC_DELETE_TRAJECTORY_AFTER_BUILD="${HSPEC_DELETE_TRAJECTORY_AFTER_BUILD:-0}"
export HSPEC_RAW_STORE_GC_AFTER_EPOCH="${HSPEC_RAW_STORE_GC_AFTER_EPOCH:-1}"
export HSPEC_SEGMENT_FSYNC_ON_SEAL="${HSPEC_SEGMENT_FSYNC_ON_SEAL:-0}"
export HSPEC_RAW_STORE_MAX_BYTES="${HSPEC_RAW_STORE_MAX_BYTES:-0}"
export HSPEC_RAW_STORE_MAX_FILES="${HSPEC_RAW_STORE_MAX_FILES:-0}"
export HSPEC_STORE_RETAIN_BATCHES="${HSPEC_STORE_RETAIN_BATCHES:-128}"
export HSPEC_RAW_STORE_BUDGET_DELETE="${HSPEC_RAW_STORE_BUDGET_DELETE:-0}"
export HSPEC_PINNED_POOL_BYTES="${HSPEC_PINNED_POOL_BYTES:-268435456}"
export HSPEC_PINNED_POOL_MAX_SLOTS="${HSPEC_PINNED_POOL_MAX_SLOTS:-64}"
export HSPEC_PINNED_POOL_BUCKET_ROWS="${HSPEC_PINNED_POOL_BUCKET_ROWS:-64,128,256,512,1024,2048,4096}"
export HSPEC_COPY_MAX_PENDING_TASKS="${HSPEC_COPY_MAX_PENDING_TASKS:-64}"
export HSPEC_COPY_MAX_PENDING_ROWS="${HSPEC_COPY_MAX_PENDING_ROWS:-0}"
export HSPEC_DROP_ON_BACKPRESSURE="${HSPEC_DROP_ON_BACKPRESSURE:-1}"
export HSPEC_BUILD_MAX_PROMPT_ROWS="${HSPEC_BUILD_MAX_PROMPT_ROWS:-0}"
export HSPEC_BUILD_MAX_PROMPT_RAW_BYTES="${HSPEC_BUILD_MAX_PROMPT_RAW_BYTES:-0}"
export HSPEC_BUILD_MAX_PROMPT_DESCS="${HSPEC_BUILD_MAX_PROMPT_DESCS:-0}"
export HSPEC_BUILD_MAX_RSS_MB="${HSPEC_BUILD_MAX_RSS_MB:-0}"
export HSPEC_PCA_METHOD="${HSPEC_PCA_METHOD:-randomized}"
export HSPEC_PCA_TILE_ROWS="${HSPEC_PCA_TILE_ROWS:-1024}"
export HSPEC_PCA_RANDOM_OVERSAMPLE="${HSPEC_PCA_RANDOM_OVERSAMPLE:-16}"
export HSPEC_PCA_RANDOM_SEED="${HSPEC_PCA_RANDOM_SEED:-202405}"
export HSPEC_PCA_COV_MAX_BYTES="${HSPEC_PCA_COV_MAX_BYTES:-134217728}"
export HSPEC_PCA_ACCUM_DTYPE="${HSPEC_PCA_ACCUM_DTYPE:-float32}"
export HSPEC_TABLE_KEYS_DTYPE="${HSPEC_TABLE_KEYS_DTYPE:-float16}"
export HSPEC_TABLE_FILE_ALIGN_BYTES="${HSPEC_TABLE_FILE_ALIGN_BYTES:-4096}"
export HSPEC_TABLE_PREFETCH_MODE="${HSPEC_TABLE_PREFETCH_MODE:-descriptor}"
export HSPEC_ALLOW_LEGACY_TABLE_PREFETCH="${HSPEC_ALLOW_LEGACY_TABLE_PREFETCH:-0}"
export HSPEC_ENABLE_ZMQ_QUERY="${HSPEC_ENABLE_ZMQ_QUERY:-0}"
export HSPEC_PROPOSER_HOT_PATH_STRICT="${HSPEC_PROPOSER_HOT_PATH_STRICT:-1}"
export HSPEC_PROPOSER_CACHE_MAX_PROMPTS="${HSPEC_PROPOSER_CACHE_MAX_PROMPTS:-512}"
export HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES="${HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES:-2147483648}"
export HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES="${HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES:-0}"
export HSPEC_PROPOSER_CACHE_MAX_ENTRIES="${HSPEC_PROPOSER_CACHE_MAX_ENTRIES:-0}"
export HSPEC_MAX_READY_PREFETCH_MATERIALIZE="${HSPEC_MAX_READY_PREFETCH_MATERIALIZE:-0}"
export HSPEC_MAX_READY_PREFETCH_BYTES="${HSPEC_MAX_READY_PREFETCH_BYTES:-268435456}"
export HSPEC_TABLE_STORE_RETAIN_VERSIONS="${HSPEC_TABLE_STORE_RETAIN_VERSIONS:-2}"
export HSPEC_TABLE_STORE_GC_AFTER_SWAP="${HSPEC_TABLE_STORE_GC_AFTER_SWAP:-1}"
export HSPEC_TABLE_STORE_FSYNC_ON_SEAL="${HSPEC_TABLE_STORE_FSYNC_ON_SEAL:-0}"

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
ROLLOUT_LENGTH_DIR="${ROLLOUT_LENGTH_DIR:-${ROLL_LEN_ROOT}/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TB_ROOT}/${RUN_NAME}}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-${LOG_DIR}/${RUN_NAME}.log}"
OUT="${OUT:-/workspace/cann-recipes-train/llm_rl/qwen3/output/train_grpo_hspec-30b.txt}"

mkdir -p "${LOG_DIR}" "${ROLL_LEN_ROOT}" "${TB_ROOT}" "${ROLLOUT_LENGTH_DIR}" "$(dirname "${OUT}")"
hspec_maybe_clean_store_dirs

if [ "${USE_HSPEC_DECODE}" != "0" ] && [ "${NODES}" != "1" ] && \
   [ "${HSPEC_EXPERIMENTAL_ALLOW_MULTI_NODE_UNSAFE:-0}" = "0" ]; then
    echo "ERROR: HSpec Phase 1 descriptor path only supports single-node; NODES=${NODES}" >&2
    exit 2
fi

{
    echo
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') hspec single run ====="
    echo "project_root=${PROJECT_ROOT}"
    echo "run_name=${RUN_NAME}"
    echo "hspec_run_name=${HSPEC_RUN_NAME}"
    echo "config_dir=${CONFIG_DIR}"
    echo "model_path=${MODEL_PATH}"
    echo "train_file=${TRAIN_FILE}"
    echo "test_file=${TEST_FILE}"
    echo "out=${OUT}"
    echo "rollout_log_path=${ROLLOUT_LOG_PATH}"
    echo "rollout_length_dir=${ROLLOUT_LENGTH_DIR}"
    echo "tensorboard_dir=${TENSORBOARD_DIR}"
    echo "use_hspec_decode=${USE_HSPEC_DECODE}"
    echo "hspec_legacy_dataproto_hs=${HSPEC_LEGACY_DATAPROTO_HS}"
    echo "hspec_strict_descriptor_mode=${HSPEC_STRICT_DESCRIPTOR_MODE}"
    echo "hspec_store_dtype=${HSPEC_STORE_DTYPE}"
    echo "hspec_store_isolation_mode=${HSPEC_STORE_ISOLATION_MODE}"
    echo "hspec_run_uid=${HSPEC_RUN_UID}"
    echo "hspec_clean_store_on_start=${HSPEC_CLEAN_STORE_ON_START}"
    echo "hspec_clean_raw_store_on_start=${HSPEC_CLEAN_RAW_STORE_ON_START}"
    echo "hspec_clean_table_store_on_start=${HSPEC_CLEAN_TABLE_STORE_ON_START}"
    echo "hspec_allow_clean_outside_project=${HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT}"
    echo "hspec_require_fresh_table_store=${HSPEC_REQUIRE_FRESH_TABLE_STORE}"
    echo "hspec_store_dir=${HSPEC_STORE_DIR}"
    echo "hspec_table_store_dir=${HSPEC_TABLE_STORE_DIR}"
    echo "hspec_infer_tp=${HSPEC_INFER_TP}"
    echo "hspec_num_shards=${HSPEC_NUM_SHARDS}"
    echo "node_rank=${NODE_RANK}"
    echo "hspec_single_node_only=${HSPEC_SINGLE_NODE_ONLY}"
    echo "hspec_topology_strict=${HSPEC_TOPOLOGY_STRICT}"
    echo "hspec_require_explicit_num_shards=${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS}"
    echo "hspec_step0_runtime_asserts=${HSPEC_STEP0_RUNTIME_ASSERTS}"
    echo "hspec_build_actor_num_cpus=${HSPEC_BUILD_ACTOR_NUM_CPUS}"
    echo "hspec_build_blas_threads=${HSPEC_BUILD_BLAS_THREADS}"
    echo "hspec_build_actor_name_prefix=${HSPEC_BUILD_ACTOR_NAME_PREFIX}"
    echo "hspec_delete_trajectory_after_build=${HSPEC_DELETE_TRAJECTORY_AFTER_BUILD}"
    echo "hspec_raw_store_gc_after_epoch=${HSPEC_RAW_STORE_GC_AFTER_EPOCH}"
    echo "hspec_segment_fsync_on_seal=${HSPEC_SEGMENT_FSYNC_ON_SEAL}"
    echo "hspec_raw_store_max_bytes=${HSPEC_RAW_STORE_MAX_BYTES}"
    echo "hspec_raw_store_max_files=${HSPEC_RAW_STORE_MAX_FILES}"
    echo "hspec_store_retain_batches=${HSPEC_STORE_RETAIN_BATCHES}"
    echo "hspec_raw_store_budget_delete=${HSPEC_RAW_STORE_BUDGET_DELETE}"
    echo "hspec_pinned_pool_bytes=${HSPEC_PINNED_POOL_BYTES}"
    echo "hspec_pinned_pool_max_slots=${HSPEC_PINNED_POOL_MAX_SLOTS}"
    echo "hspec_pinned_pool_bucket_rows=${HSPEC_PINNED_POOL_BUCKET_ROWS}"
    echo "hspec_copy_max_pending_tasks=${HSPEC_COPY_MAX_PENDING_TASKS}"
    echo "hspec_copy_max_pending_rows=${HSPEC_COPY_MAX_PENDING_ROWS}"
    echo "hspec_drop_on_backpressure=${HSPEC_DROP_ON_BACKPRESSURE}"
    echo "hspec_build_max_prompt_rows=${HSPEC_BUILD_MAX_PROMPT_ROWS}"
    echo "hspec_build_max_prompt_raw_bytes=${HSPEC_BUILD_MAX_PROMPT_RAW_BYTES}"
    echo "hspec_build_max_prompt_descs=${HSPEC_BUILD_MAX_PROMPT_DESCS}"
    echo "hspec_build_max_rss_mb=${HSPEC_BUILD_MAX_RSS_MB}"
    echo "hspec_pca_method=${HSPEC_PCA_METHOD}"
    echo "hspec_pca_tile_rows=${HSPEC_PCA_TILE_ROWS}"
    echo "hspec_pca_random_oversample=${HSPEC_PCA_RANDOM_OVERSAMPLE}"
    echo "hspec_pca_random_seed=${HSPEC_PCA_RANDOM_SEED}"
    echo "hspec_pca_cov_max_bytes=${HSPEC_PCA_COV_MAX_BYTES}"
    echo "hspec_pca_accum_dtype=${HSPEC_PCA_ACCUM_DTYPE}"
    echo "hspec_table_keys_dtype=${HSPEC_TABLE_KEYS_DTYPE}"
    echo "hspec_table_file_align_bytes=${HSPEC_TABLE_FILE_ALIGN_BYTES}"
    echo "hspec_table_prefetch_mode=${HSPEC_TABLE_PREFETCH_MODE}"
    echo "hspec_allow_legacy_table_prefetch=${HSPEC_ALLOW_LEGACY_TABLE_PREFETCH}"
    echo "hspec_enable_zmq_query=${HSPEC_ENABLE_ZMQ_QUERY}"
    echo "hspec_proposer_hot_path_strict=${HSPEC_PROPOSER_HOT_PATH_STRICT}"
    echo "hspec_proposer_cache_max_prompts=${HSPEC_PROPOSER_CACHE_MAX_PROMPTS}"
    echo "hspec_proposer_cache_max_cpu_bytes=${HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES}"
    echo "hspec_proposer_cache_max_npu_bytes=${HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES}"
    echo "hspec_proposer_cache_max_entries=${HSPEC_PROPOSER_CACHE_MAX_ENTRIES}"
    echo "hspec_max_ready_prefetch_materialize=${HSPEC_MAX_READY_PREFETCH_MATERIALIZE}"
    echo "hspec_max_ready_prefetch_bytes=${HSPEC_MAX_READY_PREFETCH_BYTES}"
    echo "hspec_table_store_retain_versions=${HSPEC_TABLE_STORE_RETAIN_VERSIONS}"
    echo "hspec_table_store_gc_after_swap=${HSPEC_TABLE_STORE_GC_AFTER_SWAP}"
    echo "hspec_table_store_fsync_on_seal=${HSPEC_TABLE_STORE_FSYNC_ON_SEAL}"
    echo "hspec_profile=${HSPEC_PROFILE}"
    echo "hspec_dump=${HSPEC_DUMP}"
    echo "hspec_async_hs_accumulate=${HSPEC_ASYNC_HS_ACCUMULATE}"
    echo "hspec_async_hs_copy_stream=${HSPEC_ASYNC_HS_COPY_STREAM}"
    echo "pca_components=${PCA_COMPONENTS}"
    echo "vllm_spec_batch_threshold=${VLLM_SPECULATIVE_BATCH_SIZE_THRE}"
    echo "nodes=${NODES}"
    echo "trainer_n_gpus_per_node=16"
    echo "infer_tp=${INFER_TP}"
    echo "max_prompt_length=${MAX_PROMPT_LENGTH}"
    echo "max_response_length=${MAX_RESPONSE_LENGTH}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "rollout_n=${ROLLOUT_N}"
    echo "train_batch_size=${TRAIN_BATCH_SIZE}"
    echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
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
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STRICT_DESCRIPTOR_MODE='"'"${HSPEC_STRICT_DESCRIPTOR_MODE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_DTYPE='"'"${HSPEC_STORE_DTYPE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_ISOLATION_MODE='"'"${HSPEC_STORE_ISOLATION_MODE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RUN_UID='"'"${HSPEC_RUN_UID}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_STORE_ON_START='"'"${HSPEC_CLEAN_STORE_ON_START}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_RAW_STORE_ON_START='"'"${HSPEC_CLEAN_RAW_STORE_ON_START}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_CLEAN_TABLE_STORE_ON_START='"'"${HSPEC_CLEAN_TABLE_STORE_ON_START}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT='"'"${HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_REQUIRE_FRESH_TABLE_STORE='"'"${HSPEC_REQUIRE_FRESH_TABLE_STORE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_DIR='"'"${HSPEC_STORE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_DIR='"'"${HSPEC_TABLE_STORE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_NUM_SHARDS='"'"${HSPEC_NUM_SHARDS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_INFER_TP='"'"${HSPEC_INFER_TP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.NODE_RANK='"'"${NODE_RANK}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_SINGLE_NODE_ONLY='"'"${HSPEC_SINGLE_NODE_ONLY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TOPOLOGY_STRICT='"'"${HSPEC_TOPOLOGY_STRICT}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS='"'"${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STEP0_RUNTIME_ASSERTS='"'"${HSPEC_STEP0_RUNTIME_ASSERTS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_ACTOR_NUM_CPUS='"'"${HSPEC_BUILD_ACTOR_NUM_CPUS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_BLAS_THREADS='"'"${HSPEC_BUILD_BLAS_THREADS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_ACTOR_NAME_PREFIX='"'"${HSPEC_BUILD_ACTOR_NAME_PREFIX}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DELETE_TRAJECTORY_AFTER_BUILD='"'"${HSPEC_DELETE_TRAJECTORY_AFTER_BUILD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RAW_STORE_GC_AFTER_EPOCH='"'"${HSPEC_RAW_STORE_GC_AFTER_EPOCH}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_SEGMENT_FSYNC_ON_SEAL='"'"${HSPEC_SEGMENT_FSYNC_ON_SEAL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RAW_STORE_MAX_BYTES='"'"${HSPEC_RAW_STORE_MAX_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RAW_STORE_MAX_FILES='"'"${HSPEC_RAW_STORE_MAX_FILES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_STORE_RETAIN_BATCHES='"'"${HSPEC_STORE_RETAIN_BATCHES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_RAW_STORE_BUDGET_DELETE='"'"${HSPEC_RAW_STORE_BUDGET_DELETE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TP_GROUP_ID='"'"${HSPEC_TP_GROUP_ID}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PINNED_POOL_BYTES='"'"${HSPEC_PINNED_POOL_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PINNED_POOL_MAX_SLOTS='"'"${HSPEC_PINNED_POOL_MAX_SLOTS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PINNED_POOL_BUCKET_ROWS='"'"${HSPEC_PINNED_POOL_BUCKET_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_COPY_MAX_PENDING_TASKS='"'"${HSPEC_COPY_MAX_PENDING_TASKS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_COPY_MAX_PENDING_ROWS='"'"${HSPEC_COPY_MAX_PENDING_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DROP_ON_BACKPRESSURE='"'"${HSPEC_DROP_ON_BACKPRESSURE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_MAX_PROMPT_ROWS='"'"${HSPEC_BUILD_MAX_PROMPT_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_MAX_PROMPT_RAW_BYTES='"'"${HSPEC_BUILD_MAX_PROMPT_RAW_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_MAX_PROMPT_DESCS='"'"${HSPEC_BUILD_MAX_PROMPT_DESCS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_BUILD_MAX_RSS_MB='"'"${HSPEC_BUILD_MAX_RSS_MB}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_METHOD='"'"${HSPEC_PCA_METHOD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_TILE_ROWS='"'"${HSPEC_PCA_TILE_ROWS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_RANDOM_OVERSAMPLE='"'"${HSPEC_PCA_RANDOM_OVERSAMPLE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_RANDOM_SEED='"'"${HSPEC_PCA_RANDOM_SEED}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_COV_MAX_BYTES='"'"${HSPEC_PCA_COV_MAX_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PCA_ACCUM_DTYPE='"'"${HSPEC_PCA_ACCUM_DTYPE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_KEYS_DTYPE='"'"${HSPEC_TABLE_KEYS_DTYPE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_FILE_ALIGN_BYTES='"'"${HSPEC_TABLE_FILE_ALIGN_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_PREFETCH_MODE='"'"${HSPEC_TABLE_PREFETCH_MODE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ALLOW_LEGACY_TABLE_PREFETCH='"'"${HSPEC_ALLOW_LEGACY_TABLE_PREFETCH}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_ENABLE_ZMQ_QUERY='"'"${HSPEC_ENABLE_ZMQ_QUERY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_HOT_PATH_STRICT='"'"${HSPEC_PROPOSER_HOT_PATH_STRICT}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_CACHE_MAX_PROMPTS='"'"${HSPEC_PROPOSER_CACHE_MAX_PROMPTS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES='"'"${HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES='"'"${HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROPOSER_CACHE_MAX_ENTRIES='"'"${HSPEC_PROPOSER_CACHE_MAX_ENTRIES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_MAX_READY_PREFETCH_MATERIALIZE='"'"${HSPEC_MAX_READY_PREFETCH_MATERIALIZE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_MAX_READY_PREFETCH_BYTES='"'"${HSPEC_MAX_READY_PREFETCH_BYTES}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_RETAIN_VERSIONS='"'"${HSPEC_TABLE_STORE_RETAIN_VERSIONS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_GC_AFTER_SWAP='"'"${HSPEC_TABLE_STORE_GC_AFTER_SWAP}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TABLE_STORE_FSYNC_ON_SEAL='"'"${HSPEC_TABLE_STORE_FSYNC_ON_SEAL}"'"' \
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
