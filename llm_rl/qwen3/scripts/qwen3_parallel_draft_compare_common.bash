# Shared implementation for the six baseline-controlled DFlash/DSpark runs.
# This file is sourced by an experiment entrypoint; do not execute it directly.

set -eo pipefail

if [[ ${COMPARE_METHOD:-} != "dflash" && ${COMPARE_METHOD:-} != "dspark" ]]; then
  echo "ERROR: COMPARE_METHOD must be dflash or dspark" >&2
  exit 2
fi
if [[ ${COMPARE_TARGET_MODE:-} != "eager" && ${COMPARE_TARGET_MODE:-} != "piecewise" && ${COMPARE_TARGET_MODE:-} != "full_graph_decode" ]]; then
  echo "ERROR: COMPARE_TARGET_MODE must be eager, piecewise, or full_graph_decode" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && $1 != "dry-run" ) ]]; then
  echo "usage: ${BASH_SOURCE[1]} [dry-run]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONFIG_DIR=${CONFIG_DIR:-"${PROJECT_ROOT}/verl/trainer/config"}
CUSTOM_OPP_PATH="${PROJECT_ROOT}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
CUSTOM_OP_API_LIB="${CUSTOM_OPP_PATH}/op_api/lib"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_qwen3_baseline_eager_tq2.sh"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

if [ -d "${CUSTOM_OPP_PATH}" ]; then
  export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
fi
if [ -d "${CUSTOM_OP_API_LIB}" ]; then
  export LD_LIBRARY_PATH="${CUSTOM_OP_API_LIB}:${LD_LIBRARY_PATH:-}"
fi

# Match the baseline's clean-room environment boundary. Method paths use
# DFLASH_DRAFT_MODEL/DSPARK_DRAFT_MODEL, which are intentionally outside it.
while IFS= read -r name; do
  case "${name}" in
    VLLM_* | VERL_*) unset "${name}" ;;
  esac
done < <(compgen -e)

unset ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

readonly VLLM_INTERNAL_DP=16
readonly SPECULATIVE_K=7
readonly PROBABILITY_BUDGET_MB=512
readonly SPEC_GPU_MEMORY_UTILIZATION=0.72
readonly BASELINE_GPU_MEMORY_UTILIZATION=0.83

case "${COMPARE_METHOD}" in
  dflash)
    DRAFT_MODEL=${DFLASH_DRAFT_MODEL:-/home/sharedata/DFlash_DSpark_open_drafts/oldabi/RedHatAI-Qwen3-30B-A3B-dflash-edcff837}
    DRAFT_PAIR_ID=${DFLASH_DRAFT_PAIR_ID:-hf-RedHatAI-Qwen3-30B-A3B-dflash-edcff8378314-cfg-8b1b97c83e1c}
    GRAPH_HCCL_BASE_PORT_DEFAULT=63000
    ;;
  dspark)
    DRAFT_MODEL=${DSPARK_DRAFT_MODEL:-/home/sharedata/DFlash_DSpark_open_drafts/oldabi/CMB-Qwen3-30B-A3B-DSpark-8886bbc}
    DRAFT_PAIR_ID=${DSPARK_DRAFT_PAIR_ID:-hf-CMBTech-Qwen3-30B-A3B-DSpark-8886bbc2c880-cfg-4549f5cb2206}
    GRAPH_HCCL_BASE_PORT_DEFAULT=63032
    ;;
esac

TARGET_ENFORCE_EAGER=False
FRAMEWORK_CUDAGRAPH_MODE=NONE
GRAPH_LABEL=${COMPARE_TARGET_MODE}
GRAPH_ARGS=()
RAY_GRAPH_ENV_ARGS=()
case "${COMPARE_TARGET_MODE}" in
  eager)
    TARGET_ENFORCE_EAGER=True
    export VLLM_ENABLE_GRAPH_MODE=0
    export HCCL_OP_EXPANSION_MODE=HOST
    export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-50021}
    GRAPH_ARGS+=(actor_rollout_ref.rollout.parallel_draft_allow_target_eager_experiment=True)
    ;;
  piecewise)
    FRAMEWORK_CUDAGRAPH_MODE=PIECEWISE
    export VLLM_ENABLE_GRAPH_MODE=1
    export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-${GRAPH_HCCL_BASE_PORT_DEFAULT}}
    GRAPH_ARGS+=(actor_rollout_ref.rollout.cudagraph_mode=PIECEWISE)
    RAY_GRAPH_ENV_ARGS+=(
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_CUDAGRAPH_MODE="'PIECEWISE'"
    )
    ;;
  full_graph_decode)
    # FULL_GRAPH_DECODE is the experiment label. The vLLM 0.14.1 enum is
    # FULL_DECODE_ONLY: full graph for uniform decode, eager for mixed/prefill.
    FRAMEWORK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
    export VLLM_ENABLE_GRAPH_MODE=1
    export VERL_VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY
    export VLLM_ASCEND_STRICT_FULL_GRAPH=1
    export VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-${GRAPH_HCCL_BASE_PORT_DEFAULT}}
    GRAPH_ARGS+=(actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY)
    RAY_GRAPH_ENV_ARGS+=(
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_CUDAGRAPH_MODE="'FULL_DECODE_ONLY'"
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_STRICT_FULL_GRAPH="'1'"
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_FULL_GRAPH_MOE_REQUIRE_AIV="'1'"
    )
    ;;
esac

# Baseline runtime settings. Only HCCL expansion/port and graph enablement above
# differ when required by a graph mode.
export TASK_QUEUE_ENABLE=2
export VLLM_USE_V1=1
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0
export VLLM_ASCEND_EAGER_COMPILE=1
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_DP_SIZE=${VLLM_INTERNAL_DP}
export VLLM_ENABLE_MC2=1
export ALL_TO_ALL_RESHARD=1
export USE_ALLTOALL_OVERLAP=1
export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
export VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=0
export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=0
export VLLM_ASCEND_USE_LEGACY_ATTENTION=0
export VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE=0
export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
export VLLM_ASCEND_MC2_GLOBAL_BS=0
export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64
export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1
export VLLM_QWEN3_MOE_REDUCE_RESULTS=1
export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1
export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
export VLLM_ROLLOUT_SLEEP_LEVEL=2
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:24"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_BUFFSIZE=800
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export MASTER_PORT=${MASTER_PORT:-23300}
export D2D_DATA_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_ENABLE_NZ=0

MODEL_PATH=${MODEL_PATH:-/home/data/Qwen3-30B-A3B}
DISTCP_PATH=${DISTCP_PATH:-/home/data/Qwen3-30B-A3B_megatron}
TRAIN_FILE=${TRAIN_FILE:-/data/deepscaler/train.parquet}
TEST_FILE=${TEST_FILE:-/data/deepscaler/test.parquet}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"${PROJECT_ROOT}/deepscaler.py"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/rl/qwen3_${COMPARE_METHOD}_${GRAPH_LABEL}_tq2_bs32_n16"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
DATASET_FRACTION=${DATASET_FRACTION:-0.005}
EXPERIMENT_NAME=${COMPARE_METHOD}_${GRAPH_LABEL}_tq2_bs32_n16

ARGS=(
  --config-path="${CONFIG_DIR}"
  --config-name=ppo_megatron_trainer.yaml
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILE}"
  data.val_files="${TEST_FILE}"
  data.train_batch_size=32
  data.max_prompt_length=1024
  data.max_response_length=16384
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=False
  data.dataloader_num_workers=0
  +data.dataset_fraction="${DATASET_FRACTION}"
  custom_reward_function.path="${REWARD_FUNCTION_PATH}"
  custom_reward_function.name=compute_score
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_fused_kernels=False
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.clip_grad=10000
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480
  actor_rollout_ref.actor.megatron.sequence_parallel=True
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=False
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.load_weight=True
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1
  actor_rollout_ref.rollout.data_parallel_size=1
  actor_rollout_ref.rollout.vllm_data_parallel_size=${VLLM_INTERNAL_DP}
  actor_rollout_ref.rollout.enforce_eager=${TARGET_ENFORCE_EAGER}
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.gpu_memory_utilization=${SPEC_GPU_MEMORY_UTILIZATION}
  actor_rollout_ref.rollout.max_num_batched_tokens=17408
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.n=16
  actor_rollout_ref.rollout.temperature=0.9
  actor_rollout_ref.rollout.top_k=50
  actor_rollout_ref.rollout.top_p=0.9
  actor_rollout_ref.rollout.ignore_eos=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480
  actor_rollout_ref.rollout.speculative_method=${COMPARE_METHOD}
  actor_rollout_ref.rollout.speculative_model="${DRAFT_MODEL}"
  actor_rollout_ref.rollout.num_speculative_tokens=${SPECULATIVE_K}
  actor_rollout_ref.rollout.draft_tensor_parallel_size=1
  actor_rollout_ref.rollout.draft_sample_method=probabilistic
  actor_rollout_ref.rollout.draft_probability_max_memory_mb=${PROBABILITY_BUDGET_MB}
  actor_rollout_ref.rollout.draft_load_format=auto
  actor_rollout_ref.rollout.rejection_sample_method=standard
  actor_rollout_ref.rollout.speculative_enforce_eager=True
  actor_rollout_ref.rollout.parallel_draft_profile_enabled=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4
  actor_rollout_ref.ref.megatron.context_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=4
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.load_weight=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
  actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}"
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.balance_batch=False
  trainer.device=npu
  trainer.val_before_train=False
  trainer.critic_warmup=0
  trainer.logger="[console,tensorboard]"
  trainer.project_name=qwen3_baseline
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.n_gpus_per_node=16
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.resume_mode=disable
  trainer.default_local_dir="${OUTPUT_DIR}"
  +trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data"
  +trainer.rollout_length_dir="${OUTPUT_DIR}/rollout_length"
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers="[[11],[13],[13],[11]]"
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11
  +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11
  +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DP_SIZE="'${VLLM_INTERNAL_DP}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ENABLE_GRAPH_MODE="'${VLLM_ENABLE_GRAPH_MODE}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE="'${HCCL_OP_EXPANSION_MODE}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.HCCL_IF_BASE_PORT="'${HCCL_IF_BASE_PORT}'"
  "${GRAPH_ARGS[@]}"
  "${RAY_GRAPH_ENV_ARGS[@]}"
)

if [ "${1:-}" = "dry-run" ]; then
  printf 'baseline_script=%s\n' "${BASELINE_SCRIPT}"
  printf 'method=%s\ntarget_mode_label=%s\nframework_cudagraph_mode=%s\n' \
    "${COMPARE_METHOD}" "${COMPARE_TARGET_MODE}" "${FRAMEWORK_CUDAGRAPH_MODE}"
  printf 'target_enforce_eager=%s\ndraft_enforce_eager=True\n' "${TARGET_ENFORCE_EAGER}"
  printf 'vllm_dp=%s\nrollout_dispatch_dp=1\ntarget_tp=1\ndraft_tp=1\n' "${VLLM_INTERNAL_DP}"
  printf 'draft_sample_method=probabilistic\nnum_speculative_tokens=%s\n' "${SPECULATIVE_K}"
  printf 'gpu_memory_utilization=%s\nbaseline_gpu_memory_utilization=%s\n' \
    "${SPEC_GPU_MEMORY_UTILIZATION}" "${BASELINE_GPU_MEMORY_UTILIZATION}"
  printf 'draft_model=%s\ndraft_pair_id=%s\n' "${DRAFT_MODEL}" "${DRAFT_PAIR_ID}"
  printf 'hccl_op_expansion_mode=%s\nhccl_if_base_port=%s\n' \
    "${HCCL_OP_EXPANSION_MODE}" "${HCCL_IF_BASE_PORT}"
  printf '%q ' python3 -m verl.trainer.main_ppo "${ARGS[@]}"
  printf '\n'
  exit 0
fi

for path in "${MODEL_PATH}" "${DISTCP_PATH}" "${DRAFT_MODEL}"; do
  if [ ! -d "${path}" ]; then
    echo "ERROR: required model/checkpoint directory is missing: ${path}" >&2
    exit 3
  fi
done
for path in "${TRAIN_FILE}" "${TEST_FILE}" "${REWARD_FUNCTION_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "ERROR: required input file is missing: ${path}" >&2
    exit 3
  fi
done
if [ -e "${OUTPUT_DIR}" ]; then
  echo "ERROR: refusing to reuse comparison output directory: ${OUTPUT_DIR}" >&2
  exit 3
fi

exec 9>"${QWEN3_COMPARE_LOCK_PATH:-/tmp/qwen3_parallel_draft_compare_16npu.lock}"
if ! flock -n 9; then
  echo "ERROR: another qwen3 comparison run owns the 16-NPU lock" >&2
  exit 4
fi

if [ "${COMPARE_TARGET_MODE}" != "eager" ]; then
  python3 "${PROJECT_ROOT}/HSpec_research_doc/repair/DFlash_DSpark_migrate/phase3/tools/hccl_port_preflight.py" \
    --base-port "${HCCL_IF_BASE_PORT}" --count 32 >/dev/null
fi

export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
mkdir -p \
  "${OUTPUT_DIR}/logs" \
  "${OUTPUT_DIR}/rollout_data" \
  "${OUTPUT_DIR}/rollout_length" \
  "${TENSORBOARD_DIR}"
LOG_FILE="${OUTPUT_DIR}/logs/${COMPARE_METHOD}_${GRAPH_LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"
MANIFEST_FILE="${OUTPUT_DIR}/comparison_manifest.txt"
{
  printf 'schema_version=qwen3.parallel-draft-comparison.v1\n'
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'baseline_script=%s\n' "${BASELINE_SCRIPT}"
  printf 'baseline_sha256=%s\n' "$(sha256sum "${BASELINE_SCRIPT}" | cut -d' ' -f1)"
  printf 'entry_script=%s\n' "${BASH_SOURCE[1]}"
  printf 'entry_sha256=%s\n' "$(sha256sum "${BASH_SOURCE[1]}" | cut -d' ' -f1)"
  printf 'common_script=%s\n' "${BASH_SOURCE[0]}"
  printf 'common_sha256=%s\n' "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
  printf 'method=%s\ntarget_mode_label=%s\nframework_cudagraph_mode=%s\n' \
    "${COMPARE_METHOD}" "${COMPARE_TARGET_MODE}" "${FRAMEWORK_CUDAGRAPH_MODE}"
  printf 'target_enforce_eager=%s\ndraft_enforce_eager=True\n' "${TARGET_ENFORCE_EAGER}"
  printf 'vllm_dp=%s\nrollout_dispatch_dp=1\ntarget_tp=1\ndraft_tp=1\n' "${VLLM_INTERNAL_DP}"
  printf 'draft_sample_method=probabilistic\nnum_speculative_tokens=%s\n' "${SPECULATIVE_K}"
  printf 'gpu_memory_utilization=%s\nbaseline_gpu_memory_utilization=%s\n' \
    "${SPEC_GPU_MEMORY_UTILIZATION}" "${BASELINE_GPU_MEMORY_UTILIZATION}"
  printf 'draft_model=%s\ndraft_pair_id=%s\n' "${DRAFT_MODEL}" "${DRAFT_PAIR_ID}"
  printf 'hccl_op_expansion_mode=%s\nhccl_if_base_port=%s\n' \
    "${HCCL_OP_EXPANSION_MODE}" "${HCCL_IF_BASE_PORT}"
  printf 'command='
  printf '%q ' python3 -m verl.trainer.main_ppo "${ARGS[@]}"
  printf '\n'
} > "${MANIFEST_FILE}"
cp "${MANIFEST_FILE}" "${LOG_FILE}"

set +e
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
TRAINING_RC=${PIPESTATUS[0]}
set -e
if [ "${TRAINING_RC}" -ne 0 ]; then
  exit "${TRAINING_RC}"
fi

# Reject silent target graph-mode fallback. Draft eager is independently
# enforced by RolloutConfig and logged in the resolved speculative manifest.
if ! grep -q "'method': '${COMPARE_METHOD}'" "${LOG_FILE}" || \
   ! grep -q "'draft_enforce_eager': True" "${LOG_FILE}"; then
  echo "ERROR: resolved speculative method/draft eager evidence is absent" >&2
  exit 5
fi
case "${COMPARE_TARGET_MODE}" in
  eager)
    grep -q "Compilation disabled, using eager mode by default" "${LOG_FILE}" || {
      echo "ERROR: target eager mode was not observed" >&2; exit 5;
    }
    ;;
  piecewise)
    grep -q "PIECEWISE compilation enabled on NPU" "${LOG_FILE}" || {
      echo "ERROR: target PIECEWISE mode was not observed" >&2; exit 5;
    }
    ;;
  full_graph_decode)
    grep -q "FULL_DECODE_ONLY compilation enabled on NPU" "${LOG_FILE}" || {
      echo "ERROR: target FULL_DECODE_ONLY mode was not observed" >&2; exit 5;
    }
    ;;
esac

echo "PASS: ${COMPARE_METHOD}/${COMPARE_TARGET_MODE} completed; log=${LOG_FILE}"
