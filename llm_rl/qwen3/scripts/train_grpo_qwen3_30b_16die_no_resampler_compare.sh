#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/train_grpo_qwen3_30b_16die_true_weight.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
    echo "Base script not found: ${BASE_SCRIPT}" >&2
    exit 1
fi

MODE=${1:-all}
shift || true

OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/../outputs/rl"}
LOG_DIR="${OUTPUT_ROOT}/logs"
ROLL_LEN_ROOT="${OUTPUT_ROOT}/rollout_length"
TB_ROOT="${OUTPUT_ROOT}/tensorboard"

mkdir -p "${LOG_DIR}" "${ROLL_LEN_ROOT}" "${TB_ROOT}"

run_case() {
    local name="$1"
    local spec_method="$2"
    shift 2
    local run_name="qwen3_30b_verl_true_weights_${name}_no_resampler"
    local log_path="${LOG_DIR}/${run_name}.log"
    local roll_len_dir="${ROLL_LEN_ROOT}/${run_name}"
    local tb_dir="${TB_ROOT}/${run_name}"

    mkdir -p "${roll_len_dir}" "${tb_dir}"

    local -a extra_args=()
    if [ "${spec_method}" = "null" ]; then
        extra_args+=("actor_rollout_ref.rollout.spec_method=null")
    else
        extra_args+=("actor_rollout_ref.rollout.spec_method=${spec_method}")
    fi

    echo "===== running ${name} ====="
    echo "log_path=${log_path}"
    echo "rollout_length_dir=${roll_len_dir}"
    echo "tensorboard_dir=${tb_dir}"

    env \
        VLLM_DYNAMIC_RL_LOG_PATH="${log_path}" \
        ROLLOUT_LENGTH_DIR="${roll_len_dir}" \
        TENSORBOARD_DIR="${tb_dir}" \
        bash "${BASE_SCRIPT}" "${extra_args[@]}" "$@"
}

case "${MODE}" in
    baseline)
        run_case "baseline" "null" "$@"
        ;;
    history_tree)
        run_case "history_tree" "history_tree" "$@"
        ;;
    dynamic_rl)
        run_case "dynamic_rl" "dynamic_rl" "$@"
        ;;
    all)
        run_case "baseline" "null" "$@"
        run_case "history_tree" "history_tree" "$@"
        run_case "dynamic_rl" "dynamic_rl" "$@"
        ;;
    *)
        echo "Usage: $0 [baseline|history_tree|dynamic_rl|all] [extra hydra args...]" >&2
        exit 1
        ;;
esac
