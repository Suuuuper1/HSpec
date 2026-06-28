#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BASELINE_DIR="${PROJECT_ROOT}/outputs/hspec_step0_baseline"
BASELINE_LOG="${BASELINE_DIR}/train.log"
BASELINE_JSON="${BASELINE_DIR}/baseline.json"

mkdir -p "${BASELINE_DIR}"

python "${SCRIPT_DIR}/validate_hspec_phase0_phase1_step0.py" --static
python "${SCRIPT_DIR}/validate_hspec_phase0_phase1_step0.py" --store-smoke
python "${SCRIPT_DIR}/validate_hspec_phase0_phase1_step0.py" --legacy-toggle

export HSPEC_LEGACY_DATAPROTO_HS=0
export HSPEC_STRICT_DESCRIPTOR_MODE="${HSPEC_STRICT_DESCRIPTOR_MODE:-1}"
export HSPEC_STORE_DTYPE="${HSPEC_STORE_DTYPE:-float16}"
export HSPEC_SINGLE_NODE_ONLY="${HSPEC_SINGLE_NODE_ONLY:-1}"
export HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS="${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS:-1}"
export HSPEC_STEP0_RUNTIME_ASSERTS="${HSPEC_STEP0_RUNTIME_ASSERTS:-1}"
export HSPEC_BUILD_ACTOR_NUM_CPUS="${HSPEC_BUILD_ACTOR_NUM_CPUS:-1}"
export HSPEC_DUMP="${HSPEC_DUMP:-1}"
export HSPEC_DUMP_DIR="${HSPEC_DUMP_DIR:-${PROJECT_ROOT}/outputs/hspec_step0_baseline_dump}"

{
    echo "HSPEC Step0 baseline run"
    echo "project_root=${PROJECT_ROOT}"
    echo "baseline_log=${BASELINE_LOG}"
    echo "baseline_json=${BASELINE_JSON}"
    echo "HSPEC_LEGACY_DATAPROTO_HS=${HSPEC_LEGACY_DATAPROTO_HS}"
    echo "HSPEC_STRICT_DESCRIPTOR_MODE=${HSPEC_STRICT_DESCRIPTOR_MODE}"
    echo "HSPEC_STORE_DTYPE=${HSPEC_STORE_DTYPE}"
    echo "HSPEC_SINGLE_NODE_ONLY=${HSPEC_SINGLE_NODE_ONLY}"
    echo "HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS=${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS}"
    echo "HSPEC_STEP0_RUNTIME_ASSERTS=${HSPEC_STEP0_RUNTIME_ASSERTS}"
    echo "HSPEC_BUILD_ACTOR_NUM_CPUS=${HSPEC_BUILD_ACTOR_NUM_CPUS}"
} > "${BASELINE_LOG}"

bash "${SCRIPT_DIR}/train_grpo_qwen2.5_1.5b_hspec-short.sh" \
  trainer.total_epochs=2 \
  "$@" 2>&1 | tee -a "${BASELINE_LOG}"

python "${SCRIPT_DIR}/validate_hspec_phase0_phase1_step0.py" --extract-baseline \
  --log-file "${BASELINE_LOG}" \
  --output-json "${BASELINE_JSON}"
