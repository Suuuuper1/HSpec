#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

export HSPEC_LEGACY_DATAPROTO_HS=0
export HSPEC_STRICT_DESCRIPTOR_MODE="${HSPEC_STRICT_DESCRIPTOR_MODE:-1}"
export HSPEC_STORE_DTYPE="${HSPEC_STORE_DTYPE:-float16}"
export HSPEC_SINGLE_NODE_ONLY="${HSPEC_SINGLE_NODE_ONLY:-1}"
export HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS="${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS:-1}"
export HSPEC_STEP0_RUNTIME_ASSERTS="${HSPEC_STEP0_RUNTIME_ASSERTS:-0}"
export HSPEC_DUMP="${HSPEC_DUMP:-1}"
export HSPEC_DUMP_DIR="${HSPEC_DUMP_DIR:-${PROJECT_ROOT}/outputs/hspec_acceptance_dump}"

echo "HSPEC Phase 1 acceptance run"
echo "project_root=${PROJECT_ROOT}"
echo "hspec_dump_dir=${HSPEC_DUMP_DIR}"
echo "hspec_legacy_dataproto_hs=${HSPEC_LEGACY_DATAPROTO_HS}"
echo "hspec_strict_descriptor_mode=${HSPEC_STRICT_DESCRIPTOR_MODE}"
echo "hspec_store_dtype=${HSPEC_STORE_DTYPE}"
echo "hspec_single_node_only=${HSPEC_SINGLE_NODE_ONLY}"
echo "hspec_require_explicit_num_shards=${HSPEC_REQUIRE_EXPLICIT_NUM_SHARDS}"
echo "hspec_step0_runtime_asserts=${HSPEC_STEP0_RUNTIME_ASSERTS}"
echo "requirement=run at least 2 epochs and verify epoch1 tables are used in epoch2"

bash "${SCRIPT_DIR}/train_grpo_qwen2.5_1.5b_hspec-short.sh" \
  trainer.total_epochs=2 \
  "$@"
