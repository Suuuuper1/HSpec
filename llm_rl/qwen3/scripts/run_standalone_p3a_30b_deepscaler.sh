#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
FROZEN_RUNNER="${PROJECT_ROOT}/HSpec_research_doc/HSpec_draft_delect_optim/s0_baseline_freeze/bin/run_frozen_profile.sh"
IDLE_TOOL="${PROJECT_ROOT}/HSpec_research_doc/HSpec_draft_delect_optim/s2_baseline_audit/tools/wait_for_s2_npu_idle.py"
EXECUTION_GATE="${PROJECT_ROOT}/outputs/hspec_draft_select_research/s13_patch3a_utility/manual_v2_20260811T065928Z/target_shadow_analysis/gate_result.json"
EXECUTION_GATE_SHA256="24e694d7a3d4cf28929c72d00d79d3e039365b86272e372b56541a8e4255e6e1"
MODEL_ARTIFACT="${PROJECT_ROOT}/HSpec_research_doc/HSpec_draft_delect_optim/s12_to_s13_transition/candidate/transition_candidate.json"
MODEL_ARTIFACT_SHA256="c3982dd40b1124d14c942dee1a010c7c441bbbbdff30d5f86b1bc1c025d0f869"
ENTRY_GATE="${PROJECT_ROOT}/HSpec_research_doc/HSpec_draft_delect_optim/s13_patch3a_utility/artifacts/s13_entry_gate.json"
ENTRY_GATE_SHA256="5afb4a4c4c57d9032775832b69a88b6ae571d9dce90edf4aaab889e5453cc6b0"

SEED=20260721
OUTPUT_ROOT=""
IDLE_TIMEOUT_SECONDS=1800
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/run_standalone_p3a_30b_deepscaler.sh [options]

Run one standalone Qwen3-30B DeepScaler P3A experiment with the frozen S13
performance parameters, without predecessor, paired-arm, ordering, or analysis
gates.

Options:
  --output-root PATH          Explicit output root (default: unique UTC path)
  --idle-timeout-seconds N    NPU/Ray idle wait timeout (default: 1800)
  --dry-run                   Validate inputs and print the resolved command
  -h, --help                  Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="${2:?--output-root requires a path}"
            shift 2
            ;;
        --idle-timeout-seconds)
            IDLE_TIMEOUT_SECONDS="${2:?--idle-timeout-seconds requires an integer}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "${IDLE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --idle-timeout-seconds must be a positive integer" >&2
    exit 2
fi

STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
if [ -z "${OUTPUT_ROOT}" ]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/outputs/hspec_draft_select_research/standalone_p3a_30b_deepscaler/${STAMP}"
fi
OUTPUT_ROOT=$(realpath -m "${OUTPUT_ROOT}")
mkdir -p "${OUTPUT_ROOT}"

verify_file_sha256() {
    local path="$1"
    local expected="$2"
    local label="$3"
    if [ ! -f "${path}" ]; then
        echo "ERROR: missing ${label}: ${path}" >&2
        exit 3
    fi
    local actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [ "${actual}" != "${expected}" ]; then
        echo "ERROR: ${label} SHA-256 mismatch" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual:   ${actual}" >&2
        exit 3
    fi
}

verify_file_sha256 "${MODEL_ARTIFACT}" "${MODEL_ARTIFACT_SHA256}" "P3A model artifact"
verify_file_sha256 "${ENTRY_GATE}" "${ENTRY_GATE_SHA256}" "P3A entry gate"
verify_file_sha256 "${EXECUTION_GATE}" "${EXECUTION_GATE_SHA256}" "P3A 30B execution gate"

for required_file in \
    "${FROZEN_RUNNER}" \
    "${IDLE_TOOL}" \
    "/data/deepscaler/train.parquet" \
    "/data/deepscaler/test.parquet"; do
    if [ ! -f "${required_file}" ]; then
        echo "ERROR: required file is missing: ${required_file}" >&2
        exit 3
    fi
done
for required_dir in \
    "/home/data/Qwen3-30B-A3B" \
    "/home/data/Qwen3-30B-A3B_megatron"; do
    if [ ! -d "${required_dir}" ]; then
        echo "ERROR: required directory is missing: ${required_dir}" >&2
        exit 3
    fi
done

DEVICE_NODE_COUNT=$(find /dev -maxdepth 1 -type c -regextype posix-extended \
    -regex '/dev/davinci[0-9]+' | wc -l)
if [ "${DEVICE_NODE_COUNT}" -ne 16 ]; then
    echo "ERROR: expected 16 /dev/davinciN device nodes, found ${DEVICE_NODE_COUNT}" >&2
    exit 4
fi

python3 - <<'PY'
import torch
import torch_npu  # noqa: F401

count = torch.npu.device_count()
if count != 16:
    raise SystemExit(f"ERROR: torch_npu reports {count} devices; expected 16")
print(f"NPU topology preflight: torch_npu devices={count}")
PY

# Ray's automatic acl.rt.get_device_count() probe returned 8 during the failed
# standalone launch even though torch_npu and /dev exposed all 16 dies. Pinning
# the resource removes that transient discovery path without changing worker
# placement or the frozen TP4 x PP4 training topology.
export RAY_OVERRIDE_RESOURCES='{"NPU":16}'
export RAY_TMPDIR="/home/sharedata/rp3a"
mkdir -p "${RAY_TMPDIR}"

RAY_TMP_AVAILABLE_KB=$(df -Pk "${RAY_TMPDIR}" | awk 'NR==2 {print $4}')
if [ "${RAY_TMP_AVAILABLE_KB}" -lt $((100 * 1024 * 1024)) ]; then
    echo "ERROR: Ray temporary filesystem has less than 100 GiB free: ${RAY_TMPDIR}" >&2
    exit 4
fi

cat > "${OUTPUT_ROOT}/standalone_preflight.txt" <<EOF
generated_at_utc=${STAMP}
expected_npu_dies=16
device_node_count=${DEVICE_NODE_COUNT}
ray_override_resources=${RAY_OVERRIDE_RESOURCES}
ray_tmpdir=${RAY_TMPDIR}
ray_tmp_available_kb=${RAY_TMP_AVAILABLE_KB}
execution_gate=${EXECUTION_GATE}
execution_gate_sha256=${EXECUTION_GATE_SHA256}
model_artifact_sha256=${MODEL_ARTIFACT_SHA256}
entry_gate_sha256=${ENTRY_GATE_SHA256}
seed=${SEED}
EOF

EXTRA_ARGS=()
if [ "${DRY_RUN}" -eq 1 ]; then
    EXTRA_ARGS+=(--dry-run)
else
    LOCK_PATH="${HSPEC_STANDALONE_NPU_LOCK_PATH:-/tmp/hspec_npu_experiment.lock}"
    exec 9>"${LOCK_PATH}"
    flock -n 9 || {
        echo "ERROR: another NPU experiment owns ${LOCK_PATH}" >&2
        exit 4
    }
    python3 "${IDLE_TOOL}" \
        --timeout-seconds "${IDLE_TIMEOUT_SECONDS}" \
        --minimum-idle-seconds 120 \
        --stable-observations 3 \
        --expected-cards 8 \
        --expected-chips 16 \
        --output "${OUTPUT_ROOT}/preflight_npu_idle_${STAMP}.json"
fi

echo "Standalone P3A output root: ${OUTPUT_ROOT}"
echo "Ray resources: ${RAY_OVERRIDE_RESOURCES}"
echo "Ray temporary root: ${RAY_TMPDIR}"

cd "${PROJECT_ROOT}"
HSPEC_S0_SKIP_VALIDATE=1 \
HSPEC_S13_FASTPATH_VERSION=p3-utility-first-batch-v1 \
HSPEC_SELECT_EXECUTION_GATE_PATH="${EXECUTION_GATE}" \
HSPEC_SELECT_EXECUTION_GATE_SHA256="${EXECUTION_GATE_SHA256}" \
HSPEC_SELECT_EXECUTION_LEVEL=performance \
MODEL_PATH=/home/data/Qwen3-30B-A3B \
TRAIN_FILE=/data/deepscaler/train.parquet \
TEST_FILE=/data/deepscaler/test.parquet \
DISTCP_PATH=/home/data/Qwen3-30B-A3B_megatron \
bash "${FROZEN_RUNNER}" \
    performance_30b \
    --seed "${SEED}" \
    --stage S13 \
    --hypothesis H-S13-P3A-UTILITY-SURVIVAL \
    --comparison C-S13-P3A-P0-R1 \
    --code-delta-id s13-patch3a \
    --s13-arm p3 \
    --output-root "${OUTPUT_ROOT}" \
    "${EXTRA_ARGS[@]}"
