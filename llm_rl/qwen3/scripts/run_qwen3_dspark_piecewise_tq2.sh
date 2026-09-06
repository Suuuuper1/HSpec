#!/usr/bin/env bash
# Baseline-controlled DSpark run: PIECEWISE target, eager draft.
COMPARE_METHOD=dspark
COMPARE_TARGET_MODE=piecewise
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_parallel_draft_compare_common.bash"
