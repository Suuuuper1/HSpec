#!/usr/bin/env bash
# Baseline-controlled DFlash run: eager target, eager draft.
COMPARE_METHOD=dflash
COMPARE_TARGET_MODE=eager
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_parallel_draft_compare_common.bash"
