#!/usr/bin/env bash
# Baseline-controlled DSpark run: FULL_GRAPH_DECODE label (FULL_DECODE_ONLY ABI), eager draft.
COMPARE_METHOD=dspark
COMPARE_TARGET_MODE=full_graph_decode
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_parallel_draft_compare_common.bash"
