# Copyright (c) 2026 HSpec Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Recipe features
export USE_HDP=0
export ROLLOUT_REBALANCE_ENABLE=0

# IMPORTANT:
# qwen3_32b_env.sh sets a SAM-oriented speculative batch threshold.
# HSpec should not be auto-disabled by batch size, so force disable the
# threshold gate in vllm_ascend.
export VLLM_SPECULATIVE_BATCH_SIZE_THRE=-1

# HSpec main switch
export USE_HSPEC_DECODE="${USE_HSPEC_DECODE:-True}"

# Basic debug / trace
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export HSPEC_DEBUG="${HSPEC_DEBUG:-0}"
export HSPEC_TRACE="${HSPEC_TRACE:-0}"
export HSPEC_DUMP="${HSPEC_DUMP:-0}"
export HSPEC_PROFILE="${HSPEC_PROFILE:-0}"
export HSPEC_DUMP_DIR="${HSPEC_DUMP_DIR:-/workspace/output/hspec_dump-rollout_1024}"

# Super params
export PCA_COMPONENTS="${PCA_COMPONENTS:-64}"

# Numba rebuild in hspec_proposer._build_batched_table_tensors()
export HSPEC_DISABLE_NUMBA_REBUILD="${HSPEC_DISABLE_NUMBA_REBUILD:-0}"
export HSPEC_NUMBA_REBUILD_MIN_ROWS="${HSPEC_NUMBA_REBUILD_MIN_ROWS:-0}"
export HSPEC_NUMBA_REBUILD_MIN_ELEMS="${HSPEC_NUMBA_REBUILD_MIN_ELEMS:-0}"

# Alignment debug
export HSPEC_ALIGN_DEBUG="${HSPEC_ALIGN_DEBUG:-0}"
export HSPEC_ALIGN_DEBUG_MAX_LOGS="${HSPEC_ALIGN_DEBUG_MAX_LOGS:-256}"
export HSPEC_ALIGN_DEBUG_PREVIEW="${HSPEC_ALIGN_DEBUG_PREVIEW:-8}"

# Optional analysis knobs
export HSPEC_ENTRY="${HSPEC_ENTRY:-0}"
export MATCH_WND="${MATCH_WND:-16}"
export HSPEC_ADVAN_NGRAM="${HSPEC_ADVAN_NGRAM:-1}"

# Fine-grained proposer timing logs
export HSPEC_GEN="${HSPEC_GEN:-0}"
export HSPEC_GEN_REQ_IDX="${HSPEC_GEN_REQ_IDX:-0}"
export HSPEC_GEN_MAX_CALLS="${HSPEC_GEN_MAX_CALLS:-0}"

# torch_npu profiler for HSpec path
export HSPEC_PROFILE="${HSPEC_PROFILE:-0}"
export HSPEC_PROFILE_STEPS="${HSPEC_PROFILE_STEPS:-12}"
export HSPEC_PROFILE_DIR="${HSPEC_PROFILE_DIR:-/home/xy/hspec_profile-19}"
export HSPEC_PROFILE_METHOD="${HSPEC_PROFILE_METHOD:-mstx}"
export HSPEC_PROFILE_LEVEL="${HSPEC_PROFILE_LEVEL:-level_none}"
export HSPEC_PROFILE_ANALYSE="${HSPEC_PROFILE_ANALYSE:-1}"
export HSPEC_PROFILE_WITH_STACK="${HSPEC_PROFILE_WITH_STACK:-0}"
export HSPEC_PROFILE_MEMORY="${HSPEC_PROFILE_MEMORY:-0}"

# Runtime logging
export HSPEC_LOG_EVERY_CALLS="${HSPEC_LOG_EVERY_CALLS:-50}"
export HSPEC_LOG_EVERY_S="${HSPEC_LOG_EVERY_S:-5}"
export HSPEC_LOG_LEVEL="${HSPEC_LOG_LEVEL:-INFO}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

