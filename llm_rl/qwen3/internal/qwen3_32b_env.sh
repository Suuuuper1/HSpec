# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0           # 0: disable rollout rebalance, 1: enable rollout rebalance
export VLLM_SPECULATIVE_BATCH_SIZE_THRE=32  # [TODO] configure SAM batch size threshold based on actual training configration. Should set to -1 when using random initialized weights.