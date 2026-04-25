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

set -e
set -o pipefail

# Install torch and torch-npu
python3 -m pip install torch==2.8.0 torch-npu==2.8.0.post2

# MindSpeed
git clone https://gitcode.com/ascend/MindSpeed.git && \
cd MindSpeed    && \
git checkout master  && \
pip3 install -r requirements.txt && \
pip3 install -e . && \
cd ..

# MindSpeed-LLM & Megatron-LM
git clone https://gitcode.com/Ascend/MindSpeed-LLM.git && \
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM  && \
git checkout core_v0.12.1   && \
cp -r megatron ../MindSpeed-LLM/    && \
cd ../MindSpeed-LLM && \
git checkout 6fe0df7fa2ce64fffee1f6304c73a1dda161187c && \
mkdir logs && \
cd ../

# Install other dependencies
python3 -m pip install torchvision==0.23.0
python3 -m pip install transformers==5.1.0
python3 -m pip install datasets==2.16.0
python3 -m pip install setuptools==79.0.1
python3 -m pip uninstall scikit-learn -y

cp -r qwen3_dense MindSpeed-LLM/examples/mcore

source /home/developer/Ascend/cann-8.5.0/set_env.sh