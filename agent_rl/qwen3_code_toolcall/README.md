# 基于verl框架和代码沙盒环境工具调用的代码强化学习实践

## 概述

本项目基于 Qwen3-1.7B 模型，采用 [verl](https://github.com/volcengine/verl) 强化学习框架，基于适配昇腾平台的代码沙盒服务 [ScaleBox](https://gitcode.com/icip-cas/ScaleBox)，实现了高效且稳定的 ***长上下文多轮工具调用 Code RL*** 训练。我们的贡献主要有：

1. 开发了一个可扩展的分布式代码执行沙盒 ScaleBox，支持大规模多机部署、主流 RL 框架兼容训练及多种模型与基准的高效统一评估；
2. 提供了融合 verl 与 ScaleBox 的统一部署镜像，支持在单一节点中同时运行 ScaleBox 服务与 verl 训练任务，同时支持零成本迁移至华为云平台 [ModelArts](https://www.huaweicloud.com/product/modelarts.html)；
3. 在昇腾环境中验证了基于 verl 框架与 ScaleBox 沙盒的 Code RL 训练效果。
4. 组织Coding Toolcall的SFT数据以及SFT策略，并在RL中引入Coding Agent多轮工具调用的训练（首个开源的基于verl支持多轮工具调用的Coding Agent RL Recipe）。
5. 提供将推测解码（EAGLE3 与 Suffix）集成至 verl + vLLM-Ascend Rollout 流水线的补丁，并实现无累计开销的逐步推测解码指标采集——包括草稿 Token 数、接受 Token 数、草稿接受率、平均接受长度及逐位置接受率。
6. 在昇腾 NPU 上验证了 EAGLE3 与 Suffix 推测解码在多轮工具调用 Code RL 训练中的效果，端到端吞吐量最高提升 **38%**，训练步骤时间缩短 **25%**，且精度无损失。

[ScaleBox](https://gitcode.com/icip-cas/ScaleBox) 是一个可扩展的分布式代码执行沙盒，其核心特性包括：

1. 可扩展的分布式代码沙盒体系
    - 支持多机分布式沙盒部署与请求负载均衡
    - 支持单元测试并行与实例级并行

2. 面向 Code RL 的统一训练接口和评估套件
    - 提供高效的批量评估接口 `common_evaluate_batch`，相较于 `run_code`，通过单次请求处理多个测试用例，显著提升训练效率
    - 内置对 LiveCodeBench、HumanEval、MBPP 等主流代码评测基准的支持，实现一键式快速评估

3. 灵活的 Special Judge 判题机制
    - 支持自定义判题逻辑，能够灵活适应具有多种正确答案的复杂编程题目

---

## 硬件要求

Atlas A2/A3 系列产品，单机八卡。

---

## 软件要求

基础配方和SD扩展共享相同的verl提交，但vLLM版本不同。SD扩展使用的是vLLM `0.13.0` 和 vLLM-Ascend `v0.13.0` 版本，这些版本相比 `0.11.0` 带来了更稳定的推测解码支持和异步实现。请注意，以下软件版本反映了测试环境的情况——CANN `8.3.RC1` 预计同样适用于SD扩展。

| 组件 | 基础样例 | 推测解码扩展 |
|---|---|---|
| verl | commit `c651b7b`（基于 v0.7.0.dev） | commit `c651b7b`（基于 v0.7.0.dev） |
| vllm | `0.11.0` | `0.13.0` |
| vllm-ascend | `v0.11.0rc1` | `v0.13.0` |
| CANN | `8.3.RC1` | `8.5.0` |

---

## 文件说明

```
├── patches
│   ├── verl                                                        # verl 补丁目录
│   │   ├── 0001-verl-feature-improve_rl_usability.patch           # Code RL 通用可用性改进（共享）
│   │   ├── 0002-enable-tool-agent-loop.patch                      # 多轮工具调用支持（共享）
│   │   ├── 0003-toolcall-reward.patch                             # 工具调用奖励（基础样例）
│   │   └── 0004-enable-eagle-specrl-clean.patch                   # Suffix/EAGLE3 推测解码集成（推测解码扩展）
│   └── vllm
│       └── 0001-enable-eagle-sprl.patch                           # vLLM 侧 EAGLE3 推测解码支持（推测解码扩展）
├── figures
│   ├── evaluation_progress.png                                     # 训练 ckpts 的测试折线图（基础样例）
│   ├── training_progress.png                                       # 训练指标进度折线图（基础样例）
│   ├── sd_nosd_accuracy.png                                        # 精度对比：推测解码 vs. 无推测解码（推测解码扩展）
│   ├── throughput_speedup.png                                      # 吞吐量加速结果（推测解码扩展）
│   └── acceptance_rate_overall.png                                 # RL 训练过程中草稿接受率变化（推测解码扩展）
├── tool_config
│   └── scalebox_tool_config.yaml                                   # ScaleBox 工具调用配置文件（共享）
├── build_dataset.py                                                 # RL 训练数据集构建脚本（共享）
├── filter_sft_data.py                                              # SFT 工具调用数据集构建脚本（基础样例）
├── scalebox.py                                                      # verl 适配 ScaleBox 的自定义奖励函数（共享）
├── download_eagle.py                                                # EAGLE3 草稿模型下载脚本（推测解码扩展）
├── run_code_rl_demo.sh                                              # RL 训练脚本（基础样例）
├── run_multi_turn_livecodebench_eval.sh                            # 多轮工具调用 LiveCodeBench 测试脚本（基础样例）
├── run_toolcall_sft_demo.sh                                        # 多轮工具调用 SFT 训练脚本（基础样例）
├── eagle3_rl_run.sh                                                 # 含 EAGLE3 推测解码的 RL 训练脚本（推测解码扩展）
├── suffix_rl_run.sh                                                 # 含 Suffix 推测解码的 RL 训练脚本（推测解码扩展）
├── no_spec_rl_run.sh                                                # 不含推测解码的 RL 训练脚本——基线（推测解码扩展）
├── process_all_the_logs_sprl.py                                     # 日志处理与指标分析脚本（推测解码扩展）
└── README.md                                                        # 说明文档
```

---

## 第一部分：基础样例——含工具调用奖励的多轮工具调用 Code RL

### 环境准备

#### 构建 Docker 镜像

1. 构建支持 Code RL 的 verl 镜像，verl.Dockerfile 与 verl_sandbox.Dockerfile 请参见 `agent_rl/qwen2_code_rl` 样例中的文件：

```bash
docker build --network=host -f verl.Dockerfile -t verl:main-c651b7b-py311-cann8.3.RC1 .
```

2. 在 verl 镜像基础上，构建支持 ScaleBox 的镜像，实现 ScaleBox 和 verl 的融合部署。首先拉取 ScaleBox 代码：

```bash
git clone https://gitcode.com/icip-cas/ScaleBox
docker build --network=host -f verl_sandbox.Dockerfile -t verl_sandbox:main-c651b7b-py311-cann8.3.RC1 .
```

#### 构建适配 ScaleBox 的 verl 框架

1. 拉取 verl 代码并切换到指定版本：

```bash
git clone https://github.com/volcengine/verl
cd verl
git checkout c651b7b4207e408875f132c4226969ef3495d408
cd ..
```

2. 应用指定 patch。为了更好地执行 Code RL 训练任务，verl 框架需应用以下 patch，主要包含以下修改：

- 在 `prime reward manager` 中，增加对 `code_contests` 数据源的支持；
- 调整 `prime reward manager` 的并发进程数，从 64 降至 32，以避免沙盒资源竞争；
- 延长 `prime reward manager` 的任务超时时间，从 300s 延长至 3000s，以支持更大批量数据下的代码执行；
- 增强训练过程中的日志打印，便于调试；
- 支持 Coding 多轮工具调用的训练逻辑；
- 增加 Toolcall reward，增强训练稳定性。

遵循下面的指令应用对应 patch：

```bash
git apply patches/verl/0001-verl-feature-improve_rl_usability.patch
git apply patches/verl/0002-enable-tool-agent-loop.patch
git apply patches/verl/0003-toolcall-reward.patch
```

#### 部署 ScaleBox 服务

1. 启动 verl_sandbox 融合镜像：

```bash
docker run -it --privileged --name=start_verl_sandbox --user root --network host \
  --shm-size 500g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  verl_sandbox:main-c651b7b-py311-cann8.3.RC1 /bin/bash
```

2. 激活 ScaleBox 服务必备环境：

```bash
source /home/ma-user/miniconda3/bin/activate sandbox-base
```

3. 部署 ScaleBox 服务。以下提供针对 Code RL 单节点训练的部署命令，更多分布式部署功能见 [ScaleBox](https://gitcode.com/icip-cas/ScaleBox) 仓库：

```bash
export HOST=0.0.0.0           # 服务器主机地址
export PORT=8080              # 服务端口
export WORKERS=32             # Uvicorn 服务并行 Worker 数量
export MAX_MEM=50000000       # 单进程最大内存占用

cd ScaleBox
make run-online > deploy_${HOST}:${PORT}.log 2>&1 &
```

验证服务是否成功部署：

```bash
curl 'http://localhost:8080/run_code' \
  -H 'Content-Type: application/json' \
  --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
```

预期返回：

```
{"status":"Success","message":"","compile_result":null,"run_result":{"status":"Finished","execution_time":0.02984905242919922,"return_code":0,"stdout":"Hello, world!\n","stderr":""}
```

### 数据集准备

#### SFT 工具调用数据构建

基于 [Gen-Verse/Open-AgentRL-SFT-3K](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) 数据，过滤出其中包含多轮 Python 工具调用的 Coding 推理数据，并将其转换格式以符合后续 RL 训练：

```bash
python build_toolcall_sft_data.py
```

#### RL 数据构建

基于 [PrimeIntellect/verifiable-coding-problems](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems) 数据，过滤其中较高质量的 Python 代码数据部分，作为 RL 训练数据（verifiable-coding-problems-python-only）：

```bash
python build_rl_dataset.py
```

### 工具调用微调

1. 准备模型权重：

```bash
hf download Qwen/Qwen3-1.7B --local-dir Qwen/Qwen3-1.7B
```

2. 工具调用的 SFT 脚本为 `run_toolcall_sft_demo.sh`，可依情况对应更改默认模型权重和数据等路径：

```bash
source /home/ma-user/miniconda3/bin/activate base
mkdir -p log/sft_run_log
bash run_toolcall_sft_demo.sh
```

3. 选用 sft_step_50 的 ckpt，合并训练好的模型权重：

```bash
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoint/multiturn-toolcall-sft-qwen-3-1b/global_step_50 \
    --target_dir checkpoint/multiturn-toolcall-sft-qwen-3-1b/global_step_50/huggingface
```

### 强化学习训练

强化学习的训练脚本为 `run_code_rl_demo.sh`，可依情况对应更改默认模型权重和数据等路径：

```bash
bash run_code_rl_demo.sh
```

### 训练结果

下图为训练中各个指标，图一为模型在训练数据（无重复数据）上的得分，图二为推理长度及截断比例，图三为工具调用交互轮次。

<p align="center">
  <img src="figures/training_progress.png" width="800" />
</p>

### 模型评测

本实验基于 [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 数据集评测模型的代码生成能力，推理参数遵循 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 相关实验设置。

**评测数据相关参数：**

- release_version: v5
- start_date: 2024-08-01
- code_execution: [ScaleBox](https://gitcode.com/icip-cas/ScaleBox)

**推理相关参数：**

- n: 4
- temperature: 0.6
- top_p: 0.95
- max_tokens: 32768

| Steps | LiveCodeBench (Pass@1) |
|-------|------------------------|
| 20  | 16.03 |
| 40 | 16.74 |
| 60 | 18.08 |
| 80 | 18.63 |
| 100 | 19.19 |
| 120 | 20.14 |
| 140 | 21.34 |
| 160 | 24.45 |
| 180 | 26.20 |
| 200 | 25.97 |
| 220 | 26.36 |
| 240 | 28.39 |

<p align="center">
  <img src="figures/evaluation_progress.png" width="800" />
</p>

---

## 第二部分：推测解码扩展

本节介绍如何在基础样例之上启用 Suffix 与 EAGLE3 推测解码。该扩展使用更新版本的 vLLM，并在 Conda 环境下完成验证。注意，本扩展直接使用 HuggingFace 上的公开 Qwen3-1.7B 模型权重，无需基础样例中的 SFT 微调步骤。

我们的分析表明，Rollout 阶段占总 RL 训练步骤时间的 **78.3%**（Qwen3-1.7B 每步 3596.5s 中的 2816.6s）。推测解码通过加速 vLLM Rollout 阶段的 Token 生成，直接解决这一瓶颈，目标是在不损失精度的前提下实现 **≥25% 的端到端训练加速**。

> **说明：** `0001-verl-feature-improve_rl_usability.patch`、`0002-enable-tool-agent-loop.patch`、`build_dataset.py` 及 `scalebox.py` 均直接沿用自基础样例，未作任何修改。本节其余文件为推测解码扩展新增内容。第一部分中的 SFT 微调与工具调用奖励步骤无需应用于推测解码扩展。推测解码扩展直接使用 HuggingFace 上的公开 Qwen3-1.7B 模型权重。

### 环境准备

#### 1. 创建 Conda 环境

```bash
conda create -n verl-specrl python=3.11 -y
conda activate verl-specrl
source /path/to/CANN_8.5.0/ascend-toolkit/set_env.sh
source /path/to/CANN_8.5.0/nnal/atb/set_env.sh
```

#### 2. 安装 vLLM

```bash
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..
```

#### 3. 安装 vLLM-Ascend

```bash
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install decorator
python -m pip install -U pip setuptools wheel
python -m pip install -U cmake ninja pybind11
python -m pip install -U "setuptools-scm>=8"
pip install --no-cache-dir torch==2.8.0 torch-npu==2.8.0
pip install torchvision==0.23.0 --no-deps
pip install -e . --no-build-isolation --no-deps
# vllm-ascend commit id: 6281c1207a7a499e9f23a42b3a1e7027469f2b10
cd ..
```

#### 4. 安装 verl

```bash
git clone https://github.com/volcengine/verl
cd verl
git checkout c651b7b4207e408875f132c4226969ef3495d408
pip install -r requirements-npu.txt
pip install click==8.2.1
pip install git+https://github.com/ShaohonChen/PyExt.git@py311support
pip install -e .
cd ..
```

#### 5. 应用补丁

```bash
# verl 补丁——在 verl 目录下执行
git apply ../patches/verl/0001-verl-feature-improve_rl_usability.patch
git apply ../patches/verl/0002-enable-tool-agent-loop.patch
git apply ../patches/verl/0004-enable-specrl-clean.patch

# vLLM 补丁——在 vllm 目录下执行
cd /path/to/vllm
git apply /path/to/cann-recipes-train/agent_rl/qwen3_code_toolcall/patches/vllm/0001-enable-eagle-sprl.patch
cd ..
```

#### 6. 修复依赖

```bash
pip install numba
pip uninstall triton-ascend triton -y
pip install transformers==4.57.6
pip install setuptools==80.10.2
pip install decorator
pip install arctic-inference==0.1.1
```

#### 部署 ScaleBox 服务

```bash
conda create -n scalebox python=3.11 -y
conda activate scalebox
git clone https://github.com/icip-cas/ScaleBox.git
cd ScaleBox
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install databases
pip install aiosqlite
```

```bash
export HOST=0.0.0.0
export PORT=8080
export WORKERS=32
export MAX_MEM=50000000

cd ScaleBox
make run-online > deploy_${HOST}:${PORT}.log 2>&1 &
```

验证服务是否成功部署：

```bash
curl 'http://localhost:8080/run_code' \
  -H 'Content-Type: application/json' \
  --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
```

预期返回：

```json
{"status":"Success","message":"","compile_result":null,"run_result":{"status":"Finished","execution_time":0.02984905242919922,"return_code":0,"stdout":"Hello, world!\n","stderr":""}}
```

### 数据集准备

直接沿用基础样例——如第一部分所述运行 `python build_dataset.py`。

### 模型准备

下载目标模型与 EAGLE3 草稿模型权重：

```bash
python download_eagle.py
```

### 强化学习训练

运行前，请在对应脚本顶部设置以下路径。

`no_spec_rl_run.sh` 与 `suffix_rl_run.sh` 所需路径：

| 变量 | 说明 |
|---|---|
| `MODEL_PATH` | Qwen3-1.7B 目标模型权重路径 |
| `DATA_PATH` | RL 训练数据集路径（通过 `build_dataset.py` 生成） |
| `ASCEND_HOME_TOOLKIT` | CANN 工具包安装路径（如 `/path/to/CANN_8.5.0/`） |

`eagle3_rl_run.sh` 所需路径：

| 变量 | 说明 |
|---|---|
| `MODEL_PATH` | Qwen3-1.7B 目标模型权重路径 |
| `DRAFT_MODEL_PATH` | EAGLE3 草稿模型权重路径（通过 `download_eagle.py` 下载） |
| `DATA_PATH` | RL 训练数据集路径（通过 `build_dataset.py` 生成） |
| `ASCEND_HOME_TOOLKIT` | CANN 工具包安装路径（如 `/path/to/CANN_8.5.0/`） |

运行不含推测解码的基线 RL 训练：

```bash
bash no_spec_rl_run.sh
```

运行含 EAGLE3 推测解码的 RL 训练：

```bash
bash eagle3_rl_run.sh
```

运行含 Suffix 推测解码的 RL 训练：

```bash
bash suffix_rl_run.sh
```

### 训练日志处理

训练完成后，将实验相关的所有日志收集至同一文件夹，然后执行：

```bash
python process_all_the_logs_sprl.py <path/to/logs/> -o <path/to/output>/combined_metrics.csv
```

对含推测解码与不含推测解码的两次训练分别执行上述命令，以生成用于对比分析的 CSV 文件。

### 训练结果

与基线相比，Suffix 与 EAGLE3 推测解码端到端吞吐量最高提升 **38%**，训练步骤时间缩短 **25%**，且精度无损失。

<p align="center">
  <img src="figures/throughput_speedup.png" width="800" />
</p>

<p align="center">
  <img src="figures/sd_nosd_accuracy.png" width="800" />
</p>

由于 EAGLE3 草稿模型在 RL 训练期间保持冻结，随着 Actor 策略逐渐偏离草稿模型的训练分布，草稿接受率呈缓慢下降趋势：

<p align="center">
  <img src="figures/acceptance_rate_overall.png" width="800" />
</p>

---

## 推测解码未来工作

1. **Ngram推测解码修复**：修复了Ngram推测解码中的一个错误。
2. **块验证**：在推测解码的拒绝采样模块中启用块验证。
3. **在线草稿模型训练**：探索在 RL 训练过程中对 EAGLE3 草稿模型进行联合训练，以缓解因策略漂移导致的接受率逐步下降问题。
4. **弹性推测解码**：探索在 RL 训练过程中自适应调整推测解码参数（如推测 Token 数量）。
5. **推测解码独立目录**：随着块验证、在线草稿模型训练、弹性推测解码、在线 MTP 等 SD 专属特性的逐步成熟，我们将重新评估是否有必要为推测解码样例建立独立目录。
