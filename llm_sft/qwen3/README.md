# Qwen3-1.7B SFT训练示例

## 硬件要求
卡数：1张A3

## 基于Dockerfile构建环境

1. 创建镜像。
   ```bash
   # 请预先下载本样例提供的Dockerfile文件：Dockerfile.cann.mindspeed.qwen3
   # 随后基于该Dockerfile创建docker image。请传入镜像名称，例如 qwen3_sft
   docker build -t qwen3_sft -f Dockerfile.cann.mindspeed.qwen3 .

   # 创建并进入镜像
   # 请设置容器名称，例如 qwen3_sft，镜像名称同上一步
   # container_name=qwen3_sft
   # image_name=qwen3_sft:latest
   # 默认docker创建为单机配置（16卡），可按照实际环境进行修改选择对应的卡数 --device=/dev/davinci0
   bash build_docker.sh
   ```

2. 源码准备。
   ```bash
   # 下载本样例所在代码仓，以master分支为例
   git clone https://gitcode.com/cann/cann-recipes-train.git

   # 添加镜像中已经准备好的依赖文件
   cd ./cann-recipes-train/llm_sft/qwen3/
   cp -r /workspace/MindSpeed-LLM MindSpeed-LLM
   cp -r qwen3_dense MindSpeed-LLM/examples/mcore

   cd MindSpeed-LLM
   ```
## 一站式平台快速启动SFT训练示例

### 环境要求
一站式平台镜像选择：CANN-8.5.0-A3 或者 CANN-8.5.0-A2

### 项目及依赖构建
在当前目录下（cann-recipes-train/llm_sft/qwen3）执行:
```bash
bash build_project_platform.sh

# 进入对应目录
cd MindSpeed-LLM
```

## 数据集准备
本样例使用的alpaca数据集准备方法如下：
```bash
# 在本样例目录下创建dataset数据集目录
mkdir -p ./dataset
cd dataset/
# 从魔塔社区下载alpaca数据集的parquet文件
pip install modelscope

modelscope download --dataset OmniData/alpaca --local_dir ./alpaca
cd ../
```

## 模型权重准备
本样例使用的Qwen3-1.7B模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放在当前目录的 ./Qwen3-1.7B 目录下
mkdir ./Qwen3-1.7B
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./Qwen3-1.7B
```

## Qwen3-1.7B
1. 数据转换，将原始数据转为训练数据格式：

   ```bash
   bash examples/mcore/qwen3_dense/data_convert_qwen3_instruction.sh
   ```

2. 模型转换，将HuggingFace格式的模型转换为Mcore训练格式：
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_hf2mcore.sh
   ```

3. SFT训练启动脚本：
   ```bash
   bash examples/mcore/qwen3_dense/tune_qwen3_4K_full_A3_ptd.sh
   ```

4. 训练后模型转换回huggingface格式
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_mcore2hf.sh
   ```

注：需要按照实际情况在以上脚本中修改 ascend-toolkit 路径，默认配置为：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
一站式平台需要修改为（以CANN 8.5.0为例）：
```
source /home/developer/Ascend/cann-8.5.0/set_env.sh
```

## 附录 相关参数说明
1. 数据准备:

   | 参数 | 说明 |
   |------|------|
   | `--input` | 输入数据文件路径（可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持.parquet/.csv/.json/.jsonl/.txt/.arrow格式， 同一个文件夹下的数据格式需要保持一致） |
   | `--tokenizer-name-or-path` | tokenizer文件路径|
   | `--output-prefix` | 输出数据文件前缀 |
   | `--handler-name` | 微调数据预处理Alpaca风格数据集时，应指定为AlpacaStyleInstructionHandler，根据--map-keys参数提取对应数据的列。|
   | `--enable-thinking` | 是否添加推理tag，例如<think> </think>|
   | `--seq-length` | 序列长度，默认4096 |
   | `--prompt-type` | 指定模型模板，默认为qwen3 |

alpaca风格数据示例：
```
[
   {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "system": "系统提示词（选填）",
      "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
      ]
   }
]
```

2. 模型转换`hf2mcore`相关参数说明：

   | 参数 | 说明 |
   |------|------|
   | `--target-tensor-parallel-size` | 张量并行 |
   | `--target-pipeline-parallel-size` | 流水线并行 |
   | `--load-dir` | 输入模型路径 |
   | `--save-dir` | 输出模型路径 |
   | `--moe-grouped-gemm` | 启用MoE grouped GEMM优化 |
   | `--model-type-hf` | HuggingFace模型类型 |
   | `--expert-tensor-parallel-size` | 专家张量并行大小，需要显式指定为1 |

3. SFT训练：

- 路径配置

   | 参数 | 说明 |
   |------|------|
   | `CKPT_LOAD_DIR` | 加载预训练权重的路径 |
   | `CKPT_SAVE_DIR` | 保存微调模型的路径 |
   | `DATA_PATH` | 训练数据集路径 |
   | `TOKENIZER_PATH` | Tokenizer文件路径 |

- 并行配置，需要和模型转换的参数**保持一致**

   | 参数 | 说明 | 默认值 |
   |------|------|--------|
   | `TP` | 张量并行大小 | 1 |
   | `PP` | 流水线并行大小 | 1 |
   | `CP` | 上下文并行大小 | 1 |
   | `CP_TYPE` | 上下文并行算法 | ulysses_cp_algo |

- 训练配置

   | 参数 | 说明 | 默认值 |
   |------|------|--------|
   | `SEQ_LENGTH` | 序列长度 | 4096 |
   | `TRAIN_ITERS` | 训练迭代次数 | 2000 |
   | `--micro-batch-size` | 单卡batch size | 1 |
   | `--global-batch-size` | 全局batch size | 32 |
   | `--lr` | 学习率 | 1.25e-6 |
   | `--lr-decay-style` | 学习率衰减方式 | cosine |
   | `--min-lr` | 最小学习率 | 1.25e-7 |
   | `--weight-decay` | 权重衰减 | 0.1 |
   | `--lr-warmup-fraction` | 预热比例 | 0.01 |
   | `--clip-grad` | 梯度裁剪 | 1.0 |

- 微调参数

   | 参数 | 说明 |
   |------|------|
   | `--finetune` | 启用微调模式 |
   | `--stage` | 训练阶段，sft表示监督微调 |
   | `--is-instruction-dataset` | 使用指令数据集 |
   | `--prompt-type` | 使用qwen3格式的prompt |

4. 模型转换 `mcore2hf`:

   | 参数 | 说明 |
   |------|------|
   | `--load-dir` | 输入模型路径，训练保存的模型路径 |
   | `--save-dir` | 输出模型路径 |
   | `--model-type-hf` | HuggingFace模型类型 |