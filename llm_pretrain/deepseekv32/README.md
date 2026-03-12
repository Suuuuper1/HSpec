# 基于TorchTitan的DeepSeek-V3.2 32K长序列预训练优化实践样例

## 概述

本样例针对DeepSeek-v3.2模型，基于[TorchTitan开源框架](https://github.com/pytorch/torchtitan)，在64卡Atlas A3集群上完成完成32K长序列预训练全流程的优化适配。优化点介绍可参见[CANN+TorchTitan: DeepSeek-V3.2 32k长序列预训练昇腾优化实践](../../docs/llm_pretrain/deepseekv32_pre_train_optimization.md)。

## 硬件要求
产品型号：Atlas A3 系列

最少卡数：64张A3

## 目录结构说明
```bash
├─torchtitan_npu              # torchtitan npu 适配目录
│  ├─config                   # 一些配置文件目录
│  ├─converter                # patch适配目录
│  │  └─features              
│  │  └─kernels               
│  ├─distributed              # 分布式并行特性目录
│  │  └─context_parallel      
│  ├─models                   # 模型适配目录 
│  │  └─deepseek_v32          # deepseekv32模型核心组件
│  │      └─infra             # 分布式策略目录
│  │      └─model             # 模型目录
│  │      └─train_configs     # 训练参数
│  ├─patches                  # torch_npu和融合算子patch目录
│  │  ├─distributed           
│  │  └─optimizer                 
│  │  └─tools                 
│  │  └─torch                 
│  │  └─torch_npu             
│  ├─tools                    # 工具函数目录
├─utils                       # 权重转换工具脚本路径
```

## 基于Dockerfile构建环境
> 环境搭建可以基于Dockerfile快速实现，Dockerfile里已配置了必要的昇腾软件和其他第三方软件的依赖。

1. 基于Dockerfile创建镜像。

```bash
# 请预先下载本样例提供的Dockerfile文件：Dockerfile.torchtitan.deepseekV32
# 随后基于该Dockerfile创建docker image。请传入镜像名称，例如 your_image_name
docker build -t your_image_name -f Dockerfile.torchtitan.deepseekV32 .

# 请设置容器名称，例如 your_docker_name，镜像名称同上一步
container_name=your_docker_name
image_name=your_image_name:latest

# 执行docker run命令创建容器，可通过-v按需挂载宿主机目录至容器
docker run -itd \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /data/:/data/ \
-v /home/:/home/ \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /dev/shm:/dev/shm \
--net=host \
--name ${container_name} \
--privileged ${image_name} /bin/bash

# 执行docker exec命令进入容器
docker exec -it -u root ${container_name} bash
```

2. 源码准备。

```bash
# 下载本样例所在代码仓，以master分支为例
git clone https://gitcode.com/cann/cann-recipes-train.git
# 进入示例所在路径
cd cann-recipes-train/llm_pretrain/deepseekv32/
# 安装相关依赖
pip install -r requirements.txt
# 安装torchtitan-npu
pip install -e ./
```

## 数据集准备

首先创建数据集路径

```
mkdir -p ./tests/assets/enwiki
```

下载[enwiki 的parquet数据](https://huggingface.co/datasets/answerdotai/enwiki)到`./tests/assets/enwiki`路径下面

可以使用下面的命令下载数据集

```bash
cd ./tests/assets/
git clone https://huggingface.co/datasets/answerdotai/enwiki
cd ../..
```

然后使用parquet转换脚本将其转换成json格式

```bash
python ./utils/parquet2json.py --input ./tests/assets/enwiki --output ./tests/assets/enwiki/data.json
```

## 模型权重准备
本样例使用的DeepSeek-V3.2模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放至样例的./assets/hf/DeepSeek-V3.2目录下（不加载权重实验也需要执行这步操作）
mkdir -p /data/models/DeepSeek-V3.2
pip install modelscope

# 下载DeepSeek-V3.2完整FP8权重至指定目录，例如 /data/models/DeepSeek-V3.2（此步骤需要目录所在磁盘有650GB以上空间）
modelscope download --model deepseek-ai/DeepSeek-V3.2 --local_dir /data/models/DeepSeek-V3.2
```

在各个节点上使用`weight_convert.sh` 脚本完成FP8到Bfloat16权重转换。

  >入参介绍：`input_fp8_hf_path`：原始fp8权重路径；`output_hf_path`：转换后输出的权重路径；`quant_mode`：量化模式

如果权重转换的运行环境为NPU，需要先执行：

```shell
cann_path=/usr/local/Ascend/ascend-toolkit/latest  # cann包安装路径
source ${cann_path}/bin/setenv.bash
```

权重转换拉起示例：

```shell
# 转换为Bfloat16权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2 --output_hf_path /data/models/DeepSeek-V3.2-bf16 --quant_mode bfloat16
```

## pretrain执行

```shell
# 示例执行，2层（1moe，1dense）裁剪模型（请提前配置toml文件中相关地址）
chmod +x ./run_pretrain.sh
NGPU=16 CONFIG_FILE="./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml" ./run_pretrain.sh --compile.enable
# 64卡671B参数全量预训练拉起脚本（要在8机上同时拉起），注意要修改run_pretrain_multinodes.sh脚本里面的IPs变量
chmod +x ./run_pretrain_multinodes.sh
CONFIG_FILE="./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_128die.toml" ./run_pretrain_multinodes.sh --compile.enable
```
