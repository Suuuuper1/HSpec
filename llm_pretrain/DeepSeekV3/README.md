# DeepSeek-V3 MXFP8/HiF8 低精度预训练优化实践样例

## 概述

本样例针对DeepSeek-V3 裁剪模型，基于[MindSpeed 框架](https://gitcode.com/Ascend/MindSpeed)，在 8 卡 Atlas A5 上完成完成8K序列MXFP8/HiF8 低精度预训练优。MXFP8/HiF8 低精度预训练介绍可参见[HiF8精度与性能双优：面向大模型训练的低精度优化实践](../../docs/llm_pretrain/deepseev3_pre_train_hifp8_mxfp8.md)。

## 硬件要求
产品型号：Atlas A5 950DT 系列

最少卡数：8 张 A5

## 构建环境

1. 手动安装相关依赖。

[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/preparing_installation.md)
```bash
# 下载并安装PyTorch框架
wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_aarch64.whl

# 下载并安装torch_npu插件
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl
#
```

2. 源码准备。

```bash
# 请根据实际路径进行替换，当前支持A5的商发cann包暂时未发布，发布后安装即可
source /usr/local/Ascend/cann/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh

# 创建代码目录工程
cd /home
mkdir train_code
cd train_code

# 下载 MindSpeed
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout master  # checkout commit from MindSpeed master
pip3 install -r requirements.txt 
pip3 install -e .
cd ..

# 下载 MindSpeed-LLM
git clone https://gitcode.com/ascend/MindSpeed-LLM.git
# 从github下载 Megatron-LM，请确保网络能访问
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout master
mkdir logs
pip3 install -r requirements.txt  # 安装其余依赖库

# 下载 cann-recipes仓对应的脚本
cd ../
git clone https://gitcode.com/cann/cann-recipes-train.git
cp ./cann-recipes-train/llm_pretrain/DeepSeekV3/run_pretrain_dsk3_A5_8P_hif8.sh ./MindSpeed-LLM
cp ./cann-recipes-train/llm_pretrain/DeepSeekV3/run_pretrain_dsk3_A5_8P_mxfp8.sh ./MindSpeed-LLM
```

## 数据集准备
首先创建数据集路径

```bash
mkdir -p ./tests/assets/enwiki
```

下载[enwiki 的parquet数据](https://huggingface.co/datasets/answerdotai/enwiki)到`./tests/assets/enwiki`路径下面

可以使用下面的命令下载数据集

```bash
cd ./tests/assets/
git clone https://huggingface.co/datasets/lsb/enwiki20230101/tree/main/data
cd ../..
```

数据集转换示例，可以参考修脚本[data_convert_deepseek3_pretrain.sh](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek3/data_convert_deepseek3_pretrain.sh)修改原始数据集路径，模型配置文件路径和目标输出路径
```bash
cd ./MindSpeed-LLM
bash examples/mcore/deepseek3/data_convert_deepseek3_pretrain.sh
```


## 模型权重准备
本样例使用的 DeepSeek-V3 模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放至样例的 ./assets/hf/DeepSeek-V3 目录下（不加载权重实验也需要执行这步操作）
mkdir -p /data/models/DeepSeek-V3
pip install modelscope

# 下载DeepSeek-V3完整模型文件 （但是不包括权重，当前是裁剪模型可以不下载权重）
modelscope download --model deepseek-ai/DeepSeek-V3 --local_dir /data/models/DeepSeek-V3
```
权重转换拉起示例，可以参考[MindSpeed-LLM中转换脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh)修改原始权重路径，以及保持的路径以及对应的切分裁剪策略
```shell
# 转换为mcore权重
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```

## pretrain执行

```shell
cd /home/train_code/MindSpeed-LLM
export GLOO_SOCKET_IFNAME=eth0
export HCCL_HOST_SOCKET_PORT_RANGE=auto
# 请根据实际路径进行替换，当前支持A5的商发cann包暂时未发布，发布后安装即可
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_TOPO_FILE_PATH=/etc/superpod_1d_noroce.json
export HCCL_CONNECT_TIMEOUT=200
export HCCL_EXEC_TIMEOUT=200

# 示例执行 MXPF8 量化训练，2层（1moe，1dense）裁剪模型
bash ./run_pretrain_dsk3_A5_8P_mxfp8.sh

# 示例执行 HiF8 量化训练，2层（1moe，1dense）裁剪模型
bash ./run_pretrain_dsk3_A5_8P_hif8.sh
```
