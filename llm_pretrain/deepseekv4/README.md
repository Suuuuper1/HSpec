# 基于TorchTitan-NPU的DeepSeek-V4-Flash训练部署指导

## 概述

本文面向 `torchtitan-npu` 仓中的 `DeepSeek-V4-Flash` 模型训练场景，介绍源码获取、数据准备、模型权重准备与 BF16 转换、镜像准备、Docker 容器拉起、训练配置说明以及多机训练启动方式。

本文的优化方案介绍和性能Benchmark可参见技术报告[DeepSeek-V4昇腾训练支持：基于CANN平台的TorchTitan-NPU + AutoFuse 极简训练优化实践](../../docs/llm_pretrain/deepseek-v4_torchtitan_npu_autofuse.md)。

## 硬件与软件要求

| 项目 | 要求 |
| --- | --- |
| 产品型号 | Atlas A3 系列 |
| 最小卡数要求 | 8机 64卡 A3 |
| 操作系统 | Linux ARM |
| 驱动版本 | Ascend HDK 25.5.1 |
| 镜像版本 | [dsv4_train_torchtitan:cann9.0.0.beta.1_v1.0](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/deepseek_train/dsv4_train_torchtitan_v1.0.tar.gz) |

> [!NOTE]
> 需使用npu-smi info 检查Ascend NPU固件和驱动正确安装，且版本为`25.5.1`。如果未安装或者版本不一致，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)，并根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。


## 源码准备

执行如下命令拉取源码：

```shell
mkdir -p /home/code && cd /home/code/
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
```

本次训练使用的配置文件为：

```shell
./torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_43layers_4k_128die.toml
```

## 获取Docker镜像

从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/deepseek_train/dsv4_train_torchtitan_v1.0.tar.gz)中下载 docker 镜像，然后上传到A3服务器的每个节点上，执行如下命令加载训练镜像：

```shell
gunzip -c dsv4_train_torchtitan*.gz | docker load
```

加载后镜像版本如下：

```shell
dsv4_train_torchtitan:cann9.0.0.beta.1_v1.0
```

## 启动Docker容器

执行如下命令启动容器：

```shell
docker run -u root -itd --name dsv4_train_torchtitan_v1.0 --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home:/home \
    -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host \
    --shm-size=128g \
    --privileged \
    dsv4_train_torchtitan:cann9.0.0.beta.1_v1.0 /bin/bash
```

进入容器：

```shell
docker exec -it dsv4_train_torchtitan_v1.0 /bin/bash
```

在容器内执行环境变量初始化：

```shell
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/cann/opp/vendors/custom_transformer/bin/set_env.bash
```

## 数据集准备

该配置默认使用仓内样例数据集：

```shell
dataset = "c4_test"
dataset_path = "./tests/assets/c4_test"
```

若直接按默认配置拉起训练，可使用仓内已有的 `c4_test` 数据集；若使用自定义数据集，请提前准备好数据目录，并在后续训练配置中修改 `dataset` 和 `dataset_path`。

## 模型权重准备

本样例使用 `DeepSeek-V4-Flash` 权重进行CPT训练，建议先从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 下载原始权重和配置文件到统一目录，例如：

```shell
mkdir -p /data/models/DeepSeek-V4-Flash
```

将原始权重转换为 BF16 权重，输出目录与训练配置文件保持一致，例如：

```shell
mkdir -p /data/models/DeepSeek-V4-Flash-bf16
```

权重转换使用 `cann-recipes-train` 仓中的 `convert_model.py` 脚本。示例如下：

```shell
cd /home/code/
git clone https://gitcode.com/cann/cann-recipes-train.git
cd cann-recipes-train/llm_pretrain/deepseekv4/utils
python3 convert_model.py \
  --input_fp8_hf_path /data/models/DeepSeek-V4-Flash \
  --output_hf_path /data/models/DeepSeek-V4-Flash-bf16 \
  --quant_type bfloat16
```

## 训练配置

本次训练使用配置文件`./torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_43layers_4k_128die.toml
`

拉起训练前请重点确认以下路径配置与实际环境一致：

```toml
[model]
hf_assets_path = "/data/models/DeepSeek-V4-Flash-bf16"

[training]
dataset = "c4_test"
dataset_path = "./tests/assets/c4_test"

[checkpoint]
initial_load_in_hf = true
initial_load_path = "/data/models/DeepSeek-V4-Flash-bf16"
```

## 启动训练

* 根据使用实际的网卡、节点 IP等，修改多机训练脚本配置，参考 `torchtitan-npu` [快速上手](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md) 文档中的“[多机训练任务](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md#多机训练任务)”一节。

* 进入各节点上的 `torchtitan-npu` 源码目录后，在所有参与训练的节点上同时执行如下命令，即可启动 `DeepSeek-V4-Flash` 多机CPT训练任务：

```shell
CONFIG_FILE=./torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_43layers_4k_128die.toml \
bash scripts/run_train_multinodes.sh
```
