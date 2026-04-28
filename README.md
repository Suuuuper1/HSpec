# cann-recipes-train

## 🚀 Latest News
- [2026/04] 新增DeepSeek-V3模型[MXFP8/HiF8 低精度预训练](llm_pretrain/DeepSeekV3/README.md)样例。
- [2026/04] 新增[DeepSeek-V4-Flash模型续训练0day支持](llm_pretrain/deepseekv4/README.md)（基于torchtitan框架）样例，支持**训练入图、AutoFuse**特性。
- [2026/02] 新增DeepSeek-V3.2模型[torchtitan框架预训练](llm_pretrain/deepseekv32/README.md)样例。
- [2026/02] 新增Qwen3系列模型RL训练使能[npugraph_ex图模式](llm_rl/qwen3/README.md)样例。
- [2025/12] 新增Qwen2.5/Qwen3模型Code RL长上下文代码生成强化学习样例。
- [2025/12] 新增Qwen3系列模型RL训练使能[SAM投机推理](llm_rl/qwen3/README.md)、[tool agent RL](agent_rl/qwen3_tool_agent/README.md)样例。
- [2025/11] [Qwen3模型长序列RL](llm_rl/qwen3/README.md)样例首次上线。
- [2025/10] [DeepSeek-R1](llm_rl/deepseek/README.md)、[Qwen2.5模型](llm_rl/qwen2_5/verl_npu_demo/README.md)样例首次上线。

## 🎉 概述
cann-recipes-train仓库旨在针对LLM与多模态模型训练业务中的典型模型、算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型训练。


## ✨ 实践列表

|实践|简介|
|-----|-----|
|[DeepSeek-R1 RL训练优化样例](llm_rl/deepseek/README.md) |基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群实现GRPO算法的高吞吐RL训练，并达到120TPS/卡的系统吞吐量。|
|[基于verl框架的Qwen2.5强化学习（入门样例）](llm_rl/qwen2_5/verl_npu_demo/README.md) |基于Qwen2.5-1.5B-Instruct模型，采用verl强化学习框架，在MATH-lighteval数学推理数据集上进行了训练。本样例只需要单卡Atlas A2环境，帮助大家快速上手，使用昇腾NPU完成RL训练任务。|
|[Qwen3-235B-A22B RL训练优化样例](llm_rl/qwen3/README.md) | 基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群实现GRPO/DAPO算法的**长序列 2k+32k**训练，GRPO达到120TPS/卡的系统吞吐量。|
|[Qwen3-32B RL训练使能SAM投机推理样例](llm_rl/qwen3/README.md) | 基于开源veRL框架，搭配MindSpeed+vLLM-Ascend框架，在Atlas A3集群，GRPO/DAPO算法的2k+32k训练场景下，使能**SAM投机推理特性**，达成**10%性能提升**。|
|[Qwen3 tool agent RL训练样例](agent_rl/qwen3_tool_agent/README.md) |基于verl/recipe中的retool项目，调用Sandbox工具，使能`asyncLLM`和`agent_loop`特性，在昇腾NPU上完成端到端agent RL训练任务。|
|[基于ScaleBox沙盒的Code RL训练样例](agent_rl/qwen2_code_rl/README.md) |基于verl框架和ScaleBox代码沙盒，支持长上下文(2k+16k) Code RL训练，Qwen3-30B-A3B在LiveCodeBench上Pass@1从46.59提升至56.27。|
|[DeepSeek-V3.2 Pretrain训练样例](llm_pretrain/deepseekv32/README.md) |基于torchtitan，在64卡Atlas A3集群上完成DeepSeek-V3.2模型32K长序列预训练复现。|
|[DeepSeek-V4-Flash 续训练样例](llm_pretrain/deepseekv4/README.md) |基于torchtitan + autufuse，使能**极简切分和训练入图**，在Atlas A3 64卡集群支持DeepSeek-V4-Flash-285B模型的续训练，吞吐达成**1100tokens/p/s**。|
|[DeepSeek-V3 MXFP8/HiF8 低精度预训练样例](llm_pretrain/DeepSeekV3/README.md) |基于MindSpeed，在 8 卡 Atlas A5 环境上完成 DeepSeek-V3 裁剪模型8k序列预训练复现。|

## 🏃 一站式平台快速体验
「一站式平台」是为开发者提供的 NPU 环境，内部已集成完整的 CANN 环境，可以直接使用。

cann-recipes-train 针对该平台在相应样例 README 中提供了简化的「快速启动」路径，帮助用户最小步骤完成 NPU 模型训练体验。当前支持的模型正在持续扩展中，敬请关注：

|实践|简介|
|-----|-----|
|[Qwen3-1.7B SFT训练样例](llm_sft/qwen3/README.md#一站式平台快速启动sft训练示例) |在一站式平台Atlas A2/A3环境中完成Qwen3-1.7B 单卡SFT训练。|

## 💡 特性介绍
本项目在探索最佳实践的过程中引入了如下特性：

|特性|介绍|
|----|---|
|SAM无损投机推理 |[docs/features/sam_speculative_decoding.md](docs/features/sam_speculative_decoding.md)|
|RL On-Policy 推理场景的序列级均衡调度引擎| [docs/features/rollout_rebalance.md](docs/features/rollout_rebalance.md)|

## 📖 目录结构说明

```
├── docs                         # 优化技术介绍文档
├── llm_rl                       # llm强化学习训练相关代码
│  ├── deepseek                  # deepseek强化学习训练相关代码
│  ├── qwen2_5                   # Qwen2.5强化学习训练相关代码
│  ├── qwen3                     # Qwen3强化学习训练相关代码
│  └── ...
├── agent_rl                     # agent强化学习训练相关代码
│  ├── qwen3_tool_agent          # Qwen3 tool agent RL训练
│  ├── qwen2_code_rl             # 基于ScaleBox沙盒的Code RL训练
│  └── ...
├── multimodal_rl                # 多模态强化学习训练相关代码
├── llm_sft                      # llm有监督微调训练相关代码
├── llm_pretrain                 # llm预训练相关代码
├── CONTRIBUTION.md
├── README.md
└── ...
```

## 🤖 智能代码助手
本仓已集成 Zread 代码仓库智能体，旨在通过 AI 技术为您提供更深度的代码理解与技术支持。

点击 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/cann-recipes-train) 徽章，进入其专属页面，开启在线智能代码学习与知识问答体验！

> ⚠️ 说明：
当前代码仓库智能体服务处于试点阶段。在使用过程中，如果您发现 AI 生成的内容存在准确性问题，或对智能助手的功能有任何改进建议，欢迎通过 Issues 与我们交流，您的反馈对我们非常重要！


## 📝 相关信息

- [贡献指南](./CONTRIBUTION.md)
- 许可证

    cann-recipes-train仓涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应许可证文本可查阅[LICENSE](./LICENSE)
- [免责声明](DISCLAIMER.md)
