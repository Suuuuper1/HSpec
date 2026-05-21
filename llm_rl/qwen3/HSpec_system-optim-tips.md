**当前问题图景**

当前链路大致是：

`NPU sample_hidden_states -> CPU pinned -> hspec_flush_and_get_all -> rollout_hidden_states 放入 DataProto.non_tensor_batch -> Ray 返回 trainer -> trainer 聚合 prompt_build_data -> Ray 发给 HSpecTableGroup actor -> actor 里 np.float32 / concatenate / SVD / build table -> epoch swap`

关键代码位置包括 [hspec_utils.py](C:/Users/HP/Desktop/workspace/HSpec_new/llm_rl/qwen3/vllm_ascend/spec_decode/hspec_utils.py:1264)、[vllm_rollout_spmd.py](C:/Users/HP/Desktop/workspace/HSpec_new/llm_rl/qwen3/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:647)、[ray_trainer.py](C:/Users/HP/Desktop/workspace/HSpec_new/llm_rl/qwen3/verl/trainer/ppo/ray_trainer.py:1369)、[hspec_table.py](C:/Users/HP/Desktop/workspace/HSpec_new/llm_rl/qwen3/vllm_ascend/spec_decode/hspec_table.py:254)。

不优点有四类：

1. **Ray 大对象路径错误**：hidden states 是 GB 级流数据，不适合放进 Ray object store。DataProto 序列化、trainer 聚合、actor 参数传递都会形成副本和 Plasma 压力。

2. **生命周期过长**：`batch` 在 HSpec 建表后继续用于 old logprob、ref、actor update，`rollout_hidden_states` 仍在 `non_tensor_batch` 中，训练 worker 不需要这些数据却会被 Ray 携带。

3. **PCA 实现内存形态差**：当前 actor 内 `astype(float32)`、`concatenate`、`np.linalg.svd` 会把 `(N,D)` 矩阵和临时 workspace 放大。对 30B 长 rollout，单 prompt 多 rollout 就可能是数百 MB 级临时内存，多个 actor 并发会叠加。

4. **5 个 HSpecTableGroup actor 是任意系统切分**：它既承担大对象接收，又承担 PCA，又承担 table 存储，还由 Ray actor heap 保存双缓冲。这种职责混合会把 Ray 调度、Plasma、Python GC、BLAS 线程和内存峰值耦合在一起。

**推荐架构**

把 HSpec 分成四层，而不是一个 Ray actor 大包办：

1. `HSpecCollector`，运行在每个 vLLM rollout worker 本地。  
   负责从 NPU 拿 anchor hidden states，写入本机 pinned buffer 或 mmap，不进入 DataProto。

2. `HSpecBuildShard`，本机 CPU 进程或受控 Ray actor，但只接收小 descriptor。  
   descriptor 只包含 `prompt_id, req_id, token_file_offset, hs_file_offset, length, reward, epoch`。大矩阵通过 mmap 文件或 shared memory 读取。

3. `HSpecTableStore`，本机 read-only mmap table store。  
   active table 按 epoch/version 存为连续数组：`mean/components/keys/token_buffer/offsets/rollout_idx/index`。swap 只更新版本元数据或原子重命名目录。

4. `HSpecCoordinator`，可以继续用 Ray。  
   只做小元数据协调：epoch barrier、build 完成状态、metrics、active version 广播。不要传 hidden states，不要传 table 大数组。

这样数据流变成：

`NPU -> local pinned/mmap raw H -> local/build shard PCA+projection -> mmap compressed table -> rollout worker mmap read/cache`

Ray 只传小对象，Raylet/Plasma 不再是 hidden-state 数据面。

**Hidden State 收集设计**

收集阶段建议保持 full hidden state，但改变承载方式：

- 每个 rollout worker 维护 append-only mmap 文件，例如 `epoch_E/worker_W/hs.bin` 和 `tokens.bin`。
- `hspec_submit_accumulate_task` 不再最终返回大 ndarray 给 rollout 层，而是直接写入本地段，并登记 descriptor。
- 使用固定大小 pinned host buffer pool，避免每步 `torch.empty(pin_memory=True)` 频繁分配和内存碎片。
- `rollout_hidden_states` 不进入 `DataProto.non_tensor_batch`。如短期保留旧路径，至少在 `build_tables_async` 后立刻从 batch 中 `pop` 掉，避免传给 actor/ref/update_actor。
- validation 阶段默认不收集 hidden states。`validate=True` 时可以使用 HSpec decode 查询，但不写 building table，不打包 hidden states。

这一步通常就能消除 Raylet/Plasma 崩溃。

**PCA 建表设计**

保留 prompt 级 PCA 语义，但不要再对完整 `(N,D)` 做一次性 SVD。

优先方案是流式 tiled PCA：

- 原始 H 以 fp16/bf16 mmap 保存。
- 对每个 prompt 先流式计算 `mean`。
- 再做分块 PCA。要求接近精确 PCA，用 tiled covariance：
  `G = Σ H^T H`，`C = G/N - μμ^T`，然后对 `D x D` 做 top-K eig。
- 关注速度，使用 randomized PCA：
  维护 `CΩ = Σ H^T(HΩ)`，`Ω` 为固定随机矩阵，`r=K+16/32`，再在小子空间求 top-K。它仍然是 PCA 近似，但计算和内存比 full SVD 小一个量级。
- 得到 `μ,W` 后，再二次扫描 mmap，分块计算 `Z=(H-μ)W^T`，写入 table keys。raw H 完成后即可删除。

这比当前 `np.concatenate + np.linalg.svd` 更符合系统实现：内存峰值从 `O(ND)` 多副本变为 `O(tile*D + D*K 或 D^2)`，且可以连续流式调度。

**CPU/NPU 调度**

不要简单把 PCA 异步丢到当前 NPU，因为 actor forward/update 也是关键路径 kernel。

推荐策略：

- 默认用 CPU build shard 做 PCA/投影，依赖 MKL/OpenBLAS，限制每个 shard 的线程数，例如每 shard 8 到 16 线程，shard 数按 NUMA 或 rollout TP group 设置。
- NPU 只作为可选加速器，用于 projection 或 randomized PCA 的大 GEMM，而且必须有预算控制。
- 建表任务需要 backpressure：当 host memory、pinned memory、NPU queue 任一超过阈值，暂停收集或延后 build，而不是继续向 Ray/ObjectStore 写大对象。

**HSpecTableGroup 重构**

当前 `_num_groups=5` 不建议保留为数据面分片。更合理的分片依据是：

- 单机 16 卡、`INFER_TP=4` 时，rollout 逻辑上有 4 个 TP group，可设 4 个 build/table shard。
- 每个 shard 明确资源：CPU core、内存预算、BLAS 线程数、mmap 目录。
- Ray actor 只做控制面，不能接收 `hidden_states_list` 这种大对象。

active table 也建议 mmap 化。在线 proposer prefetch 时不要从 Ray actor 拉大 dict，而是拿到 `{version, prompt_id, offset, shape}` 后本地 mmap 读取。worker 本地 cache 继续保留，但数据源不再经过 Ray 序列化。

**训练循环调整**

当前每 step 都 `ray.get(ray_hspec_tasks)`，导致 `timing_s/hspec_build_wait` 直接进入 step 延迟。更好的方式：

- step 内只提交 descriptor，不等待完整 build。
- build shard 在 actor update、reward、ref 等阶段后台运行。
- epoch 末尾 swap 前只等待“本 epoch 尚未完成的 prompt build”。

这个策略保持 epoch 级 table swap 语义，但把等待从每 step 的硬阻塞改成 epoch 末尾的短 barrier 或部分可用。
