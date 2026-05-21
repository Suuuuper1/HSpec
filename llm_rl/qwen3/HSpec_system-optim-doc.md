# HSpec 系统优化重构设计指导书

本文档面向当前 `vllm + vllm-ascend + verl` 强化训练框架中的 HSpec 系统重构。目标不是单点优化某个函数，而是把 HSpec 的数据面、控制面、建表计算、在线查询热路径分离，使 hidden state 收集和 PCA 建表不再挤占 Ray object store、trainer 主进程内存、NPU kernel 调度和训练 step 关键路径。

当前代码路径如下：

| 子系统 | 当前关键代码 |
| --- | --- |
| vLLM-Ascend hidden state 收集 | `vllm_ascend/spec_decode/hspec_utils.py`：`hspec_submit_accumulate_task`、`hspec_flush_and_get_all`；`vllm_ascend/worker/model_runner_v1.py`：`_hspec_submit_accumulate_hidden_states`、`_set_up_drafter` |
| verl rollout 打包 | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：`generate_sequences`、写入 `non_tensor_batch["rollout_hidden_states"]` |
| trainer 聚合和提交建表 | `verl/trainer/ppo/ray_trainer.py`：`fit`、`prompt_build_data` 聚合、`build_tables_async`、`ray.get`（`hspec_build_wait`） |
| PCA 和 table actor | `vllm_ascend/spec_decode/hspec_table.py`：`HSpecTableGroup.build_prompt_table`、`GlobalHSpecTableGroup.build_tables_async`、`swap`；`hspec_utils.py`：`fit_pca_multi_sequence` / `compute_pca` |
| 在线 proposer | `vllm_ascend/spec_decode/hspec_proposer.py`：`generate_token_ids`、`_fire_prefetch_async` / `_poll_pending`、`_build_cached_table`；`hspec_table.py`：`prefetch_batch_async` |
| 进程级初始化 | `verl/trainer/main_ppo.py`：`init_hspec_tables`；各 vLLM worker 内 `get_hspec_tables()` |

## 1. 当前问题图景

当前链路大致是：

```text
NPU sample_hidden_states
  -> CPU pinned tensor
  -> hspec_flush_and_get_all()
  -> rollout_hidden_states 放入 DataProto.non_tensor_batch
  -> Ray 返回 trainer
  -> trainer 聚合 prompt_build_data
  -> Ray 发给 HSpecTableGroup actor
  -> actor 内 astype(float32) / concatenate / SVD / build table
  -> epoch swap
```

### 1.1 数据量和内存模型

HSpec 收集对象是 token 级 anchor hidden state。单条轨迹 hidden state 大小约为：

```text
bytes_per_sequence = generated_tokens * hidden_dim * dtype_bytes
```

若 `hidden_dim=4096`、`dtype=fp16`，每个 token 约 8 KB。一个 step 若 `train_batch_size=64`、`rollout.n=8`，逻辑上最多 512 条轨迹。平均 response 长度只要达到 2048 token，原始 hidden state 就是约 8 GB 级别；若平均长度接近长上下文实验的数千到万级 token，单 step 数据面会迅速达到十几 GB 到数十 GB。当前路径里这些数据至少经过以下副本或引用生命周期：

| 位置 | 当前形态 | 系统问题 |
| --- | --- | --- |
| `hspec_utils._hspec_host_buffers` | 每个 req 一个 CPU tensor list | copy worker append 后等 rollout flush，生命周期覆盖整个 `LLM.generate()` |
| `hspec_flush_and_get_all()` 返回值 | `dict[req_id] -> np.ndarray(fp16)` | 大 dict 进入 rollout Python heap |
| `vllm_rollout_spmd.py` | object ndarray `rollout_hidden_states` | 进入 `DataProto.non_tensor_batch`，随 Ray RPC 返回 trainer |
| `ray_trainer.py` | `prompt_build_data[prompt_id]["hidden_states"]` | trainer 再聚合一份 prompt 级 Python list |
| `HSpecTableGroup` actor | Ray 参数 + actor heap | Ray object store、反序列化、actor heap 和 GC 叠加 |
| PCA 内部 | `astype(float32)`, `concatenate`, `centered`, SVD workspace | 峰值从 fp16 原始 H 放大到多份 fp32 大矩阵 |

### 1.2 当前四类不优点和落地修正方向

| 当前问题 | 当前代码 | 代价分析 | 落地修正 |
| --- | --- | --- | --- |
| Ray 大对象路径错误 | [vllm_rollout_spmd.py:760](verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:760)、[ray_trainer.py:1501](verl/trainer/ppo/ray_trainer.py:1501)、[hspec_table.py:836](vllm_ascend/spec_decode/hspec_table.py:836) | hidden states 是 GB 级数据面，不适合 DataProto/Ray object store。Ray 序列化和 Plasma 会复制大 ndarray，且 trainer 主进程被迫承载数据面。 | 用本地 mmap 或 shared memory 承载 hidden states。DataProto 只携带小 descriptor，或 rollout worker 通过 side-channel 直接向 build shard 注册 descriptor。 |
| 生命周期过长 | [batch.union(gen_batch_output)](verl/trainer/ppo/ray_trainer.py:1230)、[update_actor(batch)](verl/trainer/ppo/ray_trainer.py:1537) | HSpec hidden states 在 reward、old_logprob、ref、adv、actor update 阶段都不再需要，却仍随 batch 存活和传递。 | 建表提交后立即从 batch 删除 HSpec 大字段。重构后 batch 中不出现大字段，只保留小 descriptor，并在 build descriptor 组装后删除。 |
| PCA 内存形态差 | [compute_pca](vllm_ascend/spec_decode/hspec_utils.py:560)、[fit_pca_multi_sequence](vllm_ascend/spec_decode/hspec_utils.py:668) | `np.concatenate` 把多轨迹合成 `(N,D)`；SVD 还会产生 centered 和 workspace。对长 rollout，单 prompt 就可能产生数百 MB 临时内存。 | 改成 tiled streaming PCA。先流式求 mean，再用 tiled covariance 或 randomized PCA，最后二次扫描写 keys。 |
| HSpecTableGroup 职责混杂 | [HSpecTableGroup](vllm_ascend/spec_decode/hspec_table.py:198)、[_num_groups=5](vllm_ascend/spec_decode/hspec_table.py:195) | 同一个 Ray actor 同时接收大对象、做 PCA、保存 active/building table、服务 prefetch，导致 Ray 调度、BLAS 线程、actor heap 和 query 控制面耦合。 | 拆成 collector、build shard、table store、coordinator。Ray 只做控制面，小元数据进入 actor，大矩阵全部走本地文件或 shared memory。 |

## 2. 推荐架构

目标数据流：

```text
NPU sample_hidden_states
  -> worker-local pinned pool
  -> worker-local append-only mmap raw store
  -> descriptor 注册
  -> build shard 按 descriptor 流式读取 mmap
  -> tiled PCA / projection
  -> mmap table store
  -> coordinator 原子 swap version
  -> proposer 根据 descriptor mmap 读取并缓存到 NPU
```

### 2.1 四层架构和代码落地

| 设计层 | 当前对应代码 | 新职责 | 代码落地方案 |
| --- | --- | --- | --- |
| `HSpecCollector` | [hspec_utils.py](vllm_ascend/spec_decode/hspec_utils.py:856)、[model_runner_v1.py:2403](vllm_ascend/worker/model_runner_v1.py:2403) | vLLM rollout worker 本地收集 anchor hidden states 和 token ids；写 mmap；返回小 descriptor。 | 新增 `vllm_ascend/spec_decode/hspec_store.py` 或 `hspec_collector.py`。把 `_hspec_host_buffers` 从 `dict[str, List[Tensor]]` 改为 append-only writer；`hspec_flush_and_get_all()` 兼容保留，但新增 `hspec_flush_and_get_descriptors()`。 |
| `HSpecBuildShard` | `hspec_table.py`：`HSpecTableGroup.build_prompt_table` | 只接收 descriptor；按 prompt 读 mmap；流式 PCA 和 projection；写 table mmap。 | 新增 `hspec_builder.py`。**默认本机子进程**（§6.2 A）；若用 Ray，必须节点亲和且入参为 `HSpecTrajectoryDesc`，禁止 ndarray。 |
| `HSpecTableStore` | [PromptTableData](vllm_ascend/spec_decode/hspec_table.py:34)、[get_active_table_data_batch](vllm_ascend/spec_decode/hspec_table.py:451) | 保存 active table 为 mmap 连续数组；提供 prompt table descriptor。 | 新增 `hspec_table_store.py`。把 `mean/components/keys/token_buffer/offsets` 从 actor heap 移到 versioned table files。`get_active_table_data_batch()` 改为返回路径、offset、shape、dtype、version。 |
| `HSpecCoordinator` | [GlobalHSpecTableGroup](vllm_ascend/spec_decode/hspec_table.py:718) | 只协调 epoch、build 状态、metrics、active version；不传大数组。 | 可在 `hspec_table.py` 中保留 `GlobalHSpecTableGroup` 名称，内部重定向到 coordinator API，保证 trainer/proposer 迁移成本低。 |

### 2.2 Descriptor 数据结构

重构的核心是把 “大矩阵” 和 “元数据” 分离。建议定义不可变 descriptor：

```python
@dataclass(frozen=True)
class HSpecTrajectoryDesc:
    epoch: int
    global_step: int
    worker_rank: int
    request_id: str
    prompt_id: str
    hs_path: str
    hs_offset_rows: int
    token_path: str
    token_offset: int
    length: int
    hidden_dim: int
    hs_dtype: str
    token_dtype: str
    reward: float | None = None
```

落地要点：

- `request_id` 用于 rollout 输出和 collector 写入结果对齐。
- `prompt_id` 仍由 [prompt_id_from_token_ids](vllm_ascend/spec_decode/hspec_utils.py:211) 生成，保证 epoch 间稳定。
- `reward` 在 rollout worker 阶段未知，trainer 计算 reward 后填入或附加到 build request。
- descriptor 可以进入 `DataProto.non_tensor_batch`，因为它是小对象；hidden state 不可以进入。

### 2.3 分布式部署与节点亲和（相对 tips 的必补澄清）

tips 要求「本机 mmap + 本机 build shard」，但当前框架存在三类进程，**若不做节点亲和，descriptor 中的绝对路径会失效**：

| 进程角色 | 典型部署 | 当前 HSpec 行为 | 重构后约束 |
| --- | --- | --- | --- |
| **Driver / `RayPPOTrainer`** | 单进程，常在 head 或 CPU 节点 | 聚合 `prompt_build_data`，`build_tables_async` RPC 到全局 `hspec_table_{i}` actor | **禁止**在 driver 上读取 rollout worker 的 `hs_path`；driver 只传 **可序列化小对象**（descriptor 元数据 + reward），不读 mmap |
| **VERL `actor_rollout` worker** | 每 GPU 一 Ray actor，与 vLLM 同机 | `generate_sequences` → `LLM.generate`；flush HS 在本机 Python 进程 | **Collector 必须落在该 worker 进程内**；mmap 根目录对该 worker 节点本地可见 |
| **全局 `HSpecTableGroup` Ray actor** | `init_hspec_tables()` 创建，**默认无节点亲和**，可能调度到 head | 接收 `hidden_states_list` ndarray，PCA + 双缓冲 dict | **数据面迁出**；actor 退化为 **Coordinator**（仅 version/metrics/barrier） |

**强制设计原则（写进实现 checklist）**：

1. **数据面三段都在「产生或消费 HS 的同一物理节点」完成**：`collect (mmap write) → build (mmap read) → table store (mmap write)`。
2. **跨节点只允许传控制面**：`HSpecTrajectoryDesc` 的逻辑字段 + `node_id`/`store_root` 标识；**禁止**把 `hs_path` 指向 A 节点文件却让 B 节点 actor `open()`。
3. **Trainer 不成为 HS 汇聚点**：driver 侧 `prompt_build_data` 改为 `prompt_id → List[HSpecTrajectoryDesc]`（或 descriptor id），**不**再 `append(hs ndarray)`。

#### 2.3.1 推荐拓扑（与现有 VERL + vLLM 对齐）

单机 `trainer.nnodes=1`（当前脚本默认）：

```text
Node-0 (16× NPU)
├── verl actor_rollout workers (×16)     # 每个进程: HSpecLocalCollector + vLLM model_runner
├── HSpecBuildShard (×HSPEC_NUM_SHARDS)  # 同节点进程: 读 Node-0 的 HSPEC_STORE_DIR
├── HSpecTableStore                        # 同节点: ${HSPEC_TABLE_STORE_DIR}
└── HSpecCoordinator (Ray, 可选 head)    # 仅元数据; num_cpus 很小
```

多机 `trainer.nnodes>1`（必须显式配置，否则 Phase 1+ 不可用）：

```text
Node-k (rollout 节点)
├── actor_rollout workers on Node-k only
├── HSpecBuildShard(s) on Node-k only      # 只 build 本节点 collector 写出的 mmap
├── table_store/version_*/shard_* on Node-k local disk or shared FS mount
└── proposer prefetch: 只 mmap 本节点 table_store（或节点本地 NFS 挂载点）

Head / driver
└── Coordinator: collect build_done / active_version; 不触碰 hs.bin
```

**与 `INFER_TP` 的关系**：16 卡、`tensor_model_parallel_size=4` 时，逻辑上有 4 个 vLLM 进程组（每组 4 卡 TP）。`HSPEC_NUM_SHARDS=4` 时，建议 **shard_id = tp_group_id**，`HSPEC_STORE_DIR` 与 `HSPEC_TABLE_STORE_DIR` 按 `{node_id}/tp_{gid}/` 分子目录，避免 4 组进程争用同一 append 文件锁。

#### 2.3.2 `HSpecTrajectoryDesc` 字段补充（框架可落地）

在 §2.2 基础上，生产环境必须增加拓扑字段，且 **reward 分阶段填充**：

```python
@dataclass(frozen=True)
class HSpecTrajectoryDesc:
    epoch: int
    global_step: int
    node_id: str              # 例如 os.environ["NODE_RANK"] 或 Ray node id
    worker_rank: int          # verl worker / torch.distributed rank
    tp_group_id: int          # floor(worker_rank / infer_tp) 或 vLLM 提供的组 id
    shard_id: int             # stable_partition_id(prompt_id, HSPEC_NUM_SHARDS)
    request_id: str
    prompt_id: str
    hs_path: str              # 本节点绝对路径; 仅 build shard 在本节点 open
    hs_offset_rows: int
    token_path: str
    token_offset: int
    length: int
    hidden_dim: int
    hs_dtype: str             # "float16" | "bfloat16" | "uint16_raw_bf16"
    token_dtype: str
    reward: float | None = None   # rollout 阶段为 None; trainer 填 reward 后再 submit build
```

**Reward 生命周期（与 `ray_trainer.py` 对齐）**：

| 阶段 | 代码位置 | `reward` |
| --- | --- | --- |
| Rollout flush | `vllm_rollout_spmd.py` | `None`（reward 尚未计算） |
| Driver 组装 build 请求 | `ray_trainer.py`（`token_level_scores` 已写入 batch 后） | `batch_item.batch["token_level_scores"].sum().item()`，与现逻辑一致 |
| Phase 2 提前 reward | `ray_trainer.py` 重排后 | 仍在 **rule-based `compute_reward` 完成之后** 再 `build_tables_async`；RM 异步场景见 §7.2 |

**禁止**：在 descriptor 中携带 `np.ndarray` 或 Ray `ObjectRef` 指向 HS。

#### 2.3.3 Build Shard 的两种合法实现（澄清 §6.2 歧义）

| 模式 | 适用 | 实现要点 | 与现框架衔接 |
| --- | --- | --- | --- |
| **A. 节点本地子进程（推荐默认）** | `nnodes=1` 或每节点独立盘 | `multiprocessing.Process` / `subprocess` 启动 `HSpecBuildShard`，`HSPEC_STORE_DIR` 共享；无 Ray 传大对象 | 不依赖 `hspec_table_{i}` 做 PCA |
| **B. 节点亲和 Ray actor** | 需要 Ray 统一监控 | `HSpecBuildShard.options(resources={"node_id": ...}).remote()`，且 **仅**接收 `List[HSpecTrajectoryDesc]` | 替换 `build_tables_batch.remote(prompt_data_with_ndarray)` |

**错误模式（必须禁止）**：延续现 `init_hspec_tables()` 在集群任意节点创建 5 个 `HSpecTableGroup`，并由 **driver** 把整包 `hidden_states_list` `remote()` 进去——这与 mmap 设计矛盾。

#### 2.3.4 共享文件系统（可选）

若多机但共用 NFS/Lustre：

- `HSPEC_STORE_DIR` / `HSPEC_TABLE_STORE_DIR` 必须为 **所有 rollout 节点可读写** 的挂载路径；
- 文件名仍须含 `node_id`/`worker_rank` 前缀，避免并发写同一 `hs.bin`；
- build shard 只读「本 epoch + 本 shard 分区」子目录，降低锁竞争。

未配置共享盘时，**默认按节点私有盘**，coordinator 维护 `node_id → active_version` 映射；proposer **只 prefetch 本节点** table（多机时每节点独立 HSpec 表，epoch 语义仍为「本节点见过的 prompt」——与现 Ray 全局 actor 的「逻辑全局表」不同，需在 metrics 上标注 `hspec/table_scope=per_node`）。

## 3. Hidden State 收集设计

当前收集链路的关键代码是 [hspec_submit_accumulate_task](vllm_ascend/spec_decode/hspec_utils.py:1264)、[_hspec_copy_worker](vllm_ascend/spec_decode/hspec_utils.py:1024)、[hspec_extend_step_tokens](vllm_ascend/spec_decode/hspec_utils.py:1369) 和 [hspec_flush_and_get_all](vllm_ascend/spec_decode/hspec_utils.py:1388)。

### 3.1 每个 rollout worker 维护 append-only mmap 文件

当前设计 tip：每个 rollout worker 维护 `epoch_E/worker_W/hs.bin` 和 `tokens.bin`。

当前代码问题：

- `_hspec_host_buffers` 是 `Dict[str, List[torch.Tensor]]`，所有 request 的 CPU tensor 留在 Python heap。
- `hspec_flush_and_get_all()` 把 tensor list `torch.cat` 成 ndarray，再返回给 rollout 层。
- `vllm_rollout_spmd.py` 又把 ndarray 放进 object array，进一步延长生命周期。

落地方案：

- 在 `hspec_utils.py` 中引入进程内 singleton `HSpecLocalCollector`，初始化目录来自环境变量 `HSPEC_STORE_DIR`，默认放在本地 NVMe 或 `/dev/shm` 下的可控目录。
- `_hspec_copy_worker` 不再 append 到 `_hspec_host_buffers`，而是调用 `collector.append_hidden_rows(req_id, cpu_tensor[start:end])`。
- token ids 由 `hspec_extend_step_tokens()` 写入 collector 的 token buffer；flush 时只完成 descriptor，避免组装大 ndarray。
- `hspec_clear_store()` 改为清空当前 batch 未提交的 req 状态，不删除已经落盘的 mmap 段；真正清理由 epoch GC 负责。

建议文件布局：

```text
${HSPEC_STORE_DIR}/epoch_0003/worker_0007/
  hs.fp16.bin
  tokens.i32.bin
  desc.jsonl
  manifest.json
```

`hs.fp16.bin` 只保存连续 hidden rows。`tokens.i32.bin` 保存 token 序列。`desc.jsonl` 保存每条轨迹的 offset 和 length，便于 crash/debug。

### 3.2 `hspec_submit_accumulate_task` 直接写本地段并登记 descriptor

当前设计 tip：`hspec_submit_accumulate_task` 不再最终返回大 ndarray 给 rollout 层，而是直接写入本地段，并登记 descriptor。

当前代码问题：

- [hspec_submit_accumulate_task](vllm_ascend/spec_decode/hspec_utils.py:1264) 已经做了正确的 row mapping 和异步 D2H copy，但 copy 完成后进入 `_hspec_host_buffers`。
- [vllm_rollout_spmd.py:637](verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:637) 在 generation 后调用 `hspec_flush_and_get_all()`，被迫等待并获取全部大矩阵。

落地方案：

- 保留 `_hspec_compute_req_slices()` 的逻辑，因为它已经解决 spec decode 下 target rows、bonus row 和 accepted prefix 的对齐。
- 修改 `_HSpecAsyncCopyTask` 的消费端：copy event synchronize 后，把 `cpu_tensor[start:end]` 追加到 mmap writer，并更新 req 内部 row count。
- 新增 `hspec_flush_and_get_descriptors(request_id_to_prompt_id: dict[str, str]) -> dict[str, HSpecTrajectoryDesc]`。该函数等待 pending copy，只返回 descriptor，不返回 hidden state ndarray。
- `vllm_rollout_spmd.py` 在 `output_collect` 阶段用 `output.request_id` 查 descriptor，再把 descriptor 与 `response_ids` 对齐。

兼容策略：

- 保留 `HSPEC_LEGACY_DATAPROTO_HS=1` 走旧 `rollout_hidden_states` 路径，便于 A/B 验证。
- 默认 `HSPEC_LEGACY_DATAPROTO_HS=0`，只返回 descriptor。

### 3.3 使用固定大小 pinned host buffer pool

当前设计 tip：使用固定大小 pinned host buffer pool，避免每步 `torch.empty(pin_memory=True)` 频繁分配。

当前代码问题：

- [hspec_submit_accumulate_task](vllm_ascend/spec_decode/hspec_utils.py:1308) 每次 D2H 都 `torch.empty(..., pin_memory=True)`。
- 生成过程 token 数不稳定，会产生大量不同 shape pinned allocation。长 rollout 下 pinned memory 碎片和系统调用开销会非常明显。

落地方案：

- 在 `hspec_utils.py` 新增 `HSpecPinnedPool`，以 `(dtype, hidden_dim, bucket_rows)` 为 key 维护固定槽位。
- bucket rows 用 2 的幂或配置，例如 `64/128/256/512` rows。小 batch copy 向上取 bucket，避免 shape 爆炸。
- `hspec_submit_accumulate_task` 从 pool checkout buffer，copy worker 在 mmap write 完成后 release。
- 增加预算参数：`HSPEC_PINNED_POOL_BYTES`、`HSPEC_PINNED_POOL_MAX_SLOTS`。超过预算时退化为 pageable CPU copy 或丢弃本步 HSpec collection，而不是阻塞 NPU decode。

### 3.4 `rollout_hidden_states` 不进入 `DataProto.non_tensor_batch`

当前设计 tip：`rollout_hidden_states` 不进入 `DataProto.non_tensor_batch`。短期保留旧路径时，至少建表提交后立刻 `pop`。

当前代码问题：

- [vllm_rollout_spmd.py:760](verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:760) 把 hidden state ndarray 放进 `non_tensor_batch["rollout_hidden_states"]`。
- [DataProto.repeat](verl/protocol.py:1001) 和 [DataProto.union](verl/protocol.py:788) 会继续携带 non-tensor 字段。
- `batch` 后续会进入 old logprob、ref、actor update 等路径，大对象字段没有必要存在。

落地方案：

- `vllm_rollout_spmd.py` 改为写 `non_tensor_batch["hspec_desc"]`，元素是小 descriptor 或 descriptor id。
- `ray_trainer.py` 的 prompt 聚合逻辑从 `rollout_hidden_states` 改为 `hspec_desc`，不再读取 ndarray。
- 如果旧路径暂时保留，则在 [ray_trainer.py:1526](verl/trainer/ppo/ray_trainer.py:1526) 提交 build 后立即执行：

```python
for key in ("rollout_hidden_states", "rollout_hspec_tokens", "hspec_rollout_debug"):
    batch.non_tensor_batch.pop(key, None)
```

更理想的删除点是在 `prompt_build_data` 构造后、`update_actor(batch)` 前，保证 actor worker 不接收任何 HSpec 数据面。

### 3.5 validation 阶段默认不收集 hidden states

当前设计 tip：validation 可以使用 HSpec decode 查询，但不写 building table，不打包 hidden states。

当前代码问题：

- `model_runner_v1.py::_set_up_drafter` 中只要 `method=="hspec"` 就调用 `hspec_set_collection_enabled(True)`。
- `vllm_rollout_spmd.py` 能看到 `is_validate = prompts.meta_info.get("validate", False)`，但当前只影响 sampling 参数，不影响 hidden-state collection。

落地方案：

- 在 `vllm_rollout_spmd.py::generate_sequences` 中计算：

```python
is_validate = prompts.meta_info.get("validate", False)
collect_hspec = use_hspec and not is_validate and prompts.meta_info.get("do_sample", True)
```

- 在 `collect_hspec=False` 时：不调用 `hspec_clear_store()` 误清 prefetch 所需状态；仅跳过 `hspec_submit_accumulate_task` / flush descriptor 写入。
- 新增 `NPUModelRunner.hspec_set_runtime_collection_enabled(bool)`（`model_runner_v1.py`），由 rollout 在 `LLM.generate` 前通过现有 `collective_rpc` 模式下发（与 `hspec_prefetch_prompt_ids_batch` 同路径）。
- `hspec_submit_accumulate_task` 除全局 `_hspec_collection_enabled`（init 时 `method==hspec` 置 True）外，再检查 **runtime flag**；validation 结束后下一训练 step 必须恢复 `True`。
- **查询仍可用**：validation 可保留 `use_hspec_decode` 与 proposer prefetch（读 active table），仅关闭 **building 侧** raw H 采集。

## 4. PCA 建表设计

当前 PCA 入口是 [HSpecTableGroup.build_prompt_table](vllm_ascend/spec_decode/hspec_table.py:254)，具体 PCA 在 [fit_pca_single_sequence](vllm_ascend/spec_decode/hspec_utils.py:632)、[fit_pca_multi_sequence](vllm_ascend/spec_decode/hspec_utils.py:668) 和 [compute_pca](vllm_ascend/spec_decode/hspec_utils.py:560)。

### 4.1 保留 prompt 级 PCA 语义

当前设计 tip：保留 prompt 级 PCA 语义，但不要对完整 `(N,D)` 做一次性 SVD。

当前代码问题：

- GRPO 下同 prompt 的多条 rollout 进入 `fit_pca_multi_sequence()` 后执行 `np.concatenate(hidden_states_list, axis=0)`。
- `compute_pca()` 对完整矩阵做 mean、center、`np.linalg.svd(centered)`。
- 这保持算法语义，但内存峰值和 SVD workspace 与 RL 长输出场景不匹配。

落地方案：

- prompt 级语义保持不变：一个 prompt_id 对应一个 `mean` 和 `components`，多个 rollout 的 hidden states 仍池化。
- 输入从 `List[np.ndarray]` 改为 `List[HSpecTrajectoryDesc]`，PCA 函数以 iterator 方式读取 tiles：

```python
def iter_prompt_hidden_tiles(descs, tile_rows):
    for desc in descs:
        mm = open_hs_memmap(desc)
        for start in range(0, desc.length, tile_rows):
            yield mm[start:start + tile_rows]
```

- `HSpecTableGroup.build_prompt_table()` 拆出为 `HSpecBuildShard.build_prompt_from_descs(prompt_id, descs, rewards)`。

### 4.2 原始 H 以 fp16/bf16 mmap 保存

当前设计 tip：原始 H 以 fp16/bf16 mmap 保存。

当前代码问题：

- `hspec_flush_and_get_all()` 最终把 CPU tensor 转成 `float16 numpy`，但这个 ndarray 仍在 Python/Ray 路径中。
- `HSpecTableGroup.build_prompt_table()` 立刻 `astype(np.float32)`，导致 fp16 节省只在早期有效。

落地方案：

- collector 落盘 dtype 默认保持 `sample_hidden_states.dtype`，通常 bf16/fp16；如果 numpy 不支持 bf16 原生存储，可按 uint16 raw 保存，并在 reader 中解释。
- build shard tile 读取时局部 cast 到 fp32，tile 处理完立即释放。
- descriptor 中必须记录 `hs_dtype`、`hidden_dim` 和 `length`，避免 reader 依赖全局模型配置。

### 4.3 流式计算 mean

当前设计 tip：对每个 prompt 先流式计算 mean。

当前代码问题：

- `compute_pca()` 对完整 `hs` 调 `hs.mean(axis=0)`，要求完整矩阵已经在内存。

落地方案：

- build shard 第一遍扫描 descs，维护 `sum_h: float64 或 float32[D]` 和 `N`。
- 对每个 tile 执行 `sum_h += tile.astype(float32).sum(axis=0)`。
- `mean = sum_h / N`，写入 table store 的 prompt metadata。
- 若 `N < 2` 或有效 token 过少，直接丢弃该 prompt 或构造 zero-padded components，并增加 `hspec/discard_count`。

### 4.4 精确 tiled covariance PCA

当前设计 tip：要求接近精确 PCA 时，用 tiled covariance：

```text
G = Σ H^T H
C = G / N - μ μ^T
eig(C) -> top-K components
```

当前代码问题：

- full SVD 的内存是 `O(ND)` 多副本，长 rollout 主要瓶颈是内存和对象生命周期。
- 对 `N >> D` 的 hidden states，covariance 的内存是 `O(D^2)`，更适合流式。

落地方案：

- 新增 `compute_pca_tiled_covariance(descs, n_components, tile_rows)`。
- 第二遍扫描 descs，tile cast fp32 后执行 BLAS GEMM：`G += tile.T @ tile`。
- 得到 `C` 后用 `np.linalg.eigh(C)` 或 scipy LAPACK 求 top-K。只保留最大 K 个 eigenvectors。
- 对大 hidden_dim 需要预算判断：若 `D*D*4` 超过 `HSPEC_PCA_COV_MAX_BYTES`，自动切 randomized PCA。

系统代价：

- 内存峰值约为 `tile_rows * D * 4 + D * D * 4`。
- 对 `D=4096`，`D^2 fp32` 约 64 MB，可控；对 `D=8192` 约 256 MB，需要限制并发 shard 数。

### 4.5 randomized PCA 快速路径

当前设计 tip：关注速度时使用 randomized PCA，维护 `CΩ = Σ H^T(HΩ)`。

当前代码问题：

- 当前 `np.linalg.svd(centered)` 追求完整精确 SVD，但 HSpec 只需要稳定的低维 key，不需要全谱。

落地方案：

- 新增 `compute_pca_randomized_cov(descs, K, oversample=16/32)`，令 `r=K+oversample`。
- `Ω` 由 `prompt_id` 和 active version seed 生成，shape `(D,r)`，保证可复现。
- 扫描 centered tile，计算：

```text
T = (H_tile - μ) @ Ω
Y += (H_tile - μ)^T @ T
```

- 对 `Y(D,r)` 做 QR 得到 `Q(D,r)`。
- 再扫描一次 tile，计算小矩阵：

```text
B += ((H_tile - μ) @ Q)^T @ ((H_tile - μ) @ Q)
```

- 对 `B(r,r)` eig，`components = (Q @ eigvecs_top).T`。
- 默认策略：`HSPEC_PCA_METHOD=randomized` 用于 30B 长 rollout；`covariance` 用于短序列或 debug 精确对照。

### 4.6 二次扫描投影并写 table keys

当前设计 tip：得到 `μ,W` 后，再二次扫描 mmap，分块计算 `Z=(H-μ)W^T`，写入 table keys。raw H 完成后即可删除。

当前代码问题：

- `PromptTableData.add_rollout()` 当前要求传入完整 `projected_keys` ndarray。
- `fit_pca_multi_sequence()` 生成 `projected_list = [params.project(hs) for hs in hidden_states_list]`，又产生一份 `(N,K)` 中间数组。

落地方案：

- `PromptTableWriter` 提供 `append_projected_rollout(desc, projected_tile_iter, token_seq, reward)`。
- 对每条 trajectory 单独扫描 hidden mmap，按 tile 计算 `Z_tile`，直接写入 `keys.f16.bin`。
- value 存储继续沿用当前 value shift 语义：entry at `t` 的 value 从 `y[t+1:]` 开始，避免 draft 重复已接受 token。当前语义在 [PromptTableData.add_rollout](vllm_ascend/spec_decode/hspec_table.py:68)。
- raw hidden mmap 的 GC 由 epoch barrier 后执行：当 table version 已写入且不再需要 debug dump 时删除。

## 5. CPU/NPU 调度

### 5.1 默认 CPU build shard，不把 PCA 丢到训练 NPU

当前设计 tip：默认用 CPU build shard 做 PCA/投影，依赖 MKL/OpenBLAS，限制每个 shard 的线程数。

当前代码问题：

- 当前 PCA 在 Ray actor 进程里跑，但没有明确 CPU core、BLAS 线程和内存预算。
- 若未来简单把 PCA 放到 NPU，会和 actor forward/update、vLLM decode、rejection sampler、HSpec proposer match 竞争 NPU kernel queue。

落地方案：

- `HSpecBuildShard` 初始化时设置 `OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS` 或使用 `threadpoolctl.threadpool_limits(limits=threads)`。
- shard 数按 NUMA、物理 socket、rollout TP group 设置，不按固定 5 个 actor。
- 默认配置：

```text
HSPEC_BUILD_BACKEND=cpu
HSPEC_BUILD_SHARDS_PER_NODE=4
HSPEC_BUILD_THREADS_PER_SHARD=8
HSPEC_BUILD_MAX_INFLIGHT_PROMPTS_PER_SHARD=1 或 2
```

- 如果 CPU build 落后，可以增加 shard 或 randomized PCA，而不是占用 NPU。

### 5.2 NPU 只作为可选加速器，并必须有预算控制

当前设计 tip：NPU 只用于 projection 或 randomized PCA 的大 GEMM，而且必须有预算控制。

当前代码现状：

- 在线 proposer 已经在 [HSpecProposer.generate_token_ids](vllm_ascend/spec_decode/hspec_proposer.py:1048) 使用 NPU 做 batch projection 和 `bmm` match，这是正确的热路径优化。
- 建表侧目前在 CPU actor 中做 numpy SVD，没有 NPU build。

落地方案：

- 建表 NPU backend 只允许在 `HSPEC_BUILD_USE_NPU=1` 时启用。
- NPU build 只能跑 projection 或 randomized PCA 的大 GEMM，不跑完整 SVD。
- scheduler 必须检查训练状态：只有 rollout engine sleep、actor update 空闲、或指定独立 NPU 才允许 build kernel。
- 对同一张 NPU，不允许 build projection 与 online proposer 同时提交大 GEMM；否则 HSpec 会优化 decode token 数，却拖慢每 step 训练。

### 5.3 backpressure 和降级策略

当前设计 tip：当 host memory、pinned memory、NPU queue 任一超过阈值，暂停收集或延后 build。

当前代码问题：

- 当前路径没有全局预算。只要 rollout 产生 hidden states，就继续进入 Python heap、DataProto、Ray actor。
- `ray.get(ray_hspec_tasks)` 会把 build 落后直接转化为 step stall。

落地方案：

- collector 预算：

```text
HSPEC_COLLECT_MAX_BYTES_PER_WORKER
HSPEC_PINNED_POOL_BYTES
HSPEC_RAW_STORE_MAX_BYTES_PER_EPOCH
```

- build queue 预算：

```text
HSPEC_BUILD_QUEUE_MAX_DESCS
HSPEC_BUILD_QUEUE_MAX_BYTES
HSPEC_BUILD_MAX_PENDING_EPOCHS
```

- 超阈值策略按优先级：

| 场景 | 策略 |
| --- | --- |
| pinned pool 满 | 本次 copy 用 pageable fallback 或跳过 collection，不阻塞 decode |
| raw mmap store 超预算 | 停止收集新请求，只保留已收集 descriptor |
| build queue 落后 | epoch swap 使用已完成 prompt 的 partial table，未完成 prompt 沿用旧 active 或 miss |
| table store 超预算 | 按 prompt 最近访问、reward、entry count 做淘汰 |

指标必须上报：`hspec/collect_dropped`, `hspec/backpressure_active`, `hspec/build_queue_bytes`, `hspec/raw_store_bytes`。

## 6. HSpecTableGroup 重构

当前 `HSpecTableGroup` 同时承担构建、存储、query、metrics、ZMQ server。重构目标是拆为 **本机数据面组件 + 轻量 Coordinator**，而不是把「Ray actor」继续当作 HS/表的载体。

### 6.0 职责拆分与现码映射（相对 tips 的核心澄清）

| 组件 | 现码 | 重构后 | 禁止事项 |
| --- | --- | --- | --- |
| **HSpecLocalCollector** | `hspec_utils._hspec_host_buffers`、`_hspec_token_buffers` | 本进程写 `hs.fp16.bin` / `tokens.i32.bin` | 返回大 ndarray 给 rollout（默认路径） |
| **HSpecBuildShard** | `HSpecTableGroup.build_prompt_table` | 本节点读 mmap，流式 PCA，写 `HSpecTableStore` | `remote(hidden_states_list=...)` |
| **HSpecTableStore** | `HSpecTableGroup._active` / `_building` dict | versioned mmap + `manifest.json` | 在 Ray actor heap 常驻 `(M,K)` keys |
| **HSpecCoordinator** | `GlobalHSpecTableGroup` + `init_hspec_tables` | epoch barrier、swap 元数据、聚合 metrics | prefetch 返回含 ndarray 的 dict |

**在线查询路径（与现码一致，重构后加强）**：

- 当前 **steady-state** 已在 `HSpecProposer` 内用 **worker-local cache** + NPU `bmm`，`_fire_prefetch_async` → `prefetch_batch_async`（Ray）→ `_poll_pending` → `_build_cached_table`（**全量 copy**，`hspec_proposer.py:979-986`）。
- 重构后：prefetch 只拉 **`HSpecPromptTableDesc`**（路径/offset/shape/version），在 **rollout worker 进程**内 `np.memmap` + 一次性 H2D 建 cache；**不在** `generate_token_ids()` 内做文件 I/O。
- 现码 `GlobalHSpecTableGroup.post_query_batch`（ZMQ）为历史「远程 query」路径；`HSpecProposer` **热路径未使用** ZMQ（仅 Ray prefetch）。重构后 **废弃 ZMQ 查询**（见 §6.5）。

### 6.1 `_num_groups=5` 不作为数据面分片

当前设计 tip：单机 16 卡、`INFER_TP=4` 时，rollout 逻辑上有 4 个 TP group，可设 4 个 build/table shard。

当前代码问题：

- [_num_groups: int = 5](vllm_ascend/spec_decode/hspec_table.py:195) 是任意常量。
- `stable_partition_id(prompt_id, _num_groups)` 把 prompt 分到 5 个 Ray actor，但与硬件拓扑、TP group、NUMA、磁盘路径无关。

落地方案：

- 新配置：`HSPEC_NUM_SHARDS`，默认从 rollout TP group 或 `INFER_DP` 推导。
- 分片函数仍可用 `stable_partition_id(prompt_id, num_shards)`，但 shard 资源必须明确：CPU core set、mmap root、build threads、table cache budget。
- 对 16 卡、`INFER_TP=4` 的脚本，建议 4 个 shard，对齐 4 个 TP group 的 rollout worker 集合，降低跨 NUMA 和跨进程文件访问。
- **分片键**：`shard_id = stable_partition_id(prompt_id, HSPEC_NUM_SHARDS)`，与现 `hspec_table.py:_get_partition_id` 一致，但 **shard 进程与 TP 组/节点目录绑定**（§2.3.1），而非全局 5 个任意 Ray actor。

### 6.2 每个 shard 明确资源与部署形态

当前设计 tip：每个 shard 明确 CPU core、内存预算、BLAS 线程数、mmap 目录。

当前代码问题：

- `HSpecTableGroup.options(num_cpus=1)` 只声明 1 CPU，但实际 `np.linalg.svd` 可能调用多线程 BLAS，造成 CPU oversubscription。
- actor heap 保存 active/building table，没有内存上限。
- `init_hspec_tables()`（`hspec_table.py:1190`）在 driver 侧创建 **全局命名 actor**，与 rollout worker **不在同一调度实体**，无法直接读 worker mmap。

落地方案（**二选一，默认 A**）：

**A. 节点本地 BuildShard 进程（推荐，`nnodes=1` 与多机 per-node 通用）**

```python
# 启动时机: 每个物理节点 boot 一次，或 verl worker 首次 generate 前 lazy start
HSpecBuildShardProcess(
    shard_id=sid,
    store_root=os.environ["HSPEC_STORE_DIR"],
    table_root=os.environ["HSPEC_TABLE_STORE_DIR"],
    threads=int(os.environ.get("HSPEC_BUILD_THREADS_PER_SHARD", "8")),
    max_memory_bytes=int(os.environ.get("HSPEC_BUILD_MAX_MEMORY_BYTES", str(4 * 1024**3))),
    pca_method=os.environ.get("HSPEC_PCA_METHOD", "randomized"),
)
# IPC: multiprocessing.Queue 接收 build job; job payload = (prompt_id, List[HSpecTrajectoryDesc], rewards)
```

- 与 VERL 衔接：在 **`actor_rollout` worker 进程**内，根据本 worker 的 `tp_group_id` 连接对应 shard 的 job queue（同机 Unix socket / Queue），**不经过 driver**。
- Driver 仅当需要全局 barrier 时，通过 `HSpecCoordinator` 查询 `build_done_count`。

**B. 节点亲和 Ray actor（仅当必须用 Ray 管理 build 时）**

```python
HSpecBuildShard.options(
    num_cpus=threads,
    name=f"hspec_build_shard_{node_id}_{sid}",
    resources={f"node:{node_id}": 0.001},  # 强制与 rollout 同节点
).remote(
    shard_id=sid,
    store_root=f"{HSPEC_STORE_DIR}/{node_id}/shard_{sid:03d}",
    ...
)
```

- **入参类型**：`build_from_descs.remote(prompt_id: str, descs: List[HSpecTrajectoryDesc])`。
- **禁止签名**：`build_tables_batch.remote(prompt_data: Dict[str, Dict])` 且 `hidden_states` 为 `List[np.ndarray]`（现 `hspec_table.py:323-338`）。

**共用约束**：

- shard 启动时 `threadpoolctl.threadpool_limits(limits=threads)` 或设置 `OMP_NUM_THREADS`/`MKL_NUM_THREADS`，避免 16 个 worker × 多线程 BLAS 打满 CPU。
- table writer 写 prompt 前估算 `keys_bytes + token_bytes`；超 `HSPEC_TABLE_MAX_BYTES_PER_SHARD` 则跳过或淘汰（§5.3）。

**`init_hspec_tables()` 迁移**：

```python
def init_hspec_coordinator(...) -> None:
    # 仅创建 1 个轻量 Ray actor 或 driver 本地 Coordinator 对象
    # 不再创建 hspec_table_{0..4} 承载 PCA/表数据

def init_hspec_build_shards_on_node(node_id: str) -> None:
    # 在每个 rollout 节点调用；启动 §6.2 A 或 B
```

`main_ppo.py` 中：保留 `init_hspec_coordinator`；**build shard 初始化移到** `megatron_workers.py` / `fsdp_workers.py` 的 rollout worker `__init__`（与 vLLM engine 同进程组/同节点）。

### 6.3 active table mmap 化

当前设计 tip：active table 按 epoch/version 存为连续数组，swap 只更新版本元数据或原子重命名目录。

当前代码问题：

- `HSpecTableGroup._active` 和 `_building` 都是 Python dict，value 是 `PromptTableData`，所有 keys、tokens、offsets 常驻 actor heap。
- [get_active_table_data_batch](vllm_ascend/spec_decode/hspec_table.py:451) 把 table arrays copy 出 actor，再经 Ray 传给 proposer。

落地方案：

- 每个 version 一个目录：

```text
table_store/
  active_version.json
  version_0004/
    shard_000/
      table.bin
      manifest.json
      prompt_index.jsonl
```

- `manifest` 记录每个 prompt 的数组 offset、shape、dtype：

```json
{
  "prompt_id": "p...",
  "version": 4,
  "mean": {"path": "table.bin", "offset": 0, "shape": [4096], "dtype": "float32"},
  "components": {"offset": 16384, "shape": [64, 4096], "dtype": "float32"},
  "keys": {"offset": 1064960, "shape": [12000, 64], "dtype": "float16"},
  "token_buffer": {"offset": 2600000, "shape": [80000], "dtype": "int32"},
  "entry_offset": {"shape": [12000], "dtype": "int32"}
}
```

- `swap()` 不移动大数据，只写 `active_version.json` 或原子替换 `active` symlink/目录指针。

### 6.4 proposer prefetch 不从 Ray actor 拉大 dict

当前设计 tip：在线 proposer prefetch 时不要从 Ray actor 拉大 dict，而是拿到 `{version, prompt_id, offset, shape}` 后本地 mmap 读取。

当前代码问题：

- [HSpecProposer._poll_pending](vllm_ascend/spec_decode/hspec_proposer.py:593) 从 Ray future 得到 `table_data`，再 `_build_cached_table(data)`。
- `_build_cached_table()` 对 `mean/components/keys/rollout_seqs/offsets` 做 numpy copy，再传到 NPU。

落地方案：

- `GlobalHSpecTableGroup.prefetch_batch_async(prompt_ids)` 返回 descriptor future，不返回 arrays。
- `HSpecProposer._build_cached_table()` 改成 `_build_cached_table_from_descriptor(desc)`：

```python
mean_np = np.memmap(desc.mean.path, dtype=np.float32, mode="r", offset=desc.mean.offset, shape=desc.mean.shape)
components_np = np.memmap(...)
keys_np = np.memmap(...)
```

- proposer 本地 cache 仍保留。热路径仍然是 NPU batch projection + `bmm`，不把 mmap 访问放进 `generate_token_ids()` 的 steady-state。
- `_poll_pending()` 中完成 mmap open 和 NPU transfer；如果未 ready，当前 request 返回空 draft，保持 graceful degradation。

**新增 descriptor 类型（与 `HSpecTrajectoryDesc` 区分）**：

```python
@dataclass(frozen=True)
class HSpecArrayView:
    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: str

@dataclass(frozen=True)
class HSpecPromptTableDesc:
    version: int
    prompt_id: str
    shard_id: int
    node_id: str
    mean: HSpecArrayView
    components: HSpecArrayView
    keys: HSpecArrayView
    token_buffer: HSpecArrayView
    entry_rollout_idx: HSpecArrayView
    entry_offset: HSpecArrayView
    n_entries: int
    wnd_size: int
    max_wnd: int
    min_wnd: int
```

- `GlobalHSpecTableGroup.prefetch_batch_async` 返回 `List[Tuple[ObjectRef, List[str]]]`，future 解析为 `(version, Dict[str, HSpecPromptTableDesc | None])`，**禁止** 再返回 `mean/components/keys` ndarray dict（现 `get_active_table_data_batch`，`hspec_table.py:454-482`）。

### 6.5 ZMQ 查询路径处理（现框架遗漏项）

当前 `HSpecTableGroup` 含 ZMQ REP server（`hspec_table.py:666-689`），`GlobalHSpecTableGroup.post_query_batch` 走 ZMQ REQ（`902-938`）。

**事实**：`HSpecProposer` 在线匹配走 **本地 cache**（prefetch 后 NPU 计算），**不调用** `post_query_batch`。

**重构决策**：

| 阶段 | 行为 |
| --- | --- |
| Phase 0–2 | 保留 ZMQ 代码但默认 `HSPEC_ENABLE_ZMQ_QUERY=0`；避免误开双路径 |
| Phase 3+ | 删除 ZMQ server 与 `post_query_batch`；metrics 仍经 `report_online_metrics_async`（小 RPC） |

这样可避免「表已 mmap 化却仍维护 ZMQ 序列化 query」的额外 CPU 与端口占用。

### 6.6 Partial swap 语义（与 §7.3 / §5.3 对齐）

当 `HSPEC_SWAP_PARTIAL_ON_TIMEOUT=1` 且 epoch barrier 超时：

| 对象 | 行为 |
| --- | --- |
| 已完成 build 的 `prompt_id` | 写入 `version_{E+1}/shard_*/`，参与 swap |
| 未完成 build 的 `prompt_id` | **不**进入新 version；下一 epoch 查询仍用 **旧 active**（若有）或 miss |
| Coordinator `active_version` | 仍整体 +1；manifest 中 `prompt_index.jsonl` 只列已完成项 |
| 未完成 desc | 进入 `HSPEC_BUILD_QUEUE` 下一 epoch 或丢弃并计 `hspec/build_timeout_discard` |

**禁止**：partial swap 时把空表或半表当作完整表覆盖旧 active 导致全局 match rate 崩溃——manifest 必须 per-prompt 标记 `complete: true/false`。

## 7. 训练循环调整

### 7.0 现码 step 顺序与目标顺序（框架对照）

**当前** `ray_trainer.py::fit`（HSpec 相关）：

```text
gen → reward → old_log_prob → ref → [critic] → adv
  → 聚合 rollout_hidden_states → build_tables_async
  → update_actor(batch)          # batch 仍可能含 rollout_hidden_states
  → ray.get(hspec_build_tasks)   # hspec_build_wait
```

**Phase 0 目标**：

```text
... → build_tables_async → pop HSpec 大字段 → update_actor → （无 step 内 ray.get）
epoch end → ray.get(epoch_refs) → swap
```

**Phase 1+ 目标**：

```text
gen → reward → pop 无关字段前: 组装 desc+reward → submit build（本节点 queue）
  → old_log_prob → ref → adv → update_actor（batch 无 hspec_desc）
epoch end → barrier → swap → proposer prefetch 新 version
```

**`pop` 时机（修正 §7.4 可执行性）**：

- 必须在 **`update_actor(batch)` 之前** 执行（现码 `update_actor` 在 `1549-1535`，`ray.get` 在 `1542-1545`）。
- 建议在 `build_tables_async` **提交后立刻** pop，不要等到 step 末尾。

### 7.1 step 内只提交 descriptor，不等待完整 build

当前设计 tip：step 内只提交 descriptor，不等待完整 build。

当前代码问题：

- [ray_trainer.py:1542](verl/trainer/ppo/ray_trainer.py:1542) 每个 step 都 `ray.get(ray_hspec_tasks)`。
- 这使 `timing_s/hspec_build_wait` 直接进入训练 step 时延。

落地方案：

- trainer 初始化 `self._hspec_pending_build_refs: list[tuple[int, ObjectRef]] = []`。
- 每个 step 只提交：

```python
refs = self.hspec_tables.build_tables_async(desc_payload)
self._hspec_pending_build_refs.extend((epoch, r) for r in refs)
```

- 不在 step 内 `ray.get`。只用 `ray.wait(timeout=0)` 轮询完成状态，用于 metrics。
- 如果 build queue 超过预算，提交端做 backpressure 或丢弃低优先级 prompt。

### 7.2 build 与 reward/ref/actor update 重叠

当前设计 tip：build shard 在 actor update、reward、ref 等阶段后台运行。

当前代码问题：

- 当前 HSpec build 提交点在 reward、old_logprob、ref、adv 后，actor update 前；且随后 step 内 wait，重叠空间有限。
- reward 只依赖 prompt/response，不依赖 old_logprob/ref。为了让建表尽早开始，可以把 reward_fn 提前。

落地方案：

- 第一阶段低风险改法：保持当前 reward 位置，只删除 step 内 `ray.get`，让 build 与 actor update、下一个 step rollout 重叠。
- 第二阶段重排：generation 后尽早执行 rule-based `compute_reward(batch, reward_fn)`，拿到 sequence reward 后立即提交 HSpec descriptor build；随后再计算 old_logprob、ref、value、advantage。
- 对 `reward_model.launch_reward_fn_async=True` 的场景，build 提交可以等 `ray.get(future_reward)` 之后再 `build_tables_async`（与现 `ray_trainer.py:1329-1331` 一致）；**不可**在仅有 `token_level_scores` 之前提交带 reward 的淘汰策略。
- **Phase 1 默认**：`reward` 仅用于 `PromptTableData.rewards` 字段与淘汰，**不参与** PCA 拟合；故 Phase 2「提前 reward」只为缩短 wall-clock，不改变 PCA 数学。
- **RM 打分训练**：若 reward 来自 RM worker 而非 rule-based，build 提交点必须在 `batch.batch["token_level_scores"]` 写入之后（现逻辑），不得提前到 `gen` 后。

**Driver 提交 build 的两种接线（Phase 1 二选一）**：

| 接线 | 做法 | 适用 |
| --- | --- | --- |
| **推荐：worker 本地 submit** | rollout worker flush 后把 desc 放入本机 build queue；driver **不** 收集 HS | `nnodes>=1`，数据面不出 worker |
| **过渡：driver 转发 desc** | driver 只 `ray.get` 从 rollout 返回的 `List[HSpecTrajectoryDesc]`（体积小），再按 `shard_id` 分组 `build_shard.submit.remote(descs)` | Phase 1 兼容；仍须 **节点亲和**（§2.3.3 B） |

禁止：driver 聚合成 `prompt_build_data["hidden_states"]` 再 `build_tables_async`（现 `ray_trainer.py:1501`）。

### 7.3 epoch 末尾 swap 前等待本 epoch 未完成 build

当前设计 tip：epoch 末尾 swap 前只等待本 epoch 尚未完成的 prompt build。

当前代码问题：

- [ray_trainer.py:1630](verl/trainer/ppo/ray_trainer.py:1630) final step 和 [ray_trainer.py:1644](verl/trainer/ppo/ray_trainer.py:1644) epoch 末尾调用 `swap()`，但当前每 step 已经等待 build 完成。

落地方案：

- 删除 step wait 后，在 epoch 末尾执行：

```python
epoch_refs = [r for e, r in self._hspec_pending_build_refs if e == epoch]
ray.get(epoch_refs)
self.hspec_tables.swap()
self._hspec_pending_build_refs = [(e, r) for e, r in self._hspec_pending_build_refs if e != epoch]
```

- 支持 timeout：

```text
HSPEC_EPOCH_BUILD_BARRIER_TIMEOUT_S
HSPEC_SWAP_PARTIAL_ON_TIMEOUT=1
```

- 若 timeout，coordinator 只 promote 已完成 prompt；未完成 prompt 的 desc 留给后台清理或直接丢弃。这样不会因少数超长 prompt 拖慢整个 epoch。

### 7.4 清理 batch 中的 HSpec 字段

当前设计 tip：生命周期要短。

当前代码问题：

- 即使重构为 descriptor，`batch.non_tensor_batch` 中的 HSpec 字段也不应进入 `update_actor`。

落地方案：

- 在 `ray_trainer.py` 完成 build request 组装后执行：

```python
for key in ("hspec_desc", "rollout_hidden_states", "rollout_hspec_tokens", "hspec_rollout_debug"):
    batch.non_tensor_batch.pop(key, None)
```

- `DataProto.union()` 和 worker dispatch 之后不会再携带 HSpec 数据，避免无效序列化。

## 8. 分阶段迁移路线

### Phase 0：低风险止血

目标：不改变大架构，先减少训练 step 内阻塞和无效数据生命周期。

改动：

- 删除或配置化 [ray.get(ray_hspec_tasks)](verl/trainer/ppo/ray_trainer.py:1544)，改成 epoch barrier。
- 在 HSpec build 提交后从 `batch.non_tensor_batch` 删除 `rollout_hidden_states`。
- validation 禁止 collection。

预期收益：

- 降低 `hspec_build_wait` 对每 step 的直接影响。
- 减少 actor update/ref dispatch 的 non-tensor 携带成本。

### Phase 1：descriptor 替换 DataProto 大矩阵

目标：hidden states 不再进入 Ray object store。

改动：

- 新增 collector mmap writer 和 `hspec_flush_and_get_descriptors()`。
- `vllm_rollout_spmd.py` 改为返回 `hspec_desc`（object ndarray of `HSpecTrajectoryDesc`）。
- `ray_trainer.py`：删除 `prompt_build_data["hidden_states"]` 聚合；改为按 `prompt_id` 分组 desc + 填 `reward`；提交后 **pop** HSpec 字段（§7.4）。
- `build_tables_async` 入参改为 `Dict[str, List[HSpecTrajectoryDesc]]`；build 在 **desc 所在节点** 执行（§2.3、§6.2 A/B）。
- 环境变量：`HSPEC_STORE_DIR`、`HSPEC_TABLE_STORE_DIR`、`NODE_RANK`、`HSPEC_NUM_SHARDS`。

兼容：

- `HSPEC_LEGACY_DATAPROTO_HS=1` 保留旧路径，便于对齐验证。

**门禁**：Phase 1 合并前，在日志中确认 Ray dashboard **Object Store** 无 >1MB 的 HSpec ndarray；driver RSS 不随 `max_response_length` 线性上升。

### Phase 2：流式 PCA 和 mmap table store

目标：消除 `np.concatenate + full SVD` 的内存峰值，table 不再常驻 Ray actor heap。

改动：

- 新增 `hspec_builder.py` 和 `hspec_table_store.py`。
- `fit_pca_multi_sequence()` 保留为 debug reference，新建 tiled covariance/randomized PCA。
- table writer 直接写 versioned mmap store。
- coordinator `prefetch_batch_async()` 返回 table descriptor。

### Phase 3：proposer mmap descriptor prefetch

目标：在线 prefetch 不通过 Ray 传大 arrays。

改动：

- `HSpecProposer._poll_pending()` 消费 descriptor。
- `_build_cached_table_from_descriptor()` mmap 读取 table，转为 worker-local CPU/NPU cache。
- 保留现有 `generate_token_ids()` 热路径，不在 steady-state 做 Ray/ZMQ/文件 I/O。

### Phase 4：预算、淘汰、profiling 完整化

目标：让 HSpec 在长 rollout 和 30B/MoE 训练中可控。

改动：

- collector、build queue、table store 三类预算和 backpressure。
- table prompt 级淘汰策略：LRU、entry count、reward、recent hit rate。
- 新增 metrics：collect bytes、dropped trajectories、build queue lag、PCA time、projection time、table mmap bytes、proposer cache load time。

## 9. 正确性和性能验证

### 9.1 正确性不变量

必须保留以下不变量：

| 不变量 | 检查位置 |
| --- | --- |
| `len(hidden_states) == len(hspec_token_ids)` | collector flush descriptor、build shard 读取 desc、trainer debug |
| prompt_id 由同一份 prompt token ids 生成 | `verl/utils/dataset/rl_dataset.py`（`raw_prompt_ids`）、`vllm_rollout_spmd.py`（`vllm_inputs` / `prompt_id_from_token_ids`） |
| value shift 不重复已接受 token | `hspec_table.py`：`PromptTableData.add_rollout`（`entry_offset` 从 1 起） |
| epoch E build 的 table 只在 epoch E+1 active | trainer epoch barrier + coordinator `active_version` |
| proposer steady-state 不做 Ray get、不做 mmap I/O | `hspec_proposer.py`：`generate_token_ids`（仅读 `_cache` 中已上传 NPU 的 tensor） |
| descriptor 路径只在本节点 build 进程打开 | `HSpecTrajectoryDesc.node_id` 与 build shard 启动节点一致 |
| partial swap 不发布未完成 prompt 的空表 | `manifest` per-prompt `complete` 标志（§6.6） |

### 9.2 性能指标

| 指标 | 目标 |
| --- | --- |
| Ray object store 中 HSpec hidden state bytes | 归零 |
| `timing_s/hspec_build_wait` | step 内接近 0，只在 epoch barrier 出现 |
| trainer 主进程 RSS | 不随 hidden state 总量线性增长 |
| build shard RSS | 受 `tile_rows`、`D^2` 或 `D*r` 预算约束 |
| proposer hit path latency | 不劣化现有 batch projection/match 热路径 |
| NPU kernel queue | build 默认不占用训练 NPU |

### 9.3 A/B 验证策略

- 同一批 prompt 固定 seed，旧路径和新路径 dump table metadata，比较 `prompt_id`、entry count、value shift、token sequence。
- `HSPEC_PCA_METHOD=svd_reference` 与 `covariance/randomized` 比较 match rate、accept length、build time、RSS。
- `HSPEC_LEGACY_DATAPROTO_HS=1/0` 比较 Ray object store、trainer RSS 和 step time。
- `HSPEC_SWAP_PARTIAL_ON_TIMEOUT=0/1` 比较 epoch barrier 和下一 epoch match rate。

## 10. 最终目标状态

最终系统应满足：

- hidden state 是本机数据面，不经过 Ray object store。
- trainer 只处理 descriptor、reward 和控制面元数据。
- PCA build 是受控 CPU 后台任务，内存峰值可由公式估算和配置约束。
- active table 是 mmap store，swap 是元数据原子切换。
- proposer 热路径仍保持当前优势：worker-local cache、NPU batch projection/match、一次 host sync、CPU O(1) draft slice。
- 当 HSpec build 落后或资源不足时，系统降级为少量 prompt miss，而不是拖慢整步训练或压垮 Ray/CPU 内存。

## 11. 方案完备性评估（相对高层次目标）

高层次目标：**高性能、低耗时、低内存开销，在较低 CPU 占用、较少挤占 NPU 的前提下，尽可能提高训练速度与吞吐。**

### 11.1 目标达成度矩阵

| 维度 | 重构后预期 | 仍存在的风险 / 非“完美”点 | 缓解 |
| --- | --- | --- | --- |
| **训练 step 墙钟** | Phase 0 即可去掉 step 内 `hspec_build_wait`；Phase 1+ build 与 update/下一 gen 重叠 | epoch 末 barrier 仍可能等待最慢 prompt；长 rollout 尾部 straggler | `HSPEC_SWAP_PARTIAL_ON_TIMEOUT`、randomized PCA、按 prompt 超时丢弃 |
| **Ray / 内存** | Object store HS 归零；driver RSS 与 step token 数解耦 | Phase 0 未做 descriptor 时仍可能 Plasma 压力 | 强制 Phase 0 pop + Phase 1 门禁 |
| **CPU** | build 限线程 + shard 对齐 TP；采集无 per-step cat 大数组 | 多 shard 并发仍可能占满部分 core；与 ref/logprob 争用 | `HSPEC_BUILD_THREADS_PER_SHARD`、`HSPEC_BUILD_MAX_INFLIGHT` |
| **NPU** | 热路径仅 proposer GEMM；build 默认 CPU | 误开 `HSPEC_BUILD_USE_NPU=1` 或与 decode 叠加大 GEMM | 默认关闭；scheduler 检查 engine 状态 |
| **吞吐（tokens/s）** | 依赖 HSpec 命中率 × accept_length；系统层减少 stall | **算法层** PCA/randomized 近似可能略降 match rate | `HSPEC_PCA_METHOD=covariance` 对照；A/B match rate |
| **多机** | 文档 §2.3 已定义 per-node 表 | **非共享盘时各节点表独立**，全局 prompt 命中语义变化 | 明确 `table_scope`；或 NFS 统一 table_store |
| **工程复杂度** | 四层清晰 | 迁移 Phase 多、需维护 legacy 开关 | Phase 0→3 严格门禁与指标 |

### 11.2 结论：是否“完美优秀”

**结论：方案在系统架构层面优秀且与 tips 一致，但不宜称为“完美”；在按本文档 §2.3 / §6.0–6.6 落地前，仍存在可预见的缺口。**

**优秀之处**：

1. 正确识别并切断 **GB 级数据走 Ray/DataProto** 这一根本矛盾，与 VERL 现码 (`pickle` `DataProto`、`ray_trainer` 聚合) 精准对齐。
2. **热路径 / 冷路径分离**明确：proposer NPU 稳态不变，mmap + 流式 PCA 只影响冷路径。
3. 内存峰值从 **O(ND) 多副本** 变为 **可配置** `O(tile·D + D²)` 或 `O(D·r)`，适合 30B 长 response。
4. Phase 0 可独立交付价值，降低迁移风险。

**尚未覆盖或需实现时证明的缺口**：

1. **多机全局表**：per-node table 与「全局 prompt 共享一张表」语义不等价；多机生产需共享存储或接受命中率折损。
2. **Build 与 rollout 的精确同步**：worker 本地 queue 模式下，epoch barrier 如何等待 **所有节点** 的 shard（需 coordinator 跨节点 `build_done` 聚合）。
3. **bf16 存盘与对齐**：NPU `sample_hidden_states` 常为 bf16，mmap 用 `uint16_raw` 时的数值与 PCA 稳定性需实验验证。
4. **DataProto.repeat/union**：`hspec_desc` 在 `rollout.n>1` 时经 `batch.repeat(..., interleave=True)`（`ray_trainer.py`）后，需在 `protocol.py` 验证 object ndarray 按样本切片仍与每条轨迹一一对应，避免 descriptor 重复或丢失。
5. **效果–性能联合最优**：randomized PCA 与 partial swap 是系统友好型近似，**不保证** RL 收敛指标最优；需 offline A/B。

**总体评价**：满足「**高性能、低内存、少挤占 NPU、训练吞吐导向**」的 **必要条件**；达到「**完美**」还需：多机表语义决策、Phase 0–3 指标门禁达标、以及线上 match rate / step time 的联合验证。建议以 **Phase 0 + Phase 1（nnodes=1）** 为发布基线，再推进 mmap table 与 proposer descriptor prefetch。
