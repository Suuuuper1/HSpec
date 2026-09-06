# Qwen3 DFlash/DSpark baseline-controlled experiments

The control is `run_qwen3_baseline_eager_tq2.sh`. The six executable entries are:

| Method | Target mode | Script | vLLM ABI value |
|---|---|---|---|
| DFlash | eager | `run_qwen3_dflash_eager_tq2.sh` | no graph / `enforce_eager=True` |
| DFlash | PIECEWISE | `run_qwen3_dflash_piecewise_tq2.sh` | `PIECEWISE` |
| DFlash | FULL_GRAPH_DECODE | `run_qwen3_dflash_full_graph_decode_tq2.sh` | `FULL_DECODE_ONLY` |
| DSpark | eager | `run_qwen3_dspark_eager_tq2.sh` | no graph / `enforce_eager=True` |
| DSpark | PIECEWISE | `run_qwen3_dspark_piecewise_tq2.sh` | `PIECEWISE` |
| DSpark | FULL_GRAPH_DECODE | `run_qwen3_dspark_full_graph_decode_tq2.sh` | `FULL_DECODE_ONLY` |

`FULL_GRAPH_DECODE` is an experiment label; vLLM 0.14.1 calls the corresponding mode `FULL_DECODE_ONLY`. It captures uniform decode batches as full ACL graphs and leaves mixed/prefill batches eager. The DFlash/DSpark draft itself remains eager in all six experiments.

All baseline training/data/sampling values are copied into one shared `.bash` implementation so the entries cannot drift. The necessary differences are:

- method/checkpoint and probabilistic fixed-K=7 proposal;
- explicit Verl dispatch DP=1, model-internal vLLM DP=16, target TP1/PP1 and draft TP1;
- target graph mode; eager requires the explicit target-eager experiment opt-in;
- `gpu_memory_utilization=0.72` instead of baseline `0.83`, because the dense draft and probability workspace coexist with target KV cache; the same 0.72 is used by all six SD arms;
- AIV HCCL expansion and non-ephemeral method-specific ports for graph capture.

The actor/ref training topology, batch size 32, rollout n=16, dataset fraction 0.005, prompt/response lengths, learning parameters, task queue mode, sampling and total epochs remain baseline-identical. Phase-3 lifecycle profiling, DP trace and proposer performance profiling are disabled to avoid measurement overhead.

Use `dry-run` first, for example:

```bash
bash scripts/run_qwen3_dflash_piecewise_tq2.sh dry-run
bash scripts/run_qwen3_dspark_full_graph_decode_tq2.sh dry-run
```

The real run refuses to reuse an output directory or share the 16-NPU lock. Graph runs validate the 32-port HCCL range before Ray starts. Each output contains `comparison_manifest.txt`, binding the baseline, entrypoint and common launcher SHA256, method/checkpoint pair id, graph mode and exact argv. After training, the launcher requires logs proving the requested target mode and eager draft were actually resolved; a silent graph fallback returns a nonzero exit code.

Override checkpoint paths only with the corresponding variables:

```bash
export DFLASH_DRAFT_MODEL=/absolute/path/to/qualified-dflash
export DSPARK_DRAFT_MODEL=/absolute/path/to/qualified-dspark
```

Changing the checkpoint, 0.72 memory budget, batch/data/sampling fields, K, DP/TP/PP, HCCL mode or graph capture policy creates a different experiment and must not be pooled with this comparison.
