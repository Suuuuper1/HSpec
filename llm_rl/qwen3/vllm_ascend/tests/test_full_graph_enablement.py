import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.compilation.inductor_pass import pass_context
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.config.utils import Range

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.compilation.compiler_interface import fusion_pass_compile
from vllm_ascend.compilation.npu_graph_ex_pass_manager import (
    NpuGraphEXPassManager,
)
from vllm_ascend.utils import COMPILATION_PASS_KEY
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize(
    ("graph_mode", "compilation_mode", "enforce_eager", "expected"),
    [
        (CUDAGraphMode.NONE, CompilationMode.VLLM_COMPILE, False, False),
        (CUDAGraphMode.PIECEWISE, CompilationMode.NONE, False, False),
        (CUDAGraphMode.PIECEWISE, CompilationMode.VLLM_COMPILE, False, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, CompilationMode.NONE, False, True),
        (CUDAGraphMode.FULL, CompilationMode.VLLM_COMPILE, False, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, CompilationMode.VLLM_COMPILE, True, False),
    ],
)
def test_aclgraph_enablement_matches_full_vs_piecewise_semantics(
    graph_mode, compilation_mode, enforce_eager, expected
):
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=graph_mode,
            mode=compilation_mode,
        ),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
    )
    assert NPUModelRunner._use_aclgraph(runner) is expected


def test_aclgraph_replay_synchronizes_only_the_current_stream():
    source = inspect.getsource(ACLGraphWrapper.__call__)
    assert "torch.npu.current_stream().synchronize()" in source
    assert "torch.npu.synchronize()" not in source
    assert "ACLGraph replay synchronization scope: current_stream" in source


def test_npu_graph_ex_fusion_manager_recompiles_graph_module():
    graph = torch.fx.symbolic_trace(lambda value: value + 1)
    inputs = [torch.tensor([1.0])]
    compile_range = Range(1, 1)

    with pass_context(compile_range):
        compiled, handle = fusion_pass_compile(
            graph,
            inputs,
            {COMPILATION_PASS_KEY: NpuGraphEXPassManager()},
            compile_range,
        )

    assert compiled is not None
    assert handle is None
    assert torch.equal(compiled(*inputs), torch.tensor([2.0]))
