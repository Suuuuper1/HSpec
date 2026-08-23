import inspect
from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
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
