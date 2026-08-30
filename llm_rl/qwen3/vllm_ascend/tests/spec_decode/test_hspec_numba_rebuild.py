from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.spec_decode import hspec_proposer


def _cached_table(keys_dtype: np.dtype) -> SimpleNamespace:
    return SimpleNamespace(
        mean_cpu=np.arange(4, dtype=np.float32),
        components_t_cpu=np.ascontiguousarray(
            np.arange(12, dtype=np.float32).reshape(4, 3)
        ),
        keys_cpu=np.ascontiguousarray(
            np.arange(15, dtype=np.float32).reshape(5, 3).astype(keys_dtype)
        ),
        n_entries=5,
    )


def _proposer() -> SimpleNamespace:
    return SimpleNamespace(
        _use_numba_rebuild=True,
        _numba_rebuild_min_rows=0,
        _numba_rebuild_min_elems=0,
        _keys_device_dtype=torch.float32,
        _r1_config=SimpleNamespace(computes_topk=False),
        _record_proposer_metric=lambda *_args, **_kwargs: None,
    )


def _build(proposer: SimpleNamespace, tables: list[SimpleNamespace]):
    return hspec_proposer.HSpecProposer._build_batched_table_tensors(
        proposer,
        tables,
        torch.float32,
        torch.device("cpu"),
    )


def test_numba_array_list_rejects_float16_before_lowering():
    array = np.ones((2, 3), dtype=np.float16)
    assert not hspec_proposer._hspec_numba_array_list_compatible([array])
    assert hspec_proposer._hspec_make_numba_array_list([array]) is None


def test_float16_table_keys_use_equivalent_numpy_rebuild(monkeypatch):
    def unexpected_numba_call(*_args, **_kwargs):
        raise AssertionError("float16 arrays must not enter the Numba kernel")

    monkeypatch.setattr(
        hspec_proposer,
        "_hspec_fill_batched_components_keys_numba",
        unexpected_numba_call,
    )
    proposer = _proposer()
    outputs = _build(proposer, [_cached_table(np.float16)])
    _, components, keys, _, lengths, invalid = outputs

    assert proposer._use_numba_rebuild is True
    torch.testing.assert_close(
        components,
        torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
    )
    torch.testing.assert_close(
        keys,
        torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
    )
    assert lengths.tolist() == [5]
    assert not invalid.any()


@pytest.mark.skipif(
    not hspec_proposer._HSPEC_NUMBA_AVAILABLE,
    reason="Numba is an optional HSpec dependency",
)
def test_float32_table_keys_preserve_numba_fast_path(monkeypatch):
    calls = []
    original = hspec_proposer._hspec_fill_batched_components_keys_numba

    def observed_numba_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        hspec_proposer,
        "_hspec_fill_batched_components_keys_numba",
        observed_numba_call,
    )
    proposer = _proposer()
    outputs = _build(proposer, [_cached_table(np.float32)])

    assert calls == [True]
    assert proposer._use_numba_rebuild is True
    torch.testing.assert_close(
        outputs[2],
        torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
    )


def test_numba_runtime_failure_disables_accelerator_and_falls_back(monkeypatch):
    if not hspec_proposer._HSPEC_NUMBA_AVAILABLE:
        pytest.skip("Numba is an optional HSpec dependency")

    metrics = []
    proposer = _proposer()
    proposer._record_proposer_metric = lambda name, value: metrics.append(
        (name, value)
    )

    def broken_numba_call(*_args, **_kwargs):
        raise RuntimeError("synthetic Numba failure")

    monkeypatch.setattr(
        hspec_proposer,
        "_hspec_fill_batched_components_keys_numba",
        broken_numba_call,
    )
    outputs = _build(proposer, [_cached_table(np.float32)])

    assert proposer._use_numba_rebuild is False
    assert metrics == [("numba_rebuild_runtime_fallback_count", 1)]
    torch.testing.assert_close(
        outputs[2],
        torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
    )
