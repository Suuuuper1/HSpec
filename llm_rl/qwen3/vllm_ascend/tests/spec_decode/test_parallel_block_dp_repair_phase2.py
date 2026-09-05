from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.spec_decode.parallel_block_proposer as parallel_block
from vllm_ascend.spec_decode.parallel_block_proposer import (
    ParallelBlockDPQualification,
    ParallelBlockProposer,
    validate_parallel_block_dp_qualification,
)
from vllm_ascend.spec_decode.parallel_draft_metrics import (
    dp_repair_capability_manifest,
    phase5_capability_manifest,
)
from vllm_ascend.worker.worker import NPUWorker


def _config(
    *,
    method="dflash",
    proposal="greedy",
    dp_size=2,
    dp_rank=0,
    target_is_moe=True,
    target_vocab_size=151936,
    draft_vocab_size=151936,
    internal_draft_vocab_size=None,
):
    target_parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    draft_parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
    )
    spec = SimpleNamespace(
        method=method,
        draft_sample_method=proposal,
        draft_probability_max_bytes=(256 * 1024 * 1024 if proposal == "probabilistic" else None),
        rejection_sample_method="standard",
        num_speculative_tokens=7,
        target_parallel_config=target_parallel,
        draft_parallel_config=draft_parallel,
        target_model_config=SimpleNamespace(
            get_vocab_size=lambda: target_vocab_size
        ),
        draft_model_config=SimpleNamespace(
            is_moe=False,
            get_vocab_size=lambda: draft_vocab_size,
            hf_text_config=SimpleNamespace(
                draft_vocab_size=(
                    draft_vocab_size
                    if internal_draft_vocab_size is None
                    else internal_draft_vocab_size
                )
            ),
        ),
        enforce_eager=True,
        disable_padded_drafter_batch=False,
        parallel_draft_incremental_context_kv=False,
        parallel_draft_dynamic_k=False,
        dspark_draft_topk=None,
        parallel_draft_profile_enabled=False,
    )
    return SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            is_moe_model=target_is_moe,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        lora_config=None,
    )


@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("proposal", ["greedy", "probabilistic"])
@pytest.mark.parametrize("dp_size", [1, 2, 8])
def test_exact_gate_accepts_certified_surface(method, proposal, dp_size):
    config = _config(
        method=method,
        proposal=proposal,
        dp_size=dp_size,
        target_is_moe=dp_size > 1,
    )
    result = validate_parallel_block_dp_qualification(
        config, environ={"VLLM_DP_SIZE": str(dp_size)}
    )
    assert result == ParallelBlockDPQualification(
        method=method,
        proposal=proposal,
        requested_dp_size=dp_size,
        effective_dp_size=dp_size,
        effective_dp_rank=0,
        draft_model_kind="dense",
        draft_dp_sync_mode="local_fast_path",
    )


def _set_path(owner, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)


@pytest.mark.parametrize(
    "path,value,error",
    [
        ("speculative_config.method", "eagle", "method='eagle'"),
        ("speculative_config.draft_sample_method", "random", "draft_sample_method"),
        ("speculative_config.draft_model_config.is_moe", True, "draft model is MoE"),
        (
            "speculative_config.draft_parallel_config.enable_expert_parallel",
            True,
            "draft expert parallelism",
        ),
        (
            "speculative_config.draft_parallel_config.tensor_parallel_size",
            2,
            "draft tensor_parallel_size",
        ),
        (
            "speculative_config.target_parallel_config.pipeline_parallel_size",
            2,
            "pipeline_parallel_size",
        ),
        (
            "speculative_config.draft_parallel_config.pipeline_parallel_size",
            2,
            "draft pipeline_parallel_size",
        ),
        (
            "speculative_config.target_parallel_config.prefill_context_parallel_size",
            2,
            "prefill_context_parallel_size",
        ),
        (
            "speculative_config.target_parallel_config.decode_context_parallel_size",
            2,
            "decode_context_parallel_size",
        ),
        ("parallel_config.is_moe_model", False, "DP>1 target"),
        ("scheduler_config.async_scheduling", True, "async_scheduling"),
        ("speculative_config.enforce_eager", False, "draft eager"),
        ("speculative_config.num_speculative_tokens", 0, "fixed K"),
        ("speculative_config.disable_padded_drafter_batch", True, "padded drafter"),
        (
            "speculative_config.parallel_draft_incremental_context_kv",
            True,
            "incremental context KV",
        ),
        ("speculative_config.parallel_draft_dynamic_k", True, "dynamic K"),
        ("speculative_config.dspark_draft_topk", 128, "top-k"),
        ("cache_config.enable_prefix_caching", True, "prefix caching"),
        ("lora_config", object(), "LoRA"),
        ("speculative_config.rejection_sample_method", "legacy", "standard rejection"),
    ],
)
def test_exact_gate_rejects_unsupported_surface(path, value, error):
    config = _config()
    _set_path(config, path, value)
    with pytest.raises(NotImplementedError, match=error) as caught:
        validate_parallel_block_dp_qualification(
            config, environ={"VLLM_DP_SIZE": "2"}
        )
    assert "Certified alternatives" in str(caught.value)


def test_probabilistic_restrictions_and_requested_effective_mismatch():
    config = _config(proposal="probabilistic")
    config.speculative_config.target_parallel_config.tensor_parallel_size = 2
    config.speculative_config.draft_probability_max_bytes = None
    with pytest.raises(NotImplementedError, match="target TP is not 1"):
        validate_parallel_block_dp_qualification(
            config, environ={"VLLM_DP_SIZE": "2"}
        )
    with pytest.raises(ValueError, match="Requested/effective"):
        validate_parallel_block_dp_qualification(
            _config(), environ={"VLLM_DP_SIZE": "4"}
        )
    greedy = _config()
    greedy.speculative_config.draft_probability_max_bytes = 1024
    with pytest.raises(NotImplementedError, match="greedy proposal"):
        validate_parallel_block_dp_qualification(
            greedy, environ={"VLLM_DP_SIZE": "2"}
        )


def test_gate_rejects_noncanonical_dp_request_and_vocab_mismatch():
    with pytest.raises(ValueError, match="canonical positive integer"):
        validate_parallel_block_dp_qualification(
            _config(), environ={"VLLM_DP_SIZE": "02"}
        )
    with pytest.raises(NotImplementedError, match="vocabulary mismatch"):
        validate_parallel_block_dp_qualification(
            _config(draft_vocab_size=32000),
            environ={"VLLM_DP_SIZE": "2"},
        )


def test_gate_runs_before_eagle_or_model_allocation(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(parallel_block.EagleProposer, "__init__", fail_if_called)
    config = _config(target_is_moe=False)
    with pytest.raises(NotImplementedError, match="DP>1 target"):
        ParallelBlockProposer(config, device=SimpleNamespace())
    assert called is False


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_phase2_capability_is_self_describing_but_not_prematurely_certified(method):
    config = _config(method=method, proposal="probabilistic")
    spec = config.speculative_config
    repair = dp_repair_capability_manifest(
        spec,
        config,
        draft_model_kind="dense",
        requested_vllm_dp_size=2,
        production_dp_gt1_enabled=True,
    )
    assert repair["schema_version"] == "dflash-dspark.dp-repair-capability.v2"
    assert repair["requested_vllm_dp_size"] == 2
    assert repair["effective_vllm_dp_size"] == 2
    assert repair["draft_dp_sync_mode"] == "local_fast_path"
    assert repair["production_dp_gt1_enabled"] is True
    assert repair["certification_state"] == "phase2_candidate_r1_r2_required"
    compatibility = phase5_capability_manifest(
        spec,
        vllm_config=config,
        draft_model_kind="dense",
        certification_state="phase2_candidate_open",
    )
    assert compatibility["schema_version"] == "dflash-dspark.phase5-capability.v1"
    assert compatibility["dp_extension"]["certification_state"] == "phase2_candidate_open"


def test_dflash_mapped_reduced_vocab_capability_is_described_exactly():
    config = _config(method="dflash", internal_draft_vocab_size=32000)
    spec = config.speculative_config
    manifest = dp_repair_capability_manifest(
        spec,
        config,
        draft_model_kind="dense",
        requested_vllm_dp_size=2,
        production_dp_gt1_enabled=True,
    )

    vocabulary = manifest["certified_candidate"]["vocabulary"]
    assert vocabulary == {
        "internal_draft_vocab_size": 32000,
        "output_vocab_size": 151936,
        "target_vocab_size": 151936,
        "target_probability_vocab": True,
        "mapped_reduced_vocab": True,
        "mode": "mapped_reduced",
    }
    assert manifest["certified_candidate"]["full_vocab"] is True
    assert manifest["unsupported_fail_closed"]["dspark_topk"] is True
    assert manifest["unsupported_fail_closed"]["unmapped_reduced_vocab"] is True


def test_phase1_capability_schema_remains_available_for_frozen_analyzers():
    config = _config()
    manifest = dp_repair_capability_manifest(
        config.speculative_config, config, draft_model_kind="dense"
    )
    assert manifest["schema_version"] == "dflash-dspark.dp-repair-capability.v1"
    assert manifest["production_dp_gt1_enabled"] is False
    assert "requested_vllm_dp_size" not in manifest


def test_named_worker_rpc_returns_bounded_parallel_draft_state(monkeypatch):
    resets = []
    monkeypatch.setattr(torch.npu, "reset_peak_memory_stats", lambda: resets.append(1))
    qualification = ParallelBlockDPQualification(
        method="dflash",
        proposal="probabilistic",
        requested_dp_size=2,
        effective_dp_size=2,
        effective_dp_rank=1,
        draft_model_kind="dense",
        draft_dp_sync_mode="local_fast_path",
    )
    drafter = SimpleNamespace(
        _dp_qualification=qualification,
        flush_observability_metrics=lambda: {"phase5": {}, "dp": {}},
    )
    runner = SimpleNamespace(
        drafter=drafter,
        draft_probability_cache_snapshot=lambda: {"enabled": True},
    )
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = runner
    worker.rank = 1
    worker.local_rank = 1

    pre = worker.get_parallel_draft_worker_state(
        "pre_decode", reset_peak_memory=True
    )
    post = worker.get_parallel_draft_worker_state(
        "post_decode", flush_metrics=True
    )

    assert resets == [1]
    assert pre["qualification"] == {
        "method": "dflash",
        "proposal": "probabilistic",
        "requested_dp_size": 2,
        "effective_dp_size": 2,
        "effective_dp_rank": 1,
        "draft_model_kind": "dense",
        "draft_dp_sync_mode": "local_fast_path",
    }
    assert pre["draft_observability"] is None
    assert post["draft_observability"] == {"phase5": {}, "dp": {}}
    assert post["draft_probability_cache"] == {"enabled": True}


def test_named_worker_rpc_rejects_wrong_stage_and_missing_drafter():
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = SimpleNamespace()
    with pytest.raises(ValueError, match="unknown worker observation stage"):
        worker.get_parallel_draft_worker_state("unknown")
    with pytest.raises(RuntimeError, match="parallel-block drafter"):
        worker.get_parallel_draft_worker_state("pre_decode")
