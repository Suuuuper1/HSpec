from types import SimpleNamespace

import pytest

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_dp_topology import (
    apply_vllm_dp_environment,
    build_topology_manifest,
    build_topology_record,
    predicted_dp_group,
    resolve_vllm_data_parallel_size,
    validate_vllm_dp_layout,
)


@pytest.mark.parametrize(
    "yaml_value,environ,size,source",
    [
        (None, {}, 1, "default"),
        (None, {"VLLM_DP_SIZE": "4"}, 4, "user-env"),
        (
            None,
            {
                "VLLM_DP_SIZE": "8",
                "VERL_VLLM_DP_SIZE_SOURCE": "worker-derived",
            },
            8,
            "worker-derived",
        ),
        (2, {}, 2, "yaml"),
        (2, {"VLLM_DP_SIZE": "2"}, 2, "yaml"),
    ],
)
def test_dp_source_resolution(yaml_value, environ, size, source):
    resolved = resolve_vllm_data_parallel_size(yaml_value, environ=environ)
    assert (resolved.size, resolved.source) == (size, source)
    output = dict(environ)
    apply_vllm_dp_environment(resolved, environ=output)
    assert output["VLLM_DP_SIZE"] == str(size)
    assert output["VERL_VLLM_DP_SIZE_SOURCE"] == source


@pytest.mark.parametrize(
    "yaml_value,environ,error",
    [
        (2, {"VLLM_DP_SIZE": "4"}, "Conflicting"),
        (0, {}, "positive integer"),
        (True, {}, "positive integer"),
        (None, {"VLLM_DP_SIZE": "1.5"}, "positive integer"),
        (
            None,
            {"VLLM_DP_SIZE": "2", "VERL_VLLM_DP_SIZE_SOURCE": "unknown"},
            "Invalid VERL_VLLM_DP_SIZE_SOURCE",
        ),
    ],
)
def test_bad_dp_sources_fail_before_engine(yaml_value, environ, error):
    with pytest.raises(ValueError, match=error):
        resolve_vllm_data_parallel_size(yaml_value, environ=environ)


def test_external_dp_dp_pp_pcp_tp_layout_matches_vllm_order():
    layout = validate_vllm_dp_layout(
        world_size=8,
        vllm_dp_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        rollout_data_parallel_size=1,
    )
    assert layout.external_dp_size == 2
    assert layout.dp_groups == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert predicted_dp_group(layout, 6) == (4, 6)


def test_world_decomposition_and_rollout_dispatch_fail_closed():
    with pytest.raises(ValueError, match="not divisible"):
        validate_vllm_dp_layout(
            world_size=7,
            vllm_dp_size=2,
            tensor_parallel_size=2,
        )
    with pytest.raises(ValueError, match="request/output dispatch DP"):
        validate_vllm_dp_layout(
            world_size=4,
            vllm_dp_size=2,
            tensor_parallel_size=1,
            rollout_data_parallel_size=2,
        )
    non_parallel_layout = validate_vllm_dp_layout(
        world_size=4,
        vllm_dp_size=1,
        tensor_parallel_size=1,
        rollout_data_parallel_size=2,
        require_rollout_dispatch_one=False,
    )
    assert non_parallel_layout.rollout_data_parallel_size == 2


def test_topology_manifest_requires_complete_consistent_engine_records():
    resolved = resolve_vllm_data_parallel_size(2, environ={})
    layout = validate_vllm_dp_layout(
        world_size=4, vllm_dp_size=2, tensor_parallel_size=2
    )
    records = [
        build_topology_record(
            resolved=resolved,
            layout=layout,
            global_rank=rank,
            actual_dp_size=2,
            actual_dp_rank=(rank // 2),
            actual_dp_group_ranks=(rank % 2, rank % 2 + 2),
            method="dflash",
            draft_model_kind="dense",
        )
        for rank in range(4)
    ]
    manifest = build_topology_manifest(records)
    assert manifest["status"] == "PASS"
    assert manifest["all_engine_reflected"] is True
    assert manifest["dp_groups"] == [[0, 2], [1, 3]]
    assert manifest["draft_dp_sync_mode"] == "local_fast_path"

    incomplete = records[:-1]
    with pytest.raises(ValueError, match="each rank once"):
        build_topology_manifest(incomplete)

    overlapping = [dict(record) for record in records]
    overlapping[2]["vllm_dp_group_ranks"] = [1, 2]
    with pytest.raises(ValueError, match="disjoint world partition"):
        build_topology_manifest(overlapping)

    bad_dp_rank = [dict(record) for record in records]
    bad_dp_rank[0]["effective_vllm_dp_rank"] = 2
    with pytest.raises(ValueError, match="rank/group mismatch"):
        build_topology_manifest(bad_dp_rank)


def test_engine_reflection_mismatch_fails():
    resolved = resolve_vllm_data_parallel_size(2, environ={})
    layout = validate_vllm_dp_layout(
        world_size=4, vllm_dp_size=2, tensor_parallel_size=2
    )
    with pytest.raises(RuntimeError, match="DP size"):
        build_topology_record(
            resolved=resolved,
            layout=layout,
            global_rank=0,
            actual_dp_size=1,
        )
    with pytest.raises(RuntimeError, match="DP group"):
        build_topology_record(
            resolved=resolved,
            layout=layout,
            global_rank=0,
            actual_dp_size=2,
            actual_dp_rank=0,
            actual_dp_group_ranks=(0, 1),
        )


def test_rollout_config_distinguishes_dispatch_and_model_internal_dp():
    config = RolloutConfig(name="vllm", vllm_data_parallel_size=2)
    assert config.vllm_data_parallel_size == 2
    with pytest.raises(ValueError, match="positive integer"):
        RolloutConfig(name="vllm", vllm_data_parallel_size=0)

    parallel_kwargs = dict(
        name="vllm",
        speculative_method="dflash",
        speculative_model="/tmp/draft",
        data_parallel_size=2,
        enable_prefix_caching=False,
        parallel_draft_allow_target_eager_experiment=True,
    )
    with pytest.raises(NotImplementedError, match="request/output dispatch DP"):
        RolloutConfig(**parallel_kwargs)


def test_topology_record_accepts_parallel_config_shape():
    resolved = resolve_vllm_data_parallel_size(2, environ={})
    layout = validate_vllm_dp_layout(
        world_size=2, vllm_dp_size=2, tensor_parallel_size=1
    )
    parallel = SimpleNamespace(data_parallel_size=2, data_parallel_rank=1)
    record = build_topology_record(
        resolved=resolved,
        layout=layout,
        global_rank=1,
        actual_dp_size=parallel.data_parallel_size,
        actual_dp_rank=parallel.data_parallel_rank,
        actual_dp_group_ranks=(0, 1),
        method="dspark",
        draft_model_kind="dense",
    )
    assert record["effective_vllm_dp_rank"] == 1
    assert record["draft_model_kind"] == "dense"


def test_topology_does_not_guess_draft_kind_before_engine_reflection():
    resolved = resolve_vllm_data_parallel_size(2, environ={})
    layout = validate_vllm_dp_layout(
        world_size=2, vllm_dp_size=2, tensor_parallel_size=1
    )
    predicted = build_topology_record(
        resolved=resolved, layout=layout, global_rank=0, method="dflash"
    )
    assert predicted["draft_model_kind"] is None
    assert predicted["draft_dp_sync_mode"] is None

    reflected = build_topology_record(
        resolved=resolved,
        layout=layout,
        global_rank=0,
        actual_dp_size=2,
        actual_dp_rank=0,
        actual_dp_group_ranks=(0, 1),
        method="dflash",
        draft_model_kind="moe",
    )
    assert reflected["draft_model_kind"] == "moe"
    assert reflected["draft_dp_sync_mode"] == "cpu_group_max_pad"
    with pytest.raises(ValueError, match="only valid"):
        build_topology_record(
            resolved=resolved,
            layout=layout,
            global_rank=0,
            draft_model_kind="dense",
        )
