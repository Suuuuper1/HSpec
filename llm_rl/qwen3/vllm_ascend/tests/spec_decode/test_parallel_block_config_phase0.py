import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest

from vllm.config import (
    CacheConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.scheduler import SchedulerConfig
from vllm.config.load import LoadConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models import ModelRegistry
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer


def _base_config(architecture, *, hidden_size=32, num_layers=4):
    return {
        "architectures": [architecture],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 128,
    }


@pytest.fixture
def checkpoint_paths(tmp_path):
    target = tmp_path / "target"
    dflash = tmp_path / "dflash"
    dspark = tmp_path / "dspark"
    target.mkdir()
    dflash.mkdir()
    dspark.mkdir()
    (target / "config.json").write_text(
        json.dumps(_base_config("Qwen3ForCausalLM"))
    )
    dflash_config = _base_config(
        "DFlashDraftModel", num_layers=2
    )
    dflash_config.update(
        {
            "num_target_layers": 4,
            "layer_types": ["full_attention", "full_attention"],
            "dflash_config": {
                "target_layer_ids": [0, 3],
                "mask_token_id": 127,
            },
        }
    )
    (dflash / "config.json").write_text(json.dumps(dflash_config))
    dspark_config = _base_config(
        "Qwen3DSparkModel", num_layers=2
    )
    dspark_config.update(
        {
            "num_target_layers": 4,
            "layer_types": ["full_attention", "full_attention"],
            "target_layer_ids": [0, 3],
            "sample_from_anchor": True,
        }
    )
    (dspark / "config.json").write_text(json.dumps(dspark_config))
    return target, dflash, dspark


def _target(path):
    return ModelConfig(
        model=str(path),
        tokenizer=str(path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=128,
    )


def _spec(paths, method, **kwargs):
    target, dflash, dspark = paths
    return SpeculativeConfig(
        method=method,
        model=str(dflash if method == "dflash" else dspark),
        num_speculative_tokens=kwargs.pop("num_speculative_tokens", 7),
        target_model_config=_target(target),
        target_parallel_config=kwargs.pop(
            "target_parallel_config", ParallelConfig()
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "method,anchor,q,lookahead,additional,offset",
    [
        ("dflash", None, 8, 8, 7, 1),
        ("dspark", None, 7, 7, 6, 0),
        ("dspark", False, 8, 8, 7, 1),
    ],
)
def test_config_geometry_and_independent_loader(
    checkpoint_paths, method, anchor, q, lookahead, additional, offset
):
    spec = _spec(checkpoint_paths, method, sample_from_anchor=anchor)
    config_view = SimpleNamespace(speculative_config=spec)
    assert spec.parallel_query_count == q
    assert spec.max_num_new_slots_for_drafting == additional
    assert spec.draft_sample_offset == offset
    assert VllmConfig.num_lookahead_tokens.fget(config_view) == lookahead
    assert spec.draft_load_config.load_format == "auto"
    assert spec.uses_parallel_block_drafter()
    assert spec.uses_neural_drafter()
    assert spec.needs_aux_hidden_states()
    assert spec.rejection_sample_method == "standard"
    assert spec.parallel_window_fits(128 - q)
    assert not spec.parallel_window_fits(129 - q)


@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("k", [1, 15])
def test_certified_k_boundaries_pass(checkpoint_paths, method, k):
    assert _spec(checkpoint_paths, method, num_speculative_tokens=k)


@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("k", [0, 16])
def test_certified_k_boundaries_fail_with_npu_limit(
    checkpoint_paths, method, k
):
    with pytest.raises(ValueError, match="Ascend NPU certified range"):
        _spec(checkpoint_paths, method, num_speculative_tokens=k)


def test_vllm_scope_gates_and_geometry(checkpoint_paths):
    target, _, _ = checkpoint_paths
    target_config = _target(target)
    parallel = ParallelConfig()
    spec = _spec(checkpoint_paths, "dflash")
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        VllmConfig(
            model_config=target_config,
            parallel_config=parallel,
            speculative_config=spec,
            cache_config=CacheConfig(enable_prefix_caching=True),
        )

    spec = _spec(checkpoint_paths, "dflash")
    config = VllmConfig(
        model_config=target_config,
        parallel_config=parallel,
        speculative_config=spec,
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            async_scheduling=False,
        ),
    )
    assert config.num_speculative_tokens == 7
    assert config.num_lookahead_tokens == 8

    config.model_config.enforce_eager = False
    spec.draft_model_config.enforce_eager = False
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    class NoDeepCopy:
        def __deepcopy__(self, memo):
            del memo
            raise AssertionError("target module registry must not be deep-copied")

    target_layer_sentinel = NoDeepCopy()
    config.compilation_config.static_forward_context["target.attn"] = (
        target_layer_sentinel
    )
    proposer = ParallelBlockProposer.__new__(ParallelBlockProposer)
    proposer.vllm_config = config
    proposer.speculative_config = spec
    draft_config = proposer._draft_vllm_config()
    assert draft_config.model_config.enforce_eager is True
    assert draft_config.compilation_config.mode == CompilationMode.NONE
    assert draft_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert (
        draft_config.compilation_config.static_forward_context
        is config.compilation_config.static_forward_context
    )
    assert (
        draft_config.compilation_config.static_forward_context["target.attn"]
        is target_layer_sentinel
    )
    assert (
        draft_config.compilation_config.enabled_custom_ops
        is not config.compilation_config.enabled_custom_ops
    )
    assert (
        draft_config.compilation_config.custom_ops
        is not config.compilation_config.custom_ops
    )
    assert config.model_config.enforce_eager is False
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL

    with pytest.raises(ValueError, match="async scheduling"):
        VllmConfig(
            model_config=target_config,
            parallel_config=parallel,
            speculative_config=_spec(checkpoint_paths, "dflash"),
            cache_config=CacheConfig(enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(
                max_model_len=128,
                is_encoder_decoder=False,
                async_scheduling=True,
            ),
        )


def test_architecture_detection_hash_and_registry_skeleton(checkpoint_paths):
    target, dflash, dspark = checkpoint_paths
    detected = SpeculativeConfig(
        model=str(dflash),
        num_speculative_tokens=7,
        target_model_config=_target(target),
        target_parallel_config=ParallelConfig(),
    )
    dspark_anchor = _spec(checkpoint_paths, "dspark")
    dspark_shifted = _spec(
        checkpoint_paths, "dspark", sample_from_anchor=False
    )
    assert detected.method == "dflash"
    assert detected.compute_hash() != dspark_anchor.compute_hash()
    assert dspark_anchor.compute_hash() != dspark_shifted.compute_hash()

    for architecture, expected_name, is_placeholder in (
        ("DFlashDraftModel", "DFlashQwen3ForCausalLM", False),
        ("Qwen3DSparkModel", "Qwen3DSparkForCausalLM", True),
    ):
        model_config = detected.draft_model_config
        cls, _ = ModelRegistry.resolve_model_cls([architecture], model_config)
        assert cls.__name__ == expected_name
        assert cls.phase0_placeholder is is_placeholder
        if is_placeholder:
            with pytest.raises(NotImplementedError, match="Phase"):
                cls(vllm_config=SimpleNamespace(), prefix="")


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"num_speculative_tokens": 16}, "1 <= num_speculative_tokens"),
        ({"draft_sample_method": "probabilistic"}, "probabilistic"),
        ({"rejection_sample_method": "legacy"}, "standard"),
        ({"draft_load_config": LoadConfig(load_format="dummy")}, "dummy"),
        ({"sample_from_anchor": True}, "sample_from_anchor"),
    ],
)
def test_unsupported_dflash_options_fail_before_weight_load(
    checkpoint_paths, overrides, error
):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        _spec(checkpoint_paths, "dflash", **overrides)


def test_parallelism_and_checkpoint_mismatch_fail_closed(
    checkpoint_paths, tmp_path
):
    parallel = ParallelConfig(pipeline_parallel_size=2)
    with pytest.raises(ValueError, match="PP/PCP/DCP"):
        _spec(
            checkpoint_paths,
            "dflash",
            target_parallel_config=parallel,
            draft_tensor_parallel_size=1,
        )

    target, _, _ = checkpoint_paths
    bad = tmp_path / "bad-dflash"
    bad.mkdir()
    config = _base_config("DFlashDraftModel", hidden_size=64, num_layers=2)
    config.update(
        {
            "num_target_layers": 4,
            "layer_types": ["full_attention", "full_attention"],
            "dflash_config": {"target_layer_ids": [0, 3], "mask_token_id": 127},
        }
    )
    (bad / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden_size mismatch"):
        SpeculativeConfig(
            method="dflash",
            model=str(bad),
            num_speculative_tokens=7,
            target_model_config=_target(target),
            target_parallel_config=ParallelConfig(),
        )


def test_draft_tp_contract_and_dspark_model_requirement(checkpoint_paths):
    target, _, _ = checkpoint_paths
    target_parallel = ParallelConfig(tensor_parallel_size=4)
    for draft_tp in (1, 4):
        spec = _spec(
            checkpoint_paths,
            "dflash",
            target_parallel_config=target_parallel,
            draft_tensor_parallel_size=draft_tp,
        )
        assert spec.draft_parallel_config.tensor_parallel_size == draft_tp

    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        _spec(
            checkpoint_paths,
            "dflash",
            target_parallel_config=target_parallel,
            draft_tensor_parallel_size=2,
        )
    with pytest.raises(ValueError, match="independent draft checkpoint"):
        SpeculativeConfig(
            method="dspark",
            model=None,
            num_speculative_tokens=7,
            target_model_config=_target(target),
            target_parallel_config=ParallelConfig(),
        )

@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda config: config["dflash_config"].update(mask_token_id=-1), "mask_token_id"),
        (
            lambda config: config["dflash_config"].update(
                target_layer_ids=[0, 9]
            ),
            "outside the target layer range",
        ),
        (
            lambda config: config["dflash_config"].update(
                target_layer_ids=[0, 0]
            ),
            "must be unique",
        ),
        (lambda config: config.update(num_target_layers=5), "num_target_layers"),
        (
            lambda config: config["dflash_config"].update(
                target_layer_ids=[3, 0]
            ),
            "strictly increasing",
        ),
        (
            lambda config: config.update(
                layer_types=["full_attention", "sliding_attention"]
            ),
            "uniform full attention",
        ),
    ],
)
def test_dflash_checkpoint_contract_fields_fail_closed(
    checkpoint_paths, mutate, error
):
    _, dflash, _ = checkpoint_paths
    config_path = dflash / "config.json"
    config = json.loads(config_path.read_text())
    mutate(config)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        _spec(checkpoint_paths, "dflash")


def test_hspec_defaults_and_old_eagle_three_value_abi(checkpoint_paths):
    target, _, _ = checkpoint_paths
    target_config = _target(target)
    hspec = SpeculativeConfig(
        method="hspec",
        num_speculative_tokens=7,
        target_model_config=target_config,
        target_parallel_config=ParallelConfig(),
    )
    assert hspec.method == "hspec"
    assert hspec.draft_model_config is target_config
    assert hspec.rejection_sample_method is None
    assert not hspec.uses_parallel_block_drafter()

    source = inspect.getsource(EagleProposer.prepare_inputs_padded)
    tree = ast.parse(textwrap.dedent(source))
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    tuple_returns = [node.value for node in returns if isinstance(node.value, ast.Tuple)]
    assert tuple_returns
    assert all(len(value.elts) == 3 for value in tuple_returns)

    scheduler_source = inspect.getsource(Scheduler.__init__)
    assert "vllm_config.num_lookahead_tokens" in scheduler_source
    assert "self.num_lookahead_tokens = self.num_spec_tokens" not in scheduler_source

    with pytest.raises(ValueError, match="mutually exclusive"):
        _spec(
            checkpoint_paths,
            "dflash",
            hspec_n_components=32,
        )
