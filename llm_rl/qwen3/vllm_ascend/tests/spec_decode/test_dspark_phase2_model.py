import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash import (
    _validate_dspark_attention_config,
    dspark_aux_capture_layer_ids,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)


def _dspark_config(**overrides):
    values = {
        "vocab_size": 32,
        "mask_token_id": 31,
        "target_layer_ids": [1, 3],
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "full_attention"],
        "sliding_window": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dense_dspark_aux_outputs_translate_to_old_qwen_capture_points():
    config = _dspark_config(target_layer_ids=[1, 9, 17, 25, 33])
    assert dspark_aux_capture_layer_ids(config) == (2, 10, 18, 26, 34)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"mask_token_id": 32}, "mask_token_id"),
        ({"target_layer_ids": [3, 1]}, "ordered set"),
        ({"markov_rank": 0}, "markov_rank"),
        ({"markov_head_type": "future"}, "vanilla"),
        ({"draft_vocab_size": 31}, "equal full vocabulary"),
        ({"layer_types": ["full_attention", "sliding_attention"]}, "uniform"),
    ],
)
def test_dspark_algorithm_affecting_checkpoint_fields_fail_closed(
    override, error
):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        _validate_dspark_attention_config(_dspark_config(**override))


def test_dspark_model_surface_owns_vocab_and_exposes_markov_contract():
    assert Qwen3DSparkForCausalLM.phase0_placeholder is False
    for method in (
        "compute_draft_logits",
        "markov_embed",
        "markov_embed_into",
        "markov_bias",
        "markov_bias_into",
        "map_draft_to_target",
    ):
        assert callable(getattr(Qwen3DSparkForCausalLM, method))


def test_markov_and_confidence_projections_are_explicitly_replicated():
    markov_source = inspect.getsource(DSparkMarkovHead.__init__)
    confidence_source = inspect.getsource(DSparkConfidenceHead.__init__)
    assert "ReplicatedLinear" in markov_source
    assert "disable_tp=True" in markov_source
    assert "ReplicatedLinear" in confidence_source
    assert "disable_tp=True" in confidence_source
    assert issubclass(ReplicatedLinear, torch.nn.Module)


def test_confidence_is_checkpoint_compatible_but_not_a_model_forward_side_effect():
    source = inspect.getsource(Qwen3DSparkForCausalLM.forward)
    assert "confidence" not in source
    assert "compute_confidence" not in inspect.getsource(
        Qwen3DSparkForCausalLM.compute_draft_logits
    )


class _FakeLoadedModel:
    def __init__(self):
        self.loaded = None
        self.built = False

    def load_weights(self, weights):
        self.loaded = list(weights)

    def build_fused_context_kv_buffers(self):
        self.built = True


def _loader_wrapper():
    wrapper = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = _FakeLoadedModel()
    wrapper.lm_head = torch.nn.Module()
    wrapper.lm_head.weight = torch.nn.Parameter(torch.zeros(4, 3))
    return wrapper


def test_dspark_loader_routes_own_vocab_heads_and_skips_training_only_t2d():
    wrapper = _loader_wrapper()
    lm_weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    wrapper.load_weights(
        [
            ("embed_tokens.weight", torch.ones(4, 3)),
            ("markov_head.markov_w1.weight", torch.ones(4, 2)),
            ("confidence_head.proj.weight", torch.ones(1, 5)),
            ("lm_head.weight", lm_weight),
            ("t2d", torch.arange(4)),
        ]
    )
    assert [name for name, _ in wrapper.model.loaded] == [
        "embed_tokens.weight",
        "markov_head.markov_w1.weight",
        "confidence_head.proj.weight",
    ]
    assert torch.equal(wrapper.lm_head.weight, lm_weight)
    assert wrapper.model.built


def test_dspark_loader_fails_on_missing_lm_head_or_reduced_vocab_mapping():
    with pytest.raises(ValueError, match="lm_head"):
        _loader_wrapper().load_weights([("embed_tokens.weight", torch.ones(4, 3))])
    with pytest.raises(NotImplementedError, match="reduced-vocabulary"):
        _loader_wrapper().load_weights(
            [("d2t", torch.arange(4)), ("lm_head.weight", torch.ones(4, 3))]
        )
