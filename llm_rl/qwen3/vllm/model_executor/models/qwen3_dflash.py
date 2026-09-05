# SPDX-License-Identifier: Apache-2.0
"""Current-native Qwen3 DFlash draft model for the vLLM 0.14.1 ABI.

The draft transformer consumes a parallel query block. Context K/V is
projected from target auxiliary hidden states once, then the Ascend proposer
writes it into this model's paged caches before the query forward.
"""

import io
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import PPMissingLayer, get_draft_quant_config, maybe_prefix

logger = init_logger(__name__)

_FULL_ATTENTION = "full_attention"
_SLIDING_ATTENTION = "sliding_attention"
_MASK_EMBEDDING_FILENAME = "mask_embedding.pt"
_ALLOWED_DFLASH_CONFIG_FIELDS = {
    "causal",
    "mask_token_id",
    "target_layer_ids",
    "use_aux_hidden_state",
}


def _validate_parallel_attention_surface(
    config: Qwen3Config,
    method: str,
    *,
    allow_uniform_causal_sliding: bool = False,
) -> None:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"{method} requires one layer_types entry per draft transformer layer"
        )
    unique_layer_types = set(layer_types)
    if unique_layer_types == {_FULL_ATTENTION}:
        if getattr(config, "sliding_window", None) is not None:
            raise ValueError(
                f"{method} full-attention layers require sliding_window=None"
            )
        return
    if (
        allow_uniform_causal_sliding
        and unique_layer_types == {_SLIDING_ATTENTION}
    ):
        sliding_window = getattr(config, "sliding_window", None)
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError(
                f"{method} sliding attention requires a positive sliding_window"
            )
        return
    if len(unique_layer_types) != 1:
        raise NotImplementedError(
            f"{method} supports one uniform attention KV group "
            f"only; got layer_types={layer_types!r}"
        )
    raise NotImplementedError(
        f"{method} does not support layer type {next(iter(unique_layer_types))!r}"
    )


def _resolve_dflash_attention(config: Qwen3Config) -> tuple[int | None, bool]:
    """Resolve the uniform attention mode supported by the old eager ABI."""
    layer_types = getattr(config, "layer_types", None) or []
    raw = getattr(config, "dflash_config", None) or {}
    causal = raw.get("causal", False)
    if layer_types and set(layer_types) == {_SLIDING_ATTENTION}:
        return getattr(config, "sliding_window", None), causal
    return None, causal


def _expand_draft_logits_to_target(
    logits: torch.Tensor,
    target_ids: torch.Tensor | None,
    target_vocab_size: int,
) -> torch.Tensor:
    """Scatter reduced-vocabulary logits into target-vocabulary space."""
    if target_ids is None:
        return logits
    if target_ids.ndim != 1 or target_ids.numel() != logits.shape[-1]:
        raise ValueError("DFlash target-id mapping does not match draft logits")
    target_logits = logits.new_full(
        (*logits.shape[:-1], target_vocab_size), float("-inf")
    )
    target_logits[..., target_ids] = logits
    return target_logits


def _missing_packed_shards(
    parameter_names: Iterable[str],
    loaded_shards: dict[str, set[str | int]],
) -> dict[str, list[str | int]]:
    missing: dict[str, list[str | int]] = {}
    for name in parameter_names:
        required: set[str | int] | None = None
        if ".qkv_proj." in name:
            required = {"q", "k", "v"}
        elif ".gate_up_proj." in name:
            required = {0, 1}
        if required is None:
            continue
        absent = required - loaded_shards.get(name, set())
        if absent:
            missing[name] = sorted(absent, key=str)
    return missing


def dflash_aux_capture_layer_ids(config: Qwen3Config) -> tuple[int, ...]:
    """Translate HF output-layer ids to old-Qwen pre-layer capture points."""
    raw = getattr(config, "dflash_config", None) or {}
    layer_ids = raw.get("target_layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError("DFlash requires dflash_config.target_layer_ids")
    return tuple(int(layer_id) + 1 for layer_id in layer_ids)


def dspark_aux_capture_layer_ids(config: Qwen3Config) -> tuple[int, ...]:
    """Translate dense-DSpark target outputs to old-Qwen capture points."""
    layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(config, "target_layer_ids", None)
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError("DSpark requires target_layer_ids")
    return tuple(int(layer_id) + 1 for layer_id in layer_ids)


def dflash_target_rope_is_neox_style(target_model: nn.Module) -> bool | None:
    """Read the target RoPE layout; the DFlash checkpoint does not store it."""
    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    for module in language_model.modules():
        style = getattr(module, "is_neox_style", None)
        if isinstance(style, bool):
            return style
        rotary = getattr(module, "rotary_emb", None)
        style = getattr(rotary, "is_neox_style", None)
        if isinstance(style, bool):
            return style
    return None


def _validate_dflash_attention_config(config: Qwen3Config) -> None:
    raw = getattr(config, "dflash_config", None)
    if not isinstance(raw, dict):
        raise ValueError("DFlash requires an object-valued dflash_config")
    unknown = set(raw) - _ALLOWED_DFLASH_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "Unsupported algorithm-affecting DFlash config fields for Phase 1: "
            f"{sorted(unknown)}"
        )
    if raw.get("use_aux_hidden_state", True) is not True:
        raise NotImplementedError("Phase 1 DFlash requires auxiliary target hidden states")
    mask_token_id = raw.get("mask_token_id")
    if not isinstance(mask_token_id, int) or not 0 <= mask_token_id < config.vocab_size:
        raise ValueError(
            f"DFlash mask_token_id must be inside [0, {config.vocab_size}), "
            f"got {mask_token_id!r}"
        )
    layer_ids = raw.get("target_layer_ids")
    if (
        not isinstance(layer_ids, (list, tuple))
        or not layer_ids
        or tuple(layer_ids) != tuple(sorted(set(layer_ids)))
        or any(not isinstance(layer_id, int) or layer_id < 0 for layer_id in layer_ids)
    ):
        raise ValueError(
            "DFlash target_layer_ids must be a non-empty ordered set of "
            f"non-negative integers, got {layer_ids!r}"
        )
    causal = raw.get("causal", False)
    if not isinstance(causal, bool):
        raise ValueError(f"dflash_config.causal must be bool, got {causal!r}")
    _validate_parallel_attention_surface(
        config, "DFlash", allow_uniform_causal_sliding=True
    )
    layer_types = set(config.layer_types)
    if causal != (layer_types == {_SLIDING_ATTENTION}):
        raise NotImplementedError(
            "DFlash old eager ABI supports only non-causal full attention or "
            "causal uniform sliding attention"
        )
    draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
    if (
        isinstance(draft_vocab_size, bool)
        or not isinstance(draft_vocab_size, int)
        or not 0 < draft_vocab_size <= config.vocab_size
    ):
        raise ValueError(
            "DFlash draft_vocab_size must be in "
            f"[1, {config.vocab_size}], got {draft_vocab_size!r}"
        )


def _validate_dspark_attention_config(config: Qwen3Config) -> None:
    mask_token_id = getattr(config, "mask_token_id", None)
    if not isinstance(mask_token_id, int) or not 0 <= mask_token_id < config.vocab_size:
        raise ValueError(
            f"DSpark mask_token_id must be inside [0, {config.vocab_size}), "
            f"got {mask_token_id!r}"
        )
    layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(config, "target_layer_ids", None)
    if (
        not isinstance(layer_ids, (list, tuple))
        or not layer_ids
        or tuple(layer_ids) != tuple(sorted(set(layer_ids)))
        or any(not isinstance(layer_id, int) or layer_id < 0 for layer_id in layer_ids)
    ):
        raise ValueError(
            "DSpark target_layer_ids must be a non-empty ordered set of "
            f"non-negative integers, got {layer_ids!r}"
        )
    markov_rank = getattr(config, "markov_rank", None)
    if not isinstance(markov_rank, int) or markov_rank <= 0:
        raise ValueError(f"DSpark markov_rank must be positive, got {markov_rank!r}")
    markov_head_type = getattr(config, "markov_head_type", "vanilla")
    if markov_head_type != "vanilla":
        raise NotImplementedError(
            f"Phase 2 only supports DSpark markov_head_type='vanilla', got "
            f"{markov_head_type!r}"
        )
    for field in ("enable_confidence_head", "confidence_head_with_markov"):
        value = getattr(config, field, False)
        if not isinstance(value, bool):
            raise ValueError(f"DSpark {field} must be bool, got {value!r}")
    draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
    if draft_vocab_size != config.vocab_size:
        raise NotImplementedError(
            "Phase 2 DSpark requires an equal full vocabulary; reduced-vocab "
            "mapping is deferred with probabilistic proposal"
        )
    _validate_parallel_attention_surface(config, "DSpark")


class DFlashQwen3Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int,
        head_dim: int | None,
        rms_norm_eps: float,
        attention_bias: bool,
        sliding_window: int | None,
        causal: bool,
        is_neox_style: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError(f"num_attention_heads={num_heads} is not divisible by TP={tp_size}")
        if num_kv_heads >= tp_size and num_kv_heads % tp_size:
            raise ValueError(f"num_key_value_heads={num_kv_heads} is not divisible by TP={tp_size}")
        if num_kv_heads < tp_size and tp_size % num_kv_heads:
            raise ValueError(f"TP={tp_size} cannot replicate num_key_value_heads={num_kv_heads}")

        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.causal = causal

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            is_neox_style=is_neox_style,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self._rope_impl: Callable | None = None

    def set_rope_impl(self, implementation: Callable) -> None:
        self._rope_impl = implementation

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.view(*q_shape[:-1], -1, self.head_dim)).view(q_shape)
        k = self.k_norm(k.view(*k_shape[:-1], -1, self.head_dim)).view(k_shape)
        if self._rope_impl is None:
            q, k = self.rotary_emb(positions, q, k)
        else:
            self._rope_impl(
                positions,
                q,
                k,
                self.rotary_emb.head_size,
                self.rotary_emb.cos_sin_cache,
                self.rotary_emb.is_neox_style,
            )
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        set_default_rope_theta(config, default_theta=1000000)
        sliding_window, causal = _resolve_dflash_attention(config)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=getattr(config, "is_neox_style", True),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}
)
class DFlashQwen3Model(nn.Module):
    parallel_draft_method = "dflash"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        spec_config = vllm_config.speculative_config
        if spec_config is None or spec_config.draft_model_config is None:
            raise ValueError("DFlash model construction requires speculative draft config")
        self.config = spec_config.draft_model_config.hf_config
        if self.parallel_draft_method == "dspark":
            _validate_dspark_attention_config(self.config)
        else:
            _validate_dflash_attention_config(self.config)
        self.quant_config = get_draft_quant_config(vllm_config)
        if self.quant_config is not None:
            raise NotImplementedError(
                "Phase 1 fused Context KV requires an unquantized DFlash checkpoint"
            )
        self.vocab_size = self.config.vocab_size
        self.aux_capture_layer_ids = (
            dspark_aux_capture_layer_ids(self.config)
            if self.parallel_draft_method == "dspark"
            else dflash_aux_capture_layer_ids(self.config)
        )
        draft_vocab_size = getattr(
            self.config, "draft_vocab_size", self.config.vocab_size
        )
        # Full-vocabulary DFlash retains the original target-sharing path.
        # Reduced-vocabulary DFlash checkpoints own a full target-vocabulary
        # embedding because query anchors/masks are target token ids.
        if (
            self.parallel_draft_method == "dflash"
            and draft_vocab_size < self.config.vocab_size
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        raw_dflash = getattr(self.config, "dflash_config", None) or {}
        self.mask_token_id = (
            getattr(self.config, "mask_token_id", None)
            if self.parallel_draft_method == "dspark"
            else raw_dflash.get("mask_token_id")
        )
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False
        self.layers = nn.ModuleList(
            DFlashQwen3DecoderLayer(
                config=self.config,
                cache_config=vllm_config.cache_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}"),
            )
            for layer_idx in range(self.config.num_hidden_layers)
        )
        fc_input_size = len(self.aux_capture_layer_ids) * self.config.hidden_size
        self.fc = ReplicatedLinear(
            input_size=fc_input_size,
            output_size=self.config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.hidden_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self._max_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._context_rms_norm_impl: Callable | None = None
        self._context_rope_impl: Callable | None = None
        self._fused_context_kv_ready = False

    def set_context_rms_norm_impl(self, implementation: Callable) -> None:
        """Inject a backend-specific allocation-free RMSNorm implementation."""
        self._context_rms_norm_impl = implementation

    def set_context_rope_impl(self, implementation: Callable) -> None:
        """Inject RoPE that does not consume target-runner global slices."""
        self._context_rope_impl = implementation
        for layer in self.layers:
            layer.self_attn.set_rope_impl(implementation)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed_tokens, PPMissingLayer):
            raise RuntimeError("DFlash target embedding was not injected after load")
        embeddings = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding:
            mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeddings = torch.where(mask, self.mask_embedding.to(embeddings.dtype), embeddings)
        return embeddings

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.fc.input_size:
            raise ValueError(
                f"DFlash FC expects aux width {self.fc.input_size}, got "
                f"{hidden_states.shape[-1]} for capture layers {self.aux_capture_layer_ids}"
            )
        return self.fc(hidden_states)

    def build_fused_context_kv_buffers(self) -> None:
        attentions = [layer.self_attn for layer in self.layers]
        if not attentions:
            raise ValueError("DFlash checkpoint has no transformer layers")
        first = attentions[0]
        has_bias = first.qkv_proj.bias is not None
        if any((attn.qkv_proj.bias is not None) != has_bias for attn in attentions):
            raise ValueError("DFlash K/V projection bias must be uniform across layers")
        for attn in attentions[1:]:
            if (
                attn.kv_size != first.kv_size
                or attn.head_dim != first.head_dim
                or attn.num_kv_heads != first.num_kv_heads
                or attn.rotary_emb.is_neox_style != first.rotary_emb.is_neox_style
                or attn.rotary_emb.head_size != first.rotary_emb.head_size
            ):
                raise ValueError("DFlash fused Context KV requires uniform attention layers")

        fused_kv_weight = torch.cat(
            [attn.qkv_proj.weight[attn.q_size :] for attn in attentions], dim=0
        ).contiguous()
        fused_kv_bias = (
            torch.cat([attn.qkv_proj.bias[attn.q_size :] for attn in attentions], dim=0).contiguous()
            if has_bias
            else None
        )
        self.register_buffer("_fused_kv_weight", fused_kv_weight, persistent=False)
        self.register_buffer("_fused_kv_bias", fused_kv_bias, persistent=False)
        self._num_context_kv_layers = len(attentions)
        self._context_kv_size = first.kv_size
        self._context_head_dim = first.head_dim
        self._context_num_kv_heads = first.num_kv_heads
        max_context_tokens = self._max_context_tokens
        dtype = self._fused_kv_weight.dtype
        device = self._fused_kv_weight.device
        # The in-place Ascend RoPE kernel requires a colocated cache of the
        # same dtype. Materialize this once after loading, never per step.
        for attention in attentions:
            rope = attention.rotary_emb
            if (
                rope.cos_sin_cache.device != device
                or rope.cos_sin_cache.dtype != dtype
            ):
                rope.cos_sin_cache = rope.cos_sin_cache.to(
                    device=device, dtype=dtype
                )
        self.register_buffer("_normed_context", torch.empty(
            (max_context_tokens, self.config.hidden_size), dtype=dtype, device=device
        ), persistent=False)
        self.register_buffer("_projected_context_kv", torch.empty(
            (max_context_tokens, len(attentions) * 2 * first.kv_size),
            dtype=dtype,
            device=device,
        ), persistent=False)
        self.register_buffer("_compact_context_kv", torch.empty(
            (
                2,
                len(attentions) * max_context_tokens,
                first.num_kv_heads,
                first.head_dim,
            ),
            dtype=dtype,
            device=device,
        ), persistent=False)
        self.register_buffer("_normalized_context_k", torch.empty(
            (
                len(attentions) * max_context_tokens,
                first.num_kv_heads,
                first.head_dim,
            ),
            dtype=dtype,
            device=device,
        ), persistent=False)
        self.register_buffer("_repeated_context_positions", torch.empty(
            len(attentions) * max_context_tokens,
            dtype=torch.int32,
            device=device,
        ), persistent=False)
        self._fused_context_kv_ready = True

    @torch.inference_mode()
    def compute_context_kv(
        self, context_states: torch.Tensor, context_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._fused_context_kv_ready:
            raise RuntimeError("DFlash fused Context KV buffers were not built after weight load")
        if context_states.ndim != 2 or context_states.shape[0] != context_positions.numel():
            raise ValueError("Context hidden states and positions must have matching token counts")
        num_context = context_states.shape[0]
        if num_context > self._max_context_tokens:
            raise ValueError(
                f"DFlash context has {num_context} tokens, buffer capacity is "
                f"{self._max_context_tokens}"
            )
        normed = self._normed_context[:num_context]
        if self._context_rms_norm_impl is None:
            normed.copy_(self.hidden_norm(context_states))
        else:
            self._context_rms_norm_impl(
                normed,
                context_states,
                self.hidden_norm.weight,
                self.hidden_norm.variance_epsilon,
            )
        projected = self._projected_context_kv[:num_context]
        torch.mm(normed, self._fused_kv_weight.t(), out=projected)
        if self._fused_kv_bias is not None:
            projected.add_(self._fused_kv_bias)
        projected_token_major = projected.view(
            num_context,
            self._num_context_kv_layers,
            2,
            self._context_num_kv_heads,
            self._context_head_dim,
        )
        compact_tokens = self._num_context_kv_layers * num_context
        all_kv = self._compact_context_kv[:, :compact_tokens].view(
            2,
            self._num_context_kv_layers,
            num_context,
            self._context_num_kv_heads,
            self._context_head_dim,
        )
        for layer_idx in range(self._num_context_kv_layers):
            all_kv[0, layer_idx].copy_(projected_token_major[:, layer_idx, 0])
            all_kv[1, layer_idx].copy_(projected_token_major[:, layer_idx, 1])
        all_k = self._normalized_context_k[:compact_tokens].view(
            self._num_context_kv_layers,
            num_context,
            self._context_num_kv_heads,
            self._context_head_dim,
        )
        for layer_idx, layer in enumerate(self.layers):
            if self._context_rms_norm_impl is None:
                all_k[layer_idx].copy_(
                    layer.self_attn.k_norm(all_kv[0, layer_idx])
                )
            else:
                self._context_rms_norm_impl(
                    all_k[layer_idx],
                    all_kv[0, layer_idx],
                    layer.self_attn.k_norm.weight,
                    layer.self_attn.k_norm.variance_epsilon,
                )
        flattened_k = all_k.view(compact_tokens, self._context_kv_size)
        repeated_positions = self._repeated_context_positions[:compact_tokens]
        for layer_idx in range(self._num_context_kv_layers):
            repeated_positions[
                layer_idx * num_context : (layer_idx + 1) * num_context
            ].copy_(context_positions)
        # Some torch-npu RoPE implementations do not accept key=None. The
        # unnormalized K projection is dead after this point, so reuse it as
        # the dummy key instead of cloning a context-sized tensor every step.
        rope_scratch = all_kv[0].view_as(flattened_k)
        rotary = self.layers[0].self_attn.rotary_emb
        if self._context_rope_impl is None:
            rotated_k, _ = rotary(
                repeated_positions, flattened_k, rope_scratch
            )
            if rotated_k.data_ptr() != flattened_k.data_ptr():
                flattened_k.copy_(rotated_k)
        else:
            self._context_rope_impl(
                repeated_positions,
                flattened_k,
                rope_scratch,
                rotary.head_size,
                rotary.cos_sin_cache,
                rotary.is_neox_style,
            )
        return (
            flattened_k.view(
                self._num_context_kv_layers,
                num_context,
                self._context_num_kv_heads,
                self._context_head_dim,
            ),
            all_kv[1],
        )

    def get_context_kv_attention_layers(self) -> tuple[Attention, ...]:
        return tuple(layer.self_attn.attn for layer in self.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        loaded_shards: dict[str, set[str | int]] = {}
        for original_name, loaded_weight in weights:
            name = original_name
            if "rotary_emb.inv_freq" in name:
                continue
            for packed_name, shard_name, shard_id in stacked:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, packed_name)
                if name not in params:
                    raise ValueError(f"Unknown DFlash checkpoint weight {original_name!r}")
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if loader == default_weight_loader:
                    loader(param, loaded_weight)
                else:
                    loader(param, loaded_weight, shard_id)
                loaded.add(name)
                loaded_shards.setdefault(name, set()).add(shard_id)
                break
            else:
                if name not in params:
                    raise ValueError(f"Unknown DFlash checkpoint weight {original_name!r}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
                loaded.add(name)

        expected = {name for name in params if name != "mask_embedding"}
        if isinstance(self.embed_tokens, PPMissingLayer):
            expected = {
                name for name in expected if not name.startswith("embed_tokens.")
            }
        missing = expected - loaded
        if missing:
            raise ValueError(f"DFlash checkpoint did not initialize weights: {sorted(missing)}")
        missing_shards = _missing_packed_shards(params, loaded_shards)
        if missing_shards:
            raise ValueError(
                f"DFlash checkpoint omitted packed parameter shards: {missing_shards}"
            )
        return loaded


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    """DFlash wrapper retaining the old model-runner call surface."""

    phase0_placeholder = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        spec_config = vllm_config.speculative_config
        if spec_config is None or spec_config.draft_model_config is None:
            raise ValueError("DFlash requires speculative draft model config")
        self.draft_model_config = spec_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layers = spec_config.target_model_config.get_num_layers(
            spec_config.target_parallel_config
        )
        checkpoint_target_layers = getattr(self.config, "num_target_layers", None)
        if checkpoint_target_layers != target_layers:
            raise ValueError(
                "DFlash checkpoint target depth mismatch: "
                f"checkpoint={checkpoint_target_layers}, target={target_layers}"
            )
        raw_layer_ids = tuple(self.config.dflash_config["target_layer_ids"])
        if any(layer_id + 1 >= target_layers for layer_id in raw_layer_ids):
            raise ValueError(
                "DFlash target_layer_ids cannot be represented by the old Qwen "
                f"pre-layer capture ABI: ids={raw_layer_ids}, target_layers={target_layers}"
            )
        start_layer_id = spec_config.target_model_config.get_num_layers(
            spec_config.target_parallel_config
        )
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.draft_vocab_size = getattr(
            self.config, "draft_vocab_size", self.config.vocab_size
        )
        self.has_own_embed_tokens = self.draft_vocab_size < self.config.vocab_size
        self.has_own_lm_head = self.has_own_embed_tokens
        if self.has_own_lm_head:
            self.lm_head = ParallelLMHead(
                self.draft_vocab_size,
                self.config.hidden_size,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
            self.register_buffer(
                "draft_target_ids",
                torch.arange(self.draft_vocab_size, dtype=torch.long),
                persistent=False,
            )
        else:
            self.lm_head = PPMissingLayer()
            self.draft_id_to_target_id = None
            self.register_buffer("draft_target_ids", None, persistent=False)
        self.logits_processor = LogitsProcessor(self.draft_vocab_size)

    @property
    def aux_capture_layer_ids(self) -> tuple[int, ...]:
        return self.model.aux_capture_layer_ids

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None or is_multimodal is not None:
            raise NotImplementedError("Phase 1 DFlash is text-only")
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.lm_head, PPMissingLayer):
            raise RuntimeError("DFlash target LM head was not injected after load")
        logits = self.logits_processor(self.lm_head, hidden_states)
        return _expand_draft_logits_to_target(
            logits,
            self.draft_target_ids,
            self.config.vocab_size,
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def compute_context_kv(
        self, context_states: torch.Tensor, context_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.compute_context_kv(context_states, context_positions)

    def get_context_kv_attention_layers(self) -> tuple[Attention, ...]:
        return self.model.get_context_kv_attention_layers()

    def set_context_rms_norm_impl(self, implementation: Callable) -> None:
        self.model.set_context_rms_norm_impl(implementation)

    def set_context_rope_impl(self, implementation: Callable) -> None:
        self.model.set_context_rope_impl(implementation)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.layer_name for layer in self.get_context_kv_attention_layers()]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_weight: torch.Tensor | None = None
        d2t_weight: torch.Tensor | None = None
        t2d_weight: torch.Tensor | None = None
        unexpected: list[str] = []
        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name.removeprefix("model.")
            if name == "lm_head.weight":
                lm_head_weight = loaded_weight
            elif name == "d2t":
                d2t_weight = loaded_weight
            elif name == "t2d":
                t2d_weight = loaded_weight
            elif name.startswith("lm_head.") or name.startswith(("d2t.", "t2d.")):
                unexpected.append(name)
            elif name.startswith("embed_tokens.") and not self.has_own_embed_tokens:
                unexpected.append(name)
            else:
                model_weights.append((name, loaded_weight))
        if unexpected:
            raise ValueError(
                f"DFlash checkpoint contains unsupported vocabulary weights {unexpected[:8]}"
            )
        self.model.load_weights(model_weights)
        if self.has_own_lm_head:
            if lm_head_weight is None or d2t_weight is None or t2d_weight is None:
                raise ValueError(
                    "Reduced-vocabulary DFlash requires lm_head.weight, d2t and t2d"
                )
            parameter = self.lm_head.weight
            getattr(parameter, "weight_loader", default_weight_loader)(
                parameter, lm_head_weight
            )
            if tuple(d2t_weight.shape) != (self.draft_vocab_size,):
                raise ValueError("DFlash d2t shape does not match draft_vocab_size")
            if tuple(t2d_weight.shape) != (self.config.vocab_size,):
                raise ValueError("DFlash t2d shape does not match target vocab_size")
            base = torch.arange(
                self.draft_vocab_size,
                dtype=torch.long,
                device=d2t_weight.device,
            )
            target_ids = base + d2t_weight.to(dtype=torch.long)
            if (
                bool(torch.any(target_ids < 0))
                or bool(torch.any(target_ids >= self.config.vocab_size))
                or torch.unique(target_ids).numel() != self.draft_vocab_size
            ):
                raise ValueError("DFlash d2t must map uniquely into the target vocabulary")
            expected_t2d = torch.zeros(
                self.config.vocab_size, dtype=torch.bool, device=target_ids.device
            )
            expected_t2d[target_ids] = True
            if not torch.equal(expected_t2d, t2d_weight.to(dtype=torch.bool)):
                raise ValueError("DFlash t2d membership does not agree with d2t")
            self.draft_id_to_target_id.data.copy_(d2t_weight)
            self.draft_target_ids.copy_(target_ids)
        elif lm_head_weight is not None or d2t_weight is not None or t2d_weight is not None:
            raise ValueError(
                "Full-vocabulary DFlash must use target-owned LM head without mappings"
            )
        mask_embedding = self._read_mask_embedding()
        if mask_embedding is not None:
            if mask_embedding.numel() != self.config.hidden_size:
                raise ValueError(
                    f"mask_embedding has {mask_embedding.numel()} elements, expected "
                    f"{self.config.hidden_size}"
                )
            self.model.mask_embedding.data.copy_(
                mask_embedding.reshape(-1).to(self.model.mask_embedding)
            )
            self.model.has_separate_mask_embedding = True
        self.model.build_fused_context_kv_buffers()
        # Missing embedding/head parameters are deliberately target-owned.
        return None

    def _read_mask_embedding(self) -> torch.Tensor | None:
        if self.model.mask_token_id is None:
            return None
        local_model = Path(self.draft_model_config.model)
        if local_model.is_dir() and not (
            local_model / _MASK_EMBEDDING_FILENAME
        ).is_file():
            return None
        data = get_hf_file_bytes(
            _MASK_EMBEDDING_FILENAME,
            self.draft_model_config.model,
            self.draft_model_config.revision,
        )
        if data is None:
            return None
        state = torch.load(io.BytesIO(data), weights_only=True)
        if isinstance(state, dict):
            embedded_id = state.get("mask_token_id", self.model.mask_token_id)
            if embedded_id != self.model.mask_token_id:
                raise ValueError(
                    f"mask_embedding token id {embedded_id} does not match "
                    f"checkpoint id {self.model.mask_token_id}"
                )
            state = state["embedding"]
        if not isinstance(state, torch.Tensor):
            raise ValueError("mask_embedding.pt must contain a tensor or embedding dict")
        logger.info("Loaded separate DFlash mask embedding from %s", _MASK_EMBEDDING_FILENAME)
        return state
