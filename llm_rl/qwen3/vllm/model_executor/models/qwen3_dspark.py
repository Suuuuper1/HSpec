# SPDX-License-Identifier: Apache-2.0
"""Current-native Qwen3 DSpark draft model for the old V1 ABI.

DSpark reuses the Phase-1 DFlash parallel transformer and Context-KV path. Its
only algorithmic addition in Phase 2 is a replicated low-rank Markov head. The
checkpoint confidence head is loaded for provenance, but fixed-K greedy
proposal never calls it.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    dspark_aux_capture_layer_ids,
)
from .utils import maybe_prefix


class DSparkMarkovHead(nn.Module):
    """Replicated token embedding and rank-to-vocabulary transition bias."""

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        *,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(
            vocab_size, markov_rank, dtype=params_dtype
        )
        self.markov_w2 = ReplicatedLinear(
            input_size=markov_rank,
            output_size=vocab_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "markov_w2"),
            return_bias=False,
            disable_tp=True,
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def embed_into(
        self, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if output.shape != (token_ids.numel(), self.markov_rank):
            raise ValueError(
                "DSpark Markov embedding output shape mismatch: "
                f"{tuple(output.shape)} != {(token_ids.numel(), self.markov_rank)}"
            )
        torch.index_select(self.markov_w1.weight, 0, token_ids, out=output)
        return output

    def bias(self, markov_embedding: torch.Tensor) -> torch.Tensor:
        output = self.markov_w2(markov_embedding)
        assert isinstance(output, torch.Tensor)
        return output

    def bias_into(
        self, markov_embedding: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        expected = (markov_embedding.shape[0], self.vocab_size)
        if output.shape != expected:
            raise ValueError(
                f"DSpark Markov bias output shape mismatch: "
                f"{tuple(output.shape)} != {expected}"
            )
        torch.mm(markov_embedding, self.markov_w2.weight.t(), out=output)
        return output


class DSparkConfidenceHead(nn.Module):
    """Checkpoint-compatible confidence head, disabled in the Phase-2 runtime."""

    def __init__(
        self,
        input_size: int,
        *,
        with_markov: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_size=input_size,
            output_size=1,
            bias=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=maybe_prefix(prefix, "proj"),
            return_bias=False,
            disable_tp=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, markov_embedding: torch.Tensor
    ) -> torch.Tensor:
        inputs = (
            torch.cat((hidden_states, markov_embedding), dim=-1)
            if self.with_markov
            else hidden_states
        )
        output = self.proj(inputs.float())
        assert isinstance(output, torch.Tensor)
        return output.squeeze(-1)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash parallel backbone with checkpoint-owned DSpark heads."""

    parallel_draft_method = "dspark"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        dtype = vllm_config.model_config.dtype
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.markov_rank,
            params_dtype=dtype,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(self.config, "enable_confidence_head", False):
            with_markov = getattr(
                self.config, "confidence_head_with_markov", False
            )
            input_size = self.config.hidden_size
            if with_markov:
                input_size += self.config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_size,
                with_markov=with_markov,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    """Qwen3 DSpark model with strict full-vocabulary checkpoint ownership."""

    phase0_placeholder = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        spec_config = vllm_config.speculative_config
        if spec_config is None or spec_config.draft_model_config is None:
            raise ValueError("DSpark requires speculative draft model config")
        self.draft_model_config = spec_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layers = spec_config.target_model_config.get_num_layers(
            spec_config.target_parallel_config
        )
        checkpoint_target_layers = getattr(
            self.config, "num_target_layers", None
        )
        if checkpoint_target_layers != target_layers:
            raise ValueError(
                "DSpark checkpoint target depth mismatch: "
                f"checkpoint={checkpoint_target_layers}, target={target_layers}"
            )
        capture_layers = dspark_aux_capture_layer_ids(self.config)
        if any(layer_id >= target_layers for layer_id in capture_layers):
            raise ValueError(
                "DSpark target_layer_ids cannot be represented by the old Qwen "
                f"pre-layer capture ABI: capture={capture_layers}, "
                f"target_layers={target_layers}"
            )
        if getattr(self.config, "draft_vocab_size", self.config.vocab_size) != (
            self.config.vocab_size
        ):
            raise NotImplementedError(
                "Phase 2 DSpark requires an equal full vocabulary"
            )

        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layers,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.has_own_embed_tokens = True
        self.has_own_lm_head = True

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_draft_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_embed_into(
        self, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self.model.markov_head.embed_into(token_ids, output)

    def markov_bias(self, markov_embedding: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embedding)

    def markov_bias_into(
        self, markov_embedding: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self.model.markov_head.bias_into(markov_embedding, output)

    def compute_confidence(
        self, hidden_states: torch.Tensor, markov_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.model.confidence_head is None:
            raise RuntimeError("DSpark checkpoint has no confidence head")
        return torch.sigmoid(
            self.model.confidence_head(hidden_states, markov_embedding)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_weight: torch.Tensor | None = None
        for original_name, loaded_weight in weights:
            name = original_name.removeprefix("model.")
            if name == "lm_head.weight":
                if lm_head_weight is not None:
                    raise ValueError("DSpark checkpoint contains duplicate lm_head.weight")
                lm_head_weight = loaded_weight
            elif name == "t2d" or name.startswith("t2d."):
                continue
            elif name == "d2t" or name.startswith("d2t."):
                raise NotImplementedError(
                    "Phase 2 DSpark reduced-vocabulary mapping is not supported"
                )
            else:
                model_weights.append((name, loaded_weight))

        self.model.load_weights(model_weights)
        if lm_head_weight is None:
            raise ValueError("DSpark checkpoint did not initialize lm_head.weight")
        parameter = self.lm_head.weight
        loader = getattr(parameter, "weight_loader", default_weight_loader)
        loader(parameter, lm_head_weight)
        self.model.build_fused_context_kv_buffers()
        return None
