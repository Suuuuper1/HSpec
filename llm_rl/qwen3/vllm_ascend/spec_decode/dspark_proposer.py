# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DSpark eager, greedy, one-forward proposer for the old V1 ABI."""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.spec_decode.dflash_proposer import DFlashProposer
from vllm_ascend.spec_decode.probabilistic import sample_draft_logits


def dspark_markov_greedy(
    model: Qwen3DSparkForCausalLM,
    sample_hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    num_speculative_tokens: int,
    token_buffer: torch.Tensor,
    embedding_buffer: torch.Tensor,
    corrected_logits_buffer: torch.Tensor,
) -> torch.Tensor:
    """Apply the sequential Markov correction without per-position allocation."""
    batch_size = anchor_token_ids.numel()
    if sample_hidden_states.shape[0] != batch_size * num_speculative_tokens:
        raise ValueError(
            "DSpark sampled hidden row count must be B*K: "
            f"{sample_hidden_states.shape[0]} != "
            f"{batch_size}*{num_speculative_tokens}"
        )
    tokens = token_buffer[:batch_size, : num_speculative_tokens + 1]
    markov_embedding = embedding_buffer[:batch_size]
    corrected_logits = corrected_logits_buffer[:batch_size]
    if tokens.dtype != torch.int64:
        raise TypeError(f"DSpark token buffer must be int64, got {tokens.dtype}")
    tokens[:, 0].copy_(anchor_token_ids)

    base_logits = model.compute_draft_logits(sample_hidden_states)
    if base_logits.ndim != 2 or base_logits.shape[0] != (
        batch_size * num_speculative_tokens
    ):
        raise ValueError(
            f"DSpark base logits must be [B*K,V], got {tuple(base_logits.shape)}"
        )
    vocab_size = corrected_logits.shape[1]
    if base_logits.shape[1] != vocab_size:
        raise ValueError(
            f"DSpark logits/buffer vocabulary mismatch: "
            f"{base_logits.shape[1]} != {vocab_size}"
        )
    base_logits = base_logits.view(
        batch_size, num_speculative_tokens, vocab_size
    )

    for position in range(num_speculative_tokens):
        model.markov_embed_into(tokens[:, position], markov_embedding)
        # markov_bias_into fully overwrites this persistent buffer. Add the
        # current base row only after that overwrite so neither base logits nor
        # a prior proposal can contaminate the next call.
        model.markov_bias_into(markov_embedding, corrected_logits)
        corrected_logits.add_(base_logits[:, position])
        torch.argmax(
            corrected_logits,
            dim=-1,
            out=tokens[:, position + 1],
        )
    return model.map_draft_to_target(tokens[:, 1:])


def dspark_markov_probabilistic(
    model: Qwen3DSparkForCausalLM,
    sample_hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    num_speculative_tokens: int,
    token_buffer: torch.Tensor,
    embedding_buffer: torch.Tensor,
    corrected_logits_buffer: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply sequential Markov feedback and preserve every sampled q row."""
    batch_size = anchor_token_ids.numel()
    if sample_hidden_states.shape[0] != batch_size * num_speculative_tokens:
        raise ValueError(
            "DSpark sampled hidden row count must be B*K: "
            f"{sample_hidden_states.shape[0]} != {batch_size}*{num_speculative_tokens}"
        )
    if sampling_metadata.temperature is None:
        raise RuntimeError("DSpark probabilistic proposal requires temperatures")
    tokens = token_buffer[:batch_size, : num_speculative_tokens + 1]
    tokens[:, 0].copy_(anchor_token_ids)
    embedding = embedding_buffer[:batch_size]
    corrected = corrected_logits_buffer[:batch_size]
    base_logits = model.compute_draft_logits(sample_hidden_states)
    vocab_size = corrected.shape[1]
    if tuple(base_logits.shape) != (
        batch_size * num_speculative_tokens,
        vocab_size,
    ):
        raise ValueError(
            "DSpark base logits/probability vocabulary mismatch: "
            f"{tuple(base_logits.shape)}"
        )
    base_logits = base_logits.view(batch_size, num_speculative_tokens, vocab_size)
    probabilities = (
        None
        if sampling_metadata.all_greedy
        else torch.empty(
            batch_size,
            num_speculative_tokens,
            vocab_size,
            dtype=torch.float32,
            device=base_logits.device,
        )
    )
    for position in range(num_speculative_tokens):
        model.markov_embed_into(tokens[:, position], embedding)
        model.markov_bias_into(embedding, corrected)
        corrected.add_(base_logits[:, position])
        token_ids, step_probabilities = sample_draft_logits(
            corrected,
            sampling_metadata.temperature,
            rows_per_request=1,
            all_greedy=sampling_metadata.all_greedy,
            all_random=sampling_metadata.all_random,
            generators=sampling_metadata.generators,
        )
        tokens[:, position + 1].copy_(token_ids)
        if probabilities is not None:
            if step_probabilities is None:
                raise RuntimeError(
                    "DSpark probabilistic Markov step produced no probabilities"
                )
            probabilities[:, position].copy_(step_probabilities)
    return model.map_draft_to_target(tokens[:, 1:]), probabilities


class DSparkProposer(DFlashProposer):
    """DFlash parallel backbone plus a replicated K-step Markov greedy head."""

    method = "dspark"
    expected_model_type = Qwen3DSparkForCausalLM

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None:
        super().__init__(vllm_config, device, runner)
        config = self.speculative_config.draft_model_config.hf_config
        self._dspark_token_buffer = torch.empty(
            (self.max_batch_size, self.num_speculative_tokens + 1),
            dtype=torch.int64,
            device=device,
        )
        self._dspark_embedding_buffer = torch.empty(
            (self.max_batch_size, config.markov_rank),
            dtype=self.dtype,
            device=device,
        )
        self._dspark_corrected_logits_buffer = torch.empty(
            (self.max_batch_size, config.vocab_size),
            dtype=self.dtype,
            device=device,
        )

    def _configure_vocabulary(self, target_language_model: nn.Module) -> None:
        if not hasattr(target_language_model, "model") or not hasattr(
            target_language_model.model, "embed_tokens"
        ):
            raise AttributeError("Target Qwen3 model does not expose model.embed_tokens")
        target_embedding = target_language_model.model.embed_tokens
        target_lm_head = target_language_model.lm_head
        own_embedding = self.model.model.embed_tokens
        own_lm_head = self.model.lm_head
        target_shapes = (
            tuple(target_embedding.weight.shape),
            tuple(target_lm_head.weight.shape),
        )
        own_shapes = (
            tuple(own_embedding.weight.shape),
            tuple(own_lm_head.weight.shape),
        )
        if target_shapes[0] != target_shapes[1] or own_shapes != target_shapes:
            raise ValueError(
                "DSpark target/draft vocabulary tensor shapes are incompatible: "
                f"target={target_shapes}, draft={own_shapes}"
            )
        if (
            own_embedding.weight.data_ptr() == target_embedding.weight.data_ptr()
            or own_lm_head.weight.data_ptr() == target_lm_head.weight.data_ptr()
        ):
            raise RuntimeError(
                "DSpark checkpoint-owned embedding/LM head must not alias target weights"
            )

    def _select_draft_tokens(
        self,
        sample_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor | None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        if anchor_token_ids is None:
            raise ValueError("DSpark Markov proposal requires anchor token ids")
        if not getattr(self, "_probabilistic", False) or sampling_metadata is None:
            self._last_draft_probs = None
            return dspark_markov_greedy(
                self.model,
                sample_hidden_states,
                anchor_token_ids,
                self.num_speculative_tokens,
                self._dspark_token_buffer,
                self._dspark_embedding_buffer,
                self._dspark_corrected_logits_buffer,
            )
        tokens, self._last_draft_probs = dspark_markov_probabilistic(
            self.model,
            sample_hidden_states,
            anchor_token_ids,
            self.num_speculative_tokens,
            self._dspark_token_buffer,
            self._dspark_embedding_buffer,
            self._dspark_corrected_logits_buffer,
            sampling_metadata,
        )
        return tokens
