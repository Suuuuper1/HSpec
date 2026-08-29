# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DFlash eager, greedy, one-forward proposer for the old V1 ABI."""

from typing import Optional

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    dflash_target_rope_is_neox_style,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.context_kv import PAD_SLOT_ID, store_all_context_kv
from vllm_ascend.ops.triton.spec_decode.utils import (
    expand_parallel_queries_kernel,
    sanitize_and_copy_context_kernel,
)
from vllm_ascend.ops.triton.batch_invariant.rmsnorm import rms_norm_into
from vllm_ascend.ops.rotary_embedding import rotary_embedding_into
from vllm_ascend.spec_decode.parallel_block_proposer import ParallelBlockProposer

_TILE_SIZE = 128


class DFlashProposer(ParallelBlockProposer):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None:
        super().__init__(vllm_config, device, runner)
        if not self.speculative_config.use_dflash():
            raise ValueError("DFlashProposer requires method='dflash'")
        max_context_tokens = self.max_num_tokens
        self._context_positions = torch.empty(
            max_context_tokens, dtype=torch.int32, device=device
        )
        self._context_slots = torch.empty(
            max_context_tokens, dtype=torch.int32, device=device
        )
        self._query_input_ids = torch.empty(
            self.max_query_tokens, dtype=torch.int32, device=device
        )
        self._query_positions = torch.empty(
            self.max_query_tokens, dtype=torch.int32, device=device
        )
        self._query_slots = torch.empty(
            self.max_query_tokens, dtype=torch.int32, device=device
        )
        self._sample_indices = torch.empty(
            self.max_batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._aux_hidden_buffer: torch.Tensor | None = None

    def load_model(self, target_model: nn.Module) -> None:
        target_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase
            ).keys()
        )
        rope_style = dflash_target_rope_is_neox_style(target_model)
        if rope_style is None:
            raise RuntimeError("Could not determine target Qwen3 RoPE layout")
        self.speculative_config.draft_model_config.hf_config.is_neox_style = rope_style
        with self.maybe_eager_context:
            self.model = self._load_draft_model()
        if not isinstance(self.model, DFlashQwen3ForCausalLM):
            raise TypeError(f"Registry loaded unexpected DFlash class {type(self.model)}")
        self.model.set_context_rms_norm_impl(rms_norm_into)
        self.model.set_context_rope_impl(rotary_embedding_into)
        self.attn_layer_names = self._discover_draft_attention_layers(target_names)
        if self.model.get_draft_attn_causal() != [False] * len(self.attn_layer_names):
            raise RuntimeError("Phase 1 DFlash draft layers must all be non-causal")

        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        if not hasattr(target_language_model, "model") or not hasattr(
            target_language_model.model, "embed_tokens"
        ):
            raise AttributeError("Target Qwen3 model does not expose model.embed_tokens")
        target_embedding = target_language_model.model.embed_tokens
        target_lm_head = target_language_model.lm_head
        embedding_shape = tuple(target_embedding.weight.shape)
        lm_head_shape = tuple(target_lm_head.weight.shape)
        expected_hidden = self.model.config.hidden_size
        expected_vocab = self.model.config.vocab_size
        if embedding_shape[1] != expected_hidden or embedding_shape[0] < expected_vocab:
            raise ValueError(
                f"Target embedding shape {embedding_shape} is incompatible with "
                f"DFlash vocab/hidden ({expected_vocab}, {expected_hidden})"
            )
        if lm_head_shape != embedding_shape:
            raise ValueError(
                f"Target LM-head/embedding local shapes differ: "
                f"{lm_head_shape} != {embedding_shape}"
            )
        self.model.model.embed_tokens = target_embedding
        self.model.lm_head = target_lm_head
        self.aux_hidden_state_layer_ids = self.model.aux_capture_layer_ids
        if self.model.model.fc.input_size != (
            len(self.aux_hidden_state_layer_ids) * self.hidden_size
        ):
            raise ValueError(
                "DFlash FC width does not match target aux layer count: "
                f"fc={self.model.model.fc.input_size}, layers="
                f"{self.aux_hidden_state_layer_ids}, hidden={self.hidden_size}"
            )
        self._aux_hidden_buffer = torch.empty(
            (self.max_num_tokens, self.model.model.fc.input_size),
            dtype=self.dtype,
            device=self.device,
        )
        logger.info(
            "Loaded DFlash draft: layers=%s aux_capture_layers=%s rope_neox=%s",
            self.attn_layer_names,
            self.aux_hidden_state_layer_ids,
            rope_style,
        )

    def get_aux_hidden_state_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.aux_hidden_state_layer_ids)

    def pack_aux_hidden_states(
        self,
        aux_hidden_states: list[torch.Tensor],
        token_indices: torch.Tensor | None,
        num_tokens: int,
    ) -> torch.Tensor:
        """Pack ordered target features into a persistent FC input buffer."""
        if self._aux_hidden_buffer is None:
            raise RuntimeError("DFlash aux buffer is unavailable before draft load")
        if not 0 <= num_tokens <= self._aux_hidden_buffer.shape[0]:
            raise ValueError(
                f"DFlash aux token count {num_tokens} exceeds buffer capacity "
                f"{self._aux_hidden_buffer.shape[0]}"
            )
        if len(aux_hidden_states) != len(self.aux_hidden_state_layer_ids):
            raise ValueError(
                "Target auxiliary output count does not match DFlash checkpoint: "
                f"{len(aux_hidden_states)} != {len(self.aux_hidden_state_layer_ids)}"
            )
        if token_indices is not None:
            if token_indices.ndim != 1 or token_indices.numel() != num_tokens:
                raise ValueError(
                    "DFlash aux token_indices must be one-dimensional and "
                    f"contain num_tokens={num_tokens} entries"
                )
            if token_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError(
                    f"DFlash aux token_indices must be integer, got "
                    f"{token_indices.dtype}"
                )
            if token_indices.device != self._aux_hidden_buffer.device:
                raise ValueError("DFlash aux token_indices must remain on the NPU")
        packed = self._aux_hidden_buffer[:num_tokens]
        for feature_index, feature in enumerate(aux_hidden_states):
            if feature.ndim != 2 or feature.shape[1] != self.hidden_size:
                raise ValueError(
                    f"DFlash aux feature {feature_index} must be [T,{self.hidden_size}], "
                    f"got {tuple(feature.shape)}"
                )
            if (
                feature.dtype != packed.dtype
                or feature.device != packed.device
            ):
                raise TypeError(
                    f"DFlash aux feature {feature_index} dtype/device mismatch: "
                    f"{feature.dtype}/{feature.device} != "
                    f"{packed.dtype}/{packed.device}"
                )
            if token_indices is None and feature.shape[0] < num_tokens:
                raise ValueError(
                    f"DFlash aux feature {feature_index} has only "
                    f"{feature.shape[0]} rows for {num_tokens} tokens"
                )
            start = feature_index * self.hidden_size
            destination = packed[:, start : start + self.hidden_size]
            if token_indices is None:
                destination.copy_(feature[:num_tokens])
            else:
                torch.index_select(feature, 0, token_indices, out=destination)
        return packed

    def _build_parallel_layout(
        self,
        target_positions: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        rejected_counts: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = common_attn_metadata.num_reqs
        num_context = target_positions.shape[-1]
        num_query = batch_size * self.num_queries
        context_positions = self._context_positions[:num_context]
        context_slots = self._context_slots[:num_context]
        sanitize_and_copy_context_kernel[(batch_size,)](
            target_positions,
            common_attn_metadata.slot_mapping,
            context_positions,
            context_slots,
            common_attn_metadata.query_start_loc,
            rejected_counts,
            batch_size,
            self.runner.input_batch.block_table[0].block_size,
            0,
            PAD_SLOT_ID=PAD_SLOT_ID,
            HAS_NULL_BLOCK=False,
            TILE_SIZE=_TILE_SIZE,
        )

        query_ids = self._query_input_ids[:num_query]
        query_positions = self._query_positions[:num_query]
        query_slots = self._query_slots[:num_query]
        sample_indices = self._sample_indices[
            : batch_size * self.num_speculative_tokens
        ]
        block_table = common_attn_metadata.block_table_tensor
        grid = (max(1, triton.cdiv(num_query, _TILE_SIZE)),)
        expand_parallel_queries_kernel[grid](
            next_token_ids,
            target_positions,
            query_ids,
            query_positions,
            query_slots,
            sample_indices,
            block_table,
            block_table.stride(0),
            block_table.shape[1],
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            rejected_counts,
            self.model.model.mask_token_id,
            self.runner.input_batch.block_table[0].block_size,
            0,
            self.num_queries,
            self.num_speculative_tokens,
            batch_size,
            self.speculative_config.draft_model_config.max_model_len,
            PAD_SLOT_ID=PAD_SLOT_ID,
            HAS_NULL_BLOCK=False,
            SAMPLE_FROM_ANCHOR=False,
            TILE_SIZE=_TILE_SIZE,
        )
        return (
            context_positions,
            context_slots,
            query_ids,
            query_positions,
            query_slots,
            sample_indices,
        )

    @torch.inference_mode()
    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: Optional[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs: int = 0,
        num_decode_reqs: int = 0,
        scheduler_output: SchedulerOutput | None = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        del (
            target_token_ids,
            last_token_indices,
            sampling_metadata,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
            scheduler_output,
            num_scheduled_tokens,
        )
        batch_size = common_attn_metadata.num_reqs
        rejected = self._num_rejected_tokens[:batch_size]
        if num_rejected_tokens_gpu is None:
            rejected.zero_()
        elif num_rejected_tokens_gpu.data_ptr() != rejected.data_ptr():
            rejected.copy_(num_rejected_tokens_gpu)

        combined_hidden = self.model.combine_hidden_states(target_hidden_states)
        (
            context_positions,
            context_slots,
            query_ids,
            query_positions,
            query_slots,
            sample_indices,
        ) = self._build_parallel_layout(
            target_positions, next_token_ids, common_attn_metadata, rejected
        )
        all_key, all_value = self.model.compute_context_kv(
            combined_hidden, context_positions
        )
        store_all_context_kv(
            self.model.get_context_kv_attention_layers(),
            all_key,
            all_value,
            context_slots,
            # Slots originate from the engine-owned block table and the layout
            # kernels convert every rejected/out-of-window row to PAD. Repeating
            # torch._assert_async here falls back to CPU on Ascend and inserts a
            # host operation into every proposal step; direct-store tests retain
            # range validation at the public adapter boundary.
            validate_slot_range=False,
        )

        metadata = self.build_parallel_attention_metadata(
            common_attn_metadata, query_slots, rejected
        )
        per_layer_metadata = {
            layer_name: metadata for layer_name in self.attn_layer_names
        }
        num_query_tokens = batch_size * self.num_queries
        with set_ascend_forward_context(
            per_layer_metadata,
            self.vllm_config,
            num_tokens=num_query_tokens,
            num_tokens_across_dp=None,
            num_actual_tokens=num_query_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
        ):
            draft_token_ids = self._run_parallel_backbone(
                query_ids, query_positions, sample_indices
            )
        draft_token_ids.masked_fill_(
            self._cannot_speculate[:batch_size, None], PAD_SLOT_ID
        )
        return draft_token_ids
