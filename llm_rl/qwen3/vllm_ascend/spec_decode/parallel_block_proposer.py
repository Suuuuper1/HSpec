# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Old-ABI base for one-forward parallel-block speculative drafters."""

import copy
from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.ops.triton.spec_decode.utils import compute_rejected_tokens_kernel
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer


class ParallelBlockProposer(EagleProposer):
    """Shared lifecycle and padded-verification ABI for DFlash/DSpark.

    Existing EAGLE keeps its three-return ``prepare_inputs_padded`` method.
    Parallel drafters use the four-return method below so rejected counts stay
    device-resident and can sanitize Context KV writes.
    """

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None:
        super().__init__(vllm_config, device, runner)
        if vllm_config.parallel_config.data_parallel_size > 1:
            raise NotImplementedError("Phase 1 DFlash certifies DP=1 only")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError("Phase 1 DFlash requires async_scheduling=False")
        # Full graph still hard-codes causal FIA arguments in this ABI.
        self.use_cuda_graph = False
        self._runnable = self._run_parallel_backbone
        self.max_batch_size = runner.max_num_reqs
        self.num_queries = self.speculative_config.parallel_query_count
        self.max_query_tokens = self.max_batch_size * self.num_queries
        self._num_rejected_tokens = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._effective_seq_lens = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._can_speculate = torch.empty(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._cannot_speculate = torch.empty(
            self.max_batch_size, dtype=torch.bool, device=device
        )
        self._query_start_loc = torch.arange(
            self.max_batch_size + 1, dtype=torch.int32, device=device
        ) * self.num_queries
        self._query_start_loc_cpu = torch.arange(
            self.max_batch_size + 1, dtype=torch.int32, device="cpu"
        ) * self.num_queries
        self._actual_query_lengths = tuple(
            tuple(range(self.num_queries, (batch + 1) * self.num_queries, self.num_queries))
            for batch in range(1, self.max_batch_size + 1)
        )
        self._parallel_dummy_positions = torch.zeros(
            self.max_query_tokens, dtype=torch.int32, device=device
        )
        dummy_sample_rows = torch.arange(
            self.max_batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._parallel_dummy_sample_indices = (
            torch.div(
                dummy_sample_rows,
                self.num_speculative_tokens,
                rounding_mode="floor",
            )
            * self.num_queries
            + torch.remainder(dummy_sample_rows, self.num_speculative_tokens)
            + 1
        )

    def _draft_vllm_config(self) -> VllmConfig:
        draft_model_config = copy.copy(
            self.speculative_config.draft_model_config
        )
        draft_model_config.enforce_eager = True
        # Attention layers from both models must remain in the engine-owned
        # registry so KV-cache allocation can discover and bind the drafter.
        # The rest of CompilationConfig contains mutable custom-op/compile
        # state and must not leak back into the target graph. Deep-copy it with
        # an explicit memo entry for the registry: this isolates those fields
        # without recursively copying already-loaded target modules.
        target_compilation_config = self.vllm_config.compilation_config
        shared_forward_context = target_compilation_config.static_forward_context
        draft_compilation_config = copy.deepcopy(
            target_compilation_config,
            memo={id(shared_forward_context): shared_forward_context},
        )
        draft_compilation_config.static_forward_context = shared_forward_context
        draft_compilation_config.mode = CompilationMode.NONE
        draft_compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        return replace(
            self.vllm_config,
            model_config=draft_model_config,
            load_config=self.speculative_config.draft_load_config,
            parallel_config=self.speculative_config.draft_parallel_config,
            compilation_config=draft_compilation_config,
        )

    def _load_draft_model(self) -> nn.Module:
        draft_vllm_config = self._draft_vllm_config()
        return get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_vllm_config.model_config,
        )

    def _discover_draft_attention_layers(
        self, target_layer_names: set[str]
    ) -> list[str]:
        all_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase
        )
        discovered = sorted(set(all_layers) - target_layer_names)
        model_layers = tuple(self.model.get_context_kv_attention_layers())
        explicit_by_name = {layer.layer_name: layer for layer in model_layers}
        if len(explicit_by_name) != len(model_layers):
            raise RuntimeError("DFlash model contains duplicate attention layer names")
        explicit = sorted(explicit_by_name)
        if discovered != explicit:
            raise RuntimeError(
                "DFlash draft attention discovery mismatch: "
                f"registry={discovered}, model={explicit}"
            )
        if not discovered:
            raise RuntimeError("DFlash model registered no attention layers")
        identity_mismatches = [
            layer_name
            for layer_name in discovered
            if all_layers[layer_name] is not explicit_by_name[layer_name]
        ]
        if identity_mismatches:
            raise RuntimeError(
                "DFlash draft Attention registry identity mismatch: "
                f"{identity_mismatches}"
            )
        return discovered

    def prepare_parallel_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata, token_indices, token_indices_to_sample = super().prepare_inputs_padded(
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )
        num_reqs = metadata.num_reqs
        rejected = self._num_rejected_tokens[:num_reqs]
        block_size = 128
        compute_rejected_tokens_kernel[(1,)](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            rejected,
            num_reqs,
            BLOCK_SIZE=block_size,
        )
        return metadata, token_indices, token_indices_to_sample, rejected

    def build_parallel_attention_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        query_slot_mapping: torch.Tensor,
        rejected_counts: torch.Tensor,
    ) -> AscendMetadata:
        batch_size = common_attn_metadata.num_reqs
        effective = self._effective_seq_lens[:batch_size]
        torch.sub(
            common_attn_metadata.seq_lens[:batch_size],
            rejected_counts,
            out=effective,
        )
        effective.add_(self.num_queries)
        can_speculate = self._can_speculate[:batch_size]
        cannot_speculate = self._cannot_speculate[:batch_size]
        max_model_len = self.speculative_config.draft_model_config.max_model_len
        torch.lt(effective, max_model_len + 1, out=can_speculate)
        torch.logical_not(can_speculate, out=cannot_speculate)
        # Invalid fixed-K windows still execute padded rows so the ABI keeps a
        # stable [B,K] shape. A length of one and PAD slots make their output
        # irrelevant and keep FIA/cache addressing inside the allocation.
        effective.masked_fill_(cannot_speculate, 1)

        # Construct the narrow eager metadata directly. The generic old builder
        # pins/copies query_start_loc and materializes CPU lists on every call;
        # neither is needed by the non-causal TND FIA path.
        return AscendMetadata(
            num_actual_tokens=batch_size * self.num_queries,
            num_decode_tokens=0,
            num_prefills=batch_size,
            num_decodes=0,
            block_tables=common_attn_metadata.block_table_tensor,
            query_start_loc=self._query_start_loc[: batch_size + 1],
            seq_lens=effective,
            # torch-npu FIA accepts this device tensor despite the legacy type
            # annotation saying list[int]. This keeps rejected counts on NPU.
            seq_lens_list=effective,
            max_query_len=self.num_queries,
            actual_seq_lengths_q=self._actual_query_lengths[batch_size - 1],
            slot_mapping=query_slot_mapping,
            attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            model_runner_type=self.vllm_config.model_config.runner_type,
        )

    def _run_parallel_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, positions=positions)
        sample_hidden_states = hidden_states[sample_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        return logits.argmax(dim=-1).view(-1, self.num_speculative_tokens)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile: bool = False,
    ) -> None:
        del (
            with_prefill,
            in_graph_capturing,
            aclgraph_runtime_mode,
            batch_descriptor,
            dummy_compute_logits,
        )
        context_tokens = min(num_tokens, self.max_num_tokens)
        profile_reqs = min(max(num_reqs, 1), self.max_batch_size)
        query_tokens = profile_reqs * self.num_queries
        positions = self._get_positions(context_tokens)
        aux_hidden_buffer = getattr(self, "_aux_hidden_buffer", None)
        if aux_hidden_buffer is None:
            raise RuntimeError("Parallel draft aux buffer is unavailable during profile")
        combined_hidden = self.model.combine_hidden_states(
            aux_hidden_buffer[:context_tokens]
        )
        self.model.compute_context_kv(combined_hidden, positions)
        # Attention metadata=None makes the old backend return zeros while
        # still profiling the draft transformer allocations.
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=query_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
        ):
            hidden = self.model(
                input_ids=self.input_ids[:query_tokens],
                positions=self._parallel_dummy_positions[:query_tokens],
            )
            sample_count = profile_reqs * self.num_speculative_tokens
            sample_hidden = hidden[
                self._parallel_dummy_sample_indices[:sample_count]
            ]
            logits = self.model.compute_logits(sample_hidden)
            logits.argmax(dim=-1)
