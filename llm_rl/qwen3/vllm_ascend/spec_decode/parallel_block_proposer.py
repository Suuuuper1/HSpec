# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Old-ABI base for one-forward parallel-block speculative drafters."""

import copy
import json
from dataclasses import dataclass, replace
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
from vllm.logger import logger
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.context_kv import PAD_SLOT_ID
from vllm_ascend.ops.triton.spec_decode.utils import compute_rejected_tokens_kernel
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.probabilistic import (
    estimate_draft_probability_bytes,
    sample_draft_logits,
)
from vllm_ascend.spec_decode.parallel_draft_metrics import (
    DraftDPObserver,
    ParallelDraftMetrics,
    dp_repair_capability_manifest,
    phase5_capability_manifest,
)


@dataclass(frozen=True, slots=True)
class DraftDPPlan:
    """Validated execution lengths for one parallel-draft forward.

    ``num_tokens_across_dp`` is borrowed from the runner and is valid for the
    lifetime of the forward. Keeping the reference avoids a per-step clone.
    """

    num_actual_tokens: int
    num_tokens: int
    num_padding_tokens: int
    num_tokens_across_dp: Optional[torch.Tensor]


def _format_dp_plan_context(
    *,
    method: str,
    batch_size: int,
    num_speculative_tokens: int,
    num_queries: int,
    num_actual_tokens: int,
    num_tokens: int,
    dp_size: int,
    dp_rank: int,
    num_tokens_across_dp: Optional[torch.Tensor],
    max_query_tokens: int,
) -> str:
    try:
        padding: object = num_tokens - num_actual_tokens
    except TypeError:
        padding = "<invalid>"
    if isinstance(num_tokens_across_dp, torch.Tensor):
        if num_tokens_across_dp.device.type == "cpu":
            counts: object = tuple(num_tokens_across_dp.tolist())
        else:
            counts = (
                f"Tensor(device={num_tokens_across_dp.device}, "
                f"dtype={num_tokens_across_dp.dtype}, "
                f"shape={tuple(num_tokens_across_dp.shape)})"
            )
    else:
        counts = num_tokens_across_dp
    return (
        f"method={method}, B={batch_size}, K={num_speculative_tokens}, "
        f"Q={num_queries}, actual={num_actual_tokens}, exec={num_tokens}, "
        f"pad={padding}, DP size/rank={dp_size}/{dp_rank}, counts={counts}, "
        f"capacity={max_query_tokens}"
    )


def validate_draft_dp_plan(
    *,
    method: str,
    batch_size: int,
    num_speculative_tokens: int,
    num_queries: int,
    num_actual_tokens: int,
    num_tokens: int,
    dp_size: int,
    dp_rank: int,
    num_tokens_across_dp: Optional[torch.Tensor],
    max_query_tokens: int,
) -> DraftDPPlan:
    """Validate the logical/execution split without touching an accelerator."""

    scalar_values = {
        "B": batch_size,
        "K": num_speculative_tokens,
        "Q": num_queries,
        "actual": num_actual_tokens,
        "exec": num_tokens,
        "DP size": dp_size,
        "DP rank": dp_rank,
        "capacity": max_query_tokens,
    }

    def context() -> str:
        # Keep counts materialization off the successful per-step path.
        return _format_dp_plan_context(
            method=method,
            batch_size=batch_size,
            num_speculative_tokens=num_speculative_tokens,
            num_queries=num_queries,
            num_actual_tokens=num_actual_tokens,
            num_tokens=num_tokens,
            dp_size=dp_size,
            dp_rank=dp_rank,
            num_tokens_across_dp=num_tokens_across_dp,
            max_query_tokens=max_query_tokens,
        )

    for name, value in scalar_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Draft DP plan {name} must be an int; {context()}")
    if not method:
        raise ValueError(f"Draft DP plan method must be non-empty; {context()}")
    if batch_size < 0 or num_speculative_tokens <= 0 or num_queries <= 0:
        raise ValueError(f"Draft DP plan has invalid B/K/Q; {context()}")
    if dp_size <= 0 or not 0 <= dp_rank < dp_size:
        raise ValueError(f"Draft DP plan has invalid DP topology; {context()}")
    if max_query_tokens < 0:
        raise ValueError(f"Draft DP plan has invalid capacity; {context()}")
    if num_actual_tokens != batch_size * num_queries:
        raise ValueError(f"Draft DP plan violates actual=B*Q; {context()}")
    if not 0 <= num_actual_tokens <= num_tokens <= max_query_tokens:
        raise ValueError(
            "Draft DP plan violates 0<=actual<=exec<=capacity; " + context()
        )

    if dp_size == 1:
        if num_tokens_across_dp is not None:
            raise ValueError(f"Draft DP1 plan requires counts=None; {context()}")
        if num_tokens != num_actual_tokens:
            raise ValueError(f"Draft DP1 plan requires exec=actual; {context()}")
    else:
        counts = num_tokens_across_dp
        if not isinstance(counts, torch.Tensor):
            raise TypeError(f"Draft DP>1 plan requires a counts tensor; {context()}")
        if counts.device.type != "cpu":
            raise ValueError(f"Draft DP counts must remain on CPU; {context()}")
        if counts.dtype != torch.int32:
            raise TypeError(f"Draft DP counts must use torch.int32; {context()}")
        if counts.ndim != 1 or counts.numel() != dp_size:
            raise ValueError(
                f"Draft DP counts must have shape [{dp_size}]; {context()}"
            )
        if counts[dp_rank].item() != num_tokens:
            raise ValueError(f"Draft DP local count must equal exec; {context()}")

    return DraftDPPlan(
        num_actual_tokens=num_actual_tokens,
        num_tokens=num_tokens,
        num_padding_tokens=num_tokens - num_actual_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def validate_homogeneous_draft_capacities(
    capacities: tuple[int, ...],
    *,
    dp_size: int,
    local_capacity: int,
) -> None:
    """Fail closed before execution when DP workers preallocate differently."""

    if (
        isinstance(dp_size, bool)
        or not isinstance(dp_size, int)
        or dp_size <= 0
        or isinstance(local_capacity, bool)
        or not isinstance(local_capacity, int)
        or local_capacity < 0
        or len(capacities) != dp_size
        or any(
            isinstance(capacity, bool) or not isinstance(capacity, int)
            for capacity in capacities
        )
    ):
        raise ValueError(
            "Draft DP capacities must contain one integer per rank: "
            f"DP size={dp_size}, capacities={capacities}, "
            f"local capacity={local_capacity}"
        )
    if any(capacity != local_capacity for capacity in capacities):
        raise ValueError(
            "Heterogeneous draft DP capacities are unsupported: "
            f"DP size={dp_size}, capacities={capacities}, "
            f"local capacity={local_capacity}"
        )


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
        self._checkpoint_load_count = 0
        if vllm_config.parallel_config.data_parallel_size > 1:
            raise NotImplementedError("Phase 1/2 parallel drafters certify DP=1 only")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "Phase 1/2 parallel drafters require async_scheduling=False"
            )
        # Full graph still hard-codes causal FIA arguments in this ABI.
        self.use_cuda_graph = False
        self._runnable = self._run_parallel_backbone
        self.max_batch_size = runner.max_num_reqs
        self.num_queries = self.speculative_config.parallel_query_count
        self.max_query_tokens = self.max_batch_size * self.num_queries
        self._probabilistic = (
            self.speculative_config.draft_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None
        self._draft_model_kind = (
            "moe" if self.speculative_config.draft_model_config.is_moe else "dense"
        )
        parallel_config = self.vllm_config.parallel_config
        self._draft_dp_observer = DraftDPObserver(
            method=self.speculative_config.method,
            dp_size=parallel_config.data_parallel_size,
            dp_rank=parallel_config.data_parallel_rank,
            draft_model_kind=self._draft_model_kind,
            sample_every=self.speculative_config.parallel_draft_profile_sample_every,
        )
        self._last_draft_dp_sequence: int | None = None
        self._phase5_metrics = ParallelDraftMetrics(
            method=self.speculative_config.method,
            enabled=self.speculative_config.parallel_draft_profile_enabled,
            sample_every=self.speculative_config.parallel_draft_profile_sample_every,
            flush_every=self.speculative_config.parallel_draft_profile_flush_every,
        )
        logger.info(
            "PHASE5_DRAFT_CAPABILITY %s",
            json.dumps(
                phase5_capability_manifest(
                    self.speculative_config,
                    vllm_config=self.vllm_config,
                    draft_model_kind=self._draft_model_kind,
                ),
                sort_keys=True,
            ),
        )
        logger.info(
            "DP_REPAIR_DRAFT_CAPABILITY %s",
            json.dumps(
                dp_repair_capability_manifest(
                    self.speculative_config,
                    self.vllm_config,
                    draft_model_kind=self._draft_model_kind,
                ),
                sort_keys=True,
            ),
        )
        self._draft_probability_memory: dict[str, int] | None = None
        if self._probabilistic:
            vocab_size = self.speculative_config.draft_model_config.get_vocab_size()
            estimate = estimate_draft_probability_bytes(
                self.max_batch_size,
                self.num_speculative_tokens,
                vocab_size,
                method=self.speculative_config.method,
            )
            budget = self.speculative_config.draft_probability_max_bytes
            if budget is None or estimate["peak_bytes"] > budget:
                raise MemoryError(
                    "probabilistic draft working set exceeds the explicit startup "
                    f"budget: estimate={estimate['peak_bytes']} bytes, budget={budget}, "
                    f"B={self.max_batch_size}, K={self.num_speculative_tokens}, "
                    f"V={vocab_size}"
                )
            self._draft_probability_memory = {**estimate, "budget_bytes": budget}
            logger.info(
                "PHASE4_Q_MEMORY method=%s B=%d K=%d V=%d q_bytes=%d "
                "peak_bytes=%d budget_bytes=%d",
                self.speculative_config.method,
                self.max_batch_size,
                self.num_speculative_tokens,
                vocab_size,
                estimate["probability_bytes"],
                estimate["peak_bytes"],
                budget,
            )
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
            + self.speculative_config.draft_sample_offset
        )

    @property
    def phase5_metrics(self) -> ParallelDraftMetrics:
        """Keep pure-function tests that construct proposers via ``__new__`` valid."""
        metrics = getattr(self, "_phase5_metrics", None)
        if metrics is None:
            metrics = ParallelDraftMetrics(
                method=getattr(self, "method", "parallel"),
                enabled=False,
                sample_every=1,
                flush_every=1,
            )
            self._phase5_metrics = metrics
        return metrics

    def _validate_dp_plan(
        self,
        *,
        batch_size: int,
        num_actual_tokens: int,
        num_tokens: int,
        num_tokens_across_dp: Optional[torch.Tensor],
    ) -> DraftDPPlan:
        parallel_config = self.vllm_config.parallel_config
        return validate_draft_dp_plan(
            method=self.method,
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            num_queries=self.num_queries,
            num_actual_tokens=num_actual_tokens,
            num_tokens=num_tokens,
            dp_size=parallel_config.data_parallel_size,
            dp_rank=parallel_config.data_parallel_rank,
            num_tokens_across_dp=num_tokens_across_dp,
            max_query_tokens=self.max_query_tokens,
        )

    @property
    def draft_dp_observer(self) -> DraftDPObserver:
        """Return the DP observer, including for lightweight ``__new__`` tests."""

        observer = getattr(self, "_draft_dp_observer", None)
        if observer is None:
            parallel = self.vllm_config.parallel_config
            observer = DraftDPObserver(
                method=getattr(self, "method", "parallel"),
                dp_size=parallel.data_parallel_size,
                dp_rank=parallel.data_parallel_rank,
                draft_model_kind=getattr(self, "_draft_model_kind", "dense"),
                sample_every=64,
            )
            self._draft_dp_observer = observer
        return observer

    def _coordinate_draft_dp(
        self,
        *,
        batch_size: int,
        num_actual_tokens: int,
        kind: str,
    ) -> DraftDPPlan:
        """Run exactly one old-runner draft sync and validate its result."""

        observer = self.draft_dp_observer
        sequence, start_ns = observer.begin_sync(kind)
        num_tokens: int | None = None
        try:
            num_tokens, counts, _ = self.runner._sync_metadata_across_dp(
                num_actual_tokens,
                is_draft_model=True,
            )
            plan = self._validate_dp_plan(
                batch_size=batch_size,
                num_actual_tokens=num_actual_tokens,
                num_tokens=num_tokens,
                num_tokens_across_dp=counts,
            )
        except Exception:
            observer.finish_sync(
                sequence=sequence,
                kind=kind,
                start_ns=start_ns,
                batch_size=batch_size,
                num_queries=self.num_queries,
                num_actual_tokens=num_actual_tokens,
                num_tokens=num_tokens,
                num_padding_tokens=None,
                success=False,
            )
            raise
        observer.finish_sync(
            sequence=sequence,
            kind=kind,
            start_ns=start_ns,
            batch_size=batch_size,
            num_queries=self.num_queries,
            num_actual_tokens=num_actual_tokens,
            num_tokens=plan.num_tokens,
            num_padding_tokens=plan.num_padding_tokens,
            success=True,
        )
        self._last_draft_dp_sequence = sequence
        return plan

    def _finish_draft_dp_context(self, *, success: bool) -> None:
        sequence = self._last_draft_dp_sequence
        if sequence is None:
            return
        self.draft_dp_observer.finish_context(sequence, success=success)
        self._last_draft_dp_sequence = None

    def _begin_draft_dp_context(self) -> None:
        sequence = self._last_draft_dp_sequence
        if sequence is not None:
            self.draft_dp_observer.begin_context(sequence)

    def flush_observability_metrics(self) -> dict[str, object]:
        """Publish DP evidence; an enabled legacy Phase-5 flush may sync events."""

        phase5 = self.phase5_metrics.flush()
        dp = self.draft_dp_observer.snapshot()
        logger.info("DP_REPAIR_DRAFT_METRICS %s", json.dumps(dp, sort_keys=True))
        return {"phase5": phase5, "dp": dp}

    def _pad_draft_buffers(
        self,
        *,
        num_actual_tokens: int,
        num_tokens: int,
    ) -> None:
        """Sanitize only the execution-only tail of persistent query buffers."""

        if not 0 <= num_actual_tokens <= num_tokens <= self.max_query_tokens:
            raise ValueError(
                "Invalid draft padding range: "
                f"method={self.method}, actual={num_actual_tokens}, "
                f"exec={num_tokens}, capacity={self.max_query_tokens}"
            )
        if num_tokens == num_actual_tokens:
            return

        buffers = (
            ("_query_input_ids", self.model.model.mask_token_id),
            ("_query_positions", 0),
            ("_query_slots", PAD_SLOT_ID),
        )
        validated_buffers: list[tuple[torch.Tensor, int]] = []
        for name, value in buffers:
            buffer = getattr(self, name, None)
            if not isinstance(buffer, torch.Tensor) or buffer.ndim != 1:
                raise RuntimeError(
                    f"Draft padding requires one-dimensional {name}; "
                    f"method={self.method}, actual={num_actual_tokens}, "
                    f"exec={num_tokens}, capacity={self.max_query_tokens}"
                )
            if buffer.shape[0] < num_tokens:
                raise RuntimeError(
                    f"Draft padding exceeds {name} capacity={buffer.shape[0]}; "
                    f"method={self.method}, actual={num_actual_tokens}, "
                    f"exec={num_tokens}, capacity={self.max_query_tokens}"
                )
            validated_buffers.append((buffer, value))

        group_buffers = tuple(
            getattr(self, "_per_group_query_slot_mapping_buffers", {}).values()
        )
        for buffer in group_buffers:
            if buffer.ndim != 1 or buffer.shape[0] < num_tokens:
                raise RuntimeError(
                    "Draft per-group query slot buffer cannot cover padding: "
                    f"method={self.method}, actual={num_actual_tokens}, "
                    f"exec={num_tokens}, shape={tuple(buffer.shape)}"
                )

        for buffer, value in validated_buffers:
            buffer[num_actual_tokens:num_tokens].fill_(value)
        for buffer in group_buffers:
            buffer[num_actual_tokens:num_tokens].fill_(PAD_SLOT_ID)

    def _pad_and_build_parallel_attention_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        rejected_counts: torch.Tensor,
        plan: DraftDPPlan,
    ) -> AscendMetadata:
        """Preserve the required padding-before-metadata ordering."""

        self._pad_draft_buffers(
            num_actual_tokens=plan.num_actual_tokens,
            num_tokens=plan.num_tokens,
        )
        return self.build_parallel_attention_metadata(
            common_attn_metadata,
            self._query_slots[: plan.num_tokens],
            rejected_counts,
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
        if self._checkpoint_load_count != 0:
            raise RuntimeError(
                f"{type(self).__name__} draft checkpoint reload is forbidden; "
                "the draft is immutable across the Verl RL lifecycle"
            )
        draft_vllm_config = self._draft_vllm_config()
        model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_vllm_config.model_config,
        )
        self._checkpoint_load_count = 1
        return model

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
        grid = (max(1, triton.cdiv(num_reqs, block_size)),)
        compute_rejected_tokens_kernel[grid](
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
        anchor_token_ids: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        with self.phase5_metrics.timer("spec/draft_backbone_ms"):
            hidden_states = self.model(input_ids=input_ids, positions=positions)
        sample_hidden_states = hidden_states[sample_indices]
        return self._select_draft_tokens(
            sample_hidden_states, anchor_token_ids, sampling_metadata
        )

    def _select_draft_tokens(
        self,
        sample_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor | None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        del anchor_token_ids
        with self.phase5_metrics.timer("spec/draft_lm_head_ms"):
            logits = self.model.compute_logits(sample_hidden_states)
        if not getattr(self, "_probabilistic", False) or sampling_metadata is None:
            self._last_draft_probs = None
            token_ids = logits.argmax(dim=-1)
        else:
            if sampling_metadata.temperature is None:
                raise RuntimeError(
                    "probabilistic draft proposal requires per-request temperature"
                )
            token_ids, probabilities = sample_draft_logits(
                logits,
                sampling_metadata.temperature,
                rows_per_request=self.num_speculative_tokens,
                all_greedy=sampling_metadata.all_greedy,
                all_random=sampling_metadata.all_random,
                generators=sampling_metadata.generators,
            )
            self._last_draft_probs = (
                None
                if probabilities is None
                else probabilities.view(
                    -1, self.num_speculative_tokens, probabilities.shape[-1]
                ).contiguous()
            )
        return token_ids.view(-1, self.num_speculative_tokens)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        probabilities = self._last_draft_probs
        self._last_draft_probs = None
        return probabilities

    def probability_memory_manifest(self) -> dict[str, int] | None:
        return (
            None
            if self._draft_probability_memory is None
            else dict(self._draft_probability_memory)
        )

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
            num_tokens_across_dp,
            aclgraph_runtime_mode,
            batch_descriptor,
            dummy_compute_logits,
        )
        context_tokens = min(num_tokens, self.max_num_tokens)
        dummy_reqs = min(max(num_reqs, 1), self.max_batch_size)
        num_actual_tokens = dummy_reqs * self.num_queries
        kind = "profile" if is_profile else "dummy"
        plan = self._coordinate_draft_dp(
            batch_size=dummy_reqs,
            num_actual_tokens=num_actual_tokens,
            kind=kind,
        )
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

        mask_token_id = self.model.model.mask_token_id
        self._query_input_ids[:num_actual_tokens].fill_(mask_token_id)
        self._query_positions[:num_actual_tokens].zero_()
        self._query_slots[:num_actual_tokens].fill_(PAD_SLOT_ID)
        for slot_buffer in getattr(
            self, "_per_group_query_slot_mapping_buffers", {}
        ).values():
            slot_buffer[:num_actual_tokens].fill_(PAD_SLOT_ID)
        self._pad_draft_buffers(
            num_actual_tokens=num_actual_tokens,
            num_tokens=plan.num_tokens,
        )

        context_success = False
        try:
            self._begin_draft_dp_context()
            with set_ascend_forward_context(
                None,
                self.vllm_config,
                num_tokens=plan.num_tokens,
                num_tokens_across_dp=plan.num_tokens_across_dp,
                # Runtime dummy participates in model/DP communication but must
                # not expose any row as a cache-writing logical token.
                num_actual_tokens=(plan.num_actual_tokens if is_profile else 0),
                in_profile_run=is_profile,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                is_draft_model=True,
            ):
                hidden = self.model(
                    input_ids=self._query_input_ids[: plan.num_tokens],
                    positions=self._query_positions[: plan.num_tokens],
                )
                sample_count = dummy_reqs * self.num_speculative_tokens
                sample_hidden = hidden[
                    self._parallel_dummy_sample_indices[:sample_count]
                ]
                anchor_tokens = self._query_input_ids[
                    :num_actual_tokens:self.num_queries
                ]
                # No SamplingMetadata means deterministic argmax/Markov scratch:
                # dummy never consumes proposal RNG or publishes q.
                self._select_draft_tokens(sample_hidden, anchor_tokens, None)
            context_success = True
        finally:
            self._finish_draft_dp_context(success=context_success)
