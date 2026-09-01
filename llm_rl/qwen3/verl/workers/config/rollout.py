# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "RolloutConfig",
    "resolve_rollout_speculative_method",
]


_ROLLOUT_SPECULATIVE_METHODS = {"hspec", "dflash", "dspark"}


def resolve_rollout_speculative_method(config) -> Optional[str]:
    """Resolve the unified Phase-3 method, including the legacy HSpec switch.

    Keeping this pure helper in the config layer gives the trainer and rollout
    worker exactly the same interpretation without importing vLLM.
    """
    explicit = config.get("speculative_method", None)
    if isinstance(explicit, str):
        explicit = explicit.strip().lower() or None
    if explicit not in _ROLLOUT_SPECULATIVE_METHODS | {None}:
        raise ValueError(
            "speculative_method must be one of null, hspec, dflash or dspark; "
            f"got {explicit!r}"
        )
    legacy_hspec = bool(config.get("use_hspec_decode", False))
    if legacy_hspec and explicit is not None:
        raise ValueError(
            "use_hspec_decode and speculative_method are mutually exclusive; "
            "remove the legacy switch when using the unified field"
        )
    return "hspec" if legacy_hspec else explicit


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: Optional[str] = MISSING
    mode: str = "sync"
    seed: int = 0
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_mode: Optional[str] = None
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    # HSpec (Hidden State based Speculative Decoding)
    # These are optional knobs; when disabled they add near-zero overhead.
    use_hspec_decode: bool = False
    hspec_num_speculative_tokens: int = 5
    hspec_similarity_threshold: float = 0.9
    hspec_min_match_len: int = 1
    # Table build parameters (trainer-side; used by HSpecTableGroup)
    hspec_n_components: int = 64
    hspec_max_entries_per_prompt: int = 10000

    # Unified speculative decoding contract. ``use_hspec_decode`` remains a
    # compatibility-only alias and cannot be combined with these methods.
    speculative_method: Optional[str] = None
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 7
    draft_tensor_parallel_size: int = 1
    draft_sample_method: str = "greedy"
    draft_probability_max_memory_mb: int = 2048
    draft_load_format: str = "auto"
    rejection_sample_method: str = "standard"
    speculative_enforce_eager: bool = True
    parallel_draft_profile_enabled: bool = False
    parallel_draft_profile_sample_every: int = 64
    parallel_draft_profile_flush_every: int = 4
    parallel_draft_incremental_context_kv: bool = False
    parallel_draft_dynamic_k: bool = False
    # Keep the Phase-3 graph-certified surface unchanged by default.  The
    # 30B eager comparison must opt in explicitly and is treated as a new
    # runtime certification candidate rather than production support.
    parallel_draft_allow_target_eager_experiment: bool = False
    dspark_draft_topk: Optional[int] = None

    # Disabled by default. The Phase-3 evidence run enables this to emit a
    # bounded sampled checksum and state-machine record at lifecycle edges.
    speculative_lifecycle_audit: bool = False
    speculative_lifecycle_strict: bool = True
    speculative_lifecycle_samples_per_parameter: int = 8

    def __post_init__(self):
        """Validate the rollout config"""
        method = resolve_rollout_speculative_method(self)
        if not 1 <= self.num_speculative_tokens <= 15:
            raise ValueError("num_speculative_tokens must be in [1, 15]")
        if not 1 <= self.speculative_lifecycle_samples_per_parameter <= 64:
            raise ValueError(
                "speculative_lifecycle_samples_per_parameter must be in [1, 64]"
            )
        if method is None and self.speculative_model is not None:
            raise ValueError(
                "speculative_model requires speculative_method=dflash or dspark"
            )
        if method == "hspec" and self.speculative_model is not None:
            raise ValueError("HSpec does not accept speculative_model")
        if method in {"dflash", "dspark"}:
            if not self.speculative_model:
                raise ValueError(f"{method} requires speculative_model")
            if self.name != "vllm" or self.mode != "sync":
                raise ValueError(
                    f"{method} Phase-3 support requires name=vllm and mode=sync"
                )
            if self.tensor_model_parallel_size < 1:
                raise ValueError("target tensor parallel size must be positive")
            if self.draft_tensor_parallel_size != 1:
                raise NotImplementedError(
                    "Phase-3 DFlash/DSpark supports draft_tensor_parallel_size=1 only"
                )
            if (
                self.draft_sample_method == "probabilistic"
                and self.tensor_model_parallel_size != 1
            ):
                raise NotImplementedError(
                    "DFlash/DSpark probabilistic proposal requires target "
                    "tensor_model_parallel_size=1 until cross-rank RNG/proposal "
                    "equivalence is certified"
                )
            if self.pipeline_model_parallel_size != 1 or self.data_parallel_size != 1:
                raise NotImplementedError(
                    "Phase-3 DFlash/DSpark supports vLLM PP=1 and DP=1 only"
                )
            if self.draft_sample_method not in {"greedy", "probabilistic"}:
                raise NotImplementedError(
                    "DFlash/DSpark draft_sample_method must be greedy or probabilistic"
                )
            if self.draft_probability_max_memory_mb <= 0:
                raise ValueError("draft_probability_max_memory_mb must be positive")
            if self.rejection_sample_method != "standard":
                raise NotImplementedError(
                    "Phase-3 DFlash/DSpark requires standard rejection sampling"
                )
            if self.draft_load_format.lower() != "auto":
                raise NotImplementedError(
                    "Phase-3 DFlash/DSpark certifies draft_load_format=auto only"
                )
            if not self.speculative_enforce_eager:
                raise NotImplementedError(
                    "DFlash/DSpark draft full graph is not available in the old "
                    "Ascend ABI and must remain eager; keep "
                    "speculative_enforce_eager=true"
                )
            if self.parallel_draft_profile_sample_every <= 0:
                raise ValueError("parallel_draft_profile_sample_every must be positive")
            if self.parallel_draft_profile_flush_every <= 0:
                raise ValueError("parallel_draft_profile_flush_every must be positive")
            if self.parallel_draft_incremental_context_kv:
                raise NotImplementedError(
                    "parallel_draft_incremental_context_kv lacks old-ABI request/block "
                    "generation ownership and remains fail-closed"
                )
            if self.parallel_draft_dynamic_k:
                raise NotImplementedError(
                    "parallel_draft_dynamic_k is incompatible with the old fixed-K "
                    "scheduler/verifier ABI and remains fail-closed"
                )
            if self.dspark_draft_topk is not None:
                if method != "dspark":
                    raise ValueError("dspark_draft_topk is only valid for DSpark")
                raise NotImplementedError(
                    "dspark_draft_topk is not implemented by the reference DSpark "
                    "path and cannot enter the full-vocabulary fair baseline"
                )
            if (
                self.enforce_eager
                and not self.parallel_draft_allow_target_eager_experiment
            ):
                raise ValueError(
                    "Phase-3 DFlash/DSpark keeps the target graph enabled; "
                    "set rollout.enforce_eager=false, or explicitly set "
                    "parallel_draft_allow_target_eager_experiment=true for an "
                    "eager-only certification experiment"
                )
            if self.enable_prefix_caching:
                raise ValueError(
                    "Phase-3 DFlash/DSpark requires enable_prefix_caching=false"
                )
            if not self.free_cache_engine:
                raise ValueError(
                    "Phase-3 lifecycle certification requires free_cache_engine=true"
                )
            async_scheduling = (
                self.engine_kwargs.get("vllm", {}).get("async_scheduling", False)
                if self.engine_kwargs
                else False
            )
            if async_scheduling:
                raise ValueError("Phase-3 DFlash/DSpark requires async_scheduling=false")
        elif (
            self.parallel_draft_profile_enabled
            or self.parallel_draft_incremental_context_kv
            or self.parallel_draft_dynamic_k
            or self.parallel_draft_allow_target_eager_experiment
            or self.dspark_draft_topk is not None
        ):
            raise ValueError(
                "parallel draft Phase-5 controls require speculative_method=dflash or dspark"
            )

        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
