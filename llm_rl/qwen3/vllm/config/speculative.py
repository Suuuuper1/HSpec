# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_effective_rope_parameters
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "ernie_mtp",
    "exaone_moe_mtp",
    "qwen3_next_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
]
EagleModelTypes = Literal["eagle", "eagle3", MTPModelTypes]
ParallelBlockModelTypes = Literal["dflash", "dspark"]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    "sam",
    "hspec",
    ParallelBlockModelTypes,
    EagleModelTypes,
]
DraftSampleMethod = Literal["greedy", "probabilistic"]
RejectionSampleMethod = Literal["legacy", "standard"]


@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    # HSpec proposer configuration (optional; plugin-defined).
    # Kept in SpeculativeConfig so EngineArgs -> SpeculativeConfig validation
    # accepts these fields when method="hspec".
    hspec_similarity_threshold: float | None = None
    hspec_min_match_len: int | None = None
    hspec_n_components: int | None = Field(default=None, ge=1)
    hspec_max_entries_per_prompt: int | None = Field(default=None, ge=1)
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""
    draft_load_config: LoadConfig | None = None
    """Independent draft-weight loader. Neural drafters used by a Verl hybrid
    engine must not inherit the target-side dummy loader."""
    draft_sample_method: DraftSampleMethod = "greedy"
    """Draft-token sampling mode."""
    draft_probability_max_bytes: int | None = Field(default=None, ge=1)
    """Explicit upper bound for the probabilistic draft working set. Required
    for DFlash/DSpark probabilistic proposal so an oversized [B,K,V] q buffer is
    rejected at proposer startup instead of failing inside a rollout."""
    rejection_sample_method: RejectionSampleMethod | None = None
    """Verifier mode. Unset retains the historical mode for old methods and
    resolves to standard rejection for DFlash/DSpark."""
    sample_from_anchor: bool | None = None
    """DSpark layout override. If unset, the checkpoint value is authoritative."""
    parallel_draft_profile_enabled: bool = False
    """Enable sampled Phase-5 device timers for DFlash/DSpark."""
    parallel_draft_profile_sample_every: int = Field(default=64, ge=1)
    """Record one detailed parallel-draft proposal every N proposals."""
    parallel_draft_profile_flush_every: int = Field(default=4, ge=1)
    """Synchronize and publish after this many sampled proposals."""
    parallel_draft_incremental_context_kv: bool = False
    """Reserved Phase-5 capability. The old ABI currently rejects True."""
    parallel_draft_dynamic_k: bool = False
    """Reserved Phase-5 capability. The old fixed-K ABI rejects True."""
    dspark_draft_topk: int | None = Field(default=None, ge=1)
    """Reserved explicit DSpark algorithm variant; not a baseline shortcut."""

    # Advanced control
    disable_by_batch_size: int | None = Field(default=None, ge=2)
    """Disable speculative decoding for new incoming requests when the number
    of enqueued requests is larger than this value, if provided."""
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    @model_validator(mode="before")
    @classmethod
    def _validate_parallel_block_k_early(cls, values: Any) -> Any:
        """Give explicit-method users the certified NPU K error before Field parsing."""
        kwargs = getattr(values, "kwargs", values)
        if isinstance(kwargs, dict) and kwargs.get("method") in ("dflash", "dspark"):
            k = kwargs.get("num_speculative_tokens")
            if not isinstance(k, int) or not 1 <= k <= 15:
                raise ValueError(
                    "DFlash/DSpark Ascend NPU certified range requires "
                    "1 <= num_speculative_tokens <= 15"
                )
        return values

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        if not self.uses_parallel_block_drafter():
            # Preserve the pre-migration hash exactly for all existing methods.
            factors: list[Any] = [self.method == "eagle3"]
            return safe_hash(
                str(factors).encode(), usedforsecurity=False
            ).hexdigest()

        factors = [
            self.method,
            self.num_speculative_tokens,
            self.parallel_query_count,
            self.sample_from_anchor,
            self.needs_aux_hidden_states(),
        ]
        if self.uses_parallel_block_drafter():
            factors.append(self.draft_sample_method)
        if self.draft_model_config is not None:
            factors.append(self.draft_model_config.compute_hash())
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                factors.append(tuple(layer_ids))
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )

        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudgraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method in ("hspec", "[hspec]"):
                # Non-model proposer implemented by platform plugins.
                self.model = "hspec"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "sam":
                self.model = "sam"
            elif self.method in ("dflash", "dspark"):
                raise ValueError(
                    f"{self.method} requires an independent draft checkpoint model path"
                )
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (
            self.model is not None and self.model in ("ngram", "[ngram]")
        ):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "sam":
            # SAM is a stateful suffix-automaton proposer, not an n-gram
            # prompt lookup alias. Keep the method identity intact so platform
            # plugins can instantiate the correct proposer.
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method in ("hspec", "[hspec]"):
            # Non-model proposer implemented by platform plugins.
            # No prompt-lookup parameters are required for HSpec.
            self.method = "hspec"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            # Keep draft configs identical to target so downstream verification
            # and parallel-config checks pass.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash", "dspark"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif (
                    "DFlashDraftModel" in self.draft_model_config.architectures
                    or "dflash" in self.draft_model_config.model.lower()
                ):
                    self.method = "dflash"
                elif (
                    any(
                        arch in self.draft_model_config.architectures
                        for arch in ("Qwen3DSparkModel", "DSparkDraftModel")
                    )
                    or "dspark" in self.draft_model_config.model.lower()
                ):
                    self.method = "dspark"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run"
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                else:
                    self.method = "draft_model"
                    raise NotImplementedError(
                        "Speculative decoding with draft model is not "
                        "supported yet. Please consider using other "
                        "speculative decoding methods such as ngram, medusa, "
                        "eagle, or mtp."
                    )

                if self.uses_parallel_block_drafter():
                    self._validate_parallel_block_checkpoint()

                # Replace hf_config for EAGLE and DFlash draft models.
                if self.method in ("eagle", "eagle3", "dflash"):
                    from vllm.transformers_utils.configs import SpeculatorsConfig
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.draft_model_config.model_arch_config = (
                            self.draft_model_config.get_model_arch_config()
                        )

                if (
                    self.method == "dspark"
                    and "DSparkDraftModel" in self.draft_model_config.architectures
                ):
                    self.draft_model_config.hf_config.architectures = [
                        "Qwen3DSparkModel"
                    ]
                    self.draft_model_config.model_arch_config = (
                        self.draft_model_config.get_model_arch_config()
                    )

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.speculative_token_tree is None:
                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
                if self.uses_parallel_block_drafter():
                    self._resolve_parallel_block_options()
        return self

    def _validate_parallel_block_checkpoint(self) -> None:
        if self.target_model_config is None or self.draft_model_config is None:
            raise ValueError(f"{self.method} requires target and draft model configs")

        target = self.target_model_config
        draft = self.draft_model_config
        target_type = getattr(target.hf_text_config, "model_type", None)
        draft_type = getattr(draft.hf_text_config, "model_type", None)
        if target_type not in ("qwen3", "qwen3_moe") or draft_type != "qwen3":
            raise ValueError(
                f"{self.method} Phase 0 only supports Qwen3/Qwen3-MoE targets "
                f"with a Qwen3 draft; got target={target_type!r}, draft={draft_type!r}"
            )

        architectures = set(draft.architectures)
        allowed = (
            {"DFlashDraftModel"}
            if self.use_dflash()
            else {"Qwen3DSparkModel", "DSparkDraftModel"}
        )
        if not architectures.intersection(allowed):
            raise ValueError(
                f"Unsupported {self.method} draft architecture {sorted(architectures)}; "
                f"expected one of {sorted(allowed)}"
            )

        target_vocab = target.get_vocab_size()
        draft_vocab = draft.get_vocab_size()
        if target_vocab != draft_vocab:
            raise ValueError(
                f"{self.method} requires full equal vocabularies; "
                f"target={target_vocab}, draft={draft_vocab}"
            )

        for field in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            target_value = getattr(target.hf_text_config, field, None)
            draft_value = getattr(draft.hf_text_config, field, None)
            if target_value != draft_value:
                raise ValueError(
                    f"{self.method} target/draft {field} mismatch: "
                    f"{target_value!r} != {draft_value!r}"
                )

        target_rope = get_effective_rope_parameters(target.hf_text_config)
        draft_rope = get_effective_rope_parameters(draft.hf_text_config)
        if target_rope != draft_rope:
            raise ValueError(
                f"{self.method} target/draft effective RoPE parameters mismatch: "
                f"{target_rope!r} != {draft_rope!r}"
            )

        for token_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            target_id = getattr(target.hf_text_config, token_name, None)
            draft_id = getattr(draft.hf_text_config, token_name, None)
            if target_id is not None and draft_id is not None and target_id != draft_id:
                raise ValueError(
                    f"{self.method} target/draft {token_name} mismatch: "
                    f"{target_id} != {draft_id}"
                )

        layer_types = getattr(draft.hf_text_config, "layer_types", None) or []
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError(
                f"{self.method} Phase 0 only supports uniform full attention; "
                f"got layer_types={layer_types}"
            )
        if getattr(draft.hf_text_config, "sliding_window", None) is not None:
            raise ValueError(f"{self.method} Phase 0 does not support sliding window")

        if self.use_dflash():
            raw_config = getattr(draft.hf_text_config, "dflash_config", {}) or {}
            if not isinstance(raw_config, dict):
                raise ValueError("dflash_config must be a mapping")
            layer_ids = raw_config.get("target_layer_ids")
            if not layer_ids:
                raise ValueError("DFlash checkpoint is missing dflash_config.target_layer_ids")
            mask_token_id = raw_config.get("mask_token_id")
            if (
                not isinstance(mask_token_id, int)
                or mask_token_id < 0
                or mask_token_id >= draft_vocab
            ):
                raise ValueError(
                    f"DFlash mask_token_id must be in [0, {draft_vocab}); "
                    f"got {mask_token_id!r}"
                )
            # The checkpoint ids address target *layer outputs*. The old Qwen
            # model records auxiliary states immediately before a layer, so
            # output i is captured at old-runner point i + 1.
            draft.hf_text_config.eagle_aux_hidden_state_layer_ids = [
                layer_id + 1 for layer_id in layer_ids
            ]
        else:
            layer_ids = getattr(
                draft.hf_text_config, "dspark_target_layer_ids", None
            ) or getattr(draft.hf_text_config, "target_layer_ids", None)
            if not layer_ids:
                raise ValueError("DSpark checkpoint is missing target aux layer ids")
            # Dense DSpark ids address target layer outputs. The old Qwen model
            # captures immediately before a layer, so output i is point i + 1.
            draft.hf_text_config.eagle_aux_hidden_state_layer_ids = [
                layer_id + 1 for layer_id in layer_ids
            ]
            mask_token_id = getattr(draft.hf_text_config, "mask_token_id", None)
            if (
                not isinstance(mask_token_id, int)
                or mask_token_id < 0
                or mask_token_id >= draft_vocab
            ):
                raise ValueError(
                    f"DSpark mask_token_id must be in [0, {draft_vocab}); "
                    f"got {mask_token_id!r}"
                )
            markov_rank = getattr(draft.hf_text_config, "markov_rank", None)
            if not isinstance(markov_rank, int) or markov_rank <= 0:
                raise ValueError(
                    f"DSpark checkpoint has invalid markov_rank={markov_rank!r}"
                )
            if getattr(draft.hf_text_config, "markov_head_type", "vanilla") != "vanilla":
                raise ValueError("Phase 2 DSpark only supports markov_head_type='vanilla'")

        target_num_layers = getattr(target.hf_text_config, "num_hidden_layers", None)
        if (
            not isinstance(target_num_layers, int)
            or any(
                not isinstance(layer_id, int)
                or layer_id < 0
                or layer_id >= target_num_layers
                for layer_id in layer_ids
            )
        ):
            raise ValueError(
                f"{self.method} aux layer ids {layer_ids!r} are outside the "
                f"target layer range [0, {target_num_layers})"
            )
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"{self.method} aux layer ids must be unique")
        if list(layer_ids) != sorted(layer_ids):
            raise ValueError(
                f"{self.method} aux layer ids must be in strictly increasing order"
            )
        if any(layer_id + 1 >= target_num_layers for layer_id in layer_ids):
            raise ValueError(
                f"{self.method} aux layer ids {layer_ids!r} cannot be represented "
                f"by the old-Qwen pre-layer capture range [1, {target_num_layers})"
            )
        draft_target_layers = getattr(
            draft.hf_text_config, "num_target_layers", None
        )
        if draft_target_layers not in (None, target_num_layers):
            raise ValueError(
                f"{self.method} num_target_layers mismatch: "
                f"{draft_target_layers} != {target_num_layers}"
            )

    def _resolve_parallel_block_options(self) -> None:
        assert self.draft_model_config is not None
        if self.draft_parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "DFlash/DSpark currently require draft_tensor_parallel_size=1; "
                "replicated dense draft weights are the only certified DP surface"
            )
        if self.num_speculative_tokens is None or not 1 <= self.num_speculative_tokens <= 15:
            raise ValueError(
                "DFlash/DSpark requires 1 <= num_speculative_tokens <= 15"
            )
        hspec_fields = {
            name: getattr(self, name)
            for name in (
                "hspec_similarity_threshold",
                "hspec_min_match_len",
                "hspec_n_components",
                "hspec_max_entries_per_prompt",
            )
            if getattr(self, name) is not None
        }
        if hspec_fields:
            raise ValueError(
                f"HSpec and {self.method} configuration are mutually exclusive; "
                f"remove {sorted(hspec_fields)}"
            )
        if self.draft_sample_method not in ("greedy", "probabilistic"):
            raise ValueError(
                "DFlash/DSpark draft_sample_method must be 'greedy' or "
                "'probabilistic'"
            )
        if self.draft_sample_method == "probabilistic":
            if self.draft_probability_max_bytes is None:
                raise ValueError(
                    "DFlash/DSpark probabilistic proposal requires an explicit "
                    "draft_probability_max_bytes startup budget"
                )
            if (
                self.target_parallel_config.tensor_parallel_size != 1
                or self.draft_parallel_config.tensor_parallel_size != 1
            ):
                raise NotImplementedError(
                    "DFlash/DSpark probabilistic proposal requires target TP=1 "
                    "and draft TP=1 until cross-rank RNG/proposal equivalence is "
                    "certified"
                )
        elif self.draft_probability_max_bytes is not None:
            raise ValueError(
                "draft_probability_max_bytes is only valid with "
                "draft_sample_method='probabilistic'"
            )
        if self.rejection_sample_method is None:
            self.rejection_sample_method = "standard"
        if self.rejection_sample_method != "standard":
            raise ValueError(
                "DFlash/DSpark requires rejection_sample_method='standard'; "
                "the legacy verifier is not valid for random RL rollout"
            )
        if self.enforce_eager is False:
            raise NotImplementedError(
                "DFlash/DSpark draft full graph is not supported by the old "
                "vLLM 0.14.1 + vLLM-Ascend 0.14.0rc1 ABI: draft attention "
                "capture/update still assumes causal sparse_mode=3. Keep "
                "speculative enforce_eager=true; target graph remains supported."
            )
        self.enforce_eager = True
        if self.parallel_draft_incremental_context_kv:
            raise NotImplementedError(
                "parallel_draft_incremental_context_kv is not certified: the old "
                "scheduler does not expose request/block generations needed for "
                "safe invalidation after reject, preempt, remap, or actor update"
            )
        if self.parallel_draft_dynamic_k:
            raise NotImplementedError(
                "parallel_draft_dynamic_k is not certified: the old scheduler and "
                "padded verifier ABI require one fixed K for every active request"
            )
        if self.dspark_draft_topk is not None:
            if not self.use_dspark():
                raise ValueError("dspark_draft_topk is only valid for method='dspark'")
            raise NotImplementedError(
                "dspark_draft_topk is an uncalibrated algorithm variant and is "
                "not implemented by the reference DSpark path; use full-vocabulary "
                "DSpark for the fair baseline"
            )

        if self.draft_load_config is None:
            self.draft_load_config = LoadConfig(load_format="auto")
        load_format = str(self.draft_load_config.load_format).lower()
        if "dummy" in load_format:
            raise ValueError("DFlash/DSpark draft_load_config cannot use dummy weights")

        checkpoint_anchor = getattr(
            self.draft_model_config.hf_text_config, "sample_from_anchor", None
        )
        if self.use_dflash():
            if self.sample_from_anchor is True or checkpoint_anchor is True:
                raise ValueError("DFlash does not support sample_from_anchor=True")
            self.sample_from_anchor = False
        else:
            if self.sample_from_anchor is None:
                self.sample_from_anchor = (
                    True if checkpoint_anchor is None else bool(checkpoint_anchor)
                )
            self.draft_model_config.hf_text_config.sample_from_anchor = (
                self.sample_from_anchor
            )

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        if self.uses_parallel_block_drafter():
            assert self.target_model_config is not None
            assert self.draft_model_config is not None
            if self.target_model_config.is_multimodal_model:
                raise ValueError("DFlash/DSpark Phase 0 does not support multimodal models")
            if self.disable_padded_drafter_batch:
                raise ValueError("DFlash/DSpark requires the padded drafter path")
            if self.target_model_config.tokenizer != self.draft_model_config.tokenizer:
                raise ValueError("DFlash/DSpark target and draft tokenizers must match")
            if self.use_dspark() and (
                self.draft_model_config.model == self.target_model_config.model
            ):
                raise ValueError(
                    "Qwen3 DSpark requires an independent draft checkpoint path"
                )
            parallel = self.target_parallel_config
            unsupported_parallel = {
                "pipeline_parallel_size": parallel.pipeline_parallel_size,
                "prefill_context_parallel_size": parallel.prefill_context_parallel_size,
                "decode_context_parallel_size": parallel.decode_context_parallel_size,
            }
            enabled = {name: value for name, value in unsupported_parallel.items() if value > 1}
            if enabled:
                raise ValueError(
                    f"DFlash/DSpark Phase 0 does not support PP/PCP/DCP: {enabled}"
                )
            if self.parallel_query_count > self.draft_model_config.max_model_len:
                raise ValueError(
                    "parallel query block is larger than the draft max model length"
                )
        else:
            if self.draft_load_config is not None:
                if self.method not in ("eagle", "eagle3"):
                    raise ValueError(
                        "draft_load_config is only supported for EAGLE/EAGLE3/"
                        "DFlash/DSpark"
                    )
                if "dummy" in str(self.draft_load_config.load_format).lower():
                    raise ValueError(
                        "EAGLE/EAGLE3 draft_load_config cannot use dummy weights"
                    )
            if self.rejection_sample_method is not None:
                raise ValueError(
                    "rejection_sample_method is only supported for DFlash/DSpark "
                    "in the old-ABI implementation"
                )
            if self.draft_sample_method != "greedy":
                raise ValueError(
                    "draft_sample_method is only supported for DFlash/DSpark "
                    "in the old-ABI implementation"
                )
            if self.draft_probability_max_bytes is not None:
                raise ValueError(
                    "draft_probability_max_bytes is only supported for "
                    "DFlash/DSpark"
                )
            if self.sample_from_anchor is not None:
                raise ValueError("sample_from_anchor is only supported for DSpark")
            if self.parallel_draft_profile_enabled:
                raise ValueError(
                    "parallel_draft_profile_enabled is only supported for DFlash/DSpark"
                )
            if self.parallel_draft_incremental_context_kv:
                raise ValueError(
                    "parallel_draft_incremental_context_kv is only supported for DFlash/DSpark"
                )
            if self.parallel_draft_dynamic_k:
                raise ValueError(
                    "parallel_draft_dynamic_k is only supported for DFlash/DSpark"
                )
            if self.dspark_draft_topk is not None:
                raise ValueError("dspark_draft_topk is only supported for DSpark")

        if self.disable_by_batch_size is not None and self.disable_by_batch_size < 2:
            raise ValueError(
                "Expect the batch size threshold of disabling "
                "speculative decoding is > 1, but got "
                f"{self.disable_by_batch_size=}"
            )

        eagle3_target_supported = ["llama", "qwen", "minicpm", "gpt_oss"]
        if (
            self.method == "eagle3"
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in eagle3_target_supported
            )
        ):
            raise ValueError(
                f"Eagle3 is only supported for {eagle3_target_supported} models. "  # noqa: E501
                f"Got {self.target_model_config.hf_text_config.model_type=}"
            )

        return self

    def use_eagle(self) -> bool:
        # Historical name: downstream uses this as "needs target hidden state".
        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def use_dflash(self) -> bool:
        return self.method == "dflash"

    def use_dspark(self) -> bool:
        return self.method == "dspark"

    def uses_parallel_block_drafter(self) -> bool:
        return self.method in ("dflash", "dspark")

    def uses_neural_drafter(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def needs_aux_hidden_states(self) -> bool:
        return self.method in ("eagle3", "dflash", "dspark")

    @property
    def parallel_query_count(self) -> int:
        """Number of draft-model queries per request (Q)."""
        if not self.uses_parallel_block_drafter():
            return 0
        k = self.num_speculative_tokens or 0
        if self.use_dspark() and self.sample_from_anchor is True:
            return k
        return k + 1

    @property
    def draft_sample_offset(self) -> int:
        """First query row sampled into the K-token draft result."""
        return 0 if self.use_dspark() and self.sample_from_anchor is True else 1

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """Additional KV slots after the scheduler's existing query slot."""
        if not self.uses_parallel_block_drafter():
            return 0
        return max(0, self.parallel_query_count - 1)

    def parallel_window_fits(self, sequence_length: int) -> bool:
        """Whether a complete fixed-K query block fits the draft window."""
        if sequence_length < 0:
            raise ValueError("sequence_length must be non-negative")
        if not self.uses_parallel_block_drafter():
            return True
        assert self.draft_model_config is not None
        return (
            sequence_length + self.parallel_query_count
            <= self.draft_model_config.max_model_len
        )

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method in ("ngram", "suffix", "sam")
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
