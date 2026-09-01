# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single Verl-to-vLLM speculative configuration boundary for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.config.load import LoadConfig

from verl.workers.config.rollout import resolve_rollout_speculative_method


@dataclass(frozen=True)
class ResolvedRolloutSpeculation:
    method: str | None
    engine_kwargs: dict[str, Any]
    speculative_config: dict[str, Any] | None
    manifest: dict[str, Any]

    @property
    def uses_hspec(self) -> bool:
        return self.method == "hspec"

    @property
    def uses_parallel_draft(self) -> bool:
        return self.method in {"dflash", "dspark"}


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(key): item for key, item in value.items()}


def _remove_explicit_llm_duplicates(config, engine_kwargs: dict[str, Any]) -> None:
    expected = {
        "async_scheduling": False,
        "enable_prefix_caching": bool(config.get("enable_prefix_caching", True)),
        "enforce_eager": bool(config.get("enforce_eager", True)),
    }
    for key, value in expected.items():
        if key not in engine_kwargs:
            continue
        configured = engine_kwargs.pop(key)
        if bool(configured) != value:
            raise ValueError(
                f"engine_kwargs.vllm.{key}={configured!r} conflicts with "
                f"the Verl-owned value {value!r}"
            )


def resolve_rollout_speculation(config) -> ResolvedRolloutSpeculation:
    """Resolve one authoritative method into the old vLLM engine ABI."""
    method = resolve_rollout_speculative_method(config)
    engine_kwargs = {
        key: value
        for key, value in _dict(
            _dict(config.get("engine_kwargs", {})).get("vllm", {})
        ).items()
        if value is not None
    }
    _remove_explicit_llm_duplicates(config, engine_kwargs)

    raw_spec = engine_kwargs.get("speculative_config")
    if method is not None and raw_spec is not None:
        raise ValueError(
            "speculative_method/use_hspec_decode conflicts with "
            "engine_kwargs.vllm.speculative_config"
        )
    if method is None and raw_spec is not None:
        raw_spec_dict = _dict(raw_spec)
        raw_method = str(raw_spec_dict.get("method", "")).lower()
        if raw_method in {"hspec", "dflash", "dspark"}:
            raise ValueError(
                f"method={raw_method!r} must use rollout.speculative_method; "
                "the unified migrated-method boundary is mandatory and the raw "
                "engine escape hatch is reserved for existing methods"
            )
        raw_draft_load = raw_spec_dict.get("draft_load_config")
        raw_draft_load_format = (
            _dict(raw_draft_load).get("load_format")
            if isinstance(raw_draft_load, dict) or hasattr(raw_draft_load, "items")
            else getattr(raw_draft_load, "load_format", None)
        )
        return ResolvedRolloutSpeculation(
            method=raw_method or "external",
            engine_kwargs=engine_kwargs,
            speculative_config=raw_spec,
            manifest={
                "source": "engine_kwargs.vllm.speculative_config",
                "method": raw_method or "external",
                "model": raw_spec_dict.get("model"),
                "num_speculative_tokens": int(
                    raw_spec_dict.get("num_speculative_tokens", 0)
                ),
                "prompt_lookup_min": raw_spec_dict.get("prompt_lookup_min"),
                "prompt_lookup_max": raw_spec_dict.get("prompt_lookup_max"),
                "target_enforce_eager": bool(config.get("enforce_eager", True)),
                "draft_enforce_eager": raw_spec_dict.get("enforce_eager"),
                "draft_tensor_parallel_size": raw_spec_dict.get(
                    "draft_tensor_parallel_size"
                ),
                "draft_load_format": (
                    str(raw_draft_load_format)
                    if raw_draft_load_format is not None
                    else None
                ),
                "target_tensor_parallel_size": int(
                    config.get("tensor_model_parallel_size", 1)
                ),
                "prefix_caching": bool(
                    config.get("enable_prefix_caching", True)
                ),
                "chunked_prefill": bool(
                    config.get("enable_chunked_prefill", True)
                ),
                "mode": str(config.get("mode", "sync")),
                "seed": int(config.get("seed", 0)),
                "temperature": float(config.get("temperature", 1.0)),
                "top_p": float(config.get("top_p", 1.0)),
                "top_k": int(config.get("top_k", -1)),
                "rollout_n": int(config.get("n", 1)),
                "parallel_draft": False,
            },
        )

    speculative_config: dict[str, Any] | None = None
    if method == "hspec":
        legacy = bool(config.get("use_hspec_decode", False))
        speculative_config = {
            "method": "hspec",
            "num_speculative_tokens": int(
                config.get("hspec_num_speculative_tokens", 5)
                if legacy
                else config.get("num_speculative_tokens", 7)
            ),
            "hspec_similarity_threshold": float(
                config.get("hspec_similarity_threshold", 0.9)
            ),
            "hspec_min_match_len": int(config.get("hspec_min_match_len", 1)),
            "hspec_n_components": int(config.get("hspec_n_components", 64)),
            "hspec_max_entries_per_prompt": int(
                config.get("hspec_max_entries_per_prompt", 10000)
            ),
        }
    elif method in {"dflash", "dspark"}:
        draft_sample_method = str(config.get("draft_sample_method", "greedy"))
        speculative_config = {
            "model": str(config.get("speculative_model")),
            "method": method,
            "num_speculative_tokens": int(config.get("num_speculative_tokens", 7)),
            "draft_tensor_parallel_size": int(
                config.get("draft_tensor_parallel_size", 1)
            ),
            "draft_sample_method": draft_sample_method,
            "draft_load_config": LoadConfig(
                load_format=str(config.get("draft_load_format", "auto"))
            ),
            "rejection_sample_method": str(
                config.get("rejection_sample_method", "standard")
            ),
            "enforce_eager": bool(config.get("speculative_enforce_eager", True)),
            "parallel_draft_profile_enabled": bool(
                config.get("parallel_draft_profile_enabled", False)
            ),
            "parallel_draft_profile_sample_every": int(
                config.get("parallel_draft_profile_sample_every", 64)
            ),
            "parallel_draft_profile_flush_every": int(
                config.get("parallel_draft_profile_flush_every", 4)
            ),
            "parallel_draft_incremental_context_kv": bool(
                config.get("parallel_draft_incremental_context_kv", False)
            ),
            "parallel_draft_dynamic_k": bool(
                config.get("parallel_draft_dynamic_k", False)
            ),
        }
        if config.get("dspark_draft_topk", None) is not None:
            speculative_config["dspark_draft_topk"] = int(
                config.get("dspark_draft_topk")
            )
        if draft_sample_method == "probabilistic":
            speculative_config["draft_probability_max_bytes"] = int(
                config.get("draft_probability_max_memory_mb", 2048)
            ) * 1024 * 1024

    if speculative_config is not None:
        engine_kwargs["speculative_config"] = speculative_config
    manifest = {
        "source": (
            "legacy_use_hspec_decode"
            if bool(config.get("use_hspec_decode", False))
            else "rollout.speculative_method"
        ),
        "method": method,
        "model": config.get("speculative_model", None),
        "num_speculative_tokens": (
            speculative_config.get("num_speculative_tokens")
            if speculative_config is not None
            else 0
        ),
        "target_load_format": str(config.get("load_format", "dummy")),
        "target_enforce_eager": bool(config.get("enforce_eager", True)),
        "draft_load_format": (
            str(config.get("draft_load_format", "auto"))
            if method in {"dflash", "dspark"}
            else None
        ),
        "draft_enforce_eager": (
            bool(config.get("speculative_enforce_eager", True))
            if method in {"dflash", "dspark"}
            else None
        ),
        "draft_tensor_parallel_size": (
            int(config.get("draft_tensor_parallel_size", 1))
            if method in {"dflash", "dspark"}
            else None
        ),
        "draft_sample_method": (
            str(config.get("draft_sample_method", "greedy"))
            if method in {"dflash", "dspark"}
            else None
        ),
        "draft_probability_max_memory_mb": (
            int(config.get("draft_probability_max_memory_mb", 2048))
            if method in {"dflash", "dspark"}
            and str(config.get("draft_sample_method", "greedy")) == "probabilistic"
            else None
        ),
        "rejection_sample_method": (
            str(config.get("rejection_sample_method", "standard"))
            if method in {"dflash", "dspark"}
            else None
        ),
        "parallel_draft_profile_enabled": (
            bool(config.get("parallel_draft_profile_enabled", False))
            if method in {"dflash", "dspark"}
            else False
        ),
        "parallel_draft_profile_sample_every": (
            int(config.get("parallel_draft_profile_sample_every", 64))
            if method in {"dflash", "dspark"}
            else None
        ),
        "parallel_draft_profile_flush_every": (
            int(config.get("parallel_draft_profile_flush_every", 4))
            if method in {"dflash", "dspark"}
            else None
        ),
        "target_eager_experiment": (
            bool(config.get("parallel_draft_allow_target_eager_experiment", False))
            if method in {"dflash", "dspark"}
            else False
        ),
        "draft_full_graph": False,
        "incremental_context_kv": False,
        "dynamic_k": False,
        "dspark_draft_topk": None,
        "async_scheduling": False,
        "prefix_caching": bool(config.get("enable_prefix_caching", True)),
        "chunked_prefill": bool(config.get("enable_chunked_prefill", True)),
        "mode": str(config.get("mode", "sync")),
        "seed": int(config.get("seed", 0)),
        "temperature": float(config.get("temperature", 1.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "top_k": int(config.get("top_k", -1)),
        "rollout_n": int(config.get("n", 1)),
        "target_tensor_parallel_size": int(
            config.get("tensor_model_parallel_size", 1)
        ),
    }
    return ResolvedRolloutSpeculation(
        method=method,
        engine_kwargs=engine_kwargs,
        speculative_config=speculative_config,
        manifest=manifest,
    )
