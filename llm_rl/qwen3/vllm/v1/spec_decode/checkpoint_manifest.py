# SPDX-License-Identifier: Apache-2.0
"""Offline checkpoint manifest and compatibility gate for DFlash/DSpark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class ManifestError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"expected a JSON object in {path}")
    return value


def _tokenizer_manifest(path: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for filename in TOKENIZER_FILES:
        candidate = path / filename
        if candidate.is_file():
            files[filename] = _sha256_file(candidate)
    if not files:
        raise ManifestError(f"no tokenizer artifacts found under {path}")

    tokenizer_config = (
        _load_json(path / "tokenizer_config.json")
        if (path / "tokenizer_config.json").is_file()
        else {}
    )
    special_map = (
        _load_json(path / "special_tokens_map.json")
        if (path / "special_tokens_map.json").is_file()
        else {}
    )
    return {
        "path": str(path.resolve()),
        "revision": tokenizer_config.get("_commit_hash"),
        "files": files,
        "aggregate_sha256": _canonical_hash(files),
        "vocab_sha256": files.get("tokenizer.json") or files.get("vocab.json"),
        "special_tokens": special_map,
    }


def _weight_metadata(path: Path) -> tuple[dict[str, Any], int | None]:
    """Return safetensors file identity/shape metadata without materializing weights."""
    files = sorted(path.glob("*.safetensors"))
    metadata: dict[str, Any] = {
        "files": [
            {"name": file.name, "bytes": file.stat().st_size} for file in files
        ],
        "index_sha256": None,
        "tensor_shape_sha256": None,
    }
    index_files = sorted(path.glob("*.safetensors.index.json"))
    if index_files:
        metadata["index_sha256"] = _sha256_file(index_files[0])

    fc_input_width: int | None = None
    shapes: dict[str, list[int]] = {}
    try:
        from safetensors import safe_open

        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shape = list(handle.get_slice(key).get_shape())
                    shapes[key] = shape
                    if key.endswith("fc.weight") and len(shape) == 2:
                        fc_input_width = shape[1]
        metadata["tensor_shape_sha256"] = _canonical_hash(shapes)
        metadata["num_tensors"] = len(shapes)
    except ImportError:
        metadata["tensor_metadata_error"] = "safetensors package unavailable"
    except Exception as error:  # Corrupt/unsupported checkpoints must stay visible.
        metadata["tensor_metadata_error"] = f"{type(error).__name__}: {error}"
    return metadata, fc_input_width


def _extract_aux_layer_ids(config: dict[str, Any], method: str) -> list[int]:
    if method == "dflash":
        nested = config.get("dflash_config") or {}
        return list(nested.get("target_layer_ids") or [])
    for key in ("dspark_target_layer_ids", "target_layer_ids"):
        if config.get(key) is not None:
            return list(config[key])
    nested = config.get("dspark_config") or {}
    return list(nested.get("target_layer_ids") or [])


def _resolve_layer_causality(
    config: dict[str, Any], method: str | None
) -> list[bool] | None:
    layer_types = config.get("layer_types") or []
    if not method or not layer_types:
        return None
    nested = config.get(f"{method}_config") or {}
    override = nested.get("causal")
    if isinstance(override, bool):
        return [override] * len(layer_types)
    # Qwen DFlash/DSpark full-attention blocks are non-causal by default.
    return ["sliding" in str(layer_type).lower() for layer_type in layer_types]


def _resolve_rope_theta(config: dict[str, Any]) -> Any:
    """Normalize Transformers 4.x and 5.x Qwen RoPE config layouts."""
    if config.get("rope_theta") is not None:
        return config["rope_theta"]
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_theta")
    return None


def build_checkpoint_manifest(
    checkpoint: str | Path,
    *,
    role: str,
    tokenizer: str | Path,
    revision: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint)
    if not path.is_dir():
        raise ManifestError(f"checkpoint must be an offline directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ManifestError(f"missing checkpoint config: {config_path}")
    config = _load_json(config_path)
    tokenizer_info = _tokenizer_manifest(Path(tokenizer))
    weight_info, fc_input_width = _weight_metadata(path)

    aux_layer_ids = _extract_aux_layer_ids(config, method or "")
    dflash_config = config.get("dflash_config") or {}
    mask_embedding = path / "mask_embedding.pt"
    layer_types = config.get("layer_types") or []
    hidden_size = config.get("hidden_size")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "method": method,
        "id": path.name,
        "path": str(path.resolve()),
        "revision": revision or config.get("_commit_hash"),
        "config_sha256": _sha256_file(config_path),
        "config_canonical_sha256": _canonical_hash(config),
        "architectures": config.get("architectures") or [],
        "model_type": config.get("model_type"),
        "vocab_size": config.get("vocab_size"),
        "hidden_size": hidden_size,
        "dtype": config.get("dtype") or config.get("torch_dtype"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_target_layers": config.get("num_target_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "rope_theta": _resolve_rope_theta(config),
        "rope_scaling": config.get("rope_scaling"),
        "is_neox_style": config.get("is_neox_style"),
        "layer_types": layer_types,
        "layer_causality": _resolve_layer_causality(config, method),
        "sliding_window": config.get("sliding_window"),
        "special_token_ids": {
            key: config.get(key)
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "aux_layer_ids": aux_layer_ids,
        "aux_concat_width": (
            len(aux_layer_ids) * hidden_size
            if aux_layer_ids and isinstance(hidden_size, int)
            else None
        ),
        "draft_fc_input_width": fc_input_width,
        "mask_token_id": (
            dflash_config["mask_token_id"]
            if "mask_token_id" in dflash_config
            else config.get("mask_token_id")
        ),
        "independent_mask_embedding": {
            "present": mask_embedding.is_file(),
            "sha256": _sha256_file(mask_embedding) if mask_embedding.is_file() else None,
        },
        "sample_from_anchor": config.get(
            "sample_from_anchor", True if method == "dspark" else False
        ),
        "block_size": config.get("block_size"),
        "tokenizer": tokenizer_info,
        "weights": weight_info,
    }


def _compare_equal(
    errors: list[str],
    target: dict[str, Any],
    draft: dict[str, Any],
    fields: Iterable[str],
) -> None:
    for field in fields:
        if target.get(field) != draft.get(field):
            errors.append(
                f"{field} mismatch: target={target.get(field)!r}, "
                f"draft={draft.get(field)!r}"
            )


def compare_checkpoint_pair(
    target: dict[str, Any],
    draft: dict[str, Any],
    *,
    method: str,
    pair_id: str | None,
    num_speculative_tokens: int,
    target_tp: int,
    draft_tp: int,
    draft_sample_method: str,
    rejection_sample_method: str,
) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if method not in ("dflash", "dspark"):
        errors.append(f"unsupported method: {method!r}")
    if not 1 <= num_speculative_tokens <= 15:
        errors.append("K must be in [1, 15]")
    if draft_tp not in (1, target_tp):
        errors.append("draft TP must be 1 or equal to target TP")
    if draft_sample_method != "greedy":
        errors.append("the first certified scope requires greedy draft sampling")
    if rejection_sample_method != "standard":
        errors.append("DFlash/DSpark requires standard rejection")

    _compare_equal(
        errors,
        target,
        draft,
        (
            "vocab_size",
            "hidden_size",
            "dtype",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "rope_scaling",
            "special_token_ids",
        ),
    )
    if target["tokenizer"]["aggregate_sha256"] != draft["tokenizer"]["aggregate_sha256"]:
        errors.append("target/draft tokenizer hash mismatch")
    if target.get("model_type") not in ("qwen3", "qwen3_moe"):
        errors.append(f"unsupported target model_type={target.get('model_type')!r}")
    if draft.get("model_type") != "qwen3":
        errors.append(f"unsupported draft model_type={draft.get('model_type')!r}")

    expected_architectures = (
        {"DFlashDraftModel"}
        if method == "dflash"
        else {"Qwen3DSparkModel", "DSparkDraftModel"}
    )
    if not expected_architectures.intersection(draft.get("architectures") or []):
        errors.append(
            f"{method} architecture mismatch: {draft.get('architectures')!r}"
        )
    if any(value != "full_attention" for value in draft.get("layer_types") or []):
        errors.append("first scope requires uniform full-attention draft layers")
    if draft.get("sliding_window") is not None:
        errors.append("first scope does not support sliding-window draft layers")
    trained_block_size = draft.get("block_size")
    if isinstance(trained_block_size, int) and (
        trained_block_size < 1 or num_speculative_tokens > trained_block_size
    ):
        errors.append(
            "K exceeds the draft checkpoint block_size: "
            f"K={num_speculative_tokens}, block_size={trained_block_size}"
        )

    aux_ids = draft.get("aux_layer_ids") or []
    target_layers = target.get("num_hidden_layers")
    if not aux_ids:
        errors.append("draft checkpoint does not declare target aux layer ids")
    elif not isinstance(target_layers, int) or any(
        not isinstance(layer, int) or layer < 0 or layer >= target_layers
        for layer in aux_ids
    ):
        errors.append("draft aux layer ids are outside the target layer range")
    if draft.get("num_target_layers") not in (None, target_layers):
        errors.append("draft num_target_layers does not match target")
    if draft.get("draft_fc_input_width") is None:
        blockers.append("draft FC input width could not be read from weight metadata")
    elif draft.get("draft_fc_input_width") != draft.get("aux_concat_width"):
        errors.append(
            "draft FC input width does not equal target aux concatenation width"
        )

    if not pair_id:
        blockers.append(
            "missing explicit target/draft distillation pair id or revision attestation"
        )
    if draft["weights"].get("tensor_metadata_error"):
        blockers.append(draft["weights"]["tensor_metadata_error"])
    if draft.get("is_neox_style") is None:
        warnings.append(
            "RoPE layout is not checkpoint-authored; Phase 1 must inject it from target"
        )
    if method == "dflash" and draft.get("mask_token_id") is None:
        errors.append("DFlash checkpoint has no mask_token_id")

    status = "FAIL" if errors else "BLOCKED" if blockers else "PASS"
    return {
        "method": method,
        "status": status,
        "pair_id": pair_id,
        "num_speculative_tokens": num_speculative_tokens,
        "target_tp": target_tp,
        "draft_tp": draft_tp,
        "draft_sample_method": draft_sample_method,
        "rejection_sample_method": rejection_sample_method,
        "errors": errors,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_migration_manifest(
    *,
    target_path: str | Path,
    tokenizer_path: str | Path,
    draft_paths: dict[str, str | Path],
    draft_tokenizer_paths: dict[str, str | Path] | None = None,
    pair_ids: dict[str, str | None],
    revisions: dict[str, str | None] | None = None,
    num_speculative_tokens: int,
    target_tp: int,
    draft_tp: int,
    draft_sample_method: str = "greedy",
    rejection_sample_method: str = "standard",
) -> dict[str, Any]:
    revisions = revisions or {}
    draft_tokenizer_paths = draft_tokenizer_paths or {}
    target = build_checkpoint_manifest(
        target_path,
        role="target",
        tokenizer=tokenizer_path,
        revision=revisions.get("target"),
    )
    drafts: dict[str, Any] = {}
    compatibility: dict[str, Any] = {}
    for method, path in draft_paths.items():
        draft = build_checkpoint_manifest(
            path,
            role="draft",
            tokenizer=draft_tokenizer_paths.get(method, tokenizer_path),
            method=method,
            revision=revisions.get(method),
        )
        drafts[method] = draft
        compatibility[method] = compare_checkpoint_pair(
            target,
            draft,
            method=method,
            pair_id=pair_ids.get(method),
            num_speculative_tokens=num_speculative_tokens,
            target_tp=target_tp,
            draft_tp=draft_tp,
            draft_sample_method=draft_sample_method,
            rejection_sample_method=rejection_sample_method,
        )

    missing = sorted({"dflash", "dspark"} - set(drafts))
    statuses = [item["status"] for item in compatibility.values()]
    overall = (
        "FAIL"
        if "FAIL" in statuses
        else "BLOCKED"
        if missing or "BLOCKED" in statuses
        else "PASS"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "runner": "MRV1",
            "rollout": "sync",
            "fixed_k": True,
            "draft_eager": True,
            "full_vocab": True,
            "draft_sample_method": draft_sample_method,
            "rejection_sample_method": rejection_sample_method,
            "num_speculative_tokens": num_speculative_tokens,
            "target_tp": target_tp,
            "draft_tp": draft_tp,
            "draft_load_format": "auto",
        },
        "target": target,
        "drafts": drafts,
        "compatibility": compatibility,
        "missing_methods": missing,
        "overall_status": overall,
    }
