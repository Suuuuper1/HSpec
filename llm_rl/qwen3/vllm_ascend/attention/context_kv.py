# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validated direct paged-cache writes for parallel-block drafters."""

from collections.abc import Sequence

import torch
from vllm.attention.layer import Attention

from vllm_ascend.device.device_op import DeviceOperator

PAD_SLOT_ID = -1


def _async_assert(condition: torch.Tensor, message: str) -> None:
    assert condition.numel() == 1
    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    else:  # pragma: no cover - only for older torch test environments.
        torch._assert(condition, message)


def store_context_kv(
    attn_layer: Attention,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    virtual_engine: int = 0,
    validate_slot_range: bool = True,
) -> None:
    """Write K/V to one drafter cache without running attention.

    Slot validation remains device-side so rejected/PAD handling adds no host
    synchronization to the proposal hot path.
    """
    if not isinstance(attn_layer, Attention):
        raise TypeError(f"Expected vLLM Attention, got {type(attn_layer).__name__}")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError(
            f"Context K/V must both be [T,nkv,d], got {key.shape} and {value.shape}"
        )
    if key.shape[0] != slot_mapping.numel():
        raise ValueError("Context K/V and slot mapping token counts differ")
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("Direct Context K/V inputs must be contiguous")
    if slot_mapping.dtype != torch.int32:
        raise TypeError(f"slot_mapping must be int32, got {slot_mapping.dtype}")
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    if key.device != value.device or key.device != slot_mapping.device:
        raise ValueError("Context K/V and slot mapping must be on the same device")
    if virtual_engine < 0 or virtual_engine >= len(attn_layer.kv_cache):
        raise ValueError(f"Invalid virtual_engine={virtual_engine}")

    kv_cache = attn_layer.kv_cache[virtual_engine]
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 5:
        raise RuntimeError(
            f"DFlash cache for {attn_layer.layer_name!r} is not bound: "
            f"shape={getattr(kv_cache, 'shape', None)}"
        )
    if kv_cache.shape[0] != 2:
        raise ValueError(f"Expected [2,blocks,block,nkv,d] cache, got {kv_cache.shape}")
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    if key.dtype != key_cache.dtype or value.dtype != value_cache.dtype:
        raise TypeError(
            f"Context/cache dtype mismatch: key={key.dtype}, value={value.dtype}, "
            f"cache={key_cache.dtype}"
        )
    if key.shape[1:] != key_cache.shape[-2:]:
        raise ValueError(
            f"Context/cache head layout mismatch: {key.shape[1:]} != "
            f"{key_cache.shape[-2:]}"
        )

    if validate_slot_range:
        physical_slots = key_cache.shape[0] * key_cache.shape[1]
        _async_assert(
            (slot_mapping >= PAD_SLOT_ID).all(),
            "slot mapping below PAD_SLOT_ID",
        )
        _async_assert(
            (slot_mapping < physical_slots).all(),
            "DFlash Context KV slot exceeds physical drafter cache",
        )
    DeviceOperator.reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
    )


def store_all_context_kv(
    attn_layers: Sequence[Attention],
    all_key: torch.Tensor,
    all_value: torch.Tensor,
    slot_mapping: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    """Store layer-major ``[L,T,nkv,d]`` K/V into drafter-owned caches."""
    if all_key.ndim != 4 or all_value.shape != all_key.shape:
        raise ValueError("all_key/all_value must have identical [L,T,nkv,d] shape")
    if len(attn_layers) != all_key.shape[0]:
        raise ValueError("DFlash attention layer count does not match projected K/V")
    per_layer_slots = isinstance(slot_mapping, (list, tuple))
    if per_layer_slots and len(slot_mapping) != len(attn_layers):
        raise ValueError("Per-layer slot mapping count does not match DFlash layers")
    uniform_cache_shape = None
    if not per_layer_slots:
        for attn_layer in attn_layers:
            if not attn_layer.kv_cache or not isinstance(attn_layer.kv_cache[0], torch.Tensor):
                raise RuntimeError(f"DFlash cache for {attn_layer.layer_name!r} is not bound")
            cache_shape = tuple(attn_layer.kv_cache[0].shape)
            if uniform_cache_shape is None:
                uniform_cache_shape = cache_shape
            elif cache_shape != uniform_cache_shape:
                raise ValueError(
                    "Phase 1 DFlash requires uniform physical cache shapes, got "
                    f"{uniform_cache_shape} and {cache_shape}"
                )
    for layer_idx, attn_layer in enumerate(attn_layers):
        slots = slot_mapping[layer_idx] if per_layer_slots else slot_mapping
        store_context_kv(
            attn_layer,
            all_key[layer_idx].contiguous(),
            all_value[layer_idx].contiguous(),
            slots,
            validate_slot_range=per_layer_slots or layer_idx == 0,
        )
