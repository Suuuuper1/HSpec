# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validated direct paged-cache writes for parallel-block drafters."""

from collections.abc import Sequence

import torch
from vllm.attention.layer import Attention

from vllm_ascend.device.device_op import DeviceOperator

PAD_SLOT_ID = -1

ContextKVCache = tuple[torch.Tensor, torch.Tensor]


def _async_assert(condition: torch.Tensor, message: str) -> None:
    assert condition.numel() == 1
    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    else:  # pragma: no cover - only for older torch test environments.
        torch._assert(condition, message)


def _resolve_context_kv_cache(
    attn_layer: Attention,
    virtual_engine: int,
) -> ContextKVCache:
    """Return the physical K/V tensors from the old runner cache ABI.

    Ascend's V1 runner binds ``Attention.kv_cache`` as ``[(K, V)]``.  A
    packed ``[2, blocks, block, n_kv, head]`` tensor is also accepted for
    compatibility with vLLM backends that retain the upstream representation.
    """
    if virtual_engine < 0 or virtual_engine >= len(attn_layer.kv_cache):
        raise ValueError(f"Invalid virtual_engine={virtual_engine}")

    storage = attn_layer.kv_cache[virtual_engine]
    if isinstance(storage, torch.Tensor):
        if storage.numel() == 0:
            raise RuntimeError(
                f"DFlash cache for {attn_layer.layer_name!r} is not bound; "
                "the Attention placeholder is still installed"
            )
        if storage.ndim != 5 or storage.shape[0] != 2:
            raise ValueError(
                "Packed DFlash cache must be "
                f"[2,blocks,block,nkv,d], got {tuple(storage.shape)}"
            )
        key_cache, value_cache = storage[0], storage[1]
    elif isinstance(storage, (tuple, list)):
        if len(storage) != 2 or not all(
            isinstance(cache, torch.Tensor) for cache in storage
        ):
            raise ValueError(
                "Ascend DFlash cache must contain exactly two tensors (K,V), "
                f"got {type(storage).__name__} with length={len(storage)}"
            )
        key_cache, value_cache = storage
    else:
        raise TypeError(
            f"Unsupported DFlash cache storage for {attn_layer.layer_name!r}: "
            f"{type(storage).__name__}"
        )

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "DFlash K/V caches must both be [blocks,block,nkv,d], got "
            f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"DFlash K/V cache shapes differ: {tuple(key_cache.shape)} != "
            f"{tuple(value_cache.shape)}"
        )
    if key_cache.dtype != value_cache.dtype or key_cache.device != value_cache.device:
        raise ValueError(
            "DFlash K/V cache dtype/device differ: "
            f"{key_cache.dtype}/{key_cache.device} != "
            f"{value_cache.dtype}/{value_cache.device}"
        )
    return key_cache, value_cache


def _store_context_kv_pair(
    attn_layer: Attention,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_pair: ContextKVCache,
    *,
    validate_slot_range: bool,
) -> None:
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

    key_cache, value_cache = cache_pair
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
    _store_context_kv_pair(
        attn_layer,
        key,
        value,
        slot_mapping,
        _resolve_context_kv_cache(attn_layer, virtual_engine),
        validate_slot_range=validate_slot_range,
    )


def store_all_context_kv(
    attn_layers: Sequence[Attention],
    all_key: torch.Tensor,
    all_value: torch.Tensor,
    slot_mapping: torch.Tensor | Sequence[torch.Tensor],
    *,
    validate_slot_range: bool = True,
) -> None:
    """Store layer-major ``[L,T,nkv,d]`` K/V into drafter-owned caches.

    ``validate_slot_range=False`` is reserved for engine-produced slot maps
    that have already passed the block-table layout kernels. Public callers
    retain device-side validation by default.
    """
    if all_key.ndim != 4 or all_value.shape != all_key.shape:
        raise ValueError("all_key/all_value must have identical [L,T,nkv,d] shape")
    if len(attn_layers) != all_key.shape[0]:
        raise ValueError("DFlash attention layer count does not match projected K/V")
    per_layer_slots = isinstance(slot_mapping, (list, tuple))
    if per_layer_slots and len(slot_mapping) != len(attn_layers):
        raise ValueError("Per-layer slot mapping count does not match DFlash layers")
    cache_pairs = [
        _resolve_context_kv_cache(attn_layer, virtual_engine=0)
        for attn_layer in attn_layers
    ]
    if not per_layer_slots:
        layouts = [
            (
                tuple(key_cache.shape),
                tuple(value_cache.shape),
                key_cache.dtype,
                key_cache.device,
            )
            for key_cache, value_cache in cache_pairs
        ]
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError(
                "Phase 1 DFlash requires uniform physical cache layouts, got "
                f"{layouts}"
            )
    for layer_idx, (attn_layer, cache_pair) in enumerate(
        zip(attn_layers, cache_pairs)
    ):
        slots = slot_mapping[layer_idx] if per_layer_slots else slot_mapping
        _store_context_kv_pair(
            attn_layer,
            all_key[layer_idx].contiguous(),
            all_value[layer_idx].contiguous(),
            slots,
            cache_pair,
            validate_slot_range=validate_slot_range
            and (per_layer_slots or layer_idx == 0),
        )
