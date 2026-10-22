"""Lossless JSON containers for checkpoint configuration metadata.

Runtime callbacks are omitted from mappings, as in the original checkpoint
writer. Unknown objects fail with their field path rather than being silently
stringified. This module never changes the live training configuration.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _json_value(value, path):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    if isinstance(value, (torch.dtype, Enum)):
        # Preserve the previous dtype / AttnBackend spelling.
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: configuration key {key!r} is not a string")
            if not callable(item):
                result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path}: unsupported configuration type {type(value).__name__}")


def serialize_checkpoint_config(config):
    """Serialize before opening a destination or saving large tensor payloads."""
    value = _json_value(config, "transformer_config")
    if not isinstance(value, dict):
        raise TypeError("transformer_config must be a dataclass or mapping")
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def write_checkpoint_config(path, serialized):
    """Publish a complete metadata file by rename on the same filesystem."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
