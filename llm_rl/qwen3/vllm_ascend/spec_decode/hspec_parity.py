# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in runtime token trace for the HSpec Patch 0 parity gate."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import threading
from pathlib import Path
from typing import Mapping, Sequence


_PARITY_DIR = os.environ.get("HSPEC_S1_PARITY_DIR", "").strip()
HSPEC_S1_PARITY_ENABLED = bool(_PARITY_DIR)


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except (TypeError, ValueError):
        return int(default)


def _token_hash(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(len(tokens).to_bytes(8, byteorder="little", signed=False))
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


class HSpecS1ParityRecorder:
    """Buffer exact draft/target events and write one JSONL file per worker."""

    def __init__(self, output_dir: str | Path, flush_records: int = 256):
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._flush_records = max(int(flush_records), 1)
        self._lock = threading.Lock()
        self._buffer: list[dict[str, object]] = []
        self._sequence = 0
        self._pid = os.getpid()
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
        host = socket.gethostname().replace("/", "_")
        self._path = self._output_dir / (
            f"parity-{host}-rank-{rank}-pid-{self._pid}.jsonl"
        )

    @property
    def path(self) -> Path:
        return self._path

    def record_batch(
        self,
        *,
        request_ids: Sequence[str],
        prompt_token_ids: Sequence[Sequence[int]],
        output_prefix_token_ids: Sequence[Sequence[int]],
        draft_token_ids: Sequence[Sequence[int]],
        target_token_ids: Sequence[Sequence[int]],
        accepted_prefix_lengths: Sequence[int],
        spec_decode_active: bool,
    ) -> None:
        records: list[dict[str, object]] = []
        for index, request_id in enumerate(request_ids):
            prompt = (
                [int(token) for token in prompt_token_ids[index]]
                if index < len(prompt_token_ids)
                else []
            )
            prefix = (
                [int(token) for token in output_prefix_token_ids[index]]
                if index < len(output_prefix_token_ids)
                else []
            )
            draft = (
                [int(token) for token in draft_token_ids[index]]
                if index < len(draft_token_ids)
                else []
            )
            target = (
                [int(token) for token in target_token_ids[index]]
                if index < len(target_token_ids)
                else []
            )
            if not draft and not target:
                continue
            accepted = (
                int(accepted_prefix_lengths[index])
                if index < len(accepted_prefix_lengths)
                else 0
            )
            records.append({
                "schema_version": "hspec.s1.parity-event.v1",
                "request_id": str(request_id),
                "prompt_token_ids_sha256": _token_hash(prompt),
                "prompt_length": len(prompt),
                "output_prefix_token_ids": prefix,
                "draft_token_ids": draft,
                "target_token_ids": target,
                "accepted_prefix_len": accepted,
                "spec_decode_active": bool(spec_decode_active),
            })
        if not records:
            return

        with self._lock:
            for record in records:
                record["worker_sequence"] = self._sequence
                self._sequence += 1
                self._buffer.append(record)
            if len(self._buffer) >= self._flush_records:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        if os.getpid() != self._pid:
            raise RuntimeError("HSpec S1 parity recorder cannot be reused after fork")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in self._buffer
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()


_RECORDER: HSpecS1ParityRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def _get_recorder() -> HSpecS1ParityRecorder:
    global _RECORDER
    if not HSPEC_S1_PARITY_ENABLED:
        raise RuntimeError("HSPEC_S1_PARITY_DIR is not configured")
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = HSpecS1ParityRecorder(
                _PARITY_DIR,
                flush_records=_env_positive_int(
                    "HSPEC_S1_PARITY_FLUSH_RECORDS", 256
                ),
            )
        return _RECORDER


def record_hspec_s1_parity_batch(
    *,
    request_ids: Sequence[str],
    prompt_token_ids: Sequence[Sequence[int]],
    output_prefix_token_ids: Sequence[Sequence[int]],
    scheduled_drafts: Mapping[str, Sequence[int]],
    target_token_ids: Sequence[Sequence[int]],
    accepted_prefix_lengths: Sequence[int],
    spec_decode_active: bool,
) -> None:
    """Record one model-runner output batch when the S1 gate is enabled."""
    if not HSPEC_S1_PARITY_ENABLED:
        return
    drafts = [scheduled_drafts.get(str(request_id), ()) for request_id in request_ids]
    _get_recorder().record_batch(
        request_ids=request_ids,
        prompt_token_ids=prompt_token_ids,
        output_prefix_token_ids=output_prefix_token_ids,
        draft_token_ids=drafts,
        target_token_ids=target_token_ids,
        accepted_prefix_lengths=accepted_prefix_lengths,
        spec_decode_active=spec_decode_active,
    )


def flush_hspec_s1_parity() -> None:
    """Flush buffered parity events without initializing an unused recorder."""
    if _RECORDER is not None:
        _RECORDER.flush()


atexit.register(flush_hspec_s1_parity)
