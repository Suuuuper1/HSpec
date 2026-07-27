# Copyright 2026 Xuyi
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
"""Opt-in online reference trace for S4 extent-aware replay parity.

The module is deliberately dependency-free and inert unless
``HSPEC_S4_TRACE_DIR`` is set.  It records the selector state that cannot be
reconstructed from sealed trajectories: worker-local adaptive windows,
entry feedback, exact online projection output, and request interleaving.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import threading
from pathlib import Path
from typing import Mapping, Sequence


_TRACE_DIR = os.environ.get("HSPEC_S4_TRACE_DIR", "").strip()
HSPEC_S4_TRACE_ENABLED = bool(_TRACE_DIR)
_PRODUCER_RANK = os.environ.get(
    "RANK", os.environ.get("LOCAL_RANK", "unknown")
)


def _positive_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except (TypeError, ValueError):
        return int(default)


HSPEC_S4_TRACE_SAMPLE_EVERY = _positive_int(
    "HSPEC_S4_TRACE_SAMPLE_EVERY", 1
)
HSPEC_S4_TRACE_CAPTURE_PROJECTED = (
    os.environ.get("HSPEC_S4_TRACE_CAPTURE_PROJECTED", "1") != "0"
)


def hspec_s4_token_hash(tokens: Sequence[int]) -> str:
    """Hash a bounded token sequence with an unambiguous length prefix."""
    digest = hashlib.sha256()
    digest.update(len(tokens).to_bytes(8, byteorder="little", signed=False))
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _producer_rank() -> str:
    return _PRODUCER_RANK


def _identity_payload(
    request_id: str,
    prompt_id: str,
    decoded_len: int,
    active_table_version: int,
) -> bytes:
    fields = (
        socket.gethostname(),
        _producer_rank(),
        str(request_id),
        str(prompt_id),
        str(int(decoded_len)),
        str(int(active_table_version)),
    )
    return "\x1f".join(fields).encode("utf-8", errors="surrogatepass")


def hspec_s4_trace_query_id(
    request_id: str,
    prompt_id: str,
    decoded_len: int,
    active_table_version: int,
) -> str | None:
    """Return the deterministic sampled query id, or ``None`` when skipped."""
    if not HSPEC_S4_TRACE_ENABLED:
        return None
    payload = _identity_payload(
        request_id, prompt_id, decoded_len, active_table_version
    )
    digest = hashlib.sha256(payload).digest()
    if (
        int.from_bytes(digest[:8], byteorder="little", signed=False)
        % HSPEC_S4_TRACE_SAMPLE_EVERY
    ):
        return None
    return hashlib.sha256(b"hspec-s4-query-v1\x00" + payload).hexdigest()


class HSpecS4TraceRecorder:
    """Buffered per-worker JSONL writer for diagnostic parity events."""

    def __init__(self, output_dir: str | Path, flush_records: int = 1024):
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._flush_records = max(int(flush_records), 1)
        self._lock = threading.Lock()
        self._buffer: list[dict[str, object]] = []
        self._open_queries: dict[str, dict[str, object]] = {}
        self._sequence = 0
        self._pid = os.getpid()
        self._closed = False
        host = socket.gethostname().replace("/", "_")
        rank = _producer_rank()
        self._producer_id = f"{host}:rank-{rank}:pid-{self._pid}"
        self._path = self._output_dir / (
            f"s4-trace-{host}-rank-{rank}-pid-{self._pid}.jsonl"
        )

    @property
    def path(self) -> Path:
        return self._path

    def record_many(self, events: Sequence[Mapping[str, object]]) -> None:
        if not events:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("HSpec S4 trace recorder is closed")
            for raw_event in events:
                event = dict(raw_event)
                query_id = str(event.get("query_id", ""))
                event_type = str(event.get("event", ""))
                if event_type == "selection" and int(
                    event.get("drafted_len", 0) or 0
                ) > 0:
                    self._open_queries[query_id] = {
                        "request_id": str(event.get("request_id", "")),
                        "prompt_id": str(event.get("prompt_id", "")),
                        "active_table_version": int(
                            event.get("active_table_version", -1) or -1
                        ),
                    }
                elif event_type in {"verification", "cancellation"}:
                    self._open_queries.pop(query_id, None)
                self._append_locked(event)
            if len(self._buffer) >= self._flush_records:
                self._flush_locked()

    def _append_locked(self, event: dict[str, object]) -> None:
        event.setdefault("schema_version", "hspec.s4.online-trace.v1")
        event["producer_id"] = self._producer_id
        event["producer_sequence"] = self._sequence
        self._sequence += 1
        self._buffer.append(event)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def seal_open_queries(self, reason: str) -> int:
        """Cancel pending drafts at a lifecycle boundary and keep writing."""
        with self._lock:
            if self._closed:
                return 0
            canceled = self._cancel_open_queries_locked(reason)
            self._flush_locked()
            return canceled

    def close(self) -> None:
        """Close the producer and explicitly cancel unverified tail drafts."""
        with self._lock:
            if self._closed:
                return
            self._cancel_open_queries_locked(
                "producer_exit_before_verification"
            )
            self._closed = True
            self._flush_locked()

    def _cancel_open_queries_locked(self, reason: str) -> int:
        canceled = len(self._open_queries)
        for query_id, metadata in self._open_queries.items():
            self._append_locked({
                "event": "cancellation",
                "query_id": str(query_id),
                "request_id": str(metadata.get("request_id", "")),
                "prompt_id": str(metadata.get("prompt_id", "")),
                "active_table_version": int(
                    metadata.get("active_table_version", -1)
                ),
                "reason": str(reason),
            })
        self._open_queries.clear()
        return canceled

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        if os.getpid() != self._pid:
            raise RuntimeError("HSpec S4 trace recorder cannot be reused after fork")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in self._buffer
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()


_RECORDER: HSpecS4TraceRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def _get_recorder() -> HSpecS4TraceRecorder:
    global _RECORDER
    if not HSPEC_S4_TRACE_ENABLED:
        raise RuntimeError("HSPEC_S4_TRACE_DIR is not configured")
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = HSpecS4TraceRecorder(
                _TRACE_DIR,
                flush_records=_positive_int(
                    "HSPEC_S4_TRACE_FLUSH_RECORDS", 1024
                ),
            )
        return _RECORDER


def record_hspec_s4_trace_events(
    events: Sequence[Mapping[str, object]],
) -> None:
    if HSPEC_S4_TRACE_ENABLED and events:
        _get_recorder().record_many(events)


def flush_hspec_s4_trace() -> None:
    if _RECORDER is not None:
        _RECORDER.flush()


def seal_hspec_s4_trace_open_queries(reason: str) -> int:
    if _RECORDER is None:
        return 0
    return _RECORDER.seal_open_queries(reason)


def close_hspec_s4_trace() -> None:
    if _RECORDER is not None:
        _RECORDER.close()


atexit.register(close_hspec_s4_trace)
