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
HSPEC_S4_TRACE_ROUND_REASON = (
    "rollout_round_completed_with_unresolved_trace_outcome"
)
HSPEC_S4_TRACE_ORPHAN_REASON = (
    "trace_orphan_after_proposer_round_finalize"
)
HSPEC_S4_TRACE_SHUTDOWN_REASON = (
    "worker_shutdown_with_unresolved_trace_outcome"
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
        self._sequence = 0
        self._match_group_sequence = 0
        self._pending_selections: dict[str, dict[str, object]] = {}
        self._finalized = False
        self._pid = os.getpid()
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
            if self._finalized:
                raise RuntimeError("HSpec S4 trace recorder is already finalized")
            selection_batch = all(
                str(event.get("event", "")) == "selection" for event in events
            )
            match_group_id = None
            if selection_batch:
                match_group_id = (
                    f"{self._producer_id}:match-{self._match_group_sequence}"
                )
                self._match_group_sequence += 1
            for recorded_row, raw_event in enumerate(events):
                event = dict(raw_event)
                event.setdefault("schema_version", "hspec.s4.online-trace.v2")
                query_id = str(event.get("query_id", ""))
                event_type = str(event.get("event", ""))
                if query_id and event_type in {"verification", "cancellation"}:
                    pending = self._pending_selections.get(query_id)
                    if pending is not None:
                        for key, value in pending.items():
                            event.setdefault(key, value)
                if match_group_id is not None:
                    event.setdefault("match_group_id", match_group_id)
                    event.setdefault("match_group_recorded_row", recorded_row)
                    event.setdefault("match_group_recorded_rows", len(events))
                event["producer_id"] = self._producer_id
                event["producer_sequence"] = self._sequence
                self._sequence += 1
                self._buffer.append(event)
                if (
                    query_id
                    and event_type == "selection"
                    and int(event.get("drafted_len", 0)) > 0
                ):
                    if query_id in self._pending_selections:
                        raise ValueError(
                            f"duplicate pending S4 selection query_id={query_id}"
                        )
                    self._pending_selections[query_id] = {
                        "query_id": query_id,
                        "request_id": str(event.get("request_id", "")),
                        "prompt_id": str(event.get("prompt_id", "")),
                        "decoded_len": int(event.get("decoded_len", -1)),
                        "active_table_version": int(
                            event.get("active_table_version", -1)
                        ),
                    }
                elif query_id and event_type in {"verification", "cancellation"}:
                    self._pending_selections.pop(query_id, None)
            if len(self._buffer) >= self._flush_records:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def seal_pending(self, reason: str) -> int:
        """Write explicit cancellations while keeping the recorder reusable."""
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("HSpec S4 trace cancellation reason must be non-empty")
        with self._lock:
            if self._finalized:
                self._flush_locked()
                return 0
            canceled = self._cancel_pending_locked(normalized_reason)
            self._flush_locked()
            return canceled

    def finalize(self) -> None:
        """Close trace-only pending selections when the worker exits.

        vLLM can stop a worker after its last selection without another
        scheduler update.  Such queries have no observable verification event;
        mark them as canceled instead of leaving an ambiguous trace hole.
        """
        with self._lock:
            if self._finalized:
                self._flush_locked()
                return
            self._cancel_pending_locked(HSPEC_S4_TRACE_SHUTDOWN_REASON)
            self._finalized = True
            self._flush_locked()

    def _cancel_pending_locked(self, reason: str) -> int:
        canceled = len(self._pending_selections)
        for pending in self._pending_selections.values():
            event = {
                "schema_version": "hspec.s4.online-trace.v2",
                "event": "cancellation",
                **pending,
                "reason": str(reason),
                "producer_id": self._producer_id,
                "producer_sequence": self._sequence,
            }
            self._sequence += 1
            self._buffer.append(event)
        self._pending_selections.clear()
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


def seal_hspec_s4_trace_pending(reason: str) -> int:
    if _RECORDER is None:
        return 0
    return _RECORDER.seal_pending(reason)


def finalize_hspec_s4_trace() -> None:
    if _RECORDER is not None:
        _RECORDER.finalize()


atexit.register(finalize_hspec_s4_trace)
