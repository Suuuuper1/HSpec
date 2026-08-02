# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded asynchronous online trace for the S9 R1 shadow gate.

This observer is inert unless ``HSPEC_S9_SHADOW_DIR`` is configured.  The
decode thread only builds bounded records and performs a non-blocking queue
put; JSON serialization and filesystem writes belong to a worker-local writer
thread.  Recorder failures never change selector output and are made fatal by
the offline S9 gate through the status sidecar.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import socket
import threading
from pathlib import Path
from typing import Mapping, Sequence


_TRACE_DIR = os.environ.get("HSPEC_S9_SHADOW_DIR", "").strip()
HSPEC_S9_SHADOW_ENABLED = bool(_TRACE_DIR)
_PRODUCER_RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))


def _positive_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except (TypeError, ValueError):
        return int(default)


HSPEC_S9_SAMPLE_EVERY = _positive_int("HSPEC_S9_SAMPLE_EVERY", 1000)
HSPEC_S9_TIMING_SAMPLE_EVERY = _positive_int(
    "HSPEC_S9_TIMING_SAMPLE_EVERY", 1
)
HSPEC_S9_CAPTURE_PROJECTED = (
    os.environ.get("HSPEC_S9_CAPTURE_PROJECTED", "1") != "0"
)
HSPEC_S9_CAPTURE_ALL_DIVERGENCE = (
    os.environ.get("HSPEC_S9_CAPTURE_ALL_DIVERGENCE", "0") != "0"
)
HSPEC_S9_DIVERGENCE_SAMPLE_EVERY = _positive_int(
    "HSPEC_S9_DIVERGENCE_SAMPLE_EVERY", 256
)


class _FlushRequest:
    def __init__(self) -> None:
        self.ack = threading.Event()


def hspec_s9_token_hash(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(len(tokens).to_bytes(8, byteorder="little", signed=False))
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _query_payload(
    request_id: str,
    prompt_id: str,
    decoded_len: int,
    active_table_version: int,
) -> bytes:
    fields = (
        socket.gethostname(),
        str(_PRODUCER_RANK),
        str(request_id),
        str(prompt_id),
        str(int(decoded_len)),
        str(int(active_table_version)),
    )
    return "\x1f".join(fields).encode("utf-8", errors="surrogatepass")


def hspec_s9_query_identity(
    request_id: str,
    prompt_id: str,
    decoded_len: int,
    active_table_version: int,
) -> tuple[str, bool]:
    """Return a stable identity and deterministic ordinary-sample decision."""
    payload = _query_payload(
        request_id, prompt_id, decoded_len, active_table_version
    )
    digest = hashlib.sha256(payload).digest()
    sampled = not (
        int.from_bytes(digest[:8], byteorder="little", signed=False)
        % HSPEC_S9_SAMPLE_EVERY
    )
    query_id = hashlib.sha256(b"hspec-s9-query-v1\x00" + payload).hexdigest()
    return query_id, sampled


def hspec_s9_divergence_sampled(query_id: str) -> bool:
    digest = hashlib.sha256(
        b"hspec-s9-divergence-sample-v1\x00" + str(query_id).encode("ascii")
    ).digest()
    return not (
        int.from_bytes(digest[:8], byteorder="little", signed=False)
        % HSPEC_S9_DIVERGENCE_SAMPLE_EVERY
    )


class HSpecS9ShadowRecorder:
    """Non-blocking producer with one worker-local JSONL consumer thread."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        queue_records: int = 8192,
        flush_records: int = 256,
    ) -> None:
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._queue: queue.Queue[object] = queue.Queue(
            maxsize=max(int(queue_records), 1)
        )
        self._flush_records = max(int(flush_records), 1)
        self._pid = os.getpid()
        host = socket.gethostname().replace("/", "_")
        self._producer_id = f"{host}:rank-{_PRODUCER_RANK}:pid-{self._pid}"
        stem = f"s9-shadow-{host}-rank-{_PRODUCER_RANK}-pid-{self._pid}"
        self._path = self._output_dir / f"{stem}.jsonl"
        self._status_path = self._output_dir / f"{stem}.status.json"
        self._lock = threading.Lock()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._write_errors = 0
        self._sequence = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_main,
            name=f"hspec-s9-writer-{self._pid}",
            daemon=True,
        )
        self._thread.start()

    @property
    def path(self) -> Path:
        return self._path

    def record_many(self, events: Sequence[Mapping[str, object]]) -> bool:
        if not events:
            return True
        if os.getpid() != self._pid:
            with self._lock:
                self._dropped += len(events)
            return False
        batch = tuple(dict(event) for event in events)
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            with self._lock:
                self._dropped += len(batch)
            return False
        with self._lock:
            self._enqueued += len(batch)
        return True

    def flush(self, reason: str) -> None:
        """Drain outside the selector hot path and publish an audit sidecar."""
        if os.getpid() != self._pid:
            return
        request = _FlushRequest()
        self._queue.put(request)
        self._queue.join()
        request.ack.wait(timeout=10.0)
        self._publish_status(reason=reason, closed=False, quiescent=True)

    def close(self, reason: str = "worker_shutdown") -> None:
        if os.getpid() != self._pid:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=10.0)
        self._publish_status(reason=reason, closed=True, quiescent=True)

    def _writer_main(self) -> None:
        buffer: list[dict[str, object]] = []
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._write_buffer(buffer)
                    return
                if isinstance(item, _FlushRequest):
                    self._write_buffer(buffer)
                    item.ack.set()
                    continue
                assert isinstance(item, tuple)
                for raw_event in item:
                    event = dict(raw_event)
                    event.setdefault("schema_version", "hspec.s9.shadow-trace.v1")
                    event["producer_id"] = self._producer_id
                    event["producer_sequence"] = self._sequence
                    self._sequence += 1
                    buffer.append(event)
                if len(buffer) >= self._flush_records:
                    self._write_buffer(buffer)
            except Exception:
                dropped = (
                    len(buffer)
                    if buffer else (len(item) if isinstance(item, tuple) else 0)
                )
                with self._lock:
                    self._write_errors += 1
                    self._dropped += dropped
                buffer.clear()
                if isinstance(item, _FlushRequest):
                    item.ack.set()
            finally:
                self._queue.task_done()

    def _write_buffer(self, buffer: list[dict[str, object]]) -> None:
        if not buffer:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in buffer
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
        with self._lock:
            self._written += len(buffer)
        buffer.clear()

    def _write_status(
        self, *, reason: str, closed: bool, quiescent: bool
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "schema_version": "hspec.s9.shadow-recorder-status.v1",
                "producer_id": self._producer_id,
                "trace_path": str(self._path),
                "enqueued_records": self._enqueued,
                "written_records": self._written,
                "dropped_records": self._dropped,
                "write_errors": self._write_errors,
                "queue_unfinished_tasks": int(self._queue.unfinished_tasks),
                "reason": str(reason),
                "closed": bool(closed),
                "quiescent": bool(quiescent),
            }
        temporary = self._status_path.with_suffix(
            self._status_path.suffix + f".tmp-{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._status_path)

    def _publish_status(
        self, *, reason: str, closed: bool, quiescent: bool
    ) -> None:
        try:
            self._write_status(
                reason=reason, closed=closed, quiescent=quiescent
            )
        except Exception:
            # Missing/stale status is an offline-gate failure. It must not
            # perturb the target rollout or selector output.
            with self._lock:
                self._write_errors += 1


_RECORDER: HSpecS9ShadowRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def _get_recorder() -> HSpecS9ShadowRecorder:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = HSpecS9ShadowRecorder(
                _TRACE_DIR,
                queue_records=_positive_int("HSPEC_S9_QUEUE_RECORDS", 8192),
                flush_records=_positive_int("HSPEC_S9_FLUSH_RECORDS", 256),
            )
        return _RECORDER


def record_hspec_s9_shadow_events(
    events: Sequence[Mapping[str, object]],
) -> bool:
    if not HSPEC_S9_SHADOW_ENABLED or not events:
        return True
    return _get_recorder().record_many(events)


def flush_hspec_s9_shadow(reason: str) -> None:
    if HSPEC_S9_SHADOW_ENABLED and _RECORDER is not None:
        _RECORDER.flush(reason)


def _finalize() -> None:  # pragma: no cover - interpreter shutdown
    if _RECORDER is not None:
        try:
            _RECORDER.close()
        except Exception:
            pass


atexit.register(_finalize)
