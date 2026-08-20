# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded non-blocking sequential trace for S14 replay."""

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


_TRACE_DIR = os.environ.get("HSPEC_S14_TRACE_DIR", "").strip()
HSPEC_S14_TRACE_ENABLED = bool(_TRACE_DIR)
_PRODUCER_RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))


def _positive_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except (TypeError, ValueError):
        return default


HSPEC_S14_SAMPLE_PROMPT_EVERY = _positive_int(
    "HSPEC_S14_SAMPLE_PROMPT_EVERY", 64
)
HSPEC_S14_SAMPLE_REQUEST_EVERY = _positive_int(
    "HSPEC_S14_SAMPLE_REQUEST_EVERY", 32
)


def hspec_s14_prompt_sampled(prompt_id: str) -> bool:
    payload = str(prompt_id).encode("utf-8", errors="surrogatepass")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % HSPEC_S14_SAMPLE_PROMPT_EVERY == 0


class HSpecS14RequestSampler:
    """Deterministically sample whole request trajectories across prompts.

    The first request observed for every eligible prompt is always retained.
    Sampling individual decode queries would destroy sequential replay, while
    sampling prompts alone spends the trace budget on deep pseudoreplication.
    """

    def __init__(
        self,
        prompt_every: int = HSPEC_S14_SAMPLE_PROMPT_EVERY,
        request_every: int = HSPEC_S14_SAMPLE_REQUEST_EVERY,
    ) -> None:
        self._prompt_every = max(int(prompt_every), 1)
        self._request_every = max(int(request_every), 1)
        self._first_request_by_prompt: dict[str, str] = {}
        self._decisions: dict[tuple[str, str], bool] = {}
        self._lock = threading.Lock()

    def sampled(self, prompt_id: str, request_id: str) -> bool:
        prompt = str(prompt_id)
        request = str(request_id)
        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8", errors="surrogatepass")
        ).digest()
        if int.from_bytes(prompt_hash[:8], "little") % self._prompt_every:
            return False

        key = (prompt, request)
        with self._lock:
            cached = self._decisions.get(key)
            if cached is not None:
                return cached
            first = self._first_request_by_prompt.get(prompt)
            if first is None:
                self._first_request_by_prompt[prompt] = request
                decision = True
            elif first == request or self._request_every == 1:
                decision = True
            else:
                payload = "\x1f".join(key).encode(
                    "utf-8", errors="surrogatepass"
                )
                value = int.from_bytes(
                    hashlib.sha256(
                        b"hspec-s14-request-sample-v1\x00" + payload
                    ).digest()[:8],
                    "little",
                )
                decision = value % self._request_every == 0
            self._decisions[key] = decision
            return decision


_REQUEST_SAMPLER = HSpecS14RequestSampler()


def hspec_s14_request_sampled(prompt_id: str, request_id: str) -> bool:
    return _REQUEST_SAMPLER.sampled(prompt_id, request_id)


def hspec_s14_query_id(
    request_id: str,
    prompt_id: str,
    decoded_len: int,
    table_version: int,
    cache_generation: int,
    query_sequence: int,
) -> str:
    payload = "\x1f".join((
        socket.gethostname(), str(_PRODUCER_RANK), str(request_id),
        str(prompt_id), str(int(decoded_len)), str(int(table_version)),
        str(int(cache_generation)), str(int(query_sequence)),
    )).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(b"hspec-s14-query-v1\x00" + payload).hexdigest()


class _FlushRequest:
    def __init__(self) -> None:
        self.ack = threading.Event()


class HSpecS14TraceRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir).expanduser().resolve()
        capacity = _positive_int("HSPEC_S14_QUEUE_RECORDS", 8192)
        self._flush_records = _positive_int("HSPEC_S14_FLUSH_RECORDS", 256)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self._pid = os.getpid()
        host = socket.gethostname().replace("/", "_")
        stem = f"s14-trace-{host}-rank-{_PRODUCER_RANK}-pid-{self._pid}"
        self._path = self._output_dir / f"{stem}.jsonl"
        self._status_path = self._output_dir / f"{stem}.status.json"
        self._producer_id = f"{host}:rank-{_PRODUCER_RANK}:pid-{self._pid}"
        self._lock = threading.Lock()
        self._enqueued = self._written = self._dropped = self._write_errors = 0
        self._sequence = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_main, name=f"hspec-s14-writer-{self._pid}",
            daemon=True,
        )
        self._thread.start()

    def record_many(self, events: Sequence[Mapping[str, object]]) -> bool:
        if not events:
            return True
        batch = tuple(dict(event) for event in events)
        if os.getpid() != self._pid:
            with self._lock:
                self._dropped += len(batch)
            return False
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
        if os.getpid() != self._pid:
            return
        request = _FlushRequest()
        self._queue.put(request)
        self._queue.join()
        request.ack.wait(timeout=10.0)
        self._publish_status(reason, False)

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
        self._publish_status(reason, True)

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
                for raw in item:
                    event = dict(raw)
                    event.setdefault("schema_version", "hspec.s14.sequential-trace.v1")
                    event["producer_id"] = self._producer_id
                    event["producer_sequence"] = self._sequence
                    self._sequence += 1
                    buffer.append(event)
                if len(buffer) >= self._flush_records:
                    self._write_buffer(buffer)
            except Exception:
                with self._lock:
                    self._write_errors += 1
                    self._dropped += len(buffer)
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

    def _publish_status(self, reason: str, closed: bool) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "schema_version": "hspec.s14.trace-status.v1",
                    "producer_id": self._producer_id,
                    "trace_path": str(self._path),
                    "enqueued_records": self._enqueued,
                    "written_records": self._written,
                    "dropped_records": self._dropped,
                    "write_errors": self._write_errors,
                    "queue_unfinished_tasks": int(self._queue.unfinished_tasks),
                    "reason": str(reason), "closed": bool(closed),
                    "quiescent": True,
                }
            temporary = self._status_path.with_suffix(
                self._status_path.suffix + f".tmp-{os.getpid()}"
            )
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self._status_path)
        except Exception:
            with self._lock:
                self._write_errors += 1


_RECORDER: HSpecS14TraceRecorder | None = None
_LOCK = threading.Lock()


def _get_recorder() -> HSpecS14TraceRecorder:
    global _RECORDER
    with _LOCK:
        if _RECORDER is None:
            _RECORDER = HSpecS14TraceRecorder(_TRACE_DIR)
        return _RECORDER


def record_hspec_s14_trace_events(
    events: Sequence[Mapping[str, object]],
) -> bool:
    if not HSPEC_S14_TRACE_ENABLED or not events:
        return True
    return _get_recorder().record_many(events)


def flush_hspec_s14_trace(reason: str) -> None:
    if HSPEC_S14_TRACE_ENABLED and _RECORDER is not None:
        _RECORDER.flush(reason)


def _finalize() -> None:  # pragma: no cover
    if _RECORDER is not None:
        try:
            _RECORDER.close()
        except Exception:
            pass


atexit.register(_finalize)
