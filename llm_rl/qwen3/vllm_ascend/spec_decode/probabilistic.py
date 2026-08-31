# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Probability-preserving draft sampling and old-runner cache alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


_SAMPLING_EPS = 1e-5
_PROBABILITY_SUM_ATOL = 5e-5


class DraftSamplingWorkspace:
    """Persistent device buffers for allocation-free proposal sampling."""

    def __init__(
        self,
        max_rows: int,
        vocab_size: int,
        *,
        device: torch.device,
    ) -> None:
        if max_rows <= 0 or vocab_size <= 0:
            raise ValueError("draft sampling workspace dimensions must be positive")
        self.max_rows = int(max_rows)
        self.vocab_size = int(vocab_size)
        self.probabilities = torch.empty(
            self.max_rows, self.vocab_size, dtype=torch.float32, device=device
        )
        self.race = torch.empty_like(self.probabilities)
        self.row_temperatures = torch.empty(
            self.max_rows, dtype=torch.float32, device=device
        )
        self.safe_temperatures = torch.empty_like(self.row_temperatures)
        self.greedy_rows = torch.empty(
            self.max_rows, dtype=torch.bool, device=device
        )
        self.token_ids = torch.empty(
            self.max_rows, dtype=torch.int64, device=device
        )

    def rows(self, count: int) -> tuple[torch.Tensor, ...]:
        if not 0 < count <= self.max_rows:
            raise ValueError(
                f"draft sampling rows {count} exceed workspace {self.max_rows}"
            )
        return (
            self.probabilities[:count],
            self.race[:count],
            self.row_temperatures[:count],
            self.safe_temperatures[:count],
            self.greedy_rows[:count],
            self.token_ids[:count],
        )


def estimate_draft_probability_bytes(
    max_batch_size: int,
    num_speculative_tokens: int,
    vocab_size: int,
    *,
    method: str,
) -> dict[str, int]:
    """Return the persistent q size and a conservative proposal/cache peak."""
    for name, value in (
        ("max_batch_size", max_batch_size),
        ("num_speculative_tokens", num_speculative_tokens),
        ("vocab_size", vocab_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    row_bytes = max_batch_size * vocab_size * torch.float32.itemsize
    probability_bytes = row_bytes * num_speculative_tokens
    if method == "dflash":
        scratch_bytes = probability_bytes
    elif method == "dspark":
        scratch_bytes = 2 * row_bytes
    else:
        raise ValueError(f"unsupported probabilistic draft method: {method!r}")
    peak_bytes = probability_bytes + scratch_bytes
    return {
        "probability_bytes": probability_bytes,
        "peak_bytes": peak_bytes,
        "row_bytes": row_bytes,
        "scratch_bytes": scratch_bytes,
    }


def _validate_sampling_inputs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
    generators: Mapping[int, torch.Generator],
) -> int:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError(f"draft logits must be non-empty [rows,vocab], got {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError(f"draft logits must be floating point, got {logits.dtype}")
    if temperatures.ndim != 1 or temperatures.numel() == 0:
        raise ValueError("per-request temperatures must be a non-empty vector")
    if temperatures.device != logits.device:
        raise ValueError("draft logits and temperatures must be on the same device")
    if rows_per_request <= 0:
        raise ValueError("rows_per_request must be positive")
    batch_size = temperatures.shape[0]
    if logits.shape[0] != batch_size * rows_per_request:
        raise ValueError(
            "request-major draft row mismatch: "
            f"{logits.shape[0]} != {batch_size}*{rows_per_request}"
        )
    invalid_generators = sorted(set(generators) - set(range(batch_size)))
    if invalid_generators:
        raise ValueError(f"generator request indices are out of range: {invalid_generators}")
    return batch_size


def sample_draft_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    rows_per_request: int,
    all_greedy: bool,
    all_random: bool,
    generators: Mapping[int, torch.Generator] | None = None,
    workspace: DraftSamplingWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample request-major draft rows and preserve the exact float32 q.

    Only per-request temperature is intentionally applied. Target top-k/top-p
    and penalties remain target-side; standard rejection corrects q != p.
    """
    generators = generators or {}
    batch_size = _validate_sampling_inputs(
        logits, temperatures, rows_per_request, generators
    )
    if all_greedy:
        if workspace is None:
            return logits.argmax(dim=-1), None
        token_ids = workspace.rows(logits.shape[0])[-1]
        torch.argmax(logits, dim=-1, out=token_ids)
        return token_ids, None

    if workspace is None:
        row_temperatures = temperatures.repeat_interleave(rows_per_request)
        greedy_rows = row_temperatures < _SAMPLING_EPS
        safe_temperatures = torch.where(
            greedy_rows, torch.ones_like(row_temperatures), row_temperatures
        )
        probabilities = None
        race = None
        token_ids = None
    else:
        (
            probabilities,
            race,
            row_temperatures,
            safe_temperatures,
            greedy_rows,
            token_ids,
        ) = workspace.rows(logits.shape[0])
        row_temperatures.view(batch_size, rows_per_request).copy_(
            temperatures[:, None]
        )
        torch.lt(row_temperatures, _SAMPLING_EPS, out=greedy_rows)
        safe_temperatures.copy_(row_temperatures)
        safe_temperatures.masked_fill_(greedy_rows, 1.0)
    logits.div_(safe_temperatures.unsqueeze(-1))
    if probabilities is None:
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        torch.softmax(logits, dim=-1, dtype=torch.float32, out=probabilities)

    if not all_random:
        # The actual proposal for near-zero-temperature rows is one-hot. Store
        # that exact q even though the rejection sampler also has a greedy mask.
        greedy_token_ids = probabilities.argmax(dim=-1)
        if workspace is None:
            probabilities.mul_((~greedy_rows).unsqueeze(-1))
        else:
            probabilities.masked_fill_(greedy_rows.unsqueeze(-1), 0.0)
        row_indices = torch.arange(
            probabilities.shape[0], dtype=torch.long, device=probabilities.device
        )
        probabilities[row_indices, greedy_token_ids] += greedy_rows.to(
            probabilities.dtype
        )

    if race is None:
        race = torch.empty_like(probabilities)
    # Follow vLLM's one-kernel common path, then overwrite rows owned by private
    # generators. If every row has a private owner, skip the otherwise unused
    # global draw so unrelated global requests do not consume random numbers.
    if len(generators) != batch_size:
        race.exponential_()
    for request_index, generator in generators.items():
        start = request_index * rows_per_request
        race[start : start + rows_per_request].exponential_(generator=generator)
    torch.div(probabilities, race, out=race)
    if token_ids is None:
        token_ids = race.argmax(dim=-1)
    else:
        torch.argmax(race, dim=-1, out=token_ids)
    return token_ids, probabilities


def validate_probability_tensor(
    probabilities: torch.Tensor,
    *,
    expected_batch_size: int,
    expected_k: int,
    expected_vocab_size: int,
) -> None:
    expected = (expected_batch_size, expected_k, expected_vocab_size)
    if tuple(probabilities.shape) != expected:
        raise ValueError(
            f"draft probability shape mismatch: {tuple(probabilities.shape)} != {expected}"
        )
    if probabilities.dtype != torch.float32:
        raise TypeError(f"draft probabilities must be float32, got {probabilities.dtype}")
    if not probabilities.is_contiguous():
        raise ValueError("draft probabilities must be contiguous")
    # Phase 4 prioritizes fail-closed correctness. This bounded scalar check is
    # the only proposal-side D2H synchronization and is reported in cache stats.
    finite = torch.isfinite(probabilities).all()
    nonnegative = torch.all(probabilities >= 0)
    normalized = torch.all(
        torch.abs(probabilities.sum(dim=-1) - 1.0) <= _PROBABILITY_SUM_ATOL
    )
    valid = torch.logical_and(torch.logical_and(finite, nonnegative), normalized)
    if not bool(valid):
        raise ValueError(
            "draft probabilities contain non-finite, negative, or non-normalized rows"
        )


@dataclass
class DraftProbabilityCache:
    """Single-generation q cache between proposal and target verification."""

    enabled: bool
    num_speculative_tokens: int
    vocab_size: int
    probabilities: torch.Tensor | None = None
    request_ids: tuple[str, ...] | None = None
    generation: int = 0
    publish_count: int = 0
    consume_count: int = 0
    discard_count: int = 0
    peak_bytes: int = 0
    validation_sync_count: int = 0

    @property
    def current_bytes(self) -> int:
        if self.probabilities is None:
            return 0
        return self.probabilities.numel() * self.probabilities.element_size()

    def begin_proposal(self) -> None:
        if self.probabilities is not None or self.request_ids is not None:
            self.discard_count += 1
        self.probabilities = None
        self.request_ids = None

    def publish(
        self, probabilities: torch.Tensor | None, request_ids: Sequence[str]
    ) -> None:
        if not self.enabled:
            if probabilities is not None:
                raise RuntimeError("greedy proposal unexpectedly produced draft probabilities")
            return
        if probabilities is None:
            raise RuntimeError("probabilistic proposal produced no draft probabilities")
        ids = tuple(str(request_id) for request_id in request_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("draft probability request ids must be non-empty and unique")
        validate_probability_tensor(
            probabilities,
            expected_batch_size=len(ids),
            expected_k=self.num_speculative_tokens,
            expected_vocab_size=self.vocab_size,
        )
        self.validation_sync_count += 1
        self.generation += 1
        self.publish_count += 1
        self.probabilities = probabilities
        self.request_ids = ids
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def consume(
        self,
        request_ids: Sequence[str],
        num_draft_tokens: Sequence[int],
        *,
        require_probabilities: bool,
    ) -> torch.Tensor | None:
        try:
            if len(request_ids) != len(num_draft_tokens):
                raise ValueError("current request ids and draft lengths must align")
            active = [
                (str(request_id), int(length))
                for request_id, length in zip(request_ids, num_draft_tokens)
                if int(length) > 0
            ]
            active_ids = tuple(request_id for request_id, _ in active)
            if len(set(active_ids)) != len(active_ids):
                raise ValueError("current draft probability request ids are duplicated")
            if not require_probabilities:
                if self.probabilities is not None or self.request_ids is not None:
                    raise RuntimeError("stale draft probabilities reached a greedy verification")
                return None
            if not active:
                return None
            if self.probabilities is None or self.request_ids is None:
                raise RuntimeError("missing draft probabilities for probabilistic verification")
            if len(set(self.request_ids)) != len(self.request_ids):
                raise RuntimeError("cached draft probability request ids are duplicated")
            row_by_request = {
                request_id: index
                for index, request_id in enumerate(self.request_ids)
            }
            for request_id, length in active:
                if request_id not in row_by_request:
                    raise RuntimeError(
                        f"missing draft probability row for request {request_id!r}"
                    )
                if not 1 <= length <= self.num_speculative_tokens:
                    raise ValueError(
                        f"invalid draft length {length} for request {request_id!r}"
                    )

            if active_ids == self.request_ids and all(
                length == self.num_speculative_tokens for _, length in active
            ):
                aligned = self.probabilities.view(-1, self.vocab_size)
            else:
                aligned = torch.cat(
                    [
                        self.probabilities[row_by_request[request_id], :length]
                        for request_id, length in active
                    ],
                    dim=0,
                ).contiguous()
            expected_rows = sum(length for _, length in active)
            if tuple(aligned.shape) != (expected_rows, self.vocab_size):
                raise RuntimeError(
                    "flattened draft probability shape mismatch: "
                    f"{tuple(aligned.shape)} != {(expected_rows, self.vocab_size)}"
                )
            self.consume_count += 1
            return aligned
        finally:
            self.probabilities = None
            self.request_ids = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "generation": self.generation,
            "publish_count": self.publish_count,
            "consume_count": self.consume_count,
            "discard_count": self.discard_count,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "cached_request_count": len(self.request_ids or ()),
            "validation_sync_count": self.validation_sync_count,
        }
