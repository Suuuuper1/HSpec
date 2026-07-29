"""Default-off draft-length patterns for the S7 verification-cost experiment.

This module does not select historical entries. It only caps the value returned
by the existing P0 selector when an explicit S7 experiment environment variable
is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_NAMED_PATTERNS = {
    "all0": 0,
    "all1": 1,
    "all2": 2,
    "all4": 4,
    "all8": 8,
    "all15": 15,
}
_COVERAGE_PATTERNS = {
    "coverage25_d8": 0.25,
    "coverage50_d8": 0.50,
    "coverage75_d8": 0.75,
    "coverage100_d8": 1.00,
}
_OTHER_PATTERNS = {"mixed", "equal_sum_lowmax", "equal_sum_highmax"}
VALID_PATTERNS = frozenset(
    set(_NAMED_PATTERNS) | set(_COVERAGE_PATTERNS) | _OTHER_PATTERNS
)


def parse_pattern_sequence(value: str) -> tuple[str, ...]:
    patterns = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(set(patterns) - VALID_PATTERNS)
    if unknown:
        raise ValueError(f"unsupported S7 verification pattern(s): {unknown}")
    return patterns


def _rotated_indices(size: int, rotation: int) -> list[int]:
    if size <= 0:
        return []
    offset = int(rotation) % size
    return [(offset + index) % size for index in range(size)]


def pattern_caps(pattern: str, batch_size: int, rotation: int = 0) -> list[int]:
    """Return the desired draft cap for each active P0 candidate row."""
    batch_size = max(int(batch_size), 0)
    pattern = str(pattern).strip().lower()
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"unsupported S7 verification pattern: {pattern!r}")
    if batch_size == 0:
        return []
    order = _rotated_indices(batch_size, rotation)
    caps = [0] * batch_size
    if pattern in _NAMED_PATTERNS:
        return [_NAMED_PATTERNS[pattern]] * batch_size
    if pattern in _COVERAGE_PATTERNS:
        count = int(round(_COVERAGE_PATTERNS[pattern] * batch_size))
        if _COVERAGE_PATTERNS[pattern] > 0:
            count = max(count, 1)
        for index in order[:count]:
            caps[index] = 8
        return caps
    if pattern == "mixed":
        actions = (0, 2, 4, 8, 15)
        for rank, index in enumerate(order):
            caps[index] = actions[rank % len(actions)]
        return caps
    if pattern == "equal_sum_lowmax":
        return [4] * batch_size

    # Match all4's total budget while concentrating it into long rows.
    remaining = 4 * batch_size
    for index in order:
        if remaining <= 0:
            break
        take = min(15, remaining)
        caps[index] = take
        remaining -= take
    return caps


@dataclass
class S7VerificationPatternController:
    patterns: tuple[str, ...] = ()
    block_calls: int = 512
    call_count: int = 0

    @classmethod
    def from_environment(cls) -> "S7VerificationPatternController":
        patterns = parse_pattern_sequence(
            os.getenv("HSPEC_S7_VERIFY_PATTERN_SEQUENCE", "")
        )
        try:
            block_calls = int(os.getenv("HSPEC_S7_VERIFY_PATTERN_BLOCK_CALLS", "512"))
        except ValueError as exc:
            raise ValueError("HSPEC_S7_VERIFY_PATTERN_BLOCK_CALLS must be an integer") from exc
        if patterns and block_calls < 1:
            raise ValueError("HSPEC_S7_VERIFY_PATTERN_BLOCK_CALLS must be >= 1")
        if patterns and os.getenv("HSPEC_S7_ENGINE_TIMING", "0") != "1":
            raise ValueError(
                "S7 verification patterns require HSPEC_S7_ENGINE_TIMING=1"
            )
        if patterns and (
            os.getenv("HSPEC_S4_EXTENT_REPLAY", "0") != "0"
            or os.getenv("HSPEC_S4_TRACE_DIR", "")
        ):
            raise ValueError("S7 verification patterns cannot be combined with S4 tracing")
        return cls(patterns=patterns, block_calls=max(block_calls, 1))

    @property
    def enabled(self) -> bool:
        return bool(self.patterns)

    def next_caps(self, batch_size: int) -> tuple[str | None, list[int] | None]:
        if not self.patterns:
            return None, None
        call_index = self.call_count
        self.call_count += 1
        pattern = self.patterns[(call_index // self.block_calls) % len(self.patterns)]
        rotation = call_index // self.block_calls
        return pattern, pattern_caps(pattern, batch_size, rotation)
