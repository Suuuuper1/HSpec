import numpy as np

from vllm_ascend.spec_decode.hspec_selector_consensus import (
    HSPEC_CONSENSUS_NUMBA_AVAILABLE,
    HSPEC_CONSENSUS_MODE_C1,
    HSPEC_CONSENSUS_MODE_C2,
    HSPEC_CONSENSUS_STOP_DISAGREEMENT,
    HSPEC_CONSENSUS_WEIGHT_UTILITY,
    prefix_consensus_python,
    warm_consensus_selector,
)


def _inputs():
    utilities = np.asarray([1.0, 0.9, 0.8, 0.2], dtype=np.float64)
    p1 = np.asarray([0.8, 0.8, 0.8, 0.8], dtype=np.float64)
    rollouts = np.asarray([0, 1, 2, 3], dtype=np.int16)
    tokens = np.asarray([
        [10, 20, 30, 40],
        [11, 21, 31, 41],
        [11, 21, 32, 42],
        [10, 22, 33, 43],
    ], dtype=np.int32)
    lengths = np.full((4,), 4, dtype=np.int16)
    return utilities, p1, rollouts, tokens, lengths


def test_c1_votes_first_token_then_follows_best_compatible_entry():
    result = prefix_consensus_python(
        *_inputs(), 0.0, HSPEC_CONSENSUS_WEIGHT_UTILITY, 10.0, False,
        HSPEC_CONSENSUS_MODE_C1, 0.0, 4,
    )
    assert result[0][:result[1]].tolist() == [11, 21, 31, 41]
    assert result[2] == 1


def test_c2_prefix_always_belongs_to_one_alive_candidate():
    utilities, p1, rollouts, tokens, lengths = _inputs()
    result = prefix_consensus_python(
        utilities, p1, rollouts, tokens, lengths, 0.0,
        HSPEC_CONSENSUS_WEIGHT_UTILITY, 10.0, False,
        HSPEC_CONSENSUS_MODE_C2, 0.0, 4,
    )
    prefix = result[0][:result[1]].tolist()
    assert prefix
    assert any(row[:len(prefix)].tolist() == prefix for row in tokens)
    assert result[2] >= 0


def test_c2_stops_before_low_consensus_token_and_respects_length_cap():
    result = prefix_consensus_python(
        *_inputs(), 0.0, HSPEC_CONSENSUS_WEIGHT_UTILITY, 10.0, False,
        HSPEC_CONSENSUS_MODE_C2, 0.75, 2,
    )
    assert result[1] == 0
    assert result[3] == HSPEC_CONSENSUS_STOP_DISAGREEMENT


def test_p3_abstain_boundary_cannot_be_revived():
    utilities, p1, rollouts, tokens, lengths = _inputs()
    result = prefix_consensus_python(
        utilities, p1, rollouts, tokens, lengths, 2.0,
        HSPEC_CONSENSUS_WEIGHT_UTILITY, 1.0, False,
        HSPEC_CONSENSUS_MODE_C2, 0.0, 4,
    )
    assert result[1] == 0 and result[2] == -1


def test_rollout_normalization_reduces_duplicate_entry_mass():
    # Summed duplicate mass wins before normalization; the slightly stronger
    # independent candidate wins once every rollout carries one unit of mass.
    utilities = np.asarray([0.99, 0.98, 1.0], dtype=np.float64)
    p1 = np.ones((3,), dtype=np.float64)
    rollouts = np.asarray([0, 0, 1], dtype=np.int16)
    tokens = np.asarray([[10], [10], [20]], dtype=np.int32)
    lengths = np.ones((3,), dtype=np.int16)
    without = prefix_consensus_python(
        utilities, p1, rollouts, tokens, lengths, 0.0,
        HSPEC_CONSENSUS_WEIGHT_UTILITY, 100.0, False,
        HSPEC_CONSENSUS_MODE_C2, 0.0, 1,
    )
    with_normalization = prefix_consensus_python(
        utilities, p1, rollouts, tokens, lengths, 0.0,
        HSPEC_CONSENSUS_WEIGHT_UTILITY, 100.0, True,
        HSPEC_CONSENSUS_MODE_C2, 0.0, 1,
    )
    assert without[0][0] == 10
    assert with_normalization[0][0] == 20


def test_numba_kernel_warms_in_nopython_mode():
    if HSPEC_CONSENSUS_NUMBA_AVAILABLE:
        warm_consensus_selector()
