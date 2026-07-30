import os
import unittest
from unittest.mock import patch

import vllm.v1.engine.core as engine_core_module
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm_ascend.spec_decode.hspec_s7_benchmark import (
    S7VerificationPatternController,
    parse_pattern_sequence,
    pattern_caps,
)


class TestS7VerificationPatterns(unittest.TestCase):
    def test_default_is_disabled_and_returns_no_override(self):
        with patch.dict(os.environ, {}, clear=True):
            controller = S7VerificationPatternController.from_environment()
        self.assertFalse(controller.enabled)
        self.assertEqual(controller.next_caps(8), (None, None))

    def test_fixed_coverage_and_mixed_patterns(self):
        self.assertEqual(pattern_caps("all8", 4), [8, 8, 8, 8])
        coverage = pattern_caps("coverage25_d8", 8)
        self.assertEqual(sum(value > 0 for value in coverage), 2)
        self.assertEqual(sum(coverage), 16)
        self.assertEqual(sorted(pattern_caps("mixed", 5)), [0, 2, 4, 8, 15])

    def test_equal_sum_patterns_change_max_only(self):
        low = pattern_caps("equal_sum_lowmax", 16)
        high = pattern_caps("equal_sum_highmax", 16)
        self.assertEqual(sum(low), sum(high))
        self.assertEqual(max(low), 4)
        self.assertEqual(max(high), 15)

    def test_sequence_blocks_and_rotation_are_deterministic(self):
        controller = S7VerificationPatternController(
            patterns=("all0", "all2"), block_calls=2
        )
        observed = [controller.next_caps(3)[0] for _ in range(5)]
        self.assertEqual(observed, ["all0", "all0", "all2", "all2", "all0"])

    def test_unknown_pattern_and_invalid_block_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_pattern_sequence("all0,unknown")
        with patch.dict(
            os.environ,
            {
                "HSPEC_S7_VERIFY_PATTERN_SEQUENCE": "all0",
                "HSPEC_S7_VERIFY_PATTERN_BLOCK_CALLS": "0",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                S7VerificationPatternController.from_environment()

    def test_pattern_injection_requires_timing_and_excludes_s4(self):
        with patch.dict(
            os.environ,
            {"HSPEC_S7_VERIFY_PATTERN_SEQUENCE": "all0"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                S7VerificationPatternController.from_environment()
        with patch.dict(
            os.environ,
            {
                "HSPEC_S7_VERIFY_PATTERN_SEQUENCE": "all0",
                "HSPEC_S7_ENGINE_TIMING": "1",
                "HSPEC_S7_OBSERVER_SCHEMA": "v2",
                "HSPEC_S4_EXTENT_REPLAY": "1",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                S7VerificationPatternController.from_environment()

    def test_pattern_injection_requires_observer_v2(self):
        with patch.dict(
            os.environ,
            {
                "HSPEC_S7_VERIFY_PATTERN_SEQUENCE": "all0",
                "HSPEC_S7_ENGINE_TIMING": "1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "OBSERVER_SCHEMA=v2"):
                S7VerificationPatternController.from_environment()

    def test_inproc_client_completes_s7_timing(self):
        class FakeCore:
            hspec_s7_timing_enabled = True

            def __init__(self):
                self.completed = None
                self.posted = None

            def step_fn(self):
                return {0: "output"}, True

            def post_step(self, model_executed):
                self.posted = model_executed

            def complete_hspec_s7_step_timing(self, **kwargs):
                self.completed = kwargs

        client = object.__new__(InprocClient)
        client.engine_core = FakeCore()
        self.assertEqual(client.get_output(), "output")
        self.assertTrue(client.engine_core.posted)
        self.assertEqual(client.engine_core.completed["transport"], "inproc")
        self.assertEqual(
            client.engine_core.completed["core_end_ns"],
            client.engine_core.completed["output_queue_end_ns"],
        )

    def test_engine_timing_completion_samples_and_clears(self):
        fake = type(
            "FakeEngineCore",
            (),
            {
                "_hspec_s7_last_step_timing": {"draft_sum": 0},
                "_hspec_s7_timing_ordinal": 0,
            },
        )()
        with (
            patch.object(engine_core_module, "_HSPEC_S7_ENGINE_TIMING", True),
            patch.object(engine_core_module, "_HSPEC_S7_OBSERVER_SAMPLE_EVERY", 4),
            patch.object(engine_core_module.logger, "info") as log_info,
        ):
            EngineCore.complete_hspec_s7_step_timing(
                fake,
                loop_start_ns=0,
                core_end_ns=10,
                output_queue_end_ns=10,
                post_end_ns=20,
                transport="inproc",
            )
        self.assertIsNone(fake._hspec_s7_last_step_timing)
        self.assertEqual(fake._hspec_s7_timing_ordinal, 1)
        log_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
