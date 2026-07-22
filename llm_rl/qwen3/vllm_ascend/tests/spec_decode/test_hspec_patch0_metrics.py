import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_metrics_module():
    path = PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_metrics.py"
    name = "hspec_patch0_metrics_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


metrics = _load_metrics_module()


class TestSelectionWindowTracker(unittest.TestCase):

    def test_aligned_funnel_and_derived_metrics(self):
        tracker = metrics.HSpecSelectionMetricTracker(max_draft_tokens=15)
        window_id = tracker.begin_window(eligible_queries=3, active_table_version=2)
        self.assertIsNone(
            tracker.finalize_proposals(
                window_id,
                proposed_requests=2,
                drafted_tokens=5,
                stop_reasons={"score_gate": 1, "adaptive_window": 2},
                margin_sum=0.75,
                margin_count=3,
                timings={"total_ms": 0.3},
            )
        )
        self.assertIsNone(
            tracker.record_verification(
                window_id,
                accepted_prefix_len=2,
                emitted_tokens=3,
            )
        )
        counters = tracker.record_verification(
            window_id,
            accepted_prefix_len=0,
            emitted_tokens=1,
        )
        self.assertIsNotNone(counters)
        assert counters is not None
        self.assertEqual(counters["select_eligible_queries"], 3)
        self.assertEqual(counters["select_proposed_requests"], 2)
        self.assertEqual(counters["select_abstained_requests"], 1)
        self.assertEqual(counters["select_drafted_tokens"], 5)
        self.assertEqual(counters["select_verified_requests"], 2)
        self.assertEqual(counters["select_first_token_accepts"], 1)
        self.assertEqual(counters["select_accepted_tokens"], 2)
        self.assertEqual(counters["select_zero_accept_requests"], 1)
        self.assertEqual(counters["select_accept_len_0_count"], 1)
        self.assertEqual(counters["select_accept_len_2_count"], 1)
        self.assertTrue(all(metrics.is_selector_additive_metric(key) for key in counters))

        derived = metrics.derive_selector_metrics(counters)
        self.assertAlmostEqual(derived["select_proposal_coverage"], 2 / 3)
        self.assertAlmostEqual(derived["select_match_rate"], 0.5)
        self.assertAlmostEqual(derived["select_avg_accept_len"], 1.0)
        self.assertAlmostEqual(
            derived["select_avg_accept_length_accepted_only"], 2.0
        )
        self.assertAlmostEqual(
            derived["select_accepted_tokens_per_query"], 2 / 3
        )
        self.assertAlmostEqual(derived["select_accept_efficiency"], 0.4)
        self.assertAlmostEqual(derived["select_active_table_version_mean"], 2.0)
        self.assertAlmostEqual(
            derived["select_active_table_version_variance"], 0.0
        )

    def test_abstain_closes_without_pending_verification(self):
        tracker = metrics.HSpecSelectionMetricTracker(max_draft_tokens=15)
        window_id = tracker.begin_window(eligible_queries=4)
        counters = tracker.finalize_proposals(
            window_id,
            proposed_requests=0,
            drafted_tokens=0,
            stop_reasons={"score_gate": 4},
        )
        self.assertIsNotNone(counters)
        assert counters is not None
        self.assertEqual(counters["select_abstained_requests"], 4)
        self.assertEqual(counters["select_verified_requests"], 0)
        self.assertEqual(tracker.pending_window_count, 0)
        self.assertIsNone(tracker.record_verification(window_id, accepted_prefix_len=1))

    def test_abort_closes_once_and_duplicate_terminal_is_ignored(self):
        tracker = metrics.HSpecSelectionMetricTracker(max_draft_tokens=15)
        window_id = tracker.begin_window(eligible_queries=1)
        self.assertIsNone(
            tracker.finalize_proposals(
                window_id,
                proposed_requests=1,
                drafted_tokens=4,
            )
        )
        counters = tracker.record_cancellation(window_id)
        self.assertIsNotNone(counters)
        assert counters is not None
        self.assertEqual(counters["select_canceled_requests"], 1)
        self.assertEqual(counters["select_verified_requests"], 0)
        self.assertIsNone(tracker.record_cancellation(window_id))
        self.assertIsNone(tracker.record_verification(window_id, accepted_prefix_len=0))

    def test_unknown_stop_reason_is_bounded_and_length_mismatch_is_counted(self):
        tracker = metrics.HSpecSelectionMetricTracker(max_draft_tokens=2)
        window_id = tracker.begin_window(eligible_queries=1)
        tracker.finalize_proposals(
            window_id,
            proposed_requests=1,
            drafted_tokens=2,
            stop_reasons={"request-specific-unbounded-value": 1},
        )
        counters = tracker.record_verification(
            window_id,
            accepted_prefix_len=3,
            drafted_length_mismatch=True,
        )
        assert counters is not None
        self.assertEqual(counters["select_stop_other_count"], 1)
        self.assertEqual(counters["select_accept_len_overflow_count"], 1)
        self.assertEqual(counters["select_drafted_length_mismatch_count"], 1)
        self.assertNotIn("request-specific-unbounded-value", " ".join(counters))


class TestSelectorMetricStore(unittest.TestCase):

    def test_interval_reset_and_cumulative_monotonicity(self):
        store = metrics.HSpecSelectorMetricStore()
        store.record({
            "select_metric_windows": 1,
            "select_eligible_queries": 4,
            "not_a_selector_key": 99,
        })
        window_id, interval, cumulative = store.snapshot_and_reset()
        self.assertEqual(window_id, 1)
        self.assertEqual(interval["select_eligible_queries"], 4)
        self.assertNotIn("not_a_selector_key", interval)
        self.assertEqual(cumulative["select_eligible_queries"], 4)

        same_window_id, empty_interval, same_cumulative = store.snapshot_and_reset()
        self.assertEqual(same_window_id, 1)
        self.assertEqual(empty_interval, {})
        self.assertEqual(same_cumulative["select_eligible_queries"], 4)

        store.record({"select_metric_windows": 1, "select_eligible_queries": 3})
        next_window_id, next_interval, next_cumulative = store.snapshot_and_reset()
        self.assertEqual(next_window_id, 2)
        self.assertEqual(next_interval["select_eligible_queries"], 3)
        self.assertEqual(next_cumulative["select_eligible_queries"], 7)

    def test_abs_delta_buckets_have_fixed_boundaries(self):
        cases = {
            0: "0",
            1: "1_2",
            2: "1_2",
            3: "3_8",
            8: "3_8",
            9: "9_32",
            32: "9_32",
            33: "33_64",
            64: "33_64",
            65: "65_256",
            256: "65_256",
            257: "gt_256",
            1000000: "gt_256",
        }
        self.assertEqual(
            {value: metrics.hspec_abs_delta_bucket(value) for value in cases},
            cases,
        )


if __name__ == "__main__":
    unittest.main()
