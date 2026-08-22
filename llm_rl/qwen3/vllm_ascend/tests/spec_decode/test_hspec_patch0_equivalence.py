import ast
import copy
import inspect
import runpy
import subprocess
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
S0_FROZEN_HEAD = "8d50448b1f079d1fb8b72e7af908c8b231879cee"
S0_PROPOSER_PATH = (
    "llm_rl/qwen3/vllm_ascend/spec_decode/hspec_proposer.py"
)


def _extract_method_from_source(
    source: str,
    filename: str,
    class_name: str,
    method_name: str,
    extra_globals: Optional[dict] = None,
):
    tree = ast.parse(source, filename=filename)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    function = copy.deepcopy(item)
                    function.decorator_list = []
                    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
                    namespace = {
                        "torch": torch,
                        "Dict": Dict,
                        "Optional": Optional,
                    }
                    namespace.update(extra_globals or {})
                    exec(compile(module, filename, "exec"), namespace)
                    return namespace[method_name]
    raise AssertionError(f"Could not find {class_name}.{method_name}")


def _extract_method(
    path: Path,
    class_name: str,
    method_name: str,
    extra_globals: Optional[dict] = None,
):
    return _extract_method_from_source(
        path.read_text(encoding="utf-8"),
        str(path),
        class_name,
        method_name,
        extra_globals=extra_globals,
    )


class TestHardmaxEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.method = staticmethod(_extract_method(
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py",
            "HSpecProposer",
            "_match_projected_batch",
            extra_globals={"_HSpecMatchResult": SimpleNamespace},
        ))

    def test_default_and_observation_path_keep_exact_hardmax_output(self):
        z = torch.tensor([[1.0, 0.5], [0.5, -1.0], [2.0, 1.0]])
        keys = torch.tensor([
            [[1.0, 0.0], [0.2, 1.0], [0.9, 0.1], [-1.0, 0.0]],
            [[1.0, 1.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            [[0.0, 1.0], [1.0, 0.0], [9.0, 9.0], [8.0, 8.0]],
        ])
        invalid = torch.tensor([
            [False, False, False, False],
            [False, True, True, True],
            [False, False, True, True],
        ])
        sims = torch.bmm(keys, z.unsqueeze(-1)).squeeze(-1)
        expected_sims = sims.masked_fill(invalid, torch.finfo(sims.dtype).min)
        expected_values, expected_indices = expected_sims.max(dim=1)

        result = self.method(object(), z, keys, invalid)
        self.assertTrue(torch.equal(result.raw_top1_scores, expected_values))
        self.assertTrue(torch.equal(result.raw_top1_indices, expected_indices))
        self.assertTrue(
            torch.equal(result.candidate_scores[:, 0], expected_values)
        )
        self.assertTrue(
            torch.equal(result.candidate_indices[:, 0], expected_indices)
        )

        observed = self.method(
            object(), z, keys, invalid, observe_margin=True
        )
        self.assertTrue(torch.equal(observed.raw_top1_scores, expected_values))
        self.assertTrue(torch.equal(observed.raw_top1_indices, expected_indices))
        assert observed.margins is not None
        self.assertTrue(torch.isnan(observed.margins[1]))
        self.assertAlmostEqual(float(observed.margins[0]), 0.05, places=6)

    def test_frozen_s0_and_patch0_produce_identical_drafts(self):
        frozen_source = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "show",
                f"{S0_FROZEN_HEAD}:{S0_PROPOSER_PATH}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        frozen_method = _extract_method_from_source(
            frozen_source,
            f"{S0_FROZEN_HEAD}:{S0_PROPOSER_PATH}",
            "HSpecProposer",
            "_match_projected_batch",
        )

        generator = torch.Generator().manual_seed(20260721)
        projected = torch.randn(17, 64, generator=generator)
        keys = torch.randn(17, 23, 64, generator=generator)
        invalid = torch.rand(17, 23, generator=generator) < 0.2
        invalid[:, 0] = False
        # Exercise the original first-index tie behavior as well as masked entries.
        keys[0, 1] = keys[0, 0]
        invalid[1, -4:] = True

        frozen_scores, frozen_indices = frozen_method(
            object(), projected, keys, invalid
        )
        patch0 = self.method(
            object(), projected, keys, invalid
        )
        patch0_scores = patch0.raw_top1_scores
        patch0_indices = patch0.raw_top1_indices
        self.assertTrue(torch.equal(patch0_scores, frozen_scores))
        self.assertTrue(torch.equal(patch0_indices, frozen_indices))

        entry_drafts = torch.arange(17 * 23 * 15).reshape(17, 23, 15)
        rows = torch.arange(projected.shape[0])
        self.assertTrue(
            torch.equal(
                entry_drafts[rows, patch0_indices],
                entry_drafts[rows, frozen_indices],
            )
        )

    def test_hardmax_function_contract_and_s18_release_default_are_explicit(self):
        signature = inspect.signature(self.method)
        self.assertEqual(signature.parameters["topk"].default, 1)
        self.assertEqual(signature.parameters["sim_mode"].default, "raw")
        self.assertEqual(signature.parameters["observe_margin"].default, False)
        source = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'os.environ.get("HSPEC_SELECT_MODE", "topk_position")', source
        )
        self.assertIn('_get_env_int("HSPEC_SELECT_TOPK", 8, 1)', source)
        self.assertIn(
            "raw_top1_scores, raw_top1_indices = sims.max(dim=1)", source
        )
        tree = ast.parse(source)
        generate_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_generate_token_ids_impl":
                generate_method = node
                break
        self.assertIsNotNone(generate_method)
        assigned_names = {
            target.id
            for node in ast.walk(generate_method)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("selector_timing_enabled", assigned_names)
        self.assertIn("observe_selector_clock", assigned_names)


class TestVerificationEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.method = staticmethod(_extract_method(
            PROJECT_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py",
            "NPUModelRunner",
            "_hspec_compute_accepted_prefix_lengths",
        ))

    def test_longest_common_prefix_and_inputs_are_unchanged(self):
        fake_runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["full", "partial", "none"])
        )
        scheduler = SimpleNamespace(
            scheduled_spec_decode_tokens={
                "full": [1, 2, 3],
                "partial": [4, 5, 6],
            }
        )
        outputs = [[1, 2, 3, 9], [4, 8, 6], [7]]
        outputs_before = copy.deepcopy(outputs)
        drafts_before = copy.deepcopy(scheduler.scheduled_spec_decode_tokens)
        accepted = self.method(fake_runner, scheduler, outputs)
        self.assertEqual(accepted, [3, 1, 0])
        self.assertEqual(outputs, outputs_before)
        self.assertEqual(scheduler.scheduled_spec_decode_tokens, drafts_before)

    def test_pending_metadata_is_compact_and_fixed_schema(self):
        source = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py"
        ).read_text(encoding="utf-8")
        for key in (
            '"selector_mode"',
            '"candidate_rank"',
            '"score"',
            '"top1_top2_margin"',
            '"drafted_len"',
            '"base_pos"',
            '"matched_pos"',
            '"delta"',
            '"suffix_match"',
            '"predicted_accept_probability"',
            '"stop_reason"',
            '"metric_window_id"',
        ):
            self.assertIn(key, source)
        runner_source = (
            PROJECT_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("drafted_lengths=drafted_lengths", runner_source)
        self.assertIn("emitted_token_lengths=emitted_token_lengths", runner_source)
        self.assertIn("self.input_batch.num_reqs == 0", runner_source)
        self.assertIn("flush_observability_metrics", runner_source)
        self.assertIn("HSPEC_S1_PARITY_ENABLED", runner_source)
        self.assertIn("record_hspec_s1_parity_batch", runner_source)

    def test_legacy_metric_semantics_are_not_silently_overwritten(self):
        source = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_table.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"hspec/avg_accept_length": accept_len_sum / accept_times',
            source,
        )
        self.assertIn(
            '"select_avg_accept_len": _safe_ratio(accepted, verified)',
            (
                PROJECT_ROOT
                / "vllm_ascend"
                / "spec_decode"
                / "hspec_metrics.py"
            ).read_text(encoding="utf-8"),
        )
        self.assertIn("entry_abs_delta_bucket_verify_", source)
        self.assertNotIn('prefix = f"hspec/entry_abs_delta_{abs_delta}"', source)


class TestRolloutRoundFinalization(unittest.TestCase):

    def test_proposer_finalization_closes_pending_metric_window_once(self):
        proposer_path = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py"
        )
        source = proposer_path.read_text(encoding="utf-8")
        trace_events = []
        globals_for_cancel = {
            "HSPEC_S4_TRACE_ENABLED": True,
            "record_hspec_s4_trace_events": trace_events.extend,
        }
        cancel_method = _extract_method_from_source(
            source,
            str(proposer_path),
            "HSpecProposer",
            "_cancel_pending_request",
            globals_for_cancel,
        )
        finalize_method = _extract_method_from_source(
            source,
            str(proposer_path),
            "HSpecProposer",
            "finalize_rollout_round",
        )
        metrics_namespace = runpy.run_path(str(
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_metrics.py"
        ))
        tracker = metrics_namespace["HSpecSelectionMetricTracker"](
            max_draft_tokens=15
        )
        window_id = tracker.begin_window(eligible_queries=2)
        tracker.finalize_proposals(
            window_id,
            proposed_requests=2,
            drafted_tokens=4,
        )
        self.assertIsNone(tracker.record_verification(
            window_id,
            accepted_prefix_len=2,
        ))
        closed_metrics = []
        flush_calls = []
        proposer = SimpleNamespace(
            _accept_lengths={"r2": 0},
            _pending_verify_meta={
                "r2": {
                    "metric_window_id": window_id,
                    "s4_trace_query_id": "q2",
                }
            },
            _req_prompt_ids={"r2": "p2"},
            _cached_batch_req_ids=("r2",),
            _cached_batch_prompt_ids=["p2"],
            _batched_table_cache=object(),
            _selector_metric_tracker=tracker,
            _record_closed_selector_metrics=lambda value: (
                closed_metrics.append(value) if value is not None else None
            ),
            flush_observability_metrics=lambda: flush_calls.append(True),
        )
        proposer._cancel_pending_request = MethodType(cancel_method, proposer)
        proposer.finalize_rollout_round = MethodType(finalize_method, proposer)

        reason = "rollout_round_completed_with_unresolved_trace_outcome"
        self.assertEqual(proposer.finalize_rollout_round(reason), 1)
        self.assertEqual(tracker.pending_window_count, 0)
        self.assertEqual(len(closed_metrics), 1)
        self.assertEqual(closed_metrics[0]["select_proposed_requests"], 2)
        self.assertEqual(closed_metrics[0]["select_verified_requests"], 1)
        self.assertEqual(closed_metrics[0]["select_canceled_requests"], 1)
        self.assertEqual(trace_events[0]["reason"], reason)
        self.assertFalse(proposer._pending_verify_meta)
        self.assertFalse(proposer._accept_lengths)
        self.assertFalse(proposer._req_prompt_ids)
        self.assertEqual(proposer._cached_batch_req_ids, ())
        self.assertIsNone(proposer._batched_table_cache)

        self.assertEqual(proposer.finalize_rollout_round(reason), 0)
        self.assertEqual(len(trace_events), 1)
        self.assertEqual(len(closed_metrics), 1)
        self.assertEqual(len(flush_calls), 2)

    def test_model_runner_fails_closed_on_trace_mirror_orphan(self):
        runner_path = (
            PROJECT_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
        )
        source = runner_path.read_text(encoding="utf-8")
        seal_results = [0]
        method = _extract_method_from_source(
            source,
            str(runner_path),
            "NPUModelRunner",
            "hspec_finalize_rollout_round",
            {
                "HSPEC_S4_TRACE_ROUND_REASON": "round",
                "HSPEC_S4_TRACE_ORPHAN_REASON": "orphan",
                "HSPEC_S4_TRACE_ENABLED": True,
                "SpecDcodeType": SimpleNamespace(HSPEC="hspec"),
                "seal_hspec_s4_trace_pending": lambda reason: seal_results.pop(0),
            },
        )
        finalized = []
        runner = SimpleNamespace(
            _hspec_collect=True,
            drafter=SimpleNamespace(
                name="hspec",
                finalize_rollout_round=lambda reason: finalized.append(reason) or 2,
            ),
        )
        result = method(runner)
        self.assertEqual(result["canceled_proposals"], 2)
        self.assertEqual(result["trace_orphans"], 0)
        self.assertEqual(finalized, ["round"])

        method.__globals__["HSPEC_S4_TRACE_ENABLED"] = False
        method.__globals__["seal_hspec_s4_trace_pending"] = lambda reason: (
            self.fail("trace seal must remain inert when S4 tracing is disabled")
        )
        disabled_result = method(runner)
        self.assertEqual(disabled_result["trace_orphans"], 0)

        method.__globals__["HSPEC_S4_TRACE_ENABLED"] = True
        method.__globals__["seal_hspec_s4_trace_pending"] = lambda reason: 1
        with self.assertRaisesRegex(RuntimeError, "trace lifecycle diverged"):
            method(runner)

    def test_round_finalize_lifecycle_hooks_are_outside_decode_hot_path(self):
        worker_source = (
            PROJECT_ROOT / "vllm_ascend" / "worker" / "worker.py"
        ).read_text(encoding="utf-8")
        worker_tree = ast.parse(worker_source)
        worker_class = next(
            node for node in worker_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NPUWorker"
        )
        methods = {
            node.name: ast.get_source_segment(worker_source, node)
            for node in worker_class.body if isinstance(node, ast.FunctionDef)
        }
        sleep_source = methods["sleep"]
        self.assertLess(
            sleep_source.index("self.hspec_finalize_rollout_round()"),
            sleep_source.index("allocator.sleep("),
        )
        self.assertIn("HSPEC_S4_TRACE_SHUTDOWN_REASON", methods["shutdown"])
        self.assertIn("finalize_hspec_s4_trace()", methods["shutdown"])

        rollout_source = (
            PROJECT_ROOT
            / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
        ).read_text(encoding="utf-8")
        generate_pos = rollout_source.index(
            "outputs = self.inference_engine.generate("
        )
        finalize_pos = rollout_source.index(
            '"hspec_finalize_rollout_round"', generate_pos
        )
        flush_pos = rollout_source.index("hs_store: dict", generate_pos)
        self.assertLess(generate_pos, finalize_pos)
        self.assertLess(finalize_pos, flush_pos)


class TestGlobalMetricFormatting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        metrics_path = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_metrics.py"
        )
        metrics_namespace = runpy.run_path(str(metrics_path))
        method = _extract_method(
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_table.py",
            "GlobalHSpecTableGroup",
            "_format_selector_metrics",
        )
        method.__globals__["SELECTOR_ADDITIVE_METRIC_KEYS"] = metrics_namespace[
            "SELECTOR_ADDITIVE_METRIC_KEYS"
        ]
        method.__globals__["derive_selector_metrics"] = metrics_namespace[
            "derive_selector_metrics"
        ]
        cls.method = staticmethod(method)

    def test_interval_and_cumulative_metrics_keep_separate_denominators(self):
        formatted = self.method({
            "select_metric_window_id": 7,
            "select_metric_windows": 2,
            "select_eligible_queries": 8,
            "select_proposed_requests": 6,
            "select_drafted_tokens": 18,
            "select_verified_requests": 6,
            "select_first_token_accepts": 3,
            "select_accepted_tokens": 9,
            "select_zero_accept_requests": 3,
            "select_cumulative_metric_windows": 10,
            "select_cumulative_eligible_queries": 40,
            "select_cumulative_proposed_requests": 20,
            "select_cumulative_drafted_tokens": 60,
            "select_cumulative_verified_requests": 20,
            "select_cumulative_first_token_accepts": 10,
            "select_cumulative_accepted_tokens": 20,
        })
        self.assertEqual(formatted["hspec/select_metric_window_id"], 7.0)
        self.assertEqual(formatted["hspec/select_metrics_are_interval"], 1.0)
        self.assertAlmostEqual(formatted["hspec/select_proposal_coverage"], 0.75)
        self.assertAlmostEqual(formatted["hspec/select_accepted_tokens_per_query"], 9 / 8)
        self.assertAlmostEqual(
            formatted["hspec/select_cumulative_proposal_coverage"], 0.5
        )
        self.assertAlmostEqual(
            formatted["hspec/select_cumulative_accepted_tokens_per_query"], 0.5
        )
        self.assertEqual(formatted["hspec/select_rerank_changed_count"], 0.0)
        self.assertEqual(formatted["hspec/select_npu_topk_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
