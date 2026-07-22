import ast
import copy
import inspect
import runpy
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
S0_FROZEN_HEAD = "8d50448b1f079d1fb8b72e7af908c8b231879cee"
S0_PROPOSER_PATH = (
    "llm_rl/qwen3/vllm_ascend/spec_decode/hspec_proposer.py"
)


def _extract_method_from_source(
    source: str, filename: str, class_name: str, method_name: str
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
                    exec(compile(module, filename, "exec"), namespace)
                    return namespace[method_name]
    raise AssertionError(f"Could not find {class_name}.{method_name}")


def _extract_method(path: Path, class_name: str, method_name: str):
    return _extract_method_from_source(
        path.read_text(encoding="utf-8"), str(path), class_name, method_name
    )


class TestHardmaxEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.method = staticmethod(_extract_method(
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py",
            "HSpecProposer",
            "_match_projected_batch",
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

        values, indices = self.method(object(), z, keys, invalid)
        self.assertTrue(torch.equal(values, expected_values))
        self.assertTrue(torch.equal(indices, expected_indices))

        observed_values, observed_indices, observed_margins = self.method(
            object(), z, keys, invalid, observe_margin=True
        )
        self.assertTrue(torch.equal(observed_values, expected_values))
        self.assertTrue(torch.equal(observed_indices, expected_indices))
        assert observed_margins is not None
        self.assertTrue(torch.isnan(observed_margins[1]))
        self.assertAlmostEqual(float(observed_margins[0]), 0.05, places=6)

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
        patch0_scores, patch0_indices = self.method(
            object(), projected, keys, invalid
        )
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

    def test_topk_one_contract_is_the_default_signature(self):
        signature = inspect.signature(self.method)
        self.assertEqual(signature.parameters["observe_margin"].default, False)
        source = (
            PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_proposer.py"
        ).read_text(encoding="utf-8")
        self.assertIn('os.environ.get("HSPEC_SELECT_MODE", "hardmax")', source)
        self.assertIn('_get_env_int("HSPEC_SELECT_TOPK", 1, 1)', source)
        self.assertIn("best_sims, best_idxs = sims.max(dim=1)", source)
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
