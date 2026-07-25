import ast
import copy
import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _extract_method(path: Path, class_name: str, method_name: str):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if (
            not class_name
            and isinstance(node, ast.FunctionDef)
            and node.name == method_name
        ):
            function = copy.deepcopy(node)
            function.decorator_list = []
            module = ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            )
            namespace = {"Any": Any, "DataProto": object}
            exec(compile(module, str(path), "exec"), namespace)
            return namespace[method_name]
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                function = copy.deepcopy(item)
                function.decorator_list = []
                module = ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                )
                namespace = {
                    "Any": Any,
                    "Dict": Dict,
                    "List": List,
                    "Optional": Optional,
                    "_BatchedPromptTableCache": object,
                    "SamplingMetadata": object,
                    "SchedulerOutput": object,
                    "SpecDecodeMetadata": object,
                    "torch": torch,
                }
                exec(compile(module, str(path), "exec"), namespace)
                return namespace[method_name]
    raise AssertionError(f"missing {class_name}.{method_name}")


def _load_metrics_module():
    path = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_metrics.py"
    name = "hspec_s2_metrics_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestS2RequestFunnel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.method = staticmethod(
            _extract_method(
                PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py",
                "HSpecProposer",
                "_record_s2_baseline_funnel",
            )
        )

    @staticmethod
    def _observer(enabled: bool = True):
        observer = SimpleNamespace(
            _s2_baseline_audit_enabled=enabled,
            _cache_version=3,
            min_match_len=2,
            _cache={
                "p0": SimpleNamespace(n_entries=8),
                "p1": SimpleNamespace(n_entries=4),
                "p2": SimpleNamespace(n_entries=0),
            },
            _proposer_metric_deltas=defaultdict(float),
        )

        def record(key, value=1.0):
            observer._proposer_metric_deltas[key] += float(value)

        observer._record_proposer_metric = record
        return observer

    def test_funnel_is_strictly_nested_and_uses_actual_batch_cache(self):
        observer = self._observer()
        batch_cache = SimpleNamespace(batch_idx_to_row={0: 0, 1: 1})
        self.method(
            observer,
            prompt_ids=["p0", "p1", "p2", "", "missing"],
            anchor_indices=[4, -1, 7, 8, 9],
            sampled_token_ids=[[1, 2], [1, 2], [1, 2], [1, 2], [1]],
            batch_table_cache=batch_cache,
        )
        values = observer._proposer_metric_deltas
        expected = [5, 5, 4, 2, 2, 1, 1]
        keys = [
            "select_funnel_decode_requests",
            "select_funnel_active_table_requests",
            "select_funnel_prompt_id_ready_requests",
            "select_funnel_prompt_table_ready_requests",
            "select_funnel_batch_cache_ready_requests",
            "select_funnel_anchor_ready_requests",
            "select_funnel_eligible_queries",
        ]
        self.assertEqual([values[key] for key in keys], expected)
        self.assertTrue(all(left >= right for left, right in zip(expected, expected[1:])))

    def test_disabled_observer_has_no_metric_or_state_effect(self):
        observer = self._observer(enabled=False)
        self.method(
            observer,
            prompt_ids=["p0"],
            anchor_indices=[0],
            sampled_token_ids=[[1, 2]],
            batch_table_cache=SimpleNamespace(batch_idx_to_row={0: 0}),
        )
        self.assertEqual(dict(observer._proposer_metric_deltas), {})

    def test_no_active_table_stops_the_funnel_after_decode(self):
        observer = self._observer()
        observer._cache_version = -1
        self.method(
            observer,
            prompt_ids=["p0", "p1"],
            anchor_indices=[0, 1],
            sampled_token_ids=[[1, 2], [1, 2]],
            batch_table_cache=SimpleNamespace(batch_idx_to_row={0: 0, 1: 1}),
        )
        values = observer._proposer_metric_deltas
        self.assertEqual(values["select_funnel_decode_requests"], 2)
        self.assertEqual(values["select_funnel_active_table_requests"], 0)
        self.assertEqual(values["select_funnel_eligible_queries"], 0)


class TestS2FunnelMetricDerivation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metrics = _load_metrics_module()

    def test_ratios_keep_conditional_denominators(self):
        counters = {
            "select_funnel_decode_requests": 100,
            "select_funnel_active_table_requests": 80,
            "select_funnel_prompt_id_ready_requests": 80,
            "select_funnel_prompt_table_ready_requests": 72,
            "select_funnel_batch_cache_ready_requests": 68,
            "select_funnel_anchor_ready_requests": 66,
            "select_funnel_eligible_queries": 60,
        }
        derived = self.metrics.derive_selector_metrics(counters)
        self.assertAlmostEqual(derived["select_funnel_active_table_ratio"], 0.8)
        self.assertAlmostEqual(derived["select_funnel_prompt_table_ready_ratio"], 0.9)
        self.assertAlmostEqual(
            derived["select_funnel_batch_cache_ready_ratio"], 68 / 72
        )
        self.assertAlmostEqual(
            derived["select_funnel_end_to_end_eligible_ratio"], 0.6
        )
        self.assertTrue(
            all(
                self.metrics.is_selector_additive_metric(key)
                for key in counters
            )
        )

    def test_global_formatter_exposes_zeroes_and_conditional_ratios(self):
        method = _extract_method(
            PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_table.py",
            "GlobalHSpecTableGroup",
            "_format_selector_metrics",
        )
        method.__globals__["SELECTOR_ADDITIVE_METRIC_KEYS"] = (
            self.metrics.SELECTOR_ADDITIVE_METRIC_KEYS
        )
        method.__globals__["derive_selector_metrics"] = (
            self.metrics.derive_selector_metrics
        )
        formatted = method({
            "select_funnel_decode_requests": 10,
            "select_funnel_active_table_requests": 10,
            "select_funnel_prompt_id_ready_requests": 10,
            "select_funnel_prompt_table_ready_requests": 9,
            "select_funnel_batch_cache_ready_requests": 8,
            "select_funnel_anchor_ready_requests": 8,
            "select_funnel_eligible_queries": 0,
        })
        self.assertEqual(formatted["hspec/select_funnel_eligible_queries"], 0.0)
        self.assertAlmostEqual(
            formatted["hspec/select_funnel_batch_cache_ready_ratio"], 8 / 9
        )
        self.assertEqual(
            formatted["hspec/select_funnel_eligible_after_anchor_ratio"], 0.0
        )


class TestHSpecTpDraftSynchronization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        proposer_path = (
            PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"
        )
        cls.generate = staticmethod(
            _extract_method(proposer_path, "HSpecProposer", "generate_token_ids")
        )
        cls.sync = staticmethod(
            _extract_method(
                proposer_path,
                "HSpecProposer",
                "_sync_tp_draft_token_ids",
            )
        )

    def test_follower_uses_leader_result_without_running_local_selector(self):
        group = SimpleNamespace(world_size=4, rank_in_group=2)
        observer = SimpleNamespace(
            _in_generate_token_ids=False,
            _get_tp_draft_sync_group=lambda: group,
            _generate_token_ids_impl=lambda **_kwargs: self.fail(
                "TP follower must not execute the rank-local HSpec selector"
            ),
            _sync_tp_draft_token_ids=lambda size, drafts, sampled: (
                self.assertEqual(
                    (size, drafts, sampled),
                    (2, [[], []], [[1], [2]]),
                ) or [[7], [8, 9]]
            ),
        )
        result = self.generate(observer, [[1], [2]])
        self.assertEqual(result, [[7], [8, 9]])
        self.assertFalse(observer._in_generate_token_ids)

    def test_leader_payload_is_authoritative_and_normalized(self):
        group = SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            broadcast_object=lambda payload, src=0: payload,
        )
        observer = SimpleNamespace(
            max_draft_tokens=3,
            _get_tp_draft_sync_group=lambda: group,
            _tp_request_sync_keys=lambda _size: [("386", 12, 13)],
        )
        result = self.sync(observer, 1, [[1, 2]], [[9]])
        self.assertEqual(result, [[1, 2]])

    def test_request_state_mismatch_fails_fast(self):
        leader_payload = {
            "request_keys": [("386", 12, 13)],
            "sampled_token_ids": [(9,)],
            "draft_token_ids": [[1, 2]],
        }
        group = SimpleNamespace(
            world_size=4,
            rank_in_group=2,
            broadcast_object=lambda _payload, src=0: leader_payload,
        )
        observer = SimpleNamespace(
            max_draft_tokens=3,
            _get_tp_draft_sync_group=lambda: group,
            _tp_request_sync_keys=lambda _size: [("469", 12, 13)],
        )
        with self.assertRaisesRegex(RuntimeError, "scheduler divergence"):
            self.sync(observer, 1, [[]], [[9]])

    def test_sampled_token_mismatch_fails_fast(self):
        leader_payload = {
            "request_keys": [("386", 12, 13)],
            "sampled_token_ids": [(9,)],
            "draft_token_ids": [[1, 2]],
        }
        group = SimpleNamespace(
            world_size=4,
            rank_in_group=3,
            broadcast_object=lambda _payload, src=0: leader_payload,
        )
        observer = SimpleNamespace(
            max_draft_tokens=3,
            _get_tp_draft_sync_group=lambda: group,
            _tp_request_sync_keys=lambda _size: [("386", 12, 13)],
        )
        with self.assertRaisesRegex(RuntimeError, "sampled_equal=False"):
            self.sync(observer, 1, [[]], [[10]])

    def test_30b_scripts_enable_log_and_forward_tp_draft_sync(self):
        for script_name in (
            "train_grpo_qwen3_30b_hspec.sh",
            "train_grpo_qwen3_30b_hspec_gsm8k.sh",
        ):
            source = (PROJECT_ROOT / "scripts" / script_name).read_text(
                encoding="utf-8"
            )
            self.assertIn(
                'export HSPEC_TP_DRAFT_SYNC="${HSPEC_TP_DRAFT_SYNC:-1}"',
                source,
            )
            self.assertIn(
                'echo "hspec_tp_draft_sync=${HSPEC_TP_DRAFT_SYNC}"',
                source,
            )
            self.assertIn(
                "runtime_env.env_vars.HSPEC_TP_DRAFT_SYNC=",
                source,
            )


class TestRolloutThroughputPrimitives(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.method = staticmethod(
            _extract_method(
                PROJECT_ROOT / "verl/trainer/ppo/metric_utils.py",
                "",
                "compute_timing_metrics",
            )
        )

    def test_generated_tokens_and_seconds_are_directly_aggregatable(self):
        self.method.__globals__["torch"] = torch
        self.method.__globals__["_compute_response_info"] = lambda _batch: {
            "prompt_length": torch.tensor([3, 4]),
            "response_length": torch.tensor([5, 7]),
        }
        result = self.method(object(), {"gen": 2.0, "step": 4.0})
        self.assertEqual(result["rollout/generated_tokens"], 12.0)
        self.assertEqual(result["timing_s/gen"], 2.0)
        self.assertEqual(result["rollout/generated_tokens_per_second"], 6.0)

    def test_zero_generation_time_does_not_emit_nan_or_inf(self):
        self.method.__globals__["torch"] = torch
        self.method.__globals__["_compute_response_info"] = lambda _batch: {
            "prompt_length": torch.tensor([3]),
            "response_length": torch.tensor([1]),
        }
        result = self.method(object(), {"gen": 0.0})
        self.assertEqual(result["rollout/generated_tokens_per_second"], 0.0)


if __name__ == "__main__":
    unittest.main()
