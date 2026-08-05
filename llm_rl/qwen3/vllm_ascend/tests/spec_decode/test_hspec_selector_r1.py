import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SELECTOR_PATH = (
    PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_selector_r1.py"
)
PROPOSER_PATH = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"


def _load_selector():
    # Numba's on-disk cache serializes the defining module name. Use the
    # production import name so a unit test cannot poison worker warmup with
    # an unimportable temporary module reference.
    name = "vllm_ascend.spec_decode.hspec_selector_r1"
    spec = importlib.util.spec_from_file_location(name, SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selector = _load_selector()


def _extract_matcher():
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HSpecProposer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_match_projected_batch":
                    item.decorator_list = []
                    module = ast.fix_missing_locations(
                        ast.Module(body=[item], type_ignores=[])
                    )
                    namespace = {
                        "torch": torch,
                        "Optional": Optional,
                        "_HSpecMatchResult": SimpleNamespace,
                    }
                    exec(compile(module, str(PROPOSER_PATH), "exec"), namespace)
                    return namespace[item.name]
    raise AssertionError("missing _match_projected_batch")


MATCHER = _extract_matcher()


def _extract_prefix_budget_allocator():
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_prefix_fair_budget_caps"
    )
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"List": list}
    exec(compile(module, str(PROPOSER_PATH), "exec"), namespace)
    return namespace[node.name]


PREFIX_BUDGET_ALLOCATOR = _extract_prefix_budget_allocator()


def _extract_payload_copy():
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HSpecProposer":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "_copy_r1_selection_payload_to_host"
                ):
                    item.decorator_list = []
                    module = ast.fix_missing_locations(
                        ast.Module(body=[item], type_ignores=[])
                    )
                    namespace = {
                        "np": np,
                        "torch": torch,
                        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
                    }
                    exec(compile(module, str(PROPOSER_PATH), "exec"), namespace)
                    return namespace[item.name]
    raise AssertionError("missing _copy_r1_selection_payload_to_host")


COPY_PAYLOAD = _extract_payload_copy()


def _common_arrays():
    token_buffer = np.asarray(
        [9, 9, 3, 10, 1, 2, 3, 11], dtype=np.int32
    )
    return {
        "entry_rollout_idx": np.asarray([0, 1], dtype=np.int32),
        "entry_offset": np.asarray([3, 3], dtype=np.int32),
        "token_buffer": token_buffer,
        "rollout_offsets": np.asarray([0, 4], dtype=np.int64),
        "rollout_lens": np.asarray([4, 4], dtype=np.int32),
        "current_tokens": np.asarray([1, 2, 3], dtype=np.int32),
    }


def _rerank(fn, scores, indices, arrays, **overrides):
    kwargs = {
        "base_pos": int(arrays["current_tokens"].shape[0]) - 1,
        "n_entries": 2,
        "relative_radius": 0.0001,
        "suffix_cap": 8,
        "relative_weight": 7.719409849724556,
        "suffix_weight": 0.6382322349890022,
        "position_mode": 0,
        "position_exact_weight": 0.0,
        "position_log_weight": 0.0,
        "utility_threshold": -1.0e30,
    }
    kwargs.update(overrides)
    return fn(
        np.ascontiguousarray(scores, dtype=np.float32),
        np.ascontiguousarray(indices, dtype=np.int64),
        arrays["entry_rollout_idx"],
        arrays["entry_offset"],
        arrays["token_buffer"],
        arrays["rollout_offsets"],
        arrays["rollout_lens"],
        arrays["current_tokens"],
        kwargs["base_pos"],
        kwargs["n_entries"],
        kwargs["relative_radius"],
        kwargs["suffix_cap"],
        kwargs["relative_weight"],
        kwargs["suffix_weight"],
        kwargs["position_mode"],
        kwargs["position_exact_weight"],
        kwargs["position_log_weight"],
        kwargs["utility_threshold"],
    )


class TestR1Configuration(unittest.TestCase):
    def test_default_is_exact_hardmax_and_frozen_r1_is_explicit(self):
        default = selector.HSpecR1Config.from_environment({})
        self.assertEqual((default.mode, default.topk, default.sim_mode), (
            "hardmax", 1, "raw"
        ))
        r1 = selector.HSpecR1Config.from_environment({
            "HSPEC_SELECT_MODE": "topk_position",
            "HSPEC_SELECT_TOPK": "8",
            "HSPEC_SELECT_SIM_MODE": "cosine",
            "HSPEC_SELECT_POSITION_MODE": "none",
        })
        self.assertTrue(r1.executes_topk)
        self.assertEqual(r1.relative_radius, 0.0001)
        self.assertEqual(r1.suffix_cap, 8)

    def test_invalid_config_fails_closed_without_raising(self):
        config = selector.HSpecR1Config.from_environment({
            "HSPEC_SELECT_MODE": "topk_position",
            "HSPEC_SELECT_TOPK": "1",
            "HSPEC_SELECT_SIM_MODE": "cosine",
        })
        self.assertEqual(config.mode, "hardmax")
        self.assertIsNotNone(config.fallback_reason)
        uncalibrated = selector.HSpecR1Config.from_environment({
            "HSPEC_SELECT_MODE": "topk_position",
            "HSPEC_SELECT_TOPK": "8",
            "HSPEC_SELECT_SIM_MODE": "cosine",
            "HSPEC_SELECT_POSITION_MODE": "piecewise",
        })
        self.assertEqual(uncalibrated.mode, "hardmax")

    def test_shadow_computes_but_never_executes_topk(self):
        config = selector.HSpecR1Config.from_environment({
            "HSPEC_SELECT_MODE": "topk_position",
            "HSPEC_SELECT_TOPK": "8",
            "HSPEC_SELECT_SIM_MODE": "cosine",
            "HSPEC_SELECT_SHADOW": "1",
        })
        self.assertTrue(config.computes_topk)
        self.assertFalse(config.executes_topk)


class TestR1Matcher(unittest.TestCase):
    def test_topk_one_preserves_raw_max_and_cosine_filters_padding(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        keys = torch.tensor([
            [[2.0, 0.0], [1.0, 1.0], [99.0, 99.0]],
            [[1.0, 0.0], [0.0, 3.0], [9.0, 9.0]],
        ])
        invalid = torch.tensor([
            [False, False, True],
            [False, False, True],
        ])
        raw = MATCHER(object(), z, keys, invalid)
        expected = torch.bmm(keys, z.unsqueeze(-1)).squeeze(-1).masked_fill(
            invalid, torch.finfo(torch.float32).min
        )
        expected_values, expected_indices = expected.max(dim=1)
        self.assertTrue(torch.equal(raw.raw_top1_scores, expected_values))
        self.assertTrue(torch.equal(raw.raw_top1_indices, expected_indices))

        norms = torch.linalg.vector_norm(keys, dim=2).clamp_min(1.0e-12)
        cosine = MATCHER(
            object(), z, keys, invalid, topk=2, sim_mode="cosine",
            key_norm_batch=norms,
        )
        self.assertEqual(tuple(cosine.candidate_scores.shape), (2, 2))
        self.assertTrue((cosine.candidate_indices < 2).all())
        self.assertTrue(torch.equal(cosine.raw_top1_scores, expected_values))


class TestR1Reranker(unittest.TestCase):
    def test_suffix_can_replace_cosine_top1(self):
        arrays = _common_arrays()
        result = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 0.99995],
            [0, 1],
            arrays,
        )
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 3)
        self.assertEqual(result[4], 1)

    def test_invalid_padding_and_zero_remaining_are_never_selected(self):
        arrays = _common_arrays()
        invalid = _rerank(
            selector.rerank_one_prompt_python,
            [1.1, 1.0],
            [99, 1],
            arrays,
        )
        self.assertEqual(invalid[0], 1)
        arrays["entry_offset"] = np.asarray([4, 4], dtype=np.int32)
        empty = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 0.99999],
            [0, 1],
            arrays,
        )
        self.assertEqual(empty[0], -1)

    def test_position_interface_and_utility_abstain(self):
        arrays = _common_arrays()
        arrays["entry_offset"] = np.asarray([2, 3], dtype=np.int32)
        exact = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 1.0],
            [0, 1],
            arrays,
            suffix_weight=0.0,
            position_mode=1,
            position_exact_weight=2.0,
        )
        self.assertEqual(exact[0], 1)
        earlier_base = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 1.0],
            [0, 1],
            arrays,
            base_pos=1,
            suffix_weight=0.0,
            position_mode=1,
            position_exact_weight=2.0,
        )
        self.assertEqual(earlier_base[0], 0)
        log_nearer = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 1.0],
            [0, 1],
            arrays,
            base_pos=0,
            suffix_weight=0.0,
            position_mode=2,
            position_log_weight=1.0,
        )
        self.assertEqual(log_nearer[0], 0)
        abstain = _rerank(
            selector.rerank_one_prompt_python,
            [1.0, 0.99999],
            [0, 1],
            arrays,
            suffix_weight=0.0,
            utility_threshold=1.0,
        )
        self.assertEqual(abstain[0], -1)

    @unittest.skipUnless(
        selector.HSPEC_R1_NUMBA_AVAILABLE, "Numba is unavailable"
    )
    def test_numba_matches_reference_across_boundaries(self):
        arrays = _common_arrays()
        for scores, indices, cap in (
            ([1.0, 0.99995], [0, 1], 8),
            ([1.0, 0.7], [0, 1], 1),
            ([1.0, 0.99999], [99, 1], 4),
        ):
            expected = _rerank(
                selector.rerank_one_prompt_python,
                scores,
                indices,
                arrays,
                suffix_cap=cap,
            )
            actual = _rerank(
                selector.rerank_one_prompt_numba,
                scores,
                indices,
                arrays,
                suffix_cap=cap,
            )
            self.assertEqual(actual[:2], expected[:2])
            self.assertAlmostEqual(actual[2], expected[2], places=6)
            self.assertEqual(actual[3:], expected[3:])

    def test_hot_path_module_has_no_rpc_or_storage_dependency(self):
        source = SELECTOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse({"ray", "mmap", "json"} & imported)
        self.assertNotIn("materialize", source)
        proposer_source = PROPOSER_PATH.read_text(encoding="utf-8")
        self.assertIn("current_tokens[-tail_cap:]", proposer_source)


class TestR1IntegrationContracts(unittest.TestCase):
    def test_prefix_fair_batch_budget_is_bounded_and_depth_ordered(self):
        self.assertEqual(
            PREFIX_BUDGET_ALLOCATOR([15, 15, 2, 0], 0),
            [15, 15, 2, 0],
        )
        self.assertEqual(
            PREFIX_BUDGET_ALLOCATOR([15, 15, 2, 0], 8),
            [3, 3, 2, 0],
        )
        caps = PREFIX_BUDGET_ALLOCATOR([14] * 64, 384)
        self.assertEqual(caps, [6] * 64)
        self.assertEqual(sum(caps), 384)
        self.assertTrue(all(cap <= length for cap, length in zip(caps, [14] * 64)))

    def test_prefix_budget_preserves_short_values_and_redistributes_depth(self):
        caps = PREFIX_BUDGET_ALLOCATOR([1, 3, 10], 8)
        self.assertEqual(caps, [1, 3, 4])
        self.assertEqual(sum(caps), 8)

    def test_shadow_cannot_mutate_raw_entry_or_admission_decision(self):
        source = PROPOSER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        generate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_generate_token_ids_impl"
        )
        parents = {
            child: parent
            for parent in ast.walk(generate)
            for child in ast.iter_child_nodes(parent)
        }

        def guarded_by_not_shadow(node):
            current = node
            while current in parents:
                current = parents[current]
                if isinstance(current, ast.If) and ast.unparse(current.test) == (
                    "not self._r1_config.shadow"
                ):
                    return True
            return False

        decision_mutations = []
        admission_mutations = []
        for node in ast.walk(generate):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if isinstance(target.value, ast.Name):
                    if target.value.id == "decision_idxs_cpu":
                        decision_mutations.append(node)
                    elif target.value.id == "selected_valid":
                        admission_mutations.append(node)
        self.assertTrue(decision_mutations)
        self.assertTrue(admission_mutations)
        self.assertTrue(all(map(guarded_by_not_shadow, decision_mutations)))
        self.assertTrue(all(map(guarded_by_not_shadow, admission_mutations)))
        generate_source = ast.get_source_segment(source, generate)
        self.assertIn(
            "decision_idxs_cpu = np.asarray(idxs_cpu, dtype=np.int64).copy()",
            generate_source,
        )
        self.assertIn(
            "matched_entry_idx = int(decision_idxs_cpu[j])", generate_source
        )

    def test_candidates_and_raw_gate_share_one_pinned_sync(self):
        class Event:
            def __init__(self):
                self.record_count = 0
                self.sync_count = 0

            def record(self):
                self.record_count += 1

            def synchronize(self):
                self.sync_count += 1

        event = Event()
        proposer = SimpleNamespace(
            _r1_d2h_strategy_runtime="pinned_two_async",
            _r1_host_scores=torch.empty((4, 4), dtype=torch.float32),
            _r1_host_indices=torch.empty((4, 4), dtype=torch.int64),
            _r1_copy_event=event,
            _record_proposer_metric=lambda *args: None,
        )
        candidate_scores = torch.tensor([[1.0, 0.9, 0.8], [0.7, 0.6, 0.5]])
        candidate_indices = torch.tensor([[2, 1, 0], [0, 2, 1]])
        raw_scores = torch.tensor([4.0, 3.0])
        raw_indices = torch.tensor([1, 2])
        scores, indices, raw_s, raw_i = COPY_PAYLOAD(
            proposer,
            candidate_scores,
            candidate_indices,
            raw_scores,
            raw_indices,
        )
        np.testing.assert_array_equal(scores, candidate_scores.numpy())
        np.testing.assert_array_equal(indices, candidate_indices.numpy())
        np.testing.assert_array_equal(raw_s, raw_scores.numpy())
        np.testing.assert_array_equal(raw_i, raw_indices.numpy())
        self.assertEqual((event.record_count, event.sync_count), (1, 1))

    def test_all_launchers_default_off_and_forward_r1_configuration(self):
        scripts = (
            PROJECT_ROOT / "scripts/train_grpo_qwen2.5_1.5b_hspec.sh",
            PROJECT_ROOT / "scripts/train_grpo_qwen3_30b_hspec.sh",
            PROJECT_ROOT / "scripts/train_grpo_qwen3_30b_hspec_gsm8k.sh",
        )
        required = (
            "MODE",
            "TOPK",
            "SIM_MODE",
            "RELATIVE_RADIUS",
            "SUFFIX_CAP",
            "POSITION_MODE",
            "SHADOW",
            "SAMPLE_LOG_RATE",
        )
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            self.assertIn(
                'export HSPEC_SELECT_MODE="${HSPEC_SELECT_MODE:-hardmax}"',
                source,
            )
            self.assertIn(
                'export HSPEC_SELECT_TOPK="${HSPEC_SELECT_TOPK:-1}"',
                source,
            )
            self.assertIn(
                'export HSPEC_SELECT_SIM_MODE="${HSPEC_SELECT_SIM_MODE:-raw}"',
                source,
            )
            for suffix in required:
                self.assertIn(
                    f"runtime_env.env_vars.HSPEC_SELECT_{suffix}=", source
                )
            self.assertIn(
                "runtime_env.env_vars.HSPEC_MAX_DRAFT_TOKENS_PER_BATCH=",
                source,
            )


if __name__ == "__main__":
    unittest.main()
