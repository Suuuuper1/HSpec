import hashlib
import json
import math
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numba.typed import List as NumbaList

from vllm_ascend.spec_decode.hspec_metrics import (
    hspec_utility_score_histogram_key,
    is_selector_additive_metric,
)
from vllm_ascend.spec_decode.hspec_selector_survival import (
    HSPEC_SURVIVAL_ACTIONS,
    HSpecSurvivalConfig,
    extract_utility_features_python,
    score_utility_batch_numba,
    score_utility_one_prompt_numba,
    score_utility_one_prompt_python,
)
from vllm_ascend.spec_decode.hspec_s13_shadow import HSpecS13ShadowRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL = (
    PROJECT_ROOT
    / "HSpec_research_doc/HSpec_draft_delect_optim/"
    "s12_to_s13_transition/candidate/transition_candidate.json"
)
ENTRY_GATE = (
    PROJECT_ROOT
    / "HSpec_research_doc/HSpec_draft_delect_optim/"
    "s13_patch3a_utility/artifacts/s13_entry_gate.json"
)
PROPOSER = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"


def _load_utility_action_window():
    tree = ast.parse(PROPOSER.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_utility_action_window"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(PROPOSER), "exec"), namespace)
    return namespace[node.name]


UTILITY_ACTION_WINDOW = _load_utility_action_window()


def _load_utility_r1_work_masks():
    tree = ast.parse(PROPOSER.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_utility_r1_work_masks"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(PROPOSER), "exec"), namespace)
    return namespace[node.name]


UTILITY_R1_WORK_MASKS = _load_utility_r1_work_masks()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(*, shadow: str = "1", allow_execute: str = "0") -> dict[str, str]:
    return {
        "HSPEC_SELECT_MODE": "topk_utility",
        "HSPEC_SELECT_SHADOW": shadow,
        "HSPEC_SELECT_ALLOW_EXECUTE": allow_execute,
        "HSPEC_SELECT_MODEL_PATH": str(MODEL),
        "HSPEC_SELECT_MODEL_SHA256": _sha256(MODEL),
        "HSPEC_SELECT_MODEL_VERSION": "s12-transition-fixed-theta-bias-v1",
        "HSPEC_SELECT_PROMOTION_GATE_PATH": str(ENTRY_GATE),
        "HSPEC_SELECT_PROMOTION_GATE_SHA256": _sha256(ENTRY_GATE),
    }


def _table_arrays():
    rollouts = []
    histories = (
        [10, 11, 12, 13],
        [20, 11, 12, 13],
        [20, 21, 12, 13],
        [20, 21, 22, 13],
        [30, 31, 32, 33],
        [40, 41, 42, 43],
        [50, 51, 52, 53],
        [60, 61, 62, 63],
    )
    futures = (
        [100, 101, 102, 103, 104],
        [100, 101, 102, 199, 200],
        [100, 101, 150, 151, 152],
        [100, 141, 142, 143, 144],
        [200, 201, 202, 203, 204],
        [200, 201, 250, 251, 252],
        [300, 301, 302, 303, 304],
        [400, 401, 402, 403, 404],
    )
    offsets = []
    cursor = 0
    for history, future in zip(histories, futures):
        offsets.append(cursor)
        row = np.asarray(history + future, dtype=np.int32)
        rollouts.append(row)
        cursor += len(row)
    return {
        "scores": np.asarray(
            [0.99, 0.98, 0.97, 0.96, 0.94, 0.92, 0.90, 0.88],
            dtype=np.float32,
        ),
        "indices": np.arange(8, dtype=np.int64),
        "entry_rollout_idx": np.arange(8, dtype=np.int32),
        "entry_offset": np.full((8,), 4, dtype=np.int32),
        "token_buffer": np.concatenate(rollouts),
        "rollout_offsets": np.asarray(offsets, dtype=np.int64),
        "rollout_lens": np.full((8,), 9, dtype=np.int32),
        "key_norms": np.linspace(80.0, 120.0, 8, dtype=np.float32),
        "current": np.asarray([10, 11, 12, 13], dtype=np.int32),
    }


def _score(fn, arrays, model, threshold=None, query_norm=90.0):
    return fn(
        arrays["scores"],
        arrays["indices"],
        arrays["entry_rollout_idx"],
        arrays["entry_offset"],
        arrays["token_buffer"],
        arrays["rollout_offsets"],
        arrays["rollout_lens"],
        arrays["key_norms"],
        arrays["current"],
        query_norm,
        3,
        4,
        8,
        np.asarray(model["feature_mean"], dtype=np.float64),
        np.asarray(model["feature_scale"], dtype=np.float64),
        np.asarray(model["theta"], dtype=np.float64),
        np.asarray(model["depth_bias"], dtype=np.float64),
        np.asarray(model["length_actions"], dtype=np.int16),
        np.asarray(
            [model["costs_ms"][str(action)] for action in HSPEC_SURVIVAL_ACTIONS],
            dtype=np.float64,
        ),
        float(model["temperature"]),
        float(model["utility_threshold"] if threshold is None else threshold),
    )


class TestSurvivalConfiguration(unittest.TestCase):
    def test_validated_candidate_is_shadow_only_by_default(self):
        config = HSpecSurvivalConfig.from_environment(_environment())
        self.assertTrue(config.enabled, config.fallback_reason)
        self.assertTrue(config.shadow)
        self.assertFalse(config.executes_utility)
        self.assertEqual(config.model_sha256, _sha256(MODEL))

    def test_execute_has_a_second_explicit_interlock(self):
        blocked = HSpecSurvivalConfig.from_environment(
            _environment(shadow="0", allow_execute="0")
        )
        self.assertFalse(blocked.enabled)
        allowed = HSpecSurvivalConfig.from_environment(
            _environment(shadow="0", allow_execute="1")
        )
        self.assertFalse(allowed.enabled)
        with tempfile.TemporaryDirectory() as temporary:
            gate_path = Path(temporary) / "shadow_gate.json"
            gate_path.write_text(json.dumps({
                "schema_version": "hspec.s13.functional-shadow-gate.v2",
                "status": "PASS",
                "decision": "READY_FOR_1P5B_EXECUTION_SMOKE",
                "checks": {"shadow_parity": True},
                "model_sha256": _sha256(MODEL),
                "entry_gate_sha256": _sha256(ENTRY_GATE),
            }), encoding="utf-8")
            environment = _environment(shadow="0", allow_execute="1")
            environment.update({
                "HSPEC_SELECT_EXECUTION_LEVEL": "functional",
                "HSPEC_SELECT_EXECUTION_GATE_PATH": str(gate_path),
                "HSPEC_SELECT_EXECUTION_GATE_SHA256": _sha256(gate_path),
            })
            allowed = HSpecSurvivalConfig.from_environment(environment)
            self.assertTrue(allowed.executes_utility, allowed.fallback_reason)
            self.assertEqual(allowed.execution_level, "functional")

    def test_hash_or_gate_failure_falls_back_without_raising(self):
        bad_hash = _environment()
        bad_hash["HSPEC_SELECT_MODEL_SHA256"] = "0" * 64
        self.assertFalse(HSpecSurvivalConfig.from_environment(bad_hash).enabled)
        with tempfile.TemporaryDirectory() as temporary:
            gate = json.loads(ENTRY_GATE.read_text(encoding="utf-8"))
            gate["checks"]["coverage"] = False
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            bad_gate = _environment()
            bad_gate["HSPEC_SELECT_PROMOTION_GATE_PATH"] = str(path)
            bad_gate["HSPEC_SELECT_PROMOTION_GATE_SHA256"] = _sha256(path)
            self.assertFalse(
                HSpecSurvivalConfig.from_environment(bad_gate).enabled
            )

    def test_performance_execute_requires_target_shadow_v2_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution_gate.json"
            environment = _environment(shadow="0", allow_execute="1")
            environment["HSPEC_SELECT_EXECUTION_LEVEL"] = "performance"
            environment["HSPEC_SELECT_EXECUTION_GATE_PATH"] = str(path)

            legacy = {
                "schema_version": "hspec.s13.functional-gate.v1",
                "status": "PASS",
                "decision": "READY_FOR_30B_ONLINE_AB",
                "checks": {"functional": True},
                "model_sha256": _sha256(MODEL),
                "entry_gate_sha256": _sha256(ENTRY_GATE),
            }
            path.write_text(json.dumps(legacy), encoding="utf-8")
            environment["HSPEC_SELECT_EXECUTION_GATE_SHA256"] = _sha256(path)
            rejected = HSpecSurvivalConfig.from_environment(environment)
            self.assertFalse(rejected.enabled)

            target = dict(legacy)
            target["schema_version"] = "hspec.s13.target-shadow-gate.v2"
            path.write_text(json.dumps(target), encoding="utf-8")
            environment["HSPEC_SELECT_EXECUTION_GATE_SHA256"] = _sha256(path)
            allowed = HSpecSurvivalConfig.from_environment(environment)
            self.assertTrue(allowed.executes_utility, allowed.fallback_reason)
            self.assertEqual(allowed.execution_level, "performance")


class TestSurvivalFeaturesAndPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(MODEL.read_text(encoding="utf-8"))

    def test_feature_block_matches_frozen_offline_definitions(self):
        arrays = _table_arrays()
        features, suffix, abs_delta, remaining, status = (
            extract_utility_features_python(
                arrays["scores"],
                arrays["indices"],
                arrays["entry_rollout_idx"],
                arrays["entry_offset"],
                arrays["token_buffer"],
                arrays["rollout_offsets"],
                arrays["rollout_lens"],
                arrays["key_norms"],
                arrays["current"],
                90.0,
                3,
                4,
                8,
            )
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(suffix, [4, 3, 2, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(abs_delta, np.zeros(8, dtype=np.int32))
        np.testing.assert_array_equal(remaining, np.full(8, 5, dtype=np.int32))
        np.testing.assert_allclose(features[:, 1], suffix / 8.0)
        np.testing.assert_allclose(features[:, 3], 0.01, atol=2e-8)
        np.testing.assert_allclose(features[:, 7], 1.0)
        np.testing.assert_allclose(features[:, 10], np.arange(8) / 7.0)
        np.testing.assert_allclose(features[:, 12], 5.0 / 15.0)
        np.testing.assert_allclose(features[:, 14], [0.5] * 4 + [0.25, 0.25, 0.125, 0.125])
        np.testing.assert_allclose(features[:, 15], [0.375, 0.375, 0.375, 0.125, 0.25, 0.25, 0.125, 0.125])
        np.testing.assert_allclose(features[:, 16], 0.125)
        np.testing.assert_allclose(features[:, 17], 0.125)

    def test_numba_policy_matches_independent_numpy_argmax(self):
        arrays = _table_arrays()
        python_result = _score(
            score_utility_one_prompt_python, arrays, self.model
        )
        numba_result = _score(score_utility_one_prompt_numba, arrays, self.model)
        self.assertEqual(python_result[0:2], numba_result[0:2])
        np.testing.assert_allclose(python_result[2:4], numba_result[2:4], rtol=0, atol=1e-12)
        self.assertEqual(python_result[7], 0)

        features = extract_utility_features_python(
            arrays["scores"], arrays["indices"], arrays["entry_rollout_idx"],
            arrays["entry_offset"], arrays["token_buffer"],
            arrays["rollout_offsets"], arrays["rollout_lens"],
            arrays["key_norms"], arrays["current"], 90.0, 3, 4, 8,
        )[0]
        x = np.clip(
            (features - np.asarray(self.model["feature_mean"]))
            / np.asarray(self.model["feature_scale"]),
            -12.0,
            12.0,
        )
        logits = x @ np.asarray(self.model["theta"])
        logits = logits[:, None] + np.asarray(self.model["depth_bias"])[None, :]
        probability = 1.0 / (1.0 + np.exp(-logits))
        remaining = np.full((8,), 5)
        utilities = []
        base_cost = self.model["costs_ms"]["0"]
        for slot in range(8):
            for action in self.model["length_actions"][1:]:
                expected = probability[slot, : min(action, remaining[slot])].sum()
                penalty = (
                    self.model["costs_ms"][str(action)] - base_cost
                ) / base_cost
                utilities.append((expected - penalty, slot, action))
        expected_utility, expected_slot, expected_action = max(
            utilities, key=lambda row: row[0]
        )
        self.assertEqual(python_result[0], expected_slot)
        self.assertEqual(python_result[1], expected_action)
        self.assertAlmostEqual(python_result[2], expected_utility, places=12)
        self.assertLessEqual(python_result[1], 15)

    def test_batch_kernel_matches_individual_numba_decisions(self):
        arrays = _table_arrays()
        rows = 3

        def typed(value):
            result = NumbaList()
            result.append(value)
            result.append(value)
            return result

        current_tails = np.zeros((rows, 8), dtype=np.int32)
        current_lens = np.asarray([4, 4, 3], dtype=np.int16)
        current_tails[0, -4:] = arrays["current"]
        current_tails[1, -4:] = [20, 21, 12, 13]
        current_tails[2, -3:] = [11, 12, 13]
        scores = np.repeat(arrays["scores"][None, :], rows, axis=0)
        indices = np.repeat(arrays["indices"][None, :], rows, axis=0)
        table_rows = np.asarray([0, 1, 0], dtype=np.int32)
        query_norms = np.asarray([90.0, 91.0, 89.0], dtype=np.float32)
        base_positions = np.asarray([3, 3, 3], dtype=np.int32)
        decoded_lens = np.asarray([4, 4, 3], dtype=np.int32)
        model = self.model
        model_args = (
            np.asarray(model["feature_mean"], dtype=np.float64),
            np.asarray(model["feature_scale"], dtype=np.float64),
            np.asarray(model["theta"], dtype=np.float64),
            np.asarray(model["depth_bias"], dtype=np.float64),
            np.asarray(model["length_actions"], dtype=np.int16),
            np.asarray(
                [model["costs_ms"][str(action)] for action in HSPEC_SURVIVAL_ACTIONS],
                dtype=np.float64,
            ),
            float(model["temperature"]),
            float(model["utility_threshold"]),
        )
        batch = score_utility_batch_numba(
            scores,
            indices,
            table_rows,
            typed(arrays["entry_rollout_idx"]),
            typed(arrays["entry_offset"]),
            typed(arrays["token_buffer"]),
            typed(arrays["rollout_offsets"]),
            typed(arrays["rollout_lens"]),
            typed(arrays["key_norms"]),
            current_tails,
            current_lens,
            query_norms,
            base_positions,
            decoded_lens,
            *model_args,
        )
        for row in range(rows):
            tail_len = int(current_lens[row])
            expected = score_utility_one_prompt_numba(
                scores[row],
                indices[row],
                arrays["entry_rollout_idx"],
                arrays["entry_offset"],
                arrays["token_buffer"],
                arrays["rollout_offsets"],
                arrays["rollout_lens"],
                arrays["key_norms"],
                current_tails[row, -tail_len:],
                float(query_norms[row]),
                int(base_positions[row]),
                int(decoded_lens[row]),
                8,
                *model_args,
            )
            actual = tuple(values[row].item() for values in batch)
            self.assertEqual(actual[0:2], expected[0:2])
            np.testing.assert_allclose(actual[2:4], expected[2:4], rtol=0, atol=1e-12)
            self.assertEqual(actual[4:], expected[4:])

    def test_abstain_and_out_of_contract_width_are_explicit(self):
        arrays = _table_arrays()
        abstain = _score(
            score_utility_one_prompt_python, arrays, self.model, threshold=1e6
        )
        self.assertEqual((abstain[0], abstain[1], abstain[7]), (-1, 0, 0))
        short = dict(arrays)
        short["scores"] = arrays["scores"][:7]
        short["indices"] = arrays["indices"][:7]
        fallback = _score(
            score_utility_one_prompt_python, short, self.model
        )
        self.assertEqual(fallback[0], -2)
        self.assertNotEqual(fallback[7], 0)

    def test_nonfinite_norms_are_row_fallbacks_not_abstentions(self):
        arrays = _table_arrays()
        invalid_query = _score(
            score_utility_one_prompt_python,
            arrays,
            self.model,
            query_norm=float("nan"),
        )
        self.assertEqual(invalid_query[0], -2)
        self.assertEqual(invalid_query[7], 5)

        invalid_key = dict(arrays)
        invalid_key["key_norms"] = arrays["key_norms"].copy()
        invalid_key["key_norms"][3] = np.inf
        python_result = _score(
            score_utility_one_prompt_python, invalid_key, self.model
        )
        numba_result = _score(
            score_utility_one_prompt_numba, invalid_key, self.model
        )
        self.assertEqual(python_result[0], -2)
        self.assertEqual(python_result[7], 5)
        self.assertEqual(python_result, numba_result)

    def test_depth_bias_and_survival_are_monotone(self):
        bias = np.asarray(self.model["depth_bias"])
        self.assertTrue(np.all(bias[1:] <= bias[:-1]))
        probability = 1.0 / (1.0 + np.exp(-(0.25 + bias)))
        self.assertTrue(np.all(probability[1:] <= probability[:-1]))

    def test_length_action_never_overrides_row_fallback_or_safety_cap(self):
        self.assertEqual(UTILITY_ACTION_WINDOW(8, 0, -1, -1, True), (8, False))
        self.assertEqual(UTILITY_ACTION_WINDOW(4, 8, 0, 3, True), (4, False))
        self.assertEqual(UTILITY_ACTION_WINDOW(15, 8, 0, 3, True), (8, True))
        self.assertEqual(UTILITY_ACTION_WINDOW(15, 8, 0, 3, False), (15, False))

    def test_execute_p3_runs_r1_only_for_fallback_or_bounded_comparison(self):
        raw = np.asarray([True, True, True, False])
        status = np.asarray([0, 5, 0, 0], dtype=np.int16)
        work, compare, sequence = UTILITY_R1_WORK_MASKS(
            raw,
            status,
            utility_enabled=True,
            executes_utility=True,
            compare_every_batches=0,
            execute_batch_sequence=0,
        )
        np.testing.assert_array_equal(work, [False, True, False, False])
        self.assertFalse(np.any(compare))
        self.assertEqual(sequence, 1)

        work, compare, sequence = UTILITY_R1_WORK_MASKS(
            raw,
            status,
            utility_enabled=True,
            executes_utility=True,
            compare_every_batches=2,
            execute_batch_sequence=1,
        )
        np.testing.assert_array_equal(work, [True, True, False, False])
        np.testing.assert_array_equal(compare, [True, False, False, False])
        self.assertEqual(sequence, 2)

        shadow_work, shadow_compare, shadow_sequence = UTILITY_R1_WORK_MASKS(
            raw,
            status,
            utility_enabled=True,
            executes_utility=False,
            compare_every_batches=1,
            execute_batch_sequence=9,
        )
        np.testing.assert_array_equal(shadow_work, raw)
        self.assertFalse(np.any(shadow_compare))
        self.assertEqual(shadow_sequence, 9)


class TestSurvivalMetrics(unittest.TestCase):
    def test_utility_metrics_belong_to_fixed_additive_keyspace(self):
        for key in (
            "selector_utility_model_fallback_count",
            "selector_utility_width_fallback_count",
            "selector_utility_invalid_row_fallback_count",
            "selector_utility_batch_fallback_count",
            "select_utility_batch_kernel_queries",
            "select_utility_lazy_r1_fallback_queries",
            "select_utility_r1_compare_queries",
            "select_utility_shadow_queries",
            "select_utility_execution_action_sum",
            hspec_utility_score_histogram_key(0.2),
            "select_stop_utility_abstain_count",
            "select_stop_utility_length_count",
        ):
            self.assertTrue(is_selector_additive_metric(key), key)

    def test_shadow_recorder_conserves_nonblocking_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = HSpecS13ShadowRecorder(
                temporary, queue_records=8, flush_records=2
            )
            self.assertTrue(recorder.record_many([
                {"event": "selection", "query_id": "q0"},
                {"event": "counterfactual", "query_id": "q0"},
            ]))
            recorder.flush("test")
            status_path = next(Path(temporary).glob("*.status.json"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["enqueued_records"], 2)
            self.assertEqual(status["written_records"], 2)
            self.assertEqual(status["dropped_records"], 0)
            recorder.close("test_done")


if __name__ == "__main__":
    unittest.main()
