import ast
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECORDER_PATH = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_s9_shadow.py"
PROPOSER_PATH = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"
ANALYZER_PATH = (
    PROJECT_ROOT
    / "HSpec_research_doc/HSpec_draft_delect_optim/s9_shadow/tools/analyze_s9_shadow.py"
)


def load_module(name: str, path: Path, env=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    with patch.dict(os.environ, env or {}, clear=False):
        spec.loader.exec_module(module)
    return module


def extract_method(name: str, globals_dict: dict[str, Any]):
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HSpecProposer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    item.decorator_list = []
                    namespace = dict(globals_dict)
                    exec(
                        compile(
                            ast.fix_missing_locations(
                                ast.Module(body=[item], type_ignores=[])
                            ),
                            str(PROPOSER_PATH),
                            "exec",
                        ),
                        namespace,
                    )
                    return namespace[name]
    raise AssertionError(f"method missing: {name}")


class S9RecorderTest(unittest.TestCase):
    def test_round_flush_acks_and_writes_partial_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_module(
                "hspec_s9_shadow_flush_test",
                RECORDER_PATH,
                {"HSPEC_S9_SHADOW_DIR": str(root)},
            )
            recorder = module.HSpecS9ShadowRecorder(
                root, queue_records=16, flush_records=100
            )
            recorder.record_many([{"event": "selection", "query_id": "tail"}])
            recorder.flush("round_complete")
            self.assertEqual(len(recorder.path.read_text().splitlines()), 1)
            status = json.loads(next(root.glob("*.status.json")).read_text())
            self.assertEqual(status["written_records"], 1)
            self.assertFalse(status["closed"])
            self.assertTrue(status["quiescent"])
            recorder.close("test_complete")

    def test_async_writer_is_lossless_and_publishes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_module(
                "hspec_s9_shadow_recorder_test",
                RECORDER_PATH,
                {"HSPEC_S9_SHADOW_DIR": str(root), "RANK": "2"},
            )
            recorder = module.HSpecS9ShadowRecorder(
                root, queue_records=16, flush_records=2
            )
            self.assertTrue(recorder.record_many([
                {"event": "selection", "query_id": "q1"},
                {"event": "verification", "query_id": "q1"},
            ]))
            recorder.close("test_complete")
            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["producer_sequence"] for row in rows], [0, 1])
            status = json.loads(next(root.glob("*.status.json")).read_text())
            self.assertEqual(status["written_records"], 2)
            self.assertEqual(
                status["enqueued_records"],
                status["written_records"] + status["dropped_records"],
            )
            self.assertEqual(status["dropped_records"], 0)
            self.assertEqual(status["write_errors"], 0)
            self.assertTrue(status["closed"])
            self.assertTrue(status["quiescent"])

    def test_query_identity_is_deterministic_and_position_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_module(
                "hspec_s9_shadow_identity_test",
                RECORDER_PATH,
                {"HSPEC_S9_SHADOW_DIR": tmp, "HSPEC_S9_SAMPLE_EVERY": "1"},
            )
            first = module.hspec_s9_query_identity("r", "p", 7, 1)
            self.assertEqual(first, module.hspec_s9_query_identity("r", "p", 7, 1))
            self.assertNotEqual(first[0], module.hspec_s9_query_identity("r", "p", 8, 1)[0])
            self.assertTrue(first[1])


class S9LifecycleTest(unittest.TestCase):
    def test_online_suffix_does_not_double_append_current_emitted_tokens(self):
        proposer_source = PROPOSER_PATH.read_text(encoding="utf-8")
        rerank_call = proposer_source.index(
            "slot, suffix, utility, _, _ = self._rerank_r1_row"
        )
        current_tokens = proposer_source.rfind(
            "current_tokens = (", 0, rerank_call
        )
        self.assertNotIn(
            "valid_sampled_token_ids",
            proposer_source[current_tokens:rerank_call],
        )

        runner_path = PROJECT_ROOT / "vllm_ascend/worker/model_runner_v1.py"
        runner_source = runner_path.read_text(encoding="utf-8")
        bookkeeping_call = runner_source.index(") = self._bookkeeping_sync(")
        proposal_call = runner_source.index(
            "propose_draft_token_ids(valid_sampled_token_ids)",
            bookkeeping_call,
        )
        state_extend = runner_source.index(
            "req_state.output_token_ids.extend(sampled_ids)"
        )
        bookkeeping_return = runner_source.index(
            "return (", state_extend
        )
        self.assertLess(state_extend, bookkeeping_return)
        self.assertLess(bookkeeping_call, proposal_call)

    def test_r1_position_and_label_use_post_bookkeeping_decoded_len(self):
        source = PROPOSER_PATH.read_text(encoding="utf-8")
        self.assertIn("base_pos = active_base_positions[j]", source)
        s9_meta = source.index(
            "if self._s9_shadow_enabled and s9_record_group:"
        )
        label_boundary = source.index('"s9_label_start_len"', s9_meta)
        next_pending_write = source.index(
            "self._pending_verify_meta[req_id]", s9_meta
        )
        self.assertLess(label_boundary, next_pending_write)

    def test_counterfactual_probe_closes_on_first_mismatch(self):
        events = []
        method = extract_method(
            "_s9_advance_request",
            {
                "Any": Any,
                "Dict": Dict,
                "List": List,
                "Optional": Optional,
                "Sequence": Sequence,
                "hspec_s9_token_hash": lambda values: f"h{list(values)}",
                "record_hspec_s9_shadow_events": events.extend,
            },
        )
        proposer = SimpleNamespace(
            _s9_shadow_enabled=True,
            _s9_last_output_tokens={},
            _s9_shadow_probes={
                "r": [{
                    "query_id": "q",
                    "prompt_id": "p",
                    "decoded_len": 2,
                    "label_start_len": 4,
                    "active_table_version": 1,
                    "shadow_draft_tokens": [5, 6, 7],
                    "shadow_accept_len": 0,
                    "observed_future_tokens": [],
                }]
            },
        )
        proposer._s9_advance_request = MethodType(method, proposer)
        proposer._s9_advance_request("r", [5, 9])
        self.assertNotIn("r", proposer._s9_shadow_probes)
        self.assertEqual(events[0]["shadow_accept_len"], 1)
        self.assertTrue(events[0]["label_exact"])

    def test_counterfactual_probe_accumulates_bounded_incremental_extents(self):
        events = []
        method = extract_method(
            "_s9_advance_request",
            {
                "Any": Any,
                "Dict": Dict,
                "List": List,
                "Optional": Optional,
                "Sequence": Sequence,
                "hspec_s9_token_hash": lambda values: f"h{list(values)}",
                "record_hspec_s9_shadow_events": events.extend,
            },
        )
        proposer = SimpleNamespace(
            _s9_shadow_enabled=True,
            _s9_shadow_probes={
                "r": [{
                    "query_id": "q",
                    "prompt_id": "p",
                    "decoded_len": 4,
                    "label_start_len": 4,
                    "active_table_version": 1,
                    "shadow_draft_tokens": [5, 6, 7],
                    "shadow_accept_len": 0,
                    "observed_future_tokens": [],
                }]
            },
        )
        proposer._s9_advance_request = MethodType(method, proposer)
        proposer._s9_advance_request("r", [5])
        self.assertEqual(
            proposer._s9_shadow_probes["r"][0]["observed_future_tokens"],
            [5],
        )
        proposer._s9_advance_request("r", [6, 7, 999])
        self.assertNotIn("r", proposer._s9_shadow_probes)
        self.assertEqual(events[0]["shadow_accept_len"], 3)
        self.assertEqual(events[0]["observed_future_len"], 3)
        self.assertTrue(events[0]["label_exact"])


class S9OfflineReplayTest(unittest.TestCase):
    def test_replay_group_sampling_covers_producer_version_strata(self):
        analyzer = load_module("analyze_s9_shadow_strata_test", ANALYZER_PATH)
        groups = {
            ("p0", 0): [{
                "producer_id": "p0", "active_table_version": 1,
                "match_padded_entries": 8,
            }],
            ("p0", 10): [{
                "producer_id": "p0", "active_table_version": 1,
                "match_padded_entries": 32,
            }],
            ("p1", 0): [{
                "producer_id": "p1", "active_table_version": 2,
                "match_padded_entries": 16,
            }],
            ("p1", 10): [{
                "producer_id": "p1", "active_table_version": 2,
                "match_padded_entries": 64,
            }],
        }
        selected = analyzer.select_replay_group_keys(groups, 2)
        strata = {
            (
                groups[key][0]["producer_id"],
                groups[key][0]["active_table_version"],
            )
            for key in selected
        }
        self.assertEqual(strata, {("p0", 1), ("p1", 2)})

    def test_compact_and_full_table_replay(self):
        analyzer = load_module("analyze_s9_shadow_test", ANALYZER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
            key_path = root / "keys.bin"
            keys.tofile(key_path)
            index_dir = root / "shard_0/version_1"
            index_dir.mkdir(parents=True)
            row = {
                "desc": {
                    "prompt_id": "p",
                    "version": 1,
                    "n_entries": 3,
                    "keys": {
                        "path": str(key_path),
                        "dtype": "float32",
                        "shape": [3, 2],
                        "offset": 0,
                        "order": "C",
                    },
                }
            }
            (index_dir / "active_prompt_index.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            keys_t = torch.from_numpy(keys)
            z = torch.tensor([1.0, 0.0])
            raw = keys_t @ z
            cosine = raw / (torch.linalg.vector_norm(keys_t, dim=1) * z.norm())
            scores, indices = torch.topk(cosine, 2, sorted=True)
            relative = float((scores[0] - scores[1]) / abs(scores[0]))
            config = {
                "topk": 2,
                "relative_radius": 0.001,
                "suffix_cap": 2,
                "relative_weight": 1.0,
                "suffix_weight": 1.0,
                "position_mode": "none",
                "utility_threshold": -1e30,
            }
            event = {
                "producer_id": "worker",
                "match_group_local_id": 0,
                "match_batch_row": 0,
                "match_batch_rows": 1,
                "match_padded_entries": 3,
                "match_components": 2,
                "active_table_version": 1,
                "prompt_id": "p",
                "table_entries": 3,
                "projected_query_fp32": [1.0, 0.0],
                "raw_top1_entry_idx": 0,
                "raw_top1_score": float(raw[0]),
                "raw_gate_hit": True,
                "candidate_entry_indices": indices.tolist(),
                "candidate_scores": scores.tolist(),
                "current_suffix_tokens": [10, 11],
                "current_emitted_token_ids": [11],
                "decoded_len": 2,
                "label_start_len": 2,
                "query_pos": 1,
                "p0_window_base_pos": 1,
                "r1_config": config,
                "r1_kernel_slot": 0,
                "r1_reference_slot": 0,
                "reference_error": None,
                "candidate_features": [
                    {
                        "entry_idx": int(indices[0]), "valid": True,
                        "eligible": True, "relative_drop": 0.0,
                        "suffix": 2, "utility": 2.0,
                        "matched_pos": 1, "abs_delta": 0,
                        "history_tail_tokens": [10, 11],
                    },
                    {
                        "entry_idx": int(indices[1]), "valid": True,
                        "eligible": True, "relative_drop": relative,
                        "suffix": 1, "utility": 1.0 - relative,
                        "matched_pos": 2, "abs_delta": 1,
                        "history_tail_tokens": [9, 11],
                    },
                ],
            }
            self.assertEqual(analyzer.recompute_compact(event), [])
            wrong_boundary = dict(event)
            wrong_boundary["label_start_len"] = 3
            self.assertIn(
                "label_start_boundary",
                analyzer.recompute_compact(wrong_boundary),
            )
            wrong_suffix = dict(event)
            wrong_suffix["current_suffix_tokens"] = [10, 12]
            self.assertIn(
                "current_suffix_missing_emitted_tail",
                analyzer.recompute_compact(wrong_suffix),
            )
            replay = analyzer.replay_topk_groups(
                {("worker", 0): [event]},
                analyzer.table_index(root),
                device="cpu",
                maximum_groups=1,
                score_tolerance=1e-6,
            )
            self.assertEqual(replay["replayed_groups"], 1)
            self.assertEqual(replay["entry_errors"], 0)
            self.assertEqual(replay["score_errors"], 0)


class S9IncrementTimingTest(unittest.TestCase):
    @staticmethod
    def _timing_event(**overrides):
        values = {
            "r1_cpu_rerank_ms": 0.10,
            "shadow_identity_ms": 0.02,
            "shadow_trace_d2h_ms": 0.0,
            "shadow_cancel_ms": 0.0,
            "shadow_pending_meta_ms": 0.01,
            "shadow_record_ms": 0.03,
            "shadow_timing_emit_lagged_ms": 0.01,
            "total_ms": 4.50,
        }
        values.update(overrides.pop("selector_overrides", {}))
        return {
            "event": "batch_timing",
            "timing_schema_version": 2,
            "shadow_lifecycle_ms": 0.03,
            "selector_timings_ms": values,
            **overrides,
        }

    def test_increment_summary_excludes_existing_p0_total(self):
        analyzer = load_module("analyze_s9_shadow_timing_test", ANALYZER_PATH)
        summary = analyzer.summarize_increment_timings(
            [self._timing_event()],
            {
                "candidate_increment_p95_ms": 0.50,
                "d2h_increment_p95_ms": 0.0,
            },
        )
        self.assertEqual(summary["increment_timing_schema_errors"], 0)
        self.assertEqual(summary["increment_timing_samples"], 1)
        self.assertAlmostEqual(
            summary["production_selector_increment_model_p95_ms"], 0.60
        )
        self.assertAlmostEqual(
            summary["shadow_hot_path_increment_model_p95_ms"], 0.70
        )

    def test_legacy_or_incomplete_timing_cannot_pass_as_v2(self):
        analyzer = load_module("analyze_s9_shadow_legacy_timing_test", ANALYZER_PATH)
        legacy = self._timing_event()
        legacy.pop("timing_schema_version")
        incomplete = self._timing_event()
        incomplete["selector_timings_ms"].pop("r1_cpu_rerank_ms")
        summary = analyzer.summarize_increment_timings(
            [legacy, incomplete],
            {
                "candidate_increment_p95_ms": 0.50,
                "d2h_increment_p95_ms": 0.0,
            },
        )
        self.assertEqual(summary["increment_timing_schema_errors"], 2)
        self.assertEqual(summary["increment_timing_samples"], 0)
        self.assertTrue(
            math.isnan(summary["shadow_hot_path_increment_model_p95_ms"])
        )

    def test_gate_uses_increment_not_full_selector_latency(self):
        analyzer = load_module("analyze_s9_shadow_increment_gate_test", ANALYZER_PATH)
        contract = analyzer.read_json(
            PROJECT_ROOT
            / "HSpec_research_doc/HSpec_draft_delect_optim/s9_shadow/s9_contract.json"
        )
        replay = {
            "replayed_groups": 16,
            "entry_errors": 0,
            "score_errors": 0,
            "missing_tables": 0,
            "device": "npu:0",
            "replayed_table_versions": [1],
            "replayed_producers": ["p"],
            "replayed_producer_version_strata": [["p", 1]],
        }
        run = {
            "environment_errors": [],
            "selection_count": 1,
            "dropped_records": 0,
            "write_errors": 0,
            "unfinished_tasks": 0,
            "nonquiescent_recorders": 0,
            "recorder_accounting_errors": 0,
            "recorder_trace_count_errors": 0,
            "duplicate_status_producers": 0,
            "missing_status_producers": [],
            "shadow_output_parity_errors": 0,
            "compact_reference_error_count": 0,
            "bounded_trace_errors": 0,
            "selector_contract_error_count": 0,
            "full_table_replay": replay,
            "trace_table_versions": [1],
            "trace_producers": ["p"],
            "trace_producer_version_strata": [["p", 1]],
            "lifecycle_errors": 0,
            "label_linkage_errors": 0,
            "exact_divergence_labels": 10,
            "baseline_divergence_atq": 0.5,
            "shadow_divergence_atq": 1.0,
            "selected_rank_one_based_mean": 1.0,
            "selected_suffix_mean": contract["reference_distribution"][
                "s6_test_selected_suffix"
            ],
            "max_decoded_len": 1024,
            "timing_samples": 200,
            "increment_timing_samples": 200,
            "increment_timing_schema_errors": 0,
            "selector_total_p95_ms": 4.50,
            "shadow_hot_path_increment_model_p95_ms": 1.0,
            "max_padded_entries": 128,
            "max_timing_padded_entries": 128,
            "max_active_rows": 16,
            "max_timing_components": 64,
            "min_actual_entries": 64,
            "max_actual_entries": 128,
            "topology": {
                "INFER_TP": "4",
                "TRAIN_TP": "4",
                "TRAIN_PP": "4",
                "TRAIN_EP": "4",
                "MAX_PROMPT_LENGTH": "1024",
                "MAX_RESPONSE_LENGTH": "16384",
            },
        }
        gate = analyzer.build_gate(dict(run), dict(run), contract)
        self.assertTrue(gate["checks"]["increment_timing_schema_v2"])
        self.assertTrue(
            gate["checks"][
                "large_run_selector_increment_model_p95_under_budget"
            ]
        )
        self.assertNotIn("large_run_selector_p95_under_budget", gate["checks"])
        self.assertTrue(gate["promotion_allowed"])

        slow = dict(run)
        slow["shadow_hot_path_increment_model_p95_ms"] = 2.0
        slow_gate = analyzer.build_gate(dict(run), slow, contract)
        self.assertFalse(
            slow_gate["checks"][
                "large_run_selector_increment_model_p95_under_budget"
            ]
        )
        self.assertFalse(slow_gate["promotion_allowed"])

        uncovered = dict(run)
        uncovered["max_timing_padded_entries"] = 131065
        uncovered_gate = analyzer.build_gate(dict(run), uncovered, contract)
        self.assertFalse(
            uncovered_gate["checks"]["large_run_increment_model_shape_covered"]
        )
        self.assertFalse(uncovered_gate["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
