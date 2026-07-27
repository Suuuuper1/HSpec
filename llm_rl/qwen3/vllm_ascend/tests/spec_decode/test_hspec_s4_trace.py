import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "spec_decode" / "hspec_s4_trace.py"
)
MODEL_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "worker" / "model_runner_v1.py"
)
WORKER_PATH = Path(__file__).resolve().parents[2] / "worker" / "worker.py"


def _load_trace_module(name: str, output_dir: Path):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(
        os.environ,
        {
            "HSPEC_S4_TRACE_DIR": str(output_dir),
            "HSPEC_S4_TRACE_SAMPLE_EVERY": "1",
            "HSPEC_S4_TRACE_CAPTURE_PROJECTED": "1",
            "RANK": "3",
        },
        clear=False,
    ):
        spec.loader.exec_module(module)
    return module


class HSpecS4TraceTest(unittest.TestCase):
    def test_query_identity_is_deterministic_and_boundary_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_trace_module("hspec_s4_trace_identity_test", Path(tmp))
            first = module.hspec_s4_trace_query_id("r", "p", 7, 1)
            self.assertEqual(
                first, module.hspec_s4_trace_query_id("r", "p", 7, 1)
            )
            self.assertNotEqual(
                first, module.hspec_s4_trace_query_id("r", "p", 8, 1)
            )

    def test_recorder_writes_ordered_schema_versioned_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = _load_trace_module("hspec_s4_trace_recorder_test", root)
            recorder = module.HSpecS4TraceRecorder(root, flush_records=2)
            recorder.record_many([
                {"event": "selection", "query_id": "q1"},
                {"event": "verification", "query_id": "q1"},
            ])
            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["producer_sequence"] for row in rows], [0, 1])
            self.assertTrue(
                all(
                    row["schema_version"] == "hspec.s4.online-trace.v1"
                    for row in rows
                )
            )
            self.assertTrue(all(":rank-3:" in row["producer_id"] for row in rows))

    def test_token_hash_has_length_domain_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_trace_module("hspec_s4_trace_hash_test", Path(tmp))
            self.assertNotEqual(
                module.hspec_s4_token_hash([1, 2]),
                module.hspec_s4_token_hash([1, 2, 0]),
            )

    def test_close_cancels_only_unverified_tail_drafts_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = _load_trace_module("hspec_s4_trace_close_test", root)
            recorder = module.HSpecS4TraceRecorder(root, flush_records=100)
            recorder.record_many([
                {
                    "event": "selection",
                    "query_id": "verified",
                    "request_id": "r1",
                    "prompt_id": "p1",
                    "active_table_version": 1,
                    "drafted_len": 2,
                },
                {"event": "verification", "query_id": "verified"},
                {
                    "event": "selection",
                    "query_id": "no-draft",
                    "request_id": "r2",
                    "drafted_len": 0,
                },
                {
                    "event": "selection",
                    "query_id": "pending",
                    "request_id": "r3",
                    "prompt_id": "p3",
                    "active_table_version": 2,
                    "drafted_len": 3,
                },
            ])
            recorder.close()
            recorder.close()

            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            cancellations = [
                row for row in rows if row["event"] == "cancellation"
            ]
            self.assertEqual(len(cancellations), 1)
            self.assertEqual(cancellations[0]["query_id"], "pending")
            self.assertEqual(
                cancellations[0]["reason"],
                "producer_exit_before_verification",
            )
            self.assertEqual(
                [row["producer_sequence"] for row in rows], list(range(5))
            )

            with self.assertRaisesRegex(RuntimeError, "closed"):
                recorder.record_many([
                    {"event": "selection", "query_id": "late"}
                ])

    def test_seal_cancels_pending_queries_without_closing_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = _load_trace_module("hspec_s4_trace_seal_test", root)
            recorder = module.HSpecS4TraceRecorder(root, flush_records=100)
            recorder.record_many([{
                "event": "selection",
                "query_id": "first",
                "request_id": "r1",
                "drafted_len": 2,
            }])
            self.assertEqual(
                recorder.seal_open_queries(
                    "scheduler_batch_drained_before_verification"
                ),
                1,
            )
            self.assertEqual(
                recorder.seal_open_queries(
                    "scheduler_batch_drained_before_verification"
                ),
                0,
            )
            recorder.record_many([{
                "event": "selection",
                "query_id": "second",
                "request_id": "r2",
                "drafted_len": 0,
            }])
            recorder.close()

            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in rows],
                ["selection", "cancellation", "selection"],
            )
            self.assertEqual(
                rows[1]["reason"],
                "scheduler_batch_drained_before_verification",
            )
            self.assertEqual(
                [row["producer_sequence"] for row in rows], list(range(3))
            )

    def test_model_runner_seals_trace_when_scheduler_batch_drains(self):
        source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("scheduler_output.finished_req_ids", source)
        self.assertIn("self.input_batch.num_reqs == 0", source)
        self.assertIn("seal_hspec_s4_trace_open_queries(", source)
        self.assertIn(
            '"scheduler_batch_drained_before_verification"', source
        )

    def test_worker_shutdown_closes_trace_recorder(self):
        source = WORKER_PATH.read_text(encoding="utf-8")
        self.assertIn("def shutdown(self) -> None:", source)
        self.assertIn("close_hspec_s4_trace()", source)


if __name__ == "__main__":
    unittest.main()
