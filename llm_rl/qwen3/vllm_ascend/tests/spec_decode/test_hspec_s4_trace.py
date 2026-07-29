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
                {"event": "selection", "query_id": "q1", "drafted_len": 2},
                {"event": "selection", "query_id": "q2", "drafted_len": 2},
            ])
            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["producer_sequence"] for row in rows], [0, 1])
            self.assertTrue(
                all(
                    row["schema_version"] == "hspec.s4.online-trace.v2"
                    for row in rows
                )
            )
            self.assertTrue(all(":rank-3:" in row["producer_id"] for row in rows))
            self.assertEqual(rows[0]["match_group_id"], rows[1]["match_group_id"])
            self.assertEqual(
                [row["match_group_recorded_row"] for row in rows], [0, 1]
            )
            self.assertEqual(
                [row["match_group_recorded_rows"] for row in rows], [2, 2]
            )

    def test_finalize_closes_only_unresolved_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = _load_trace_module("hspec_s4_trace_finalize_test", root)
            recorder = module.HSpecS4TraceRecorder(root, flush_records=100)
            recorder.record_many([
                {
                    "event": "selection",
                    "query_id": "q1",
                    "request_id": "r1",
                    "prompt_id": "p1",
                    "decoded_len": 7,
                    "active_table_version": 1,
                    "drafted_len": 2,
                },
                {
                    "event": "selection",
                    "query_id": "q2",
                    "request_id": "r2",
                    "prompt_id": "p2",
                    "decoded_len": 8,
                    "active_table_version": 1,
                    "drafted_len": 2,
                },
                {
                    "event": "selection",
                    "query_id": "q3",
                    "request_id": "r3",
                    "prompt_id": "p3",
                    "decoded_len": 9,
                    "active_table_version": 1,
                    "drafted_len": 0,
                },
            ])
            recorder.record_many([{
                "event": "verification",
                "query_id": "q1",
            }])
            recorder.finalize()
            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            terminal = [
                row for row in rows
                if row.get("reason")
                == "worker_shutdown_with_unresolved_trace_outcome"
            ]
            self.assertEqual([row["query_id"] for row in terminal], ["q2"])
            self.assertEqual(
                [row["producer_sequence"] for row in rows], [0, 1, 2, 3, 4]
            )

    def test_token_hash_has_length_domain_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_trace_module("hspec_s4_trace_hash_test", Path(tmp))
            self.assertNotEqual(
                module.hspec_s4_token_hash([1, 2]),
                module.hspec_s4_token_hash([1, 2, 0]),
            )


if __name__ == "__main__":
    unittest.main()
