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


if __name__ == "__main__":
    unittest.main()
