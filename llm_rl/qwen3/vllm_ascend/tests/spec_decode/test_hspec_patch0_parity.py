import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_parity_module():
    path = PROJECT_ROOT / "vllm_ascend" / "spec_decode" / "hspec_parity.py"
    name = "hspec_patch0_parity_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parity = _load_parity_module()


class TestParityRecorder(unittest.TestCase):

    def test_exact_tokens_are_buffered_and_flushed(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = parity.HSpecS1ParityRecorder(temporary, flush_records=2)
            recorder.record_batch(
                request_ids=["r0", "r1"],
                prompt_token_ids=[[11, 12], [11, 12]],
                output_prefix_token_ids=[[21], [21, 22]],
                draft_token_ids=[[31, 32], []],
                target_token_ids=[[31, 40], [23]],
                accepted_prefix_lengths=[1, 0],
                spec_decode_active=True,
            )
            self.assertTrue(recorder.path.is_file())
            records = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["draft_token_ids"], [31, 32])
            self.assertEqual(records[0]["target_token_ids"], [31, 40])
            self.assertEqual(records[0]["accepted_prefix_len"], 1)
            self.assertEqual(records[1]["output_prefix_token_ids"], [21, 22])
            self.assertEqual(
                records[0]["prompt_token_ids_sha256"],
                records[1]["prompt_token_ids_sha256"],
            )

    def test_empty_rows_are_not_emitted_and_manual_flush_preserves_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = parity.HSpecS1ParityRecorder(temporary, flush_records=8)
            recorder.record_batch(
                request_ids=["empty", "tail"],
                prompt_token_ids=[[1], [2]],
                output_prefix_token_ids=[[], [3]],
                draft_token_ids=[[], [4]],
                target_token_ids=[[], [5]],
                accepted_prefix_lengths=[0, 0],
                spec_decode_active=False,
            )
            self.assertFalse(recorder.path.exists())
            recorder.flush()
            records = recorder.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["request_id"], "tail")


if __name__ == "__main__":
    unittest.main()
