import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.spec_decode.hspec_store import (
    HSpecLocalCollector,
    get_hspec_store_dtype,
    get_hspec_store_hidden_filename,
)


class HSpecS4StoreDtypeTest(unittest.TestCase):
    def test_float32_alias_and_filename_are_explicit(self):
        with patch.dict("os.environ", {"HSPEC_STORE_DTYPE": "fp32"}):
            self.assertEqual(get_hspec_store_dtype(), "float32")
            self.assertEqual(get_hspec_store_hidden_filename(), "hs.fp32.bin")

    def test_float32_budget_uses_four_bytes_per_hidden_value(self):
        with patch.dict("os.environ", {"HSPEC_STORE_DTYPE": "float32"}):
            self.assertEqual(
                HSpecLocalCollector._estimate_payload_bytes(3, 5, 2),
                3 * 5 * 4 + 2 * 4,
            )

    def test_float32_store_losslessly_carries_bfloat16_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "HSPEC_STORE_DTYPE": "float32",
                "HSPEC_STORE_DIR": str(root / "raw"),
                "HSPEC_TABLE_STORE_DIR": str(root / "table"),
                "HSPEC_NUM_SHARDS": "1",
                "HSPEC_STRICT_PROMPT_ID_ON_SEAL": "1",
            }
            with patch.dict("os.environ", env, clear=False):
                collector = HSpecLocalCollector()
                source = torch.tensor(
                    [[1.25, -2.5], [3.0, 0.125]], dtype=torch.bfloat16
                )
                collector.append_hidden_and_tokens(
                    "req-1", source, [11, 12], epoch=1
                )
                descriptors = collector.flush_descriptors(
                    {"req-1": "p-test"}, epoch=1, global_step=2
                )
                desc = descriptors["req-1"]
                self.assertEqual(desc.hs_dtype, "float32")
                self.assertEqual(Path(desc.hs_path).name, "hs.fp32.bin")
                stored = np.fromfile(desc.hs_path, dtype=np.float32).reshape(2, 2)
                np.testing.assert_array_equal(stored, source.float().numpy())


if __name__ == "__main__":
    unittest.main()
