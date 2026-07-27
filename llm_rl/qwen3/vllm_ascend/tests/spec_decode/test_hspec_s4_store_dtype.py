import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.spec_decode.hspec_builder import (
    HSpecPCAConfig,
    _as_owned_writable_fp32_work_tile,
    build_prompt_table_to_store,
    fit_prompt_pca_streaming,
)
from vllm_ascend.spec_decode.hspec_store import (
    HSpecLocalCollector,
    get_hspec_store_dtype,
    get_hspec_store_hidden_filename,
    iter_hspec_hidden_tiles,
)
from vllm_ascend.spec_decode.hspec_table_store import HSpecTableStoreWriter


class HSpecS4StoreDtypeTest(unittest.TestCase):
    @staticmethod
    def _pca_config(method: str) -> HSpecPCAConfig:
        return HSpecPCAConfig(
            method=method,
            n_components=2,
            tile_rows=2,
            randomized_oversample=1,
            randomized_seed=202405,
            covariance_max_bytes=1024 * 1024,
            accum_dtype="float32",
            keys_dtype="float16",
        )

    def test_float16_remains_the_default_store_dtype(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_hspec_store_dtype(), "float16")

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

    def test_pca_work_tile_copies_only_non_owned_or_read_only_input(self):
        owned = np.arange(8, dtype=np.float32).reshape(2, 4).copy()
        self.assertTrue(owned.flags.owndata)
        self.assertIs(_as_owned_writable_fp32_work_tile(owned), owned)

        read_only = owned.view()
        read_only.flags.writeable = False
        work = _as_owned_writable_fp32_work_tile(read_only)
        self.assertIsNot(work, read_only)
        self.assertTrue(work.flags.owndata)
        self.assertTrue(work.flags.writeable)
        np.testing.assert_array_equal(work, read_only)

    def test_float16_descriptor_reuses_its_fp32_conversion_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "HSPEC_STORE_DTYPE": "float16",
                "HSPEC_STORE_DIR": str(root / "raw"),
                "HSPEC_TABLE_STORE_DIR": str(root / "table"),
                "HSPEC_NUM_SHARDS": "1",
                "HSPEC_STRICT_PROMPT_ID_ON_SEAL": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                collector = HSpecLocalCollector()
                collector.append_hidden_and_tokens(
                    "req-1",
                    torch.ones((2, 4), dtype=torch.bfloat16),
                    [11, 12],
                    epoch=0,
                )
                desc = collector.flush_descriptors(
                    {"req-1": "p-test"}, epoch=0, global_step=1
                )["req-1"]

                tiles = iter_hspec_hidden_tiles(desc, 2, dtype=np.float32)
                try:
                    _, tile = next(tiles)
                    self.assertTrue(tile.flags.owndata)
                    self.assertTrue(tile.flags.writeable)
                    self.assertIs(_as_owned_writable_fp32_work_tile(tile), tile)
                finally:
                    tiles.close()

    def test_float32_descriptor_builds_both_pca_methods_without_mutating_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "HSPEC_STORE_DTYPE": "float32",
                "HSPEC_STORE_DIR": str(root / "raw"),
                "HSPEC_TABLE_STORE_DIR": str(root / "table"),
                "HSPEC_NUM_SHARDS": "1",
                "HSPEC_STRICT_PROMPT_ID_ON_SEAL": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                collector = HSpecLocalCollector()
                source = torch.tensor(
                    [
                        [1.0, 0.0, 2.0, -1.0],
                        [0.0, 1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0, 1.0],
                        [-1.0, 2.0, 1.0, 0.0],
                        [1.0, -1.0, 2.0, 2.0],
                        [2.0, 2.0, -1.0, 1.0],
                    ],
                    dtype=torch.bfloat16,
                )
                collector.append_hidden_and_tokens(
                    "req-1", source[:3], [11, 12, 13], epoch=0
                )
                collector.append_hidden_and_tokens(
                    "req-1", source[3:], [14, 15, 16], epoch=0
                )
                desc = collector.flush_descriptors(
                    {"req-1": "p-test"}, epoch=0, global_step=1
                )["req-1"]

                raw_before = Path(desc.hs_path).read_bytes()
                tiles = iter_hspec_hidden_tiles(desc, 2, dtype=np.float32)
                try:
                    _, first_tile = next(tiles)
                    self.assertFalse(first_tile.flags.writeable)
                    self.assertFalse(first_tile.flags.owndata)
                finally:
                    tiles.close()

                for method in ("randomized", "covariance"):
                    result = fit_prompt_pca_streaming(
                        desc.prompt_id,
                        [desc],
                        config=self._pca_config(method),
                    )
                    self.assertEqual(result.method, method)
                    self.assertEqual(result.n_samples, 6)
                    self.assertEqual(result.components.shape, (2, 4))

                writer = HSpecTableStoreWriter(
                    root=root / "table", shard_id=0, version=1
                )
                table_desc, metrics = build_prompt_table_to_store(
                    prompt_id=desc.prompt_id,
                    descs=[desc],
                    writer=writer,
                    n_components=2,
                    max_entries=100,
                    pca_config=self._pca_config("randomized"),
                )
                manifest = writer.seal()

                self.assertEqual(table_desc.n_entries, 5)
                self.assertEqual(metrics.n_entries, 5)
                self.assertEqual(manifest["prompt_count"], 1)
                self.assertEqual(manifest["entry_count"], 5)
                self.assertGreater(manifest["bytes_committed"], 0)
                self.assertEqual(Path(desc.hs_path).read_bytes(), raw_before)


if __name__ == "__main__":
    unittest.main()
