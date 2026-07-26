import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import vllm_ascend.spec_decode.hspec_utils as hspec_utils

from vllm_ascend.spec_decode.hspec_store import (
    HSpecLocalCollector,
    HSpecTrajectoryDesc,
)


class TestS3CollectionMetadata(unittest.TestCase):

    def _environment(self, root: Path) -> dict[str, str]:
        return {
            "HSPEC_STORE_DIR": str(root / "raw"),
            "HSPEC_TABLE_STORE_DIR": str(root / "table"),
            "HSPEC_STORE_DTYPE": "float16",
            "HSPEC_NUM_SHARDS": "2",
            "HSPEC_INFER_TP": "2",
            "HSPEC_STRICT_PROMPT_ID_ON_SEAL": "1",
            "HSPEC_SEGMENT_FSYNC_ON_SEAL": "0",
            "HSPEC_RAW_STORE_MAX_BYTES": "0",
            "HSPEC_RAW_STORE_MAX_FILES": "0",
            "HSPEC_COLLECT_MAX_BYTES_PER_WORKER": "0",
            "HSPEC_RAW_STORE_MAX_BYTES_PER_EPOCH": "0",
            "HSPEC_S3_PHASE_A_COLLECTION": "1",
        }

    def test_execute_calls_remain_distinct_extents_and_health_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, self._environment(root), clear=False):
                collector = HSpecLocalCollector()
                collector.bind_prompt_id("req-1", "prompt-1")
                collector.append_hidden_and_tokens(
                    "req-1", torch.ones((2, 4)), [10, 11], epoch=0
                )
                collector.append_hidden_and_tokens(
                    "req-1", torch.ones((3, 4)), [12, 13, 14], epoch=0
                )
                desc = collector.flush_descriptors(
                    epoch=0,
                    global_step=1,
                    runtime_metrics={"copy_backpressure_drop": 0},
                )["req-1"]

            self.assertEqual([extent.length for extent in desc.extents or ()], [2, 3])
            self.assertEqual(desc.chunk_count, 2)
            self.assertFalse(desc.has_gap)
            self.assertEqual(
                [sum(ext.length for ext in desc.extents[:index + 1]) - 1
                 for index in range(len(desc.extents or ()))],
                [1, 4],
            )
            health_files = list((root / "raw").rglob("collection_health.jsonl"))
            self.assertEqual(len(health_files), 1)
            health = json.loads(health_files[0].read_text(encoding="utf-8"))
            self.assertEqual(health["epoch"], 0)
            self.assertEqual(health["desc_count"], 1)
            self.assertEqual(health["runtime_metrics"]["copy_backpressure_drop"], 0)

    def test_gap_reason_survives_descriptor_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, self._environment(root), clear=False):
                collector = HSpecLocalCollector()
                collector.bind_prompt_id("req-2", "prompt-2")
                collector.append_hidden_and_tokens(
                    "req-2", torch.ones((1, 3)), [20], epoch=1
                )
                collector.mark_gap(["req-2"], "copy_submit_error")
                collector.append_hidden_and_tokens(
                    "req-2", torch.ones((1, 3)), [21], epoch=1
                )
                desc = collector.flush_descriptors(
                    epoch=1, global_step=2, runtime_metrics={"copy_submit_error": 1}
                )["req-2"]

            self.assertTrue(desc.has_gap)
            self.assertEqual(desc.gap_reasons, ("copy_submit_error",))
            payload = json.loads(
                Path(desc.hs_path).with_name("desc.jsonl").read_text(encoding="utf-8")
            )
            restored = HSpecTrajectoryDesc(**payload)
            self.assertTrue(restored.has_gap)
            self.assertEqual(restored.gap_reasons, ("copy_submit_error",))

            string_reason = HSpecTrajectoryDesc(**{
                **payload,
                "gap_reasons": "copy_submit_error",
            })
            self.assertEqual(string_reason.gap_reasons, ("copy_submit_error",))

    def test_legacy_descriptor_without_gap_fields_remains_compatible(self):
        desc = HSpecTrajectoryDesc(
            epoch=0,
            global_step=0,
            node_id="node",
            worker_rank=0,
            tp_group_id=0,
            shard_id=0,
            request_id="request",
            prompt_id="prompt",
            hs_path="/tmp/hs.bin",
            token_path="/tmp/tokens.bin",
            length=1,
            hidden_dim=2,
            hs_dtype="float16",
            token_dtype="int32",
        )
        self.assertFalse(desc.has_gap)
        self.assertIsNone(desc.gap_reasons)

    def test_health_snapshot_does_not_consume_trainer_metric_interval(self):
        class CollectorStub:
            runtime_metrics = None

            def flush_descriptors(self, **kwargs):
                self.runtime_metrics = kwargs.get("runtime_metrics")
                return {}

        collector = CollectorStub()
        hspec_utils.hspec_collect_runtime_metrics(reset=True)
        hspec_utils._hspec_metric_add("copy_submit_error", 2)
        try:
            with patch.dict(
                os.environ, {"HSPEC_S3_PHASE_A_COLLECTION": "1"}, clear=False
            ), patch(
                "vllm_ascend.spec_decode.hspec_store.get_hspec_local_collector",
                return_value=collector,
            ):
                hspec_utils.hspec_flush_and_get_descriptors(epoch=0, global_step=1)
            self.assertEqual(collector.runtime_metrics["copy_submit_error"], 2)
            trainer_interval = hspec_utils.hspec_collect_runtime_metrics(reset=True)
            self.assertEqual(trainer_interval["copy_submit_error"], 2)
        finally:
            hspec_utils.hspec_collect_runtime_metrics(reset=True)


if __name__ == "__main__":
    unittest.main()
