#!/usr/bin/env python3
"""Atomically persist effective launcher values independently of stdout."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


LAUNCHER_FIELDS = {
    "train_file": "TRAIN_FILE",
    "test_file": "TEST_FILE",
    "dataset_fraction": "DATASET_FRACTION",
    "gpu_memory_utilization": "GPU_MEMORY_UTILIZATION",
    "max_prompt_length": "MAX_PROMPT_LENGTH",
    "max_response_length": "MAX_RESPONSE_LENGTH",
    "max_num_seqs": "MAX_NUM_SEQS",
    "rollout_n": "ROLLOUT_N",
    "train_batch_size": "TRAIN_BATCH_SIZE",
    "task_queue_enable": "TASK_QUEUE_ENABLE",
}


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--launcher-script", required=True)
    args = parser.parse_args()
    assignments = {
        output: os.environ.get(environment, "")
        for output, environment in LAUNCHER_FIELDS.items()
    }
    payload = {
        "schema_version": "hspec.launcher-metadata.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "launcher_script": str(Path(args.launcher_script).resolve()),
        "effective_launcher_assignments": assignments,
        "selector_profile": os.environ.get("HSPEC_SELECTOR_PROFILE", ""),
        "hspec_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("HSPEC_")
        },
        "run_identity": {
            "run_name": os.environ.get("RUN_NAME", ""),
            "hspec_run_name": os.environ.get("HSPEC_RUN_NAME", ""),
            "hspec_run_uid": os.environ.get("HSPEC_RUN_UID", ""),
            "output_log": os.environ.get("OUT", ""),
        },
    }
    atomic_write(Path(args.output).resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
