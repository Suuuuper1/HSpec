#!/usr/bin/env python3
"""Generate the Phase-0 offline target/draft compatibility manifest."""

import argparse
import json
from pathlib import Path

from vllm.v1.spec_decode.checkpoint_manifest import (
    ManifestError,
    build_migration_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dflash")
    parser.add_argument("--dspark")
    parser.add_argument(
        "--dflash-tokenizer",
        help="DFlash tokenizer artifacts; defaults to --tokenizer/--target",
    )
    parser.add_argument(
        "--dspark-tokenizer",
        help="DSpark tokenizer artifacts; defaults to --tokenizer/--target",
    )
    parser.add_argument("--dflash-pair-id")
    parser.add_argument("--dspark-pair-id")
    parser.add_argument("--target-revision")
    parser.add_argument("--dflash-revision")
    parser.add_argument("--dspark-revision")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--draft-sample-method", default="greedy")
    parser.add_argument("--rejection-sample-method", default="standard")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit nonzero unless both methods pass all research gates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    drafts = {
        method: path
        for method, path in (("dflash", args.dflash), ("dspark", args.dspark))
        if path
    }
    try:
        manifest = build_migration_manifest(
            target_path=args.target,
            tokenizer_path=args.tokenizer or args.target,
            draft_paths=drafts,
            draft_tokenizer_paths={
                method: path
                for method, path in (
                    ("dflash", args.dflash_tokenizer),
                    ("dspark", args.dspark_tokenizer),
                )
                if path
            },
            pair_ids={
                "dflash": args.dflash_pair_id,
                "dspark": args.dspark_pair_id,
            },
            revisions={
                "target": args.target_revision,
                "dflash": args.dflash_revision,
                "dspark": args.dspark_revision,
            },
            num_speculative_tokens=args.num_speculative_tokens,
            target_tp=args.target_tp,
            draft_tp=args.draft_tp,
            draft_sample_method=args.draft_sample_method,
            rejection_sample_method=args.rejection_sample_method,
        )
    except ManifestError as error:
        raise SystemExit(f"manifest error: {error}") from error

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Phase 0 manifest: {manifest['overall_status']} ({output})")
    if manifest["missing_methods"]:
        print(f"Missing draft checkpoints: {', '.join(manifest['missing_methods'])}")
    return int(args.require_pass and manifest["overall_status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
