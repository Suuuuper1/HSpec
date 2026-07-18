#!/usr/bin/env python3
"""Offline A2 error-type decomposition for HSpec draft selection.

This tool reuses the exact blocked top-k retrieval and descriptor/table-store
read helpers from the A1 oracle-ceiling script, then adds the four A2
counterfactuals:

1. current top-1 + current static length rule
2. current top-1 + oracle optimal length
3. top-k oracle best entry + current static length rule
4. top-k oracle best entry + oracle optimal length

The main output is a query-point level decomposition of ranking error and
length-control error under the token-boundary query approximation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_hspec_a1_topk_oracle as a1  # noqa: E402


DEFAULT_ANALYSIS_UNIVERSE = "spec_eligible"
DEFAULT_CURRENT_LENGTH_MODE = "static"
DEFAULT_ABS_DELTA_CAP_ENABLED = True
DEFAULT_ABS_DELTA_SAFE_THRESHOLD = 2
DEFAULT_ABS_DELTA_MID_THRESHOLD = 64
DEFAULT_ABS_DELTA_MID_CAP = 8
DEFAULT_ABS_DELTA_FAR_CAP = 4


def _load_optional_step_inventory(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "rows" in payload and isinstance(payload["rows"], list):
            rows = payload["rows"]
        elif "step_inventory" in payload and isinstance(payload["step_inventory"], list):
            rows = payload["step_inventory"]
        else:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "global_step" not in row:
            continue
        result[int(row["global_step"])] = row
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    return str(value).strip().lower() not in {"", "0", "false", "off", "no"}


def _resolve_abs_delta_config(args: argparse.Namespace) -> dict[str, int | bool]:
    return {
        "enabled": bool(_coerce_bool(args.abs_delta_cap_enabled)),
        "safe_threshold": max(int(args.abs_delta_safe_threshold), 0),
        "mid_threshold": max(int(args.abs_delta_mid_threshold), 0),
        "mid_cap": max(int(args.abs_delta_mid_cap), 1),
        "far_cap": max(int(args.abs_delta_far_cap), 1),
    }


def _main_universe_membership(row: dict[str, Any], universe: str) -> bool:
    if not bool(row.get("table_present", False)):
        return False
    if universe == "all":
        return True
    if universe == "spec_eligible":
        return bool(row.get("eligible_by_threshold", False))
    raise ValueError(f"unsupported analysis universe: {universe!r}")


def _apply_abs_delta_cap(
    *,
    window: int,
    abs_delta: int,
    horizon: int,
    min_wnd: int,
    config: dict[str, int | bool],
) -> int:
    wnd = int(window)
    if not bool(config["enabled"]):
        return wnd
    abs_delta_i = int(abs_delta)
    safe_threshold = int(config["safe_threshold"])
    mid_threshold = max(int(config["mid_threshold"]), safe_threshold)
    mid_cap = max(int(config["mid_cap"]), 1)
    far_cap = max(int(config["far_cap"]), 1)
    if abs_delta_i <= safe_threshold:
        return wnd
    if abs_delta_i <= mid_threshold:
        cap = max(int(min_wnd), min(mid_cap, int(horizon)))
    else:
        cap = max(int(min_wnd), min(far_cap, int(horizon)))
    return min(wnd, cap)


def _compute_static_current_length(
    *,
    table_data: dict[str, Any],
    entry_idx: int,
    query_token_idx: int,
    horizon: int,
    abs_delta_config: dict[str, int | bool],
) -> dict[str, int]:
    ridx = int(table_data["entry_rollout_idx"][entry_idx])
    off = int(table_data["entry_offset"][entry_idx])
    seq = table_data["rollout_seqs"][ridx]
    tail_len = max(int(len(seq)) - off, 0)
    base_pos = int(query_token_idx)
    matched_pos = int(off) - 1
    abs_delta = abs(matched_pos - base_pos)

    wnd_size = int(table_data.get("wnd_size", 8))
    max_wnd = int(table_data.get("max_wnd", max(int(horizon), 1)))
    min_wnd = int(table_data.get("min_wnd", 1))
    if max_wnd < min_wnd:
        max_wnd = min_wnd
    baseline_wnd = max(min_wnd, min(wnd_size, max_wnd))
    capped_wnd = _apply_abs_delta_cap(
        window=baseline_wnd,
        abs_delta=abs_delta,
        horizon=int(horizon),
        min_wnd=min_wnd,
        config=abs_delta_config,
    )
    capped_wnd = max(min_wnd, min(capped_wnd, int(horizon)))
    current_len = min(capped_wnd, tail_len)
    return {
        "base_pos": base_pos,
        "matched_pos": matched_pos,
        "abs_delta": abs_delta,
        "tail_len": tail_len,
        "wnd_size": baseline_wnd,
        "wnd_after_cap": capped_wnd,
        "current_len": current_len,
        "min_wnd": min_wnd,
        "max_wnd": max_wnd,
    }


def _safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num) / float(den)


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _compute_error_label(
    *,
    H_total: int,
    H_len: int,
    H_rank: int,
    H_coupled: int,
) -> str:
    if int(H_total) <= 0:
        return "no_headroom"
    if int(H_len) > 0 and int(H_rank) == 0:
        return "length_only"
    if int(H_rank) > 0 and int(H_len) == 0:
        return "ranking_only"
    if int(H_coupled) > 0:
        return "dual_error_coupled"
    return "dual_error_overlap"


def _summarize_decomposition(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_rows:
        return {
            "count": 0,
            "sum_baseline_accept": 0.0,
            "sum_top1_oracle_len": 0.0,
            "sum_topk_current_len": 0.0,
            "sum_joint_oracle": 0.0,
            "mean_baseline_accept": 0.0,
            "mean_top1_oracle_len": 0.0,
            "mean_topk_current_len": 0.0,
            "mean_joint_oracle": 0.0,
            "sum_headroom_total": 0.0,
            "sum_headroom_len": 0.0,
            "sum_headroom_rank": 0.0,
            "sum_headroom_len_after_rank": 0.0,
            "sum_headroom_rank_after_len": 0.0,
            "sum_phi_rank": 0.0,
            "sum_phi_len": 0.0,
            "sum_headroom_coupled": 0.0,
            "rank_share": 0.0,
            "len_share": 0.0,
            "coupling_share": 0.0,
            "error_label_histogram": {},
            "error_label_ratio": {},
        }

    hist: dict[str, int] = defaultdict(int)
    for row in query_rows:
        hist[str(row["error_label"])] += 1

    sum_b = float(sum(int(row["top1_current_accept_len"]) for row in query_rows))
    sum_l = float(sum(int(row["top1_full_match_len"]) for row in query_rows))
    sum_r = float(sum(int(row["best_current_accept_len"]) for row in query_rows))
    sum_j = float(sum(int(row["best_joint_full_match_len"]) for row in query_rows))
    sum_H_total = float(sum(int(row["H_total"]) for row in query_rows))
    sum_H_len = float(sum(int(row["H_len"]) for row in query_rows))
    sum_H_rank = float(sum(int(row["H_rank"]) for row in query_rows))
    sum_H_len_after_rank = float(sum(int(row["H_len_after_rank"]) for row in query_rows))
    sum_H_rank_after_len = float(sum(int(row["H_rank_after_len"]) for row in query_rows))
    sum_phi_rank = float(sum(float(row["phi_rank"]) for row in query_rows))
    sum_phi_len = float(sum(float(row["phi_len"]) for row in query_rows))
    sum_H_coupled = float(sum(int(row["H_coupled"]) for row in query_rows))

    ratio = {
        key: _safe_div(value, len(query_rows))
        for key, value in sorted(hist.items())
    }
    return {
        "count": len(query_rows),
        "sum_baseline_accept": sum_b,
        "sum_top1_oracle_len": sum_l,
        "sum_topk_current_len": sum_r,
        "sum_joint_oracle": sum_j,
        "mean_baseline_accept": _safe_div(sum_b, len(query_rows)),
        "mean_top1_oracle_len": _safe_div(sum_l, len(query_rows)),
        "mean_topk_current_len": _safe_div(sum_r, len(query_rows)),
        "mean_joint_oracle": _safe_div(sum_j, len(query_rows)),
        "sum_headroom_total": sum_H_total,
        "sum_headroom_len": sum_H_len,
        "sum_headroom_rank": sum_H_rank,
        "sum_headroom_len_after_rank": sum_H_len_after_rank,
        "sum_headroom_rank_after_len": sum_H_rank_after_len,
        "sum_phi_rank": sum_phi_rank,
        "sum_phi_len": sum_phi_len,
        "sum_headroom_coupled": sum_H_coupled,
        "rank_share": _safe_div(sum_phi_rank, sum_H_total),
        "len_share": _safe_div(sum_phi_len, sum_H_total),
        "coupling_share": _safe_div(sum_H_coupled, sum_H_total),
        "error_label_histogram": dict(sorted(hist.items())),
        "error_label_ratio": ratio,
    }


def _group_query_rows_by_topk(query_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[int(row["topk_requested"])].append(row)
    return grouped


def _write_topk_plot_a2(out_dir: Path, summary_by_topk: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if a1.plt is None:  # pragma: no cover - optional dependency
        outputs["warning"] = "matplotlib is unavailable; plot was skipped"
        return outputs

    topk_values = sorted(int(key) for key in summary_by_topk.keys())
    if not topk_values:
        outputs["warning"] = "no topk values to plot"
        return outputs

    def _series(universe: str, metric: str) -> list[float]:
        return [
            float(summary_by_topk[str(topk)][universe][metric])
            for topk in topk_values
        ]

    fig, axes = a1.plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    universe_specs = [
        ("all_query_points", "All Query Points", axes[0]),
        ("spec_eligible_query_points", "Spec Eligible", axes[1]),
    ]
    for universe_key, title, row_axes in universe_specs:
        row_axes[0].plot(topk_values, _series(universe_key, "mean_baseline_accept"), marker="o", label="baseline")
        row_axes[0].plot(topk_values, _series(universe_key, "mean_top1_oracle_len"), marker="o", label="top1+oracle_len")
        row_axes[0].plot(topk_values, _series(universe_key, "mean_topk_current_len"), marker="o", label="topk+current_len")
        row_axes[0].plot(topk_values, _series(universe_key, "mean_joint_oracle"), marker="o", label="joint_oracle")
        row_axes[0].set_title(f"{title}: Counterfactual Means")
        row_axes[0].set_xlabel("topk")
        row_axes[0].set_ylabel("accept len")
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend()

        row_axes[1].plot(topk_values, _series(universe_key, "rank_share"), marker="o", label="rank_share")
        row_axes[1].plot(topk_values, _series(universe_key, "len_share"), marker="o", label="len_share")
        row_axes[1].plot(topk_values, _series(universe_key, "coupling_share"), marker="o", label="coupling_share")
        row_axes[1].set_title(f"{title}: Error Shares")
        row_axes[1].set_xlabel("topk")
        row_axes[1].set_ylabel("share")
        row_axes[1].grid(True, alpha=0.3)
        row_axes[1].legend()

    plot_path = out_dir / "topk_sweep_a2.png"
    fig.savefig(plot_path, dpi=160)
    a1.plt.close(fig)
    outputs["png"] = str(plot_path)
    return outputs


def _mark_candidate_flags(
    candidate_records: list[dict[str, Any]],
    candidate_start: int,
    *,
    step: int,
    prompt_id: str,
    request_id: str,
    external_request_id: str,
    tp_group_id: int,
    query_token_idx: int,
    best_current_entry_idx: int,
    best_current_rank: int,
    best_joint_entry_idx: int,
    best_joint_rank: int,
) -> None:
    for cand_idx in range(candidate_start, len(candidate_records)):
        cand = candidate_records[cand_idx]
        if (
            int(cand["global_step"]) != int(step)
            or str(cand["prompt_id"]) != str(prompt_id)
            or str(cand["request_id"]) != str(request_id)
            or str(cand.get("external_request_id", "")) != str(external_request_id)
            or int(cand.get("tp_group_id", 0)) != int(tp_group_id)
            or int(cand["query_token_idx"]) != int(query_token_idx)
        ):
            continue
        if (
            int(cand["candidate_entry_idx"]) == int(best_current_entry_idx)
            and int(cand["candidate_rank_by_sim"]) == int(best_current_rank)
        ):
            cand["is_best_current"] = True
        if (
            int(cand["candidate_entry_idx"]) == int(best_joint_entry_idx)
            and int(cand["candidate_rank_by_sim"]) == int(best_joint_rank)
        ):
            cand["is_best_joint"] = True


def _analyze_prompt_group(
    *,
    step: int,
    epoch: int,
    prompt_id: str,
    shard_id: int,
    version_record: dict[str, Any],
    desc_items: list[dict[str, Any]],
    table_desc: Any | None,
    table_cache: Any,
    prompt_desc_cache: dict[tuple[int, int], dict[str, Any]],
    hspec_store_mod: Any,
    hspec_table_store_mod: Any,
    topk_values: list[int],
    horizon: int,
    threshold: float,
    query_block_size: int,
    key_block_size: int,
    compute_backend: str,
    torch_device: str,
    abs_delta_config: dict[str, int | bool],
    analysis_universe: str,
    query_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    step_counters: dict[str, int],
) -> None:
    max_topk = max(int(value) for value in topk_values)
    trajectories: list[dict[str, Any]] = []
    total_query_points = 0

    for item in desc_items:
        desc = item["desc"]
        hs, tokens = hspec_store_mod.materialize_hspec_trajectory(desc)
        hs_np = np.ascontiguousarray(np.asarray(hs, dtype=np.float32))
        tok_np = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
        if hs_np.shape[0] != tok_np.shape[0]:
            raise RuntimeError(
                f"trajectory token/hidden mismatch: request_id={desc.request_id} "
                f"hs_rows={hs_np.shape[0]} token_len={tok_np.shape[0]}"
            )
        query_count = max(int(tok_np.shape[0]) - 1, 0)
        total_query_points += query_count
        trajectories.append(
            {
                "desc": desc,
                "hs": hs_np,
                "tokens": tok_np,
                "query_count": query_count,
            }
        )

    if table_desc is None:
        for traj in trajectories:
            tokens = traj["tokens"]
            desc = traj["desc"]
            for token_idx in range(traj["query_count"]):
                query_row = {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "prompt_id": str(prompt_id),
                    "request_id": str(desc.request_id),
                    "external_request_id": str(
                        getattr(desc, "external_request_id", "")
                        or hspec_store_mod.hspec_external_request_id(str(desc.request_id))
                    ),
                    "shard_id": int(shard_id),
                    "tp_group_id": int(getattr(desc, "tp_group_id", 0)),
                    "query_token_idx": int(token_idx),
                    "query_token_id": int(tokens[token_idx]),
                    "response_len": int(tokens.shape[0]),
                    "table_present": False,
                    "table_version": int(version_record["version"]),
                    "table_active_epoch": int(version_record["active_epoch"]),
                    "table_entry_count": 0,
                    "top1_sim": None,
                    "top2_sim": None,
                    "top1_margin": None,
                    "top1_entry_idx": None,
                    "top1_current_len": None,
                    "top1_full_match_len": None,
                    "top1_current_accept_len": None,
                    "best_current_entry_idx": None,
                    "best_current_rank": None,
                    "best_current_current_len": None,
                    "best_current_full_match_len": None,
                    "best_current_accept_len": None,
                    "best_joint_entry_idx": None,
                    "best_joint_rank": None,
                    "best_joint_full_match_len": None,
                    "H_total": None,
                    "H_len": None,
                    "H_rank": None,
                    "H_len_after_rank": None,
                    "H_rank_after_len": None,
                    "phi_rank": None,
                    "phi_len": None,
                    "H_coupled": None,
                    "error_label": "table_miss",
                    "eligible_by_threshold": False,
                    "analysis_universe_main": False,
                    "query_target_len": int(min(horizon, tokens.shape[0] - token_idx - 1)),
                }
                for topk_requested in topk_values:
                    missing_row = dict(query_row)
                    missing_row["topk_requested"] = int(topk_requested)
                    missing_row["topk_effective"] = 0
                    query_records.append(missing_row)
        step_counters["num_query_points_total"] += total_query_points
        step_counters["num_query_points_table_miss"] += total_query_points
        return

    table_data = table_cache.get_or_load(
        int(shard_id),
        int(version_record["version"]),
        str(prompt_id),
        loader=lambda: hspec_table_store_mod.materialize_prompt_table(table_desc),
    )
    table_data = a1._prepare_table_data(table_data)

    mean = np.ascontiguousarray(np.asarray(table_data["mean"], dtype=np.float32))
    components = np.ascontiguousarray(np.asarray(table_data["components"], dtype=np.float32))
    keys = np.ascontiguousarray(np.asarray(table_data["keys"], dtype=np.float32))
    rewards = table_data.get("rewards")
    if rewards is not None:
        rewards = np.ascontiguousarray(np.asarray(rewards, dtype=np.float32))

    all_queries: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    for traj in trajectories:
        hs_np = traj["hs"]
        tokens = traj["tokens"]
        desc = traj["desc"]
        for token_idx in range(traj["query_count"]):
            all_queries.append(hs_np[token_idx])
            all_meta.append(
                {
                    "request_id": str(desc.request_id),
                    "external_request_id": str(
                        getattr(desc, "external_request_id", "")
                        or hspec_store_mod.hspec_external_request_id(str(desc.request_id))
                    ),
                    "tp_group_id": int(getattr(desc, "tp_group_id", 0)),
                    "query_token_idx": int(token_idx),
                    "query_token_id": int(tokens[token_idx]),
                    "response_len": int(tokens.shape[0]),
                    "target": a1._target_slice(tokens, token_idx, horizon),
                }
            )

    if not all_queries:
        return

    q_matrix = np.ascontiguousarray(np.stack(all_queries, axis=0), dtype=np.float32)
    z_queries = np.ascontiguousarray((q_matrix - mean) @ components.T, dtype=np.float32)
    topk_vals, topk_idxs = a1._exact_topk_blocked(
        z_queries,
        keys,
        topk=max_topk,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
        compute_backend=compute_backend,
        torch_device=torch_device,
    )

    step_counters["num_query_points_total"] += len(all_meta)
    step_counters["num_query_points_with_table"] += len(all_meta)

    for q_idx, meta in enumerate(all_meta):
        target = meta["target"]
        sims = topk_vals[q_idx]
        idxs = topk_idxs[q_idx]

        top1_sim = float(sims[0])
        top2_sim = float(sims[1]) if sims.shape[0] > 1 else None
        top1_margin = top1_sim - top2_sim if top2_sim is not None else None
        eligible = bool(top1_sim >= threshold)

        candidate_start = len(candidate_records)
        valid_mask = np.asarray(idxs >= 0, dtype=np.bool_)
        candidate_entry_indices = np.ascontiguousarray(idxs[valid_mask], dtype=np.int32)
        candidate_sims = np.ascontiguousarray(sims[valid_mask], dtype=np.float32)

        if candidate_entry_indices.size == 0:
            raise RuntimeError(
                f"no valid A2 candidates for step={step} prompt={prompt_id} query={q_idx}"
            )

        draft_matrix, draft_lens, rollout_idx_arr, entry_offsets_arr, tail_lens_arr = a1._extract_draft_batch(
            table_data,
            candidate_entry_indices,
            horizon,
        )
        full_match_lens = a1._prefix_match_lens_batch(draft_matrix, draft_lens, target)
        num_candidates = int(candidate_entry_indices.shape[0])

        current_len_arr = np.empty((num_candidates,), dtype=np.int32)
        matched_pos_arr = np.empty((num_candidates,), dtype=np.int32)
        base_pos_arr = np.empty((num_candidates,), dtype=np.int32)
        abs_delta_arr = np.empty((num_candidates,), dtype=np.int32)
        wnd_size_arr = np.empty((num_candidates,), dtype=np.int32)
        wnd_after_cap_arr = np.empty((num_candidates,), dtype=np.int32)
        current_accept_arr = np.empty((num_candidates,), dtype=np.int32)

        prefix_best_current_accept = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_current_rank = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_current_entry = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_current_len = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_current_full_match = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_joint_full = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_joint_rank = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_joint_entry = np.empty((num_candidates,), dtype=np.int32)
        prefix_best_joint_current_len = np.empty((num_candidates,), dtype=np.int32)

        running_best_current_accept = -1
        running_best_current_rank = -1
        running_best_current_entry = -1
        running_best_current_len = -1
        running_best_current_full_match = -1
        running_best_current_sim = -math.inf

        running_best_joint_full = -1
        running_best_joint_rank = -1
        running_best_joint_entry = -1
        running_best_joint_current_len = -1
        running_best_joint_sim = -math.inf

        for rank_idx, entry_i in enumerate(candidate_entry_indices.tolist(), start=1):
            row_idx = rank_idx - 1
            len_info = _compute_static_current_length(
                table_data=table_data,
                entry_idx=entry_i,
                query_token_idx=int(meta["query_token_idx"]),
                horizon=horizon,
                abs_delta_config=abs_delta_config,
            )
            current_len_arr[row_idx] = int(len_info["current_len"])
            matched_pos_arr[row_idx] = int(len_info["matched_pos"])
            base_pos_arr[row_idx] = int(len_info["base_pos"])
            abs_delta_arr[row_idx] = int(len_info["abs_delta"])
            wnd_size_arr[row_idx] = int(len_info["wnd_size"])
            wnd_after_cap_arr[row_idx] = int(len_info["wnd_after_cap"])
            current_accept_arr[row_idx] = min(int(full_match_lens[row_idx]), int(current_len_arr[row_idx]))

            sim_value = float(candidate_sims[row_idx])
            full_match_len = int(full_match_lens[row_idx])
            current_len = int(current_len_arr[row_idx])
            current_accept = int(current_accept_arr[row_idx])
            entry_reward = (
                float(rewards[entry_i])
                if rewards is not None and entry_i < len(rewards)
                else math.nan
            )

            candidate_records.append(
                {
                    "epoch": int(epoch),
                    "global_step": int(step),
                    "topk_max_requested": int(max_topk),
                    "prompt_id": str(prompt_id),
                    "request_id": str(meta["request_id"]),
                    "external_request_id": str(meta["external_request_id"]),
                    "shard_id": int(shard_id),
                    "tp_group_id": int(meta["tp_group_id"]),
                    "table_version": int(version_record["version"]),
                    "table_active_epoch": int(version_record["active_epoch"]),
                    "query_token_idx": int(meta["query_token_idx"]),
                    "candidate_rank_by_sim": int(rank_idx),
                    "candidate_entry_idx": int(entry_i),
                    "candidate_sim": sim_value,
                    "candidate_reward": None if math.isnan(entry_reward) else float(entry_reward),
                    "candidate_rollout_idx": int(rollout_idx_arr[row_idx]),
                    "candidate_entry_offset": int(entry_offsets_arr[row_idx]),
                    "candidate_matched_pos": int(matched_pos_arr[row_idx]),
                    "candidate_base_pos": int(base_pos_arr[row_idx]),
                    "candidate_abs_delta": int(abs_delta_arr[row_idx]),
                    "candidate_tail_len": int(tail_lens_arr[row_idx]),
                    "candidate_wnd_size": int(wnd_size_arr[row_idx]),
                    "candidate_wnd_after_cap": int(wnd_after_cap_arr[row_idx]),
                    "candidate_current_len": int(current_len),
                    "candidate_full_match_len": int(full_match_len),
                    "candidate_current_accept_len": int(current_accept),
                    "is_top1": bool(rank_idx == 1),
                    "is_best_current": False,
                    "is_best_joint": False,
                }
            )

            if (
                current_accept > running_best_current_accept
                or (current_accept == running_best_current_accept and sim_value > running_best_current_sim)
            ):
                running_best_current_accept = current_accept
                running_best_current_rank = rank_idx
                running_best_current_entry = int(entry_i)
                running_best_current_len = int(current_len)
                running_best_current_full_match = int(full_match_len)
                running_best_current_sim = sim_value

            if (
                full_match_len > running_best_joint_full
                or (full_match_len == running_best_joint_full and sim_value > running_best_joint_sim)
            ):
                running_best_joint_full = full_match_len
                running_best_joint_rank = rank_idx
                running_best_joint_entry = int(entry_i)
                running_best_joint_current_len = int(current_len)
                running_best_joint_sim = sim_value

            prefix_best_current_accept[row_idx] = int(running_best_current_accept)
            prefix_best_current_rank[row_idx] = int(running_best_current_rank)
            prefix_best_current_entry[row_idx] = int(running_best_current_entry)
            prefix_best_current_len[row_idx] = int(running_best_current_len)
            prefix_best_current_full_match[row_idx] = int(running_best_current_full_match)
            prefix_best_joint_full[row_idx] = int(running_best_joint_full)
            prefix_best_joint_rank[row_idx] = int(running_best_joint_rank)
            prefix_best_joint_entry[row_idx] = int(running_best_joint_entry)
            prefix_best_joint_current_len[row_idx] = int(running_best_joint_current_len)

        top1_entry_idx = int(candidate_entry_indices[0])
        top1_current_len = int(current_len_arr[0])
        top1_full_match_len = int(full_match_lens[0])
        top1_current_accept_len = int(current_accept_arr[0])

        step_counters["num_query_points_usable"] += 1
        if eligible:
            step_counters["num_query_points_spec_eligible"] += 1

        for topk_requested in topk_values:
            eff_k = min(int(topk_requested), num_candidates)
            eff_idx = eff_k - 1
            best_current_entry_idx = int(prefix_best_current_entry[eff_idx])
            best_current_rank = int(prefix_best_current_rank[eff_idx])
            best_current_len = int(prefix_best_current_len[eff_idx])
            best_current_full_match = int(prefix_best_current_full_match[eff_idx])
            best_current_accept = int(prefix_best_current_accept[eff_idx])
            best_joint_entry_idx = int(prefix_best_joint_entry[eff_idx])
            best_joint_rank = int(prefix_best_joint_rank[eff_idx])
            best_joint_full_match = int(prefix_best_joint_full[eff_idx])
            best_joint_current_len = int(prefix_best_joint_current_len[eff_idx])

            if int(topk_requested) == int(max_topk):
                _mark_candidate_flags(
                    candidate_records,
                    candidate_start,
                    step=step,
                    prompt_id=prompt_id,
                    request_id=str(meta["request_id"]),
                    external_request_id=str(meta["external_request_id"]),
                    tp_group_id=int(meta["tp_group_id"]),
                    query_token_idx=int(meta["query_token_idx"]),
                    best_current_entry_idx=best_current_entry_idx,
                    best_current_rank=best_current_rank,
                    best_joint_entry_idx=best_joint_entry_idx,
                    best_joint_rank=best_joint_rank,
                )

            b = int(top1_current_accept_len)
            l = int(top1_full_match_len)
            r = int(best_current_accept)
            j = int(best_joint_full_match)
            if not (0 <= b <= l <= j <= horizon):
                raise RuntimeError(
                    "A2 invariant violated for top1/joint decomposition: "
                    f"step={step} prompt={prompt_id} query={meta['query_token_idx']} "
                    f"topk={topk_requested} b={b} l={l} j={j} horizon={horizon}"
                )
            if not (0 <= b <= r <= j <= horizon):
                raise RuntimeError(
                    "A2 invariant violated for current-length oracle decomposition: "
                    f"step={step} prompt={prompt_id} query={meta['query_token_idx']} "
                    f"topk={topk_requested} b={b} r={r} j={j} horizon={horizon}"
                )

            H_total = int(j - b)
            H_len = int(l - b)
            H_rank = int(r - b)
            H_len_after_rank = int(j - r)
            H_rank_after_len = int(j - l)
            phi_rank = 0.5 * ((r - b) + (j - l))
            phi_len = 0.5 * ((l - b) + (j - r))
            H_coupled = int(j - max(l, r))

            if phi_rank < -1e-6 or phi_len < -1e-6:
                raise RuntimeError(
                    "A2 Shapley contribution became negative: "
                    f"step={step} prompt={prompt_id} query={meta['query_token_idx']} "
                    f"topk={topk_requested} phi_rank={phi_rank} phi_len={phi_len}"
                )
            if abs((phi_rank + phi_len) - H_total) > 1e-6:
                raise RuntimeError(
                    "A2 Shapley sum mismatch: "
                    f"step={step} prompt={prompt_id} query={meta['query_token_idx']} "
                    f"topk={topk_requested} phi_rank={phi_rank} phi_len={phi_len} H_total={H_total}"
                )

            error_label = _compute_error_label(
                H_total=H_total,
                H_len=H_len,
                H_rank=H_rank,
                H_coupled=H_coupled,
            )

            query_row = {
                "epoch": int(epoch),
                "global_step": int(step),
                "topk_requested": int(topk_requested),
                "topk_effective": int(eff_k),
                "prompt_id": str(prompt_id),
                "request_id": str(meta["request_id"]),
                "external_request_id": str(meta["external_request_id"]),
                "shard_id": int(shard_id),
                "tp_group_id": int(meta["tp_group_id"]),
                "query_token_idx": int(meta["query_token_idx"]),
                "query_token_id": int(meta["query_token_id"]),
                "response_len": int(meta["response_len"]),
                "table_present": True,
                "table_version": int(version_record["version"]),
                "table_active_epoch": int(version_record["active_epoch"]),
                "table_entry_count": int(keys.shape[0]),
                "top1_sim": float(top1_sim),
                "top2_sim": _float_or_none(top2_sim),
                "top1_margin": _float_or_none(top1_margin),
                "top1_entry_idx": int(top1_entry_idx),
                "top1_current_len": int(top1_current_len),
                "top1_full_match_len": int(top1_full_match_len),
                "top1_current_accept_len": int(top1_current_accept_len),
                "best_current_entry_idx": int(best_current_entry_idx),
                "best_current_rank": int(best_current_rank),
                "best_current_current_len": int(best_current_len),
                "best_current_full_match_len": int(best_current_full_match),
                "best_current_accept_len": int(best_current_accept),
                "best_joint_entry_idx": int(best_joint_entry_idx),
                "best_joint_rank": int(best_joint_rank),
                "best_joint_current_len": int(best_joint_current_len),
                "best_joint_full_match_len": int(best_joint_full_match),
                "H_total": H_total,
                "H_len": H_len,
                "H_rank": H_rank,
                "H_len_after_rank": H_len_after_rank,
                "H_rank_after_len": H_rank_after_len,
                "phi_rank": float(phi_rank),
                "phi_len": float(phi_len),
                "H_coupled": H_coupled,
                "error_label": str(error_label),
                "eligible_by_threshold": bool(eligible),
                "analysis_universe_main": False,
                "query_target_len": int(len(target)),
            }
            query_row["analysis_universe_main"] = _main_universe_membership(query_row, analysis_universe)
            query_records.append(query_row)


def _make_step_summary_row(
    *,
    epoch: int,
    step: int,
    step_counters: dict[str, int],
    all_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    analysis_universe: str,
) -> dict[str, Any]:
    all_summary = _summarize_decomposition(all_rows)
    eligible_summary = _summarize_decomposition(eligible_rows)
    main_summary = _summarize_decomposition(main_rows)
    return {
        "epoch": int(epoch),
        "global_step": int(step),
        **step_counters,
        "analysis_universe": str(analysis_universe),
        "num_query_points_main": int(main_summary["count"]),
        "main_baseline_mean_accept": float(main_summary["mean_baseline_accept"]),
        "main_top1_oracle_mean_accept": float(main_summary["mean_top1_oracle_len"]),
        "main_topk_current_mean_accept": float(main_summary["mean_topk_current_len"]),
        "main_joint_oracle_mean_accept": float(main_summary["mean_joint_oracle"]),
        "main_rank_share": float(main_summary["rank_share"]),
        "main_len_share": float(main_summary["len_share"]),
        "main_coupling_share": float(main_summary["coupling_share"]),
        "main_length_only_ratio": float(main_summary["error_label_ratio"].get("length_only", 0.0)),
        "main_ranking_only_ratio": float(main_summary["error_label_ratio"].get("ranking_only", 0.0)),
        "main_dual_error_coupled_ratio": float(main_summary["error_label_ratio"].get("dual_error_coupled", 0.0)),
        "main_dual_error_overlap_ratio": float(main_summary["error_label_ratio"].get("dual_error_overlap", 0.0)),
        "all_query_points_count": int(all_summary["count"]),
        "all_query_points_rank_share": float(all_summary["rank_share"]),
        "all_query_points_len_share": float(all_summary["len_share"]),
        "all_query_points_coupling_share": float(all_summary["coupling_share"]),
        "spec_eligible_count": int(eligible_summary["count"]),
        "spec_eligible_rank_share": float(eligible_summary["rank_share"]),
        "spec_eligible_len_share": float(eligible_summary["len_share"]),
        "spec_eligible_coupling_share": float(eligible_summary["coupling_share"]),
    }


def _analyze_command(args: argparse.Namespace) -> int:
    if args.current_length_mode != DEFAULT_CURRENT_LENGTH_MODE:
        raise ValueError(
            "only current-length-mode=static is implemented in the first A2 version"
        )

    manifest = a1._read_env_manifest(Path(args.run_manifest) if args.run_manifest else None)
    hspec_store_dir_value = args.hspec_store_dir or manifest.get("HSPEC_STORE_DIR", "")
    hspec_table_store_dir_value = args.hspec_table_store_dir or manifest.get("HSPEC_TABLE_STORE_DIR", "")
    if not hspec_store_dir_value:
        raise ValueError("hspec store dir is required")
    if not hspec_table_store_dir_value:
        raise ValueError("hspec table store dir is required")

    target_steps = set(a1._parse_step_spec(args.target_steps))
    topk_values = a1._parse_positive_int_spec(args.topk)
    threshold = float(
        args.similarity_threshold
        if args.similarity_threshold is not None
        else manifest.get("A1_HSPEC_SIMILARITY_THRESHOLD", a1.DEFAULT_THRESHOLD)
    )
    horizon = int(
        args.max_draft_tokens
        if args.max_draft_tokens is not None
        else manifest.get("A1_HSPEC_NUM_SPECULATIVE_TOKENS", a1.DEFAULT_DRAFT_HORIZON)
    )
    abs_delta_config = _resolve_abs_delta_config(args)

    step_inventory = _load_optional_step_inventory(Path(args.step_inventory) if args.step_inventory else None)
    for step in sorted(target_steps):
        inventory_row = step_inventory.get(step)
        if inventory_row is None:
            continue
        epoch_value = inventory_row.get("epoch")
        if epoch_value is not None and int(epoch_value) < 1:
            raise RuntimeError(f"step={step} belongs to epoch 0 and is not eligible for A2")

    hspec_store_mod, hspec_table_store_mod, hspec_utils_mod = a1._load_hspec_modules(a1.REPO_ROOT)
    descriptors = a1._load_target_descriptors(
        Path(hspec_store_dir_value).expanduser(),
        hspec_store_mod,
        target_steps,
    )
    version_records = (
        a1._load_version_catalog_from_json(Path(args.table_version_catalog))
        if args.table_version_catalog
        else a1._scan_table_versions(Path(hspec_table_store_dir_value).expanduser(), hspec_table_store_mod)[0]
    )
    version_map = a1._version_index(version_records)
    num_shards = a1._infer_num_shards(version_records)

    grouped_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in descriptors:
        grouped_steps[int(item["global_step"])].append(item)

    missing_steps = sorted(step for step in target_steps if step not in grouped_steps)
    if missing_steps:
        raise RuntimeError(f"target steps not found in raw descriptors: {missing_steps}")

    query_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    prompt_desc_cache: dict[tuple[int, int], dict[str, Any]] = {}
    table_cache = a1._PromptTableCache(max_prompts=int(args.table_cache_prompts))

    for step in sorted(target_steps):
        items = grouped_steps[step]
        epochs = sorted({int(item["epoch"]) for item in items})
        if len(epochs) != 1:
            raise RuntimeError(f"step has mixed epochs: step={step} epochs={epochs}")
        epoch = int(epochs[0])
        if epoch < 1:
            raise RuntimeError(f"step={step} belongs to epoch 0 and is not eligible for A2")
        empty_prompt_ids = sum(1 for item in items if not str(item["prompt_id"]))
        if empty_prompt_ids:
            raise RuntimeError(
                f"step={step}: {empty_prompt_ids}/{len(items)} descriptors have empty prompt_id. "
                "Raw-store seal mapping is broken; refusing silent skip. "
                "See HSpec_reaserch_doc/optim_draft_select/exp/hspec_a1_prompt_id_disk_fix_guide.md"
            )

        deduped_items = a1._dedupe_tp_fanout_for_step(items, hspec_utils_mod)
        for item in deduped_items:
            prompt_id = str(item["prompt_id"])
            expected_shard = int(hspec_utils_mod.stable_partition_id(prompt_id, num_shards))
            if int(item["shard_id"]) != expected_shard:
                raise RuntimeError(
                    "descriptor shard mismatch after prompt_id repair: "
                    f"step={step} request_id={item['request_id']} prompt_id={prompt_id} "
                    f"desc.shard_id={item['shard_id']} expected_shard={expected_shard} "
                    f"num_shards={num_shards}"
                )

        target_active_epoch = epoch - 1
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in deduped_items:
            prompt_id = str(item["prompt_id"])
            by_prompt[prompt_id].append(item)

        step_counters = {
            "num_desc_scanned": len(items),
            "num_desc_after_tp_dedupe": len(deduped_items),
            "num_desc_analyzed": len(deduped_items),
            "num_prompts_analyzed": len(by_prompt),
            "num_query_points_total": 0,
            "num_query_points_usable": 0,
            "num_query_points_with_table": 0,
            "num_query_points_table_miss": 0,
            "num_query_points_spec_eligible": 0,
        }

        for prompt_id, prompt_items in sorted(by_prompt.items()):
            shard_ids = sorted({int(item["shard_id"]) for item in prompt_items})
            if len(shard_ids) != 1:
                raise RuntimeError(
                    f"prompt spans multiple shards in one step: step={step} prompt_id={prompt_id} shards={shard_ids}"
                )
            shard_id = int(shard_ids[0])
            version_record = version_map.get(shard_id, {}).get(target_active_epoch)
            if version_record is None:
                raise RuntimeError(
                    "missing table version mapping for target step: "
                    f"step={step} epoch={epoch} shard={shard_id} target_active_epoch={target_active_epoch}"
                )
            prompt_descs = a1._load_prompt_active_index(
                version_record,
                hspec_table_store_mod,
                prompt_desc_cache,
            )
            table_desc = prompt_descs.get(prompt_id)
            _analyze_prompt_group(
                step=step,
                epoch=epoch,
                prompt_id=prompt_id,
                shard_id=shard_id,
                version_record=version_record,
                desc_items=prompt_items,
                table_desc=table_desc,
                table_cache=table_cache,
                prompt_desc_cache=prompt_desc_cache,
                hspec_store_mod=hspec_store_mod,
                hspec_table_store_mod=hspec_table_store_mod,
                topk_values=topk_values,
                horizon=int(horizon),
                threshold=float(threshold),
                query_block_size=int(args.query_block_size),
                key_block_size=int(args.key_block_size),
                compute_backend=str(args.compute_backend),
                torch_device=str(args.torch_device),
                abs_delta_config=abs_delta_config,
                analysis_universe=str(args.analysis_universe),
                query_records=query_records,
                candidate_records=candidate_records,
                step_counters=step_counters,
            )

        step_query_rows = [row for row in query_records if int(row["global_step"]) == int(step)]
        for topk_requested in topk_values:
            step_topk_rows = [
                row for row in step_query_rows
                if int(row["topk_requested"]) == int(topk_requested)
            ]
            step_all_rows = [row for row in step_topk_rows if bool(row["table_present"])]
            step_eligible_rows = [
                row for row in step_all_rows if bool(row["eligible_by_threshold"])
            ]
            step_main_rows = [
                row for row in step_all_rows
                if _main_universe_membership(row, str(args.analysis_universe))
            ]
            row = _make_step_summary_row(
                epoch=epoch,
                step=step,
                step_counters=step_counters,
                all_rows=step_all_rows,
                eligible_rows=step_eligible_rows,
                main_rows=step_main_rows,
                analysis_universe=str(args.analysis_universe),
            )
            row["topk_requested"] = int(topk_requested)
            step_rows.append(row)

    all_query_rows = [row for row in query_records if bool(row["table_present"])]
    eligible_query_rows = [row for row in all_query_rows if bool(row["eligible_by_threshold"])]
    summary_by_topk: dict[str, Any] = {}
    for topk_requested in topk_values:
        topk_all_rows = [
            row for row in all_query_rows
            if int(row["topk_requested"]) == int(topk_requested)
        ]
        topk_eligible_rows = [
            row for row in eligible_query_rows
            if int(row["topk_requested"]) == int(topk_requested)
        ]
        topk_main_rows = [
            row for row in topk_all_rows
            if _main_universe_membership(row, str(args.analysis_universe))
        ]
        summary_by_topk[str(int(topk_requested))] = {
            "main_universe": _summarize_decomposition(topk_main_rows),
            "all_query_points": _summarize_decomposition(topk_all_rows),
            "spec_eligible_query_points": _summarize_decomposition(topk_eligible_rows),
        }

    unique_step_rows = [
        row for row in step_rows
        if int(row["topk_requested"]) == int(topk_values[0])
    ]

    summary = {
        "topk_values": topk_values,
        "num_steps_analyzed": len(target_steps),
        "num_desc_scanned": int(sum(int(row["num_desc_scanned"]) for row in unique_step_rows)),
        "num_desc_after_tp_dedupe": int(
            sum(int(row["num_desc_after_tp_dedupe"]) for row in unique_step_rows)
        ),
        "num_desc_analyzed": int(sum(int(row["num_desc_analyzed"]) for row in unique_step_rows)),
        "num_query_points_total": int(sum(int(row["num_query_points_total"]) for row in unique_step_rows)),
        "num_query_points_table_present": int(sum(int(row["num_query_points_with_table"]) for row in unique_step_rows)),
        "num_query_points_spec_eligible": int(sum(int(row["num_query_points_spec_eligible"]) for row in unique_step_rows)),
        "num_query_points_main_universe": int(
            sum(int(row["num_query_points_main"]) for row in unique_step_rows)
        ),
        "analysis_config": {
            "topk_values": topk_values,
            "max_draft_tokens": int(horizon),
            "similarity_threshold": float(threshold),
            "analysis_universe": str(args.analysis_universe),
            "current_length_mode": str(args.current_length_mode),
            "query_block_size": int(args.query_block_size),
            "key_block_size": int(args.key_block_size),
            "table_cache_prompts": int(args.table_cache_prompts),
            "compute_backend": str(args.compute_backend),
            "torch_device": str(args.torch_device),
            "target_steps": sorted(target_steps),
            "abs_delta_cap_enabled": bool(abs_delta_config["enabled"]),
            "abs_delta_safe_threshold": int(abs_delta_config["safe_threshold"]),
            "abs_delta_mid_threshold": int(abs_delta_config["mid_threshold"]),
            "abs_delta_mid_cap": int(abs_delta_config["mid_cap"]),
            "abs_delta_far_cap": int(abs_delta_config["far_cap"]),
        },
        "by_topk": summary_by_topk,
    }

    out_dir = Path(args.out_dir).resolve()
    a1._ensure_dir(out_dir)
    a1._write_json(out_dir / "summary_by_topk.json", summary)
    a1._write_csv(out_dir / "step_summary_by_topk.csv", step_rows)
    query_outputs = a1._write_records(out_dir / "query_records_by_topk", query_records)
    candidate_outputs = a1._write_records(out_dir / "candidate_records_max_topk", candidate_records)
    plot_outputs = _write_topk_plot_a2(out_dir, summary_by_topk)
    a1._write_json(
        out_dir / "artifact_paths.json",
        {
            "query_records_by_topk": query_outputs,
            "candidate_records_max_topk": candidate_outputs,
            "plots": plot_outputs,
        },
    )

    if len(topk_values) == 1:
        topk_key = str(int(topk_values[0]))
        legacy_summary = {
            "num_steps_analyzed": int(summary["num_steps_analyzed"]),
            "num_desc_scanned": int(summary["num_desc_scanned"]),
            "num_desc_after_tp_dedupe": int(summary["num_desc_after_tp_dedupe"]),
            "num_desc_analyzed": int(summary["num_desc_analyzed"]),
            "num_query_points_total": int(summary["num_query_points_total"]),
            "num_query_points_table_present": int(summary["num_query_points_table_present"]),
            "num_query_points_spec_eligible": int(summary["num_query_points_spec_eligible"]),
            "num_query_points_main_universe": int(summary["num_query_points_main_universe"]),
            "analysis_config": summary["analysis_config"],
            "main_universe": summary_by_topk[topk_key]["main_universe"],
            "all_query_points": summary_by_topk[topk_key]["all_query_points"],
            "spec_eligible_query_points": summary_by_topk[topk_key]["spec_eligible_query_points"],
        }
        a1._write_json(out_dir / "summary.json", legacy_summary)
        a1._write_csv(
            out_dir / "step_summary.csv",
            [row for row in step_rows if int(row["topk_requested"]) == int(topk_values[0])],
        )
        a1._write_records(
            out_dir / "query_records",
            [row for row in query_records if int(row["topk_requested"]) == int(topk_values[0])],
        )
        a1._write_records(out_dir / "candidate_records", candidate_records)

    print(f"A2 analysis written to {out_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=str, default=None, help="optional collect-run env manifest")
    parser.add_argument("--hspec-store-dir", type=str, default=None, help="descriptor raw store root")
    parser.add_argument("--hspec-table-store-dir", type=str, default=None, help="table store root")
    parser.add_argument("--step-inventory", type=str, default=None, help="optional A1 step_inventory.json")
    parser.add_argument("--table-version-catalog", type=str, default=None, help="optional A1 table_version_catalog.json")
    parser.add_argument("--target-steps", type=str, required=True, help="comma list or ranges, e.g. 12,13,20-25")
    parser.add_argument("--topk", type=str, default="8", help="one or more positive integers, e.g. 8 or 2,4,8")
    parser.add_argument("--max-draft-tokens", type=int, default=None, help="draft horizon H; defaults from manifest or 15")
    parser.add_argument("--similarity-threshold", type=float, default=None, help="spec-eligibility threshold; defaults from manifest or 0.85")
    parser.add_argument(
        "--analysis-universe",
        type=str,
        choices=("spec_eligible", "all"),
        default=DEFAULT_ANALYSIS_UNIVERSE,
        help="main analysis universe for step-level summary fields",
    )
    parser.add_argument(
        "--current-length-mode",
        type=str,
        choices=(DEFAULT_CURRENT_LENGTH_MODE,),
        default=DEFAULT_CURRENT_LENGTH_MODE,
        help="current draft-length approximation mode",
    )
    parser.add_argument("--query-block-size", type=int, default=256, help="number of query rows per blocked similarity pass")
    parser.add_argument("--key-block-size", type=int, default=8192, help="number of keys per blocked similarity pass")
    parser.add_argument("--table-cache-prompts", type=int, default=64, help="max materialized prompt tables kept in the LRU")
    parser.add_argument(
        "--compute-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default=a1.DEFAULT_COMPUTE_BACKEND,
        help="backend for blocked exact top-k retrieval",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        choices=("auto", "cpu", "cuda", "npu"),
        default=a1.DEFAULT_TORCH_DEVICE,
        help="device used when compute-backend=torch or auto resolves to torch",
    )
    parser.add_argument("--abs-delta-cap-enabled", type=str, default="1", help="whether to apply the static abs-delta window cap")
    parser.add_argument("--abs-delta-safe-threshold", type=int, default=DEFAULT_ABS_DELTA_SAFE_THRESHOLD, help="safe abs-delta threshold")
    parser.add_argument("--abs-delta-mid-threshold", type=int, default=DEFAULT_ABS_DELTA_MID_THRESHOLD, help="mid abs-delta threshold")
    parser.add_argument("--abs-delta-mid-cap", type=int, default=DEFAULT_ABS_DELTA_MID_CAP, help="mid abs-delta cap")
    parser.add_argument("--abs-delta-far-cap", type=int, default=DEFAULT_ABS_DELTA_FAR_CAP, help="far abs-delta cap")
    parser.add_argument("--out-dir", type=str, required=True, help="directory to write analysis artifacts")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return _analyze_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
