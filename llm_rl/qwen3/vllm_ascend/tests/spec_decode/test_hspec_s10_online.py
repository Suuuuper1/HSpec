import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
S10_DIR = (
    PROJECT_ROOT
    / "HSpec_research_doc/HSpec_draft_delect_optim/s10_online_ab"
)
ANALYZER_PATH = S10_DIR / "tools/analyze_s10_online_ab.py"


def _load_analyzer():
    name = "hspec_s10_analyzer_under_test"
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_analyzer()


def _step_line(step: int, epoch: int, arm: str, performance: bool) -> str:
    if arm == "p0":
        first, accepted, zero, gen_seconds, step_seconds = 40, 80, 60, 10.0, 12.0
        online = {}
    else:
        first, accepted, zero, gen_seconds, step_seconds = 60, 120, 40, 8.0, 10.0
        online = {
            "hspec/select_r1_execution_batches": 10,
            "hspec/select_r1_execution_queries": 100,
            "hspec/select_r1_changed_entry_count": 90,
            "hspec/select_r1_rank_one_based_sum": 101,
            "hspec/select_r1_suffix_sum": 300,
            "hspec/select_r1_cpu_rerank_samples": 10,
            "hspec/select_r1_cpu_rerank_us_le_700": 10,
        }
    metrics = {
        "training/epoch": epoch,
        "hspec/phase4_metrics_sampled": 1,
        "hspec/select_metrics_are_interval": 1,
        "hspec/select_metric_window_id": step,
        "hspec/select_eligible_queries": 100,
        "hspec/select_proposed_requests": 100,
        "hspec/select_abstained_requests": 0,
        "hspec/select_drafted_tokens": 400,
        "hspec/select_verified_requests": 100,
        "hspec/select_first_token_accepts": first,
        "hspec/select_accepted_tokens": accepted,
        "hspec/select_emitted_tokens": accepted + 100,
        "hspec/select_zero_accept_requests": zero,
        "hspec/select_canceled_requests": 0,
        "hspec/select_drafted_length_mismatch_count": 0,
        "hspec/select_funnel_decode_requests": 100,
        "hspec/select_funnel_active_table_requests": 100,
        "hspec/select_funnel_prompt_id_ready_requests": 100,
        "hspec/select_funnel_prompt_table_ready_requests": 100,
        "hspec/select_funnel_batch_cache_ready_requests": 100,
        "hspec/select_funnel_anchor_ready_requests": 100,
        "hspec/select_funnel_eligible_queries": 100,
        "hspec/select_accept_len_0_count": zero,
        "hspec/select_accept_len_2_count": first,
        "rollout/generated_tokens": 1000,
        "timing_s/gen": gen_seconds,
        "timing_s/step": step_seconds,
        "critic/rewards/mean": 0.2,
        "response_length/mean": 100.0,
        "actor/kl_loss": 0.001,
        "perf/max_memory_allocated_gb": 40.0,
    }
    metrics.update(online)
    fields = " - ".join(f"{key}:{value}" for key, value in metrics.items())
    prefix = "\x1b[36m(TaskRunner pid=1)\x1b[0m "
    return f"{prefix}step:{step} - {fields}\n"


def _write_run(
    root: Path,
    *,
    profile: str,
    arm: str,
    seed: int,
    performance: bool,
    contract: dict,
) -> None:
    run_dir = root / profile / f"S10-{profile}-{arm}-{seed}"
    run_dir.mkdir(parents=True)
    epochs = range(5) if performance else range(2)
    log_path = run_dir / "train.log"
    log_path.write_text(
        "".join(_step_line(index + 1, epoch, arm, performance) for index, epoch in enumerate(epochs)),
        encoding="utf-8",
    )
    environment = analyzer.expected_environment(contract, arm)
    manifest = {
        "schema_version": "hspec.s0.run-manifest.v1",
        "run_id": f"S10-{profile}-{arm}-{seed}",
        "profile_id": profile,
        "stage_id": "S10",
        "comparison_id": "C-S10-R1-ONLINE",
        "hypothesis_id": "H-S10-R1-ONLINE-END-TO-END",
        "seed": seed,
        "status": "completed",
        "exit_code": 0,
        "effective_hspec_environment": environment,
        "deviations": [{"id": "s10-online-ab"}],
        "git": {"head": "synthetic"},
        "inputs": {
            "train_data": {"sha256": "train"},
            "validation_data": {"sha256": "test"},
            "model": {"identity_files": {"config.json": "model"}},
            "dist_checkpoint": {"identity_files": {}, "shard_sizes": {}},
        },
        "artifacts": {"train_log": str(log_path)},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class TestS10Analyzer(unittest.TestCase):
    def test_semantic_log_scan_ignores_metric_and_config_cooccurrence(self):
        log = "\n".join((
            "Warning: The watchdog timeout 600000ms is less than or equal to "
            "HCCL execution timeout 1836000ms!",
            "step:1 - hspec/build_timeout_discard:0.0 - "
            "hspec/build_timeout_unfinished_prompts:0.0 - "
            "hspec/phase4_metrics_error_count:0.0",
            "StructuredOutputsConfig(disable_fallback=False), "
            "compilation_config={'cudagraph_mode': 'PIECEWISE'}",
        ))
        audit = analyzer.scan_runtime_log_events(log)
        self.assertEqual(audit["policy"], analyzer.LOG_EVENT_POLICY)
        self.assertTrue(all(count == 0 for count in audit["counts"].values()))
        self.assertEqual(
            audit["nonfailure_observations"][
                "timeout_threshold_configuration_warning"
            ],
            1,
        )

    def test_semantic_log_scan_preserves_real_failure_events(self):
        log = "\n".join((
            "worker raised TimeoutError while waiting for HCCL",
            "ERROR: NPU graph fallback triggered after graph capture failed",
            "RuntimeError: NPU out of memory",
            "Traceback (most recent call last):",
        ))
        audit = analyzer.scan_runtime_log_events(log)
        self.assertEqual(
            audit["counts"],
            {"oom": 1, "timeout": 1, "graph_fallback": 1, "traceback": 1},
        )
        self.assertEqual(
            [item["line_number"] for item in audit["evidence"]],
            [1, 2, 3, 4],
        )

    def test_historical_analyzer_exception_is_exact_and_functional_only(self):
        contract = analyzer.read_json(S10_DIR / "s10_contract.json")
        environment = analyzer.expected_environment(contract, "r1")
        historical = contract["analysis_remediation"][
            "accepted_historical_functional_provenance"
        ][0]
        environment.update(historical["environment"])
        errors, compatibility = analyzer.validate_environment(
            contract, "r1", "functional_1p5b", environment,
            historical["git_head"],
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            compatibility["mode"], "exact_historical_functional_provenance"
        )
        errors, _ = analyzer.validate_environment(
            contract, "r1", "performance_30b", environment,
            historical["git_head"],
        )
        self.assertGreaterEqual(len(errors), 1)
        environment["HSPEC_S10_IMPLEMENTATION_SHA256"] = "different"
        errors, _ = analyzer.validate_environment(
            contract, "r1", "functional_1p5b", environment,
            historical["git_head"],
        )
        self.assertGreaterEqual(len(errors), 1)

    def test_exact_independent_bootstrap_uses_run_not_query_as_unit(self):
        interval = analyzer.exact_independent_bootstrap_ci(
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]
        )
        self.assertEqual(interval["samples"], 729)
        self.assertAlmostEqual(interval["low"], 0.1)
        self.assertAlmostEqual(interval["high"], 0.1)

    def test_synthetic_functional_then_six_run_final_gate_passes(self):
        contract = analyzer.read_json(S10_DIR / "s10_contract.json")
        with tempfile.TemporaryDirectory() as temporary:
            gate_root = Path(temporary)
            for arm in ("p0", "r1"):
                _write_run(
                    gate_root / "functional" / arm / "seed_20260721",
                    profile="functional_1p5b", arm=arm, seed=20260721,
                    performance=False, contract=contract,
                )
            functional = analyzer.analyze_functional(
                gate_root, gate_root / "functional_analysis", contract, True
            )
            self.assertEqual(functional["status"], "PASS", functional["diagnostics"])
            analyzer.atomic_write(gate_root / "functional_analysis/PASS", "PASS\n")

            for seed in (20260721, 20260722, 20260723):
                for arm in ("p0", "r1"):
                    _write_run(
                        gate_root / "performance" / arm / f"seed_{seed}",
                        profile="performance_30b", arm=arm, seed=seed,
                        performance=True, contract=contract,
                    )
            final = analyzer.analyze_final(
                gate_root, gate_root / "analysis", contract, True
            )
            self.assertEqual(final["status"], "PASS", final["diagnostics"])
            self.assertTrue(final["checks"]["atq_uplift_at_least_10pct"])
            self.assertTrue(final["checks"]["one_requested_metric_significant"])
            self.assertTrue(final["checks"]["selector_increment_p95_under_budget_every_run"])
            self.assertAlmostEqual(
                final["effects"]["rollout_tokens_per_second"]["relative_ratio"],
                1.25,
            )

    def test_v2_functional_analysis_is_authoritative_and_must_be_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate_root = Path(temporary)
            canonical = gate_root / "functional_analysis"
            canonical.mkdir()
            analyzer.atomic_json(canonical / "gate_result.json", {
                "mode": "functional", "status": "FAIL",
                "may_run_performance": False, "checks": {"old": False},
            })
            analyzer.atomic_write(canonical / "FAIL", "FAIL\n")
            remediation = gate_root / "functional_analysis_v2"
            remediation.mkdir()
            analyzer.atomic_json(remediation / "gate_result.json", {
                "mode": "functional", "status": "PASS",
                "may_run_performance": True, "checks": {"fixed": True},
                "analysis_provenance": analyzer.current_provenance(),
                "log_event_policy": analyzer.LOG_EVENT_POLICY,
            })
            analyzer.atomic_write(remediation / "PASS", "PASS\n")
            result, selected, errors = analyzer.load_authoritative_functional_gate(
                gate_root
            )
            self.assertEqual(errors, [])
            self.assertEqual(selected, remediation)
            self.assertEqual(result["status"], "PASS")

    def test_source_scope_excludes_target_and_reward_paths(self):
        self.assertTrue(analyzer.target_path_scope_is_untouched())
        source = (
            PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if self._r1_config.executes_topk:", source)
        self.assertIn("select_r1_execution_queries", source)
        self.assertNotIn("record_hspec_s9_shadow_events([{'event': 's10'", source)


if __name__ == "__main__":
    unittest.main()
