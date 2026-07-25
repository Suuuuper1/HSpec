import ast
import copy
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROPOSER_PATH = PROJECT_ROOT / "vllm_ascend/spec_decode/hspec_proposer.py"
SCRIPT_PATHS = (
    PROJECT_ROOT / "scripts/train_grpo_qwen3_30b_hspec.sh",
    PROJECT_ROOT / "scripts/train_grpo_qwen3_30b_hspec_gsm8k.sh",
)

SCRIPT_DEFAULTS = {
    "HSPEC_PROPOSER_CACHE_MAX_PROMPTS": 512,
    "HSPEC_PROPOSER_CACHE_MAX_CPU_BYTES": 2 * 1024**3,
    "HSPEC_PROPOSER_CACHE_MAX_NPU_BYTES": 0,
    "HSPEC_PROPOSER_CACHE_MAX_ENTRIES": 0,
    "HSPEC_PROPOSER_MAX_PROMPT_CPU_BYTES": 64 * 1024**2,
    "HSPEC_PROPOSER_MAX_PROMPT_ENTRIES": 160_000,
    "HSPEC_PROPOSER_MAX_PROMPT_TOKEN_BYTES": 0,
    "HSPEC_MAX_READY_PREFETCH_MATERIALIZE": 0,
    "HSPEC_MAX_READY_PREFETCH_BYTES": 512 * 1024**2,
    "HSPEC_PROPOSER_PREFETCH_WINDOW_MAX_BYTES": 2 * 1024**3,
    "HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES": 768 * 1024**2,
    "HSPEC_PROPOSER_BATCH_CACHE_MAX_TOTAL_ENTRIES": 0,
    "HSPEC_PROPOSER_BATCH_CACHE_MAX_BMM_ELEMS": 192 * 1024**2,
    "HSPEC_PROPOSER_PREFIX_CACHE": 0,
    "HSPEC_PROPOSER_STORE_PER_PROMPT_NPU": 0,
}


def _eval_integer(node: ast.AST, constants: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_integer(node.left, constants)
        right = _eval_integer(node.right, constants)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
    raise AssertionError(f"unsupported integer expression: {ast.dump(node)}")


def _extract_proposer_env_defaults() -> dict[str, int]:
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    constants: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("_DEFAULT_"):
            constants[target.id] = _eval_integer(node.value, constants)

    defaults: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_get_env_int":
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        try:
            default = _eval_integer(node.args[1], constants)
        except (AssertionError, KeyError):
            continue
        defaults[str(node.args[0].value)] = default
    return defaults


def _extract_proposer_method(method_name: str):
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "HSpecProposer":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                function = copy.deepcopy(item)
                function.decorator_list = []
                module = ast.fix_missing_locations(
                    ast.Module(body=[function], type_ignores=[])
                )
                namespace = {
                    "List": List,
                    "_CachedPromptTable": object,
                    "torch": SimpleNamespace(dtype=object),
                }
                exec(compile(module, str(PROPOSER_PATH), "exec"), namespace)
                return namespace[method_name]
    raise AssertionError(f"missing HSpecProposer.{method_name}")


class TestHSpecCapacityConfiguration(unittest.TestCase):

    def test_framework_defaults_keep_cpu_tables_and_bound_npu_workspace(self):
        defaults = _extract_proposer_env_defaults()
        expected = {
            "HSPEC_MAX_READY_PREFETCH_BYTES": 512 * 1024**2,
            "HSPEC_PROPOSER_MAX_PROMPT_CPU_BYTES": 64 * 1024**2,
            "HSPEC_PROPOSER_MAX_PROMPT_ENTRIES": 160_000,
            "HSPEC_PROPOSER_MAX_PROMPT_TOKEN_BYTES": 0,
            "HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES": 768 * 1024**2,
            "HSPEC_PROPOSER_BATCH_CACHE_MAX_TOTAL_ENTRIES": 0,
            "HSPEC_PROPOSER_BATCH_CACHE_MAX_BMM_ELEMS": 192 * 1024**2,
        }
        self.assertEqual({name: defaults[name] for name in expected}, expected)

    def test_30b_scripts_export_log_and_forward_every_capacity_knob(self):
        for path in SCRIPT_PATHS:
            source = path.read_text(encoding="utf-8")
            with self.subTest(script=path.name):
                for name, expected in SCRIPT_DEFAULTS.items():
                    export_pattern = re.compile(
                        rf'^export {name}="\$\{{{name}:-([0-9]+)\}}"$',
                        re.MULTILINE,
                    )
                    match = export_pattern.search(source)
                    self.assertIsNotNone(match, f"missing export for {name}")
                    self.assertEqual(int(match.group(1)), expected)
                    self.assertIn(
                        f"ray_init.runtime_env.env_vars.{name}=",
                        source,
                        f"{name} is not forwarded to Ray workers",
                    )
                    self.assertRegex(
                        source,
                        rf'echo "[^"]+=\$\{{{name}\}}"',
                        f"{name} is not captured in the run manifest",
                    )
                self.assertIn(
                    "actor_rollout_ref.rollout.hspec_max_entries_per_prompt=160000",
                    source,
                )
                self.assertIn(
                    'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"',
                    source,
                )
                self.assertIn(
                    'echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"',
                    source,
                )

    def test_30b_capacity_preserves_sampler_headroom(self):
        rows = 64
        entries = 8 * (16_384 - 1)
        hidden_dim = 2048
        components = 64
        fp32_bytes = 4

        workspace_bytes = rows * (
            hidden_dim * fp32_bytes
            + hidden_dim * components * fp32_bytes
            + entries * components * fp32_bytes
            + 8
            + entries
        )
        bmm_elements = rows * entries * components

        self.assertLessEqual(
            entries,
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_MAX_PROMPT_ENTRIES"],
        )
        # The failed configuration admitted this 2.04 GiB full workspace.
        # Physical NPU guards must deliberately select a subset instead.
        self.assertGreater(
            workspace_bytes,
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES"],
        )
        self.assertGreater(
            bmm_elements,
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_BATCH_CACHE_MAX_BMM_ELEMS"],
        )

        per_row_bytes = workspace_bytes // rows
        rows_by_bytes = (
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES"]
            // per_row_bytes
        )
        rows_by_bmm = (
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_BATCH_CACHE_MAX_BMM_ELEMS"]
            // (entries * components)
        )
        selected_rows = min(rows, rows_by_bytes, rows_by_bmm)
        self.assertEqual(selected_rows, 23)
        self.assertGreaterEqual(selected_rows, 8)
        self.assertLessEqual(
            selected_rows * per_row_bytes,
            SCRIPT_DEFAULTS["HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES"],
        )

    def test_entry_sum_is_disabled_but_physical_guards_select_safe_subset(self):
        method = _extract_proposer_method("_select_batch_cache_subset")
        entries = 8 * (16_384 - 1)
        observer = SimpleNamespace(
            _batch_cache_max_total_entries=0,
            _batch_cache_max_bmm_elems=SCRIPT_DEFAULTS[
                "HSPEC_PROPOSER_BATCH_CACHE_MAX_BMM_ELEMS"
            ],
            _batch_cache_max_npu_bytes=SCRIPT_DEFAULTS[
                "HSPEC_PROPOSER_BATCH_CACHE_MAX_NPU_BYTES"
            ],
        )
        observer._estimate_batched_table_cache_nbytes_from_dims = (
            lambda *, num_rows, hidden_dim, m_max, k_max, dtype: num_rows
            * (hidden_dim * 4 + hidden_dim * k_max * 4 + m_max * k_max * 4 + 8 + m_max)
        )
        observer._estimate_batched_table_cache_bmm_elems = (
            lambda *, num_rows, m_max, k_max: num_rows * m_max * k_max
        )
        tables = [
            SimpleNamespace(
                n_entries=entries,
                mean_cpu=SimpleNamespace(shape=(2048,)),
                components_t_cpu=SimpleNamespace(shape=(2048, 64)),
            )
            for _ in range(64)
        ]

        selected_indices, selected_tables, total_entries, bmm_elements = method(
            observer,
            list(range(64)),
            tables,
            object(),
        )

        expected_rows = 23
        self.assertEqual(selected_indices, list(range(expected_rows)))
        self.assertEqual(selected_tables, tables[:expected_rows])
        self.assertEqual(total_entries, expected_rows * entries)
        self.assertEqual(bmm_elements, expected_rows * entries * 64)


if __name__ == "__main__":
    unittest.main()
