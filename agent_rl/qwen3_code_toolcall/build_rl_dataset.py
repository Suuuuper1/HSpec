# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import logging
from typing import List, Dict, Any
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
DATASET_NAME = "PrimeIntellect/verifiable-coding-problems"
DATASET_SPLIT = "train"
TRAIN_FILE = "train.parquet"
VAL_FILE_PARQUET = "validation.parquet"
VAL_FILE_JSONL = "validation.jsonl"
TEST_SPLIT_SIZE = 0.01
RANDOM_SEED = 42
TEST_CASES_THRESHOLD = 5  # Minimum number of test cases required


def change_test_list_format(old_tests_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts a list of individual test case dicts into a single dict
    grouping all inputs and outputs.

    Example:
    From: [{'fn_name': 'f', 'input': [1], 'output': 2}, {'fn_name': 'f', 'input': [3], 'output': 4}]
    To:   {'fn_name': 'f', 'type': 'function_call', 'input': [[1], [3]], 'output': [2, 4]}
    """
    if not old_tests_list:
        return {}

    first_test = old_tests_list[0]
    # Pre-allocate list space using list comprehension to avoid frequent memory allocation
    new_tests: Dict[str, Any] = {
        "input": [test["input"] for test in old_tests_list],
        "output": [test["output"] for test in old_tests_list],
    }

    if "fn_name" in first_test:
        new_tests["fn_name"] = first_test["fn_name"]
        new_tests["type"] = "function_call"
    else:
        new_tests["fn_name"] = None
        new_tests["type"] = "stdin_stdout"

    return new_tests


def process_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single entry from the source dataset, filtering and
    reformatting it for the target structure.
    """
    empty_data = {
        "prompt": [{"role": "user", "content": entry["prompt"]}],
        "solutions": [""],
        "reward_model": {"ground_truth": "", "style": "rule"},
        "data_source": entry["source"],
    }

    gold_standard_solution = entry.get("gold_standard_solution")
    if not (
        gold_standard_solution
        and gold_standard_solution.startswith("```python")
        and gold_standard_solution.endswith("```")
    ):
        return empty_data

    tests_str = entry.get("verification_info")
    if not isinstance(tests_str, str):
        return empty_data

    try:
        tests = ast.literal_eval(tests_str)
    except (ValueError, SyntaxError):
        try:
            tests = json.loads(tests_str)
        except json.JSONDecodeError:
            return empty_data

    if not isinstance(tests, dict) or tests.get("language") != "python":
        return empty_data

    test_cases = tests.get("test_cases", [])
    # Filter out entries with fewer than 5 test cases (len > 4 means >= 5)
    if not (
        isinstance(test_cases, list)
        and len(test_cases) >= TEST_CASES_THRESHOLD
        and "input" in test_cases[0]
        and "output" in test_cases[0]
    ):
        return empty_data

    formatted_tests = change_test_list_format(test_cases)

    return {
        "prompt": [{"role": "user", "content": entry["prompt"]}],
        "solutions": [gold_standard_solution],
        "reward_model": {"ground_truth": json.dumps(formatted_tests), "style": "rule"},
        "data_source": entry["source"],
    }


def main():
    """
    Main function to load, process, and save the dataset.
    """
    # Load the dataset
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, trust_remote_code=True)
    logger.info(f"Original dataset size: {len(ds)}")
    logger.info(f"First entry keys: {list(ds[0].keys())}")

    # Process and filter the dataset
    processed_ds = ds.map(process_entry, remove_columns=ds.column_names)
    logger.info(f"Processed dataset size (before filtering): {len(processed_ds)}")

    filtered_ds = processed_ds.filter(lambda x: x["solutions"] != [""])
    logger.info(f"Filtered dataset size: {len(filtered_ds)}")
    if len(filtered_ds) > 0:
        logger.info(f"First filtered entry:\n{filtered_ds[0]}")

    # Split the dataset into train and validation sets
    split_dataset = filtered_ds.train_test_split(
        test_size=TEST_SPLIT_SIZE, seed=RANDOM_SEED
    )
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(val_ds)}")

    # Save datasets to files
    train_ds.to_parquet(TRAIN_FILE)
    val_ds.to_parquet(VAL_FILE_PARQUET)
    val_ds.to_json(VAL_FILE_JSONL, orient="records", lines=True)
    logger.info(f"Successfully saved train set to {TRAIN_FILE}")
    logger.info(
        f"Successfully saved validation set to {VAL_FILE_PARQUET} and {VAL_FILE_JSONL}"
    )


if __name__ == "__main__":
    main()
