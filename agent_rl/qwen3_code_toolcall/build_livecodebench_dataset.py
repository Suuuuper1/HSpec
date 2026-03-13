# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Build LiveCodeBench dataset in parquet format for evaluation.

Based on the verl-recipe data processing pipeline:
https://github.com/verl-project/verl-recipe/blob/main/r1/data_process.py
"""
import argparse
import base64
import json
import os
import pickle
import zlib
from functools import partial
import logging

from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def build_livecodebench_dataset():

    def process_livecodebench(example):
        """
        Construct Query Prompt 
        From https://github.com/LiveCodeBench/LiveCodeBench/blob/\
        998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        """
        query_prompt = (
            f"You will be given a question (problem specification) and will generate a correct Python program "
            f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
        )
        if example["starter_code"]:
            query_prompt += (
                f"You will use the following starter code to write the solution to the problem and enclose your "
                f"code within delimiters.\n```python\n{example['starter_code']}\n```"
            )
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
                "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                "```python\n# YOUR CODE HERE\n```"
            )

        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception as e:
            try:
                private_test_cases = json.loads(
                    pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
                )
            except Exception as e2:
                logger.error("Failed to parse test_cases JSON: %s", e2)
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "input": [t["input"] for t in full_test_cases],
            "output": [t["output"] for t in full_test_cases],
            "type": "function_call" if metadata.get("func_name", None) else "stdin_stdout",  
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed


    def lcb_map_fn(example, idx, data_source):
        question, solution = process_livecodebench(example)
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "Code",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx},
        }
        return data

    data_source = "livecodebench/code_generation_lite"
    logger.info(f"Loading the {data_source} dataset from huggingface...")
    dataset = load_dataset(data_source, split="test")
    
    dataset = dataset.filter(lambda line: "2024-08-00T00:00:00" <= str(line["contest_date"]) < "2025-01-00T00:00:00")
    map_fn = partial(lcb_map_fn, data_source=data_source)

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/livecodebench", help="path to save the dataset")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy dataset")

    args = parser.parse_args()

    test_dataset = build_livecodebench_dataset()

    data_dir = args.data_dir
    hdfs_dir = args.hdfs_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(data_dir, "test.parquet"))

    if hdfs_dir is not None:
        if not os.path.exists(hdfs_dir):
            os.makedirs(hdfs_dir, exist_ok=True)
        copy(src=data_dir, dst=hdfs_dir)


if __name__ == "__main__":
    main()
