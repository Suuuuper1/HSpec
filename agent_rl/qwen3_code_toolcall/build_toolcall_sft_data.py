# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import logging
import re

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ds = load_dataset("Gen-Verse/Open-AgentRL-SFT-3K", split="train")

py_block = re.compile(r"```python[\s\S]*?```", re.IGNORECASE)


def is_code_trace(ex):
    msgs = ex["messages"]

    # Must contain at least one tool call from assistant messages.
    has_tool_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)

    # Must contain at least one tool response message.
    has_tool_resp = any(m.get("role") == "tool" for m in msgs)

    # The last assistant message must include a python code block and not be a boxed answer.
    last_a = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    if not last_a:
        return False
    c = last_a.get("content", "")
    has_final_python = bool(py_block.search(c))
    not_boxed = ("\\boxed" not in c)

    return has_tool_call and has_tool_resp and has_final_python and not_boxed

code_ds = ds.filter(is_code_trace)

logger.info("all: %s code_only: %s", len(ds), len(code_ds))
code_ds.to_parquet("toolcall_sft.parquet", compression="zstd")
logger.info("wrote toolcall_sft.parquet")
