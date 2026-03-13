# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 4  # Number of retries for API calls
INITIAL_RETRY_DELAY = 40
API_TIMEOUT = 10
# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_TOOL_CALL_BONUS_PER_CALL = 0.005
DEFAULT_MAX_REWARDED_TOOL_CALLS = 4


def _parse_test_cases(ground_truth):
    """
    Parse and validate test cases from ground_truth.

    Returns:
        tuple: (test_cases, error_result) where error_result is None if successful,
               or a tuple (score, metadata) if parsing failed.
    """
    test_cases = ground_truth
    if isinstance(test_cases, dict):
        if not test_cases or "input" not in test_cases or "output" not in test_cases:
            logger.error("Invalid test_cases structure.")
            logger.error("%s ...", str(test_cases)[:100])
            return None, (0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}])
        return test_cases, None

    try:
        test_cases = json.loads(test_cases)
    except json.JSONDecodeError as e1: # For LiveCodeBench compatibility
        try:
            import pickle
            import zlib
            import base64
            test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))
        except json.JSONDecodeError as e2:
            logger.error("Failed to parse test_cases JSON: %s", e)
            return None, (0.0, [{"error": "Invalid test_cases JSON format"}])

    if not test_cases or "input" not in test_cases or "output" not in test_cases:
        logger.error("Invalid test_cases structure.")
        logger.error("%s ...", str(test_cases)[:100])
        return None, (0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}])

    return test_cases, None


def _process_api_response(api_response, error_msg, test_cases, solution):
    """
    Process the sandbox API response and build metadata.

    Returns:
        dict: Metadata dictionary with score and execution details.
    """
    metadata = {
        "input": str(test_cases),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
        "score": 0.0,
    }

    if error_msg:
        metadata["status"] = "api_error"
        logger.error("Sandbox Error Report: API error occurred: %s", error_msg)
        generation_to_log = solution[:200] + "..." if len(solution) > 200 else solution
        logger.error("Sandbox Error Report: Generation: %s", generation_to_log)
        return metadata

    if not api_response:
        return metadata

    logger.debug("Sandbox Debug Report: API Response: %s", api_response)
    metadata["api_response"] = api_response
    metadata["api_status"] = api_response.get("status")

    compile_result = api_response.get("compile_result")
    if compile_result:
        metadata["compile_status"] = compile_result.get("status")
        metadata["compile_duration"] = compile_result.get("execution_time")
        metadata["compile_stderr"] = compile_result.get("stderr")

    run_result = api_response.get("run_result")
    if run_result:
        metadata["run_status"] = run_result.get("status")
        metadata["stdout"] = run_result.get("stdout")
        metadata["stderr"] = run_result.get("stderr")
        metadata["exit_code"] = run_result.get("return_code")
        metadata["duration"] = run_result.get("execution_time")

    if api_response.get("accepted", None) is True:
        metadata["status"] = "success"
        metadata["score"] = 1.0
    else:
        metadata["status"] = "wrong_answer"
        cases = api_response.get("tests", [])
        total_cases = len(cases)
        passed_cases = sum(1 for test in cases if test and test.get("passed", False))
        if total_cases > 0:
            metadata["score"] = passed_cases / total_cases

    return metadata


def _get_tool_call_count(extra_info: Optional[Dict[str, Any]]) -> int:
    if not extra_info or not isinstance(extra_info, dict):
        return 0
    if "tool_call_count" in extra_info:
        try:
            return int(extra_info["tool_call_count"])
        except (TypeError, ValueError):
            return 0
    tool_calls = extra_info.get("tool_calls")
    if isinstance(tool_calls, list):
        return len(tool_calls)
    tool_call_names = extra_info.get("tool_call_names")
    if isinstance(tool_call_names, list):
        return len(tool_call_names)
    return 0


def _compute_code_score_and_metadata(
    solution_str: str,
    ground_truth: Any,
    sandbox_fusion_url: Optional[str],
    timeout: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    # 1. Extract code and language from solution_str
    # Remove <think>.*</think> tags if they exist
    solution = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    language_str = re.search(r"```(\w+)", solution_str)
    if language_str:
        language = language_str.group(1).strip()
    else:
        # Default to Python if no language is specified
        language = "python"

    # 2. Parse test cases
    test_cases, error_result = _parse_test_cases(ground_truth)
    if error_result is not None:
        return float(error_result[0]), error_result[1]

    try:
        # 3. Call sandbox API
        api_response, error_msg = call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url,
            code=solution,
            in_outs=test_cases,
            timeout=timeout,
            language=language,
        )

        # 4. Process API response
        metadata = _process_api_response(api_response, error_msg, test_cases, solution)
        score = float(metadata.get("score", 0.0))
        final_metadata = [metadata]
        logger.info("Sandbox Info Report: Results: %s", score)

    except Exception as e:
        score = 0.0
        final_metadata = [{"error": f"Unhandled exception: {e}"}]

    return float(score), final_metadata


def _format_score_result(
    score: float,
    metadata: List[Dict[str, Any]],
    return_dict: bool,
    include_metadata: bool,
    extra_fields: Optional[Dict[str, Any]] = None,
):
    if return_dict:
        result = {"score": float(score)}
        if extra_fields:
            result.update(extra_fields)
        if include_metadata:
            result["metadata"] = json.dumps(metadata, ensure_ascii=True)
        return result
    return float(score), metadata


def _get_tool_call_bonus_config(kwargs: Dict[str, Any]) -> Tuple[float, int]:
    tool_call_bonus_per_call = kwargs.get(
        "tool_call_bonus_per_call",
        kwargs.get("tool_call_reward", DEFAULT_TOOL_CALL_BONUS_PER_CALL),
    )
    try:
        tool_call_bonus_per_call = float(tool_call_bonus_per_call)
    except (TypeError, ValueError):
        tool_call_bonus_per_call = DEFAULT_TOOL_CALL_BONUS_PER_CALL

    max_rewarded_tool_calls = kwargs.get("max_rewarded_tool_calls", DEFAULT_MAX_REWARDED_TOOL_CALLS)
    try:
        max_rewarded_tool_calls = int(max_rewarded_tool_calls)
    except (TypeError, ValueError):
        max_rewarded_tool_calls = DEFAULT_MAX_REWARDED_TOOL_CALLS
    max_rewarded_tool_calls = max(max_rewarded_tool_calls, 0)
    return tool_call_bonus_per_call, max_rewarded_tool_calls


def _compute_tool_call_bonus(
    extra_info: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Tuple[int, float, float, int]:
    tool_call_bonus_per_call, max_rewarded_tool_calls = _get_tool_call_bonus_config(kwargs)
    tool_call_count = max(_get_tool_call_count(extra_info), 0)
    rewarded_tool_call_count = min(tool_call_count, max_rewarded_tool_calls)
    tool_call_bonus = float(rewarded_tool_call_count) * tool_call_bonus_per_call
    return tool_call_count, tool_call_bonus, tool_call_bonus_per_call, max_rewarded_tool_calls


def compute_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """
    Computes training reward: code correctness score + tool-call bonus.
    """
    sandbox_fusion_url = kwargs.get("sandbox_fusion_url")
    return_dict = kwargs.get("return_dict", False)
    include_metadata = kwargs.get("include_metadata", False)
    timeout = kwargs.get("timeout", 30)

    base_score, final_metadata = _compute_code_score_and_metadata(
        solution_str=solution_str,
        ground_truth=ground_truth,
        sandbox_fusion_url=sandbox_fusion_url,
        timeout=timeout,
    )
    tool_call_count, tool_call_bonus, tool_call_bonus_per_call, max_rewarded_tool_calls = _compute_tool_call_bonus(
        extra_info=extra_info,
        kwargs=kwargs,
    )
    final_reward = float(base_score) + tool_call_bonus

    return _format_score_result(
        score=final_reward,
        metadata=final_metadata,
        return_dict=return_dict,
        include_metadata=include_metadata,
        extra_fields={
            "code_score": float(base_score),
            "base_score": float(base_score),  # backward-compatible alias
            "tool_call_count": tool_call_count,
            "tool_call_bonus": tool_call_bonus,
            "tool_call_bonus_per_call": tool_call_bonus_per_call,
            "max_rewarded_tool_calls": max_rewarded_tool_calls,
            "tool_call_reward": tool_call_bonus_per_call,  # backward-compatible alias
        },
    )


def _build_sandbox_payload(code: str, language: str, timeout: int, in_outs: Any) -> str:
    """Build JSON payload for sandbox API request."""
    return json.dumps({
        "completion": code,
        "config": {
            "language": language,
            "compile_timeout": timeout,
            "run_timeout": timeout,
            "provided_data": {"test_cases": in_outs},
            "extra": {"run_all_cases": True, "total_timeout": 30},
        },
    })


def _execute_single_request(url: str, payload: str, timeout: int) -> requests.Response:
    """Execute a single HTTP POST request to the sandbox API."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    return requests.post(url, headers=headers, data=payload, timeout=timeout)


def _try_single_api_call(url: str, payload: str, timeout: int, attempt: int, log_prefix: str):
    """
    Try a single API call attempt.

    Returns:
        tuple: (response, error) - response is not None on success,
               error starts with "RETRY:" if should retry.
    """
    try:
        logger.info("%sAttempt %d/%d: Calling sandbox API at %s",
                    log_prefix, attempt + 1, MAX_RETRIES, url)
        response = _execute_single_request(url, payload, timeout)

        if response.status_code in [429, 500, 502, 503, 504]:
            logger.warning("%sReceived status %d", log_prefix, response.status_code)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            return None, "RETRY:%sReceived status %d" % (log_prefix, response.status_code)

        response.raise_for_status()
        logger.info("%sSandbox API call successful on attempt %d", log_prefix, attempt + 1)
        return response, None

    except Exception as e:
        return None, "%sError: %s" % (log_prefix, e)


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    in_outs: Any,
    timeout: int,
    language: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the remote sandbox API to execute code and retries on specific HTTP errors.

    Args:
        sandbox_fusion_url (str): The URL of the sandbox fusion API endpoint.
        code (str): The source code to be executed.
        in_outs (any): The test cases to be used for evaluation.
        timeout (int): The timeout in seconds for compilation and execution.
        language (str): The programming language of the code.

    Returns:
        tuple: (response_json, error_message) - response_json is None on failure.
    """
    request_id = str(uuid.uuid4())
    log_prefix = "[Request ID: %s] " % request_id

    if language not in SUPPORTED_LANGUAGES:
        error_msg = "%sUnsupported language: %s" % (log_prefix, language)
        logger.error(error_msg)
        return None, error_msg

    payload = _build_sandbox_payload(code, language, timeout, in_outs)
    request_timeout = timeout * 2 + API_TIMEOUT
    last_error = None

    for attempt in range(MAX_RETRIES):
        response, last_error = _try_single_api_call(
            sandbox_fusion_url, payload, request_timeout, attempt, log_prefix
        )
        if response is not None:
            return response.json(), None
        if last_error and not last_error.startswith("RETRY:"):
            break
        if last_error:
            last_error = last_error[6:]  # Remove "RETRY:" prefix

    logger.error("%sSandbox API call failed. Last error: %s", log_prefix, last_error)
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed"
