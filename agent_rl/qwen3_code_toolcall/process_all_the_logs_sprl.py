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

import re
import csv
import glob
import logging
import argparse
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

PATH_PATTERN = re.compile(
    r"'path':\s*'[^']*checkpoint/([^/]+)/global_step_(\d+)"
)
STEP_PATTERN = re.compile(r'step:(\d+)')

# Validation patterns (dict-style log lines)
VAL_PATTERNS_DICT = {
    'val_code_contests_reward': re.compile(
        r"'val-(?:core|aux)/code_contests/(?:reward|score)/mean@1':"
        r"\s*([\d.e\-+]+)"
    ),
    'val_apps_reward': re.compile(
        r"'val-(?:core|aux)/apps/(?:reward|score)/mean@1':\s*([\d.e\-+]+)"
    ),
    'val_taco_reward': re.compile(
        r"'val-(?:core|aux)/taco/(?:reward|score)/mean@1':\s*([\d.e\-+]+)"
    ),
    'val_num_turns_min': re.compile(r"'val-aux/num_turns/min':\s*(\d+)"),
    'val_num_turns_max': re.compile(r"'val-aux/num_turns/max':\s*(\d+)"),
    'val_num_turns_mean': re.compile(
        r"'val-aux/num_turns/mean':\s*([\d.e\-+]+)"
    ),
}

# Validation patterns (colon-style log lines)
VAL_PATTERNS_COLON = {
    'val_code_contests_reward': re.compile(
        r'val-core/code_contests/reward/mean@1:([\d.e\-+]+)'
    ),
    'val_apps_reward': re.compile(
        r'val-core/apps/reward/mean@1:([\d.e\-+]+)'
    ),
    'val_taco_reward': re.compile(
        r'val-core/taco/reward/mean@1:([\d.e\-+]+)'
    ),
    'val_num_turns_min': re.compile(r'val-aux/num_turns/min:(\d+)'),
    'val_num_turns_max': re.compile(r'val-aux/num_turns/max:(\d+)'),
    'val_num_turns_mean': re.compile(
        r'val-aux/num_turns/mean:([\d.e\-+]+)'
    ),
}

# Training metrics patterns
TRAIN_PATTERNS = {
    'critic_score_mean': re.compile(
        r'critic/score/mean:([\d.e\-+]+)'
    ),
    'prompt_length_mean': re.compile(
        r'prompt_length/mean:([\d.e\-+]+)'
    ),
    'response_length_mean': re.compile(
        r'response_length/mean:([\d.e\-+]+)'
    ),
    'response_length_non_aborted_mean': re.compile(
        r'response_length_non_aborted/mean:([\d.e\-+]+)'
    ),
    'response_length_clip_ratio': re.compile(
        r'response_length/clip_ratio:([\d.e\-+]+)'
    ),
    'num_turns_mean': re.compile(
        r'(?:val-aux/)?num_turns/mean:([\d.e\-+]+)'
    ),
    'num_turns_min': re.compile(
        r'(?:val-aux/)?num_turns/min:([\d.e\-+]+)'
    ),
    'num_turns_max': re.compile(
        r'(?:val-aux/)?num_turns/max:([\d.e\-+]+)'
    ),
    'time_per_step': re.compile(
        r'perf/time_per_step:([\d.e\-+]+)'
    ),
    'throughput_tokens_per_s': re.compile(
        r'perf/throughput:([\d.e\-+]+)'
    ),
    'entropy': re.compile(
        r'actor/entropy:([\d.e\-+]+)'
    ),
    'timing_s_step': re.compile(
        r'timing_s/step:([\d.e\-+]+)'
    ),
    'timing_s_gen': re.compile(
        r'timing_s/gen:([\d.e\-+]+)'
    ),
    'timing_s_update_actor': re.compile(
        r'timing_s/update_actor:([\d.e\-+]+)'
    ),
    'timing_s_ref': re.compile(
        r'timing_s/ref:([\d.e\-+]+)'
    ),
    'timing_s_old_log_prob': re.compile(
        r'timing_s/old_log_prob:([\d.e\-+]+)'
    ),
    'timing_s_adv': re.compile(
        r'timing_s/adv:([\d.e\-+]+)'
    ),
    'timing_s_reward': re.compile(
        r'timing_s/reward:([\d.e\-+]+)'
    ),
}

# Speculative decoding patterns
SPEC_PATTERNS = {
    'spec_decode_num_drafts': re.compile(
        r'spec_decode/num_drafts:([\d.e\-+]+)'
    ),
    'spec_decode_num_draft_tokens': re.compile(
        r'spec_decode/num_draft_tokens:([\d.e\-+]+)'
    ),
    'spec_decode_num_accepted_tokens': re.compile(
        r'spec_decode/num_accepted_tokens:([\d.e\-+]+)'
    ),
    'spec_decode_draft_acceptance_rate': re.compile(
        r'spec_decode/draft_acceptance_rate:([\d.e\-+]+)'
    ),
    'spec_decode_mean_acceptance_length': re.compile(
        r'spec_decode/mean_acceptance_length:([\d.e\-+]+)'
    ),
}
for _i in range(5):
    SPEC_PATTERNS[f'spec_decode_acceptance_rate_pos_{_i}'] = re.compile(
        rf'spec_decode/acceptance_rate_pos_{_i}:([\d.e\-+]+)'
    )

# ---------------------------------------------------------------------------
# CSV field list
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    'configuration', 'global_step', 'rl_step',
    'critic_score_mean', 'entropy',
    'prompt_length_mean', 'response_length_mean',
    'response_length_non_aborted_mean',
    'response_length_clip_ratio',
    'num_turns_mean', 'num_turns_min', 'num_turns_max',
    'time_per_step', 'throughput_tokens_per_s',
    'timing_s_step', 'timing_s_gen', 'timing_s_update_actor',
    'timing_s_ref', 'timing_s_old_log_prob', 'timing_s_adv',
    'timing_s_reward',
    'spec_decode_num_drafts', 'spec_decode_num_draft_tokens',
    'spec_decode_num_accepted_tokens',
    'spec_decode_draft_acceptance_rate',
    'spec_decode_mean_acceptance_length',
    'spec_decode_acceptance_rate_pos_0',
    'spec_decode_acceptance_rate_pos_1',
    'spec_decode_acceptance_rate_pos_2',
    'spec_decode_acceptance_rate_pos_3',
    'spec_decode_acceptance_rate_pos_4',
]

VAL_FIELDNAMES = [
    'configuration',
    'val_code_contests_reward', 'val_apps_reward', 'val_taco_reward',
    'val_overall_avg_reward',
    'val_num_turns_min', 'val_num_turns_max', 'val_num_turns_mean',
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _apply_patterns(line, patterns):
    """Apply a dict of compiled patterns to a line, return matched values."""
    result = {}
    for key, pattern in patterns.items():
        match = pattern.search(line)
        if match:
            result[key] = match.group(1)
    return result


def _cast_train_metrics(raw):
    """Cast raw string values from TRAIN_PATTERNS to int/float."""
    metrics = {}
    int_keys = {'num_turns_min', 'num_turns_max'}
    for key, val in raw.items():
        metrics[key] = int(val) if key in int_keys else float(val)
    return metrics


def _cast_val_metrics(raw, int_keys=None):
    """Cast raw string values from validation patterns to int/float."""
    if int_keys is None:
        int_keys = {'val_num_turns_min', 'val_num_turns_max'}
    metrics = {}
    for key, val in raw.items():
        metrics[key] = int(val) if key in int_keys else float(val)
    return metrics


def _parse_path_line(line):
    """Extract configuration and global_step from a path line."""
    match = PATH_PATTERN.search(line)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def _parse_spec_metrics(line):
    """Extract speculative decoding metrics from a line."""
    raw = _apply_patterns(line, SPEC_PATTERNS)
    return {k: float(v) for k, v in raw.items()}


def _parse_train_metrics(line):
    """Extract training metrics from a line."""
    raw = _apply_patterns(line, TRAIN_PATTERNS)
    return _cast_train_metrics(raw)

# ---------------------------------------------------------------------------
# Step-0 validation extraction
# ---------------------------------------------------------------------------


def _extract_val_from_line(line, patterns):
    """Extract validation metrics from a single line using given patterns."""
    raw = _apply_patterns(line, patterns)
    return _cast_val_metrics(raw) if raw else None


def extract_step0_validation_metrics(log_file):
    """Extract initial validation metrics from the step:0 line."""
    with open(log_file, 'r') as fh:
        for line in fh:
            if 'step:0' not in line:
                continue
            if 'val-core/code_contests/reward/mean@1' not in line:
                continue
            return _extract_val_from_line(line, VAL_PATTERNS_COLON)
    return None


# ---------------------------------------------------------------------------
# Per-line metric parsing
# ---------------------------------------------------------------------------


def _handle_path_line(line, config_name, state):
    """Update parser state when a path line is encountered."""
    config, global_step = _parse_path_line(line)
    if config is not None:
        if not config_name:
            state['current_config'] = config
        state['current_global_step'] = global_step
    return True


def _handle_init_validation_line(line, config_name, state, init_validation_data):
    """Update init_validation_data when an initial validation line is found."""
    config = config_name or state.get('current_config')
    if not config:
        return
    raw = _apply_patterns(line, VAL_PATTERNS_DICT)
    if not raw:
        return
    metrics = _cast_val_metrics(raw)
    if config in init_validation_data:
        init_validation_data[config].update(metrics)
    else:
        init_validation_data[config] = metrics


def _handle_rl_step_line(line, config_name, state, data):
    """Append RL-step metrics to data when a training step line is found."""
    step_match = STEP_PATTERN.search(line)
    if not step_match:
        return

    config = config_name or state.get('current_config') or 'baseline'
    global_step = state.get('current_global_step') or 0

    metrics = {'rl_step': int(step_match.group(1))}
    metrics.update(_parse_train_metrics(line))
    metrics.update(_parse_spec_metrics(line))
    metrics['configuration'] = config
    metrics['global_step'] = global_step

    data[config][global_step].append(metrics)

# ---------------------------------------------------------------------------
# organize_metrics
# ---------------------------------------------------------------------------


def organize_metrics(log_file, config_name=None):
    """Organize metrics by configuration and global step."""
    data = defaultdict(lambda: defaultdict(list))
    init_validation_data = {}
    state = {'current_config': config_name, 'current_global_step': None}

    step0_validation = extract_step0_validation_metrics(log_file)
    if step0_validation:
        cfg = config_name or 'baseline'
        init_validation_data[cfg] = step0_validation

    with open(log_file, 'r') as fh:
        for line in fh:
            if PATH_PATTERN.search(line):
                _handle_path_line(line, config_name, state)
            elif 'Initial validation metrics:' in line:
                _handle_init_validation_line(
                    line, config_name, state, init_validation_data
                )
            else:
                _handle_rl_step_line(line, config_name, state, data)

    return data, init_validation_data


# ---------------------------------------------------------------------------
# process_all_logs
# ---------------------------------------------------------------------------


def process_all_logs(log_dir, config_name=None):
    """Process all log files in a directory and combine their data."""
    all_data = defaultdict(lambda: defaultdict(list))
    all_init_validation = {}

    log_files = sorted(glob.glob(str(Path(log_dir) / '*.log')))

    if not log_files:
        logger.warning("No .log files found in %s", log_dir)
        return all_data, all_init_validation

    logger.info("Found %d log files:", len(log_files))
    for log_file in log_files:
        logger.info("  - %s", Path(log_file).name)

    for log_file in log_files:
        logger.info("Processing: %s", Path(log_file).name)
        file_data, init_val_data = organize_metrics(log_file, config_name)

        for config in file_data:
            for global_step in file_data[config]:
                all_data[config][global_step].extend(
                    file_data[config][global_step]
                )

        all_init_validation.update(init_val_data)

    return all_data, all_init_validation


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_to_csv(data, output_file):
    """Write organized metrics to CSV file."""
    rows = []
    for config in sorted(data.keys()):
        for global_step in sorted(data[config].keys()):
            for metrics in sorted(
                data[config][global_step],
                key=lambda x: x.get('rl_step', 0)
            ):
                row = {field: metrics.get(field, '') for field in CSV_FIELDNAMES}
                row['configuration'] = config
                row['global_step'] = global_step
                rows.append(row)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Successfully wrote %d rows to %s", len(rows), output_file)


def write_init_validation_to_csv(init_validation_data, output_file, mode='w'):
    """Write initial validation metrics to CSV file."""
    rows = []
    for config in sorted(init_validation_data.keys()):
        metrics = init_validation_data[config]
        reward_keys = (
            'val_code_contests_reward', 'val_apps_reward', 'val_taco_reward'
        )
        rewards = []
        for k in reward_keys:
            if k in metrics:
                rewards.append(metrics[k])
        overall_avg = sum(rewards) / len(rewards) if rewards else ''
        row = {field: metrics.get(field, '') for field in VAL_FIELDNAMES}
        row['configuration'] = config
        row['val_overall_avg_reward'] = overall_avg
        rows.append(row)

    file_exists = Path(output_file).exists()
    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=VAL_FIELDNAMES)
        if not file_exists or mode == 'w':
            writer.writeheader()
        writer.writerows(rows)

    action = 'Appended' if mode == 'a' else 'Wrote'
    logger.info("%s %d initial validation row(s) to %s", action, len(rows), output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Extract RL training metrics from all log files in a directory '
            'and combine into one CSV'
        )
    )
    parser.add_argument(
        'log_dir', type=str,
        help='Directory containing log files (will process all .log files)'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='combined_rl_metrics.csv',
        help='Output CSV file name (default: combined_rl_metrics.csv)'
    )
    parser.add_argument(
        '-c', '--config', type=str, default=None,
        help='Configuration name (optional, extracted from logs if not provided)'
    )
    parser.add_argument(
        '--validation-only', action='store_true',
        help='Only extract and write initial validation metrics'
    )
    parser.add_argument(
        '--validation-output', type=str, default='init_validation_metrics.csv',
        help='Output CSV for validation metrics (default: init_validation_metrics.csv)'
    )
    parser.add_argument(
        '--append', action='store_true',
        help='Append to existing validation CSV instead of overwriting'
    )
    return parser


def _run_validation_only(args, init_validation_data):
    """Handle --validation-only execution path."""
    if not init_validation_data:
        logger.warning("No initial validation data found in log files.")
        return
    write_mode = 'a' if args.append else 'w'
    logger.info("Writing initial validation metrics to: %s", args.validation_output)
    write_init_validation_to_csv(
        init_validation_data, args.validation_output, mode=write_mode
    )


def main():
    args = _build_arg_parser().parse_args()

    logger.info("Processing all log files in: %s", args.log_dir)
    data, init_validation_data = process_all_logs(args.log_dir, args.config)

    if args.validation_only:
        _run_validation_only(args, init_validation_data)
        return

    if data:
        logger.info("Writing combined RL training results to: %s", args.output)
        write_to_csv(data, args.output)
    else:
        logger.warning("No RL training data extracted from log files.")

    if init_validation_data:
        logger.info("Writing initial validation metrics to: %s", args.validation_output)
        write_init_validation_to_csv(
            init_validation_data, args.validation_output, mode='w'
        )


if __name__ == '__main__':
    main()