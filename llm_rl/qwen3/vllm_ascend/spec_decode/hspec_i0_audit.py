"""Opt-in I0 correctness evidence. No file I/O or tensor copies in decode.

Bounded samples are exported by an existing rollout-boundary collective RPC.
This observer is not enabled by production profiles or performance benchmarks.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def record_weight_update(runner, loaded_count):
    """Boundary-only sampled fingerprint; never advertised as a full hash."""
    import torch
    if loaded_count == 0:
        raise RuntimeError('I0 generation cannot use unloaded dummy weights')
    values, names = [], []
    for name, param in runner.model.named_parameters():
        if param.numel() and param.is_floating_point():
            flat = param.detach().reshape(-1)
            indices = torch.linspace(0, flat.numel()-1, 8, device=flat.device).long()
            values.append(flat[indices].float())
            names.append(name)
        if len(values) == 4:
            break
    if not values:
        raise RuntimeError('I0 could not fingerprint target weights')
    runner._hspec_i0_weight_updates = getattr(runner, '_hspec_i0_weight_updates', 0) + 1
    runner._hspec_i0_weight_fingerprint = digest([names, torch.cat(values).cpu().tolist()])


def record_trainer_event(kind, step, epoch, **details):
    """Small control-plane record after an operation has actually returned."""
    run_dir = os.environ.get('HSPEC_I0_RUN_DIR')
    if not run_dir:
        raise ValueError('I0 trainer evidence requires an isolated run directory')
    atomic_json(Path(run_dir) / 'trainer_events' / f'{kind}_e{epoch}_s{step}.json',
                {'schema_version': 'hspec.i0.trainer-event.v1', 'kind': kind,
                 'step': int(step), 'epoch': int(epoch), **details})


def config_values(config):
    """Snapshot existing values without revalidating runtime dataclass types."""
    from dataclasses import fields, is_dataclass
    from collections.abc import Mapping
    from enum import Enum
    from omegaconf import OmegaConf
    if OmegaConf.is_config(config):
        return config_values(OmegaConf.to_container(config, resolve=True))
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: config_values(getattr(config, field.name)) for field in fields(config)}
    if isinstance(config, Mapping):
        return {key: config_values(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_values(value) for value in config]
    if isinstance(config, Enum):
        return config_values(config.value)
    if isinstance(config, Path):
        return str(config)
    if config is None or isinstance(config, (str, bool, int, float)):
        return config
    raise TypeError(f'Unsupported I0 config value type: {type(config).__qualname__}')


def write_rollout_evidence(engine, resolved, config, non_tensor_batch, *, step, epoch, rank, validate, ordinal):
    """Control plane only, after flush/finalize. Failure must be visible in I0."""
    from omegaconf import OmegaConf
    run_dir = os.environ.get('HSPEC_I0_RUN_DIR', '')
    if not run_dir:
        raise ValueError('HSPEC_I0_AUDIT requires HSPEC_I0_RUN_DIR')
    workers = engine.llm_engine.collective_rpc('hspec_i0_runtime_snapshot', args=())
    descriptors = non_tensor_batch.get('hspec_desc', [])
    payload = {'schema_version': 'hspec.i0.rollout.v1', 'step': step, 'epoch': epoch,
               'rollout_rank': rank, 'validate': bool(validate), 'ordinal': ordinal,
               'resolved_speculation': resolved.manifest,
               'rollout_config': config_values(config),
               'descriptor_count': sum(d is not None for d in descriptors),
               'workers': workers}
    atomic_json(Path(run_dir) / 'runtime' / f'rank{rank}_round{ordinal}.json', payload)


class HSpecI0Audit:
    def __init__(self):
        self.counts = Counter()
        self.shapes = Counter()
        self.samples = {'ar': [], 'spec': []}
        self.chain = {'ar': '0' * 64, 'spec': '0' * 64}

    def shape(self, mode, logical, padded, requests):
        key = (str(mode), int(logical), int(padded), int(requests))
        if key in self.shapes or len(self.shapes) < 512:
            self.shapes[key] += 1
        else:
            self.counts['shape_overflow'] += 1

    def outcome(self, request_ids, scheduled, emitted, accepted):
        paths = [list(scheduled.get(r, [])) for r in request_ids]
        kind = 'spec' if any(paths) else 'ar'
        self.counts[kind + '_batches'] += 1
        self.counts['output_opportunities'] += sum(bool(ids) for ids in emitted)
        self.counts['matched_proposals'] += sum(int(a)>0 for a in accepted)
        self.counts['scheduled_tokens'] += sum(map(len, paths))
        self.counts['emitted_tokens'] += sum(map(len, emitted))
        self.counts['accepted_tokens'] += sum(accepted)
        for path, out, a in zip(paths, emitted, accepted):
            if not 0 <= a <= min(len(path), len(out)):
                raise RuntimeError('I0 accepted/scheduled/emitted conservation failed')
            if path:
                self.counts['verified_proposals'] += 1
                self.counts['verified_length_' + str(len(path))] += 1
        payload = {'requests': list(request_ids), 'scheduled': paths}
        # InputProcessor appends a local 8-hex nonce to external request IDs.
        # Independent external-launcher engines need the shared external ID.
        stable_ids = []
        for request in request_ids:
            prefix, sep, nonce = str(request).rpartition('-')
            stable_ids.append(prefix if sep and len(nonce) == 8
                              and all(c in '0123456789abcdef' for c in nonce) else str(request))
        self.chain[kind] = digest([self.chain[kind], {'requests': stable_ids, 'scheduled': paths}])
        if len(self.samples[kind]) < 16:
            self.samples[kind].append({**payload, 'emitted': [list(x) for x in emitted],
                                       'accepted': list(accepted)})

    def snapshot(self):
        return {'counts': dict(self.counts), 'draft_chain': dict(self.chain),
                'samples': self.samples,
                'shapes': [{'mode': k[0], 'logical': k[1], 'padded': k[2],
                            'requests': k[3], 'calls': v} for k,v in sorted(self.shapes.items())]}


def runtime_snapshot(runner):
    """Boundary only. Captures actual runner values and code identity."""
    import torch
    import vllm.v1.sample.rejection_sampler as upstream
    import vllm_ascend.sample.rejection_sampler as ascend
    from vllm.distributed.parallel_state import get_tp_group
    spec = runner.speculative_config
    drafter = runner.drafter
    survival = getattr(drafter, '_survival_config', None)
    group = get_tp_group()
    sources = [Path(ascend.__file__), Path(__file__),
               Path(__file__).parents[1] / 'worker/model_runner_v1.py',
               Path(__file__).parents[1] / 'ops/triton/reject_sample.py']
    return {
        'schema_version': 'hspec.i0.worker.v1',
        'rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        'tp_ranks': list(group.ranks), 'tp_rank': int(group.rank_in_group),
        'method': spec.method if spec else None,
        'requested_verifier': spec.rejection_sample_method if spec else None,
        'actual_verifier': getattr(getattr(runner, 'rejection_sampler', None), 'verifier_mode', None) if spec else None,
        'patched_dispatch': upstream.rejection_sample is ascend.rejection_sample,
        'has_triton': bool(ascend.HAS_TRITON),
        'collection_capable': bool(runner._hspec_collect),
        'collect_only': bool((runner.vllm_config.additional_config or {}).get('hspec_collect_only', False)),
        'async_scheduling': bool(runner.use_async_scheduling),
        'drafter': type(drafter).__name__ if drafter else None,
        'selector_mode': getattr(drafter, '_selector_mode', None),
        'survival_enabled': bool(getattr(survival, 'enabled', False)),
        'survival_executes': bool(getattr(survival, 'executes_utility', False)),
        'survival_fallback': getattr(survival, 'fallback_reason', None),
        'i4': {'mode': getattr(getattr(drafter, '_i4_selector', None), 'mode', 'off'),
               'init_error': getattr(drafter, '_i4_init_error', None),
               'model_version': getattr(getattr(getattr(drafter, '_i4_selector', None), 'scorer', None), 'model_version', None)},
        'pending_verify': len(getattr(drafter, '_pending_verify_meta', {})),
        'probability_cache_enabled': bool(getattr(runner._draft_probability_cache, 'enabled', False)),
        'model': runner.model_config.model,
        'anchor': {'stage': 'final_post_norm_hidden',
                   'layers': getattr(runner.model_config.hf_config, 'num_hidden_layers', None),
                   'hidden_dim': getattr(runner.model_config.hf_config, 'hidden_size', None)},
        'weight_updates': getattr(runner, '_hspec_i0_weight_updates', 0),
        'sampled_weight_fingerprint': getattr(runner, '_hspec_i0_weight_fingerprint', None),
        'proposer_queries': int(getattr(drafter, '_stat_queries', 0)),
        'proposer_hits': int(getattr(drafter, '_stat_hits', 0)),
        'proposer_draft_tokens': int(getattr(drafter, '_stat_total_draft_len', 0)),
        'dtype': str(runner.model_config.dtype),
        'enforce_eager': bool(runner.model_config.enforce_eager),
        'graph_config': str(runner.compilation_config.cudagraph_mode),
        'parallel_config': {'tp': runner.parallel_config.tensor_parallel_size,
                            'pp': runner.parallel_config.pipeline_parallel_size,
                            'dp': runner.parallel_config.data_parallel_size,
                            'ep': runner.parallel_config.enable_expert_parallel},
        'device_name': torch.npu.get_device_name(torch.npu.current_device()),
        'environment': {k:v for k,v in sorted(os.environ.items())
                        if k.startswith(('HSPEC_', 'VLLM_', 'HCCL_', 'ASCEND_', 'OMP_', 'MKL_'))
                        or k in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'VERL_LOGGING_LEVEL',
                                 'HYDRA_FULL_ERROR', 'RAY_DEDUP_LOGS', 'NODE_RANK', 'MATCH_WND',
                                 'VERL_VLLM_CUDAGRAPH_MODE')},
        'source_sha256': {str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        'audit': runner._hspec_i0_audit.snapshot() if runner._hspec_i0_audit else None,
    }
