"""Streaming expert coverage and in-place Ascend actor-weight reload.

Checks run at weight-update boundaries, never in decode. No full-model copy or
per-expert device synchronization is needed to validate names and shapes.
"""
from contextlib import contextmanager
import re


class ExpertWeightCoverage:
    _name = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

    def __init__(self, model, hidden_size, intermediate_size):
        self.hidden = hidden_size
        self.intermediate = intermediate_size
        self.owned = {}
        self.required = set()
        self.seen = set()
        self.remote_empty = 0
        for prefix, module in model.named_modules():
            if not (hasattr(module, 'w13_weight') and hasattr(module, 'w2_weight')):
                continue
            n = int(module.global_num_experts)
            mapping = getattr(module, '_expert_map', None)
            ids = set(range(n)) if mapping is None else {
                i for i, value in enumerate(mapping.cpu().tolist()) if value >= 0}
            self.owned[prefix] = ids
            self.required.update((prefix, i, projection) for i in ids
                                 for projection in ('gate_proj', 'up_proj', 'down_proj'))

    def wrap(self, weights):
        for name, tensor in weights:
            match = self._name.fullmatch(name)
            if match is None:
                yield name, tensor
                continue
            prefix, expert, projection = match.groups()
            expert = int(expert)
            if prefix not in self.owned:
                raise ValueError(f'actor expert has no target module: {name}')
            local = expert in self.owned[prefix]
            if tensor.numel() == 0:
                if local:
                    raise ValueError(f'missing local expert weight: {name}; EP export and target layout differ')
                self.remote_empty += 1
                continue
            if local:
                key = (prefix, expert, projection)
                if key in self.seen:
                    raise ValueError(f'duplicate local expert weight: {name}')
                expected = ((self.hidden, self.intermediate) if projection == 'down_proj'
                            else (self.intermediate, self.hidden))
                if tuple(tensor.shape) != expected:
                    raise ValueError(f'actor expert shape mismatch: {name}: {tuple(tensor.shape)} != {expected}')
                self.seen.add(key)
            yield name, tensor

    def finish(self):
        missing = self.required - self.seen
        if missing:
            raise ValueError(f'actor weight synchronization missed {len(missing)} local expert shards; '
                             f'examples={sorted(missing)[:3]}')
        return {'expected_expert_shards': len(self.required),
                'loaded_expert_shards': len(self.seen), 'remote_empty_shards': self.remote_empty}


@contextmanager
def ascend_moe_reload_views(model, hidden_size):
    """Restore loader views without moving the storage used by captured graphs.

Ascend's unquantized postprocessor transposes/allocates weights. Calling it a
second time after actor reload would transpose again and change graph pointers.
Only already-processed Ascend unquantized MoE modules use this path; other
quantizers and first-time loader layouts keep the native postprocessor.
"""
    saved = []
    methods = []
    seen_methods = set()
    try:
        for _, module in model.named_modules():
            qm = getattr(module, 'quant_method', None)
            if qm is None or type(qm).__name__ != 'AscendUnquantizedFusedMoEMethod':
                continue
            w13, w2 = module.w13_weight, module.w2_weight
            if w13.ndim != 3 or w2.ndim != 3:
                raise ValueError('Ascend MoE reload requires rank-3 weights')
            runtime = w13.shape[1] == hidden_size and w2.shape[2] == hidden_size
            if not runtime:
                if w13.shape[2] != hidden_size or w2.shape[1] != hidden_size:
                    raise ValueError('unknown Ascend MoE weight layout')
                continue
            for param in (w13, w2):
                data = param.data
                saved.append((param, data))
                param.data = data.transpose(1, 2)
            if id(qm) not in seen_methods:
                seen_methods.add(id(qm))
                methods.append((qm, qm.process_weights_after_loading))
                qm.process_weights_after_loading = lambda layer: None

        def restore():
            for param, data in saved:
                param.data = data

        yield restore
    finally:
        for param, data in saved:
            param.data = data
        for qm, method in methods:
            qm.process_weights_after_loading = method
