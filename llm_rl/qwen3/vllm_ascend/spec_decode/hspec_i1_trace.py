"""I1 opt-in observation, independent of selector and verifier randomness.

Only bounded CPU snapshots enter the queue. Serialization, hashes and writes
run on the writer thread; a rollout boundary seals the stream. No target
forward or RPC is introduced in decode. The sampled arm adds one packed
async probability gather/copy per verified batch, billed in the manifest.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue, Full
import random
from threading import Thread, Lock
import time
import uuid

SCHEMA = "hspec.i1.query.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class TraceWriter:
    """Whole-record backpressure: never turn a lost outcome into a negative."""
    def __init__(self, directory, capacity=256, max_records=512):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = Queue(maxsize=capacity)
        self.capacity = capacity
        self.max_records = max_records
        self.queued_records = self.peak_records = 0
        self.record_lock = Lock()
        self.dropped = 0
        self.written = 0
        self.bytes = 0
        self.error = None
        self.closed = False
        self.thread = Thread(target=self._write, name="hspec-i1-writer", daemon=True)
        self.thread.start()

    def _write(self):
        try:
            with (self.directory / "queries.jsonl.partial").open("w") as stream:
                while True:
                    batch = self.queue.get()
                    try:
                        if batch is None:
                            break
                        for record in batch:
                            transfer = record.pop('_teacher_transfer', None)
                            if transfer is not None:
                                buffer, event, nodes = transfer
                                event.synchronize()  # writer thread, never decode
                                probabilities = buffer.tolist()
                                mass = 1.; teacher = []
                                for prefix, children, start, end, chosen_index in nodes:
                                    probs = probabilities[start:end]
                                    teacher.append({'prefix': prefix, 'children': children,
                                        'probabilities': probs+[max(0.,1-sum(probs))],
                                        'parent_mass': mass, 'inclusion_probability': 1.,
                                        'complete': True, 'source': 'runtime_existing_target_probs'})
                                    mass *= probabilities[chosen_index]
                                record['teacher'] = teacher
                            line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                            stream.write(line)
                            self.bytes += len(line.encode())
                            self.written += 1
                    finally:
                        if batch is not None:
                            with self.record_lock:
                                self.queued_records -= len(batch)
                        self.queue.task_done()
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:
            self.error = repr(exc)

    def submit(self, records):
        if self.closed:
            raise RuntimeError("closed I1 writer")
        if not records:
            return
        with self.record_lock:
            if self.queued_records + len(records) > self.max_records:
                self.dropped += len(records)
                return
            self.queued_records += len(records)
            self.peak_records = max(self.peak_records,self.queued_records)
        try:
            self.queue.put_nowait(records)
        except Full:
            with self.record_lock:
                self.queued_records -= len(records)
                self.dropped += len(records)

    def seal(self, metadata):
        if self.closed:
            raise RuntimeError("duplicate seal")
        # Boundary only. A failed writer must not cause an unbounded queue wait.
        deadline = time.monotonic() + 60
        while self.thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("I1 seal queue timeout")
            try:
                self.queue.put(None, timeout=.1)
                break
            except Full:
                continue
        self.thread.join(timeout=60)
        self.closed = True
        if self.thread.is_alive() or self.error:
            raise RuntimeError(f"I1 writer failed: {self.error or 'seal timeout'}")
        path = self.directory / "queries.jsonl"
        (self.directory / "queries.jsonl.partial").replace(path)
        result = {"schema": "hspec.i1.stream.v1", "sealed": True,
                  "query_schema": SCHEMA, "records": self.written, "bytes": self.bytes,
                  "queue_capacity_batches": self.capacity, "queue_capacity_records": self.max_records,
                  "queue_peak_records": self.peak_records, "dropped_records": self.dropped,
                  "data_sha256": file_sha(path), **metadata}
        atomic_json(self.directory / "manifest.json", result)
        return result


class I1Observer:
    def __init__(self, runner, probability=.02, capacity=256, max_pending=512):
        if not 0 < probability <= 1 or capacity <= 0 or max_pending <= 0:
            raise ValueError("invalid I1 sampling/buffer limits")
        self.runner = runner
        self.probability = probability
        self.capacity = capacity
        self.max_pending = max_pending
        self.rng = random.Random(int(os.getenv("HSPEC_I1_SEED", "90117")))
        self.engine = uuid.uuid4().hex
        self.round = 0
        self.serial = 0
        self.writer = None
        self.context = None
        self.pending = {}
        self.lifecycle = {}
        self.recently_closed = OrderedDict()
        self.current = {}
        self.counts = Counter()
        self.shape = {}
        self.pins = {}
        self.observer_ns = 0

    def configure(self, context):
        if self.writer is not None:
            self.finish("rollout_boundary")
        if context["verifier"] != "standard":
            raise ValueError("I1 labels require standard verifier")
        self.context = dict(context)
        self.round += 1
        self.counts = Counter()
        self.observer_ns = 0
        self.writer = TraceWriter(Path(context["trace_dir"]) /
                                  f"{self.engine}_round{self.round}", self.capacity)

    def begin(self, req_ids):
        if self.writer is None:
            raise RuntimeError("I1 observer lacks rollout context")
        self.current = {}
        self.counts["opportunities"] += len(req_ids)
        for rid in req_ids:
            self.serial += 1
            if self.rng.random() >= self.probability:
                continue
            if len(self.pending) + len(self.current) >= self.max_pending:
                self.counts["admission_drops"] += 1
                continue
            state = self.runner.requests.get(rid)
            if state is None:
                self.counts["missing_request_state"] += 1
                continue
            prompt = list(state.prompt_token_ids or [])
            output = list(state.output_token_ids)
            if len(prompt) + len(output) > 32768:
                self.counts["context_limit_drops"] += 1
                continue
            self.current[rid] = {"schema": SCHEMA, "query_id": f"{self.engine}:{self.round}:{self.serial}",
                "engine": self.engine, "request_id": rid, "generation": self.round,
                "proposal_round": self.serial, "decode_position": len(output),
                "sample_probability": self.probability, "lineage": dict(self.context),
                "context_tokens": prompt + output, "prompt_tokens": prompt,
                "freeze_ns": time.monotonic_ns(), "candidates": [],
                "availability": {"cache_anchor_ready": False, "legal_candidates": False,
                                 "raw_gate": False, "model_applicable": False,
                                 "positive_utility": None},
                "system": {"predicted_shape": dict(self.shape)}, "shadow_action": None}
            hf_config=getattr(getattr(self.runner,'model_config',None),'hf_config',None)
            eos=getattr(hf_config,'eos_token_id',None)
            self.current[rid]['eos_token_id']=eos if isinstance(eos,int) else None
            self.current[rid]['eos_token_ids']=eos if isinstance(eos,list) else ([eos] if eos is not None else [])
            params=getattr(state,'sampling_params',None)
            if params is not None:
                self.current[rid]['lineage']['recipe']={key:getattr(params,key) for key in
                    ('temperature','top_p','top_k','min_p','repetition_penalty','presence_penalty',
                     'frequency_penalty','ignore_eos','max_tokens')}
                self.current[rid]['sampling_stop']={'stop':params.stop,'stop_token_ids':params.stop_token_ids,
                    'min_tokens':params.min_tokens,'include_stop_str_in_output':params.include_stop_str_in_output}

    def candidates(self, proposer, active, tables, scores, indices, raw, norms, req_ids):
        start = time.perf_counter_ns()
        self.counts["cache_anchor_ready"] += len(active)
        for row, batch_idx in enumerate(active):
            rid = req_ids[batch_idx]
            gate = float(raw[row]) >= proposer.similarity_threshold
            self.counts["raw_gate"] += int(gate)
            record = self.current.get(rid)
            if record is None:
                continue
            cached = tables[row]
            current = self.runner.requests[rid].output_token_ids
            candidates = []
            for slot, (entry, score) in enumerate(zip(indices[row], scores[row])):
                entry = int(entry)
                if not 0 <= entry < cached.n_entries or not math.isfinite(float(score)):
                    continue
                rollout = int(cached.entry_rollout_idx[entry])
                offset = int(cached.entry_offset[entry])
                historical = cached.get_rollout_tokens(rollout)
                suffix = 0
                while suffix < min(8, offset, len(current)) and int(historical[offset-1-suffix]) == int(current[-1-suffix]):
                    suffix += 1
                tokens = list(map(int, cached.get_draft_tokens(entry, 15)))
                window = min(int(cached.get_effective_window(entry)), proposer.max_draft_tokens)
                window = proposer._apply_abs_delta_cap(window=window,
                    abs_delta=abs(offset - max(len(current), 1)), min_wnd=int(cached.min_wnd))
                source = cached.i1_lineage[rollout] if cached.i1_lineage else None
                candidates.append({"entry_id": entry, "slot": slot, "tokens": tokens,
                    "cosine": finite(score) if proposer._r1_config.computes_topk else None,
                    "key_norm": finite(cached.key_norm_cpu[entry]) if cached.key_norm_cpu is not None else None,
                    "suffix": suffix, "rollout": rollout, "offset": offset,
                    "source": source, "available_length": max(0, len(historical)-offset),
                    "legal_depth": min(window, len(tokens)), "table_ref": cached.i1_table_ref})
            record["candidates"] = candidates
            if proposer._r1_config.computes_topk and len(candidates)==8 and cached.key_norm_cpu is not None:
                import numpy as np
                from vllm_ascend.spec_decode.hspec_selector_survival import extract_utility_features_python
                features,_,_,_,status = extract_utility_features_python(
                    scores[row], indices[row], cached.entry_rollout_idx, cached.entry_offset,
                    cached.token_buffer, cached.rollout_token_offset, cached.rollout_token_len,
                    cached.key_norm_cpu, np.asarray(current,dtype=np.int32), float(norms[row]),
                    max(len(current)-1,0),len(current),cached.n_entries)
                record['p3a_feature_status']=int(status)
                if status==0:record['p3a_features']=features.tolist()
            record["prompt_id"] = proposer._get_prompt_ids_for_batch([rid])[0]
            record["raw_score"] = finite(raw[row])
            record["query_norm"] = finite(norms[row]) if proposer._r1_config.computes_topk else None
            record["score_space"] = "cosine" if proposer._r1_config.computes_topk else "raw"
            record["raw_threshold"] = proposer.similarity_threshold
            record["table_version"] = proposer._cache_version
            record["cache_generation"] = proposer._cache_generation
            state=self.runner.requests[rid]
            params=getattr(state,'sampling_params',None)
            response_limit=getattr(params,'max_tokens',None)
            context_remaining=max(0,int(getattr(self.runner,'max_model_len',32768))-len(record['context_tokens']))
            response_remaining=max(0,int(response_limit)-len(current)) if response_limit is not None else context_remaining
            record['legal_caps']={'context_remaining':context_remaining,'response_remaining':response_remaining,
                'batch_budget':int(proposer._max_draft_tokens_per_batch),'max_draft':int(proposer.max_draft_tokens),
                'scheduler_feasible_known':False,'note':'physical scheduler may further shorten; next-step actual length is authoritative'}
            for candidate in candidates:
                candidate['controller_depth']=candidate['legal_depth']
                candidate['legal_depth']=min(candidate['legal_depth'],context_remaining,response_remaining)
            record["availability"].update(cache_anchor_ready=True, raw_gate=gate,
                legal_candidates=any(c["legal_depth"] > 0 for c in candidates))
            record["freeze_ns"] = time.monotonic_ns()
            # Keep immutable table arrays and version metadata alive until outcome.
            self.pins[record["query_id"]] = cached
        self.observer_ns += time.perf_counter_ns() - start

    def proposed(self, req_ids, drafts, proposer):
        records = []
        for rid, draft in zip(req_ids, drafts):
            self.counts["proposed"] += int(bool(draft))
            if rid in self.lifecycle:
                self.counts['superseded_proposals'] += 1
                self.lifecycle.pop(rid)
            if draft:
                self.recently_closed.pop(rid, None)
                self.lifecycle[rid] = tuple(draft)
            record = self.current.get(rid)
            if record is None:
                continue
            if rid in self.pending:
                old = self.pending.pop(rid)
                old["outcome"] = {"status": "not_scheduled", "reason": "superseded"}
                records.append(old)
                self.pins.pop(old["query_id"], None)
            meta = proposer._pending_verify_meta.get(rid, {})
            record["action"] = {"entry_id": meta.get("matched_entry_idx"),
                "tokens": list(map(int, draft)), "length": len(draft),
                "behavior_probability": 1.0, "policy": proposer._r1_config.mode,
                "G": None, "S": None}
            # These booleans describe the existing policy, not PCS applicability.
            record["availability"]["model_applicable"] = record.get('p3a_feature_status')==0
            utility=meta.get('selected_utility')
            record["availability"]["positive_utility"] = bool(draft) if getattr(getattr(proposer,'_survival_config',None),'executes_utility',False) else None
            record['decision_detail']={key:meta.get(key) for key in (
                'stop_reason','utility_action','utility_model_version','predicted_accept_probability',
                'abs_delta_cap_applied','utility_action_cap_applied','batch_draft_budget_cap_applied',
                'base_effective_wnd_at_match','effective_wnd_at_match','entry_bias_at_match','entry_hits_at_match')}

            if draft:
                self.pending[rid] = record
            else:
                record["outcome"] = {"status": "abstained", "reason": "existing_selector_or_caps"}
                records.append(record)
                self.pins.pop(record["query_id"], None)
        self.current.clear()
        self.writer.submit(records)

    def capture_probabilities(self, lengths, probabilities):
        """Sampled arm: one packed async D2H of already transformed target p.

        A single verified path exposes complete sibling probabilities at its
        parents, including off-policy hypothetical parents after rejection.
        Other candidate paths are NOT evaluated. Greedy probes use offline AR.
        """
        if self.writer is None or not self.pending:
            return
        if self.context.get('recipe', {}).get('temperature') == 0:
            self.counts['teacher_greedy_skipped'] += 1
            return
        import torch
        rows=[]; tokens=[]; units=[]; offset=0
        for rid, length in zip(self.runner.input_batch.req_ids, lengths):
            record=self.pending.get(rid)
            if record is not None and length and record['lineage'].get('recipe',{}).get('temperature')!=0:
                paths=[c['tokens'] for c in record['candidates']]
                draft=record['action']['tokens']; nodes=[]
                for j in range(min(length,len(draft))):
                    prefix=draft[:j]
                    children=sorted({p[j] for p in paths if len(p)>j and p[:j]==prefix})
                    if not children:break
                    start=len(tokens);rows.extend([offset+j]*len(children));tokens.extend(children)
                    end=len(tokens);chosen_index=len(tokens);rows.append(offset+j);tokens.append(draft[j])
                    nodes.append((prefix,children,start,end,chosen_index))
                if nodes:units.append((record,nodes))
            offset+=length
        if not tokens:return
        row_index=torch.tensor(rows,device=probabilities.device,dtype=torch.long)
        token_index=torch.tensor(tokens,device=probabilities.device,dtype=torch.long)
        packed=probabilities[row_index,token_index]
        buffer=torch.empty(len(tokens),dtype=torch.float32,device='cpu',pin_memory=True)
        buffer.copy_(packed,non_blocking=True)
        event=torch.npu.Event();event.record()
        for record,nodes in units:
            record['_teacher_transfer']=(buffer,event,nodes)
        self.counts['teacher_gather_batches']+=1
        self.counts['teacher_D2H_bytes']+=4*len(tokens)
        self.counts['teacher_index_H2D_bytes']+=16*len(tokens)

    def outcomes(self, req_ids, scheduled, emitted, accepted):
        if self.writer is None:
            return
        records = []
        for rid, ids, count in zip(req_ids, emitted, accepted):
            draft = list(scheduled.get(rid, []))
            self.counts["scheduled"] += int(bool(draft))
            self.counts["verified"] += int(bool(draft) and bool(ids))
            live = self.lifecycle.pop(rid, None)
            leader = self.context.get('tp_rank',0)==0
            if leader and live is None and draft:
                self.counts['orphans'] += 1
                self.counts['late_outcomes'] += int(rid in self.recently_closed)
            if live is not None:
                if tuple(draft) != live[:len(draft)]:
                    raise RuntimeError('I1 unsampled proposal identity mismatch')
                self.counts['closed_verified' if draft and ids else 'closed_not_scheduled'] += 1
                self._remember_closed(rid)
            record = self.pending.pop(rid, None)
            if record is None:
                continue  # A known unsampled proposal has no label record.
            proposed = record["action"]["tokens"]
            if draft != proposed[:len(draft)] or not 0 <= count <= len(draft) <= len(proposed):
                raise RuntimeError("I1 proposal/scheduler/acceptance conservation failed")
            record["outcome"] = {"status": "verified" if draft and ids else "not_scheduled",
                "scheduled_tokens": draft, "scheduled_length": len(draft),
                "verified_length": len(draft) if ids else 0, "accepted": int(count) if ids else None,
                "emitted_tokens": list(map(int, ids)), "emitted_scope": "raw_verifier",
                "frontend_mask": None, "reason": "scheduler_clip" if len(draft)<len(proposed) else "none",
                "verified_ns": time.monotonic_ns(), "actual_shape": dict(self.shape),
                "rejected": bool(ids) and count < len(draft),
                "correction": int(ids[count]) if ids and count < len(draft) and len(ids)>count else None,
                "bonus": int(ids[count]) if ids and count == len(draft) and len(ids)>count else None}
            records.append(record)
            self.pins.pop(record["query_id"], None)
        self.writer.submit(records)

    def _remember_closed(self, rid):
        self.recently_closed[rid] = None
        self.recently_closed.move_to_end(rid)
        if len(self.recently_closed) > 4096:
            self.recently_closed.popitem(last=False)

    def cancel(self, rid, reason):
        if self.lifecycle.pop(rid, None) is not None:
            self.counts['closed_cancelled'] += 1
            self._remember_closed(rid)
        record = self.pending.pop(rid, None)
        if record is not None:
            record["outcome"] = {"status": "cancelled", "reason": reason}
            self.pins.pop(record["query_id"], None)
            self.writer.submit([record])

    def finish(self, reason):
        if self.writer is None:
            return None
        for rid in tuple(set(self.pending)|set(self.lifecycle)):
            self.cancel(rid, reason)
        self.counts['conservation_residual'] = (self.counts['proposed'] - self.counts['closed_verified'] -
            self.counts['closed_not_scheduled'] - self.counts['closed_cancelled'] - self.counts['superseded_proposals'])
        result = self.writer.seal({"lineage": self.context, "counts": dict(self.counts),
            "pending": len(self.pending), "pins": len(self.pins), "orphans": self.counts["orphans"],
            "late_outcomes": self.counts["late_outcomes"], "observer_candidate_cpu_ns": self.observer_ns})
        self.writer = None
        return result
