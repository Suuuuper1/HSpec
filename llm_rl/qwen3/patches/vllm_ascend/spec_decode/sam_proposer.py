# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType


class SAM:
    """Incremental suffix automaton used as a per-request token proposer."""

    @dataclass(slots=True)
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, n_predicts: int = 3):
        if n_predicts < 1:
            raise ValueError("n_predicts must be positive")
        self.n_predicts = n_predicts
        self.reset()

    def reset(self) -> None:
        self.states: list[SAM.SAMState] = [
            SAM.SAMState(next={}, link=-1, length=0, min_endpos=0)
        ]
        # SAM end positions are one-based. The sentinel keeps token storage
        # aligned with min_endpos without copying proposal slices.
        self.input_ids: list[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.cur_index = 0
        self.cur_length = 0

    def expand_state(self, state: SAMState) -> int:
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int) -> None:
        self.max_length += 1
        cur = self.expand_state(
            SAM.SAMState(
                next={}, link=-1, length=self.max_length,
                min_endpos=self.max_length,
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                source = self.states[q]
                clone = self.expand_state(
                    SAM.SAMState(
                        next=source.next.copy(), link=source.link,
                        length=source.length, min_endpos=source.min_endpos,
                    )
                )
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next.get(token) == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = clone
                self.states[cur].link = clone
        self.last = cur

    def transfer_state(self, index: int, length: int, token: int) -> tuple[int, int]:
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            return self.states[index].next[token], length + 1
        return 0, 0

    def transfer_cur_state(self, token: int) -> None:
        self.cur_index, self.cur_length = self.transfer_state(
            self.cur_index, self.cur_length, token
        )

    def add_tokens(self, tokens: Sequence[int] | np.ndarray) -> None:
        for raw_token in tokens:
            token = int(raw_token)
            self.transfer_cur_state(token)
            self.add_state(token)
            self.input_ids.append(token)

    def lookup(self, token: int) -> tuple[int, int]:
        return self.transfer_state(self.cur_index, self.cur_length, int(token))

    def to_ancestor(self, index: int) -> int:
        if index != 0:
            length_to_end = self.max_length - self.states[index].min_endpos
            while self.states[index].link != 0 and self.n_predicts > length_to_end:
                index = self.states[index].link
                length_to_end = self.max_length - self.states[index].min_endpos
        return index

    def gen_draft(self, index: int) -> list[int]:
        index = self.to_ancestor(index)
        endpos = self.states[index].min_endpos
        if endpos == 0:
            return []
        return self.input_ids[endpos + 1:endpos + self.n_predicts + 1]

    def propose(self, context_token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Reference one-shot proposal helper used by zero-NPU tests."""
        if not len(context_token_ids):
            return np.empty(0, dtype=np.int64)
        self.reset()
        self.add_tokens(context_token_ids[:-1])
        index, _ = self.lookup(int(context_token_ids[-1]))
        return np.asarray(self.gen_draft(index), dtype=np.int64)


class SAMDecodingProposer(Proposer):
    """Old-vLLM-ABI adapter for incremental per-request SAM proposal."""

    def __init__(self, vllm_config, device, runner):
        self.n_predicts = int(
            vllm_config.speculative_config.num_speculative_tokens
        )
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.all_proposers: dict[str, SAM] = {}
        self.name = SpecDcodeType.SAM
        self.device = device
        self.runner = runner

    def clear_request(self, request_id: str) -> None:
        self.all_proposers.pop(request_id, None)

    def propose(
        self,
        request_id: str,
        old_token_ids: Sequence[int] | np.ndarray,
        new_token_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        if not len(new_token_ids):
            return np.empty(0, dtype=np.int64)

        proposer = self.all_proposers.get(request_id)
        if proposer is None:
            proposer = SAM(n_predicts=self.n_predicts)
            proposer.add_tokens(old_token_ids)
            self.all_proposers[request_id] = proposer

        # The last emitted token is the query. Earlier accepted/recovered
        # tokens must become history before matching that query.
        proposer.add_tokens(new_token_ids[:-1])
        index, _ = proposer.lookup(int(new_token_ids[-1]))
        draft = proposer.gen_draft(index)
        proposer.add_tokens(new_token_ids[-1:])
        return np.asarray(draft, dtype=np.int64)

    def load_model(self, *args, **kwargs) -> None:
        # SAM has no neural draft weights.
        return None

    @torch.inference_mode()
    def dummy_run(self, *args, **kwargs) -> None:
        return None

    def generate_token_ids(
        self, valid_sampled_token_ids, *args, **kwargs
    ) -> list[list[int]]:
        input_batch = self.runner.input_batch
        draft_token_ids: list[list[int]] = []

        for i, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            # Bookkeeping has already appended this step's emitted tokens to
            # token_ids_cpu in the Ascend runner. Use sampled_ids explicitly so
            # this adapter does not depend on that write ordering.
            num_tokens = int(input_batch.num_tokens_no_spec[i])
            num_new = len(sampled_ids)
            old_end = num_tokens - num_new
            if old_end < 0 or num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            draft = self.propose(
                req_id,
                input_batch.token_ids_cpu[i, :old_end],
                sampled_ids,
            )
            max_draft = max(0, self.max_model_len - num_tokens - 1)
            draft_token_ids.append(draft[:max_draft].tolist())

        # Finished/preempted request state must not grow for the lifetime of an
        # RL rollout worker. Recomputed requests rebuild from token_ids_cpu.
        active_req_ids = set(input_batch.req_id_to_index)
        for stale_req_id in self.all_proposers.keys() - active_req_ids:
            self.clear_request(stale_req_id)

        return draft_token_ids
