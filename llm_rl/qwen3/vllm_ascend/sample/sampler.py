import os

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

# ``npu_apply_top_k_top_p`` is the fast path used by vLLM-Ascend.  Its
# cumulative-probability mask is numerically correct for distinct logits, but
# the order of equal logits at the top-p boundary is not part of the operator
# contract.  HF's TopPLogitsWarper uses a stable descending sort, so equal
# logits are resolved by the smaller vocabulary id first.  Keep the default
# hybrid policy cheap for the common case and make the full reference policy
# selectable for qualification/reproducibility.
_TOP_P_POLICY_ENV = "HSPEC_ASCEND_TOP_P_POLICY"
# Preserve the upstream Ascend behavior unless a correctness campaign opts in
# explicitly.  I1 sets ``HSPEC_ASCEND_TOP_P_POLICY=hf_stable``; ordinary
# vLLM/verl/HSpec/EAGLE experiments therefore retain the native policy.
_TOP_P_POLICY_DEFAULT = "legacy"
_TOP_P_POLICIES = {"legacy", "hybrid", "hf_stable"}


def top_p_policy() -> str:
    """Return the process-wide top-p tie policy used by this backend."""
    policy = os.getenv(_TOP_P_POLICY_ENV, _TOP_P_POLICY_DEFAULT).strip().lower()
    if policy not in _TOP_P_POLICIES:
        raise ValueError(
            f"{_TOP_P_POLICY_ENV} must be one of {sorted(_TOP_P_POLICIES)}, "
            f"got {policy!r}"
        )
    return policy


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendSampler(Sampler):

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler()
        self.async_exponential_event = torch.npu.Event()

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def do_async_exponential(self, b_s, head_dim, generators):
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)


class AscendTopKTopPSampler(TopKTopPSampler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if get_ascend_config().enable_async_exponential:
            # Add synchronize to prevent synchronize error.
            self.async_event.synchronize()
            return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
        return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    # Qualification and reproducibility runs may request the fully stable
    # device-side implementation.  Bypass the custom kernel for active top-p
    # rows so its unspecified boundary ordering cannot enter the result.  A
    # p==1 row remains a no-op/top-k fast path.
    if p is not None and top_p_policy() == "hf_stable":
        active = p < 1.0
        if bool(active.any().item()):
            if bool(active.all().item()):
                return _apply_top_k_top_p_hf_stable(logits, k, p)
            fast = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)
            rows = torch.nonzero(active, as_tuple=False).flatten()
            stable_k = k.index_select(0, rows) if k is not None else None
            stable = _apply_top_k_top_p_hf_stable(
                logits.index_select(0, rows).clone(), stable_k,
                p.index_select(0, rows)
            )
            fast.index_copy_(0, rows, stable)
            return fast
        # No top-p filtering is active.  Keep the operation device-side and
        # avoid invoking the custom kernel for p==1, whose tiling path is not
        # available for every diagnostic vocabulary shape.
        if k is None:
            return logits
        return _apply_top_k_top_p_hf_stable(logits, k, p)
    fast = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)
    if p is None or top_p_policy() == "legacy":
        return fast

    # p==1 is a no-op for top-p.  Avoid the boundary scan in this case.  The
    # custom op returns a separate tensor and leaves ``logits`` unchanged;
    # this is part of its Python wrapper contract and lets us inspect the
    # original values without an additional full-vocabulary clone.
    active = p < 1.0
    if not bool(active.any().item()):
        return fast

    # A custom top-p result has a finite value exactly where a token survived.
    # If more than one surviving boundary value is equal, the custom kernel's
    # unspecified tie order can disagree with HF even when it kept all ties.
    # Route only those rows to the stable reference implementation.  The
    # comparison is exact in the input dtype (BF16 ties are the issue being
    # repaired); near ties retain the optimized operator and are covered by
    # the numerical TV gate.
    finite = torch.isfinite(fast)
    finite_count = finite.sum(dim=-1)
    boundary = torch.where(finite, logits, torch.full_like(logits, float("inf"))).amin(dim=-1)
    boundary_ties = (logits == boundary.unsqueeze(-1)).sum(dim=-1) > 1
    if top_p_policy() == "hf_stable":
        repair_rows = active
    else:
        repair_rows = active & (finite_count > 0) & boundary_ties
    if not bool(repair_rows.any().item()):
        return fast

    rows = torch.nonzero(repair_rows, as_tuple=False).flatten()
    stable_logits = logits.index_select(0, rows).clone()
    stable_k = k.index_select(0, rows) if k is not None else None
    stable_p = p.index_select(0, rows)
    stable = _apply_top_k_top_p_hf_stable(stable_logits, stable_k, stable_p)
    # ``fast`` is writable for all supported Ascend kernels.  index_copy_ keeps
    # the output shape/dtype and preserves the low-overhead fast path rows.
    fast.index_copy_(0, rows, stable)
    return fast


def _apply_top_k_top_p_hf_stable(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
) -> torch.Tensor:
    """Apply the I1 reference-compatible stable top-k/top-p mask on NPU.

    The implementation intentionally stays device-side.  It is used for
    boundary-tie rows by the hybrid path and for all top-p rows in the
    explicit ``hf_stable`` qualification mode; no CPU vocabulary transfer or
    host-side sampling is introduced.
    """
    if k is not None:
        vocab = logits.shape[-1]
        positive = k > 0
        if bool(positive.any().item()):
            # ``top_k`` metadata is normalized by vLLM, but clamp here so the
            # diagnostic path remains well-defined for a direct probe.
            kk = k.to(torch.long).clamp(min=1, max=vocab)
            max_k = int(kk.max().item())
            kth = logits.topk(max_k, dim=-1).values.gather(1, (kk - 1).unsqueeze(1))
            no_op = (k <= 0) | (k >= vocab)
            logits.masked_fill_((logits < kth) & ~no_op.unsqueeze(1), -float("inf"))

    # The I1 teacher contract uses a minimal-set top-p rule: sort descending
    # stably and remove every item whose cumulative mass is above p.  Preserve
    # the first item for p below its own mass.  (The previous Ascend kernel
    # effectively used an ascending threshold and retained an entire equal
    # boundary group.)  Stable sort resolves an equal-logit group by its
    # original (therefore lower) vocabulary index.
    sorted_logits, sorted_indices = torch.sort(
        logits, dim=-1, descending=True, stable=True
    )
    sorted_probs = sorted_logits.softmax(dim=-1, dtype=torch.float32)
    remove = sorted_probs.cumsum(dim=-1) > p.unsqueeze(1)
    remove[:, 0] = False
    # ``src`` must be a distinct tensor: some NPU scatter implementations do
    # not guarantee correctness when source and destination alias.
    remove = remove.scatter(dim=-1, index=sorted_indices, src=remove.clone())
    logits.masked_fill_(remove, -float("inf"))
    return logits

apply_top_k_top_p = _apply_top_k_top_p_ascendc if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3] else _apply_top_k_top_p_pytorch
