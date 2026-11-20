"""Shared Ascend GLM5-next TP8 projection arithmetic."""

import os

import torch

from sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops import (
    npu_mm_batch_invariant,
)


def projection_shard(x, weight):
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    output = npu_mm_batch_invariant(rows, weight.contiguous().transpose(0, 1))
    return output.reshape(*x.shape[:-1], weight.shape[0])


def ordered_shard_sum(partials, *, allow_fused=True):
    if allow_fused and os.getenv("GLM53_TUNE_ORDERED_SUM") == "1":
        from sglang.srt.hardware_backend.npu.glm53_ordered_reduce import try_ordered_sum

        result = try_ordered_sum(partials)
        if result is not None:
            return result

    total = partials[0].float()
    for partial in partials[1:]:
        total = total + partial.float()
    return total.to(partials[0].dtype)


def gathered_eight_shard_sum(group, partial):
    if group.world_size != 8:
        raise RuntimeError("GLM5-next parity reduction requires eight ranks")
    gathered = group.all_gather(partial.contiguous(), dim=0).reshape(8, *partial.shape)
    # The decode graph uses at most 32 token rows. Larger dynamic prefill shapes
    # retain the eager ordered sum to avoid specialization and launch overhead.
    allow_fused = partial.ndim > 0 and partial.shape[-1] > 0 and partial.numel() <= 32 * partial.shape[-1]
    return ordered_shard_sum(gathered.unbind(0), allow_fused=allow_fused)
