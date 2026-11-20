"""Exact-order reduction of eight contiguous BF16 projection shards on Ascend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_eight(
    p0, p1, p2, p3, p4, p5, p6, p7, result,
    size: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < size
    value = tl.load(p0 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p1 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p2 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p3 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p4 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p5 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p6 + offset, mask=valid, other=0).to(tl.float32)
    value = value + tl.load(p7 + offset, mask=valid, other=0).to(tl.float32)
    tl.store(result + offset, value.to(tl.bfloat16), mask=valid)


def try_ordered_sum(partials):
    first = partials[0]
    if len(partials) != 8 or first.device.type != "npu" or first.dtype != torch.bfloat16:
        return None
    if torch.is_grad_enabled() and any(value.requires_grad for value in partials):
        return None
    if not all(
        value.is_contiguous()
        and value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in partials
    ):
        return None
    result = torch.empty_like(first)
    if first.numel():
        block = 1024
        _sum_eight[(triton.cdiv(first.numel(), block),)](
            *partials, result, first.numel(), block, num_warps=1,
        )
    return result
