"""Reduce BF16 MoE partials in one source-rank order after exchanging token slices."""
import torch
import torch.distributed as dist


def reduce_rank_order(partials, sizes, group):
    world=group.size()
    rank=group.rank()
    if len(sizes)!=world or sum(sizes)!=partials.shape[0]:
        raise ValueError('MoE split sizes do not match the EP rank group')
    local_rows=sizes[rank]
    shape=partials.shape[1:]
    slices=partials.new_empty((world*local_rows,*shape))
    dist.all_to_all_single(
        slices,partials.contiguous(),output_split_sizes=[local_rows]*world,
        input_split_sizes=sizes,group=group,
    )
    received=slices.reshape(world,local_rows,*shape)
    total=received[0].float()
    for source in received[1:]:
        total.add_(source.float())
    return total.to(partials.dtype)
