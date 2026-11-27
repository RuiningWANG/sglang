"""Project training and serving indexer inputs with identical rounding."""
import torch


def project_rows(value, weight):
    from sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops import npu_mm_batch_invariant

    rows = value.reshape(-1, value.shape[-1]).contiguous()
    result = npu_mm_batch_invariant(rows, weight.contiguous().transpose(0, 1))
    return result.reshape(*value.shape[:-1], weight.shape[0])


def project_head_weights(value, weight, heads):
    # The release PoolKeyIndexer applies the head-dimension scale in the operator.
    return (project_rows(value.float(), weight.float()) * heads**-0.5).to(torch.bfloat16).contiguous()
