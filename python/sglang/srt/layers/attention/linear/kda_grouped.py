"""Group independent KDA requests while reusing the accepted numerical kernels."""
from itertools import accumulate

import torch
import torch.nn.functional as F


def grouped_decode_conv(mixed_qkv, weight, bias, conv_states, cache_indices):
    slots = cache_indices.long().clamp_min(0)
    valid = cache_indices > 0
    history = conv_states.index_select(0, slots)
    history = torch.where(valid[:, None, None], history, torch.zeros_like(history))
    window = torch.cat((history, mixed_qkv.unsqueeze(-1)), dim=-1)
    output = F.silu(F.conv1d(window, weight.unsqueeze(1), bias, groups=weight.shape[0]))
    updated = torch.where(valid[:, None, None], window[..., 1:], torch.zeros_like(history))
    conv_states.index_copy_(0, slots, updated.to(conv_states.dtype))
    return output.squeeze(-1).to(mixed_qkv.dtype)


def grouped_prefill_recurrence(dispatcher, values, *, A_log, dt_bias, ssm_states,
                               cache_indices, query_start_loc, lengths, lower_bound,
                               checkpoints, track_indices):
    """Advance live slots once and preserve separate radix-prefix snapshots."""
    bounds = list(accumulate((0, *lengths)))

    def run(inputs, slots, offsets):
        q, k, v, gate, beta = inputs
        return dispatcher.decode(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
            a=gate.contiguous(), b=beta.contiguous(), A_log=A_log, dt_bias=dt_bias,
            ssm_states=ssm_states, cache_indices=slots,
            query_start_loc=offsets, lower_bound=lower_bound,
        )

    intermediate = [i for i, size in checkpoints.items() if size < lengths[i]]
    if intermediate:
        selection = cache_indices.new_tensor(intermediate, dtype=torch.long)
        source = cache_indices.index_select(0, selection).long()
        destination = track_indices.index_select(0, selection).long()
        ssm_states.index_copy_(0, destination, ssm_states.index_select(0, source))
        prefix_values = [
            torch.cat([value[:, bounds[i]:bounds[i] + checkpoints[i]]
                       for i in intermediate], dim=1)
            for value in values
        ]
        prefix_bounds = list(accumulate((0, *(checkpoints[i] for i in intermediate))))
        run(prefix_values, track_indices.index_select(0, selection),
            query_start_loc.new_tensor(prefix_bounds))

    output = run(values, cache_indices, query_start_loc)
    final = [i for i, size in checkpoints.items() if size == lengths[i]]
    if final:
        selection = cache_indices.new_tensor(final, dtype=torch.long)
        source = cache_indices.index_select(0, selection).long()
        destination = track_indices.index_select(0, selection).long()
        ssm_states.index_copy_(0, destination, ssm_states.index_select(0, source))
    return output
