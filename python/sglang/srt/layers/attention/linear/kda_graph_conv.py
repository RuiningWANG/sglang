"""Decode through the same Conv1d + SiLU path with static graph shapes."""

import torch
import torch.nn.functional as F


def decode_graph_conv(mixed_qkv, weight, bias, conv_states, cache_indices):
    """Consume one token per request, reserving state slot zero for graph padding."""
    slots = cache_indices.to(torch.int64)
    valid = slots > 0
    slots = slots.clamp_min(0)
    rows = []
    for i in range(mixed_qkv.shape[0]):
        slot = slots[i:i + 1]
        # The native decode first gathers [1, channels, width - 1], appends the
        # current token, then executes a bias-free depthwise Conv1d and SiLU.
        history = conv_states.index_select(0, slot)
        history = torch.where(valid[i:i + 1, None, None], history, torch.zeros_like(history))
        current = mixed_qkv[i:i + 1].unsqueeze(-1)
        window = torch.cat((history, current), dim=-1)
        y = F.silu(F.conv1d(window, weight.unsqueeze(1), bias, groups=weight.shape[0]))
        rows.append(y.squeeze(-1).to(mixed_qkv.dtype))
        new_history = torch.where(valid[i:i + 1, None, None], window[..., 1:], torch.zeros_like(history))
        conv_states.index_copy_(0, slot, new_history.to(conv_states.dtype))
    return torch.cat(rows, dim=0)
