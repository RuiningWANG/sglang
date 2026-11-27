"""The BF16 GLM5.3 EP16 layout used by both rollout and the trainer."""
import torch

from glm53_zero_mismatch import enabled


def use_local_ep16(config, parallel) -> bool:
    """Describe the standard dispatch path with rank-local expert IDs."""
    return (
        enabled()
        and (parallel.tp_size, parallel.attn_dp_size, parallel.moe_ep_size) == (16, 16, 16)
        and parallel.moe_tp_size == 1
        and parallel.dwdp_size == 1
        and config.num_experts == 288
        and config.num_local_experts == 18
        and config.top_k == 8
        and config.num_fused_shared_experts == 0
        and config.params_dtype == torch.bfloat16
        and not config.apply_router_weight_on_input
        and not config.no_combine
    )
