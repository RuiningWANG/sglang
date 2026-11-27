"""Select the same KDA radix checkpoints as SGLang's packed prefill."""


def snapshot_prefix_lengths(mask, tracked, prefixes, lengths, chunk_size):
    checkpoints = {}
    for request, (keep, tag, prefix, total) in enumerate(
        zip(mask, tracked, prefixes, lengths, strict=True)
    ):
        if not keep:
            continue
        # An aligned tag selects the final state. For intermediate checkpoints
        # the scheduler tags the desired boundary plus one to select chunk h.
        relative_tag = tag - prefix
        size = total if relative_tag % chunk_size == 0 else (
            relative_tag // chunk_size * chunk_size
        )
        if not 0 < size <= total:
            raise ValueError(
                f"KDA cache checkpoint at {size} is outside request {request} ({total} tokens)"
            )
        checkpoints[request] = size
    return checkpoints

