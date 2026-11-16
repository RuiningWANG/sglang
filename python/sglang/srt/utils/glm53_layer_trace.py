import os
from contextvars import ContextVar
from pathlib import Path

import torch


_counter = 0
mhc_trace = ContextVar("glm53_mhc_trace", default=None)


def _trace_flag():
    directory = os.getenv("GLM53_TRACE_DIR")
    return Path(directory) / "trace-phase" if directory else None


def trace_requested():
    flag = _trace_flag()
    return flag is not None and flag.is_file()


def start(ids, positions, batch):
    global _counter
    flag = _trace_flag()
    if flag is None or (
        torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
    ):
        return None
    try:
        phase = flag.read_text().strip()
    except FileNotFoundError:
        return None
    if not phase or batch.batch_size != 1:
        return None
    source_ids = batch.input_ids if ids is None else ids
    source_pos = positions if positions is not None else batch.positions
    if source_ids is None or source_pos is None:
        print(
            f"GLM53_TRACE_SKIPPED phase={phase} missing token IDs or positions",
            flush=True,
        )
        return None
    pos = source_pos.detach().cpu().reshape(-1)
    token_ids = source_ids.detach().cpu().reshape(-1)
    if pos.numel() != token_ids.numel():
        print(
            f"GLM53_TRACE_SKIPPED phase={phase} IDs={token_ids.numel()} positions={pos.numel()}",
            flush=True,
        )
        return None
    trace = {
        "phase": phase,
        "mode": str(batch.forward_mode),
        "ids": token_ids,
        "positions": pos,
        "values": {},
        "trace_dir": str(flag.parent),
    }
    _counter += 1
    trace["call"] = _counter
    return trace


def record(trace, label, tensor):
    if trace is not None and tensor.ndim and tensor.shape[0] <= trace["ids"].numel():
        trace["values"][label] = tensor.detach().cpu()


def finish(trace):
    if trace is not None:
        directory = Path(trace["trace_dir"]) / "tensor-trace"
        directory.mkdir(exist_ok=True)
        torch.save(trace, directory / f"{trace['call']:05d}.pt")


def record_kda(batch, name, tensor):
    trace = getattr(batch, "glm53_kda_trace", None)
    if trace is not None:
        record(trace, "layer-00/kda/" + name, tensor)


def record_mhc(label, tensor):
    if tensor is not None:
        record(mhc_trace.get(), "layer-33/mhc/" + label, tensor)
