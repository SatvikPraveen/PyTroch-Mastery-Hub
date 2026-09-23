"""
Distributed-training helpers (DistributedDataParallel).

Everything degrades gracefully to single-process behaviour when no process
group is initialised, so the same script runs with plain ``python`` and with
``torchrun --nproc_per_node=N``.

Typical use::

    from pytorch_mastery_hub.utils import distributed as dist_utils

    dist_utils.init_distributed()  # reads RANK/WORLD_SIZE/LOCAL_RANK
    device = dist_utils.local_device()
    model = dist_utils.wrap_ddp(model.to(device))
    sampler = dist_utils.make_sampler(train_ds, shuffle=True)
    loader = DataLoader(train_ds, sampler=sampler, batch_size=64)
    for epoch in range(epochs):
        dist_utils.set_epoch(loader, epoch)  # different shuffle per epoch
        ...
        metrics = dist_utils.reduce_dict({"loss": loss})  # averaged over ranks
    dist_utils.cleanup()
"""

from __future__ import annotations

import contextlib
import functools
import os
import socket
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DistributedSampler, Sampler

__all__ = [
    "all_gather_object",
    "all_gather_tensors",
    "all_reduce_mean",
    "barrier",
    "cleanup",
    "find_free_port",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "local_device",
    "local_rank",
    "main_process_first",
    "main_process_only",
    "make_sampler",
    "reduce_dict",
    "set_epoch",
    "spawn",
    "unwrap_ddp",
    "wrap_ddp",
]

F = TypeVar("F", bound=Callable[..., Any])


# ------------------------------------------------------------------ introspection


def is_distributed() -> bool:
    """True when a process group is initialised (``torchrun`` / ``spawn``)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def local_rank() -> int:
    """Rank on this machine (``LOCAL_RANK`` set by torchrun; 0 otherwise)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def local_device() -> torch.device:
    """``cuda:<local_rank>`` when CUDA is available, else CPU/MPS via get_device()."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    from .device_utils import get_device

    return get_device()


# ------------------------------------------------------------------- lifecycle


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def init_distributed(
    backend: str | None = None,
    *,
    timeout_minutes: float = 30,
    init_method: str | None = None,
) -> bool:
    """
    Initialise the default process group from ``torchrun`` environment variables.

    Returns ``False`` (and does nothing) when ``WORLD_SIZE`` is absent or 1, so
    scripts stay runnable without a launcher. ``backend`` defaults to NCCL when
    CUDA is available and gloo otherwise.
    """
    if is_distributed():
        return True
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(
        backend=backend,
        init_method=init_method or "env://",
        timeout=timedelta(minutes=timeout_minutes),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return True


def cleanup() -> None:
    """Destroy the process group if one exists."""
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


# ------------------------------------------------------------------ collectives


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average ``tensor`` over all ranks (returns a copy; no-op single process)."""
    if not is_distributed():
        return tensor
    out = tensor.detach().clone().float()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / get_world_size()


def reduce_dict(
    values: Mapping[str, float | torch.Tensor], average: bool = True
) -> dict[str, float]:
    """Reduce a dict of scalars across ranks in one collective (keys must match on all ranks)."""
    if not values:
        return {}
    keys = sorted(values)
    device = (
        torch.device("cuda", local_rank())
        if (is_distributed() and dist.get_backend() == "nccl")
        else "cpu"
    )
    stacked = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        if average:
            stacked /= get_world_size()
    return dict(zip(keys, stacked.tolist()))


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate ``tensor`` from every rank along dim 0 (supports unequal lengths)."""
    if not is_distributed():
        return tensor
    tensor = tensor.detach()
    world = get_world_size()
    local_len = torch.tensor([tensor.shape[0]], device=tensor.device)
    lens = [torch.zeros_like(local_len) for _ in range(world)]
    dist.all_gather(lens, local_len)
    max_len = int(max(int(n.item()) for n in lens))
    padded = torch.zeros((max_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[: int(n.item())] for g, n in zip(gathered, lens)], dim=0)


def all_gather_object(obj: Any) -> list[Any]:
    """Gather arbitrary picklable objects from all ranks."""
    if not is_distributed():
        return [obj]
    out: list[Any] = [None] * get_world_size()
    dist.all_gather_object(out, obj)
    return out


# ------------------------------------------------------------------- model/data


def wrap_ddp(
    model: nn.Module,
    *,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Wrap in DistributedDataParallel when distributed; returns ``model`` unchanged otherwise."""
    if not is_distributed():
        return model
    device_ids = [local_rank()] if torch.cuda.is_available() else None
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        **kwargs,
    )


def unwrap_ddp(model: nn.Module) -> nn.Module:
    inner = getattr(model, "module", None)
    return (
        inner
        if isinstance(model, nn.parallel.DistributedDataParallel) and isinstance(inner, nn.Module)
        else model
    )


def make_sampler(
    dataset: Dataset[Any], *, shuffle: bool = True, seed: int = 0, drop_last: bool = False
) -> Sampler[Any] | None:
    """A ``DistributedSampler`` when distributed, else ``None`` (let DataLoader shuffle)."""
    if not is_distributed():
        return None
    return DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)


def set_epoch(loader: Any, epoch: int) -> None:
    """Call ``sampler.set_epoch`` if the loader's sampler supports it (needed for per-epoch shuffling)."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


# ------------------------------------------------------------------- decorators


def main_process_only(fn: F) -> F:
    """Run ``fn`` only on rank 0 (returns ``None`` elsewhere). Good for logging/saving."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_main_process():
            return fn(*args, **kwargs)
        return None

    return wrapper  # type: ignore[return-value]


@contextlib.contextmanager
def main_process_first() -> Iterator[None]:
    """
    Let rank 0 run the block first, then the others (e.g. dataset download /
    tokenizer cache), avoiding N processes racing on the same files.
    """
    if not is_distributed():
        yield
        return
    if not is_main_process():
        dist.barrier()
    try:
        yield
    finally:
        if is_main_process():
            dist.barrier()


# ------------------------------------------------------------------ test helper


def _spawn_entry(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def spawn(
    fn: Callable[..., Any],
    world_size: int = 2,
    *,
    backend: str = "gloo",
    args: tuple[Any, ...] = (),
) -> None:
    """
    Run ``fn(rank, world_size, *args)`` in ``world_size`` processes on one machine.

    Uses a file-based rendezvous, so it needs no free port and works in tests
    and notebooks. Use ``torchrun`` for real multi-GPU jobs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "rendezvous")
        torch.multiprocessing.spawn(
            _spawn_entry,
            args=(world_size, backend, init_file, fn, args),
            nprocs=world_size,
            join=True,
        )
