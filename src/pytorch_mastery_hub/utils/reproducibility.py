"""
Reproducibility helpers: global seeding, deterministic kernels and RNG scoping.

Deep learning results are only comparable when every random source is pinned.
``seed_everything`` covers Python, NumPy, PyTorch (CPU + CUDA + MPS) and the
``PYTHONHASHSEED`` used by ``hash()``; ``deterministic=True`` additionally
forces deterministic cuDNN/cuBLAS kernels at some cost in speed.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = [
    "RNGState",
    "capture_rng_state",
    "isolated_rng",
    "make_generator",
    "restore_rng_state",
    "seed_everything",
    "seed_worker",
]


@dataclass
class RNGState:
    """Snapshot of every RNG the library touches."""

    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None = None


def seed_everything(seed: int = 42, *, deterministic: bool = False, warn_only: bool = True) -> int:
    """
    Seed every random number generator in the process.

    Args:
        seed: Seed value (0 <= seed < 2**32).
        deterministic: Also request deterministic algorithms. Sets
            ``torch.backends.cudnn.deterministic``, disables cuDNN autotuning and
            calls :func:`torch.use_deterministic_algorithms`. Some ops have no
            deterministic implementation; with ``warn_only=True`` PyTorch warns
            instead of raising.
        warn_only: Passed through to :func:`torch.use_deterministic_algorithms`.

    Returns:
        The seed that was applied (useful when a caller passes ``None`` upstream).
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, all CUDA devices and MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return seed


def seed_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` for :class:`torch.utils.data.DataLoader`.

    PyTorch seeds each worker's torch RNG, but NumPy and ``random`` would
    otherwise be forked with identical state, producing duplicated augmentations
    across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a seeded :class:`torch.Generator` (for DataLoader shuffling, splits, ...)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def capture_rng_state() -> RNGState:
    """Snapshot the current state of all RNGs."""
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_states,
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore a snapshot taken with :func:`capture_rng_state`."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    # A state loaded with torch.load(map_location=<accelerator>) may have been moved;
    # generators only accept CPU uint8 tensors.
    torch.set_rng_state(state.torch_cpu.detach().to("cpu", torch.uint8))
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.detach().to("cpu", torch.uint8) for s in state.torch_cuda])


@contextlib.contextmanager
def isolated_rng(seed: int | None = None) -> Iterator[None]:
    """
    Run a block with its own RNG state, restoring the outer state afterwards.

    Useful for weight initialisation or data sampling that must not disturb the
    surrounding training run::

        with isolated_rng(0):
            init_weights(model)
    """
    state = capture_rng_state()
    try:
        if seed is not None:
            seed_everything(seed)
        yield
    finally:
        restore_rng_state(state)
