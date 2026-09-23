"""
Device selection and placement helpers.

Centralises the "which accelerator do I have?" logic so notebooks, examples and
the training loop all agree. Supports CUDA, Apple Silicon (MPS) and CPU, with an
environment-variable override (``PYTORCH_HUB_DEVICE``) for reproducible runs.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from typing import Any, TypeVar

import torch

__all__ = [
    "autocast_dtype",
    "device_info",
    "get_device",
    "move_to_device",
    "print_system_info",
    "synchronize",
]

T = TypeVar("T")

_ENV_OVERRIDE = "PYTORCH_HUB_DEVICE"


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def get_device(prefer: str | torch.device | None = None) -> torch.device:
    """
    Return the best available device.

    Resolution order:

    1. ``prefer`` argument (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``).
    2. ``PYTORCH_HUB_DEVICE`` environment variable.
    3. CUDA, then MPS, then CPU.

    A requested accelerator that is not available silently falls back to CPU so
    that the same code runs on laptops and GPU boxes.

    Args:
        prefer: Optional explicit device string or object.

    Returns:
        A :class:`torch.device`.
    """
    requested: str | None
    if prefer is not None:
        requested = str(prefer)
    else:
        requested = os.environ.get(_ENV_OVERRIDE)

    if requested:
        requested = requested.lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if requested == "mps" and _mps_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        # Requested accelerator unavailable: fall through to auto-detection.

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device | str | None = None) -> torch.dtype | None:
    """
    Pick a sensible mixed-precision dtype for ``device``.

    * CUDA: ``bfloat16`` on Ampere+ (compute capability >= 8), else ``float16``.
    * CPU: ``bfloat16`` (the only dtype CPU autocast supports).
    * MPS: ``float16``.

    Returns ``None`` when autocast should be disabled.
    """
    dev = torch.device(device) if device is not None else get_device()
    if dev.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dev.type == "cpu":
        return torch.bfloat16
    if dev.type == "mps":
        return torch.float16
    return None


def move_to_device(obj: T, device: torch.device | str, non_blocking: bool = False) -> T:
    """
    Recursively move tensors inside (nested) lists, tuples, dicts or dataclasses.

    Non-tensor leaves are returned untouched, so a batch such as
    ``{"input_ids": Tensor, "meta": {"id": 3}}`` works.
    """
    result: Any
    if isinstance(obj, torch.Tensor):
        result = obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, Mapping):
        result = type(obj)((k, move_to_device(v, device, non_blocking)) for k, v in obj.items())  # type: ignore[call-arg]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        result = type(obj)(*(move_to_device(v, device, non_blocking) for v in obj))
    elif isinstance(obj, (list, tuple)):
        result = type(obj)(move_to_device(v, device, non_blocking) for v in obj)
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            setattr(obj, name, move_to_device(getattr(obj, name), device, non_blocking))
        result = obj
    else:
        result = obj
    return result


def synchronize(device: torch.device | str | None = None) -> None:
    """Block until all queued kernels on ``device`` have finished (for timing)."""
    dev = torch.device(device) if device is not None else get_device()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def device_info() -> dict[str, Any]:
    """Collect a JSON-serialisable snapshot of the compute environment."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": _mps_available(),
        "num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def print_system_info() -> None:
    """Pretty-print :func:`device_info`."""
    info = device_info()
    width = max(len(k) for k in info)
    print("System information")
    print("-" * (width + 24))
    for key, value in info.items():
        print(f"  {key:<{width}} : {value}")
