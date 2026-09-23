"""
Memory measurement utilities for CPU, CUDA and MPS.

``get_memory_usage`` returns a uniform dict regardless of backend, and
``MemoryTracker`` records named snapshots so you can see which phase of a
pipeline (data loading, forward, backward, optimizer step) allocates the most.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .device_utils import get_device

try:  # POSIX only; Windows has no `resource` module
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI
    resource = None  # type: ignore[assignment]

__all__ = ["MemorySnapshot", "MemoryTracker", "clear_memory", "get_memory_usage", "tensor_bytes"]

_MB = 1024**2


def _windows_working_set_bytes() -> int:  # pragma: no cover - Windows only
    """Current working set via psapi.GetProcessMemoryInfo (no psutil dependency)."""
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)  # type: ignore[attr-defined]
    return int(pmc.WorkingSetSize)


def _process_rss_mb() -> float:
    """Resident set size of this process in MB (psutil-free, cross-platform)."""
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        return usage / _MB if sys.platform == "darwin" else usage / 1024
    if sys.platform == "win32":  # pragma: no cover
        return _windows_working_set_bytes() / _MB
    return 0.0  # pragma: no cover


def get_memory_usage(device: torch.device | str | None = None) -> dict[str, float]:
    """
    Current memory statistics in megabytes.

    Keys always present: ``process_rss_mb``. On CUDA additionally
    ``allocated_mb``, ``reserved_mb``, ``max_allocated_mb``, ``total_mb``; on
    MPS ``allocated_mb`` and ``driver_mb``.
    """
    dev = torch.device(device) if device is not None else get_device()
    stats: dict[str, float] = {"process_rss_mb": round(_process_rss_mb(), 2)}
    if dev.type == "cuda" and torch.cuda.is_available():
        stats.update(
            allocated_mb=torch.cuda.memory_allocated(dev) / _MB,
            reserved_mb=torch.cuda.memory_reserved(dev) / _MB,
            max_allocated_mb=torch.cuda.max_memory_allocated(dev) / _MB,
            total_mb=torch.cuda.get_device_properties(dev).total_memory / _MB,
        )
    elif dev.type == "mps" and hasattr(torch, "mps"):
        stats.update(
            allocated_mb=torch.mps.current_allocated_memory() / _MB,
            driver_mb=torch.mps.driver_allocated_memory() / _MB,
        )
    return {k: round(v, 2) for k, v in stats.items()}


def clear_memory(device: torch.device | str | None = None) -> None:
    """Run the garbage collector and release cached accelerator memory."""
    gc.collect()
    dev = torch.device(device) if device is not None else get_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def tensor_bytes(*tensors: torch.Tensor) -> int:
    """Total bytes of storage referenced by ``tensors`` (shared storage counted once)."""
    seen: set[int] = set()
    total = 0
    for t in tensors:
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += t.untyped_storage().nbytes()
    return total


@dataclass
class MemorySnapshot:
    tag: str
    elapsed_s: float
    process_rss_mb: float
    allocated_mb: float | None = None
    reserved_mb: float | None = None
    delta_mb: float | None = None


class MemoryTracker:
    """
    Record memory at named points and report deltas.

    Example::

        tracker = MemoryTracker()
        tracker.snapshot("start")
        out = model(x)
        tracker.snapshot("forward")
        out.sum().backward()
        tracker.snapshot("backward")
        print(tracker.report())

    Also usable as a context manager, which snapshots ``enter``/``exit``.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.snapshots: list[MemorySnapshot] = []
        self._start = time.monotonic()

    def _key_metric(self, stats: dict[str, float]) -> float:
        # Prefer accelerator allocation when available, else process RSS.
        return stats.get("allocated_mb", stats["process_rss_mb"])

    def snapshot(self, tag: str = "") -> MemorySnapshot:
        stats = get_memory_usage(self.device)
        current = self._key_metric(stats)
        prev = self._key_metric(asdict_to_stats(self.snapshots[-1])) if self.snapshots else None
        snap = MemorySnapshot(
            tag=tag or f"snapshot_{len(self.snapshots)}",
            elapsed_s=round(time.monotonic() - self._start, 4),
            process_rss_mb=stats["process_rss_mb"],
            allocated_mb=stats.get("allocated_mb"),
            reserved_mb=stats.get("reserved_mb"),
            delta_mb=None if prev is None else round(current - prev, 2),
        )
        self.snapshots.append(snap)
        return snap

    def peak(self) -> float:
        """Highest key metric seen across snapshots (MB)."""
        if not self.snapshots:
            return 0.0
        return max(self._key_metric(asdict_to_stats(s)) for s in self.snapshots)

    def reset(self) -> None:
        self.snapshots.clear()
        self._start = time.monotonic()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def report(self) -> str:
        """Human-readable table of all snapshots."""
        if not self.snapshots:
            return "MemoryTracker: no snapshots"
        has_alloc = any(s.allocated_mb is not None for s in self.snapshots)
        header = f"{'tag':<20} {'t(s)':>8} {'rss MB':>10}"
        if has_alloc:
            header += f" {'alloc MB':>10} {'delta MB':>10}"
        lines = [header, "-" * len(header)]
        for s in self.snapshots:
            row = f"{s.tag:<20} {s.elapsed_s:>8.3f} {s.process_rss_mb:>10.1f}"
            if has_alloc:
                row += f" {s.allocated_mb or 0:>10.1f} {s.delta_mb or 0:>+10.1f}"
            else:
                row += f" {'':>10} {s.delta_mb or 0:>+10.1f}" if s.delta_mb is not None else ""
            lines.append(row)
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.snapshots]

    def __enter__(self) -> MemoryTracker:
        self.snapshot("enter")
        return self

    def __exit__(self, *exc: object) -> None:
        self.snapshot("exit")


def asdict_to_stats(snap: MemorySnapshot) -> dict[str, float]:
    stats = {"process_rss_mb": snap.process_rss_mb}
    if snap.allocated_mb is not None:
        stats["allocated_mb"] = snap.allocated_mb
    return stats
