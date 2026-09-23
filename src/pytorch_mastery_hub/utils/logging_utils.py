"""
Logging helpers: idempotent logger setup and a lightweight metrics logger.

``setup_logger`` can be called repeatedly (e.g. every time a notebook cell is
re-run) without stacking duplicate handlers. ``MetricsLogger`` records scalar
metrics per step to memory and optionally to a JSON-lines file that can be read
back with pandas.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import IO, Any

__all__ = ["DEFAULT_FORMAT", "MetricsLogger", "get_logger", "setup_logger"]

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_HANDLER_TAG = "_pytorch_hub_handler"


def setup_logger(
    name: str = "pytorch_mastery_hub",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = DEFAULT_FORMAT,
    stream: IO[str] | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create (or reconfigure) a logger with a console and optional file handler.

    Calling this twice with the same ``name`` replaces the handlers this
    function installed previously instead of adding more, so log lines are not
    duplicated in notebooks.

    Args:
        name: Logger name. Use ``__name__`` in library code.
        level: Logging level (int or name such as ``"DEBUG"``).
        log_file: If given, also write to this file (parents are created).
        fmt: ``logging.Formatter`` format string.
        stream: Console stream; defaults to ``sys.stdout`` so output interleaves
            correctly with ``print`` in notebooks.
        propagate: Whether records bubble up to the root logger.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)
    logger.propagate = propagate

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream or sys.stdout)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger (``pytorch_mastery_hub.<name>``)."""
    base = "pytorch_mastery_hub"
    return logging.getLogger(base if not name else f"{base}.{name}")


class MetricsLogger:
    """
    Record scalar metrics per step, in memory and optionally as JSON lines.

    Example::

        metrics = MetricsLogger("runs/exp1/metrics.jsonl")
        for step, batch in enumerate(loader):
            loss = ...
            metrics.log(step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
        print(metrics.history["loss"])
        df = metrics.to_dataframe()  # requires pandas
    """

    def __init__(self, path: str | Path | None = None, *, flush_every: int = 1) -> None:
        self.path = Path(path) if path is not None else None
        self.flush_every = max(1, flush_every)
        self.history: dict[str, list[float]] = defaultdict(list)
        self.steps: list[int] = []
        self._records: list[dict[str, Any]] = []
        self._fh: IO[str] | None = None
        self._start = time.monotonic()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(self, step: int, **metrics: float) -> dict[str, Any]:
        """Record ``metrics`` for ``step``; returns the stored record."""
        record: dict[str, Any] = {
            "step": int(step),
            "elapsed_s": round(time.monotonic() - self._start, 4),
        }
        for key, value in metrics.items():
            fvalue = float(value)
            record[key] = fvalue
            self.history[key].append(fvalue)
        self.steps.append(int(step))
        self._records.append(record)

        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            if len(self._records) % self.flush_every == 0:
                self._fh.flush()
        return record

    def last(self, key: str) -> float | None:
        """Most recent value of ``key`` or ``None``."""
        values = self.history.get(key)
        return values[-1] if values else None

    def best(self, key: str, mode: str = "min") -> tuple[int, float] | None:
        """``(step, value)`` of the best value of ``key``."""
        values = self.history.get(key)
        if not values:
            return None
        pick = min if mode == "min" else max
        idx = pick(range(len(values)), key=values.__getitem__)
        return self.steps[idx], values[idx]

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def to_dataframe(self) -> Any:
        """Return the records as a ``pandas.DataFrame`` (pandas imported lazily)."""
        import pandas as pd

        return pd.DataFrame(self._records)

    @classmethod
    def load(cls, path: str | Path) -> MetricsLogger:
        """Rebuild a logger's history from a JSON-lines file."""
        inst = cls()
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                step = rec.pop("step")
                rec.pop("elapsed_s", None)
                inst.log(step, **rec)
        return inst

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._records)
