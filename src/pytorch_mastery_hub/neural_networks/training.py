"""
Training engine for PyTorch Mastery Hub.

Two layers of API:

* Functional: :func:`train_epoch` / :func:`validate_epoch` - one epoch each,
  return a dict of averaged metrics. Good for notebooks.
* :class:`Trainer` - a small but complete training loop with

  - device-agnostic automatic mixed precision (CUDA fp16/bf16, CPU bf16),
  - gradient accumulation and gradient-norm clipping,
  - per-step or per-epoch LR schedulers (``ReduceLROnPlateau`` handled),
  - exponential moving average of weights (:class:`~.ema.ModelEMA`),
  - optional ``torch.compile``,
  - a typed :class:`Callback` protocol (early stopping, checkpointing, ...),
  - resumable checkpoints (model, optimizer, scheduler, scaler, EMA, RNG,
    epoch, global step, history).

The loop is deliberately explicit (~200 lines) so it can be read as a
reference implementation of what Lightning/Accelerate do under the hood.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..utils import distributed as dist_utils
from ..utils.device_utils import autocast_dtype, get_device, move_to_device
from ..utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from .ema import ModelEMA

__all__ = [
    "Callback",
    "EarlyStopping",
    "EarlyStoppingCallback",
    "LambdaCallback",
    "LearningRateSchedulerCallback",
    "ModelCheckpoint",
    "ModelCheckpointCallback",
    "ProgressCallback",
    "Trainer",
    "TrainerConfig",
    "train_epoch",
    "train_with_mixed_precision",
    "validate_epoch",
]

MetricFn = Callable[[torch.Tensor, torch.Tensor], "torch.Tensor | float"]
Precision = Literal["no", "auto", "fp16", "bf16"]


# --------------------------------------------------------------------------- helpers


def _accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy in percent for logits ``(N, C)`` and integer targets."""
    pred = output.argmax(dim=1)
    return 100.0 * (pred == target.view_as(pred)).float().mean().item()


def _is_classification(output: torch.Tensor, target: torch.Tensor) -> bool:
    return output.dim() >= 2 and output.size(1) > 1 and not target.is_floating_point()


def _unpack_batch(batch: Any) -> tuple[Any, torch.Tensor]:
    """Support ``(inputs, targets)`` tuples and HF-style dict batches."""
    if isinstance(batch, Mapping):
        b = dict(batch)
        target = b.pop("labels", None)
        if target is None:
            target = b.pop("targets", None)
        if target is None:
            raise KeyError("dict batches must contain a 'labels' or 'targets' key")
        return b, target
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError("batch must be an (inputs, targets) pair or a dict with 'labels'")


def _forward(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def _make_scaler(enabled: bool) -> Any:
    """GradScaler compatible with torch 2.0 (torch.cuda.amp) and >= 2.3 (torch.amp)."""
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - old torch


def _resolve_precision(precision: Precision | bool, device: torch.device) -> torch.dtype | None:
    if precision in (False, "no"):
        return None
    if precision in (True, "auto"):
        return autocast_dtype(device)
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"precision must be one of no/auto/fp16/bf16, got {precision!r}")


class _RunningMean:
    """Weighted running mean for a set of metrics."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, values: Mapping[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.sums[k] += float(v) * n
            self.counts[k] += n

    def averages(self) -> dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}


def _compute_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    metrics: Mapping[str, MetricFn] | None,
    auto_accuracy: bool,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if metrics:
        for name, fn in metrics.items():
            val = fn(output.detach(), target)
            out[name] = float(val.item() if isinstance(val, torch.Tensor) else val)
    if auto_accuracy and "accuracy" not in out and _is_classification(output, target):
        out["accuracy"] = _accuracy(output.detach(), target)
    return out


# ------------------------------------------------------------------- functional API


def train_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: nn.Module | Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str | None = None,
    scheduler: Any | None = None,
    clip_grad_norm: float | None = None,
    accumulation_steps: int = 1,
    *,
    precision: Precision | bool = "no",
    metrics: Mapping[str, MetricFn] | None = None,
    scaler: Any | None = None,
    ema: ModelEMA | None = None,
    non_blocking: bool = True,
) -> dict[str, float]:
    """
    Train ``model`` for one epoch and return averaged metrics.

    Args:
        model: Module to train (moved to ``device`` if needed).
        dataloader: Yields ``(inputs, targets)`` pairs or dicts with ``labels``.
        criterion: Loss function ``(output, target) -> scalar``.
        optimizer: Optimizer whose ``step`` is called every
            ``accumulation_steps`` batches.
        device: Target device; auto-detected when ``None``.
        scheduler: Per-*step* LR scheduler (e.g. OneCycleLR). Epoch-level
            schedulers should be stepped by the caller.
        clip_grad_norm: If set, clip the global grad norm before each step.
        accumulation_steps: Number of micro-batches per optimizer step.
        precision: ``"no"`` | ``"auto"`` | ``"fp16"`` | ``"bf16"`` (or bool).
        metrics: Extra ``name -> fn(output, target)`` metrics averaged per epoch.
        scaler: Optional externally managed ``GradScaler`` (for fp16).
        ema: Optional :class:`ModelEMA` updated after each optimizer step.
        non_blocking: Use async host-to-device copies (pinned memory).

    Returns:
        ``{"loss": ..., "accuracy": ..., **metrics, "grad_norm": ...}``
        (``accuracy`` only for classification outputs, ``grad_norm`` only when
        clipping).
    """
    dev = torch.device(device) if device is not None else get_device()
    model.to(dev)
    model.train()

    amp_dtype = _resolve_precision(precision, dev)
    use_scaler = amp_dtype == torch.float16 and dev.type == "cuda"
    scaler = scaler if scaler is not None else _make_scaler(use_scaler)
    autocast_ctx: Callable[[], Any] = (
        (lambda: torch.autocast(device_type=dev.type, dtype=amp_dtype))
        if amp_dtype is not None
        else contextlib.nullcontext
    )

    running = _RunningMean()
    optimizer.zero_grad(set_to_none=True)
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs, target = _unpack_batch(move_to_device(batch, dev, non_blocking))
        n = int(target.shape[0]) if target.dim() > 0 else 1

        with autocast_ctx():
            output = _forward(model, inputs)
            loss = criterion(output, target)
        scaler.scale(loss / accumulation_steps).backward()

        step_now = (batch_idx + 1) % accumulation_steps == 0
        batch_logs: dict[str, float] = {"loss": loss.item()}
        batch_logs.update(_compute_metrics(output.float(), target, metrics, auto_accuracy=True))

        if step_now:
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                batch_logs["grad_norm"] = float(gn)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        running.update(batch_logs, n)
        num_batches += 1

    # Flush a partial accumulation window so no gradients are silently dropped.
    if num_batches % accumulation_steps != 0:
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

    return running.averages()


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: nn.Module | Callable[..., torch.Tensor],
    device: torch.device | str | None = None,
    compute_metrics: Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]] | None = None,
    *,
    precision: Precision | bool = "no",
    metrics: Mapping[str, MetricFn] | None = None,
    non_blocking: bool = True,
) -> dict[str, float]:
    """
    Evaluate ``model`` over ``dataloader``; returns averaged metrics.

    ``compute_metrics(all_outputs, all_targets)`` (if given) is called once on
    the concatenated CPU predictions for dataset-level metrics such as F1 or
    AUROC; ``metrics`` are per-batch and averaged.
    """
    dev = torch.device(device) if device is not None else get_device()
    model.to(dev)
    model.eval()
    amp_dtype = _resolve_precision(precision, dev)
    autocast_ctx: Callable[[], Any] = (
        (lambda: torch.autocast(device_type=dev.type, dtype=amp_dtype))
        if amp_dtype is not None
        else contextlib.nullcontext
    )

    running = _RunningMean()
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in dataloader:
        inputs, target = _unpack_batch(move_to_device(batch, dev, non_blocking))
        n = int(target.shape[0]) if target.dim() > 0 else 1
        with autocast_ctx():
            output = _forward(model, inputs)
            loss = criterion(output, target)
        output = output.float()
        logs = {"loss": loss.item()}
        logs.update(_compute_metrics(output, target, metrics, auto_accuracy=True))
        running.update(logs, n)
        if compute_metrics is not None:
            outputs.append(output.cpu())
            targets.append(target.cpu())

    result = running.averages()
    if compute_metrics is not None and outputs:
        result.update(compute_metrics(torch.cat(outputs), torch.cat(targets)))
    return result


def train_with_mixed_precision(
    model: nn.Module,
    train_loader: Iterable[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str | None = None,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Backwards-compatible alias for ``train_epoch(..., precision="auto")``."""
    return train_epoch(
        model, train_loader, criterion, optimizer, device, precision="auto", scaler=scaler
    )


# ----------------------------------------------------------------------- callbacks


class Callback:
    """
    Base class for :class:`Trainer` hooks. Override any subset of methods.

    ``logs`` passed to ``on_epoch_end`` is a flat dict such as
    ``{"loss": 0.4, "accuracy": 87.5, "val_loss": 0.5, "val_accuracy": 84.1,
    "lr": 1e-3, "epoch_time": 3.2}``.
    """

    def on_train_begin(self, trainer: Trainer) -> None: ...

    def on_train_end(self, trainer: Trainer) -> None: ...

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None: ...

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None: ...

    def on_batch_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None: ...

    def on_validation_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None: ...


class LambdaCallback(Callback):
    """Build a callback from plain functions: ``LambdaCallback(on_epoch_end=fn)``."""

    def __init__(self, **hooks: Callable[..., None]) -> None:
        for name, fn in hooks.items():
            if not hasattr(Callback, name):
                raise ValueError(f"Unknown hook {name!r}")
            setattr(self, name, fn)


class _Monitor:
    """Shared best-value tracking for EarlyStopping / ModelCheckpoint."""

    def __init__(self, monitor: str, mode: str, min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best = math.inf if mode == "min" else -math.inf

    def value(self, logs: Mapping[str, float]) -> float | None:
        if self.monitor in logs:
            return logs[self.monitor]
        # Accept "val_loss" when only "loss" exists (no validation loader) and vice versa.
        alt = self.monitor.removeprefix("val_")
        return logs.get(alt)

    def improved(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta


class EarlyStopping(Callback):
    """
    Stop training when ``monitor`` stops improving for ``patience`` epochs.

    With ``restore_best_weights=True`` the model is reset to the best epoch's
    weights when training stops (also on normal completion).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        self._m = _Monitor(monitor, mode, min_delta)
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = -1
        self.stopped_epoch: int | None = None
        self.best_weights: dict[str, torch.Tensor] | None = None

    @property
    def best_score(self) -> float:
        return self._m.best

    @property
    def monitor(self) -> str:
        return self._m.monitor

    def on_train_begin(self, trainer: Trainer) -> None:
        self.wait = 0
        self.stopped_epoch = None

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        current = self._m.value(logs)
        if current is None:
            return
        if self._m.improved(current):
            self._m.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.detach().clone() for k, v in trainer.unwrapped_model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best {self.monitor}={self._m.best:.4f} at epoch {self.best_epoch + 1})"
                    )

    def on_train_end(self, trainer: Trainer) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            trainer.unwrapped_model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch + 1}")


class ModelCheckpoint(Callback):
    """
    Save a resumable checkpoint after each epoch.

    ``filepath`` may contain ``{epoch}`` and any log key, e.g.
    ``"ckpt/epoch{epoch:02d}-val_loss{val_loss:.3f}.pt"``.
    """

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = False,
        verbose: bool = True,
    ) -> None:
        self.filepath = str(filepath)
        self._m = _Monitor(monitor, mode)
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        self.best_path: Path | None = None
        self.last_path: Path | None = None

    @property
    def best_score(self) -> float:
        return self._m.best

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        current = self._m.value(logs)
        is_best = current is not None and self._m.improved(current)
        if is_best:
            self._m.best = current  # type: ignore[assignment]

        if is_best or not self.save_best_only:
            path = Path(self.filepath.format(epoch=epoch + 1, **logs))
            trainer.save_checkpoint(path, extra={"epoch_logs": logs})
            if is_best:
                self.best_path = path
            if self.verbose:
                tag = "best" if is_best else "epoch"
                print(f"Saved {tag} checkpoint to {path}")

        if self.save_last:
            self.last_path = Path(self.filepath).with_name("last.pt")
            trainer.save_checkpoint(self.last_path, extra={"epoch_logs": logs})


class LearningRateSchedulerCallback(Callback):
    """
    Step an epoch-level scheduler after every epoch.

    ``ReduceLROnPlateau`` is stepped with the monitored metric. Prefer passing
    ``scheduler=`` to :class:`Trainer`, which does this automatically; this
    callback exists for schedulers created outside the trainer.
    """

    def __init__(self, scheduler: Any, monitor: str = "val_loss") -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = logs.get(self.monitor, logs.get(self.monitor.removeprefix("val_")))
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()


class ProgressCallback(Callback):
    """Print a one-line summary every ``print_freq`` epochs."""

    def __init__(self, print_freq: int = 1) -> None:
        self.print_freq = max(1, print_freq)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        if (epoch + 1) % self.print_freq == 0:
            body = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items() if k != "epoch_time")
            print(
                f"Epoch {epoch + 1}/{trainer.config.epochs} - {logs.get('epoch_time', 0):.1f}s - {body}"
            )


# Backwards-compatible names
EarlyStoppingCallback = EarlyStopping
ModelCheckpointCallback = ModelCheckpoint


# -------------------------------------------------------------------------- trainer


@dataclass
class TrainerConfig:
    """All knobs of :class:`Trainer` in one serialisable place."""

    epochs: int = 10
    precision: Precision | bool = "no"
    accumulation_steps: int = 1
    clip_grad_norm: float | None = None
    ema_decay: float | None = None
    ema_warmup_steps: int = 0
    compile: bool = False
    seed: int | None = None
    non_blocking: bool = True
    scheduler_interval: Literal["auto", "step", "epoch"] = "auto"
    monitor: str = "val_loss"
    log_every_n_steps: int = 0
    verbose: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")


class Trainer:
    """
    Reference training loop with mixed precision, EMA, callbacks and resume.

    Example::

        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.AdamW(model.parameters(), 3e-4),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10),
            config=TrainerConfig(epochs=10, precision="auto", ema_decay=0.999),
            callbacks=[EarlyStopping(patience=3), ModelCheckpoint("ckpt/best.pt")],
        )
        history = trainer.fit(train_loader, val_loader)
        trainer.evaluate(test_loader)

    The legacy positional signature
    ``Trainer(model, criterion, optimizer, device, scheduler, clip_grad_norm,
    accumulation_steps)`` is still accepted.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device | str | None = None,
        scheduler: Any | None = None,
        clip_grad_norm: float | None = None,
        accumulation_steps: int | None = None,
        *,
        config: TrainerConfig | None = None,
        callbacks: Iterable[Callback] | None = None,
        metrics: Mapping[str, MetricFn] | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        if clip_grad_norm is not None:
            self.config.clip_grad_norm = clip_grad_norm
        if accumulation_steps is not None:
            self.config.accumulation_steps = accumulation_steps

        self.device = torch.device(device) if device is not None else get_device()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = dict(metrics or {})
        self.callbacks: list[Callback] = list(callbacks or [])

        if self.config.seed is not None:
            seed_everything(self.config.seed)

        self._raw_model = model.to(self.device)
        self.model: nn.Module = (
            torch.compile(self._raw_model) if self.config.compile else self._raw_model  # type: ignore[assignment]
        )

        self.amp_dtype = _resolve_precision(self.config.precision, self.device)
        self.scaler = _make_scaler(self.amp_dtype == torch.float16 and self.device.type == "cuda")
        self.ema = (
            ModelEMA(
                self.unwrapped_model,
                decay=self.config.ema_decay,
                warmup_steps=self.config.ema_warmup_steps,
            ).to(self.device)
            if self.config.ema_decay is not None
            else None
        )

        self.history: dict[str, list[float]] = defaultdict(list)
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False

    # ------------------------------------------------------------------ properties

    @property
    def unwrapped_model(self) -> nn.Module:
        """The model without ``torch.compile``/DDP wrapping (use for state_dict)."""
        return dist_utils.unwrap_ddp(self._raw_model)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def _scheduler_is_per_step(self) -> bool:
        if self.scheduler is None:
            return False
        if self.config.scheduler_interval != "auto":
            return self.config.scheduler_interval == "step"
        per_step = (
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.CyclicLR,
            torch.optim.lr_scheduler.LambdaLR,
        )
        return isinstance(self.scheduler, per_step)

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ----------------------------------------------------------------- core steps

    def training_step(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Forward + loss for one batch. Override for custom losses/inputs.

        Must return ``(loss, logs)``; ``logs`` may contain extra scalars.
        """
        inputs, target = _unpack_batch(batch)
        with self._autocast():
            output = _forward(self.model, inputs)
            loss = self.criterion(output, target)
        logs = _compute_metrics(output.float(), target, self.metrics, auto_accuracy=True)
        return loss, logs

    def _optimizer_step(self) -> float | None:
        grad_norm: float | None = None
        if self.config.clip_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None and self._scheduler_is_per_step():
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self._raw_model)
        return grad_norm

    def train_one_epoch(self, loader: Iterable[Any]) -> dict[str, float]:
        self.model.train()
        running = _RunningMean()
        accum = self.config.accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        pending = 0

        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, self.device, self.config.non_blocking)
            _, target = _unpack_batch(batch)
            n = int(target.shape[0]) if target.dim() > 0 else 1

            loss, logs = self.training_step(batch)
            self.scaler.scale(loss / accum).backward()
            pending += 1
            logs = {"loss": loss.item(), **logs}

            if pending == accum:
                gn = self._optimizer_step()
                pending = 0
                self.global_step += 1
                if gn is not None:
                    logs["grad_norm"] = gn
                if (
                    self.config.log_every_n_steps
                    and self.global_step % self.config.log_every_n_steps == 0
                    and self.config.verbose
                ):
                    print(f"  step {self.global_step}: loss={logs['loss']:.4f} lr={self.lr:.2e}")
                for cb in self.callbacks:
                    cb.on_batch_end(self, self.global_step, logs)

            running.update(logs, n)

        if pending:  # flush a partial accumulation window
            self._optimizer_step()
            self.global_step += 1
        return running.averages()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Any], *, use_ema: bool | None = None) -> dict[str, float]:
        """Evaluate on ``loader``; uses EMA weights when available unless ``use_ema=False``."""
        use_ema = (self.ema is not None) if use_ema is None else (use_ema and self.ema is not None)
        ctx = self.ema.swap(self._raw_model) if use_ema and self.ema else contextlib.nullcontext()
        with ctx:
            return validate_epoch(
                self.model,
                loader,
                self.criterion,
                self.device,
                precision=self.config.precision,
                metrics=self.metrics,
                non_blocking=self.config.non_blocking,
            )

    @torch.no_grad()
    def predict(self, loader: Iterable[Any], *, use_ema: bool | None = None) -> torch.Tensor:
        """Concatenated model outputs (on CPU) for every batch in ``loader``."""
        use_ema = (self.ema is not None) if use_ema is None else (use_ema and self.ema is not None)
        ctx = self.ema.swap(self._raw_model) if use_ema and self.ema else contextlib.nullcontext()
        self.model.eval()
        outputs = []
        with ctx:
            for batch in loader:
                batch = move_to_device(batch, self.device, self.config.non_blocking)
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                if isinstance(inputs, Mapping):
                    inputs = {k: v for k, v in inputs.items() if k not in ("labels", "targets")}
                with self._autocast():
                    outputs.append(_forward(self.model, inputs).float().cpu())
        return torch.cat(outputs)

    # ------------------------------------------------------------------------ fit

    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        epochs: int | None = None,
        verbose: bool | None = None,
    ) -> dict[str, list[float]]:
        """
        Train for ``epochs`` (defaults to ``config.epochs``) and return history.

        History keys: ``loss``, ``accuracy`` (classification), any extra metric,
        ``val_*`` counterparts, ``lr`` and ``epoch_time``.
        """
        if epochs is not None:
            self.config.epochs = epochs
        if verbose is not None:
            self.config.verbose = verbose
        self.stop_training = False

        for cb in self.callbacks:
            cb.on_train_begin(self)

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            t0 = time.perf_counter()
            dist_utils.set_epoch(train_loader, epoch)  # per-epoch shuffle under DDP
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            logs: dict[str, float] = dist_utils.reduce_dict(self.train_one_epoch(train_loader))
            if val_loader is not None:
                val_logs = dist_utils.reduce_dict(self.evaluate(val_loader))
                logs.update({f"val_{k}": v for k, v in val_logs.items()})
                for cb in self.callbacks:
                    cb.on_validation_end(self, epoch, logs)

            if self.scheduler is not None and not self._scheduler_is_per_step():
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = logs.get(self.config.monitor, logs.get("loss"))
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            logs["lr"] = self.lr
            logs["epoch_time"] = time.perf_counter() - t0
            for k, v in logs.items():
                self.history[k].append(v)

            if self.config.verbose:
                self._print_epoch(epoch, logs)
            # Callbacks that checkpoint must see the number of *completed* epochs.
            self.current_epoch = epoch + 1
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if self.stop_training:
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        return dict(self.history)

    def _print_epoch(self, epoch: int, logs: Mapping[str, float]) -> None:
        parts = [f"Epoch {epoch + 1}/{self.config.epochs}", f"{logs['epoch_time']:.1f}s"]
        parts += [f"{k}: {v:.4f}" for k, v in logs.items() if k not in ("epoch_time", "lr")]
        parts.append(f"lr: {logs['lr']:.2e}")
        print(" - ".join(parts))

    # --------------------------------------------------------------- checkpoints

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume: weights, optimizer, scheduler, AMP, EMA, RNG."""
        state: dict[str, Any] = {
            "model": self.unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "history": dict(self.history),
            "config": self.config.__dict__,
            "rng": capture_rng_state(),
        }
        return state

    def load_state_dict(self, state: Mapping[str, Any], *, restore_rng: bool = True) -> None:
        self.unwrapped_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
        self.current_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.history = defaultdict(list, {k: list(v) for k, v in state.get("history", {}).items()})
        if restore_rng and state.get("rng") is not None:
            restore_rng_state(state["rng"])

    def save_checkpoint(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        # Load to CPU: module/optimizer load_state_dict copy into device tensors, and
        # RNG states must stay on CPU (map_location="mps"/"cuda" would break them).
        # weights_only=False is required for the RNGState / numpy generator objects that
        # save_checkpoint writes; only load checkpoints you produced yourself.
        state = torch.load(Path(path), map_location="cpu", weights_only=False)  # nosec B614
        self.load_state_dict(state, restore_rng=restore_rng)
        extra: dict[str, Any] = state.get("extra", {})
        return extra
