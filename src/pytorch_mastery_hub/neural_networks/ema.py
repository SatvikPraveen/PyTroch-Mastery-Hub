"""
Exponential moving average (EMA) of model weights.

Keeping an EMA copy of the parameters and evaluating with it is a cheap,
reliable way to improve generalisation (used by Mean Teacher, BYOL, diffusion
models, and most modern image-classification recipes).
"""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

__all__ = ["ModelEMA", "steps_to_reach"]


class ModelEMA:
    """
    Maintain ``ema = decay * ema + (1 - decay) * model`` after each optimizer step.

    Args:
        model: The live model. A deep copy is taken for the shadow weights.
        decay: Target decay (0.999 - 0.9999 are typical).
        warmup_steps: If > 0 the effective decay ramps up as
            ``min(decay, (1 + step) / (warmup_steps + step))`` so the EMA is not
            dominated by the random initial weights early in training.
        update_buffers: Also copy buffers (BatchNorm running stats). They are
            copied, not averaged, matching the timm implementation.

    Example::

        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            loss = step(model, batch)
            optimizer.step()
            ema.update(model)
        with ema.swap(model):
            evaluate(model)  # uses EMA weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        *,
        warmup_steps: int = 0,
        update_buffers: bool = True,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_buffers = update_buffers
        self.step = 0
        self.module = copy.deepcopy(_unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def effective_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1 + self.step) / (self.warmup_steps + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the live ``model`` weights into the shadow copy."""
        self.step += 1
        d = self.effective_decay()
        src = _unwrap(model)
        ema_params = dict(self.module.named_parameters())
        for name, p in src.named_parameters():
            ema_p = ema_params[name]
            if p.dtype.is_floating_point:
                ema_p.lerp_(p.detach().to(ema_p.dtype), 1.0 - d)
            else:
                ema_p.copy_(p)
        if self.update_buffers:
            ema_bufs = dict(self.module.named_buffers())
            for name, b in src.named_buffers():
                ema_bufs[name].copy_(b)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model`` with the EMA weights (irreversible)."""
        _unwrap(model).load_state_dict(self.module.state_dict(), strict=True)

    @contextlib.contextmanager
    def swap(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily load EMA weights into ``model``; restores on exit."""
        target = _unwrap(model)
        backup = {k: v.detach().clone() for k, v in target.state_dict().items()}
        target.load_state_dict(self.module.state_dict(), strict=True)
        try:
            yield model
        finally:
            target.load_state_dict(backup, strict=True)

    def to(self, device: torch.device | str) -> ModelEMA:
        self.module.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step": self.step,
            "module": self.module.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]
        self.step = state["step"]
        self.module.load_state_dict(state["module"])

    def __repr__(self) -> str:
        return f"ModelEMA(decay={self.decay}, warmup_steps={self.warmup_steps}, step={self.step})"


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip torch.compile / DataParallel / DDP wrappers."""
    orig = getattr(model, "_orig_mod", None)
    if isinstance(orig, nn.Module):
        return orig
    inner = getattr(model, "module", None)
    if isinstance(inner, nn.Module):
        return inner
    return model


def steps_to_reach(decay: float, fraction: float = 0.99) -> int:
    """How many updates until the EMA has absorbed ``fraction`` of new weights."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    return math.ceil(math.log(1 - fraction) / math.log(decay))
