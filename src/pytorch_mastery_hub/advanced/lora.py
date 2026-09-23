"""
LoRA: Low-Rank Adaptation of large models (Hu et al., 2021).

Freeze a pretrained weight ``W`` and learn a rank-``r`` update
``ΔW = (alpha / r) · B @ A`` with ``A ∈ R^{r×in}``, ``B ∈ R^{out×r}``. Only
``A`` and ``B`` train, cutting optimizer memory by orders of magnitude, and the
update can be merged back into ``W`` for zero-overhead inference.

Usage::

    apply_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    mark_only_lora_trainable(model)
    ...train...
    torch.save(lora_state_dict(model), "adapter.pt")   # tiny file
    merge_lora(model)                                  # fold into base weights
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "LoRALinear",
    "apply_lora",
    "lora_parameters",
    "lora_state_dict",
    "mark_only_lora_trainable",
    "merge_lora",
    "unmerge_lora",
]


class LoRALinear(nn.Module):
    """
    ``nn.Linear`` wrapper adding a trainable low-rank path.

    Args:
        base: The frozen linear layer to adapt (kept by reference).
        r: Rank of the update. ``0`` disables the adapter (pass-through).
        alpha: Scaling numerator; effective scale is ``alpha / r``.
        dropout: Dropout on the input of the LoRA branch.
        init: ``"kaiming"`` (A ~ Kaiming-uniform, B = 0, the paper default) or
            ``"gaussian"``. B starts at zero so the adapted model equals the base
            model at step 0.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init: str = "kaiming",
    ) -> None:
        super().__init__()
        if r < 0:
            raise ValueError("r must be >= 0")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            if init == "kaiming":
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            elif init == "gaussian":
                nn.init.normal_(self.lora_A, std=1.0 / r)
            else:
                raise ValueError(f"unknown init {init!r}")
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def delta_weight(self) -> Tensor:
        """``scaling * B @ A`` in the base weight's dtype."""
        assert self.lora_A is not None and self.lora_B is not None
        return (self.lora_B @ self.lora_A * self.scaling).to(self.base.weight.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into ``base.weight`` (inference speed = base model)."""
        if self.r > 0 and not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.r > 0 and self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.r == 0 or self.merged:
            return out
        assert self.lora_A is not None and self.lora_B is not None
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + lora * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def _matches(name: str, patterns: Iterable[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(leaf, p) for p in patterns)


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    *,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    init: str = "kaiming",
) -> list[str]:
    """
    Replace matching ``nn.Linear`` sub-modules with :class:`LoRALinear` in place.

    ``target_modules`` are glob patterns matched against either the full dotted
    name (``"layers.0.attn.q_proj"``) or the leaf name (``"q_proj"``).

    Returns:
        Names of the modules that were wrapped.
    """
    patterns = list(target_modules)
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if (
            isinstance(module, nn.Linear)
            and not isinstance(module, LoRALinear)
            and _matches(name, patterns)
        ):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout, init=init))
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"No nn.Linear matched target_modules={patterns}")
    return wrapped


def lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield name, m


def mark_only_lora_trainable(model: nn.Module, *, train_bias: bool = False) -> int:
    """Freeze everything except LoRA ``A``/``B`` (and optionally all biases). Returns trainable count."""
    for name, p in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        p.requires_grad_(is_lora or (train_bias and name.endswith("bias")))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the adapter weights - what you save/share instead of the full model."""
    return {
        k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k
    }


def merge_lora(model: nn.Module) -> int:
    """Merge every adapter into its base weight. Returns number merged."""
    n = 0
    for _, m in lora_modules(model):
        m.merge()
        n += 1
    return n


def unmerge_lora(model: nn.Module) -> int:
    n = 0
    for _, m in lora_modules(model):
        m.unmerge()
        n += 1
    return n
