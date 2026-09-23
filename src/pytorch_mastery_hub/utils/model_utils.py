"""
Model introspection and manipulation helpers.

* :func:`count_parameters` / :func:`get_model_size` for quick sizing.
* :func:`model_summary` prints a per-layer table (shapes + params) using
  forward hooks, without any third-party dependency.
* :func:`freeze` / :func:`unfreeze` for transfer learning.
* :func:`param_groups_with_weight_decay` implements the standard "no decay for
  biases and normalisation weights" optimizer grouping.
* :func:`init_weights` applies a named initialisation scheme to a whole model.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

__all__ = [
    "LayerInfo",
    "count_parameters",
    "freeze",
    "get_model_size",
    "init_weights",
    "model_summary",
    "param_groups_with_weight_decay",
    "summarize",
    "unfreeze",
]

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}

Nonlinearity = Literal["linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu"]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters in ``model`` (optionally only those requiring grad)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def get_model_size(model: nn.Module, unit: str = "MB", include_buffers: bool = True) -> float:
    """
    In-memory footprint of parameters (and buffers) in ``unit``.

    This is the size of the weights themselves, i.e. what ``state_dict``
    serialises to, not activations or optimizer state.
    """
    unit = unit.upper()
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {sorted(_UNITS)}, got {unit!r}")
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    if include_buffers:
        total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / _UNITS[unit]


@dataclass
class LayerInfo:
    """One row of :func:`summarize`."""

    name: str
    type: str
    output_shape: list[int] | None
    params: int
    trainable: int
    children: list[LayerInfo] = field(default_factory=list)


def _shape_of(out: Any) -> list[int] | None:
    if isinstance(out, torch.Tensor):
        return list(out.shape)
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return list(out[0].shape)
    return None


def summarize(
    model: nn.Module,
    input_size: Sequence[int] | Sequence[Sequence[int]] | None = None,
    *,
    input_data: Any = None,
    device: torch.device | str | None = None,
    dtypes: Sequence[torch.dtype] | None = None,
    depth: int = 1,
) -> list[LayerInfo]:
    """
    Collect per-layer output shapes and parameter counts via forward hooks.

    Args:
        model: The module to inspect.
        input_size: Shape of a single dummy input *including* the batch
            dimension, or a list of shapes for multi-input models. Ignored if
            ``input_data`` is given.
        input_data: A ready-made input (tensor or tuple of tensors).
        device: Where to run the dummy forward pass (defaults to the model's).
        dtypes: dtype per input (defaults to ``float32``).
        depth: How many levels of nested modules to report (1 = direct children).

    Returns:
        Flat list of :class:`LayerInfo` in execution order.
    """
    infos: list[LayerInfo] = []
    handles = []

    def register(module: nn.Module, prefix: str, level: int) -> None:
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name

            def hook(mod: nn.Module, _inp: Any, out: Any, full_name: str = full) -> None:
                params = sum(p.numel() for p in mod.parameters(recurse=False))
                trainable = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
                # Leaf modules report their own params; containers report totals.
                if not list(mod.children()):
                    params = sum(p.numel() for p in mod.parameters())
                    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                infos.append(
                    LayerInfo(full_name, type(mod).__name__, _shape_of(out), params, trainable)
                )

            handles.append(child.register_forward_hook(hook))
            if level < depth:
                register(child, full, level + 1)

    register(model, "", 1)

    if input_data is None:
        if input_size is None:
            raise ValueError("Provide either input_size or input_data")
        shapes: list[Sequence[int]]
        if isinstance(input_size[0], int):
            shapes = [input_size]  # type: ignore[list-item]
        else:
            shapes = list(input_size)  # type: ignore[arg-type]
        dtypes = list(dtypes) if dtypes else [torch.float32] * len(shapes)
        dev = torch.device(device) if device is not None else _model_device(model)
        input_data = tuple(torch.zeros(*s, dtype=d, device=dev) for s, d in zip(shapes, dtypes))
    elif not isinstance(input_data, (tuple, list)):
        input_data = (input_data,)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(*input_data)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return infos


def model_summary(
    model: nn.Module,
    input_size: Sequence[int] | Sequence[Sequence[int]] | None = None,
    *,
    input_data: Any = None,
    device: torch.device | str | None = None,
    depth: int = 1,
    print_summary: bool = True,
) -> str:
    """
    Render a Keras/torchsummary-style table of the model.

    Returns the table as a string (and prints it unless ``print_summary=False``).
    When no input is supplied only the parameter columns are filled.
    """
    if input_size is not None or input_data is not None:
        rows = summarize(model, input_size, input_data=input_data, device=device, depth=depth)
    else:
        rows = [
            LayerInfo(
                name,
                type(m).__name__,
                None,
                sum(p.numel() for p in m.parameters()),
                sum(p.numel() for p in m.parameters() if p.requires_grad),
            )
            for name, m in model.named_children()
        ]

    name_w = max([len("Layer (type)")] + [len(f"{r.name} ({r.type})") for r in rows])
    shape_w = max([len("Output shape")] + [len(str(r.output_shape or "-")) for r in rows])
    header = f"{'Layer (type)':<{name_w}}  {'Output shape':<{shape_w}}  {'Params':>12}"
    line = "=" * len(header)
    lines = [line, header, line]
    for r in rows:
        lines.append(
            f"{f'{r.name} ({r.type})':<{name_w}}  {r.output_shape or '-'!s:<{shape_w}}  {r.params:>12,}"
        )
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    lines += [
        line,
        f"Total params:         {total:,}",
        f"Trainable params:     {trainable:,}",
        f"Non-trainable params: {total - trainable:,}",
        f"Model size:           {get_model_size(model):.2f} MB",
        line,
    ]
    text = "\n".join(lines)
    if print_summary:
        print(text)
    return text


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) or name.startswith(p) for p in patterns)


def freeze(model: nn.Module, patterns: Iterable[str] | None = None) -> int:
    """
    Set ``requires_grad=False`` on parameters whose name matches ``patterns``.

    Patterns are glob-style (``"encoder.*"``) or prefixes (``"encoder"``). With
    no patterns the whole model is frozen. Returns the number of tensors frozen.
    """
    pats = list(patterns) if patterns is not None else None
    n = 0
    for name, p in model.named_parameters():
        if pats is None or _matches(name, pats):
            p.requires_grad_(False)
            n += 1
    return n


def unfreeze(model: nn.Module, patterns: Iterable[str] | None = None) -> int:
    """Inverse of :func:`freeze`."""
    pats = list(patterns) if patterns is not None else None
    n = 0
    for name, p in model.named_parameters():
        if pats is None or _matches(name, pats):
            p.requires_grad_(True)
            n += 1
    return n


_NORM_TYPES: tuple[type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
)
_NO_DECAY_TYPES: tuple[type[nn.Module], ...] = (*_NORM_TYPES, nn.Embedding)


def param_groups_with_weight_decay(
    model: nn.Module,
    weight_decay: float,
    *,
    no_decay_names: Sequence[str] = ("bias",),
    extra_no_decay_types: Sequence[type[nn.Module]] = (),
) -> list[dict[str, Any]]:
    """
    Split parameters into decayed / non-decayed optimizer groups.

    Biases, normalisation layers and embeddings are conventionally excluded from
    weight decay (as in BERT, GPT and timm). Returns a list usable directly as
    ``torch.optim.AdamW(param_groups_with_weight_decay(model, 0.01), lr=...)``.
    """
    no_decay_types = _NO_DECAY_TYPES + tuple(extra_no_decay_types)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full = f"{module_name}.{param_name}" if module_name else param_name
            if (
                isinstance(module, no_decay_types)
                or any(full.endswith(s) for s in no_decay_names)
                or param.ndim < 2
            ):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def init_weights(
    model: nn.Module, method: str = "kaiming_normal", nonlinearity: Nonlinearity = "relu"
) -> None:
    """
    Initialise all Linear/Conv weights in-place with a named scheme.

    ``method`` is one of ``xavier_uniform``, ``xavier_normal``,
    ``kaiming_uniform``, ``kaiming_normal``, ``orthogonal``, ``trunc_normal``.
    Biases are zeroed, normalisation layers reset to weight=1/bias=0.
    """
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    for module in model.modules():
        if isinstance(module, (nn.Linear, *conv_types)):
            w = module.weight
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(w)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(w)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(w, nonlinearity=nonlinearity)
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(w, nonlinearity=nonlinearity)
            elif method == "orthogonal":
                nn.init.orthogonal_(w, gain=nn.init.calculate_gain(nonlinearity))
            elif method == "trunc_normal":
                nn.init.trunc_normal_(w, std=0.02)
            else:
                raise ValueError(f"Unknown init method {method!r}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, _NORM_TYPES):
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if isinstance(weight, torch.Tensor):
                nn.init.ones_(weight)
            if isinstance(bias, torch.Tensor):
                nn.init.zeros_(bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
