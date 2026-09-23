"""
Property-based tests (Hypothesis) for numerically sensitive helpers.

Instead of a handful of hand-picked inputs, each property is checked on
hundreds of randomly generated tensors, including degenerate shapes and
extreme magnitudes.
"""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from pytorch_mastery_hub.advanced.lora import LoRALinear
from pytorch_mastery_hub.fundamentals.math_utils import (
    cross_entropy,
    log_softmax,
    normalize,
    softmax,
    standardize,
)
from pytorch_mastery_hub.fundamentals.tensor_ops import safe_divide
from pytorch_mastery_hub.neural_networks.attention import (
    KVCache,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary,
)
from pytorch_mastery_hub.utils.device_utils import move_to_device
from pytorch_mastery_hub.utils.metrics import accuracy

settings.register_profile("ci", max_examples=60, deadline=None)
settings.load_profile("ci")

floats = st.floats(-50, 50, allow_nan=False, allow_infinity=False, width=32)
matrices = hnp.arrays(
    np.float32, hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8), elements=floats
)


@given(matrices, st.floats(0.1, 10.0))
def test_softmax_is_a_distribution(arr, temperature):
    x = torch.from_numpy(arr)
    p = softmax(x, dim=-1, temperature=temperature)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0]), atol=1e-5)
    # temperature -> shift-invariance: softmax(x + c) == softmax(x)
    assert torch.allclose(p, softmax(x + 7.0, dim=-1, temperature=temperature), atol=1e-5)


@given(matrices)
def test_log_softmax_matches_log_of_softmax(arr):
    x = torch.from_numpy(arr)
    ls = log_softmax(x)
    assert torch.allclose(ls.exp(), softmax(x), atol=1e-5)
    # log-probabilities of a distribution log-sum-exp to 0
    assert torch.allclose(torch.logsumexp(ls, dim=-1), torch.zeros(x.shape[0]), atol=1e-4)


@given(matrices)
def test_cross_entropy_matches_manual(arr):
    logits = torch.from_numpy(arr)
    n, c = logits.shape
    target = torch.arange(n) % c
    manual = -log_softmax(logits)[torch.arange(n), target].mean()
    assert torch.allclose(cross_entropy(logits, target), manual, atol=1e-5)
    assert cross_entropy(logits, target, reduction="none").shape == (n,)


@given(matrices, st.sampled_from([1.0, 2.0]))
def test_normalize_has_unit_norm(arr, p):
    x = torch.from_numpy(arr)
    y = normalize(x, p=p, dim=-1)
    norms = y.norm(p=p, dim=-1)
    nonzero = x.norm(p=p, dim=-1) > 1e-6
    assert torch.allclose(norms[nonzero], torch.ones_like(norms[nonzero]), atol=1e-4)


@given(
    hnp.arrays(
        np.float32,
        hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=8),
        elements=floats,
    )
)
def test_standardize_zero_mean_unit_var(arr):
    x = torch.from_numpy(arr)
    y = standardize(x, dim=0)
    var_ok = x.std(0) > 1e-2  # degenerate (constant) columns are dominated by eps
    assert torch.allclose(y.mean(0)[var_ok], torch.zeros(int(var_ok.sum())), atol=1e-3)
    assert torch.allclose(y.std(0)[var_ok], torch.ones(int(var_ok.sum())), atol=1e-2)


@given(matrices, matrices)
def test_safe_divide_is_finite(a, b):
    if a.shape != b.shape:
        b = np.resize(b, a.shape).astype(np.float32)
    out = safe_divide(torch.from_numpy(a), torch.from_numpy(b).abs())
    assert torch.isfinite(out).all()


@given(st.integers(1, 32), st.integers(2, 10), st.integers(1, 5))
def test_accuracy_matches_manual(n, c, k):
    k = min(k, c)
    logits = torch.randn(n, c)
    y = torch.randint(0, c, (n,))
    topk = logits.topk(k, dim=1).indices
    manual = (topk == y[:, None]).any(1).float().mean().item()
    assert accuracy(logits, y, topk=k) == pytest.approx(manual)
    assert 0.0 <= accuracy(logits, y) <= 1.0


@given(st.integers(2, 64), st.floats(1.0, 100.0))
def test_rmsnorm_is_scale_invariant(dim, scale):
    norm = RMSNorm(dim)
    x = torch.randn(3, dim)
    assert torch.allclose(norm(x), norm(x * scale), atol=1e-4)


@given(st.sampled_from([4, 8, 16, 32]), st.integers(0, 100))
def test_rotary_preserves_norm(dim, position):
    """A rotation must not change vector length."""
    rope = RotaryEmbedding(dim, max_seq_len=128)
    cos, sin = rope(1, offset=position)
    x = torch.randn(1, 1, 1, dim)
    assert torch.allclose(apply_rotary(x, cos, sin).norm(), x.norm(), atol=1e-5)


@given(st.integers(1, 3), st.integers(1, 6), st.sampled_from([(4, 1), (4, 2), (4, 4)]))
def test_kv_cache_matches_full_attention(batch, seq, heads):
    num_heads, num_kv = heads
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, num_heads, num_kv).eval()
    x = torch.randn(batch, seq, 32)
    cache = KVCache.empty(batch, num_kv, seq, 32 // num_heads)
    steps = torch.cat([mha(x[:, i : i + 1], cache=cache) for i in range(seq)], 1)
    assert torch.allclose(steps, mha(x), atol=1e-5)


@given(st.integers(1, 8), st.floats(0.5, 32.0))
def test_lora_merge_is_lossless(r, alpha):
    torch.manual_seed(0)
    base = nn_linear = torch.nn.Linear(6, 5)
    lora = LoRALinear(nn_linear, r=r, alpha=alpha)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(4, 6)
    y = lora(x)
    lora.merge()
    assert torch.allclose(lora(x), y, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(lora(x), y, atol=1e-5)
    assert base is lora.base


@given(st.lists(st.integers(0, 5), min_size=0, max_size=4))
def test_move_to_device_is_structure_preserving(shape):
    obj = {"t": torch.zeros(shape), "nested": [torch.ones(1), ("s", 3)], "n": None}
    out = move_to_device(obj, "cpu")
    assert out["t"].shape == obj["t"].shape and out["nested"][1] == ("s", 3) and out["n"] is None
