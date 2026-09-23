"""Tests for LoRA adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pytorch_mastery_hub.advanced.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
    unmerge_lora,
)
from pytorch_mastery_hub.neural_networks.attention import TransformerConfig, TransformerLM


@pytest.fixture
def base():
    torch.manual_seed(0)
    return nn.Linear(16, 8)


class TestLoRALinear:
    def test_identity_at_init(self, base):
        lora = LoRALinear(base, r=4, alpha=8)
        x = torch.randn(3, 16)
        assert torch.allclose(lora(x), base(x))
        assert not base.weight.requires_grad and lora.lora_A.requires_grad
        assert lora.in_features == 16 and lora.out_features == 8
        assert lora.weight is base.weight and lora.bias is base.bias

    def test_merge_unmerge_roundtrip(self, base):
        lora = LoRALinear(base, r=4, alpha=8)
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(3, 16)
        unmerged = lora(x)
        w0 = base.weight.clone()
        lora.merge()
        assert lora.merged and not torch.allclose(base.weight, w0)
        assert torch.allclose(lora(x), unmerged, atol=1e-5)
        lora.merge()  # idempotent
        lora.unmerge()
        assert torch.allclose(base.weight, w0, atol=1e-6)
        assert "merged=False" in repr(lora)

    def test_rank_zero_passthrough(self, base):
        lora = LoRALinear(base, r=0)
        x = torch.randn(2, 16)
        assert torch.equal(lora(x), base(x))
        lora.merge()  # no-op
        assert lora.lora_A is None

    def test_init_variants_and_errors(self, base):
        LoRALinear(base, r=2, init="gaussian")
        with pytest.raises(ValueError):
            LoRALinear(base, r=2, init="bogus")
        with pytest.raises(ValueError):
            LoRALinear(base, r=-1)

    def test_gradients_only_reach_adapter(self, base):
        lora = LoRALinear(base, r=4, dropout=0.1)
        lora(torch.randn(5, 16)).sum().backward()
        assert base.weight.grad is None
        assert lora.lora_A.grad is not None and lora.lora_B.grad is not None


class TestApplyLoRA:
    @pytest.fixture
    def lm(self):
        torch.manual_seed(0)
        return TransformerLM(
            TransformerConfig(vocab_size=30, d_model=16, num_layers=2, num_heads=2, max_seq_len=16)
        )

    def test_wraps_targets_and_preserves_output(self, lm):
        ids = torch.randint(0, 30, (2, 5))
        before = lm(ids)
        wrapped = apply_lora(lm, ["q_proj", "v_proj"], r=2, alpha=4)
        assert len(wrapped) == 4 and all(w.endswith(("q_proj", "v_proj")) for w in wrapped)
        assert isinstance(lm.layers[0].attn.q_proj, LoRALinear)
        assert isinstance(lm.layers[0].attn.k_proj, nn.Linear) and not isinstance(
            lm.layers[0].attn.k_proj, LoRALinear
        )
        assert torch.allclose(lm(ids), before, atol=1e-6)

    def test_glob_on_full_name(self, lm):
        assert apply_lora(lm, ["layers.1.*.o_proj"], r=2) == ["layers.1.attn.o_proj"]

    def test_no_match_raises(self, lm):
        with pytest.raises(ValueError):
            apply_lora(lm, ["does_not_exist"])

    def test_trainable_params_and_state_dict(self, lm):
        apply_lora(lm, ["q_proj", "v_proj"], r=2)
        n = mark_only_lora_trainable(lm)
        assert n == sum(p.numel() for p in lora_parameters(lm))
        assert n == 4 * (2 * 16 + 16 * 2)
        sd = lora_state_dict(lm)
        assert len(sd) == 8 and all("lora_" in k for k in sd)
        assert mark_only_lora_trainable(lm, train_bias=True) >= n

    def test_training_updates_only_adapters_and_merge(self, lm):
        apply_lora(lm, ["q_proj", "v_proj", "gate", "up"], r=4)
        mark_only_lora_trainable(lm)
        frozen = lm.layers[0].attn.q_proj.base.weight.clone()
        opt = torch.optim.Adam(lora_parameters(lm), lr=1e-2)
        ids = torch.randint(0, 30, (4, 6))
        first = None
        for _ in range(10):
            logits = lm(ids[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, 30), ids[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first or loss.item()
        assert loss.item() < first
        assert torch.equal(frozen, lm.layers[0].attn.q_proj.base.weight)

        lm.eval()
        out = lm(ids)
        assert merge_lora(lm) == 8
        assert torch.allclose(lm(ids), out, atol=1e-5)
        assert unmerge_lora(lm) == 8
        assert torch.allclose(lm(ids), out, atol=1e-5)
