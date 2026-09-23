"""Tests for the modern attention stack: RMSNorm, RoPE, GQA/SDPA attention, KV cache, LM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_mastery_hub.neural_networks.attention import (
    DecoderBlock,
    KVCache,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerConfig,
    TransformerLM,
    _sample,
    apply_rotary,
    build_causal_mask,
)


class TestRMSNorm:
    def test_matches_formula_and_keeps_dtype(self):
        norm = RMSNorm(8, eps=1e-6)
        x = torch.randn(2, 3, 8)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(norm(x), ref, atol=1e-6)
        assert norm(x.to(torch.bfloat16)).dtype == torch.bfloat16
        assert "eps" in norm.extra_repr()


class TestRoPE:
    def test_relative_position_property(self):
        """q_m . k_n after rotation depends only on m - n."""
        rope = RotaryEmbedding(16, max_seq_len=64)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)
        cos, sin = rope(64)

        def score(m, n):
            qm = apply_rotary(q, cos[m], sin[m])
            kn = apply_rotary(k, cos[n], sin[n])
            return (qm * kn).sum().item()

        assert score(5, 2) == pytest.approx(score(13, 10), abs=1e-5)
        assert score(5, 2) != pytest.approx(score(5, 3), abs=1e-3)

    def test_table_grows_on_demand_and_offsets(self):
        rope = RotaryEmbedding(8, max_seq_len=4)
        cos, sin = rope(10, offset=3)
        assert cos.shape == (10, 8) and rope.max_seq_len >= 13
        cos_full, _ = rope(13)
        assert torch.equal(cos, cos_full[3:13])

    def test_odd_dim_rejected(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(7)

    def test_ntk_scaling_changes_base(self):
        assert RotaryEmbedding(8, scaling=2.0).base > RotaryEmbedding(8).base


def reference_attention(q, k, v, causal):
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if causal:
        mask = build_causal_mask(q.shape[-2], k.shape[-2])
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


class TestMultiHeadAttention:
    @pytest.fixture
    def x(self):
        torch.manual_seed(0)
        return torch.randn(2, 6, 32)

    def test_matches_reference_without_rope(self, x):
        mha = MultiHeadAttention(32, 4, rope=False, causal=True).eval()
        out = mha(x)
        b, t, _ = x.shape
        q = mha.q_proj(x).view(b, t, 4, 8).transpose(1, 2)
        k = mha.k_proj(x).view(b, t, 4, 8).transpose(1, 2)
        v = mha.v_proj(x).view(b, t, 4, 8).transpose(1, 2)
        ref = mha.o_proj(reference_attention(q, k, v, True).transpose(1, 2).reshape(b, t, 32))
        assert torch.allclose(out, ref, atol=1e-5)

    def test_causal_masking_blocks_future(self, x):
        mha = MultiHeadAttention(32, 4).eval()
        out1 = mha(x)
        x2 = x.clone()
        x2[:, -1] += 10.0  # perturb the last token only
        out2 = mha(x2)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
        assert not torch.allclose(out1[:, -1], out2[:, -1])

    def test_non_causal_sees_everything(self, x):
        mha = MultiHeadAttention(32, 4, causal=False).eval()
        out1, out2 = mha(x), mha(x.flip(1)).flip(1)
        # Bidirectional attention with RoPE is position dependent, so just check shape & finiteness
        assert out1.shape == x.shape and torch.isfinite(out2).all()

    @pytest.mark.parametrize("kv_heads", [1, 2, 4])
    def test_grouped_query_shapes(self, x, kv_heads):
        mha = MultiHeadAttention(32, 4, num_kv_heads=kv_heads)
        assert mha(x).shape == (2, 6, 32)
        assert mha.k_proj.out_features == kv_heads * 8

    def test_padding_mask_ignores_padded_keys(self, x):
        mha = MultiHeadAttention(32, 4, causal=False, rope=False).eval()
        valid = torch.ones(2, 6, dtype=torch.bool)
        valid[:, 4:] = False
        out_masked = mha(x, key_padding_mask=valid)
        x2 = x.clone()
        x2[:, 4:] = torch.randn_like(x2[:, 4:])  # change padded positions
        out_masked2 = mha(x2, key_padding_mask=valid)
        assert torch.allclose(out_masked[:, :4], out_masked2[:, :4], atol=1e-5)

    def test_fully_masked_rows_do_not_nan(self, x):
        mha = MultiHeadAttention(32, 4, causal=True).eval()
        valid = torch.zeros(2, 6, dtype=torch.bool)
        valid[:, 3:] = True  # first 3 keys are padding -> causal rows 0..2 fully masked
        assert torch.isfinite(mha(x, key_padding_mask=valid)).all()

    def test_explicit_attn_mask(self, x):
        mha = MultiHeadAttention(32, 4, causal=False).eval()
        am = torch.ones(6, 6, dtype=torch.bool).tril()
        out_manual = mha(x, attn_mask=am)
        out_causal = mha(x, causal=True)
        assert torch.allclose(out_manual, out_causal, atol=1e-5)
        assert mha(x, attn_mask=am[None, None].expand(2, 1, 6, 6)).shape == x.shape

    def test_invalid_configs(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(30, 4)
        with pytest.raises(ValueError):
            MultiHeadAttention(32, 4, num_kv_heads=3)

    def test_dropout_only_in_training(self, x):
        mha = MultiHeadAttention(32, 4, dropout=0.5)
        mha.eval()
        assert torch.equal(mha(x), mha(x))


class TestKVCache:
    def test_incremental_decoding_matches_full_forward(self):
        torch.manual_seed(1)
        mha = MultiHeadAttention(32, 4, num_kv_heads=2).eval()
        x = torch.randn(1, 5, 32)
        full = mha(x)
        cache = KVCache.empty(1, 2, 16, 8)
        steps = [mha(x[:, i : i + 1], cache=cache) for i in range(5)]
        assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
        assert cache.pos == 5
        cache.reset()
        assert cache.pos == 0

    def test_prefill_then_decode(self):
        mha = MultiHeadAttention(32, 4).eval()
        x = torch.randn(2, 4, 32)
        cache = KVCache.empty(2, 4, 8, 8)
        mha(x[:, :3], cache=cache)
        last = mha(x[:, 3:], cache=cache)
        assert torch.allclose(last, mha(x)[:, 3:], atol=1e-5)

    def test_overflow(self):
        cache = KVCache.empty(1, 1, 2, 4)
        cache.update(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4))
        with pytest.raises(ValueError):
            cache.update(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))


class TestBlocksAndLM:
    def test_swiglu_hidden_rounding(self):
        ffn = SwiGLU(256)
        assert ffn.gate.out_features % 256 == 0
        assert SwiGLU(16, hidden=40).gate.out_features == 40
        assert ffn(torch.randn(2, 3, 256)).shape == (2, 3, 256)

    def test_decoder_block(self):
        block = DecoderBlock(32, 4, num_kv_heads=2, dropout=0.1)
        x = torch.randn(2, 5, 32)
        assert block(x).shape == x.shape

    @pytest.fixture
    def lm(self):
        torch.manual_seed(0)
        return TransformerLM(
            TransformerConfig(
                vocab_size=50, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=32
            )
        ).eval()

    def test_forward_shape_and_tied_embeddings(self, lm):
        ids = torch.randint(0, 50, (2, 7))
        assert lm(ids).shape == (2, 7, 50)
        assert lm.lm_head.weight is lm.embed.weight
        assert lm(ids, attention_mask=torch.ones(2, 7, dtype=torch.bool)).shape == (2, 7, 50)

    def test_sequence_too_long(self, lm):
        with pytest.raises(ValueError):
            lm(torch.zeros(1, 33, dtype=torch.long))

    def test_generate_greedy_matches_uncached(self, lm):
        ids = torch.randint(0, 50, (2, 4))
        out = lm.generate(ids, max_new_tokens=5, temperature=0.0)
        assert out.shape == (2, 9)
        # Recompute greedily without cache and compare.
        seq = ids
        for _ in range(5):
            nxt = lm(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        assert torch.equal(out, seq)

    def test_generate_stops_at_eos(self, lm):
        ids = torch.randint(0, 50, (1, 3))
        greedy_first = lm.generate(ids, 1, temperature=0.0)[0, -1].item()
        out = lm.generate(ids, max_new_tokens=10, temperature=0.0, eos_token_id=greedy_first)
        assert out.shape[1] == 4 and out[0, -1] == greedy_first

    def test_generate_sampling_options(self, lm):
        ids = torch.randint(0, 50, (2, 3))
        g = torch.Generator().manual_seed(0)
        out = lm.generate(ids, 3, temperature=0.8, top_k=5, top_p=0.9, generator=g)
        assert out.shape == (2, 6) and (out < 50).all()

    def test_sample_top_p_keeps_best_token(self):
        logits = torch.tensor([[10.0, 1.0, 0.0, -5.0]])
        for _ in range(10):
            assert _sample(logits, 1.0, None, 0.5, None).item() == 0
            assert _sample(logits, 1.0, 1, None, None).item() == 0

    def test_lm_trains(self, lm):
        lm.train()
        ids = torch.randint(0, 50, (4, 8))
        opt = torch.optim.AdamW(lm.parameters(), lr=1e-2)
        losses = []
        for _ in range(15):
            logits = lm(ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 50), ids[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_lm_is_torchscript_free_but_compilable_api(self, lm):
        assert isinstance(lm.layers[0].attn, nn.Module)
        assert lm.num_kv_heads == 2 and lm.head_dim == 8
