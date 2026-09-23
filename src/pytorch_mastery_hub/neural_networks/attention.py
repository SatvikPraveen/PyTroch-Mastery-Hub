"""
Modern transformer building blocks (LLaMA/Mistral-style).

Everything here runs on :func:`torch.nn.functional.scaled_dot_product_attention`
so it automatically dispatches to FlashAttention-2 / memory-efficient kernels on
supported GPUs and to a math fallback elsewhere.

Components:

* :class:`RMSNorm` - root-mean-square layer norm (no mean subtraction, no bias).
* :class:`RotaryEmbedding` - rotary position embeddings (RoPE) with cached
  cos/sin tables and optional NTK-style base scaling.
* :class:`KVCache` - pre-allocated key/value cache for autoregressive decoding.
* :class:`MultiHeadAttention` - multi-head, multi-query or grouped-query
  attention (``num_kv_heads``) with RoPE, causal masking, padding masks and
  KV-cache support.
* :class:`SwiGLU` - gated feed-forward network used by LLaMA/PaLM.
* :class:`DecoderBlock` - pre-norm block combining the above.
* :class:`TransformerLM` - tiny decoder-only language model with greedy /
  sampled :meth:`generate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DecoderBlock",
    "KVCache",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TransformerConfig",
    "TransformerLM",
    "apply_rotary",
    "build_causal_mask",
]


class RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * weight`` - cheaper than LayerNorm, same effect."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.numel()}, eps={self.eps}"


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Rotate ``x`` (``[B, H, T, D]``) by position-dependent angles.

    ``cos``/``sin`` are ``[T, D]`` (or broadcastable) tables from
    :class:`RotaryEmbedding`.
    """
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings (Su et al., 2021).

    Encodes *relative* position by rotating query/key pairs of dimensions;
    the dot product between a query at position ``m`` and key at ``n`` then
    depends only on ``m - n``.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Length of the pre-computed table (grown on demand).
        base: Frequency base (10 000 in the paper; larger extends context).
        scaling: NTK-aware scaling factor for context extension (1.0 = off).
    """

    def __init__(
        self, dim: int, max_seq_len: int = 4096, base: float = 10_000.0, scaling: float = 1.0
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even")
        self.dim = dim
        self.base = base * scaling ** (dim / (dim - 2)) if scaling != 1.0 else base
        self.inv_freq: Tensor
        self.cos_cached: Tensor
        self.sin_cached: Tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_seq_len)

    def _build(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self, seq_len: int, offset: int = 0, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` tables of shape ``[seq_len, dim]`` starting at ``offset``."""
        needed = offset + seq_len
        if needed > self.max_seq_len:
            self._build(max(needed, 2 * self.max_seq_len))
        cos = self.cos_cached[offset:needed]
        sin = self.sin_cached[offset:needed]
        if device is not None and cos.device != device:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


@dataclass
class KVCache:
    """
    Pre-allocated key/value cache for one attention layer.

    Shapes are ``[batch, num_kv_heads, max_seq_len, head_dim]``. ``update``
    writes new keys/values at ``pos`` and returns the valid prefix.
    """

    k: Tensor
    v: Tensor
    pos: int = 0

    @classmethod
    def empty(
        cls,
        batch: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> KVCache:
        shape = (batch, num_kv_heads, max_seq_len, head_dim)
        return cls(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        )

    @property
    def max_seq_len(self) -> int:
        return int(self.k.shape[2])

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        t = k.shape[2]
        if self.pos + t > self.max_seq_len:
            raise ValueError(f"KV cache overflow: {self.pos + t} > {self.max_seq_len}")
        self.k[:, :, self.pos : self.pos + t] = k
        self.v[:, :, self.pos : self.pos + t] = v
        self.pos += t
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]

    def reset(self) -> None:
        self.pos = 0


def build_causal_mask(q_len: int, k_len: int, device: torch.device | None = None) -> Tensor:
    """Boolean ``[q_len, k_len]`` mask: True where query ``i`` may attend key ``j``."""
    offset = k_len - q_len
    return torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(diagonal=offset)


class MultiHeadAttention(nn.Module):
    """
    Grouped-query attention with rotary embeddings on fused SDPA.

    Args:
        d_model: Model width.
        num_heads: Query heads.
        num_kv_heads: Key/value heads. ``num_heads`` = MHA, ``1`` = multi-query,
            in between = grouped-query (LLaMA-2 70B, Mistral).
        dropout: Attention-probability dropout (training only).
        bias: Whether projections have biases.
        rope: Apply rotary embeddings to q/k.
        max_seq_len: Initial RoPE table length.
        causal: Default masking mode for :meth:`forward`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        rope: bool = True,
        rope_base: float = 10_000.0,
        max_seq_len: int = 4096,
        causal: bool = True,
    ) -> None:
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.causal = causal

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base) if rope else None

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        cache: KVCache | None = None,
        causal: bool | None = None,
    ) -> Tensor:
        """
        Args:
            x: ``[B, T, d_model]`` input (self-attention).
            key_padding_mask: ``[B, S]`` bool, True for *valid* keys (HF style
                ``attention_mask``). Positions marked False are never attended.
            attn_mask: Optional ``[T, S]`` or ``[B, 1, T, S]`` bool mask, True =
                attend. Combined with the causal and padding masks.
            cache: If given, keys/values are appended and attention runs over the
                full cached prefix (autoregressive decoding).
            causal: Override the layer default.
        """
        bsz, t, _ = x.shape
        causal = self.causal if causal is None else causal
        offset = cache.pos if cache is not None else 0

        q = self.q_proj(x).view(bsz, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(t, offset, device=x.device)
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)
        s = k.shape[2]

        if self.groups > 1:  # expand kv heads to match query heads
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        mask = self._build_mask(t, s, key_padding_mask, attn_mask, causal, x.device)
        # SDPA's is_causal fast path only applies to square, unmasked attention.
        use_is_causal = mask is None and causal and t == s and t > 1
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_is_causal,
        )
        out = out.transpose(1, 2).reshape(bsz, t, self.num_heads * self.head_dim)
        return self.o_proj(out)

    @staticmethod
    def _build_mask(
        t: int,
        s: int,
        key_padding_mask: Tensor | None,
        attn_mask: Tensor | None,
        causal: bool,
        device: torch.device,
    ) -> Tensor | None:
        if key_padding_mask is None and attn_mask is None:
            if causal and t != s:  # decoding with a cache: one query row over s keys
                return build_causal_mask(t, s, device)
            return None
        mask = torch.ones(1, 1, t, s, dtype=torch.bool, device=device)
        if causal:
            mask = mask & build_causal_mask(t, s, device)
        if key_padding_mask is not None:
            mask = mask & key_padding_mask.bool()[:, None, None, :]
        if attn_mask is not None:
            am = attn_mask.bool()
            mask = mask & (am if am.dim() == 4 else am[None, None])
        # Guard fully-masked rows (all-padding) against NaNs by allowing self-attention.
        fully_masked = ~mask.any(dim=-1, keepdim=True)
        if fully_masked.any():
            eye = torch.eye(t, s, dtype=torch.bool, device=device)[None, None]
            mask = mask | (fully_masked & eye)
        return mask


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` feed-forward (Shazeer 2020)."""

    def __init__(
        self, d_model: int, hidden: int | None = None, bias: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        # LLaMA sizes the hidden dim to 2/3 * 4d rounded to a multiple of 256.
        hidden = hidden or int(2 * (4 * d_model) / 3)
        hidden = ((hidden + 255) // 256) * 256 if hidden > 256 else hidden
        self.gate = nn.Linear(d_model, hidden, bias=bias)
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        causal: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, norm_eps)
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=causal,
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        x = x + self.resid_dropout(
            self.attn(
                self.attn_norm(x),
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                cache=cache,
            )
        )
        return x + self.ffn(self.ffn_norm(x))


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int | None = None
    ffn_hidden: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-6


class TransformerLM(nn.Module):
    """
    Decoder-only language model over token ids.

    ``forward(ids)`` returns logits ``[B, T, vocab]``; :meth:`generate` does
    KV-cached autoregressive sampling.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(
                c.d_model,
                c.num_heads,
                c.num_kv_heads,
                ffn_hidden=c.ffn_hidden,
                dropout=c.dropout,
                max_seq_len=c.max_seq_len,
                norm_eps=c.norm_eps,
            )
            for _ in range(c.num_layers)
        )
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)
        # GPT-2 style: scale residual projections by depth.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.num_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_kv_heads or self.config.num_heads

    @property
    def head_dim(self) -> int:
        return self.config.d_model // self.config.num_heads

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> Tensor:
        if input_ids.shape[1] > self.config.max_seq_len and caches is None:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len {self.config.max_seq_len}"
            )
        x = self.drop(self.embed(input_ids))
        for i, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask=attention_mask, cache=caches[i] if caches else None)
        return self.lm_head(self.norm(x))

    def new_caches(
        self, batch: int, max_seq_len: int | None = None, dtype: torch.dtype | None = None
    ) -> list[KVCache]:
        device = self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        return [
            KVCache.empty(
                batch,
                self.num_kv_heads,
                max_seq_len or self.config.max_seq_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in self.layers
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Autoregressively extend ``input_ids`` (``[B, T]``) using a KV cache.

        ``temperature=0`` gives greedy decoding. ``top_k``/``top_p`` apply
        nucleus / top-k filtering before sampling.
        """
        self.eval()
        bsz, t = input_ids.shape
        caches = self.new_caches(bsz, max_seq_len=t + max_new_tokens)
        logits = self(input_ids, caches=caches)[:, -1]
        out = input_ids
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            next_token = _sample(logits, temperature, top_k, top_p, generator)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
                finished |= next_token == eos_token_id
            out = torch.cat([out, next_token[:, None]], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
            logits = self(next_token[:, None], caches=caches)[:, -1]
        return out


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p  # keep tokens until mass exceeds p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
