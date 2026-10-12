# ch2_arch.py -- MiniQwen: a minimal runnable Qwen-style decoder (CPU only)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class QwenConfig:
    """Hyperparameter bundle. Deliberately tiny so a forward pass takes
    seconds on a laptop CPU."""
    def __init__(self, vocab=256, d_model=64, n_layers=2, n_heads=4,
                 n_kv_heads=2, d_ff=176, max_seq=128, rope_theta=10000.0):
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.rope_theta = rope_theta


class RMSNorm(nn.Module):
    """Root-mean-square layer norm used by Qwen2/3 (no mean-centering, no bias)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        ms = x.float().pow(2).mean(-1, keepdim=True)   # mean square
        x = x * torch.rsqrt(ms + self.eps)             # divide by the RMS only
        return self.weight * x.to(x.dtype)


def rope_cache(seq_len, d_head, theta=10000.0):
    """Precompute the cos/sin tables of rotary position embedding (RoPE)."""
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv)                        # [T, d_head/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """HF Qwen2-style: rotate the first and second halves. x: [B, H, T, dh]"""
    T = x.size(-2)
    cos = cos[:T][None, None]                          # [1,1,T,dh/2]
    sin = sin[:T][None, None]
    x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """GQA self-attention. Qwen2 keeps bias on Q/K/V projections but not on
    the output projection."""
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.nkv = cfg.n_heads, cfg.n_kv_heads
        self.dh = cfg.d_model // self.nh
        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.dh, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.dh, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.dh, bias=True)
        self.o_proj = nn.Linear(self.nh * self.dh, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        q = apply_rope(q, cos, sin)                    # rotate q/k onto their positions
        k = apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)            # GQA: expand KV heads
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(y), att                     # keep att for inspection


class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) )."""
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm residual block: normalize before each sublayer."""
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        h, _ = self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + h                                      # attention residual
        x = x + self.mlp(self.post_attention_layernorm(x))  # FFN residual
        return x


class MiniQwen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        dh = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cache(T, dh, self.cfg.rope_theta)
        mask = torch.triu(torch.ones(1, 1, T, T, dtype=torch.bool), 1)
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:                        # next-token cross entropy
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1))
        return logits, loss
