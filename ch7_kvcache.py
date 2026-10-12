# ch7_kvcache.py -- KV cache: why autoregressive decoding needs it
# (place in the same directory as ch2_arch.py)
import time
import torch
from ch2_arch import MiniQwen, QwenConfig, rope_cache, apply_rope

torch.manual_seed(42)
cfg = QwenConfig(vocab=256, d_model=256, n_layers=4,
                 n_heads=8, n_kv_heads=4, d_ff=704)
model = MiniQwen(cfg).eval()


@torch.no_grad()
def prefill_and_cache(idx):
    """One forward pass over the whole prefix, storing each layer's K/V."""
    T = idx.size(1)
    dh = cfg.d_model // cfg.n_heads
    cos, sin = rope_cache(T + 4, dh, cfg.rope_theta)  # 4 extra slots for decode steps
    mask = torch.triu(torch.ones(1, 1, T, T, dtype=torch.bool), 1)
    past = []
    x = model.embed_tokens(idx)
    for layer in model.layers:
        h = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(h).view(1, T, cfg.n_heads, dh).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(1, T, cfg.n_kv_heads, dh).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(1, T, cfg.n_kv_heads, dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past.append((k, v))                           # ★ this is all we cache
        rep = cfg.n_heads // cfg.n_kv_heads
        att = (q @ k.repeat_interleave(rep, 1).transpose(-2, -1)) / dh ** 0.5
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v.repeat_interleave(rep, 1)).transpose(1, 2).reshape(1, T, -1)
        x = x + layer.self_attn.o_proj(y)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    logits = model.lm_head(model.norm(x))
    return logits[:, -1], past, cos, sin


@torch.no_grad()
def step_cached(tok, pos, past, cos, sin):
    """Forward the newest token only: reuse cached K/V, no recomputation."""
    dh = cfg.d_model // cfg.n_heads
    t = torch.tensor([pos])
    x = model.embed_tokens(tok)
    for li, layer in enumerate(model.layers):
        h = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(h).view(1, 1, cfg.n_heads, dh).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(1, 1, cfg.n_kv_heads, dh).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(1, 1, cfg.n_kv_heads, dh).transpose(1, 2)
        q = apply_rope(q, cos[t], sin[t])
        k = apply_rope(k, cos[t], sin[t])
        pk, pv = past[li]
        rep = cfg.n_heads // cfg.n_kv_heads
        kk = torch.cat([pk, k], 2).repeat_interleave(rep, 1)   # concat K
        vv = torch.cat([pv, v], 2).repeat_interleave(rep, 1)   # concat V
        att = (q @ kk.transpose(-2, -1)) / dh ** 0.5
        att = att.softmax(-1)                       # the last token needs no causal mask
        y = (att @ vv).transpose(1, 2).reshape(1, 1, -1)
        x = x + layer.self_attn.o_proj(y)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(model.norm(x))[:, -1]


@torch.no_grad()
def step_naive(full_idx):
    """Naive route: recompute the whole prefix, take the last logits."""
    return model(full_idx)[0][:, -1]


def bench(fn, n=10):
    fn()                                             # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    return (time.perf_counter() - t0) / n * 1000, out


print(f"{'context':>8} | {'no cache (ms/step)':>19} | {'cache (ms/step)':>17}")
all_eq = True
for T in (64, 128, 256, 512):
    idx = torch.randint(0, 256, (1, T))
    logits_last, past, cos, sin = prefill_and_cache(idx)
    nxt = logits_last.argmax(-1, keepdim=True)       # the new token after the prefix
    t_cache, out_c = bench(lambda: step_cached(nxt, T, past, cos, sin))
    t_naive, out_n = bench(lambda: step_naive(torch.cat([idx, nxt], 1)))
    all_eq &= bool(out_c.argmax() == out_n.argmax())
    print(f"{T:>8} | {t_naive:>19.2f} | {t_cache:>17.2f}")
print(f"next-token identical for both = {all_eq}"
      f" (the cache only saves compute, never changes results)")
