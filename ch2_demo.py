# ch2_demo.py -- verify the four components of MiniQwen one by one
# (place in the same directory as ch2_arch.py)
import math
import torch
from ch2_arch import (QwenConfig, RMSNorm, rope_cache, apply_rope,
                      SwiGLU, Attention, Block, MiniQwen)

torch.manual_seed(42)

# ---------- 1) RMSNorm: divide by the root mean square only ----------
x = torch.randn(1, 4, 64)
rms_before = x.pow(2).mean(-1)                       # mean square per position
y = RMSNorm(64)(x)
rms_after = y.pow(2).mean(-1)
print(f"[RMSNorm]  mean square before ≈ {rms_before.mean():.4f}"
      f"  →  after ≈ {rms_after.mean():.4f}")

# ---------- 2) RoPE: norm preserved + depends only on relative position ----------
dh = 16
q = torch.randn(2, 4, 1, dh).repeat(1, 1, 8, 1)      # one vector at 8 positions
k = torch.randn(2, 4, 1, dh).repeat(1, 1, 8, 1)
cos, sin = rope_cache(8, dh)
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
norm_diff = (qr.norm(dim=-1) - q.norm(dim=-1)).abs().max()
score = (qr @ kr.transpose(-2, -1))                  # score[m,n] = <R_m q, R_n k>
shift = (score[..., 1:, 1:] - score[..., :-1, :-1]).abs().max()
print(f"[RoPE]     max |‖R(q)‖−‖q‖| = {norm_diff:.2e}")
print(f"[RoPE]     shift invariance max |score(m+1,n+1)−score(m,n)| = {shift:.2e}")

# ---------- 3) SwiGLU: shapes through the gated FFN ----------
ffn = SwiGLU(QwenConfig(d_model=64, d_ff=176))
h = torch.randn(1, 8, 64)
out = ffn(h)
print(f"[SwiGLU]   gate/up output [1, 8, 176] → down output {tuple(out.shape)}")

# ---------- 4) GQA + causal mask ----------
cfg = QwenConfig(vocab=256, d_model=64, n_heads=4, n_kv_heads=2)
attn = Attention(cfg)
x = torch.randn(1, 8, 64)
cos, sin = rope_cache(8, 16)
mask = torch.triu(torch.ones(1, 1, 8, 8, dtype=torch.bool), 1)
q = attn.q_proj(x).view(1, 8, 4, 16).transpose(1, 2)
k = attn.k_proj(x).view(1, 8, 2, 16).transpose(1, 2)
print(f"[GQA]      q heads = {q.shape[1]}, k/v heads = {k.shape[1]}"
      f" (every {4 // 2} q heads share 1 KV group)")

# ---------- 5) Whole-model sanity check: params / logits / init loss ----------
model = MiniQwen(cfg)
n_params = sum(p.numel() for p in model.parameters())
idx = torch.randint(0, 256, (1, 8))
targets = torch.randint(0, 256, (1, 8))
logits, loss = model(idx, targets)
print(f"[MiniQwen] params = {n_params:,}; logits shape = {tuple(logits.shape)}")
print(f"[MiniQwen] loss at random init = {loss:.3f} (theory: ln 256 = "
      f"{math.log(256):.3f}, i.e. uniform guessing)")
