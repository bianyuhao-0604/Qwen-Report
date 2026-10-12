# ch7_sampling.py -- temperature / top-k / top-p: three decoding interventions
import torch
import torch.nn.functional as F

torch.manual_seed(42)

tokens = ["the", "cat", "dog", "on", "fish"]         # toy vocabulary (5 words)
logits = torch.tensor([4.0, 2.0, 1.0, 0.5, -1.0])

# ---------- Temperature: sharpens or flattens the distribution ----------
for T in (0.7, 1.0, 1.5):
    p = F.softmax(logits / T, dim=-1)
    row = "  ".join(f"{t}:{v:.2f}" for t, v in zip(tokens, p))
    print(f"T = {T:<4}  {row}")

# ---------- Top-k: keep only the k most probable tokens ----------
k = 2
topv, topi = torch.topk(logits, k)
mask = torch.full_like(logits, float("-inf"))
mask[topi] = logits[topi]
p_topk = F.softmax(mask, -1)
print(f"after top-k={k}:  "
      + "  ".join(f"{t}:{v:.2f}" for t, v in zip(tokens, p_topk)))

# ---------- Top-p: keep the smallest set whose cumulative mass reaches p ----------
def top_p_mask(logits, p):
    order = torch.argsort(logits, descending=True)
    probs = F.softmax(logits[order], -1)
    cum = torch.cumsum(probs, -1)
    keep = order[cum - probs < p]                    # the batch before crossing p
    m = torch.full_like(logits, float("-inf"))
    m[keep] = logits[keep]
    return m

for p in (0.6, 0.9):
    m = top_p_mask(logits, p)
    kept = [t for t, v in zip(tokens, m) if v != float("-inf")]
    print(f"top-p={p} keeps {kept}")

# ---------- Full sampling: temperature 0.8 + top-p 0.9, 12 draws ----------
m = top_p_mask(logits / 0.8, 0.9)
samples = [tokens[int(torch.multinomial(F.softmax(m, -1), 1))]
           for _ in range(12)]
print("12 samples:", " ".join(samples))
