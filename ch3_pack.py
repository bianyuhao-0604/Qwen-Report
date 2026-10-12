# ch3_pack.py -- pre-training data packing: fit variable-length docs into
# fixed-length sequences (place in the same directory as ch2_arch.py)
import torch
import torch.nn.functional as F
from ch2_arch import MiniQwen, QwenConfig

torch.manual_seed(42)

# 5 fake "documents": token sequences of different lengths (in reality these
# are the input_ids produced by a tokenizer)
docs = [torch.randint(0, 200, (n,)).tolist()
        for n in (37, 251, 118, 22, 444)]
total = sum(len(d) for d in docs)

# ---- Option A: naive padding (align to the longest doc in the batch) ----
max_len = max(len(d) for d in docs)
padded = max_len * len(docs)                         # slots occupied
waste = 1 - total / padded
print(f"[naive padding] 5 docs hold {total} tokens but occupy {padded} slots"
      f" ({waste:.0%} wasted)")

# ---- Option B: concat-and-chunk (packing, the pre-training standard) ----
BLOCK = 256
flat = [t for d in docs for t in d]                  # 1 separator between docs
n_blocks = len(flat) // BLOCK
pack = torch.tensor(flat[:n_blocks * BLOCK]).view(n_blocks, BLOCK)
print(f"[packing] {len(flat)} tokens → {n_blocks} fixed-length sequences of {BLOCK},"
      f" utilization 100% (tail {len(flat) % BLOCK} tokens dropped)")

# Labels = the input itself (the model shifts internally: position i
# predicts position i+1)
inputs = targets = pack

# Verify: feed MiniQwen and compute one pre-training loss
cfg = QwenConfig(vocab=256, d_model=64, n_heads=4, n_kv_heads=2)
model = MiniQwen(cfg)
idx = inputs.clamp(0, 255)                           # this demo vocab has 256 ids
_, loss = model(idx, targets)
logits = model(idx)[0]                               # verify the loss definition manually
manual = F.cross_entropy(logits[:, :-1].reshape(-1, 256), targets[:, 1:].reshape(-1))
print(f"[check] MiniQwen built-in loss = {loss:.4f}, hand-written CE = {manual:.4f}"
      f" (must match)")
print(f"[check] each sequence of length {BLOCK} supervises {BLOCK - 1} positions"
      f" (position i predicts position i+1)")
