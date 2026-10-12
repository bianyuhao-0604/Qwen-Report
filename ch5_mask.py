# ch5_mask.py -- the core SFT trick: compute the loss on the answer only
# (prompt masking). Place in the same directory as ch2_arch.py.
import torch
import torch.nn.functional as F
from ch2_arch import MiniQwen, QwenConfig

torch.manual_seed(42)

# ---------- Qwen ChatML template (toy version; markers treated as tokens) ----------
IM_U, IM_S, IM_E = "<|user|>", "<|assistant|>", "<|end|>"
example = [(IM_U + "\n1+1=\n" + IM_E + "\n" + IM_S + "\n2" + IM_E)]

# ---------- Char-level vocabulary + 3 special markers ----------
specials = [IM_U, IM_S, IM_E, "\n"]
alphabet = sorted(set("0123456789+-*/=?. abcdefghijklmnopqrstuvwxyz"))
vocab = specials + alphabet
stoi = {t: i for i, t in enumerate(vocab)}

def encode(s):                                       # split markers first, then chars
    for sp in specials[:-1]:
        s = s.replace(sp, f"\x00{sp}\x00")
    out = []
    for part in s.split("\x00"):
        out.append([stoi[part]] if part in stoi else [stoi[c] for c in part])
    return [t for blk in out for t in blk]

ids = encode(example[0])
split = ids.index(stoi[IM_S]) + 2                    # tokens after IM_S + \n are the answer

# Build labels: set every prompt position to -100 (no loss there)
labels = ids.copy()
for i in range(split):
    labels[i] = -100
tok_line, lab_line = [], []
for t, l in zip(ids, labels):
    tok_line.append(vocab[t] if vocab[t] != "\n" else "\\n")
    lab_line.append("-100" if l == -100 else vocab[l])
print("token : " + " ".join(f"{t:>10}" for t in tok_line[:14]))
print("label : " + " ".join(f"{l:>10}" for l in lab_line[:14]))
print(f"total tokens = {len(ids)}, supervised = {len(ids) - split} (answer part only)")

# ---------- Loss with vs. without the mask (random-init model; checks -100) ----------
cfg = QwenConfig(vocab=len(vocab), d_model=64, n_heads=4, n_kv_heads=2)
model = MiniQwen(cfg)
x = torch.tensor([ids]); y = torch.tensor([labels])
_, loss_masked = model(x, y)
_, loss_all = model(x, x)                            # no mask
print(f"loss without mask = {loss_all:.3f}   loss with mask = {loss_masked:.3f}"
      f" (the latter averages over answer tokens only)")

# Equivalence check: hand-written "shift + ignore -100" cross entropy
logits = model(x)[0]
manual = F.cross_entropy(logits[0, :-1], y[0, 1:], ignore_index=-100)
print(f"manual cross entropy = {manual:.3f} (must match the masked loss)")
