# ch5_lora.py -- LoRA: train two tiny matrices, freeze everything else
# (place in the same directory as ch2_arch.py)
import copy
import torch
import torch.nn as nn
from ch2_arch import MiniQwen, QwenConfig

torch.manual_seed(42)


class LoRALinear(nn.Module):
    """Frozen W plus a low-rank bypass delta W = B·A. Input x: [..., in]."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base                                # original weights, frozen
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # Gaussian init
        self.B = nn.Parameter(torch.zeros(d_out, r))         # zero init
        self.scale = alpha / r

    def forward(self, x):
        delta = (x @ self.A.T @ self.B.T) * self.scale  # low-rank update
        return self.base(x) + delta


def inject_lora(model, r=8):                         # freeze all, then swap q/o to LoRA
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.layers:
        block.self_attn.q_proj = LoRALinear(block.self_attn.q_proj, r)
        block.self_attn.o_proj = LoRALinear(block.self_attn.o_proj, r)
    return model


cfg = QwenConfig(vocab=256, d_model=64, n_heads=4, n_kv_heads=2)
model = MiniQwen(cfg)
n_total = sum(p.numel() for p in model.parameters())

model_lora = inject_lora(copy.deepcopy(model), r=8)
n_train_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
print(f"total params          = {n_total:,}")
print(f"full fine-tuning      = {n_total:,} trainable (100%)")
print(f"LoRA trainable        = {n_train_lora:,} "
      f"({n_train_lora / n_total:.1%}, r=8 adapters on q/o only)")

# Key property 1: B starts at zero → training starts exactly at the base model
x = torch.randn(2, 10, 64)
lora = model_lora.layers[0].self_attn.q_proj
with torch.no_grad():
    delta = lora(x) - lora.base(x)
print(f"bypass output |Δ| at start = {delta.abs().max():.1e} (B=0 guarantees a lossless start)")

# Key property 2: merge before deployment, zero extra inference cost
W = model_lora.layers[0].self_attn.q_proj            # W = base + scale·B·A
merged = W.base.weight.data + W.scale * (W.B @ W.A)
print(f"merged weight keeps its shape = {tuple(merged.shape)} (no bypass at inference)")
