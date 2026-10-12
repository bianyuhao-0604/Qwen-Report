# ch6_dpo.py -- DPO: no reward model needed; update the policy from
# preference pairs directly
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Scenario: for the same prompt, the model's logit of a response token
# (single-token simplified example)
logits_pi = torch.tensor([2.2, 1.8])   # policy [chosen (good), rejected (bad)]
logits_ref = torch.tensor([2.0, 1.9])  # reference model (SFT), same position
beta = 0.1                             # strength of the penalty for drifting away

def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    """L = -log σ( β·[ (logπ_c - logπ_r) - (logπref_c - logπref_r) ] )"""
    gap_pi = (pi_c - pi_r).sum()
    gap_ref = (ref_c - ref_r).sum()
    return -F.logsigmoid(beta * (gap_pi - gap_ref))

loss = dpo_loss(logits_pi[0], logits_pi[1], logits_ref[0], logits_ref[1], beta)
print(f"policy preference gap   logπ(y+)/logπ(y-) = {logits_pi[0] - logits_pi[1]:.2f}")
print(f"reference preference gap logπref(y+)/logπref(y-) = {logits_ref[0] - logits_ref[1]:.2f}")
print(f"DPO loss (β=0.1) = {loss.item():.4f}")

# Gradient check: the good answer's logit should rise, the bad one's should fall
logits_pi = logits_pi.clone().requires_grad_(True)
loss = dpo_loss(logits_pi[0], logits_pi[1], logits_ref[0], logits_ref[1], beta)
loss.backward()
g = logits_pi.grad
print(f"∂L/∂logπ(y+) = {g[0]:+.4f} (negative → rises), "
      f"∂L/∂logπ(y-) = {g[1]:+.4f} (positive → falls)")

# Role of β: larger β hugs the reference model more; smaller β drifts further
for b in (0.05, 0.1, 0.5, 2.0):
    print(f"β = {b:<4} DPO loss = "
          f"{dpo_loss(logits_pi[0], logits_pi[1], logits_ref[0], logits_ref[1], b):.4f}")
