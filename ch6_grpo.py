# ch6_grpo.py -- GRPO: group-relative advantages, a policy gradient without a
# critic (single-step decision simplification)
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Task: given an arithmetic question, pick the answer digit in {0..9} in one
# step. Reward = 1 if correct, 0 otherwise.
QUESTIONS = [("1+1=", 2), ("2+3=", 5), ("6-4=", 2), ("3*3=", 9),
             ("8-5=", 3), ("2*4=", 8), ("7-6=", 1), ("4+4=", 8)]
chars = sorted(set("".join(q for q, _ in QUESTIONS)) | set("0123456789"))
stoi = {c: i for i, c in enumerate(chars)}
ANS_VOCAB = 10                                       # output space = digits 0-9


class TinyPolicy(nn.Module):                         # pooled prompt → linear → digit dist
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(chars), 64)
        self.head = nn.Linear(64, ANS_VOCAB)

    def forward(self, q):                            # q: char-id sequence
        h = self.emb(q).mean(0)                      # [64]
        return self.head(h)                          # [10]


G, ITERS, K = 8, 200, 5                              # group size / iters / sampling temp
policy = TinyPolicy()
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
hist = []
for it in range(1, ITERS + 1):
    logliks, advs, rewards = [], [], []
    for q, ans in QUESTIONS:
        idx = torch.tensor([stoi[c] for c in q])
        logits = policy(idx)
        dist = torch.distributions.Categorical(logits=logits / K)
        acts = dist.sample((G,))                     # sample G answers per group
        r = (acts == ans).float()                    # rewards within the group
        adv = (r - r.mean()) / (r.std() + 1e-4)      # GRPO group-normalized advantage
        logp = dist.log_prob(acts)
        logliks.append(logp); advs.append(adv); rewards.append(r.mean())

    loss = -(torch.cat(logliks) * torch.cat(advs)).mean()   # policy gradient
    opt.zero_grad(); loss.backward(); opt.step()
    mean_r = torch.stack(rewards).mean().item()
    hist.append(mean_r)
    if it % 25 == 0 or it == 1:
        print(f"iter {it:>3}  mean reward = {mean_r:.3f}  (random baseline ≈ 0.1)")

# Greedy decisions after training
print("-" * 40)
acc = 0
for q, ans in QUESTIONS:
    idx = torch.tensor([stoi[c] for c in q])
    pred = int(policy(idx).argmax())
    acc += pred == ans
    print(f"{q:<5} → {pred} (correct {ans})")
print(f"greedy accuracy = {acc}/{len(QUESTIONS)}")

# ---------- Reward curve ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
k = 5                                                # moving average for readability
smooth = [sum(hist[max(0, i - k):i + 1]) / len(hist[max(0, i - k):i + 1])
          for i in range(len(hist))]
plt.figure(figsize=(7, 3.4))
plt.plot(range(1, ITERS + 1), hist, color="#B9C4CE", lw=0.8, alpha=0.9)
plt.plot(range(1, ITERS + 1), smooth, color="#2C6E9E", lw=2.2, label="mean reward (moving average)")
plt.axhline(0.1, color="#D1495B", ls="--", lw=1.2, label="random baseline 0.1")
plt.annotate(f"{smooth[-1]:.2f}", (ITERS, smooth[-1]), textcoords="offset points",
             xytext=(-4, 8), fontsize=9, color="#2C6E9E", ha="right")
plt.xlabel("GRPO iteration"); plt.ylabel("group mean reward")
plt.ylim(0, 1.05); plt.legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("../figs/reward_curve.png")
print("reward curve saved → ../figs/reward_curve.png")
