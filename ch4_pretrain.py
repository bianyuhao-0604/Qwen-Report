# ch4_pretrain.py -- a complete pre-training loop on a toy corpus (~20 s on CPU)
import torch
from ch2_arch import MiniQwen, QwenConfig

torch.manual_seed(42)

# ---------- Corpus and vocabulary (char-level, easy to inspect) ----------
corpus = "the cat sat on the mat. the dog sat on the log. " * 40
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in corpus])
print(f"corpus {len(corpus)} chars, vocab size = {len(chars)}")

# ---------- Model and data windows ----------
cfg = QwenConfig(vocab=len(chars), d_model=64, n_layers=2,
                 n_heads=4, n_kv_heads=2, d_ff=176, max_seq=64)
model = MiniQwen(cfg)
print(f"model params = {sum(p.numel() for p in model.parameters()):,}")

def get_batch(bs=8, T=64):
    ix = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[i:i + T] for i in ix])
    return x, x          # labels aligned with the input; the model shifts internally

# ---------- Training: cross entropy + AdamW + cosine decay ----------
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
STEPS, log_at, log_val = 400, [], []
val = data[-300:]                                    # hold out 300 chars for validation
for step in range(1, STEPS + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 40 == 0 or step == 1:
        with torch.no_grad():                        # validation in 64-length windows
            acc, cnt = 0.0, 0
            for i in range(0, len(val) - cfg.max_seq, cfg.max_seq):
                w = val[i:i + cfg.max_seq].unsqueeze(0)
                _, vloss = model(w, w)
                acc += vloss.item(); cnt += 1
            vloss = acc / cnt
        log_at.append((step, loss.item())); log_val.append((step, vloss))
        print(f"step {step:>3}  train {loss.item():.4f}  val {vloss:.4f}")

# ---------- Sampling: the trained model should continue in the corpus style ----------
@torch.no_grad()
def generate(prompt="the cat ", n=60, greedy=True):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits, _ = model(idx[:, -cfg.max_seq:])
        if greedy:                                   # greedy: take the argmax each step
            nxt = logits[:, -1].argmax(-1, keepdim=True)
        else:                                        # or sample with a temperature
            probs = torch.softmax(logits[:, -1] / 0.8, -1)
            nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], 1)
    return prompt + "".join(itos[i] for i in idx[0, len(prompt):].tolist())

print("greedy continuation:", repr(generate()))
print("sampled continuation (T=0.8):", repr(generate(greedy=False)))

# ---------- Loss curve (saved as 300 dpi PNG) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams.update({"axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300})
xs, ys = zip(*log_at); vx, vy = zip(*log_val)
plt.figure(figsize=(7, 3.4))
plt.plot(xs, ys, color="#2C6E9E", lw=2, label="training loss")
plt.plot(vx, vy, color="#D1495B", lw=2, ls="--", label="validation loss")
plt.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points",
             xytext=(2, 8), fontsize=9, color="#2C6E9E")
plt.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
             xytext=(-8, 10), fontsize=9, color="#2C6E9E", ha="right")
plt.annotate(f"{vy[-1]:.2f}", (vx[-1], vy[-1]), textcoords="offset points",
             xytext=(-2, -14), fontsize=9, color="#D1495B", ha="right")
plt.xlabel("training step"); plt.ylabel("cross-entropy loss")
plt.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig("../figs/loss_curve.png")
print("loss curve saved → ../figs/loss_curve.png")
