# ch5_sft_run.py -- a complete SFT run: 4 arithmetic dialogues teach MiniQwen
# to answer
import torch
import torch.nn.functional as F
from ch2_arch import MiniQwen, QwenConfig

torch.manual_seed(42)

# ---------- Data: 4 instruction samples (real projects use thousands to millions) ----------
DATA = [("1+1=", "2"), ("2+3=", "5"), ("6-4=", "2"), ("3*3=", "9")]
IM_U, IM_S, IM_E = "<|user|>", "<|assistant|>", "<|end|>"
PAD = "<pad>"
specials = [IM_U, IM_S, IM_E, PAD, "\n"]
alphabet = sorted(set("0123456789+-*/=?. abcdefghijklmnopqrstuvwxyz"))
vocab = specials + alphabet
stoi = {t: i for i, t in enumerate(vocab)}
PAD_ID, E_ID = stoi[PAD], stoi[IM_E]


def render(q, a):                                    # ChatML-style concatenation
    return (f"{IM_U}\n{q}\n{IM_E}\n{IM_S}\n{a}{IM_E}")


def encode(s):
    for sp in [IM_U, IM_S, IM_E]:
        s = s.replace(sp, f"\x00{sp}\x00")
    out = []
    for part in s.split("\x00"):
        out.append([stoi[part]] if part in stoi else [stoi[c] for c in part])
    return [t for blk in out for t in blk]


def build_batch():
    seqs, lab_seqs = [], []
    for q, a in DATA:
        ids = encode(render(q, a))
        labels = ids.copy()
        ans = ids.index(stoi[IM_S]) + 2              # answer starts here
        for i in range(ans):
            labels[i] = -100                         # mask the prompt part
        seqs.append(ids); lab_seqs.append(labels)
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD_ID)
    y = torch.full((len(seqs), L), -100)
    for i, (s, lab) in enumerate(zip(seqs, lab_seqs)):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(lab)] = torch.tensor(lab)          # -100 also covers padding
    return x, y


def train_sft(steps=300, lr=3e-3, verbose=True):
    x, y = build_batch()
    cfg = QwenConfig(vocab=len(vocab), d_model=64, n_heads=4, n_kv_heads=2)
    model = MiniQwen(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        _, loss = model(x, y)                        # the same 4 samples, repeatedly
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (step % 60 == 0 or step == 1):
            print(f"step {step:>3}  SFT loss = {loss.item():.4f}")
    return model


@torch.no_grad()
def chat(model, q, max_new=6):                       # greedy decode until <|end|>
    ids = encode(f"{IM_U}\n{q}\n{IM_E}\n{IM_S}\n")
    for _ in range(max_new):
        logits = model(torch.tensor([ids]))[0][0, -1]
        nxt = int(logits.argmax())
        if nxt == E_ID:
            break
        ids.append(nxt)
    tail = ids[ids.index(stoi[IM_S]) + 2:]
    return "".join(vocab[t] for t in tail if vocab[t] not in specials)


if __name__ == "__main__":
    model = train_sft()
    print("-" * 34)
    for q, a in DATA:
        pred = chat(model, q)
        mark = "OK" if pred == a else "FAIL"
        print(f"{mark}  user: {q:<6} model: {pred} (target {a})")
