# ch7_generate.py -- assembling the pieces: temperature + top-p in a full
# generation loop (reuses the Chapter 5 SFT model)
import torch
import torch.nn.functional as F
from ch5_sft_run import train_sft, stoi, vocab, IM_S, IM_E

torch.manual_seed(42)


@torch.no_grad()
def generate(model, prompt_ids, max_new=8, temperature=1.0, top_p=1.0, greedy=False):
    """A minimal generate(): recompute the whole sequence each step and read
    the last position (fine for a toy model; real deployments plug in the
    KV cache from Section 7.2)."""
    ids = list(prompt_ids)
    for _ in range(max_new):
        logits = model(torch.tensor([ids]))[0][0, -1]      # distribution at the last slot
        if temperature != 1.0:
            logits = logits / temperature
        if greedy:
            nxt = int(logits.argmax())
        else:
            if top_p < 1.0:                                # top-p truncation
                order = torch.argsort(logits, descending=True)
                probs = F.softmax(logits[order], -1)
                cum = torch.cumsum(probs, -1)
                keep = order[cum - probs < top_p]
                m = torch.full_like(logits, float("-inf")); m[keep] = logits[keep]
                logits = m
            probs = F.softmax(logits, -1)
            nxt = int(torch.multinomial(probs, 1))
        if nxt == stoi[IM_E]:
            break
        ids.append(nxt)
    return ids[ids.index(stoi[IM_S]) + 2:]


def to_text(tail):
    return "".join(vocab[t] for t in tail if vocab[t] not in
                   ("<|user|>", "<|assistant|>", "<|end|>", "<pad>", "\n"))


def encode_prompt(p):
    for sp in ["<|user|>", "<|assistant|>", "<|end|>"]:
        p = p.replace(sp, f"\x00{sp}\x00")
    ids = []
    for part in p.split("\x00"):
        ids += [stoi[part]] if part in stoi else [stoi[c] for c in part]
    return ids


if __name__ == "__main__":
    model = train_sft(steps=300, verbose=False)            # reuse the Chapter 5 SFT model
    for q in ("1+1=", "2+3=", "3*3="):
        ids = encode_prompt(f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n")
        out = generate(model, ids, greedy=True)            # greedy: deterministic answers
        print(f"user: {q:<6} → assistant: {to_text(out)!r}")
    # one sampled run at temperature 0.9 + top-p 0.9 (tiny answer space: still right)
    q = "1+1="
    ids = encode_prompt(f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n")
    out = generate(model, ids, temperature=0.9, top_p=0.9)
    print(f"sampled(T=0.9, top-p=0.9) {q} → {to_text(out)!r}")
