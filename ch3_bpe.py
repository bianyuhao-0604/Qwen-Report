# ch3_bpe.py -- a mini byte-level BPE tokenizer in ~50 lines (pure Python, offline)
from collections import Counter


def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


# Training corpus in bytes (multilingual on purpose: EN + ZH -- byte-level
# BPE must handle non-ASCII text, see merge #3 in the output below).
text = ("the quick brown fox jumps over the lazy dog. " * 6 +
        "语言模型把文本切成token。 " * 6)
ids = list(text.encode("utf-8"))
vocab = {i: bytes([i]) for i in range(256)}          # ids 0-255 = single bytes
merges = {}
for step in range(24):                               # learn 24 merges
    (a, b), _ = get_pairs(ids).most_common(1)[0]
    merges[(a, b)] = 256 + step
    ids = merge(ids, (a, b), 256 + step)
    vocab[256 + step] = vocab[a] + vocab[b]

print(f"corpus bytes = {len(text.encode('utf-8'))}, tokens after training = {len(ids)}"
      f" (compression ≈ {len(text.encode('utf-8')) / len(ids):.1f}x)")
print("top 6 merges (bytes form; non-ASCII shown as hex):")
for (a, b), _ in [(k, v) for k, v in sorted(merges.items(), key=lambda x: x[1])][:6]:
    print(f"  #{merges[(a,b)] - 256:>2}  {vocab[a]!r} + {vocab[b]!r}"
          f" -> {vocab[merges[(a,b)]]!r}")


def encode(s):                                       # greedily apply learned merges
    ids = list(s.encode("utf-8"))
    for (a, b), nid in sorted(merges.items(), key=lambda x: x[1]):
        ids = merge(ids, (a, b), nid)
    return ids


def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


s = "the dog and the fox"
print(f"encode({s!r}) first 10 tokens = {encode(s)[:10]}")
print(f"roundtrip intact = {decode(encode(s)) == s}")
word = "café"
print(f"out-of-corpus chars fall back to bytes: encode({word!r}) = {encode(word)}"
      f" (é takes 2 bytes → 2 tokens; decode restores = {decode(encode(word))!r})")
