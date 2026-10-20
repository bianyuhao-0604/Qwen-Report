
# Qwen-Report

Companion code repository for **MiniQwen: Building Qwen from Scratch** —
a hands-on guide where we build a ~126K-parameter MiniQwen and verify,
chapter by chapter, every claim the book makes about the Qwen architecture.

Every chapter ships runnable scripts. If the text says "Qwen does X",
you can reproduce X with a single command a few seconds later.

## Repository Structure

| Chapter | Files | Contents |
|---------|-------|----------|
| Ch. 2 — Architecture | `ch2_arch.py` `ch2_demo.py` `ch2_rope.py` `ch2_transformer.py` | RMSNorm, RoPE, GQA, the full decoder stack |
| Ch. 3 — Tokenization | `ch3_bpe.py` `ch3_pack.py` | BPE training and sequence packing |
| Ch. 4 — Pretraining | `ch4_data.py` `ch4_pretrain.py` | Data pipeline and the pretraining loop |
| Ch. 5 — Fine-tuning | `ch5_lora.py` `ch5_mask.py` `ch5_sft_run.py` | LoRA, loss masking, SFT run |
| Ch. 6 — Alignment | `ch6_dpo.py` `ch6_grpo.py` | DPO and GRPO |
| Ch. 7 — Inference | `ch7_generate.py` `ch7_kvcache.py` `ch7_sampling.py` | Decoding, KV cache, sampling strategies |

Other directories: `resources/`, `figures/`, `img/` — assets used by the
book and this README.

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU is enough for most chapters; a GPU speeds up
  Chapters 4–6)

### Installation

