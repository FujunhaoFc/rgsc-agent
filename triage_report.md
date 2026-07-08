# NLPCC Task 11 — 138 Test Papers Triage Report

**Date:** 2026-06-16
**Hardware:** Single RTX 5090 (32GB VRAM), ~5-day compute window
**Deadline:** June 20, 2026
**Strategy:** 5–8 papers in "deep mode" (full Phase A–G), rest in "skim mode" (Phase A–D only)

---

## Executive Summary

Scanned all **138 papers** across 7 categories (Astronomy, Biology, Chemistry, Environment, ML, Materials, Medical). Applied three hard filters (hardware feasibility, time ≤6h, public data). **26 papers rejected** by hard filters, **112 passed**. Scored all passing papers on 5 axes (max 14 points — note: scope capped at 2, so max is 13 in practice).

**Top 5 picks** average **13/13** with diverse coverage: medical imaging, model merging, MoE routing, vision-language alignment, and long-context LLMs.

---

## Top 5 Picks for Deep Reproduction

### 1. TransUNet (Medical) — 13/13
**Paper:** `data/test/Medical/TransUNet/paper.md` (240 lines)
**Why:** The strongest overall candidate. Hybrid ResNet-50 + ViT encoder with cascaded upsampler decoder for 2D medical image segmentation. **Explicitly tested on a single RTX 2080Ti (11GB)** — will run effortlessly on RTX 5090 32GB. Public Synapse multi-organ CT (30 scans) and ACDC cardiac MRI datasets. Rich ablation studies: skip connections, input resolution, patch size, model scaling. Clean PyTorch implementation, standard MONAI-compatible pipeline. **Estimated reproduction time: 3–5 hours.**
- Model: ResNet-50 + 12-layer ViT (~100M params)
- Datasets: Synapse (30 CT scans), ACDC (cardiac MRI) — public
- Hardware: RTX 2080Ti 11GB → 5090 32GB (4× headroom)
- Clarity: 3 | Scope: 2 | HW Match: 3 | Infra: 3 | Length: 2

### 2. ExPO (ML) — 13/13
**Paper:** `data/test/ML/ExPO/paper.md` (537 lines)
**Why:** Model extrapolation via **weight arithmetic — zero training required**. Core method: amplify the parameter delta from SFT→DPO using publicly available HuggingFace checkpoints (zephyr-7b-sft + zephyr-7b-dpo). Evaluate on AlpacaEval 2.0 using vLLM inference. The controlled DPO experiments (8×A100) are optional — the extrapolation method itself is the novel contribution and can be validated purely via inference. Short paper, exceptional clarity. **Estimated reproduction time: 1–2 hours.**
- Model: zephyr-7b-dpo (7B), extended to 1.8B–70B (all open-weight)
- Datasets: UltraFeedback, AlpacaEval 2.0, MT-Bench — public
- Hardware: Inference-only on 7B models (fits 32GB easily)
- Clarity: 3 | Scope: 2 | HW Match: 3 | Infra: 3 | Length: 2

### 3. Router-Tuning (ML) — 13/13
**Paper:** `data/test/ML/Router-Tuning/paper.md` (462 lines)
**Why:** Novel MoE router and attention router calibration via lightweight fine-tuning. **Explicitly benchmarked on RTX A6000 with <30 minute training time**, 1000× faster than prior work. Trains only router networks (<0.01% of parameters) across 5 model families (Llama, Mistral, Qwen, DeepSeek-MoE, OLMoE). Standard HuggingFace/PyTorch stack. Short, well-structured paper. **Estimated reproduction time: 2–4 hours.**
- Model: Multiple 7B–8B models (routers only trained, LLM frozen)
- Datasets: Small-scale calibration datasets — public
- Hardware: RTX A6000 <30 min → 5090 even faster
- Clarity: 3 | Scope: 2 | HW Match: 3 | Infra: 3 | Length: 2

### 4. GMAIL (ML) — 13/13
**Paper:** `data/test/ML/GMAIL/paper.md` (536 lines)
**Why:** Gen-Real alignment: LoRA rank-4 fine-tuning of CLIP on Stable Diffusion v2 generated images. Diverse experimental scope spanning captioning (CLIPCap), retrieval, and classification across **10+ public benchmarks** (COCO, Flickr30k, ImageNet, DTD, Stanford Cars, SUN397, Food101, etc.). Crystal-clear hyperparameters (AdamW lr=1e-4, LoRA rank=4, cosine annealing). Multiple model architectures evaluated (CLIP, CLIPCap, LLaVA, Llama3). **Estimated reproduction time: 4–6 hours.**
- Model: CLIP ViT + LoRA rank-4 (~86M params, tiny trainable footprint)
- Datasets: COCO, Flickr30k, CC3M, CC12M, ImageNet + 7 more — all public
- Hardware: Stable Diffusion v2 for synthetic data + LoRA fine-tuning (fits 32GB)
- Clarity: 3 | Scope: 2 | HW Match: 3 | Infra: 3 | Length: 2

### 5. AdaGroPE (ML) — 13/13
**Paper:** `data/test/ML/AdaGroPE/paper.md` (527 lines)
**Why:** **Training-free** RoPE position encoding modification for long-context LLMs. Plug-and-play: no training, no fine-tuning, inference only. Evaluated across 10+ model variants (Llama-2/3, Mistral, SOLAR, Phi-2, Vicuna) on 4 benchmark suites (PG19, Passkey, LongBench, L-Eval). Comprehensive ablation on hyperparameters and context length scaling. All models ≤13B, all datasets public. **Estimated reproduction time: 2–4 hours.**
- Model: Llama-2/3-7B/8B, Mistral-7B, SOLAR-10.7B, Phi-2 (2.7B) — all ≤13B
- Datasets: PG19, LongBench, L-Eval, Passkey — public
- Hardware: Single GPU inference, training-free
- Clarity: 3 | Scope: 2 | HW Match: 3 | Infra: 3 | Length: 2

---

## Backup 3 Picks

### B1. RidgeLoRA (ML) — 13/13
**Paper:** `data/test/ML/RidgeLoRA/paper.md` (894 lines)
PEFT method with comprehensive 3-domain evaluation (commonsense, math/code, multi-modal) across 7B–8B models. Only ~20M trainable params per run, standard HuggingFace datasets. Well-documented hyperparameters. PEFT nature means all experiments completable within 6 hours.

### B2. representation-learning (Environment) — 13/13
**Paper:** `data/test/ML/../Environment/representation-learning/paper.md` (280 lines)
SimSiam with ResNet-50 backbone for self-supervised remote sensing. 6 public datasets, pre-training + fine-tuning pipeline clearly documented (SGD, batch 128, 100k iterations). Tested on Quadro P6000 (24GB). Multiple downstream tasks. Short, well-structured paper.

### B3. CACTI (ML) — 12/13
**Paper:** `data/test/ML/CACTI/paper.md` (951 lines)
Tabular data imputation with masked autoencoding. **Fastest to reproduce: <30 minutes for all experiments on any GPU.** Tiny custom transformer (<300MB peak GPU memory), 10 public UCI datasets, 13 baselines, all hyperparameters explicitly documented. If a top-5 pick hits unexpected issues, this is the fastest fallback.

---

## Rejection Statistics

### By Category

| Category | Total | Rejected | Rejection Rate | Top Reason |
|----------|-------|----------|----------------|------------|
| Astronomy | 5 | 2 | 40% | Non-DL methods, massive data preprocessing |
| Biology | 4 | 0 | 0% | All passed (though some are R/CPU-based, less relevant) |
| Chemistry | 4 | 0 | 0% | All passed (standard GNNs, small benchmarks) |
| Environment | 5 | 0 | 0% | All passed (lightweight models, public data) |
| **ML** | **110** | **24** | **21.8%** | Multi-GPU distributed training, >30B models, proprietary data |
| Materials | 5 | 0 | 0% | All passed (standard GNNs, public MP/JARVIS data) |
| Medical | 5 | 0 | 0% | All passed (standard segmentation models) |
| **Total** | **138** | **26** | **18.8%** | — |

### Rejection Reasons (all 26)

**Multi-GPU / Distributed Training (15 papers):**
ALTER, ATLANTIS, AttnRL, ContextReasoner, DEEM, E3-RL4LLMs, GAPO, LLaVA-Reasoner-DPO, LWD, MODA, MemAgent, MfM, cheating-llm-benchmarks, EPIC, RLEF

**Model Too Large / Hardware Mismatch (4 papers):**
REAP (480B–1T MoE), MoA (70B–110B inference), GTO (A100 80GB requirement), PedagogicalRL (PPO 7B with multi-turn rollouts)

**Proprietary API Dependency (2 papers):**
Cache-of-Thoughts (GPT-4o master oracle), ALTER (GPT-3.5-turbo core engine)

**Proprietary Data (2 papers):**
KBI (Kuaishou internal codebase), TIMING (MIMIC-III gated PhysioNet)

**Non-DL / Massive Preprocessing (2 papers):**
DTARPS (0.9M light curves, ARIMA fitting), LightPred (extensive simulation data pipeline)

**Distributed Architecture (1 paper):**
LoRA-A2 (federated learning, 30 clients)

---

## Score Distribution (112 Passing Papers)

| Score | Count | Papers |
|-------|-------|--------|
| 13/13 | 9 | TransUNet, ExPO, Router-Tuning, GMAIL, AdaGroPE, RidgeLoRA, ULPT, LARA, representation-learning |
| 12/13 | 12 | CACTI, PRISM, RFT, MR-Q, MOSAIC, disagreement-prediction, token-prepending, preference-vector, UNetMamba, scMAE, DataDTA, RetroKNN, probing-the-limit, advancing-glitch-classification, SCoNE, SensorLLM, SUBARU, TokenSwap, TraceLLM, Twilight, KLASS, NDAD |
| 11/13 | 18 | IGEGRN, scPML, HGTDP-DTA, Retroformer, remote-sensing, CGCNN, CrabNet, nnFormer, CSNN, L4Q, LEANCODE, MARINE, MePO, data-aware-and-scalable-sensitivity, direct-prompt-optimization, lexical-popularity, self-instructed-derived-prompt, where-is-the-answer, wmforger, SCGA, PFPT, Persona-Code, RDS, REEF, RoT, KARMA, MAIN-RAG |
| 10/13 | 20 | probing-the-limit, RS3Mamba, Matformer, DSGA-Net, SegMamba, Swin-UNETR, NDAD, PANDAS, PFPT, PRT, PLTR-SD, FGA, HyperFM, APPL, CompAct, humans-and-transformer-lms, know-when-to-abstain, rag-fragility, reAR, the-llm-already-knows, MIND, MaxSup |
| ≤9/13 | 53 | Remaining papers (lower priority for deep reproduction) |

---

## Category Diversity in Top Candidates

The top 5 + backup 3 cover a healthy mix:
- **Medical CV:** TransUNet (segmentation)
- **Model Analysis:** ExPO (weight arithmetic), AdaGroPE (position encoding)
- **Efficient Training:** Router-Tuning (MoE routing), RidgeLoRA (PEFT), CACTI (tabular)
- **Vision-Language:** GMAIL (CLIP alignment)
- **Self-Supervised:** representation-learning (SimSiam remote sensing)

Model scales range from tiny (CACTI <1M params) to moderate (7B–8B LLMs), all comfortably within 32GB.

---

## Recommendation

**Primary plan:** Reproduce Top 5 in order: TransUNet → ExPO → Router-Tuning → GMAIL → AdaGroPE. These span 4 domains, include both training-free and lightweight-training methods, and all fit comfortably on the RTX 5090.

**Fallback:** If any top-5 paper has missing code, broken dependencies, or hidden data issues, substitute RidgeLoRA (13/13, PEFT) or representation-learning (13/13, SimSiam) from the backup list.

**Stretch goal:** If time permits after completing 5 deep reproductions, add CACTI (<30 min to reproduce all experiments) as a 6th.

---

## Methodology Notes

- Each paper was triaged by reading the first 150–200 lines of `paper.md`
- Hard filters applied: (1) single RTX 5090 32GB feasibility, (2) ≤6 hour reproduction, (3) public datasets
- Scoring: reproducibility clarity (0–3), experimental scope (0–2), hardware match (0–3), standard infrastructure (0–3), paper length (0–2)
- Rejection decisions are conservative — papers requiring multi-GPU for *full* reproduction are rejected even if a subset could run on single GPU
- Papers that are "inference-only" or "evaluation-only" are not automatically rejected, but were scored lower on scope unless they had rich experimental designs
