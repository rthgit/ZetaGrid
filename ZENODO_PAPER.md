# RTH-LM: A Fractal Temporal Convolutional Language Model

**Author:** Christian Quintino De Luca  
**Affiliation:** RTH Italia (Research & Technology Hub), Milan, Italy  
**Email:** [info@rthitalia.com](mailto:info@rthitalia.com)  
**Date:** February 2026  
**License:** CC BY-NC 4.0 (Research) / Commercial License (RTH Italia)  
**Repository:** [https://github.com/rthgit/ZetaGrid](https://github.com/rthgit/ZetaGrid)

---

## Abstract

We introduce **RTH-LM**, a **Fractal Gated Causal Temporal Convolutional Network (TCN)** for language modeling, designed as an alternative to attention-centric architectures. RTH-LM targets linear-time inference in sequence length and improved data/compute efficiency under constrained training regimes. The model family is organized around a modular separation between a compact shared frozen core (the **Genome**) and trainable low-rank adapters (the **Soul**), enabling rapid domain specialization with minimal update artifacts.

This paper presents: (i) the Fractal TCN backbone and scaling strategy, (ii) the Genome/Soul modular deployment design, and (iii) initial training signals showing stable convergence under deliberately restricted data. We also provide a conservative hardware feasibility analysis indicating that a **120B-parameter scaled variant**—deployed with **2-bit weight-only quantization**—can fit on **80GB-class GPUs**, depending on runtime state and inference-engine overhead. Finally, we outline a practical pathway to a **1T-parameter vision** using a compact **8–9× H100 80GB cluster** for sharded inference and incremental expansion workflows.

---

## 1. Introduction

Transformer architectures dominate modern language modeling, but their operational profile often becomes the limiting factor in real deployments: quadratic attention costs, high memory pressure at long context, and a prevailing reliance on extremely large pretraining corpora. These constraints raise barriers for independent research groups and small companies, and they increase energy requirements at scale.

**RTH-LM** explores a different axis of scaling: **architectural structure over brute-force data scaling**. The core hypothesis is that long-range dependency modeling can be achieved using deep causal temporal convolutions combined with gating and a fractal block expansion strategy, reducing reliance on explicit attention mechanisms.

### 1.1 Contributions

1.  **Fractal Gated Causal TCN Backbone**: a deep, dilated, causal convolutional stack with gating and residual routing for autoregressive language modeling, designed for linear-time inference in sequence length.
2.  **Genome/Soul Modularity**: a deployable separation between a shared frozen Genome core and trainable Soul adapters (LoRA-style), enabling fast specialization with minimal retraining and small update artifacts.
3.  **Constrained-Regime Training Signals**: training is intentionally performed on a small curated dataset to emphasize architectural learning dynamics and feasibility under limited compute/data.
4.  **Conservative Memory & Hardware Feasibility**: a planning-grade VRAM model for a 120B scaled variant under 2-bit weight-only quantization, with explicit assumptions and bounded estimates.

### 1.2 Scope and Non-Claims

This paper focuses on feasibility, training stability, and modular deployment design. It does not claim parity with frontier-scale instruction-tuned Transformer systems trained on trillions of tokens. Instead, it addresses: **How far can capacity and usability be pushed under tight data/compute constraints using a non-attention backbone?**

---

## 2. Background and Motivation

### 2.1 Why Replace Attention?

Attention provides flexible token-to-token routing but incurs costs that become dominant at long context and high-throughput serving. Many real-world deployments are constrained not by FLOPs alone but by memory bandwidth, allocator fragmentation, and context-state growth. **RTH-LM** aims to reduce these constraints using temporal mixing via convolution and gating, relying on depth/dilation schedules rather than all-pairs interactions.

### 2.2 Temporal Convolutions for Sequence Modeling

Causal dilated convolutions can cover long receptive fields using dilation schedules. With sufficient depth, the model can integrate information across large contexts with predictable compute, which is attractive for streaming and on-prem inference.

---

## 3. RTH-LM Architecture

### 3.1 Overview

RTH-LM consists of: (i) tokenizer/embeddings, (ii) a Fractal Gated Causal TCN backbone, (iii) an output head, and (iv) optional modular adapters (the **Soul**) for domain specialization. The backbone is organized as a stack of Fractal Blocks, each composed of multiple gated causal convolutional layers with residual pathways and normalization.

### 3.2 Gated Causal Convolutional Layer

Let $x \in \mathbb{R}^{T \times d}$ be the sequence representation. Each layer performs:

1.  $h = \text{Conv1D}_{\text{causal, dilated}}(x)$
2.  $[h_g, h_v] = \text{split}(W h)$
3.  $y = \sigma(h_g) \odot \phi(h_v)$
4.  $x' = x + \text{Dropout}(W_o y)$
5.  $\hat{x} = \text{Norm}(x')$

### 3.3 Fractal Block Expansion

The fractal property refers to a scalable block composition strategy:
*   A base model is built from repeated block templates (a repeated micro-architecture).
*   Larger models are formed by mirroring/replicating block groups and re-initializing only minimal routing/scaling parameters.
*   Scaling is followed by brief stabilization training and/or adapter re-optimization.

### 3.4 The Frozen Genome

RTH-LM defines a compact shared core parameter artifact called the **Genome**, reused across instances. In the reference configuration motivating this paper, the Genome is engineered to be storage-efficient (single-digit GB class depending on serialization and quantization).

### 3.5 The Liquid Soul (Adapters)

Domain adaptation is performed via trainable low-rank adapters, called the **Soul**. Souls are small relative to the full model footprint and can be swapped to change domain behavior (coding, technical writing, creative) without full retraining.

---

## 4. Training Methodology

### 4.1 Objective

RTH-LM is trained using autoregressive next-token prediction:
$\mathcal{L} = - \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$

### 4.2 Data Regime

Training is intentionally constrained to emphasize architectural learning dynamics:
*   **Dataset size:** ~1.5GB curated text (scientific + narrative mix)
*   **Rationale:** feasibility under minimal data, not maximal benchmark parity

### 4.3 Compute Setup (Reference Run)

*   **Hardware:** single NVIDIA A40
*   **Precision:** mixed precision (FP16/BF16, engine-dependent)
*   **Stability tools:** gradient accumulation, checkpointing, norm management
*   **Duration:** ~24 hours end-to-end training loop

---

## 5. Results and Observations (Initial Signals)

### 5.1 Convergence Snapshot

At the end of the referenced constrained-regime run, we report the following training snapshot:
*   **Step:** 15,000
*   **Training loss:** ≈ 1.0
*   **Perplexity:** ≈ 2.8

These values are reported as an empirical training signal for feasibility; broader generalization evaluation is outlined in Appendix A.

---

## 6. Transformer Baseline: Practical Cost & Time Comparison

To contextualize RTH-LM, we provide a practical comparison against an “economical” Transformer baseline, focusing on deployment scaling drivers: context-state growth, long-context overhead, and planning-grade cost/time implications.

| Dimension | Economical Transformer (typical) | RTH-LM (Fractal TCN) |
| :--- | :--- | :--- |
| **Long-context memory** | KV cache grows with context and layers; often FP16/BF16 | No classical KV cache; runtime state can be engineered as streaming buffers |
| **Inference scaling** | Full attention can be $O(N^2)$; optimized variants mitigate | Backbone compute designed for linear-time in sequence length |
| **Domain adaptation** | Full fine-tune costly; adapters common (LoRA) | Genome/Soul split: compact Souls enable fast specialization |
| **Evidence under small data** | Small data often insufficient for broad generality | Stable snapshot at 15k steps (loss≈1.0, ppl≈2.8) |

**Table 1:** Planning-grade comparison.

---

## 7. Memory Model and Hardware Feasibility

### 7.1 Weight Storage Under 2-bit Quantization (120B)

Idealized raw size:
$120 \times 10^9 \times 2 / 8 = 30.0 \text{ GB}$

Conservative planning range: Weights (2-bit, weight-only): **~30–36 GB VRAM** (implementation-dependent).

### 7.2 Runtime State: “KV-Equivalent” Upper Bound

For conservative comparison, we report a Transformer-style KV-cache equivalent (FP16) upper bound under reference assumptions $L = 36$, $G_{kv} = 8$, $d_h = 64$. FP16 uses 2 bytes per element:
$\text{KV/token} \approx 2 \times L \times (G_{kv} \times d_h) \times 2 \text{ bytes} \approx 72 \text{ KB/token}$

KV-equivalent state (upper bound): 8k ~0.59 GB; 32k ~2.36 GB; 128k ~9.44 GB.

### 7.3 Total VRAM Estimate (Batch=1)

**Total (128k, batch=1): ~45–57 GB VRAM** (weights + upper-bound state + runtime overhead). This is compatible with **80GB-class GPUs** (H100/A100 class), subject to engine details.

---

## 8. Cluster Feasibility: The 1T-Parameter Vision on 8–9× H100

A 1T-parameter model quantized to 2-bit weight-only has raw size:
$1 \times 10^{12} \times 2 / 8 = 250 \text{ GB}$

We report a conservative range of **~250–300 GB** for weights across GPUs. While 4×80GB is the theoretical minimum for weights-only, production inference benefits from **6× H100 80GB** as a baseline (weights + overhead + long-context state), and **8–9 GPUs** for throughput, replicas, and multi-tenant Soul serving.

---

## Appendix A: Evaluation Plan and Minimal Evidence Pack

Minimal reviewer-proof evidence bundle:
*   Held-out perplexity on a fixed validation split.
*   1–2 lightweight benchmarks or subsets to detect collapse/overfit.
*   Fixed qualitative prompt suite (10–20 prompts) with deterministic decoding.

## Appendix B: Fractal Expansion Protocol (Planning-Grade)

Replication/mirroring of block groups followed by re-initialization of only stabilization parameters (norm gains, routing scalars, minimal mixing coefficients) and a short stabilization run. Adapter training (Souls) performs domain specialization while keeping the expanded Genome stable for deployment.

## Appendix C: Reproducibility Checklist

Commit hash + environment; tokenizer hash; dataset manifest + split IDs; full training config (seq, batch, optimizer, LR schedule); logs (step-time, loss, ppl, memory) and checkpoint hashes.

---

**Citation:**  
De Luca, C. Q. (2026). *RTH-LM: A Fractal Temporal Convolutional Language Model*. RTH Italia. Repository: [https://github.com/rthgit/ZetaGrid](https://github.com/rthgit/ZetaGrid)

---

**Contact:**  
[info@rthitalia.com](mailto:info@rthitalia.com) | RTH Italia
