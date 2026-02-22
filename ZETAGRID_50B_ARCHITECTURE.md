# ZetaGrid 25B — Architecture Documentation

## Overview

**ZetaGrid 25B** is a **non-Transformer** language model that combines **evolutionary genome search** with **gradient-based fine-tuning** over a **TCN (Temporal Convolutional Network)** backbone.

The training is split into two distinct phases:

| Phase | Method | What it does |
|-------|--------|-------------|
| **Phase 1** | Evolutionary search | Evolves a 25B-param ternary genome to predict the next byte |
| **Phase 2** | Gradient descent | Freezes genome as TCN backbone, trains LoRA adapters with cross-entropy |

---

## Model Specifications

```
Parameters:     25 Billion (ternary int8 genome)
Trainable:      ~300M (LoRA adapters + embeddings)
Memory:         6.98 GB (int8) → 14 GB (BF16 on GPU)
Vocabulary:     256 (byte-level)
Context:        256 tokens (Phase 2)
Backbone:       TCN — Gated Causal Depthwise Convolutions
Architecture:   D=4096, FF=16384, L=32 layers
Tokenizer:      Byte-level (0–255) + SentencePiece 100K (future)
Precision:      INT8 (Phase 1) → BF16 (Phase 2)
```

---

## Phase 1: Evolutionary Genome Search

### Concept

Phase 1 treats the entire model as a **genome** — a flat 1D array of ~7 billion int8 values ({−1, 0, +1}). The genome evolves through **mutation and selection** to minimize next-byte prediction error, without any backpropagation.

### Genome Structure

```python
genome = np.zeros(PHYSICAL_SIZE, dtype=np.int8)
# Values: -1, 0, +1 (ternary weights)
# Size: 6.98 GB = 6,979,321,856 bytes
# Each byte = 1 ternary weight
```

### Training Loop

At each generation, the algorithm:

1. **Mutate**: Copy the best genome, flip 0.5% of values randomly
2. **Sample weights**: Take a random 128-byte window from the genome
3. **Sample data**: Take a random batch of byte sequences from training data
4. **Predict next byte**: Compute `tanh(dot(input, weights))`
5. **Evaluate**: Compare prediction to actual next byte (MSE loss)
6. **Select**: If trial loss < best loss, keep the mutated genome

```python
while gen < 350000:
    # 1. MUTATE
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * 0.005)  # 0.5% mutation rate
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations)
    genome_trial[mut_indices] = cp.random.randint(-1, 2, size=n_mutations, dtype=cp.int8)
    
    # 2. SAMPLE WEIGHTS from genome
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    # 3. PREPARE INPUT BATCH (raw bytes, normalized to [0,1])
    input_chunk = tokens[offset : offset + BATCH_SIZE * SEQ_LEN]
    inputs = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    
    # 4. TARGET = next byte in sequence
    targets = tokens[offset+1 : ...].reshape(BATCH_SIZE, SEQ_LEN)[:, -1] / 255.0
    
    # 5. PREDICT next byte
    predictions = cp.tanh(cp.dot(inputs, weights))
    loss = float(cp.mean((predictions - targets) ** 2))
    
    # 6. SELECT (keep if better)
    if loss < best_loss:
        best_loss = loss
        cp.copyto(genome_best, genome_trial)
```

### Next-Byte Prediction Mechanism

The forward pass in Phase 1 is a **single dot product**:

```
Input:   [B, 128] — last 128 bytes, normalized to [0, 1]
Weights: [128]    — random window from genome
Output:  [B]      — tanh(dot(input, weights)) ∈ [-1, 1]
Target:  [B]      — actual next byte / 255.0
Loss:    MSE(output, target)
```

This is intentionally simple — the intelligence comes from the **evolutionary search** exploring 25 billion weights across 350,000 generations.

### Phase 1 Results

- **Platform**: Kaggle T4 (13B) → RunPod A40 (25B)
- **Generations**: 302,000 → 350,000
- **Speed**: 3–5 Hz (generations/second)
- **Final Loss**: 0.000001 (MSE)
- **Datasets**: 22 GB (OpenWebText, Amazon Reviews, BookCorpus, Code)

### What Phase 1 Achieves

- ✅ Genome learns statistical patterns of byte sequences
- ✅ Zero backpropagation required
- ✅ Explores global optima via evolutionary search
- ✅ Memory efficient (int8 ternary weights)
- ❌ Cannot generate coherent text alone (no embedding, no layers, no softmax)

---

## Phase 2: Gradient Fine-Tuning (TCN + LoRA)

### Concept

Phase 2 converts the evolved genome into a **real neural network** by:

1. **Reshaping** genome bytes into weight matrices for TCN layers
2. **Freezing** these weights (they don't update during training)
3. **Adding trainable LoRA adapters** on top
4. **Training with cross-entropy** on next-byte prediction

This is analogous to **QLoRA** — the genome provides the frozen "pretrained" backbone, and small adapters learn language generation.

### Architecture: TCN Backbone (Non-Transformer)

```
Input bytes → Embedding (256 → 4096) → Positional Embedding
            ↓
    ┌───────────────────────────────┐
    │   32× Gated Causal TCN Layer  │
    │                               │
    │   x → RMSNorm                 │
    │     → FrozenLinear (genome)   │  ← 25B frozen params
    │     + LoRA adapter            │  ← trainable
    │     → Causal DWConv1D         │
    │     → SiLU(a) × σ(g)         │  ← gated activation
    │     → FrozenLinear (genome)   │  ← 25B frozen params
    │     + LoRA adapter            │  ← trainable
    │     → residual + scale        │
    └───────────────────────────────┘
            ↓
    RMSNorm → Linear (tied) → Logits (256 classes)
            ↓
    Softmax → Sample → Next Byte
```

### Gated Causal TCN Layer (Detail)

Each layer uses **gated causal depthwise convolutions** — the same backbone used in KAM-LLM:

```python
def forward(self, x):
    res = x                                    # Residual connection
    x = RMSNorm(x)                             # Pre-normalization
    
    # In-projection: genome frozen + LoRA trainable
    ag = FrozenLinear(x) + LoRA_in(x)          # [B, T, 2*FF]
    a, g = ag.chunk(2)                         # Split into activation + gate
    
    # Causal depthwise conv (no future leakage)
    a = CausalDWConv1D(a, dilation=d)          # [B, T, FF]
    
    # Gated activation
    y = SiLU(a) * sigmoid(g)                   # [B, T, FF]
    
    # Out-projection: genome frozen + LoRA trainable
    out = FrozenLinear(y) + LoRA_out(y)        # [B, T, D]
    
    return res + out * scale                   # Residual + learnable scale
```

**Dilation cycle**: [1, 2, 4, 8, 16, 32, 64, 128] — gives exponentially growing receptive field.

### Genome → Weight Matrices

The 6.98 GB genome is sliced into structured weight matrices:

```python
# Per TCN layer:
in_proj:  genome[offset : offset + D×2FF]  → shape (2*FF, D)    # 134M params
dwconv:   genome[...] → shape (FF, 1, K)                        # 49K params
out_proj: genome[...] → shape (D, FF)                            # 67M params
# Total per layer: ~201M params
# 32 layers: ~6.4B bytes from genome

# Scaling: Xavier-adjusted for ternary density (~10% non-zero)
scale = 1.0 / sqrt(fan_in × 0.1)
weight_bf16 = genome_int8.to(bf16) × scale
```

### LoRA Adapters (Trainable)

Each layer has two LoRA adapters (rank 128):

```python
class LoRA:
    A: (rank, in_features)   # Normal init
    B: (out_features, rank)  # Zero init (starts as identity)
    
    forward(x) = x @ A.T @ B.T
```

- **Rank**: 128
- **Per layer**: ~7M trainable params
- **Total trainable**: ~300M params
- **Total model**: 25B frozen + 300M trainable

### Training Configuration

```
Loss:               Cross-entropy (softmax over 256 byte classes)
Optimizer:          AdamW (β1=0.9, β2=0.95, wd=0.1)
Learning Rate:      3e-4 with warmup + cosine decay
Batch Size:         8 × 4 grad_accum = 32 effective
Sequence Length:    256 bytes
Gradient Clip:      1.0
Mixed Precision:    BF16 (autocast)
Grad Checkpointing: Yes (saves VRAM, recomputes activations)
```

### VRAM Budget (A40 — 48 GB)

```
Genome backbone (BF16):    14.0 GB
Trainable adapters:         0.5 GB
Adam optimizer states:      2.0 GB
Gradients:                  0.5 GB
Activations (checkpointed): 2.0 GB
Training data:              2.6 GB
─────────────────────────────────
Total:                     21.6 GB / 48 GB
```

### Phase 2 Training Progress

```
Random baseline:  Loss = 5.55 (ln(256))
Step 25:          Loss = 6.10   PPL = 450     (warming up)
Step 50:          Loss = 2.92   PPL = 18.5    (learning fast!)
```

### Text Generation (Autoregressive)

```python
def generate(prompt):
    idx = encode_to_bytes(prompt)          # UTF-8 → byte list
    
    for step in range(max_tokens):
        logits = model(idx[-context:])     # Forward through TCN
        logits = logits[:, -1, :] / temp   # Last position, apply temperature
        
        # Top-k sampling
        top_k_filter(logits, k=50)
        probs = softmax(logits)
        next_byte = multinomial(probs)     # Sample from distribution
        
        idx = concat(idx, next_byte)
    
    return decode_bytes_to_text(idx)       # Bytes → UTF-8 string
```

---

## Key Innovations

### 1. Two-Phase Training (Evolution → Gradient)
Phase 1 explores the global loss landscape without gradients. Phase 2 refines locally with gradient descent. The genome provides a pre-evolved initialization that traditional random init cannot.

### 2. Non-Transformer Backbone (TCN)
Gated causal depthwise convolutions instead of self-attention. No KV cache needed. Linear complexity in sequence length.

### 3. Ternary Weights + LoRA
25B ternary frozen weights (int8) provide massive capacity at minimal memory. Small trainable adapters learn language generation on top.

### 4. Byte-Level Modeling
No tokenizer needed — works directly on raw UTF-8 bytes. Universal across languages and formats.

### 5. Fractal Scaling
Genome can be expanded from 13B → 25B → 50B via pattern replication with diversity noise, preserving evolved patterns.

---

## File Structure

```
zetagrid_50b/
├── zetagrid_25b_production.npy     # Phase 1 genome (6.6 GB)
├── models/
│   └── tokenizer.model             # SentencePiece 100K (future use)
├── data/pretrain/
│   └── KAM_SFT_MASTER.bin          # Training data (2.6 GB)
├── phase2_checkpoints/
│   └── zeta25b_FINAL.pt            # Phase 2 trained model
└── ZETAGRID_PHASE2_GRADIENT.py     # Phase 2 training script
```

---

## Future: Phase 3 (SFT) & QuLP Quantization

- **Phase 3**: Supervised fine-tuning on instruction-response pairs
- **QuLP**: 2-bit quantization with Hessian incoherence processing for deployment
- **Stigma integration**: Fractal expert routing for mixture-of-experts
- **KAM memory**: External ring-buffer memory bank for long-context

---

**Created**: February 2026  
**Architecture**: Evolutionary TCN (Non-Transformer)  
**Training**: Kaggle T4 (13B Phase 1) → RunPod A40 (25B Phase 1+2)  
**Status**: Phase 2 Training In Progress  
