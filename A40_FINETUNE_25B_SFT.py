#!/usr/bin/env python3
"""
ZETAGRID 25B - SFT FINE-TUNING
Generalist model with supervised fine-tuning
"""

import os
import numpy as np
import cupy as cp
import time

print("=" * 70)
print("ZETAGRID 25B - SFT FINE-TUNING")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
MODEL_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
SFT_DATASET = f"{BASE_DIR}/data/pretrain/KAM_SFT_MASTER.bin"
OUTPUT_MODEL = f"{BASE_DIR}/zetagrid_25b_sft_generalist.npy"

# Training config
BATCH_SIZE = 4096
SEQ_LEN = 256
START_GEN = 0
TARGET_GEN = 100000  # 100K generations
MUTATION_RATE = 0.002  # Lower for fine-tuning

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: LOADING 25B MODEL")
print("=" * 70)

print(f"Loading: {MODEL_PATH}")
genome_np = np.load(MODEL_PATH)
print(f"✅ Loaded: {len(genome_np)/1e9:.2f}GB")

print("Transferring to GPU...")
genome_best = cp.array(genome_np, dtype=cp.int8)
genome_trial = cp.zeros_like(genome_best)
del genome_np
print("✅ Model on GPU")

PHYSICAL_SIZE = len(genome_best)
vram_used = PHYSICAL_SIZE * 2 / 1e9
print(f"   VRAM used: {vram_used:.2f}GB")

# ============================================================
# LOAD SFT DATASET
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: LOADING SFT DATASET")
print("=" * 70)

print(f"Loading: {SFT_DATASET}")
sft_data_np = np.fromfile(SFT_DATASET, dtype=np.uint8)
print(f"✅ Loaded: {len(sft_data_np)/1e9:.2f}GB ({len(sft_data_np)/1e6:.0f}M tokens)")

print("Transferring to GPU...")
sft_data = cp.array(sft_data_np, dtype=np.uint8)
del sft_data_np
print("✅ Dataset on GPU")

vram_total = vram_used + len(sft_data) / 1e9
print(f"   Total VRAM: {vram_total:.2f}GB / 48GB")

# Calculate max offset
max_offset = len(sft_data) - (BATCH_SIZE * SEQ_LEN) - 1
print(f"Max offset: {max_offset:,}")

# ============================================================
# SFT FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: SFT FINE-TUNING")
print("=" * 70)

print(f"Generations: {START_GEN:,} → {TARGET_GEN:,}")
print(f"Mutation rate: {MUTATION_RATE*100:.1f}%")
print(f"Sequence length: {SEQ_LEN}\n")

start_time = time.time()
gen = START_GEN
best_loss = 9999.0

while gen < TARGET_GEN:
    gen += 1
    
    # Mutate
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * MUTATION_RATE)
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations, dtype=cp.int64)
    genome_trial[mut_indices] = cp.random.randint(-1, 2, size=n_mutations, dtype=cp.int8)
    
    # Sample weights
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    # Sample SFT data
    offset = np.random.randint(0, max_offset)
    
    input_chunk = sft_data[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    
    target_chunk = sft_data[offset + 1 : offset + 1 + BATCH_SIZE * SEQ_LEN]
    targets = target_chunk.reshape(BATCH_SIZE, SEQ_LEN)[:, -1].astype(cp.float32) / 255.0
    
    # Evaluate
    predictions = cp.tanh(cp.dot(input_chunk, weights))
    loss = float(cp.mean((predictions - targets) ** 2))
    
    # Select
    if loss < best_loss:
        best_loss = loss
        cp.copyto(genome_best, genome_trial)
    
    # Save every 20K (less frequent)
    if gen % 20000 == 0:
        print(f"\n💾 Saving Gen {gen:,}...", end=" ", flush=True)
        genome_cpu = cp.asnumpy(genome_best)
        np.save(f"{BASE_DIR}/zetagrid_25b_sft_checkpoint.npy", genome_cpu)
        del genome_cpu
        print(f"✅ Best: {best_loss:.6f}")
    
    # Logging
    if gen % 50 == 0:
        elapsed = time.time() - start_time
        hz = gen / elapsed if elapsed > 0 else 0
        eta_min = (TARGET_GEN - gen) / hz / 60 if hz > 0 else 0
        print(f"Gen {gen:,} | {hz:.1f}Hz | Loss: {loss:.5f} | Best: {best_loss:.5f} | ETA: {eta_min:.0f}min")

# ============================================================
# FINAL SAVE
# ============================================================

print(f"\n💾 Saving FINAL 25B SFT generalist model...")
genome_cpu = cp.asnumpy(genome_best)
np.save(OUTPUT_MODEL, genome_cpu)

print("\n" + "=" * 70)
print("SFT FINE-TUNING COMPLETE!")
print("=" * 70)
print(f"Model: 25B SFT Generalist")
print(f"Final Gen: {gen:,}")
print(f"Best Loss: {best_loss:.6f}")
print(f"Time: {(time.time()-start_time)/3600:.1f}h")
print(f"File: {OUTPUT_MODEL}")
print("\n✅ 25B SFT GENERALIST READY!")
