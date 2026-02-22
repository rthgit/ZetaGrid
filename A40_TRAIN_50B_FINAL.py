#!/usr/bin/env python3
"""
ZETAGRID 50B - A40 FINAL (DISK OPTIMIZED)
Expand to 50B, save only final checkpoint
"""

import os
import numpy as np
import cupy as cp
import time
from pathlib import Path

print("=" * 70)
print("ZETAGRID 50B - A40 PRODUCTION")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
CHECKPOINT_13B = f"{BASE_DIR}/models/zetagrid_checkpoint_13B.npy"
PRETRAIN_DIR = f"{BASE_DIR}/data/pretrain"

# ============================================================
# PHASE 1: EXPAND TO 50B ON GPU
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: EXPANDING 13B → 50B")
print("=" * 70)

print(f"Loading 13B checkpoint...")
genome_13b_np = np.load(CHECKPOINT_13B)
print(f"✅ Loaded: {len(genome_13b_np)/1e9:.2f}GB")

# Target size: 12GB = 50B params
GB = 12
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024
print(f"\nTarget: {PHYSICAL_SIZE/1e9:.2f}GB (50B params)")

print("Expanding on GPU...")
genome_13b_gpu = cp.array(genome_13b_np, dtype=cp.int8)
del genome_13b_np  # Free RAM

# Create 50B on GPU
genome_50b = cp.zeros(PHYSICAL_SIZE, dtype=cp.int8)

# Replicate pattern
replication_factor = PHYSICAL_SIZE // len(genome_13b_gpu)
print(f"Replication: {replication_factor}x")

for i in range(replication_factor):
    start = i * len(genome_13b_gpu)
    end = min(start + len(genome_13b_gpu), PHYSICAL_SIZE)
    genome_50b[start:end] = genome_13b_gpu[:end-start]

# Fill remaining
if end < PHYSICAL_SIZE:
    remaining = PHYSICAL_SIZE - end
    genome_50b[end:] = genome_13b_gpu[:remaining]

del genome_13b_gpu  # Free GPU memory

# Add diversity (10% noise)
print("Adding diversity noise...")
n_noise = int(PHYSICAL_SIZE * 0.1)
noise_idx = cp.random.randint(0, PHYSICAL_SIZE, size=n_noise, dtype=cp.int64)
genome_50b[noise_idx] = cp.random.randint(-1, 2, size=n_noise, dtype=cp.int8)

print(f"✅ Expanded to {PHYSICAL_SIZE/1e9:.2f}GB (50B params)")

# Setup for training
genome_best = genome_50b
genome_trial = cp.zeros(PHYSICAL_SIZE, dtype=cp.int8)

# ============================================================
# PHASE 2: LOAD DATASETS TO GPU
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: LOADING DATASETS")
print("=" * 70)

datasets = []
bin_files = sorted(Path(PRETRAIN_DIR).glob("*.bin"))

for bin_file in bin_files:
    print(f"Loading {bin_file.name}...", end=" ", flush=True)
    data_np = np.fromfile(str(bin_file), dtype=np.uint8)
    data_gpu = cp.array(data_np, dtype=cp.uint8)
    datasets.append(data_gpu)
    del data_np
    print(f"✅ {len(data_gpu)/1e6:.0f}M tokens")

print(f"\n✅ {len(datasets)} datasets loaded")

# Pre-calculate offsets
BATCH_SIZE = 8192
SEQ_LEN = 128
max_offsets = []
for data in datasets:
    max_off = len(data) - (BATCH_SIZE * SEQ_LEN) - 1
    max_offsets.append(max(0, max_off))

valid_idx = [i for i, m in enumerate(max_offsets) if m > 0]

# ============================================================
# PHASE 3: EVOLUTION (Gen 302K → 350K)
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: EVOLUTIONARY TRAINING (50B)")
print("=" * 70)

START_GEN = 302000
TARGET_GEN = 350000
MUTATION_RATE = 0.005

print(f"Generation: {START_GEN:,} → {TARGET_GEN:,}")
print(f"Model: 50B parameters")
print(f"Mutation: {MUTATION_RATE*100:.1f}%\n")

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
    
    # Sample data
    dataset_idx = np.random.choice(valid_idx)
    tokens = datasets[dataset_idx]
    offset = np.random.randint(0, max_offsets[dataset_idx])
    
    # Prepare batch
    input_chunk = tokens[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    
    target_chunk = tokens[offset + 1 : offset + 1 + BATCH_SIZE * SEQ_LEN]
    targets = target_chunk.reshape(BATCH_SIZE, SEQ_LEN)[:, -1].astype(cp.float32) / 255.0
    
    # Evaluate
    predictions = cp.tanh(cp.dot(input_chunk, weights))
    loss = float(cp.mean((predictions - targets) ** 2))
    
    # Select
    if loss < best_loss:
        best_loss = loss
        cp.copyto(genome_best, genome_trial)
    
    # SAVE ONLY EVERY 10K (not 1K) to save disk space
    if gen % 10000 == 0:
        print(f"\n💾 Saving Gen {gen:,}...", end=" ", flush=True)
        genome_cpu = cp.asnumpy(genome_best)
        # Overwrite same file to save space
        np.save(f"{BASE_DIR}/zetagrid_50b_checkpoint.npy", genome_cpu)
        del genome_cpu
        print(f"✅ Best: {best_loss:.6f}")
    
    # Logging
    if gen % 50 == 0:
        elapsed = time.time() - start_time
        hz = (gen - START_GEN) / elapsed if elapsed > 0 else 0
        eta_min = (TARGET_GEN - gen) / hz / 60 if hz > 0 else 0
        print(f"Gen {gen:,} | {hz:.1f}Hz | Loss: {loss:.5f} | Best: {best_loss:.5f} | ETA: {eta_min:.0f}min | DS: {dataset_idx}")

# ============================================================
# FINAL SAVE
# ============================================================

print(f"\n💾 Saving FINAL 50B model...")
genome_cpu = cp.asnumpy(genome_best)
np.save(f"{BASE_DIR}/zetagrid_50b_production.npy", genome_cpu)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Model: 50B parameters ({PHYSICAL_SIZE/1e9:.2f}GB)")
print(f"Final Gen: {gen:,}")
print(f"Best Loss: {best_loss:.6f}")
print(f"Time: {(time.time()-start_time)/3600:.1f}h")
print(f"File: zetagrid_50b_production.npy")
print("\n✅ 50B MODEL READY!")
