#!/usr/bin/env python3
"""
ZETAGRID 50B - A40 MEMORY OPTIMIZED
Evolution only, no expansion, direct GPU training
"""

import os
import numpy as np
import cupy as cp
import time
from pathlib import Path

print("=" * 70)
print("ZETAGRID 50B - A40 MEMORY OPTIMIZED TRAINING")
print("=" * 70)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
CHECKPOINT_13B = f"{BASE_DIR}/models/zetagrid_checkpoint_13B.npy"
PRETRAIN_DIR = f"{BASE_DIR}/data/pretrain"

# Training config
BATCH_SIZE = 8192
SEQ_LEN = 128
START_GEN = 302000
TARGET_GEN = 350000
MUTATION_RATE = 0.005

print(f"\nTraining: Gen {START_GEN} → {TARGET_GEN}")

# ============================================================
# LOAD CHECKPOINT DIRECTLY TO GPU
# ============================================================

print("\n" + "=" * 70)
print("LOADING CHECKPOINT TO GPU")
print("=" * 70)

print(f"Loading: {CHECKPOINT_13B}")
genome_np = np.load(CHECKPOINT_13B)
print(f"✅ Loaded: {len(genome_np)/1e9:.2f}GB")

print("Transferring to GPU...")
genome_best = cp.array(genome_np, dtype=cp.int8)
genome_trial = cp.zeros_like(genome_best)
del genome_np  # Free RAM immediately
print("✅ Model on GPU")

PHYSICAL_SIZE = len(genome_best)
print(f"Model size: {PHYSICAL_SIZE/1e9:.2f}GB")

# ============================================================
# LOAD DATASETS TO GPU (ONE AT A TIME)
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS TO GPU")
print("=" * 70)

datasets = []
dataset_names = []

bin_files = sorted(Path(PRETRAIN_DIR).glob("*.bin"))
print(f"Found {len(bin_files)} .bin files\n")

for bin_file in bin_files:
    print(f"Loading: {bin_file.name}...", end=" ", flush=True)
    
    # Load to numpy
    data_np = np.fromfile(str(bin_file), dtype=np.uint8)
    
    # Transfer to GPU
    data_gpu = cp.array(data_np, dtype=cp.uint8)
    datasets.append(data_gpu)
    dataset_names.append(bin_file.name)
    
    # Free RAM immediately
    del data_np
    
    print(f"✅ {len(data_gpu)/1e6:.0f}M tokens")

print(f"\n✅ {len(datasets)} datasets loaded to GPU")

# Pre-calculate max offsets
max_offsets = []
for data in datasets:
    max_off = len(data) - (BATCH_SIZE * SEQ_LEN) - 1
    max_offsets.append(max(0, max_off))

valid_idx = [i for i, m in enumerate(max_offsets) if m > 0]
print(f"Valid datasets: {len(valid_idx)}/{len(datasets)}")

# ============================================================
# EVOLUTIONARY TRAINING
# ============================================================

print("\n" + "=" * 70)
print("EVOLUTIONARY TRAINING")
print("=" * 70)

start_time = time.time()
gen = START_GEN
best_loss = 9999.0

print(f"Starting at Gen {gen:,}\n")

while gen < TARGET_GEN:
    gen += 1
    
    # MUTATE
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
    
    # Auto-save
    if gen % 1000 == 0:
        print(f"\n💾 Saving Gen {gen:,}...", end=" ", flush=True)
        genome_cpu = cp.asnumpy(genome_best)
        np.save(f"{BASE_DIR}/zetagrid_gen{gen}.npy", genome_cpu)
        del genome_cpu
        print(f"✅ Best: {best_loss:.6f}")
    
    # Logging
    if gen % 50 == 0:
        elapsed = time.time() - start_time
        hz = (gen - START_GEN) / elapsed if elapsed > 0 else 0
        eta_min = (TARGET_GEN - gen) / hz / 60 if hz > 0 else 0
        
        print(f"Gen {gen:,} | {hz:.1f}Hz | Loss: {loss:.5f} | Best: {best_loss:.5f} | ETA: {eta_min:.0f}min | DS: {dataset_idx}")

# Final save
print(f"\n💾 Saving final checkpoint...")
genome_cpu = cp.asnumpy(genome_best)
np.save(f"{BASE_DIR}/zetagrid_13b_final.npy", genome_cpu)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Final generation: {gen:,}")
print(f"Best loss: {best_loss:.6f}")
print(f"Total time: {(time.time()-start_time)/3600:.1f}h")
print(f"Saved: zetagrid_13b_final.npy")
print("\n✅ Done!")
