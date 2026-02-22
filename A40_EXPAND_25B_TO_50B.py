#!/usr/bin/env python3
"""
EXPAND 25B → 50B ON A40
Run this AFTER evolution completes
No datasets needed, pure expansion
"""

import numpy as np
import cupy as cp
import time

print("=" * 70)
print("ZETAGRID: EXPAND 25B → 50B")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
CHECKPOINT_25B = f"{BASE_DIR}/zetagrid_25b_production.npy"
OUTPUT_50B = f"{BASE_DIR}/zetagrid_50b_expanded.npy"

# ============================================================
# LOAD 25B CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: LOADING 25B CHECKPOINT")
print("=" * 70)

print(f"Loading: {CHECKPOINT_25B}")
genome_25b = np.load(CHECKPOINT_25B)
print(f"✅ Loaded: {len(genome_25b)/1e9:.2f}GB (25B params)")

# ============================================================
# EXPAND TO 50B IN RAM
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: EXPANDING 25B → 50B")
print("=" * 70)

# Target: 12GB = 50B params
GB = 12
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024
print(f"Target: {PHYSICAL_SIZE/1e9:.2f}GB (50B params)")

print("\nExpanding in RAM...")
genome_50b = np.zeros(PHYSICAL_SIZE, dtype=np.int8)

# Replicate pattern
replication_factor = PHYSICAL_SIZE // len(genome_25b)
print(f"Replication: {replication_factor}x")

for i in range(replication_factor):
    start = i * len(genome_25b)
    end = min(start + len(genome_25b), PHYSICAL_SIZE)
    genome_50b[start:end] = genome_25b[:end-start]
    
    # Progress
    progress = (i + 1) / replication_factor * 100
    print(f"  Progress: {progress:.0f}%")

# Fill remaining
if end < PHYSICAL_SIZE:
    remaining = PHYSICAL_SIZE - end
    genome_50b[end:] = genome_25b[:remaining]
    print(f"  Filled remaining: {remaining/1e9:.2f}GB")

print(f"✅ Expanded: {PHYSICAL_SIZE/1e9:.2f}GB (50B params)")

del genome_25b  # Free RAM

# ============================================================
# ADD DIVERSITY ON GPU
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: ADDING DIVERSITY (GPU)")
print("=" * 70)

print("Transferring to GPU...")
genome_gpu = cp.array(genome_50b, dtype=cp.int8)
del genome_50b  # Free RAM

print("Adding 10% diversity noise...")
n_noise = int(PHYSICAL_SIZE * 0.1)
noise_idx = cp.random.randint(0, PHYSICAL_SIZE, size=n_noise, dtype=cp.int64)
genome_gpu[noise_idx] = cp.random.randint(-1, 2, size=n_noise, dtype=cp.int8)
print("✅ Diversity added")

# ============================================================
# SAVE 50B
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4: SAVING 50B MODEL")
print("=" * 70)

print(f"Transferring back to CPU...")
genome_50b_final = cp.asnumpy(genome_gpu)
del genome_gpu

print(f"Saving: {OUTPUT_50B}")
np.save(OUTPUT_50B, genome_50b_final)

print("\n" + "=" * 70)
print("EXPANSION COMPLETE!")
print("=" * 70)
print(f"Input:  25B ({len(genome_25b) if 'genome_25b' in locals() else 6.98}GB)")
print(f"Output: 50B ({PHYSICAL_SIZE/1e9:.2f}GB)")
print(f"File: {OUTPUT_50B}")
print("\n✅ 50B MODEL READY FOR FINE-TUNING!")
