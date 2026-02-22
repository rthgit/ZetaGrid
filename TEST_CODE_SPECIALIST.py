#!/usr/bin/env python3
"""
Test Zetagrid 25B Code Specialist
Compare with base 25B model
"""

import numpy as np
import cupy as cp
import time

print("=" * 70)
print("ZETAGRID 25B CODE SPECIALIST - TEST")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
BASE_MODEL = f"{BASE_DIR}/zetagrid_25b_production.npy"
CODE_MODEL = f"{BASE_DIR}/zetagrid_25b_code_specialist.npy"
CODE_DATASET = f"{BASE_DIR}/data/code/python_code_public.bin"

# ============================================================
# LOAD MODELS
# ============================================================

print("\n[1/4] Loading models...")

print("  Loading base model...")
base_genome = np.load(BASE_MODEL)
base_gpu = cp.array(base_genome, dtype=cp.int8)
del base_genome

print("  Loading code specialist...")
code_genome = np.load(CODE_MODEL)
code_gpu = cp.array(code_genome, dtype=cp.int8)
del code_genome

print(f"✅ Both models loaded ({len(base_gpu)/1e9:.2f}GB each)")

# ============================================================
# LOAD CODE DATASET
# ============================================================

print("\n[2/4] Loading code dataset...")
code_data = cp.array(np.fromfile(CODE_DATASET, dtype=np.uint8))
print(f"✅ Dataset loaded: {len(code_data)/1e9:.2f}GB")

# ============================================================
# TEST ON CODE DATA
# ============================================================

print("\n[3/4] Testing on code samples...")

SEQ_LEN = 256
BATCH_SIZE = 2048
TESTS = 20

base_losses = []
code_losses = []

max_offset = len(code_data) - (BATCH_SIZE * SEQ_LEN) - 1

for i in range(TESTS):
    # Sample code
    offset = np.random.randint(0, max_offset)
    
    input_chunk = code_data[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    
    target_chunk = code_data[offset + 1 : offset + 1 + BATCH_SIZE * SEQ_LEN]
    targets = target_chunk.reshape(BATCH_SIZE, SEQ_LEN)[:, -1].astype(cp.float32) / 255.0
    
    # Test base model
    w_start = np.random.randint(0, len(base_gpu) - SEQ_LEN)
    weights_base = base_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    pred_base = cp.tanh(cp.dot(input_chunk, weights_base))
    loss_base = float(cp.mean((pred_base - targets) ** 2))
    base_losses.append(loss_base)
    
    # Test code specialist
    w_start = np.random.randint(0, len(code_gpu) - SEQ_LEN)
    weights_code = code_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    pred_code = cp.tanh(cp.dot(input_chunk, weights_code))
    loss_code = float(cp.mean((pred_code - targets) ** 2))
    code_losses.append(loss_code)
    
    print(f"  Test {i+1}/{TESTS}: Base={loss_base:.5f} | Code={loss_code:.5f} | {'✅' if loss_code < loss_base else '⚠️'}")

# ============================================================
# RESULTS
# ============================================================

print("\n[4/4] Results Summary")
print("=" * 70)

avg_base = np.mean(base_losses)
avg_code = np.mean(code_losses)
std_base = np.std(base_losses)
std_code = np.std(code_losses)

improvement = ((avg_base - avg_code) / avg_base * 100) if avg_base > 0 else 0

print(f"\n📊 BASE MODEL (25B)")
print(f"   Average Loss: {avg_base:.5f}")
print(f"   Std Dev: {std_base:.5f}")

print(f"\n💻 CODE SPECIALIST (25B)")
print(f"   Average Loss: {avg_code:.5f}")
print(f"   Std Dev: {std_code:.5f}")

print(f"\n📈 IMPROVEMENT")
print(f"   Loss Reduction: {improvement:.1f}%")
print(f"   Wins: {sum(1 for b, c in zip(base_losses, code_losses) if c < b)}/{TESTS}")

if improvement > 10:
    print("\n✅ CODE SPECIALIZATION: EXCELLENT")
    print("   → Model significantly improved on code tasks")
elif improvement > 0:
    print("\n⚠️ CODE SPECIALIZATION: MODERATE")
    print("   → Model shows some improvement")
else:
    print("\n❌ CODE SPECIALIZATION: NEEDS MORE TRAINING")
    print("   → Consider longer fine-tuning or better dataset")

print("\n" + "=" * 70)
