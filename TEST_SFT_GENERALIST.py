#!/usr/bin/env python3
"""
Test Zetagrid 25B SFT Generalist
Compare with base 25B model
"""

import numpy as np
import cupy as cp
import time

print("=" * 70)
print("ZETAGRID 25B SFT GENERALIST - TEST")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
BASE_MODEL = f"{BASE_DIR}/zetagrid_25b_production.npy"
SFT_MODEL = f"{BASE_DIR}/zetagrid_25b_sft_generalist.npy"
SFT_DATASET = f"{BASE_DIR}/data/pretrain/KAM_SFT_MASTER.bin"

# ============================================================
# LOAD MODELS
# ============================================================

print("\n[1/4] Loading models...")

print("  Loading base model...")
base_genome = np.load(BASE_MODEL)
base_gpu = cp.array(base_genome, dtype=cp.int8)
del base_genome

print("  Loading SFT generalist...")
sft_genome = np.load(SFT_MODEL)
sft_gpu = cp.array(sft_genome, dtype=cp.int8)
del sft_genome

print(f"✅ Both models loaded ({len(base_gpu)/1e9:.2f}GB each)")

# ============================================================
# LOAD SFT DATASET
# ============================================================

print("\n[2/4] Loading SFT dataset...")
sft_data = cp.array(np.fromfile(SFT_DATASET, dtype=np.uint8))
print(f"✅ Dataset loaded: {len(sft_data)/1e9:.2f}GB")

# ============================================================
# TEST ON SFT DATA
# ============================================================

print("\n[3/4] Testing on SFT samples...")

SEQ_LEN = 256
BATCH_SIZE = 2048
TESTS = 20

base_losses = []
sft_losses = []

max_offset = len(sft_data) - (BATCH_SIZE * SEQ_LEN) - 1

for i in range(TESTS):
    # Sample SFT data
    offset = np.random.randint(0, max_offset)
    
    input_chunk = sft_data[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    
    target_chunk = sft_data[offset + 1 : offset + 1 + BATCH_SIZE * SEQ_LEN]
    targets = target_chunk.reshape(BATCH_SIZE, SEQ_LEN)[:, -1].astype(cp.float32) / 255.0
    
    # Test base model
    w_start = np.random.randint(0, len(base_gpu) - SEQ_LEN)
    weights_base = base_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    pred_base = cp.tanh(cp.dot(input_chunk, weights_base))
    loss_base = float(cp.mean((pred_base - targets) ** 2))
    base_losses.append(loss_base)
    
    # Test SFT model
    w_start = np.random.randint(0, len(sft_gpu) - SEQ_LEN)
    weights_sft = sft_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    pred_sft = cp.tanh(cp.dot(input_chunk, weights_sft))
    loss_sft = float(cp.mean((pred_sft - targets) ** 2))
    sft_losses.append(loss_sft)
    
    status = '✅' if loss_sft < loss_base else '⚠️'
    print(f"  Test {i+1}/{TESTS}: Base={loss_base:.5f} | SFT={loss_sft:.5f} | {status}")

# ============================================================
# RESULTS
# ============================================================

print("\n[4/4] Results Summary")
print("=" * 70)

avg_base = np.mean(base_losses)
avg_sft = np.mean(sft_losses)
std_base = np.std(base_losses)
std_sft = np.std(sft_losses)

improvement = ((avg_base - avg_sft) / avg_base * 100) if avg_base > 0 else 0
wins = sum(1 for b, s in zip(base_losses, sft_losses) if s < b)

print(f"\n📊 BASE MODEL (25B)")
print(f"   Average Loss: {avg_base:.5f}")
print(f"   Std Dev: {std_base:.5f}")

print(f"\n🎯 SFT GENERALIST (25B)")
print(f"   Average Loss: {avg_sft:.5f}")
print(f"   Std Dev: {std_sft:.5f}")

print(f"\n📈 IMPROVEMENT")
print(f"   Loss Reduction: {improvement:.1f}%")
print(f"   Wins: {wins}/{TESTS}")

if improvement > 15:
    print("\n✅ SFT FINE-TUNING: EXCELLENT")
    print("   → Model significantly improved on SFT tasks")
    print("   → Ready for production use")
elif improvement > 5:
    print("\n⚠️ SFT FINE-TUNING: GOOD")
    print("   → Model shows improvement")
    print("   → Consider longer training for better results")
elif improvement > 0:
    print("\n⚠️ SFT FINE-TUNING: MODERATE")
    print("   → Model shows some improvement")
    print("   → May need more training or better dataset")
else:
    print("\n❌ SFT FINE-TUNING: NO IMPROVEMENT")
    print("   → Base model performs better")
    print("   → Use base model instead")

print("\n" + "=" * 70)

# ============================================================
# SPEED TEST
# ============================================================

print("\n[BONUS] Speed Comparison")
print("=" * 70)

# Test inference speed
SPEED_TESTS = 100

print(f"\nRunning {SPEED_TESTS} inference tests...")

# Base model speed
start = time.time()
for _ in range(SPEED_TESTS):
    w_start = np.random.randint(0, len(base_gpu) - SEQ_LEN)
    weights = base_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    offset = np.random.randint(0, max_offset)
    input_chunk = sft_data[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    pred = cp.tanh(cp.dot(input_chunk, weights))
cp.cuda.Stream.null.synchronize()
base_time = time.time() - start

# SFT model speed
start = time.time()
for _ in range(SPEED_TESTS):
    w_start = np.random.randint(0, len(sft_gpu) - SEQ_LEN)
    weights = sft_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    offset = np.random.randint(0, max_offset)
    input_chunk = sft_data[offset : offset + BATCH_SIZE * SEQ_LEN]
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN).astype(cp.float32) / 255.0
    pred = cp.tanh(cp.dot(input_chunk, weights))
cp.cuda.Stream.null.synchronize()
sft_time = time.time() - start

base_hz = SPEED_TESTS / base_time
sft_hz = SPEED_TESTS / sft_time

print(f"\n⚡ BASE MODEL: {base_hz:.1f} inferences/sec")
print(f"⚡ SFT MODEL:  {sft_hz:.1f} inferences/sec")
print(f"   Speed difference: {((sft_hz - base_hz) / base_hz * 100):.1f}%")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
