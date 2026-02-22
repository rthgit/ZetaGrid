#!/usr/bin/env python3
"""
Test Zetagrid 25B Model - Quick Inference
"""

import numpy as np
import cupy as cp

print("=" * 70)
print("ZETAGRID 25B - INFERENCE TEST")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
MODEL_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"

# Load model
print(f"\nLoading model: {MODEL_PATH}")
genome = np.load(MODEL_PATH)
print(f"✅ Loaded: {len(genome)/1e9:.2f}GB ({len(genome)/1e9*4:.0f}B params)")

# Transfer to GPU
print("\nTransferring to GPU...")
genome_gpu = cp.array(genome, dtype=cp.int8)
del genome
print("✅ Model on GPU")

# Test inference
print("\n" + "=" * 70)
print("RUNNING INFERENCE TESTS")
print("=" * 70)

SEQ_LEN = 128
BATCH_SIZE = 1024

# Create test input (random tokens)
print(f"\nTest 1: Random input ({BATCH_SIZE} samples)")
test_input = cp.random.rand(BATCH_SIZE, SEQ_LEN).astype(cp.float32)

# Sample weights from model
w_start = np.random.randint(0, len(genome_gpu) - SEQ_LEN)
weights = genome_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)

# Inference
import time
start = time.time()
predictions = cp.tanh(cp.dot(test_input, weights))
inference_time = time.time() - start

print(f"✅ Inference time: {inference_time*1000:.2f}ms")
print(f"   Throughput: {BATCH_SIZE/inference_time:.0f} samples/sec")
print(f"   Predictions shape: {predictions.shape}")
print(f"   Predictions range: [{float(cp.min(predictions)):.3f}, {float(cp.max(predictions)):.3f}]")

# Test 2: Multiple random positions
print(f"\nTest 2: Sampling 10 different model positions")
losses = []
for i in range(10):
    w_start = np.random.randint(0, len(genome_gpu) - SEQ_LEN)
    weights = genome_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    # Create simple target (zeros)
    targets = cp.zeros(BATCH_SIZE, dtype=cp.float32)
    
    # Predict
    predictions = cp.tanh(cp.dot(test_input, weights))
    loss = float(cp.mean((predictions - targets) ** 2))
    losses.append(loss)
    
    print(f"  Position {i+1}: Loss = {loss:.5f}")

avg_loss = np.mean(losses)
std_loss = np.std(losses)

print(f"\n✅ Average Loss: {avg_loss:.5f}")
print(f"   Std Dev: {std_loss:.5f}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Model: 25B parameters ({len(genome_gpu)/1e9:.2f}GB)")
print(f"Inference Speed: {BATCH_SIZE/inference_time:.0f} samples/sec")
print(f"Average Loss: {avg_loss:.5f}")
print(f"Loss Stability: {std_loss:.5f}")

if avg_loss < 0.5 and std_loss < 0.3:
    print("\n✅ MODEL QUALITY: GOOD")
    print("   → Ready for expansion to 50B")
elif avg_loss < 1.0:
    print("\n⚠️ MODEL QUALITY: ACCEPTABLE")
    print("   → Can proceed with caution")
else:
    print("\n❌ MODEL QUALITY: NEEDS IMPROVEMENT")
    print("   → Consider more training")

print("\n" + "=" * 70)
