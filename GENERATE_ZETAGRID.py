#!/usr/bin/env python3
"""
ZETAGRID 25B - PROPER AUTOREGRESSIVE GENERATION
Uses genome for deterministic byte-level text generation
"""

import numpy as np
import cupy as cp
from sentencepiece import SentencePieceProcessor
import time

print("=" * 70)
print("ZETAGRID 25B - AUTOREGRESSIVE TEXT GENERATION")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
TOKENIZER_PATH = f"{BASE_DIR}/models/tokenizer.model"
BASE_MODEL = f"{BASE_DIR}/zetagrid_25b_production.npy"
SFT_MODEL = f"{BASE_DIR}/zetagrid_25b_sft_generalist.npy"

SEQ_LEN = 128

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n[1/4] Loading tokenizer...")
tokenizer = SentencePieceProcessor()
tokenizer.Load(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size()
print(f"✅ Tokenizer loaded: {VOCAB_SIZE:,} tokens")

# ============================================================
# LOAD MODELS
# ============================================================

print("\n[2/4] Loading models...")

print("  Loading base model...")
base_genome = np.load(BASE_MODEL)
base_gpu = cp.array(base_genome, dtype=cp.int8)
del base_genome

print("  Loading SFT model...")
sft_genome = np.load(SFT_MODEL)
sft_gpu = cp.array(sft_genome, dtype=cp.int8)
del sft_genome

print(f"✅ Both models loaded ({len(base_gpu)/1e9:.2f}GB each)")

PHYSICAL_SIZE = len(base_gpu)

# ============================================================
# GENERATION FUNCTION - AUTOREGRESSIVE
# ============================================================

def generate_text_autoregressive(genome_gpu, prompt, max_bytes=200, weight_offset=0):
    """
    Generate text autoregressively using genome weights
    
    Args:
        genome_gpu: Model weights on GPU
        prompt: Starting text
        max_bytes: Maximum bytes to generate
        weight_offset: Starting position in genome for weights (for diversity)
    """
    
    # Convert prompt to bytes
    byte_sequence = list(prompt.encode('utf-8'))
    
    print(f"\n  Generating from: \"{prompt}\"")
    print(f"  Starting bytes: {len(byte_sequence)}")
    
    # Generate bytes one at a time
    for step in range(max_bytes):
        # Prepare context (last SEQ_LEN bytes)
        if len(byte_sequence) >= SEQ_LEN:
            context = byte_sequence[-SEQ_LEN:]
        else:
            # Pad with zeros
            context = [0] * (SEQ_LEN - len(byte_sequence)) + byte_sequence
        
        # Convert to GPU (same as training)
        input_tensor = cp.array(context, dtype=cp.float32) / 255.0
        
        # Use FIXED weights from genome (deterministic)
        # Cycle through genome for different weight sets
        w_start = (weight_offset + step * SEQ_LEN) % (PHYSICAL_SIZE - SEQ_LEN)
        weights = genome_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
        
        # Predict next byte (same as training)
        prediction = float(cp.tanh(cp.dot(input_tensor, weights)))
        
        # Convert prediction [-1, 1] to byte [0, 255]
        next_byte = int((prediction + 1.0) * 127.5)
        next_byte = max(0, min(255, next_byte))
        
        # Add to sequence
        byte_sequence.append(next_byte)
        
        # Try to decode periodically
        if step % 10 == 0:
            try:
                text = bytes(byte_sequence).decode('utf-8', errors='ignore')
                # Stop if we have a good amount of text
                if len(text) > len(prompt) + 50:
                    break
            except:
                pass
    
    # Final decode
    try:
        final_text = bytes(byte_sequence).decode('utf-8', errors='ignore')
        return final_text
    except:
        return prompt + " [decode error]"

# ============================================================
# TEST PROMPTS
# ============================================================

print("\n[3/4] Generating text samples...")
print("=" * 70)

prompts = [
    "Hello",
    "The future",
    "Python",
]

for i, prompt in enumerate(prompts):
    print(f"\n{'='*70}")
    print(f"PROMPT {i+1}: \"{prompt}\"")
    print(f"{'='*70}")
    
    # Generate with base model
    print("\n🔵 BASE MODEL:")
    try:
        base_text = generate_text_autoregressive(base_gpu, prompt, max_bytes=100, weight_offset=0)
        print(f"   {base_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Generate with SFT model
    print("\n🟢 SFT MODEL:")
    try:
        sft_text = generate_text_autoregressive(sft_gpu, prompt, max_bytes=100, weight_offset=0)
        print(f"   {sft_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("[4/4] INTERACTIVE MODE")
print("=" * 70)
print("Type prompts to test (or 'quit' to exit)\n")

while True:
    try:
        user_prompt = input("📝 Prompt: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_prompt:
            continue
        
        print(f"\n{'='*70}")
        
        # Base
        print("🔵 BASE:")
        base_text = generate_text_autoregressive(base_gpu, user_prompt, max_bytes=150)
        print(f"   {base_text}")
        
        # SFT
        print("\n🟢 SFT:")
        sft_text = generate_text_autoregressive(sft_gpu, user_prompt, max_bytes=150)
        print(f"   {sft_text}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
