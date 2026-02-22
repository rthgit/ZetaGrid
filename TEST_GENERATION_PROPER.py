#!/usr/bin/env python3
"""
ZETAGRID 25B - PROPER TEXT GENERATION
Based on actual training loop logic
"""

import numpy as np
import cupy as cp
from sentencepiece import SentencePieceProcessor

print("=" * 70)
print("ZETAGRID 25B - TEXT GENERATION (PROPER)")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
TOKENIZER_PATH = f"{BASE_DIR}/models/tokenizer.model"
BASE_MODEL = f"{BASE_DIR}/zetagrid_25b_production.npy"
SFT_MODEL = f"{BASE_DIR}/zetagrid_25b_sft_generalist.npy"

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
SEQ_LEN = 128

# ============================================================
# GENERATION FUNCTION (MATCHES TRAINING LOGIC)
# ============================================================

def generate_text(genome_gpu, prompt, max_tokens=50):
    """Generate text using same logic as training"""
    
    # Encode prompt to tokens
    input_ids = tokenizer.encode(prompt)
    
    # Convert tokens to bytes (like training data)
    text_bytes = prompt.encode('utf-8')
    byte_sequence = list(text_bytes)
    
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Input bytes: {len(byte_sequence)}")
    
    # Generate tokens
    for step in range(max_tokens):
        # Prepare input (last SEQ_LEN bytes)
        if len(byte_sequence) >= SEQ_LEN:
            context = byte_sequence[-SEQ_LEN:]
        else:
            # Pad with zeros
            context = [0] * (SEQ_LEN - len(byte_sequence)) + byte_sequence
        
        # Convert to GPU tensor (same as training)
        input_tensor = cp.array(context, dtype=cp.float32) / 255.0
        
        # Sample weights from genome (same as training)
        w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
        weights = genome_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
        
        # Predict next byte (same as training)
        prediction = cp.tanh(cp.dot(input_tensor, weights))
        
        # Convert prediction to byte value
        pred_value = float(prediction)
        next_byte = int((pred_value + 1.0) * 127.5)  # Map [-1,1] to [0,255]
        next_byte = max(0, min(255, next_byte))
        
        # Add to sequence
        byte_sequence.append(next_byte)
        
        # Try to decode
        try:
            text = bytes(byte_sequence).decode('utf-8', errors='ignore')
            # Stop if we have enough valid text
            if len(text) > len(prompt) + 20:
                break
        except:
            continue
    
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
    "The future of AI",
    "Python is",
    "Hello world",
]

for i, prompt in enumerate(prompts):
    print(f"\n{'='*70}")
    print(f"PROMPT {i+1}: \"{prompt}\"")
    print(f"{'='*70}")
    
    # Generate with base model
    print("\n🔵 BASE MODEL:")
    try:
        base_text = generate_text(base_gpu, prompt, max_tokens=30)
        print(f"   OUTPUT: {base_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Generate with SFT model
    print("\n🟢 SFT MODEL:")
    try:
        sft_text = generate_text(sft_gpu, prompt, max_tokens=30)
        print(f"   OUTPUT: {sft_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
