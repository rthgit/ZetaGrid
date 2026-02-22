#!/usr/bin/env python3
"""
ZETAGRID 25B - TEXT GENERATION TEST
Compare Base vs SFT with actual text output
"""

import numpy as np
import cupy as cp
from sentencepiece import SentencePieceProcessor

print("=" * 70)
print("ZETAGRID 25B - TEXT GENERATION COMPARISON")
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

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_text(genome_gpu, prompt, max_tokens=50, temperature=0.8):
    """Generate text from model"""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    
    generated = input_ids.copy()
    SEQ_LEN = 128
    
    for _ in range(max_tokens):
        # Prepare input (last SEQ_LEN tokens)
        context = generated[-SEQ_LEN:] if len(generated) >= SEQ_LEN else generated
        
        # Pad if needed
        if len(context) < SEQ_LEN:
            context = [0] * (SEQ_LEN - len(context)) + context
        
        # Convert to tensor
        input_tensor = cp.array(context, dtype=cp.float32) / VOCAB_SIZE
        
        # Sample weights from genome
        w_start = np.random.randint(0, len(genome_gpu) - SEQ_LEN)
        weights = genome_gpu[w_start : w_start + SEQ_LEN].astype(cp.float32)
        
        # Predict next token logits
        logits = cp.dot(input_tensor, weights)
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        probs = cp.exp(logits - cp.max(logits))
        probs = probs / cp.sum(probs)
        
        # Sample token
        probs_cpu = cp.asnumpy(probs)
        
        # Map to vocab range (handle scalar)
        if probs_cpu.ndim == 0:
            token_id = int(np.abs(float(probs_cpu)) * VOCAB_SIZE) % VOCAB_SIZE
        else:
            token_id = int(np.abs(probs_cpu[-1]) * VOCAB_SIZE) % VOCAB_SIZE
        
        # Add to generated
        generated.append(token_id)
        
        # Stop on EOS
        if token_id == tokenizer.eos_id():
            break
    
    # Decode
    text = tokenizer.decode(generated)
    return text

# ============================================================
# TEST PROMPTS
# ============================================================

print("\n[3/4] Generating text samples...")
print("=" * 70)

prompts = [
    "The future of artificial intelligence is",
    "In a world where technology",
    "Once upon a time, there was",
    "The most important thing in life is",
    "Python programming is"
]

for i, prompt in enumerate(prompts):
    print(f"\n{'='*70}")
    print(f"PROMPT {i+1}: \"{prompt}\"")
    print(f"{'='*70}")
    
    # Generate with base model
    print("\n🔵 BASE MODEL:")
    try:
        base_text = generate_text(base_gpu, prompt, max_tokens=30)
        print(f"   {base_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Generate with SFT model
    print("\n🟢 SFT MODEL:")
    try:
        sft_text = generate_text(sft_gpu, prompt, max_tokens=30)
        print(f"   {sft_text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("[4/4] INTERACTIVE MODE")
print("=" * 70)
print("\nType prompts to compare models (or 'quit' to exit)")

while True:
    try:
        user_prompt = input("\n📝 Your prompt: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_prompt:
            continue
        
        print(f"\n{'='*70}")
        
        # Base
        print("🔵 BASE MODEL:")
        base_text = generate_text(base_gpu, user_prompt, max_tokens=40)
        print(f"   {base_text}")
        
        # SFT
        print("\n🟢 SFT MODEL:")
        sft_text = generate_text(sft_gpu, user_prompt, max_tokens=40)
        print(f"   {sft_text}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
