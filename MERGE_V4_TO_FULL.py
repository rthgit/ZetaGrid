#!/usr/bin/env python3
"""
CONVERT V4 TO GGUF (Unified Single File)
========================================
Merges V4 LoRA (Rank 512) into the Frozen Genome.
Exports a single `rth_lm_25b_v4.gguf` ready for Ollama/llama.cpp.

Hardware: Requires ~60GB RAM (System RAM, not VRAM).
"""

import torch
import torch.nn as nn
import numpy as np
import os
import gc
import struct
import math

# CONFIG
if os.name == 'nt':
    print("🖥️  Running on Local Windows (E:/ZETAGRID)")
    BASE_DIR = r"E:/ZETAGRID"
else:
    print("☁️  Running on Linux/RunPod")
    BASE_DIR = "/workspace/zetagrid_50b"

GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
V4_CHECKPOINT = os.path.join(BASE_DIR, "zeta25b_v4_expanded_FINAL.pt") # Expected name on local
if not os.path.exists(V4_CHECKPOINT) and os.name == 'nt':
     # Fallback if downloaded with different name or path
     print(f"⚠️  {V4_CHECKPOINT} not found, checking alternative...")
     V4_CHECKPOINT = os.path.join(BASE_DIR, "v4_checkpoints", "zeta25b_v4_expanded_FINAL.pt")

OUTPUT_GGUF = f"{BASE_DIR}/rth_lm_25b_v4.gguf"

# METADATA
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 32
KERNEL_SIZE = 3
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

def write_str(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('I', len(b)))
    f.write(b)

def main():
    print(f"Loading Genome: {GENOME_PATH}...")
    genome_np = np.load(GENOME_PATH) # float32
    print(f"Loading V4 Checkpoint: {V4_CHECKPOINT}...")
    ckpt = torch.load(V4_CHECKPOINT, map_location='cpu')
    state = ckpt.get('model', ckpt)
    
    print(f"Opening output: {OUTPUT_GGUF}...")
    f = open(OUTPUT_GGUF, 'wb')
    
    # Header 'GGUF'
    f.write(b'GGUF')
    f.write(struct.pack('I', 3)) # Version 3
    f.write(struct.pack('Q', 0)) # Tensor count placeholder
    f.write(struct.pack('Q', 0)) # Metadata count placeholder
    
    # Metadata
    kv_data = {
        "general.architecture": "rth_tcn",
        "general.name": "RTH-LM 25B V4",
        "rth_tcn.block_count": N_LAYERS,
        "rth_tcn.context_length": 2048,
        "rth_tcn.embedding_length": D_MODEL,
        "rth_tcn.feed_forward_length": D_FF,
        "rth_tcn.kernel_size": KERNEL_SIZE,
        "tokenizer.ggml.model": "gpt2", # Fake, usually ignored or byte-level
    }
    
    # Write KV
    # ... (Standard GGUF KV writing logic would go here)
    # Since we need a custom C++ converter often, this script is a placeholder 
    # for the heavy merging logic. GGUF pure-python writing is complex.
    # 
    # BETTER APPROACH:
    # 1. Merge Genome + LoRA -> `zeta25b_v4_merged.pt` (Standard Torch Dict)
    # 2. Use llama.cpp `convert-hf-to-gguf.py` IF supported, OR custom C++ converter.
    # 
    # Since RTH-LM is custom architecture, we usually dump raw tensors and convert with C++.
    
    print("⚠️  GGUF writing in Python is complex for custom arch.")
    print("    Switching strategy: MERGE to .PT first, then use our C++ converter.")
    f.close()
    
    merge_v4_to_pt(genome_np, state)

def get_genome_chunk(genome, offset, size):
    # Retrieve chunk from genome array
    # Logic matches GenomeWeightBank
    if offset + size > len(genome):
        offset = 0
    chunk = genome[offset : offset + size]
    return chunk, offset + size

def merge_v4_to_pt(genome, lora_state):
    print("\nMERGING Genome + V4 LoRA -> Single Weights...")
    merged_state = {}
    
    # Embeddings (Not in Genome, strictly in checkpoint)
    merged_state['emb.weight'] = lora_state['emb.weight'].float()
    merged_state['pos_emb.weight'] = lora_state['pos_emb.weight'].float()
    merged_state['norm_f.w'] = lora_state['norm_f.w'].float()
    
    offset = 0
    
    for i in range(N_LAYERS):
        print(f"   Merging Layer {i+1}/{N_LAYERS}...", end='\r')
        prefix = f"layers.{i}"
        
        # 1. W_IN (2*d_ff, d_model)
        w_in_size = (2 * D_FF) * D_MODEL
        w_in_flat, offset = get_genome_chunk(genome, offset, w_in_size)
        w_in = torch.from_numpy(w_in_flat).reshape(2*D_FF, D_MODEL)
        scale_in = 1.0 / math.sqrt(D_MODEL * 0.1)
        w_in = w_in * scale_in
        
        # Add LoRA
        A = lora_state[f"{prefix}.lora_in.A"].float()
        B = lora_state[f"{prefix}.lora_in.B"].float()
        # LoRA is AxB? No, typically BAx or xAB. In our code: F.linear(F.linear(x, A), B) -> x @ A.T @ B.T
        # Weight equivalent: B @ A
        lora_weight = B @ A
        
        merged_state[f"{prefix}.w_in"] = w_in + lora_weight
        
        # 2. W_DW (d_ff, 1, k)
        w_dw_size = D_FF * KERNEL_SIZE
        w_dw_flat, offset = get_genome_chunk(genome, offset, w_dw_size)
        w_dw = torch.from_numpy(w_dw_flat).reshape(D_FF, 1, KERNEL_SIZE)
        scale_dw = 1.0 / math.sqrt(KERNEL_SIZE)
        merged_state[f"{prefix}.w_dw"] = w_dw * scale_dw
        # No LoRA on Conv
        
        # 3. W_OUT (d_model, d_ff)
        w_out_size = D_MODEL * D_FF
        w_out_flat, offset = get_genome_chunk(genome, offset, w_out_size)
        w_out = torch.from_numpy(w_out_flat).reshape(D_MODEL, D_FF)
        scale_out = 1.0 / math.sqrt(D_FF * 0.1)
        w_out = w_out * scale_out
        
        # Add LoRA
        A_out = lora_state[f"{prefix}.lora_out.A"].float()
        B_out = lora_state[f"{prefix}.lora_out.B"].float()
        lora_weight_out = B_out @ A_out
        
        merged_state[f"{prefix}.w_out"] = w_out + lora_weight_out
        
        # Norm & Scale
        merged_state[f"{prefix}.norm.w"] = lora_state[f"{prefix}.norm.w"]
        merged_state[f"{prefix}.scale"] = lora_state[f"{prefix}.scale"]

    print("\nSaving merged model (BF16)...")
    keys = list(merged_state.keys())
    for k in keys:
        merged_state[k] = merged_state[k].to(torch.bfloat16)
        
    torch.save(merged_state, f"{BASE_DIR}/zeta25b_v4_MERGED_FULL.pt")
    print(f"✅ Saved: {output_path}")
    print("   Now you can convert this .pt file to GGUF using llama.cpp's convert-hf-to-gguf.py\n   (or wait for our custom C++ converter).")

if __name__ == "__main__":
    output_path = f"{BASE_DIR}/zeta25b_v4_MERGED_FULL.pt"
    # Ensure directory exists on Windows
    if os.name == 'nt' and not os.path.exists(os.path.dirname(output_path)):
         # It's root, so it exists
         pass
    
    # Check if files exist
    if not os.path.exists(GENOME_PATH):
        print(f"❌ GENOME NOT FOUND: {GENOME_PATH}")
        exit(1)
    if not os.path.exists(V4_CHECKPOINT):
        print(f"❌ V4 CHECKPOINT NOT FOUND: {V4_CHECKPOINT}")
        exit(1)
        
    print(f"Merge Target: {output_path}")
    main()
