#!/usr/bin/env python3
"""
MERGE_CODE_LOW_RAM.py
======================
Merges CODE SPECIALIST (V5) LoRA (Rank 512) into Frozen Genome (7B).
Uses `safetensors` streaming to save memory (Low RAM friendly).
Splits output into 5GB shards.

Requirements:
- pip install safetensors torch numpy

Output:
- E:/ZETAGRID/rth_lm_25b_code_sharded/
    - model-00001-of-00010.safetensors
    - model.safetensors.index.json
    - config.json
"""

import os
import sys
import torch
import numpy as np
import json
import math
import gc

try:
    from safetensors.torch import save_file
except ImportError:
    print("❌ 'safetensors' not installed. Please run: pip install safetensors")
    sys.exit(1)

# CONFIG
if os.name == 'nt':
    print("🖥️  Running on Local Windows (E:/ZETAGRID)")
    BASE_DIR = r"E:/ZETAGRID"
else:
    print("☁️  Running on Linux/RunPod")
    BASE_DIR = "/workspace/zetagrid_50b"

GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
CODE_CHECKPOINT = os.path.join(BASE_DIR, "zeta25b_code_FINAL.pt")
# Fallback check
if not os.path.exists(CODE_CHECKPOINT) and os.name == 'nt':
     alt = os.path.join(BASE_DIR, "code_checkpoints", "zeta25b_code_FINAL.pt")
     if os.path.exists(alt):
         CODE_CHECKPOINT = alt

OUTPUT_DIR = os.path.join(BASE_DIR, "rth_lm_25b_code_sharded")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MODEL SPECS
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 128
KERNEL_SIZE = 3
SHARD_SIZE_LIMIT = 4 * 1024**3  # 4GB per shard

def get_genome_chunk(genome, offset, size):
    # Retrieve chunk from genome array
    if offset + size > len(genome):
        offset = 0
    chunk = genome[offset : offset + size]
    return chunk, offset + size

def main():
    print(f"Loading Genome: {GENOME_PATH}...")
    # Use mmap_mode='r' to keep RAM usage extremely low
    genome_np = np.load(GENOME_PATH, mmap_mode='r') 
    
    print(f"Loading CODE LoRA: {CODE_CHECKPOINT}...")
    # Load LoRA to CPU RAM (~4GB)
    ckpt = torch.load(CODE_CHECKPOINT, map_location='cpu')
    lora_state = ckpt.get('model', ckpt)
    
    print(f"Output Directory: {OUTPUT_DIR}")
    
    current_shard = {}
    current_shard_size = 0
    shard_idx = 1
    weight_map = {}
    
    offset_genome = 0
    
    # EMBEDDINGS (Not in Genome, purely from LoRA check)
    print("\nProcessing Embeddings...")
    common_layers = ['emb.weight', 'pos_emb.weight', 'norm_f.w']
    for k in common_layers:
        t = lora_state[k].to(torch.bfloat16)
        current_shard[k] = t
        current_shard_size += t.numel() * 2 # BF16 = 2 bytes
        weight_map[k] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    
    # LAYERS
    print(f"   Fractal Expansion: 32 Physical -> {N_LAYERS} Virtual Layers")
    
    for i in range(N_LAYERS):
        # FRACTAL TILING: Reuse 32 layers 4 times
        src_idx = i % 32
        
        print(f"   Merging Layer {i+1}/{N_LAYERS} (Source: {src_idx})...", end='\r')
        
        # Keys in V4 Checkpoint use src_idx
        src_prefix = f"layers.{src_idx}"
        # Keys in Output use i
        tgt_prefix = f"layers.{i}"
        
        # 1. W_IN (2*d_ff, d_model)
        # Genome Offset also needs to loop if we want "Fractal Genome"
        # OR does the genome have 32 layers of unique weights?
        # "25B Genome" suggests the GENOME itself is large.
        # But if we only have 6GB of weights in `zetagrid_25b_production.npy`, that's 6B params.
        # If the USER insists on 25B, we must repeat the genome too.
        # We will reset offset_genome every 32 layers.
        
        if src_idx == 0:
            offset_genome = 0 # Reset genome pointer for next block
        
        w_in_size = (2 * D_FF) * D_MODEL
        w_in_flat, offset_genome = get_genome_chunk(genome_np, offset_genome, w_in_size)
        w_in = torch.from_numpy(w_in_flat.copy()).reshape(2*D_FF, D_MODEL) # Copy to RAM for op
        scale_in = 1.0 / math.sqrt(D_MODEL * 0.1)
        w_in = w_in * scale_in
        
        # Add LoRA
        # Use src_prefix to find weights in checkpoint
        A = lora_state[f"{src_prefix}.lora_in.A"].float()
        B = lora_state[f"{src_prefix}.lora_in.B"].float()
        lora_weight = B @ A
        merged_w_in = (w_in + lora_weight).to(torch.bfloat16)
        
        k_in = f"{tgt_prefix}.w_in"
        current_shard[k_in] = merged_w_in
        current_shard_size += merged_w_in.numel() * 2
        weight_map[k_in] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        
        # 2. W_DW (d_ff, 1, k)
        w_dw_size = D_FF * KERNEL_SIZE
        w_dw_flat, offset_genome = get_genome_chunk(genome_np, offset_genome, w_dw_size)
        w_dw = torch.from_numpy(w_dw_flat.copy()).reshape(D_FF, 1, KERNEL_SIZE)
        scale_dw = 1.0 / math.sqrt(KERNEL_SIZE)
        merged_w_dw = (w_dw * scale_dw).to(torch.bfloat16)
        
        k_dw = f"{tgt_prefix}.w_dw"
        current_shard[k_dw] = merged_w_dw
        current_shard_size += merged_w_dw.numel() * 2
        weight_map[k_dw] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        
        # 3. W_OUT (d_model, d_ff)
        w_out_size = D_MODEL * D_FF
        w_out_flat, offset_genome = get_genome_chunk(genome_np, offset_genome, w_out_size)
        w_out = torch.from_numpy(w_out_flat.copy()).reshape(D_MODEL, D_FF)
        scale_out = 1.0 / math.sqrt(D_FF * 0.1)
        w_out = w_out * scale_out
        
        # Add LoRA
        A_out = lora_state[f"{src_prefix}.lora_out.A"].float()
        B_out = lora_state[f"{src_prefix}.lora_out.B"].float()
        lora_weight_out = B_out @ A_out
        merged_w_out = (w_out + lora_weight_out).to(torch.bfloat16)
        
        k_out = f"{tgt_prefix}.w_out"
        current_shard[k_out] = merged_w_out
        current_shard_size += merged_w_out.numel() * 2
        weight_map[k_out] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        
        # Norm & Scale
        # Read from source layer (src_idx) -> Write to target layer (i)
        k_norm_src = f"{src_prefix}.norm.w"
        k_norm_tgt = f"{tgt_prefix}.norm.w"
        current_shard[k_norm_tgt] = lora_state[k_norm_src].to(torch.bfloat16)
        weight_map[k_norm_tgt] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        
        k_scale_src = f"{src_prefix}.scale"
        k_scale_tgt = f"{tgt_prefix}.scale"
        current_shard[k_scale_tgt] = lora_state[k_scale_src].to(torch.bfloat16)
        weight_map[k_scale_tgt] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"

        # Check Flush
        if current_shard_size >= SHARD_SIZE_LIMIT:
            flush_shard(current_shard, shard_idx)
            current_shard = {}
            current_shard_size = 0
            shard_idx += 1
            gc.collect()

    # Final flush
    if current_shard:
        flush_shard(current_shard, shard_idx)
    
    # Write Index
    total_shards = shard_idx
    print(f"\nWriting index.json (Total shards: {total_shards})...")
    
    # Fix placeholders in weight_map
    final_map = {}
    for k, v in weight_map.items():
        # model-00001-of-XXXXX -> model-00001-of-00010
        final_map[k] = v.replace("XXXXX", f"{total_shards:05d}")
    
    # Rename files on disk to match total count
    for idx in range(1, total_shards + 1):
        old_name = os.path.join(OUTPUT_DIR, f"model-{idx:05d}-of-temp.safetensors")
        new_name = os.path.join(OUTPUT_DIR, f"model-{idx:05d}-of-{total_shards:05d}.safetensors")
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"   Renamed shard {idx}: {os.path.basename(new_name)}")

    index_data = {"metadata": {"total_size": 50 * 1024**3}, "weight_map": final_map}
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)
        
    print(f"\n✅ MERGE COMPLETE! Output in: {OUTPUT_DIR}")
    print(f"   Use this folder with llama.cpp: ./convert-hf-to-gguf.py {OUTPUT_DIR} --outfile rth_lm_25b_code.gguf")

def flush_shard(shard, idx):
    fname = os.path.join(OUTPUT_DIR, f"model-{idx:05d}-of-temp.safetensors")
    print(f"\n   💾 Saving Shard {idx}... ({len(shard)} tensors)")
    save_file(shard, fname)
    print(f"      OK: {fname}")

if __name__ == "__main__":
    if not os.path.exists(GENOME_PATH):
        print(f"❌ GENOME NOT FOUND: {GENOME_PATH}")
        exit(1)
    if not os.path.exists(CODE_CHECKPOINT):
        print(f"❌ CODE CHECKPOINT NOT FOUND: {CODE_CHECKPOINT}")
        exit(1)
    main()
