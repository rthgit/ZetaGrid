#!/usr/bin/env python3
"""
CONVERT_QULP_TO_GGUF.py
=======================
Converts RTH-LM 25B QULP (2-bit) files directly to GGUF.
Produces a small (~6GB) GGUF file suitable for mobile/distribution.

NOTE: This GGUF uses a CUSTOM SCHEME for quantization.
Standard llama.cpp will NOT run this without a custom kernel.
It saves weights as split tensors:
  - strict_weight.q (uint8 packed 2-bit)
  - strict_weight.s (fp16 scales)
  - strict_weight.z (fp16 zeros)

Usage:
    python CONVERT_QULP_TO_GGUF.py --model v4   --input "E:/ZETAGRID/rth_lm_25b_v4.qulp"
    python CONVERT_QULP_TO_GGUF.py --model code --input "E:/ZETAGRID/rth_lm_25b_code.qulp"
"""

import os
import sys
import argparse
import torch
import numpy as np
try:
    import gguf
except ImportError:
    print("❌ 'gguf' library not found. Install with: pip install gguf")
    sys.exit(1)

# ARCHITECTURE PARAMS
CTX_LEN = 2048
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 128
KERNEL_SIZE = 3

def convert_qulp_to_gguf(input_file, model_type):
    print(f"\n📦 Converting QULP -> GGUF (Mobile/Small)")
    print(f"   Input:  {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    out_path = input_file.replace(".qulp", ".gguf")
    print(f"   Target: {out_path}")
    
    print("   Loading QULP file (this loads ~6GB into RAM)...")
    # Load the dict. RAM usage: ~6-7GB. Safe for 32GB system.
    data = torch.load(input_file, map_location='cpu')
    model_dict = data.get("model", data)
    metadata = data.get("metadata", {})
    
    print(f"   Loaded. Metadata: {metadata}")
    
    gguf_writer = gguf.GGUFWriter(out_path, "rth-tcn")
    
    # 1. METADATA
    print("   Writing GGUF Header...")
    gguf_writer.add_name(f"RTH-LM 25B {model_type.upper()} (QULP 2-bit)")
    gguf_writer.add_context_length(CTX_LEN)
    gguf_writer.add_embedding_length(D_MODEL)
    gguf_writer.add_block_count(N_LAYERS)
    gguf_writer.add_feed_forward_length(D_FF)
    gguf_writer.add_uint32("rth_tcn.kernel_size", KERNEL_SIZE)
    # Mark as F16 mostly, though weights are custom
    gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q2_K) 
    
    tokens = [bytes([i]) for i in range(256)]
    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_token_list(tokens)
    
    # 2. TENSORS
    print("   Writing Tensors...")
    
    for k, v in model_dict.items():
        # Rename RTH -> GGUF
        # layers.0.w_in -> blk.0.w_in
        new_k = k
        if k.startswith("layers."):
            parts = k.split('.')
            idx = parts[1]
            suffix = ".".join(parts[2:])
            new_k = f"blk.{idx}.{suffix}"
        
        # Check if Quantized struct or Raw Tensor
        if isinstance(v, dict) and 'q' in v:
            # It's a QULP 2-bit struct
            # Save components as separate tensors
            
            # 1. Quantized Data (uint8)
            q_data = v['q'].numpy() # Already packed uint8
            # GGUF library (python) only supports I8, not U8.
            # We view as int8 to preserve bits.
            q_data = q_data.view(np.int8)
            gguf_writer.add_tensor(f"{new_k}.q", q_data)
            
            # 2. Scales (fp16)
            s_data = v['s'].numpy().astype(np.float16)
            gguf_writer.add_tensor(f"{new_k}.s", s_data)
            
            # 3. Zeros (fp16)
            z_data = v['z'].numpy().astype(np.float16)
            gguf_writer.add_tensor(f"{new_k}.z", z_data)
            
        elif isinstance(v, torch.Tensor):
            # Regular tensor (norm, emb, etc)
            # Ensure FP16 or FP32
            # BFloat16 -> Float16
            if v.dtype == torch.bfloat16:
                arr = v.float().numpy().astype(np.float16)
            else:
                arr = v.float().numpy().astype(np.float32)
                
            gguf_writer.add_tensor(new_k, arr)
            
        else:
            print(f"⚠️ Unknown type for {k}: {type(v)}")

    print(f"   💾 Writing to disk...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"   ✅ DONE! GGUF Size: {size_gb:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=['v4', 'code'])
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    
    convert_qulp_to_gguf(args.input, args.model)
