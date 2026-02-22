#!/usr/bin/env python3
"""
QUANTIZE_TO_QULP.py
===================
Converts RTH-LM 25B Sharded Safetensors -> QULP 2-bit Format (~6GB).
Handles V4 and Code models.

Usage:
    python QUANTIZE_TO_QULP.py --model v4   --input "E:/ZETAGRID/rth_lm_25b_v4_sharded"
    python QUANTIZE_TO_QULP.py --model code --input "E:/ZETAGRID/rth_lm_25b_code_sharded"
"""

import os
import sys
import glob
import argparse
import torch
import numpy as np
from safetensors.torch import load_file
import gc

# 2-BIT QUANTIZATION LOGIC
def quantize_tensor_2bit(tensor):
    """
    Quantizes a float/bf16 tensor to 2-bit (stored as uint8).
    Returns: q_data (uint8), scales (fp16), zeros (fp16), shape, pad
    """
    # 1. Pad to multiple of 4 (for potential bitpacking, though we store uint8 here for simplicity/compatibility)
    # Actually, let's just flatten
    orig_shape = tensor.shape
    t = tensor.float().flatten()
    
    # 2. Min-Max quantization
    # We want 4 levels: 0, 1, 2, 3
    
    # Block-wise quantization (Group Size 128) for accuracy
    group_size = 128
    pad = 0
    if t.numel() % group_size != 0:
        pad = group_size - (t.numel() % group_size)
        t = torch.nn.functional.pad(t, (0, pad), value=0.0)
    
    t = t.view(-1, group_size)
    
    # Min/Max per group
    min_val = t.min(dim=1, keepdim=True)[0]
    max_val = t.max(dim=1, keepdim=True)[0]
    
    # Scale & Zero
    scale = (max_val - min_val) / 3.0
    scale[scale == 0] = 1e-8 # Avoid div by zero
    zero = min_val
    
    # Quantize: (x - zero) / scale
    q = ((t - zero) / scale).round().clamp(0, 3).to(torch.uint8)
    
    return q.flatten(), scale.to(torch.float16), zero.to(torch.float16), orig_shape, pad

def process_model(model_type, input_dir):
    out_name = "rth_lm_25b_v4.qulp" if model_type == 'v4' else "rth_lm_25b_code.qulp"
    out_path = os.path.join(input_dir, "..", out_name)
    out_path = os.path.abspath(out_path)
    
    print(f"\n🧊 Quantizing {model_type.upper()} to 2-BIT QULP...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {out_path}")
    
    # Find shards
    shards = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not shards:
        print("❌ No .safetensors found!")
        sys.exit(1)
        
    final_dict = {
        "metadata": {
            "version": "qulp_v1", 
            "type": model_type,
            "layers": 128,
            "quantization": "2bit_group128"
        },
        "model": {}
    }
    
    total_params = 0
    compressed_size = 0
    
    for shard in shards:
        print(f"   Processing shard: {os.path.basename(shard)}...", end='\r')
        sd = load_file(shard)
        
        for k, v in sd.items():
            # If weight (Linear or Conv) -> Quantize
            # Skip norms, scales, embeddings (keep FP16)
            is_weight = ("w_in" in k or "w_out" in k or "w_dw" in k or "lora" in k)
            if is_weight and v.numel() > 1024:
                # Quantize
                q_data, scales, zeros, shape, pad = quantize_tensor_2bit(v)
                
                # Store packed struct
                final_dict["model"][k] = {
                    "q": q_data.cpu(),         # The 2-bit indices (stored as uint8 for now)
                    "s": scales.flatten().cpu(), # Scales
                    "z": zeros.flatten().cpu(),  # Zeros
                    "sh": shape,               # Original shape
                    "p": pad                   # Padding
                }
                
                # Calc size (simulated 2-bit packing for sanity check)
                # In this file on disk, q_data is uint8 (1 byte per param).
                # REAL 2-bit packing would divide this by 4.
                # But PyTorch pickle doesn't support sub-byte tensors types natively easily without extensions.
                # SO: We save as uint8. File will be ~12GB (INT8 size).
                # To get 6GB (4-bit equivalent) or 3GB (2-bit), we need manual bitpacking.
                
                # IMPLEMENTING MANUAL BITPACKING FOR TRUE SIZE
                # 4 values (0..3) per byte.
                # 00 01 10 11 -> 1 byte
                
                q_np = q_data.numpy()
                # Reshape to (N/4, 4)
                if q_np.size % 4 != 0:
                    # Should be covered by padding loop above but double check
                    pad_extra = 4 - (q_np.size % 4)
                    q_np = np.pad(q_np, (0, pad_extra), 'constant')
                    final_dict["model"][k]["p"] += pad_extra
                
                q_reshaped = q_np.reshape(-1, 4)
                # Pack: (d0 << 6) | (d1 << 4) | (d2 << 2) | d3
                packed = (q_reshaped[:, 0] << 6) | (q_reshaped[:, 1] << 4) | (q_reshaped[:, 2] << 2) | q_reshaped[:, 3]
                packed_tensor = torch.from_numpy(packed.astype(np.uint8))
                
                final_dict["model"][k]["q"] = packed_tensor
                
                compressed_size += packed_tensor.numel() # 1 byte represents 4 params
                
            else:
                # Keep FP16
                final_dict["model"][k] = v.to(torch.float16).cpu()
                compressed_size += v.numel() * 2
                
            total_params += v.numel()
        
        del sd
        gc.collect()

    print(f"\n   💾 Saving dictionary to disk... (This performs the compression)")
    torch.save(final_dict, out_path)
    
    file_size = os.path.getsize(out_path) / 1024**3
    print(f"   ✅ DONE! Final File Size: {file_size:.2f} GB")
    print(f"      To use this, you need a QULP-compatible loader.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=['v4', 'code'])
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    
    process_model(args.model, args.input)
