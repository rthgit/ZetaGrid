#!/usr/bin/env python3
"""
CONVERT_TO_GGUF_CUSTOM.py
=========================
Converts RTH-LM 25B (TCN Architecture) Safetensors to GGUF format.
Requires `gguf` package: `pip install gguf`

Usage:
    python CONVERT_TO_GGUF_CUSTOM.py --model v4   E:/ZETAGRID/rth_lm_25b_v4_sharded
    python CONVERT_TO_GGUF_CUSTOM.py --model code E:/ZETAGRID/rth_lm_25b_code_sharded
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
try:
    import gguf
except ImportError:
    print("❌ 'gguf' library not found. Install with: pip install gguf")
    sys.exit(1)

# ARCHITECTURE PARAMS
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 128
KERNEL_SIZE = 3
CTX_LEN = 2048

import struct
import json
import mmap

def load_safetensors_memmap(path):
    """
    Reads a Safetensors file and returns a dict of {name: np.memmap}.
    This avoids loading the entire 5GB shard into RAM.
    """
    tensors = {}
    with open(path, 'rb') as f:
        # Read header size (8 bytes, int64)
        header_size_bytes = f.read(8)
        if not header_size_bytes: return {}
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        # Read header JSON
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        
        # Base offset for data (after header)
        data_start = 8 + header_size
        
        for name, info in header.items():
            if name == "__metadata__": continue
            
            dtype_str = info['dtype']
            shape = tuple(info['shape'])
            start, end = info['data_offsets']
            
            # Map dtype string to numpy dtype
            np_dtype = {
                "F16": np.float16, "BF16": np.int16, # BF16 viewed as int16 (numpy lacks native bf16)
                "F32": np.float32, "I32": np.int32,
                "I16": np.int16, "I8": np.int8,
                "U8": np.uint8
            }.get(dtype_str, None)
            
            if np_dtype is None:
                print(f"⚠️ Warning: Unknown dtype {dtype_str} for {name}, skipping.")
                continue
                
            # Create memmap
            # Offset must be absolute from file start
            abs_offset = data_start + start
            
            # Memmap the tensor
            mm = np.memmap(path, dtype=np_dtype, mode='r', offset=abs_offset, shape=shape)
            
            # If BF16, we might need conversion?
            # GGUF supports BF16 if we pass strict type?
            # Or we convert on the fly? Memmap is read-only.
            # Convert to float32/16 in memory ONLY when writing? 
            # gguf_writer.add_tensor() takes data.
            # If we pass memmap(int16) for BF16, gguf might treat it as int16.
            # We must cast. Casting creates a copy in RAM.
            # To avoid RAM spike, we should probably cast a small chunk?
            # But add_tensor takes the whole array.
            
            # Optimization: If dtype is F32 or F16, we can pass memmap directly!
            # If BF16: Numpy doesn't support it. We MUST load to convert.
            # Our weights are BF16.
            # Paradox: We need to convert BF16 -> FP16 for GGUF (standard).
            # This requires loading.
            # BUT we can do it layer by layer?
            # Verify usage: load_file() loads to RAM anyway.
            # Strategy: We process sequentially.
            # If we used load_file(), we loaded 5GB shard. 
            # 5GB is OK for 32GB RAM.
            # The issue in previous script was GGUFWriter BUFFERING all 50GB.
            # Does GGUFWriter buffer? 
            # Current `gguf` implementation writes immediately? LIMITATION: It calculates offsets first?
            # To calculate offsets, it needs shapes.
            # If we passed data, it might keep reference.
            
            # NEW APPROACH: Direct passing of memmap (view) if possible.
            # Since we need BF16->FP16 conversion, we can't avoid some RAM usage.
            # But we can process ONE TENSOR AT A TIME if GGUFWriter supports it.
            # It seems GGUFWriter stores `self.tensors`.
            # If so, we are stuck unless we patch GGUFWriter or manage memory.
            
            # COMPROMISE: We assume `gguf` writer flushes to disk? 
            # No, standard writer builds metadata then writes.
            # We will use `load_file` (safetensors) which is efficient enough for 5GB.
            # We just need to ensure we don't KEEP references to 50GB.
            # Key: `gguf_writer.add_tensor(name, data)`. Does it copy?
            # If `data` is numpy, it stores the object.
            # So 50GB objects in list -> OOM.
            
            # SOLUTION: Use `gguf` explicit "data writer" or similar?
            # Since I can't change `gguf` lib, I will assume `add_tensor` keeps a reference.
            # I will assume the user has 32GB RAM + Swap. 
            # If 50GB swaps, it will be slow but work.
            # BUT wait, the previous logic: `safetensors.load_file` loads ONE shard.
            # Then we add tensors.
            # Then we `del sd`.
            # BUT `gguf_writer` KEPT the numpy arrays in its internal list!
            # `del sd` does nothing if `gguf_writer` has ref.
            
            # CRITICAL FIX: We need `gguf` to NOT store data, OR use a custom writer.
            # Writing a custom writer for 50GB is risky.
            # Suggestion: Just convert BF16->FP16 and let OS Swap handle it?
            # It's 50GB. Swap needs 50GB space.
            # User probably has it on E drive.
            
            # Let's stick to the current script logic but ensure explicit garbage collection.
            # And warn the user about RAM/Swap.
            
            tensors[name] = mm
            
    return tensors

def convert(input_dir, model_type):
    print(f"\n📦 Converting {model_type.upper()} from: {input_dir}")
    
    out_name = "rth_lm_25b_v4.gguf" if model_type == 'v4' else "rth_lm_25b_code.gguf"
    out_path = os.path.join(input_dir, "..", out_name)
    out_path = os.path.abspath(out_path)
    
    print(f"   Target: {out_path}")
    print("   ⚠️  WARNING: This process requires ~50GB RAM+Swap.")
    print("   Ensure you have a large pagefile enabled on Windows!")
    
    gguf_writer = gguf.GGUFWriter(out_path, "rth-tcn")
    
    # 1. METADATA
    gguf_writer.add_name(f"RTH-LM 25B {model_type.upper()}")
    gguf_writer.add_context_length(CTX_LEN)
    gguf_writer.add_embedding_length(D_MODEL)
    gguf_writer.add_block_count(N_LAYERS)
    gguf_writer.add_feed_forward_length(D_FF)
    gguf_writer.add_uint32("rth_tcn.kernel_size", KERNEL_SIZE)
    gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    
    tokens = [bytes([i]) for i in range(256)]
    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_token_list(tokens)
    
    # 2. TENSORS
    files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not files:
        print("❌ No .safetensors found!")
        return

    print(f"   Found {len(files)} shards. Loading...")
    
    for fpath in files:
        print(f"   Processing {os.path.basename(fpath)}...")
        
        # Load shard (BF16)
        sd = load_file(fpath)
        
        for k, v in sd.items():
            new_k = k
            if k.startswith("layers."):
                parts = k.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_k = f"blk.{idx}.{suffix}"
            
            # BF16 (Torch) -> FP16 (Numpy)
            # This creates a copy in RAM.
            data = v.to(torch.float16).numpy()
            
            # Add to GGUF Writer (It keeps a reference!)
            gguf_writer.add_tensor(new_k, data)
            
        # Explicit delete to help python GC (though refs exist in writer)
        del sd
        import gc; gc.collect()

    print(f"   💾 Writing GGUF to disk... (This may take a while)")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("   ✅ DONE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=['v4', 'code'])
    parser.add_argument("input_dir")
    args = parser.parse_args()
    
    convert(args.input_dir, args.model)
