#!/usr/bin/env python3
"""
Download Code Dataset from HuggingFace (NO Kaggle needed)
Dataset: codeparrot/github-code (Python subset, 10-15GB)
"""

import os
import sys

print("=" * 70)
print("DOWNLOADING CODE DATASET FROM HUGGINGFACE")
print("=" * 70)

# Install dependencies
print("\n[1/4] Installing dependencies...")
os.system("pip install -q datasets numpy")

from datasets import load_dataset
import numpy as np

# Setup
BASE_DIR = "/workspace/zetagrid_50b"
OUTPUT_DIR = f"{BASE_DIR}/data/code"
OUTPUT_FILE = f"{OUTPUT_DIR}/python_code_hf.bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[2/4] Downloading Python code from HuggingFace...")
print("Dataset: codeparrot/github-code (Python only)")
print("Target: 10-15GB")
print("Note: Using streaming to avoid full download\n")

try:
    # Load dataset with streaming (efficient)
    dataset = load_dataset(
        "codeparrot/github-code",
        streaming=True,
        split="train",
        languages=["Python"]  # Python only
    )
    
    print("✅ Dataset stream opened")
    
    print(f"\n[3/4] Sampling code files...")
    
    all_code = []
    total_size = 0
    target_size = 12 * 1024 * 1024 * 1024  # 12GB target
    
    file_count = 0
    
    for item in dataset:
        # Extract code
        code = item.get('code', '')
        
        if len(code) < 100:  # Skip very small files
            continue
        
        code_bytes = code.encode('utf-8', errors='ignore')
        all_code.append(code_bytes)
        total_size += len(code_bytes)
        file_count += 1
        
        # Progress
        if file_count % 1000 == 0:
            print(f"  Files: {file_count:,} | Size: {total_size/1e9:.2f}GB")
        
        # Stop at target
        if total_size >= target_size:
            print(f"\n  ✅ Reached target: {total_size/1e9:.2f}GB")
            break
    
    print(f"\n[4/4] Converting to binary format...")
    
    # Concatenate
    full_code = b'\n'.join(all_code)
    
    # Convert to uint8
    code_array = np.frombuffer(full_code, dtype=np.uint8)
    
    # Save
    print(f"Saving to: {OUTPUT_FILE}")
    code_array.tofile(OUTPUT_FILE)
    
    final_size = os.path.getsize(OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"File: {OUTPUT_FILE}")
    print(f"Size: {final_size/1e9:.2f}GB")
    print(f"Tokens: {len(code_array):,}")
    print(f"Files: {file_count:,}")
    print("\n✅ Ready for 25B code fine-tuning!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying alternative: The Stack dataset...")
    
    # Fallback: The Stack (smaller, curated)
    try:
        dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            streaming=True,
            split="train"
        )
        
        print("✅ Using The Stack dataset (Python)")
        
        all_code = []
        total_size = 0
        target_size = 12 * 1024 * 1024 * 1024
        file_count = 0
        
        for item in dataset:
            code = item.get('content', '')
            
            if len(code) < 100:
                continue
            
            code_bytes = code.encode('utf-8', errors='ignore')
            all_code.append(code_bytes)
            total_size += len(code_bytes)
            file_count += 1
            
            if file_count % 1000 == 0:
                print(f"  Files: {file_count:,} | Size: {total_size/1e9:.2f}GB")
            
            if total_size >= target_size:
                break
        
        # Save
        full_code = b'\n'.join(all_code)
        code_array = np.frombuffer(full_code, dtype=np.uint8)
        code_array.tofile(OUTPUT_FILE)
        
        print(f"\n✅ Saved: {os.path.getsize(OUTPUT_FILE)/1e9:.2f}GB")
        
    except Exception as e2:
        print(f"\n❌ Fallback also failed: {e2}")
        sys.exit(1)
