#!/usr/bin/env python3
"""
Download 10GB GitHub Code Dataset to RunPod
From: simiotic/github-code-snippets (64GB total)
Sample: First 10GB for code fine-tuning
"""

import os
import sys

print("=" * 70)
print("DOWNLOADING GITHUB CODE DATASET (10GB)")
print("=" * 70)

# Install dependencies
print("\n[1/4] Installing dependencies...")
os.system("pip install -q kagglehub pandas")

import kagglehub
import pandas as pd
import numpy as np

# Setup
BASE_DIR = "/workspace/zetagrid_50b"
OUTPUT_DIR = f"{BASE_DIR}/data/code"
OUTPUT_FILE = f"{OUTPUT_DIR}/github_code_10gb.bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[2/4] Downloading dataset from Kaggle...")
print("Dataset: simiotic/github-code-snippets")
print("Note: This may take 10-15 minutes...")

# Download dataset
try:
    # Load dataset (will download to cache)
    dataset_path = kagglehub.dataset_download("simiotic/github-code-snippets")
    print(f"✅ Downloaded to: {dataset_path}")
    
    # Find CSV/parquet files
    import glob
    data_files = glob.glob(f"{dataset_path}/**/*.csv", recursive=True)
    data_files += glob.glob(f"{dataset_path}/**/*.parquet", recursive=True)
    
    print(f"\n[3/4] Processing files...")
    print(f"Found {len(data_files)} data files")
    
    # Process files and sample 10GB
    all_code = []
    target_size = 10 * 1024 * 1024 * 1024  # 10GB
    current_size = 0
    
    for i, file_path in enumerate(data_files):
        print(f"  Processing file {i+1}/{len(data_files)}: {os.path.basename(file_path)}")
        
        # Load file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)
        
        # Extract code column (adjust column name if needed)
        code_column = None
        for col in ['code', 'content', 'snippet', 'text']:
            if col in df.columns:
                code_column = col
                break
        
        if code_column is None:
            print(f"    ⚠️ No code column found, using first text column")
            code_column = df.select_dtypes(include=['object']).columns[0]
        
        # Sample code
        for code in df[code_column].dropna():
            code_bytes = code.encode('utf-8', errors='ignore')
            all_code.append(code_bytes)
            current_size += len(code_bytes)
            
            # Stop when we reach 10GB
            if current_size >= target_size:
                print(f"    ✅ Reached 10GB target")
                break
        
        if current_size >= target_size:
            break
        
        print(f"    Current size: {current_size/1e9:.2f}GB")
    
    print(f"\n[4/4] Converting to binary format...")
    
    # Concatenate all code
    full_code = b'\n'.join(all_code)
    
    # Convert to uint8 array (tokenize by byte)
    code_array = np.frombuffer(full_code, dtype=np.uint8)
    
    # Save as .bin
    print(f"Saving to: {OUTPUT_FILE}")
    code_array.tofile(OUTPUT_FILE)
    
    final_size = os.path.getsize(OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"File: {OUTPUT_FILE}")
    print(f"Size: {final_size/1e9:.2f}GB")
    print(f"Tokens: {len(code_array):,}")
    print("\n✅ Ready for 25B code fine-tuning!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure Kaggle API credentials are set:")
    print("   export KAGGLE_USERNAME=your_username")
    print("   export KAGGLE_KEY=your_key")
    print("2. Or place kaggle.json in ~/.kaggle/")
    sys.exit(1)
