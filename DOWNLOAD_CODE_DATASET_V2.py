#!/usr/bin/env python3
"""
Download High-Quality Code Dataset
Best option: 150k Python Dataset from Kaggle (clean, filtered, ~10GB)
"""

import os
import sys

print("=" * 70)
print("DOWNLOADING 150K PYTHON CODE DATASET")
print("=" * 70)

# Install dependencies
print("\n[1/5] Installing dependencies...")
os.system("pip install -q kaggle numpy")

import kaggle
import numpy as np
import zipfile
import glob

# Setup
BASE_DIR = "/workspace/zetagrid_50b"
OUTPUT_DIR = f"{BASE_DIR}/data/code"
OUTPUT_FILE = f"{OUTPUT_DIR}/python_code_150k.bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[2/5] Downloading dataset from Kaggle...")
print("Dataset: AmeerHamza/150k-python-dataset")
print("Size: ~10GB, 150K high-quality Python files")
print("Quality: Filtered, no duplicates, permissive licenses")

try:
    # Download dataset
    kaggle.api.dataset_download_files(
        'ameerhamza/150k-python-dataset',
        path=OUTPUT_DIR,
        unzip=True
    )
    
    print("✅ Download complete")
    
    print(f"\n[3/5] Finding Python files...")
    
    # Find all .py files
    py_files = glob.glob(f"{OUTPUT_DIR}/**/*.py", recursive=True)
    print(f"Found {len(py_files)} Python files")
    
    if len(py_files) == 0:
        # Try alternative structure
        py_files = glob.glob(f"{OUTPUT_DIR}/**/*.txt", recursive=True)
        print(f"Found {len(py_files)} text files (alternative)")
    
    print(f"\n[4/5] Processing files...")
    
    all_code = []
    total_size = 0
    
    for i, file_path in enumerate(py_files):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(py_files)} files ({total_size/1e9:.2f}GB)")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
                code_bytes = code.encode('utf-8', errors='ignore')
                all_code.append(code_bytes)
                total_size += len(code_bytes)
        except Exception as e:
            continue
    
    print(f"  ✅ Processed all files: {total_size/1e9:.2f}GB")
    
    print(f"\n[5/5] Converting to binary format...")
    
    # Concatenate with newlines
    full_code = b'\n'.join(all_code)
    
    # Convert to uint8 array
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
    print(f"Files: {len(py_files)}")
    print("\n✅ Ready for 25B code fine-tuning!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure Kaggle API credentials are set:")
    print("   mkdir -p ~/.kaggle")
    print("   nano ~/.kaggle/kaggle.json")
    print('   {"username":"YOUR_USERNAME","key":"YOUR_KEY"}')
    print("   chmod 600 ~/.kaggle/kaggle.json")
    sys.exit(1)
