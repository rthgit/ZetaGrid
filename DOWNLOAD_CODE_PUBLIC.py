#!/usr/bin/env python3
"""
Download Public Python Code - NO AUTH NEEDED
Source: GitHub Archive via wget (public, free)
"""

import os
import sys
import glob

print("=" * 70)
print("DOWNLOADING PUBLIC PYTHON CODE")
print("=" * 70)

BASE_DIR = "/workspace/zetagrid_50b"
OUTPUT_DIR = f"{BASE_DIR}/data/code"
OUTPUT_FILE = f"{OUTPUT_DIR}/python_code_public.bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n[1/3] Downloading Python repositories...")
print("Source: Public GitHub repos (no auth)")

# Download multiple Python projects
repos = [
    "https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip",
    "https://github.com/donnemartin/system-design-primer/archive/refs/heads/master.zip",
    "https://github.com/vinta/awesome-python/archive/refs/heads/master.zip",
    "https://github.com/pallets/flask/archive/refs/heads/main.zip",
    "https://github.com/django/django/archive/refs/heads/main.zip",
    "https://github.com/psf/requests/archive/refs/heads/main.zip",
    "https://github.com/numpy/numpy/archive/refs/heads/main.zip",
    "https://github.com/pandas-dev/pandas/archive/refs/heads/main.zip",
    "https://github.com/scikit-learn/scikit-learn/archive/refs/heads/main.zip",
    "https://github.com/pytorch/pytorch/archive/refs/heads/main.zip",
]

temp_dir = f"{OUTPUT_DIR}/temp"
os.makedirs(temp_dir, exist_ok=True)

for i, repo_url in enumerate(repos):
    print(f"\n  [{i+1}/{len(repos)}] {repo_url.split('/')[-3]}")
    zip_file = f"{temp_dir}/repo_{i}.zip"
    
    # Download
    os.system(f"wget -q -O {zip_file} {repo_url}")
    
    # Unzip
    os.system(f"unzip -q -o {zip_file} -d {temp_dir}")
    
    # Remove zip
    os.remove(zip_file)

print("\n✅ Downloaded all repositories")

print("\n[2/3] Extracting Python files...")

import numpy as np

py_files = glob.glob(f"{temp_dir}/**/*.py", recursive=True)
print(f"Found {len(py_files)} Python files")

all_code = []
total_size = 0

for i, file_path in enumerate(py_files):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(py_files)} ({total_size/1e9:.2f}GB)")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
            if len(code) > 50:  # Skip tiny files
                code_bytes = code.encode('utf-8', errors='ignore')
                all_code.append(code_bytes)
                total_size += len(code_bytes)
    except:
        continue

print(f"\n✅ Extracted {total_size/1e9:.2f}GB of Python code")

print("\n[3/3] Converting to binary...")

# Concatenate
full_code = b'\n'.join(all_code)

# Convert to uint8
code_array = np.frombuffer(full_code, dtype=np.uint8)

# Save
code_array.tofile(OUTPUT_FILE)

# Cleanup
os.system(f"rm -rf {temp_dir}")

final_size = os.path.getsize(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE!")
print("=" * 70)
print(f"File: {OUTPUT_FILE}")
print(f"Size: {final_size/1e9:.2f}GB")
print(f"Tokens: {len(code_array):,}")
print(f"Files: {len(py_files)}")
print("\n✅ Ready for code fine-tuning!")

# Update fine-tuning script path
finetune_script = f"{BASE_DIR}/A40_FINETUNE_25B_CODE.py"
if os.path.exists(finetune_script):
    with open(finetune_script, 'r') as f:
        content = f.read()
    
    content = content.replace(
        'CODE_DATASET = f"{BASE_DIR}/data/code/python_code_150k.bin"',
        'CODE_DATASET = f"{BASE_DIR}/data/code/python_code_public.bin"'
    )
    
    with open(finetune_script, 'w') as f:
        f.write(content)
    
    print("✅ Updated fine-tuning script")
