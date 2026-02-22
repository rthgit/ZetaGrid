#!/usr/bin/env python3
"""
DOWNLOAD CODE DATASET (~5GB)
============================
Downloads code-focused datasets for Code Specialist Soul.
Uses authenticated-free sources only.

Sources:
- TheStack v1 dedup (Python, JS, C, Java, Go, Rust) ~3GB
- CodeSearchNet (functions + docstrings) ~1GB
- Golden Mix code entries (local, 3x oversample) ~0.5GB
- Repair Mix (local, 1x) for instruction following ~1GB
"""

import os
import sys
import json
import gc
import time
import numpy as np

BASE_DIR = "/workspace/zetagrid_50b"
OUTPUT_DIR = f"{BASE_DIR}/data/pretrain"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_BIN = f"{OUTPUT_DIR}/code_5gb.bin"

def ensure_datasets():
    try:
        import datasets
        print(f"✅ datasets v{datasets.__version__}")
    except ImportError:
        print("📦 Installing datasets...")
        os.system("pip install datasets -q")
    return __import__('datasets')

def stream_code(ds, text_field, target_bytes, label, log_every=50000):
    texts = []
    total = 0
    for i, row in enumerate(ds):
        t = row.get(text_field, '')
        if isinstance(t, str) and len(t) > 30:
            texts.append(t)
            total += len(t)
        if total >= target_bytes:
            break
        if i % log_every == 0 and i > 0:
            print(f"   ... {label}: {total/1e9:.2f}GB ({i:,} items)")
            sys.stdout.flush()
    print(f"   ✅ {label}: {total/1e9:.2f}GB ({len(texts):,} items)")
    return texts

def download_thestack(ds_lib, target_bytes=3_000_000_000):
    """TheStack v1 dedup — multiple languages."""
    print("\n📖 [1/4] TheStack (Python, JS, C, Java, Go, Rust) ~3GB...")
    languages = ["python", "javascript", "c", "java", "go", "rust"]
    all_texts = []
    per_lang = target_bytes // len(languages)
    
    for lang in languages:
        try:
            print(f"   🔧 {lang}...")
            ds = ds_lib.load_dataset("bigcode/the-stack-dedup", 
                                      data_dir=f"data/{lang}",
                                      split="train", streaming=True)
            texts = stream_code(ds, "content", per_lang, lang, log_every=20000)
            all_texts.extend(texts)
            del texts; gc.collect()
        except Exception as e:
            print(f"   ⚠️ {lang} failed: {e}")
            # Fallback per-language
            try:
                ds = ds_lib.load_dataset(f"codeparrot/codeparrot-clean",
                                          split="train", streaming=True)
                texts = stream_code(ds, "content", per_lang, f"codeparrot-{lang}", log_every=20000)
                all_texts.extend(texts)
                del texts; gc.collect()
            except:
                pass
    
    if not all_texts:
        # Ultimate fallback: use a smaller, open dataset
        print("   🔄 Fallback: nampdn-ai/tiny-codes...")
        try:
            ds = ds_lib.load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
            all_texts = stream_code(ds, "response", target_bytes, "tiny-codes")
        except Exception as e:
            print(f"   ❌ Fallback failed: {e}")
    
    return all_texts

def download_codesearchnet(ds_lib, target_bytes=1_000_000_000):
    """CodeSearchNet — functions with docstrings (great for understanding)."""
    print("\n📖 [2/4] CodeSearchNet (functions + docs) ~1GB...")
    try:
        ds = ds_lib.load_dataset("code_search_net", "all", split="train", streaming=True,
                                  trust_remote_code=True)
        texts = []
        total = 0
        for i, row in enumerate(ds):
            func = row.get('func_code_string', row.get('whole_func_string', ''))
            doc = row.get('func_documentation_string', '')
            if func and len(func) > 30:
                entry = f"# {doc}\n{func}" if doc else func
                texts.append(entry)
                total += len(entry)
            if total >= target_bytes:
                break
            if i % 50000 == 0 and i > 0:
                print(f"   ... CSN: {total/1e9:.2f}GB ({i:,})")
                sys.stdout.flush()
        print(f"   ✅ CodeSearchNet: {total/1e9:.2f}GB ({len(texts):,} items)")
        return texts
    except Exception as e:
        print(f"   ❌ CodeSearchNet failed: {e}")
        # Fallback
        try:
            print("   🔄 Fallback: sahil2801/CodeAlpaca-20k...")
            ds = ds_lib.load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
            texts = []
            total = 0
            for i, row in enumerate(ds):
                prompt = row.get('prompt', row.get('instruction', ''))
                completion = row.get('completion', row.get('output', ''))
                entry = f"# Task: {prompt}\n{completion}"
                if len(entry) > 30:
                    texts.append(entry)
                    total += len(entry)
                if total >= target_bytes:
                    break
            # Oversample if small
            if total < target_bytes // 2:
                factor = max(1, int(target_bytes / total)) 
                texts = texts * min(factor, 10)
                total *= min(factor, 10)
            print(f"   ✅ CodeAlpaca (fallback): {total/1e9:.2f}GB ({len(texts):,})")
            return texts
        except Exception as e2:
            print(f"   ❌ Fallback failed: {e2}")
            return []

def download_evol_instruct(ds_lib, target_bytes=500_000_000):
    """EvolInstruct Code — instruction-tuned code Q&A."""
    print("\n📖 [3/4] Code Instruction Data ~0.5GB...")
    try:
        ds = ds_lib.load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        texts = []
        total = 0
        for i, row in enumerate(ds):
            q = row.get('instruction', '')
            a = row.get('output', '')
            entry = f"### Question:\n{q}\n\n### Answer:\n{a}"
            if len(entry) > 50:
                texts.append(entry)
                total += len(entry)
            if total >= target_bytes:
                break
        print(f"   ✅ EvolInstruct Code: {total/1e9:.2f}GB ({len(texts):,} items)")
        return texts
    except Exception as e:
        print(f"   ❌ EvolInstruct failed: {e}")
        return []

def load_local_jsonl(path, oversample=1):
    print(f"\n📖 [4/4] Local: {os.path.basename(path)} (x{oversample})...")
    texts = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if 'messages' in obj:
                    for msg in obj['messages']:
                        c = msg.get('content', '')
                        if isinstance(c, str) and len(c) > 5:
                            texts.append(c)
                            count += 1
                else:
                    text = obj.get('text', obj.get('content', obj.get('output', '')))
                    if isinstance(text, str) and len(text) > 10:
                        texts.append(text)
                        count += 1
            except:
                if len(line) > 10:
                    texts.append(line)
                    count += 1
    if oversample > 1:
        texts = texts * oversample
        count *= oversample
    total = sum(len(t) for t in texts)
    print(f"   ✅ {os.path.basename(path)}: {count:,} entries → {total/1e6:.0f}MB")
    return texts

def main():
    start = time.time()
    print("=" * 70)
    print("CODE DATASET BUILDER (~5GB)")
    print("=" * 70)
    
    ds_lib = ensure_datasets()
    all_texts = []
    
    # 1. TheStack
    texts = download_thestack(ds_lib)
    all_texts.extend(texts); del texts; gc.collect()
    
    # 2. CodeSearchNet
    texts = download_codesearchnet(ds_lib)
    all_texts.extend(texts); del texts; gc.collect()
    
    # 3. EvolInstruct Code
    texts = download_evol_instruct(ds_lib)
    all_texts.extend(texts); del texts; gc.collect()
    
    # 4. Local repair_mix (has SFT code entries)
    repair_path = f"{BASE_DIR}/repair_mix.jsonl"
    if os.path.exists(repair_path):
        texts = load_local_jsonl(repair_path, oversample=1)
        all_texts.extend(texts); del texts; gc.collect()
    
    if not all_texts:
        print("❌ No data! Aborting.")
        sys.exit(1)
    
    # Save
    print(f"\n{'='*70}")
    print(f"MERGING {len(all_texts):,} texts → {FINAL_BIN}")
    print(f"{'='*70}")
    with open(FINAL_BIN, 'wb') as f:
        for i, text in enumerate(all_texts):
            f.write(text.encode('utf-8'))
            f.write(b'\n')
            if (i+1) % 200000 == 0:
                print(f"   ... {i+1:,}/{len(all_texts):,}")
                sys.stdout.flush()
    
    size = os.path.getsize(FINAL_BIN)
    elapsed = (time.time() - start) / 60
    print(f"\n{'='*70}")
    print(f"✅ DONE in {elapsed:.0f} min | {size/1e9:.2f}GB | {len(all_texts):,} entries")
    print(f"   Output: {FINAL_BIN}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
