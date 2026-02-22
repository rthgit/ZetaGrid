#!/usr/bin/env python3
"""
DOWNLOAD & PREPARE 10GB KNOWLEDGE DATASET (V2 - FIXED SOURCES)
===============================================================
Downloads from HuggingFace and converts to raw .bin format.

Dataset Mix (adjusted):
- Wikipedia EN: ~4GB  (wikimedia/wikipedia 20231101.en)
- Wikipedia IT: ~1GB  (wikimedia/wikipedia 20231101.it)
- StackExchange: ~1.5GB (flax-sentence/stackexchange_titlebody_best)
- C4 (sample): ~1.5GB (allenai/c4)
- Books: ~1GB (emozilla/pg19)
- Code: ~1GB (codeparrot/github-code)
- Golden Mix (local): 26MB x5
- Repair Mix (local): 1.4GB x1

Total: ~11GB raw UTF-8 bytes
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

FINAL_BIN = f"{OUTPUT_DIR}/knowledge_10gb.bin"

# ============================================================
# HELPERS
# ============================================================

def ensure_datasets():
    try:
        import datasets
        print(f"✅ datasets library v{datasets.__version__}")
    except ImportError:
        print("📦 Installing datasets library...")
        os.system("pip install datasets -q")
    return __import__('datasets')

def stream_to_texts(ds, text_field, target_bytes, label, log_every=100000):
    """Generic streamer: iterate dataset, extract text field, stop at target bytes."""
    texts = []
    total = 0
    for i, row in enumerate(ds):
        t = row.get(text_field, '')
        if isinstance(t, str) and len(t) > 50:
            texts.append(t)
            total += len(t)
        if total >= target_bytes:
            break
        if i % log_every == 0 and i > 0:
            print(f"   ... {label}: {total/1e9:.2f}GB ({i:,} items)")
            sys.stdout.flush()
    print(f"   ✅ {label}: {total/1e9:.2f}GB ({len(texts):,} items)")
    sys.stdout.flush()
    return texts

# ============================================================
# DOWNLOAD FUNCTIONS (FIXED SOURCES)
# ============================================================

def download_wikipedia_en(ds_lib, target_bytes=4_000_000_000):
    print("\n📖 [1/6] Wikipedia EN (~4GB)...")
    try:
        ds = ds_lib.load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        return stream_to_texts(ds, "text", target_bytes, "Wiki-EN")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        # Fallback: try plain text dataset
        try:
            print("   🔄 Trying fallback: graelo/wikipedia...")
            ds = ds_lib.load_dataset("graelo/wikipedia", "20230601.en", split="train", streaming=True)
            return stream_to_texts(ds, "text", target_bytes, "Wiki-EN-fallback")
        except Exception as e2:
            print(f"   ❌ Fallback also failed: {e2}")
            return []

def download_wikipedia_it(ds_lib, target_bytes=1_000_000_000):
    print("\n📖 [2/6] Wikipedia IT (~1GB)...")
    try:
        ds = ds_lib.load_dataset("wikimedia/wikipedia", "20231101.it", split="train", streaming=True)
        return stream_to_texts(ds, "text", target_bytes, "Wiki-IT")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        try:
            print("   🔄 Trying fallback: graelo/wikipedia IT...")
            ds = ds_lib.load_dataset("graelo/wikipedia", "20230601.it", split="train", streaming=True)
            return stream_to_texts(ds, "text", target_bytes, "Wiki-IT-fallback")
        except Exception as e2:
            print(f"   ❌ Fallback failed: {e2}")
            return []

def download_stackexchange(ds_lib, target_bytes=1_500_000_000):
    print("\n📖 [3/6] StackExchange (~1.5GB)...")
    try:
        # Use a simpler, more reliable dataset
        ds = ds_lib.load_dataset("flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl",
                                  split="train", streaming=True)
        texts = []
        total = 0
        for i, row in enumerate(ds):
            title = row.get('title_body', row.get('title', ''))
            answer = row.get('upvoted_answer', row.get('answer', ''))
            t = f"Q: {title}\nA: {answer}" if answer else title
            if len(t) > 50:
                texts.append(t)
                total += len(t)
            if total >= target_bytes:
                break
            if i % 100000 == 0 and i > 0:
                print(f"   ... SE: {total/1e9:.2f}GB ({i:,} items)")
                sys.stdout.flush()
        print(f"   ✅ StackExchange: {total/1e9:.2f}GB ({len(texts):,} items)")
        return texts
    except Exception as e:
        print(f"   ❌ StackExchange failed: {e}")
        # Fallback: use OpenAssistant conversations
        try:
            print("   🔄 Trying fallback: OpenAssistant...")
            ds = ds_lib.load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
            texts = []
            total = 0
            for i, row in enumerate(ds):
                t = row.get('text', '')
                if len(t) > 50:
                    texts.append(t)
                    total += len(t)
                if total >= target_bytes:
                    break
            print(f"   ✅ OpenAssistant (fallback): {total/1e9:.2f}GB")
            return texts
        except:
            return []

def download_c4(ds_lib, target_bytes=1_500_000_000):
    print("\n📖 [4/6] C4 Web Text (~1.5GB)...")
    try:
        ds = ds_lib.load_dataset("allenai/c4", "en", split="train", streaming=True,
                                  trust_remote_code=True)
        return stream_to_texts(ds, "text", target_bytes, "C4")
    except Exception as e:
        print(f"   ❌ C4 failed: {e}")
        # Fallback: use FineWeb
        try:
            print("   🔄 Trying fallback: FineWeb-Edu...")
            ds = ds_lib.load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True,
                                      name="sample-10BT")
            return stream_to_texts(ds, "text", target_bytes, "FineWeb")
        except Exception as e2:
            print(f"   ❌ Fallback failed: {e2}")
            return []

def download_books(ds_lib, target_bytes=1_000_000_000):
    print("\n📖 [5/6] Books (~1GB)...")
    try:
        ds = ds_lib.load_dataset("emozilla/pg19", split="train", streaming=True)
        return stream_to_texts(ds, "text", target_bytes, "Books", log_every=500)
    except Exception as e:
        print(f"   ❌ Books failed: {e}")
        # Fallback: use Gutenberg
        try:
            print("   🔄 Trying fallback: sedthh/gutenberg_english...")
            ds = ds_lib.load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
            return stream_to_texts(ds, "TEXT", target_bytes, "Gutenberg", log_every=500)
        except:
            return []

def download_code(ds_lib, target_bytes=1_000_000_000):
    print("\n📖 [6/6] Code (~1GB)...")
    try:
        ds = ds_lib.load_dataset("codeparrot/github-code", streaming=True, split="train",
                                  languages=["Python", "JavaScript", "C", "Java"],
                                  trust_remote_code=True)
        return stream_to_texts(ds, "code", target_bytes, "Code")
    except Exception as e:
        print(f"   ❌ github-code failed: {e}")
        try:
            print("   🔄 Trying fallback: bigcode/starcoderdata...")
            ds = ds_lib.load_dataset("bigcode/starcoderdata", data_dir="python",
                                      split="train", streaming=True)
            return stream_to_texts(ds, "content", target_bytes, "StarCoder")
        except Exception as e2:
            print(f"   ❌ Fallback failed: {e2}")
            return []

# ============================================================
# LOCAL JSONL LOADER
# ============================================================

def load_local_jsonl(path, oversample=1):
    print(f"\n📖 Loading local: {os.path.basename(path)} (x{oversample})...")
    texts = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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

# ============================================================
# MERGE & SAVE
# ============================================================

def save_texts_to_bin(all_texts, output_path):
    print(f"\n{'='*70}")
    print(f"MERGING {len(all_texts):,} texts → .bin")
    print(f"{'='*70}")
    
    # Write incrementally to avoid RAM spike
    print("   Writing to disk...")
    with open(output_path, 'wb') as f:
        for i, text in enumerate(all_texts):
            f.write(text.encode('utf-8'))
            f.write(b'\n')
            if (i+1) % 500000 == 0:
                print(f"   ... {i+1:,}/{len(all_texts):,}")
                sys.stdout.flush()
    
    size = os.path.getsize(output_path)
    print(f"   ✅ Saved: {output_path} ({size/1e9:.2f}GB)")
    return size

# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    
    print("=" * 70)
    print("KNOWLEDGE DATASET BUILDER V2 (10GB)")
    print("=" * 70)
    
    ds_lib = ensure_datasets()
    
    all_texts = []
    sources_ok = 0
    
    # Download each source
    for name, func in [
        ("Wikipedia EN", lambda: download_wikipedia_en(ds_lib)),
        ("Wikipedia IT", lambda: download_wikipedia_it(ds_lib)),
        ("StackExchange", lambda: download_stackexchange(ds_lib)),
        ("C4", lambda: download_c4(ds_lib)),
        ("Books", lambda: download_books(ds_lib)),
        ("Code", lambda: download_code(ds_lib)),
    ]:
        try:
            texts = func()
            if texts:
                all_texts.extend(texts)
                sources_ok += 1
                del texts
                gc.collect()
            print(f"   📊 Running total: {sum(len(t) for t in all_texts[-1000:])/1e6:.0f}MB (last 1K) | {len(all_texts):,} texts")
        except Exception as e:
            print(f"   ❌ {name} completely failed: {e}")
    
    # Local data
    golden_path = f"{BASE_DIR}/golden_mix_220b.jsonl"
    if os.path.exists(golden_path):
        texts = load_local_jsonl(golden_path, oversample=5)
        all_texts.extend(texts); del texts; gc.collect()
    
    repair_path = f"{BASE_DIR}/repair_mix.jsonl"
    if os.path.exists(repair_path):
        texts = load_local_jsonl(repair_path, oversample=1)
        all_texts.extend(texts); del texts; gc.collect()
    
    if not all_texts:
        print("❌ No data collected! Aborting.")
        sys.exit(1)
    
    # Save
    save_texts_to_bin(all_texts, FINAL_BIN)
    del all_texts; gc.collect()
    
    elapsed = (time.time() - start) / 60
    print(f"\n{'='*70}")
    print(f"✅ DONE in {elapsed:.0f} minutes")
    print(f"   Sources OK: {sources_ok}/6")
    print(f"   Output: {FINAL_BIN}")
    print(f"   Size: {os.path.getsize(FINAL_BIN)/1e9:.2f}GB")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
