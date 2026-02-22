#!/usr/bin/env python3
"""
Download clean UTF-8 text dataset for ZetaGrid Phase 2 training.
Saves raw UTF-8 bytes as .bin file.
"""

import os
import numpy as np

SAVE_PATH = "/workspace/zetagrid_50b/data/pretrain/clean_text_utf8.bin"

print("=" * 60)
print("DOWNLOADING CLEAN UTF-8 TEXT DATASET")
print("=" * 60)

# Install datasets if needed
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets -q")
    from datasets import load_dataset

# Download multiple text datasets for variety
all_text = []

# 1. WikiText-103 (~500MB of clean English text)
print("\n[1/3] Downloading WikiText-103...")
try:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    all_text.append(text)
    print(f"   ✅ WikiText-103: {len(text)/1e6:.0f}M chars")
    del ds
except Exception as e:
    print(f"   ⚠️ WikiText failed: {e}")

# 2. TinyStories (~2GB of clean short stories)
print("\n[2/3] Downloading TinyStories...")
try:
    ds = load_dataset("roneneldan/TinyStories", split="train")
    text = "\n\n".join(ds["text"][:500000])  # First 500K stories
    all_text.append(text)
    print(f"   ✅ TinyStories: {len(text)/1e6:.0f}M chars")
    del ds
except Exception as e:
    print(f"   ⚠️ TinyStories failed: {e}")

# 3. BookCorpus subset
print("\n[3/3] Downloading C4 subset (English)...")
try:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    total_chars = 0
    target = 500_000_000  # 500M chars
    for i, item in enumerate(ds):
        t = item["text"]
        if len(t) > 100:
            texts.append(t)
            total_chars += len(t)
        if total_chars >= target:
            break
        if i % 50000 == 0 and i > 0:
            print(f"   ... {total_chars/1e6:.0f}M chars collected")
    text = "\n\n".join(texts)
    all_text.append(text)
    print(f"   ✅ C4 subset: {len(text)/1e6:.0f}M chars")
    del ds, texts
except Exception as e:
    print(f"   ⚠️ C4 failed: {e}")

if not all_text:
    print("\n❌ No datasets downloaded! Trying fallback...")
    # Fallback: download raw text from web
    import urllib.request
    urls = [
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
        ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein"),
        ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes"),
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick"),
        ("https://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities"),
        ("https://www.gutenberg.org/files/1080/1080-0.txt", "A Modest Proposal"),
        ("https://www.gutenberg.org/files/74/74-0.txt", "Adventures of Tom Sawyer"),
        ("https://www.gutenberg.org/files/16/16-0.txt", "Peter Pan"),
        ("https://www.gutenberg.org/files/46/46-0.txt", "A Christmas Carol"),
    ]
    for url, name in urls:
        try:
            data = urllib.request.urlopen(url).read().decode('utf-8', errors='ignore')
            all_text.append(data)
            print(f"   ✅ {name}: {len(data)/1e3:.0f}K chars")
        except:
            pass

# Combine all text
print(f"\nCombining {len(all_text)} sources...")
combined = "\n\n".join(all_text)
print(f"Total text: {len(combined)/1e6:.0f}M chars")

# Convert to UTF-8 bytes
print("Converting to UTF-8 bytes...")
raw_bytes = combined.encode('utf-8')
data = np.frombuffer(raw_bytes, dtype=np.uint8).copy()

# Verify quality
ascii_pct = np.sum((data >= 32) & (data <= 126)) / len(data) * 100
null_pct = np.sum(data == 0) / len(data) * 100
print(f"\n📊 QUALITY CHECK:")
print(f"   Total size: {len(data)/1e6:.0f}M bytes")
print(f"   ASCII readable: {ascii_pct:.1f}% (should be >80%)")
print(f"   Null bytes: {null_pct:.1f}% (should be 0%)")
print(f"   First 200 chars: {combined[:200]}")

# Save
print(f"\nSaving to {SAVE_PATH}...")
data.tofile(SAVE_PATH)
print(f"✅ Saved! {os.path.getsize(SAVE_PATH)/1e6:.0f}M bytes")

# Show sample
print(f"\n📝 Sample text:")
print(combined[1000:1500])
