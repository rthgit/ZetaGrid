# 🦅 KAGGLE V25 (DATASET FIX)

## 📚 Dataset Options
1. **Wikipedia Small** (Built-in, fast)
2. **The Pile** (HuggingFace, massive)
3. **Custom Upload**

## 1. CODICE V25:

```python
# ==========================================
# 🦅 ZETAGRID V25 (DATASET FIX)
# ==========================================
import os
import time
import numpy as np
import cupy as cp

print("📦 ZETAGRID V25 (DATASET FIX)...")

# --- DATASET SELECTION ---
DATASET_MODE = "wikipedia"  # Options: "wikipedia", "pile", "custom"

if DATASET_MODE == "custom":
    # Upload your file to Kaggle Input
    DATASET_PATH = "/kaggle/input/your-dataset/data.txt"
    print(f"📚 Loading Custom: {DATASET_PATH}")
    with open(DATASET_PATH, "rb") as f:
        text_data = f.read()

elif DATASET_MODE == "pile":
    # The Pile from HuggingFace (requires datasets library)
    print("📚 Loading The Pile from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/pile", split="train", streaming=True)
        # Take first 100MB
        text_chunks = []
        total_size = 0
        for item in ds:
            chunk = item['text'].encode('utf-8')
            text_chunks.append(chunk)
            total_size += len(chunk)
            if total_size > 100 * 1024 * 1024:  # 100MB
                break
        text_data = b''.join(text_chunks)
        print(f"   ✅ Loaded {len(text_data)/1e6:.1f}MB from Pile")
    except:
        print("   ❌ HuggingFace failed, falling back to Wikipedia")
        DATASET_MODE = "wikipedia"

if DATASET_MODE == "wikipedia":
    # Simple Wikipedia download (always works)
    print("📚 Loading Wikipedia Sample...")
    import urllib.request
    
    # Use wikipedia dump (reliable)
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2"
    
    # Fallback: Use Gutenberg (very reliable)
    print("   Using Project Gutenberg texts...")
    texts = []
    urls = [
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride & Prejudice
        "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
        "https://www.gutenberg.org/files/11/11-0.txt",      # Alice
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock
    ]
    
    for url in urls:
        try:
            data = urllib.request.urlopen(url).read()
            texts.append(data)
            print(f"   ✅ Downloaded {len(data)/1024:.0f}KB")
        except:
            print(f"   ⚠️  Skipped {url}")
    
    text_data = b''.join(texts)
    print(f"   ✅ Total: {len(text_data)/1e6:.1f}MB")

# --- LOAD TO GPU ---
tokens_gpu = cp.array(np.frombuffer(text_data, dtype=np.uint8))
DATA_LEN = len(tokens_gpu)
print(f"📚 Dataset Ready: {DATA_LEN/1e6:.1f}M Chars on GPU")

# --- MODEL ---
GB = 5
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024
PARAMS_B = (PHYSICAL_SIZE * 4) / 1e9

print(f"🧠 Model: {PARAMS_B:.0f}B Params")

genome_best = cp.zeros(PHYSICAL_SIZE, dtype=cp.int8)
genome_trial = cp.zeros(PHYSICAL_SIZE, dtype=cp.int8)

print("🧬 EVOLUTION STARTED")

start = time.time()
gen = 0
best_loss = 9999.0
BATCH_SIZE = 8192
SEQ_LEN = 128

while True:
    gen += 1
    
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * 0.005)
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations, dtype=cp.int64)
    genome_trial[mut_indices] = cp.random.randint(-1, 2, size=n_mutations, dtype=cp.int8)
    
    max_offset = DATA_LEN - (BATCH_SIZE * SEQ_LEN) - 2
    if max_offset < 0: max_offset = 0
    offset = np.random.randint(0, max(1, max_offset))
    
    input_chunk = tokens_gpu[offset : offset + BATCH_SIZE * SEQ_LEN]
    if len(input_chunk) < BATCH_SIZE * SEQ_LEN:
        continue
    
    input_chunk = input_chunk.reshape(BATCH_SIZE, SEQ_LEN)
    inputs = input_chunk.astype(cp.float32) / 255.0
    
    target_chunk = tokens_gpu[offset + 1 : offset + 1 + BATCH_SIZE * SEQ_LEN]
    if len(target_chunk) < BATCH_SIZE * SEQ_LEN:
        continue
    
    target_chunk = target_chunk.reshape(BATCH_SIZE, SEQ_LEN)
    targets = target_chunk[:, -1].astype(cp.float32) / 255.0
    
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    predictions = cp.tanh(cp.dot(inputs, weights))
    
    diff = predictions - targets
    loss = float(cp.mean(diff ** 2))
    
    if loss > 0.000001:
        if loss < best_loss:
            best_loss = loss
            cp.copyto(genome_best, genome_trial)
    
    if gen % 50 == 0:
        dt = time.time() - start
        hz = gen / dt
        tparams_s = (PHYSICAL_SIZE * 4 * hz) / 1e12
        print(f"🧬 Gen {gen} | {hz:.1f} Hz | Loss: {best_loss:.6f} | {tparams_s:.2f} T-Params/s")
```
