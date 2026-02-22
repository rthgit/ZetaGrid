# 🦅 KAGGLE V24 (LARGE DATASET)

## 📚 Dataset Upgrade
- **WikiText-103**: ~500MB, 100M+ tokens
- **Supporto Custom**: Upload your own dataset

## 1. CODICE V24:

```python
# ==========================================
# 🦅 ZETAGRID V24 (LARGE DATASET)
# ==========================================
import os
import time
import numpy as np
import urllib.request
import zipfile
import cupy as cp

print("📦 ZETAGRID V24 (LARGE DATASET)...")

# --- DATASET SELECTION ---
USE_CUSTOM = False  # Set True to use uploaded file

if USE_CUSTOM:
    # Upload your dataset file to Kaggle, then specify path
    DATASET_PATH = "/kaggle/input/your-dataset/data.txt"
    print(f"📚 Using Custom Dataset: {DATASET_PATH}")
    with open(DATASET_PATH, "rb") as f:
        text_data = f.read()
else:
    # Download WikiText-103 (Large)
    if not os.path.exists("wikitext103.txt"):
        print("📚 Downloading WikiText-103 (~500MB)...")
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        print("   Downloading zip...")
        urllib.request.urlretrieve(url, "wiki103.zip")
        
        print("   Extracting...")
        with zipfile.ZipFile("wiki103.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Combine train files
        print("   Combining files...")
        train_path = "wikitext-103-raw/wiki.train.raw"
        with open(train_path, "rb") as f:
            text_data = f.read()
        
        with open("wikitext103.txt", "wb") as f:
            f.write(text_data)
        
        print("   ✅ WikiText-103 Ready")
    else:
        print("📚 Loading WikiText-103 from cache...")
        with open("wikitext103.txt", "rb") as f:
            text_data = f.read()

tokens_gpu = cp.array(np.frombuffer(text_data, dtype=np.uint8))
DATA_LEN = len(tokens_gpu)
print(f"📚 Dataset: {DATA_LEN/1e6:.1f}M Chars on GPU")

# --- MODEL ALLOCATION ---
GB = 5
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024
PARAMS_B = (PHYSICAL_SIZE * 4) / 1e9

print(f"🧠 Model: {GB}GB ({PARAMS_B:.0f}B Params)")

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
    
    # MUTATE
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * 0.005)
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations, dtype=cp.int64)
    genome_trial[mut_indices] = cp.random.randint(-1, 2, size=n_mutations, dtype=cp.int8)
    
    # EVALUATE
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
    
    # Fractal Weights
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    predictions = cp.tanh(cp.dot(inputs, weights))
    
    diff = predictions - targets
    loss = float(cp.mean(diff ** 2))
    
    # SELECT
    if loss > 0.000001:
        if loss < best_loss:
            best_loss = loss
            cp.copyto(genome_best, genome_trial)
    
    if gen % 50 == 0:
        dt = time.time() - start
        hz = gen / dt
        tparams_s = (PHYSICAL_SIZE * 4 * hz) / 1e12
        print(f"🧬 Gen {gen} | {hz:.1f} Hz | Loss: {best_loss:.6f} | {tparams_s:.2f} T-Params/s | Dataset: {DATA_LEN/1e6:.0f}M")
```

## 2. Custom Dataset:
Per usare il tuo dataset:
1. Carica file su Kaggle (Add Data)
2. Set `USE_CUSTOM = True`
3. Aggiorna `DATASET_PATH`
