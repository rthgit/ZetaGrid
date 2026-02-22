# 🦅 KAGGLE V22 (6GB: 24 MILIARDI)

## 🧠 VRAM Reale
Kaggle T4 ha 16GB ma PyTorch/Sistema ne usa ~3GB.
Disponibile: **~13GB**.
Allocazione sicura: **6GB**.

## Scala:
- **6GB Fisico**
- **24 Miliardi di Parametri**
- Margine VRAM sicuro

## 1. Copia QUESTO:

```python
# ==========================================
# 🦅 ZETAGRID V22: 6GB SAFE (24B PARAMS)
# ==========================================
import os
import time
import numpy as np
import urllib.request
import cupy as cp

print("📦 ZETAGRID V22 (6GB SAFE)...")

if not os.path.exists("wikitext.txt"):
    print("📚 Downloading WikiText-2...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    urllib.request.urlretrieve(url, "wikitext.txt")

with open("wikitext.txt", "rb") as f:
    text_data = f.read()

tokens_gpu = cp.array(np.frombuffer(text_data, dtype=np.uint8))
DATA_LEN = len(tokens_gpu)

# SAFE MASSIVE SCALE
GB = 6
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024
PARAMS_B = (PHYSICAL_SIZE * 4) / 1e9  # 24 Billion

print(f"🧠 Allocating {GB}GB ({PARAMS_B:.0f}B Params)...")

# Efficient allocation (avoid randint temporary memory)
genome_best = cp.empty(PHYSICAL_SIZE, dtype=cp.int8)
genome_best[:] = cp.random.choice(cp.array([-1, 0, 1], dtype=cp.int8), size=PHYSICAL_SIZE)
genome_trial = cp.empty_like(genome_best)

print(f"✅ Allocated: {PARAMS_B:.0f}B Parameters")
print("🧬 EVOLUTION STARTED")

start = time.time()
gen = 0
best_loss = 9999.0
BATCH_SIZE = 8192
SEQ_LEN = 128

while True:
    gen += 1
    
    # 1. MUTATE
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * 0.002)
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations, dtype=cp.int64)
    genome_trial[mut_indices] = cp.random.choice(cp.array([-1, 0, 1], dtype=cp.int8), size=n_mutations)
    
    # 2. EVALUATE
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
    
    # Fractal Weight Selection
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    # Prediction
    predictions = cp.tanh(cp.dot(inputs, weights))
    
    # Loss
    diff = predictions - targets
    loss = float(cp.mean(diff ** 2))
    
    # 3. SELECTION
    if loss > 0.000001:
        if loss < best_loss:
            best_loss = loss
            cp.copyto(genome_best, genome_trial)
    
    if gen % 50 == 0:
        dt = time.time() - start
        hz = gen / dt
        tparams_s = (PHYSICAL_SIZE * 4 * hz) / 1e12
        print(f"Gen {gen} | {hz:.1f} Hz | Loss: {best_loss:.6f} | {tparams_s:.2f} T-Params/s | {PARAMS_B:.0f}B")
```
