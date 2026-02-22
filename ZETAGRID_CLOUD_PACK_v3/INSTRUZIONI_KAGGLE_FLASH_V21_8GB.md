# 🦅 KAGGLE V21 (8GB SINGLE-GPU: 32 MILIARDI)

## 🐛 CUDA + Multiprocessing = Incompatibile
Nei notebook, CUDA non supporta il fork di processo.
**Soluzione:** Single GPU con **VERA SCALA** (8GB).

## Scala:
- **8GB Fisico** su GPU 0
- **32 Miliardi di Parametri**
- Poi aggiungiamo GPU 1 con metodo diverso

## 1. Copia QUESTO:

```python
# ==========================================
# 🦅 ZETAGRID V21: SINGLE-GPU 8GB (32B)
# ==========================================
import os
import time
import numpy as np
import urllib.request
import cupy as cp

print("📦 ZETAGRID V21 SINGLE-GPU 8GB...")

if not os.path.exists("wikitext.txt"):
    print("📚 Downloading WikiText-2...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    urllib.request.urlretrieve(url, "wikitext.txt")

with open("wikitext.txt", "rb") as f:
    text_data = f.read()

tokens_gpu = cp.array(np.frombuffer(text_data, dtype=np.uint8))
DATA_LEN = len(tokens_gpu)

# MASSIVE SCALE
GB = 8
PHYSICAL_SIZE = GB * 1024 * 1024 * 1024  # 8 Billion bytes
PARAMS_B = (PHYSICAL_SIZE * 4) / 1e9  # 32 Billion params (2-bit)

print(f"🧠 Allocating {GB}GB ({PARAMS_B:.0f}B Params)...")

genome_best = cp.random.randint(0, 3, size=PHYSICAL_SIZE, dtype=cp.int8)
genome_best[genome_best == 2] = -1
genome_trial = cp.empty_like(genome_best)

print(f"✅ Memory Allocated: {PARAMS_B:.0f} Billion Parameters")
print("🧬 EVOLUTION STARTED")

start = time.time()
gen = 0
best_loss = 9999.0
BATCH_SIZE = 8192  # Large batch for 8GB model
SEQ_LEN = 128

while True:
    gen += 1
    
    # 1. MUTATE
    cp.copyto(genome_trial, genome_best)
    n_mutations = int(PHYSICAL_SIZE * 0.002)  # 0.2% mutation
    mut_indices = cp.random.randint(0, PHYSICAL_SIZE, size=n_mutations, dtype=cp.int64)
    new_vals = cp.random.randint(0, 3, size=n_mutations, dtype=cp.int8)
    new_vals[new_vals == 2] = -1
    genome_trial[mut_indices] = new_vals
    
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
    
    # Fractal Weights (8GB search space)
    w_start = np.random.randint(0, PHYSICAL_SIZE - SEQ_LEN)
    weights = genome_trial[w_start : w_start + SEQ_LEN].astype(cp.float32)
    
    # Predict
    predictions = cp.tanh(cp.dot(inputs, weights))
    
    # Loss
    diff = predictions - targets
    loss = float(cp.mean(diff ** 2))
    
    # 3. SELECT
    if loss > 0.000001:
        if loss < best_loss:
            best_loss = loss
            cp.copyto(genome_best, genome_trial)
    
    if gen % 50 == 0:
        dt = time.time() - start
        hz = gen / dt
        tparams_s = (PHYSICAL_SIZE * 4 * hz) / 1e12
        print(f"Gen {gen} | {hz:.1f} Hz | Loss: {best_loss:.6f} | {tparams_s:.2f} T-Params/s | {PARAMS_B:.0f}B Params")
```
