# 🚀 COME USARE LA GPU MEGLIO
## Da 600 TPS a 17,000+ TPS: Guida Pratica

---

## 📊 Risultati Ottenuti

| Configurazione | TPS | Speedup |
|----------------|-----|---------|
| PyTorch Standard | 600 | 1x |
| GORILLA batch 1×32 | 6,700 | **11x** |
| GORILLA batch 4×8 | 10,600 | **17x** |
| GORILLA batch 32×1 | 15,500 | **26x** |
| **GORILLA batch 48×1** | **16,500** | **27x** |

---

## 🔧 Tecniche di Ottimizzazione

### 1. Enable Hardware Acceleration
```python
torch.backends.cudnn.benchmark = True      # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmuls (2x speedup)
```

### 2. Use BFloat16 Instead of Float32
```python
model = model.cuda().bfloat16()  # Half memory, 2x speed, stable training
```

### 3. Fused Optimizer
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)  # CUDA-fused
```

### 4. Maximize Batch Size
- Aumenta `batch_size` fino al limite della VRAM
- Riduci `accum_steps` proporzionalmente
- **Esempio L4 (24GB):** batch_size=48, accum_steps=1

### 5. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint
x = checkpoint(block, x, use_reentrant=False)  # Trade compute for memory
```

### 6. Scaled Dot-Product Attention (Flash Attention)
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Auto Flash Attention
```

### 7. DataLoader Optimization
```python
DataLoader(dataset, batch_size=48, shuffle=True,
    num_workers=8,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    collate_fn=...)
```

### 8. Mixed Precision Autocast
```python
with torch.amp.autocast('cuda'):
    _, loss = model(batch, batch)
```

---

## 🦍 Architettura GORILLA/ZetaGrid

### EchoAttention (Shared Q/K)
```python
qk = self.c_qk(x)  # Single projection for Q and K
v = self.c_v(x)    # Separate V projection
y = F.scaled_dot_product_attention(qk, qk, v, is_causal=True)
```
**Vantaggio:** 33% meno parametri nell'attention, stessa qualità.

### MirrorFFN (Weight Tying)
```python
h = F.gelu(self.c_fc(x))
return F.linear(h, self.c_fc.weight.t()) * 0.9 + self.bias_out
```
**Vantaggio:** 50% meno parametri nel FFN, usa la trasposta del peso di input.

---

## 📈 Formula TPS

```
TPS = (seq_len × batch_size × accum_steps) / tempo_step
```

**Per massimizzare TPS:**
1. ↑ batch_size (fino a limite VRAM)
2. ↓ accum_steps (meno overhead sync)
3. ✓ Ottimizzazioni hardware attive

---

## ⚠️ Limiti VRAM per GPU

| GPU | VRAM | Batch Max Stimato |
|-----|------|-------------------|
| T4 | 16 GB | 32 |
| L4 | 24 GB | 48 |
| A40 | 48 GB | 96+ |
| A100 | 80 GB | 128+ |

---

## 🎯 Checklist Pre-Training

- [ ] `torch.backends.cudnn.benchmark = True`
- [ ] `torch.backends.cuda.matmul.allow_tf32 = True`
- [ ] Model in `.bfloat16()`
- [ ] `fused=True` nell'optimizer
- [ ] Batch size massimizzato
- [ ] `num_workers >= 4` nel DataLoader
- [ ] `pin_memory=True`
- [ ] Gradient checkpointing attivo

---

## 💡 Debugging Performance

```python
# Check GPU utilization
nvidia-smi -l 1

# Check VRAM usage
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

**Se GPU Util < 90%:** Aumenta batch size o num_workers.
**Se OOM:** Riduci batch size o attiva gradient checkpointing.

---

*Documentato durante ottimizzazione su RunPod L4 - 22 Gennaio 2026*

---

## 🏎️ Appendice: Prestazioni ZED-HLS — Analisi Investor-Safe (v1.1)

### 1. Risultati preliminari (Validazione HLS)

Abbiamo sintetizzato su **FPGA target Xilinx VU9P** un blocco specializzato per ZetaGrid ("ZED") ottimizzato per **dataflow, residency e batching leggero**. L’obiettivo non è competere come acceleratore general purpose, ma ridurre drasticamente overhead e latenza sui colli di bottiglia critici (packing/quant e memory paging).

### 2. KPI "Proof-Grade" (dati da report HLS: csynth + stima clock)

| Metrica                        |           Valore | Note                                                          |
| ------------------------------ | ---------------: | ------------------------------------------------------------- |
| **Clock target**               |  **364–423 MHz** | range tra soluzioni/config diverse                            |
| **K1 (Packer) throughput raw** | **≈46.7 Gbit/s** | 365M "words"/s × 128-bit datapath *(definizione in report)*   |
| **K1 footprint logico**        |     **1125 LUT** | footprint "small" rispetto a VU9P (lascia margine al compute) |
| **K3 (Paging ops rate)**       | **≈364 M ops/s** | primitive di gestione pagine memoria in HW                    |
| **Latenza on-chip (K-path)**   |   **~25–110 ns** | 10–40 cicli @ 364–423 MHz, pipeline streaming **II=1**        |

> Nota: i KPI sopra sono **on-chip** (datapath + scheduling interno). La latenza end-to-end host↔device dipende da PCIe/driver/OS e viene misurata separatamente nel pilot.

### 3. Latenza vs Throughput

Perché usare FPGA invece di una GPU H100?

> **Latenza:** il datapath **on-chip** opera a latenza sub-microsecondo; la latenza end-to-end host↔device è nell'ordine dei **microsecondi** e dipende da PCIe/driver/OS.

| Scenario                                        | GPU (H100/4090)    | ZED-HLS (FPGA)                 | Esito                                |
| ----------------------------------------------- | ------------------ | ------------------------------ | ------------------------------------ |
| **Training volumetrico (batch alto)**           | 🏆                 | Buono                          | GPU vince sul volume                 |
| **Critical-path offload batch=1 (pack/KV ops)** | overhead variabile | **µs-class** (target)          | FPGA vince sulla reattività          |
| **Efficienza**                                  | centinaia di watt  | **<50W** *(stima di progetto)* | potenziale vantaggio Watt/operazione |

### 4. Equivalenza Hardware (Il "Claim")

Non stiamo costruendo una GPU generica. Stiamo costruendo un **RTH Neural Engine** specializzato.

* **Apple M2 Ultra Neural Engine**: ~**31.6 TOPS (INT8)** (dato pubblico).
* **ZED-HLS Target**: **18 – 25 TOPS (INT8 equivalent)** *su operatori target* (packing/quant + memory primitives), in funzione di clock, parallelismo e utilizzo DSP.

**Conclusione**
ZED-HLS non è "una GPU in miniatura": è un **motore proprietario** che scarica la CPU dalle primitive più costose (packing/memory) e riduce drasticamente l'overhead per i task critici realtime dove "aspettare il batch" è un lusso.

**Disclosure / Measurement scope:** I KPI riportati derivano da HLS synthesis (csynth) e stime di clock; le misure end-to-end host↔device saranno incluse nell'Evidence Pack del pilot (XRT/PCIe).
