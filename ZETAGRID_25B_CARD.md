---
language: en
license: cc-by-nc-4.0
tags:
- zetagrid
- cpu-da
- tcn
- fractal
- 25b
datasets:
- custom
metrics:
- loss
---

# 📇 Model Card: RTH-LM (25B)

## Model Details
- **Name:** RTH-LM (25B)
- **Architecture:** Fractal Gated Causal TCN (Temporal Convolutional Network)
- **Parameters:** 7B (Physical) / 25B (Effective Fractal Capacity)
- **Author:** Christian Quintino De Luca (RTH Italia)
- **Release Date:** February 2026
- **License:** CC BY-NC 4.0 (Research) / Commercial (Enterprise)
- **Paper (Figshare):** https://doi.org/10.6084/m9.figshare.31376560

RTH-LM (25B) is a **Fractal TCN (Temporal Convolutional Network)** Language Model, designed for high-efficiency inference on CPU/Consumer Hardware and massive scalability on GPUs.

Unlike Traditional Transformers, ZetaGrid uses a **Gated Causal TCN backbone** with **Fractal Scaling**, allowing it to model long-range dependencies with significantly lower memory overhead during inference.

---

## 📊  Model Specs

| Feature | Specification |
| :--- | :--- |
| **Parameters** | 25 Billion (25B) |
| **Architecture** | Fractal Gated TCN (Non-Transformer) |
| **Layers** | 32 (Phase 2) |
| **Context Window** | 256 - 1024 (Fractal Expansion Capable) |
| **Training Data** | 1.48 GB Cleaned Text (Wiki/Books) |
| **Final Loss** | **1.0675** (Phase 2) |
| **Quantization** | QULP 2-bit (Supported) |

---

## 🚀 Usage (Inference)

### Prerequisites
Use the ZetaGrid reference repository and download the model artifacts from this Hugging Face repository.

```bash
# Clone the repo
git clone https://github.com/rthgit/ZetaGrid
cd ZetaGrid
```

### Running the Model (Python)
Place the required artifacts in the working directory, or update the paths in the script:

- `zeta25b_step15000.pt` - Soul/checkpoint weights
- `zetagrid_25b_production.npy` - Genome weight bank

```python
import torch
from ZETAGRID_INFERENCE import load_model, generate

# Load 25B Model
model = load_model("zeta25b_step15000.pt", genome="zetagrid_25b_production.npy")

# Generate
text = generate(model, "The future of AI is")
print(text)
```

### QULP 2-bit Inference (Ultra-Low Memory)
If using the QULP artifact, download `zeta25b_2bit.qulp` from the model repository and run the matching local inference script when available:

```bash
python QULP_INFERENCE.py --model zeta25b_2bit.qulp
```

---

## 🧬 Architecture: The "Fractal Soul"

ZetaGrid is **NOT** a Transformer. It is a TCN-based organism.
- **Genome:** A fixed 7GB "DNA" bank of weights (`zetagrid_25b_production.npy`).
- **Phenotype:** The model layers are "grown" from this genome on the fly.
- **Training:** Only the "Soul" (LoRA Adapters + Norms) is trained (~300MB), making the model extremely portable.
- **Fractal Scaling:** The 25B model can be fractally expanded to 50B, 100B+ by duplicating layers and adding self-linear noise.

---

## 📈 Performance

- **Phase 1 (Evolution):** 200 Generations of Genome Optimization.
- **Phase 2 (Gradient):** 15,000 Steps of TCN+LoRA Fine-Tuning.
- **Convergence:** Beat target loss of 1.5, achieving **1.0675**.
- **Capabilities:** Narrative coherence, English syntax mastery, abstract reasoning.

---

## 📜 License
CC BY-NC 4.0 (Creative Commons Non-Commercial) for Research.
**Commercial Use:** Requires a license from **RTH Italia**.
For inquiries: info@rthitalia.com
