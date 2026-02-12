# 🌌 ZetaGrid
**The Fractal TCN Language Model**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Model Size: 25B](https://img.shields.io/badge/Model%20Size-25B-green.svg)](https://huggingface.co/rth-italia/zetagrid-25b)
[![Architecture: TCN](https://img.shields.io/badge/Architecture-Fractal%20TCN-purple.svg)](https://arxiv.org/abs/1803.01271)
[![Inference: CPU](https://img.shields.io/badge/Inference-CPU%20Friendly-orange.svg)](https://github.com/rth-italia/cpu-da)

**ZetaGrid is not a Transformer.**
It is an organism grown from a static genome, designed for infinite scalability and extreme inference efficiency on consumer hardware.

---

## 🧬 The "Soul" Architecture
ZetaGrid abandons the quadratic complexity of Attention for the linear efficiency of **Gated Causal Temporal Convolutional Networks (TCNs)**.

- **Non-Transformer Backbone:** Uses dilated convolutions to model long-range dependencies.
- **Fractal Scaling:** The 25B model is a "seed" that can be fractally expanded to 50B, 100B, and beyond by duplicating layer blocks and adding self-linear noise.
- **Genome-Based:** The model weights are generated on the fly from a compressed "Genome" file (`zetagrid_25b_production.npy`), reducing the trainable parameter footprint to just ~300MB.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/rthgit/ZetaGrid.git
cd ZetaGrid
pip install torch numpy
```

### 2. Download Weights
Due to GitHub limits, you must download the large model files separately:
- **`zetagrid_25b_production.npy`** (The Genome - 7GB)
- **`zeta25b_step15000.pt`** (The Checkpoint - 500MB)

*(Link to Weights TBA - Check Releases)*

### 3. Run Inference (Python)
```python
import torch
from ZETAGRID_INFERENCE import load_model, generate

# Load the Fractal Model
model = load_model("zeta25b_step15000.pt", genome="zetagrid_25b_production.npy")

# Generate Text
prompt = "The future of Artificial Intelligence is"
print(generate(model, prompt, max_new=100))
```

---

## ⚡ Quantization (QULP)
Run the 25B model on a laptop with <2GB RAM using our **QULP 2-bit Quantization**.

```bash
# Quantize the model (requires full weights)
python QULP_2BIT_QUANTIZER.py

# Run Inference
python QULP_INFERENCE.py --model zeta25b_2bit.qulp
```

---

## 📈 Performance (Phase 2)
- **Loss:** 1.0675 (Validation)
- **Context:** 1024 Tokens (Expandable)
- **Training:** Trained on 1.5GB of high-quality scientific and narrative text.

---

## 🛡️ License
MIT License.
Created by **RTH Italia** (Cpu-DA Project).
