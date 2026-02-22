---
license: other
license_name: cc-by-nc-4.0-commercial
license_link: LICENSE.md
language:
- en
metrics:
- perplexity
- loss
library_name: generic
pipeline_tag: text-generation
tags:
- tcn
- fractal
- rth-lm
- non-transformer
- cpu-da
---

# 🌌 RTH-LM (25B) — Unified V2 Release
![RTH Logo](rth_logo.png)

**The Fractal TCN Language Model**
*(Powered by ZetaGrid Architecture)*

[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.31376560-blue.svg)](https://doi.org/10.6084/m9.figshare.31376560)
[![License: Research](https://img.shields.io/badge/License-CC_BY_NC_4.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Model Size: 25B](https://img.shields.io/badge/Model%20Size-25B-green.svg)](https://huggingface.co/rth-italia/rth-lm-25b)
[![Architecture: TCN](https://img.shields.io/badge/Architecture-Fractal_TCN-purple.svg)](https://doi.org/10.6084/m9.figshare.31376560)

---

## 🛑 Beyond Transformers. The Fractal Future.

**RTH-LM 25B** is a breakthrough in efficiency. It proves that you can build intelligence without the quadratic overhead of Attention, using a **Gated Causal Temporal Convolutional Network (TCN)**.

- **128 Fractal Layers:** The V2 release expands the physical 32-layer seed into a 128-layer "Fractal" model, reaching 25B parameter capacity.
- **2-bit Quantized:** Designed to be resilient to extreme compression. The 50GB model fits into **~6.7GB** using QULP 2-bit quantization.
- **Green & Sustainable:** Optimized for local hardware and CPUs.

---

## 🧬 Architecture (ZetaGrid)

RTH-LM abandons the quadratic complexity of Attention for the linear efficiency of **TCNs**.

### 1. The Unified Model (v2)
In the V2 release, the "Genome" (base intelligence) and "Soul" (learned adapters) are merged. You no longer need separate files—just the sharded safetensors or a single GGUF.

### 2. Fractal Scaling
We scale the 7B seed to 25B by "tiling" the genome across 128 layers. This creates a massive memory-efficient model that captures long-range syntax with constant memory per step.

---

## 🚀 Usage

### ⚙️ Option 1: Native Python (Best for Research)
Clone the repo and run the optimized v2 inference script (supports sharded safetensors).

```bash
git clone https://github.com/rthgit/ZetaGrid
cd ZetaGrid
python RTH_LM_INFERENCE_v2.py --model "/path/to/rth_lm_25b_v4_sharded"
```

### 📦 Option 2: GGUF & Ollama (Best for Users)
The model is available in GGUF format for use with `llama.cpp` and `Ollama`.

```bash
# Generate Modelfile
ollama create rth-lm -f Modelfile_RTH-LM
ollama run rth-lm "Tell me a story about a digital soul"
```

---

## ⚖️ License & Commercialization

**RTH-LM is Dual-Licensed:**

### 🎓 Research & Personal Use: **Free (CC BY-NC 4.0)**
Free to use and modify for non-commercial research.

### 💼 Commercial Use: **Paid License Required**
Contact **RTH Italia** for enterprise deployment and commercial pipelines.

**[Contact Christian Quintino De Luca](mailto:info@rthitalia.com)**

---
*Created by **Christian Quintino De Luca** (RTH Italia) - Redefining AI Efficiency.*
