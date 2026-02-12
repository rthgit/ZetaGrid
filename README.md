# 🌌 ZED.AI (25B)
**The Fractal TCN Language Model**
*(Powered by ZetaGrid Architecture)*

[![License: Research](https://img.shields.io/badge/License-CC_BY_NC_4.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Commercial: Contact Us](https://img.shields.io/badge/Commercial-Contact_Us-gold.svg)](mailto:commercial@rth-italia.com)
[![Model Size: 25B](https://img.shields.io/badge/Model%20Size-25B-green.svg)](https://huggingface.co/rth-italia/zed-ai-25b)
[![Architecture: TCN](https://img.shields.io/badge/Architecture-Fractal_TCN-purple.svg)](https://arxiv.org/abs/1803.01271)
[![Sustainability: Green AI](https://img.shields.io/badge/Energy-Eco_Friendly-brightgreen.svg)](https://github.com/rth-italia/cpu-da)

---

## 🛑 Transformers are Dinosaurs. The Future is ZED.

**ZED.AI 25B** is not just another LLM. It is a statement.
It proves that you **don't need trillions of tokens** or millions of dollars in energy to build intelligence.

- **Built in 24 Hours.** We trained this 25B model from scratch in a single day.
- **Trained on <2GB Data.** We deliberately restricted the pre-training dataset to just 1.5GB of high-quality text. Why? To prove that the **Architecture is King**. The model learns structure, syntax, and reasoning not by brute-force memorization, but by efficient architectural design.
- **Green & Sustainable.** While Transformers burn forests of GPU energy, ZED.AI runs cool and efficient. It is the democratic, on-premise alternative.

---

## 🧬 The "Soul" Architecture (ZetaGrid)

ZED.AI abandons the quadratic complexity of Attention for the linear efficiency of **Gated Causal Temporal Convolutional Networks (TCNs)**.

### 1. The Frozen Genome (7GB)
The backbone of the model is a static, compressed "Genome" (`zetagrid_25b_production.npy`). This DNA encodes the fundamental patterns of language processing. It is shared across all models.

### 2. The Liquid Soul (LoRA)
We only train the "Soul" (Low-Rank Adapters). This means:
- **Instant Adaptation:** You can swap "Souls" in seconds. Load a `Coding Soul`, an `Art Soul`, or an `Enterprise Soul` without reloading the 25B weights.
- **Tiny Footprint:** A fully trained 25B model's unique weights are just **~300MB**.
- **Massive Scalability:** We can fractally expand this 25B seed to **50B, 100B, or 1T parameters** by duplicating layer blocks and letting them evolve.

---

## 🚀 The Vision: 1 Trillion Parameters on H100s

This 25B model is just the seed.
With a small cluster of H100 GPUs and a larger dataset, the **Fractal Architecture** allows us to scale to **1 Trillion Parameters** in a fraction of the time it takes to train a GPT-4 class model.

We are building a future where:
- **Training doesn't cost millions.**
- **Models run on local hardware.**
- **AI is modular, not monolithic.**

---

## ⚖️ License & Commercialization

**ZED.AI is Dual-Licensed:**

### 🎓 Research & Personal Use: **Free (CC BY-NC 4.0)**
You are free to use, modify, and experiment with ZED.AI for non-commercial purposes. We encourage researchers to explore the TCN architecture.

### 💼 Commercial Use: **Paid License Required**
If you want to use ZED.AI for commercial products, enterprise deployment, or generating revenue, you must obtain a commercial license from **RTH Italia**.
Our architecture offers unparalleled cost savings for inference at scale—invest in the future, not the past.

**[Contact Christian Quintino De Luca for Commercial Inquiries](mailto:info@rthitalia.com)**

---

## 🛠️ Usage

### 1. Download The Artifacts
Due to GitHub limits, download the large files from our HuggingFace or request access.
- **Genome:** `zetagrid_25b_production.npy` (7GB)
- **Weights:** `zeta25b_step15000.pt` (500MB)

### 2. Run Inference (Python)
```python
import torch
from ZETAGRID_INFERENCE import load_model, generate

# Load 25B Model
model = load_model("zeta25b_step15000.pt", genome="zetagrid_25b_production.npy")

# Generate
print(generate(model, "The future of AI is modular,", max_new=100))
```

---
*Created by **Christian Quintino De Luca** (RTH Italia) - Redefining AI Efficiency.*
