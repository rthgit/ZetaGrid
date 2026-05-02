# 📄 Paper Release: RTH-LM (Figshare) & Architectural Deep-Dive

Hello Community! 🚀

We are excited to share the official technical paper for **RTH-LM**, published on Figshare. This marks the first major release of our **Fractal Gated Causal TCN** architecture.

Unlike traditional Transformers, RTH-LM eliminates the KV-cache pressure by using causal convolutions and a modular **Genome/Soul** design.

### What's inside this release:
1.  **Technical Paper:** Deep dive into Fractal Block expansion and 2-bit quantization feasibility.
2.  **25B Seed Weights:** The foundation for the RTH ecosystem.
3.  **Inference Script:** A lightweight engine to run the model on CUDA.

### We are looking for:
- 🧪 **Reviewers:** Technical feedback on the TCN backbone.
- 📊 **Benchmark Partners:** Help us test the limits of long-context recall.
- 🎨 **Adapter Devs:** Interested in training specialized "Souls".

Check the **Roadmap** in the PINNED discussion below for 120B/1T scaling updates!

---

# 🗺️ Roadmap: Scaling to 120B and 1T
We are currently simulating the VRAM models for:
- **120B Variant:** Goal is to run on a single H100 (80GB) using 2-bit weight-only quantization.
- **1T Variant:** Sharded inference across a compact 8-9x H100 cluster.

Stay tuned for the "Infinite Context" update!
