# 🛠️ ZetaGrid-Ollama Patch: Setup Guide

To run **RTH-LM** (Fractal TCN) natively in your local environment via Ollama, you need to add support for the TCN operators to the underlying `llama.cpp` engine.

### 📦 Prerequisites
1.  **GGUF Model:** Download [rth_lm_25b_v1.gguf](https://huggingface.co/RthItalia/Rth-lm-25b)
2.  **Ollama Source:** Clone the official repository or use our fork.

### 🛠️ Step 1: Add Custom Kernels
Copy the provided C++ files into the `llama.cpp` source tree:
- Move `rth_tcn_ops.cpp` and `rth_tcn_ops.h` to `llama.cpp/src/`
- Register `GGML_OP_CAUSAL_CONV1D` and `GGML_OP_FRACTAL_GATE` in `ggml.c`

### 🏗️ Step 2: Compile
Rebuild `llama.cpp` or your Ollama binary:
```bash
make -j
# or for Ollama
go generate ./...
go build .
```

### 🚀 Step 3: Create & Run
Use the provided `Modelfile_RTH-LM` to register the model:
```bash
ollama create rth-lm -f Modelfile_RTH-LM
ollama run rth-lm
```

---
**Why this matters:** RTH-LM is a pioneer in non-Transformer architectures for local inference. By applying this patch, you are at the forefront of the TCN revolution. 🫡🌌🦙
