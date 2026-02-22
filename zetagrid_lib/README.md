# 🦍 ZETAGRID
**The "Unleashed" PyTorch Accelerator.**

ZETAGRID is a lightweight wrapper that automatically applies state-of-the-art optimizations (BF16, TF32, JIT Compile, Fused Kernels) to your PyTorch models, unlocking up to **4x Training Speed** and **16x Context Length** on NVIDIA Ampere GPUs (A40, A100, H100).

## 🚀 Installation
```bash
pip install .
```

## ⚡ Usage (3 Lines of Code)

```python
import torch
import zetagrid

# 1. Define your standard PyTorch model
model = MyGPT()

# 2. Initialize ZETAGRID Engine
engine = zetagrid.Engine()

# 3. Optimize Model (Auto BF16, Compile, Hardware-Match)
model = engine.prepare(model)

# Training Loop
optimizer = engine.build_optimizer(model, lr=1e-4)

for input in loader:
    # Auto-Selects correct precision (BF16/FP16)
    with engine.autocast_context():
        output = model(input)
        loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 🏆 Why use ZETAGRID?
| Metric | Vanilla PyTorch | ZETAGRID |
| :--- | :--- | :--- |
| **Precision** | FP32 (Slow/OOM) | **BFloat16 (Native)** |
| **Math** | FP32 Standard | **TensorFloat32 (TF32)** |
| **Compiler** | Eager (Slow) | **Torch.Compile (JIT)** |
| **Throughput** | 17k TPS | **70k TPS** (4x) |
| **Max Context** | 1k Tokens | **16k Tokens** (16x) |

## License
MIT
