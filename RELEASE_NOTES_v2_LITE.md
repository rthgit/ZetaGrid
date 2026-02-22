# ZetaGrid 25B v2 "Lite" (Release Notes)

This release contains the **"Lite"** (High-Efficiency) configuration of the ZetaGrid 25B architecture.
It is designed for minimal VRAM usage and maximum speed, using a compressed fractal structure.

## 📊 Specification Comparison

| Feature | **v1 (Original)** | **v2 (Lite - This Release)** | Impact |
| :--- | :--- | :--- | :--- |
| **Physical Parameters** | ~3.5B (BF16) | ~1.8B (BF16) | **50% Smaller Footprint** |
| **Fractal Capacity** | 25B Equivalent | 12B Equivalent | Lite is still powerful but less dense. |
| **FF Dim (Width)** | 16384 | **8192** | Faster Inference, Less RAM. |
| **LoRA Rank (Soul)** | 128 | **64** | Soul is extremely portable (~165MB). |
| **Soul Size** | ~933 MB | **~165 MB** | **5x Smaller Download**. |
| **Context Window** | 2048 | 2048 | Same. |

## 🛠️ Artifacts Included
1.  **Basic Repair (`zeta_25B_v2.pt` / `zeta_25B_v2_soul.pt`):**
    *   Trained on Golden Mix (Identity) + WikiText.
    *   Status: **Stable (Loss ~1.0)**.
    *   Use Case: General Conversational / RTH Persona.

2.  **Code Specialist (`zeta_code_step100.pt`):**
    *   Trained on Evol-Instruct-Code (80k).
    *   Status: **Early Checkpoint (Step 100)**.
    *   Use Case: Coding Assistant (Python/PyTorch Specialist).

## 🚀 How to Use
This model require the `ZetaGridV2` codebase with `D_FF=8192`.
It is NOT compatible with v1 checkpoints directly (due to dimension change).

### For OLLAMA (GGUF)
Use the provided `rth_lm_25b_v2_lite.gguf`. It will run on 8GB VRAM cards easily.

---
*Verified by RTH Italia Research Team*
