# Scaling Analysis: RTH-LM 120B & Hardware Requirements

**Objective:** Validate feasibility of running a 120B parameter RTH-LM model on consumer/enterprise hardware.

## 1. 2-Bit Quantization Breakdown

For a 120 Billion parameter model quantized to 2-bit (e.g., QuIP/QuIP# PTQ):

*   **Raw Weight Size:** $120 \times 10^9 \times 2 \text{ bits} / 8 = 30.0 \text{ GB}$
*   **Effective Size (Implementation Dependent):** Including scales, metadata, and group packing, the realistic footprint is **~30–36 GB**.

## 2. Kernel/State Memory (KV-Equivalent)

*Note: As a Fractal TCN, RTH-LM does not use a traditional Attention KV Cache. However, for the purpose of establishing a robust upper bound, we assume a memory footprint equivalent to a Gated Linear Attention or standard Transformer KV cache.*

**Upper Bound Estimation (Standard KV assumptions):**
*   **Per Token:** ~72 KB (Upper bound for efficient state management)
*   **8k Context:** ~0.6 GB
*   **32k Context:** ~2.4 GB
*   **128k Context:** ~9.5 GB

## 3. Total VRAM Requirements (Single Stream, Batch=1)

| Component | Size (GB) |
| :--- | :--- |
| **Model Weights (2-bit)** | 30.0 - 36.0 |
| **State/KV Cache (128k)** | ~9.5 (Upper Bound) |
| **Runtime Overhead** | 6.0 - 12.0 (Workspace/Allocation) |
| **Total VRAM** | **~45.5 - 57.5 GB** |

## 4. Conclusion: Hardware Feasibility (120B)

Even with conservative estimates for overhead and a massive 128k context window, the total memory requirement ($\approx 57.5$ GB) fits comfortably within the **80GB** envelope of a **Single NVIDIA H100 or A100**. This leaves **>20GB** of headroom for throughput and batching.

---

## 5. The 1T-Parameter Vision: Cluster Analysis

Scaling to **1 Trillion Parameters** (1T) is the next frontier for the RTH-LM fractal architecture.

### 5.1 Weight Analysis (2-bit)
*   **Raw Weights (2-bit):** $1 \times 10^{12} \times 2 / 8 = 250 \text{ GB}$
*   **Realistic Footprint:** **~250–300 GB** VRAM across GPUs.

### 5.2 Deployment Strategy (8-9× H100 Cluster)
While 4×80GB (320GB Total) is the theoretical minimum to hold the weights, production-grade inference requires additional overhead for long-context states and throughput optimization.

*   **Baseline (6× H100 80GB):** Sufficient for weights + overhead + context-state.
*   **Production Cluster (8-9× H100 80GB):** Recommended for:
    1.  **High Throughput:** Sharded inference for faster decoding.
    2.  **Redundancy:** Hot-swapping replicas.
    3.  **Multi-Tenant Souls:** Holding multiple "Soul" adapters in memory simultaneously.

**Verdict:** The 1T Vision is achievable with a single, compact rack of 8x GPUs, making extreme-scale AI accessible to private enterprise.

