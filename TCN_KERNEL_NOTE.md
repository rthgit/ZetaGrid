# 🧪 TCN Kernel Specification for llama.cpp

Per rendere RTH-LM compatibile con **Ollama**, dobbiamo integrare questi due kernel nel backend **ggml**.

### 1. Causal Conv1D (Prototipo)
La convoluzione causale è il motore del TCN. A differenza delle convoluzioni standard, questa "guarda" solo ai token passati.
- **Dilatazione:** Il parametro cruciale che permette al TCN di avere una memoria a lunghissimo raggio con pochi layer.
- **Complessità Inference:** $O(N)$ lineare, perfetta per sequenze lunghissime dove il Transformer (Quadratico) rallenta.

### 2. Fractal Fusion Logic
Implementiamo l'operatore di gating che fonde i due rami del grafo frattale:
$$Y = \text{SiLU}(A_{\text{mixed}}) \otimes \sigma(G_{\text{gate}})$$
Questo assicura che il modello possa "dimenticare" o "enfatizzare" specifiche informazioni temporali durante la generazione.

### 🛡️ Stato della Missione
- [x] Header C++ definiti.
- [x] Logica reference (CPU) scritta in `rth_tcn_ops.cpp`.
- [ ] Implementazione SIMD (AVX/NEON) per velocità desktop.
- [ ] Implementazione CUDA per velocità server.

---
**ZetaGrid Engineering Team**  
*Building the future of local, efficient LLM inference.* 🫡🌌🦙⚔️🏗️
