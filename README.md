# ZetaGrid / RTH-LM

Reference repository for RTH-LM, a non-Transformer language-model research line based on the ZetaGrid Fractal Gated Causal TCN architecture.

This GitHub repository is intentionally kept lightweight. Large model artifacts are published on Hugging Face, while the technical paper is published on Figshare.

## Published Artifacts

| Artifact | Purpose | Location |
| --- | --- | --- |
| RTH-LM 25B | General language model release | https://huggingface.co/RthItalia/Rth-lm-25b |
| RTH-Code 25B | Code-specialist Soul release | https://huggingface.co/RthItalia/Rth-lm-code-25b |
| Technical paper | Architecture and feasibility report | https://doi.org/10.6084/m9.figshare.31376560 |
| Source repository | ZetaGrid scripts, docs, and reference tooling | https://github.com/rthgit/ZetaGrid |

## Repository Scope

This repository should contain:

- Architecture and inference source code.
- Quantization/conversion utilities.
- Technical documentation and model cards.
- Small configuration files and release notes.
- Public diagrams, logos, and lightweight metadata.

This repository should not contain:

- Full model weights (`.pt`, `.safetensors`, `.gguf`, `.qulp`, `.npy`).
- ONNX external tensor dumps or generated ONNX working directories.
- Local benchmark output, release zips, caches, or temporary deployment files.

Large artifacts belong on Hugging Face model repositories. Paper and presentation artifacts belong on Figshare or in small exported documentation form.

## Current Public Releases

### RTH-LM 25B

RTH-LM is an experimental Fractal Gated Causal TCN language model. It is designed to test whether a non-attention architecture can provide useful language-model behavior with lower inference memory pressure than Transformer-style systems.

Key published claims should be read as early research evidence, not as frontier-model parity:

- 7B physical Genome / 25B effective fractal capacity.
- Genome/Soul separation for reusable core weights and small trainable specialization artifacts.
- QULP 2-bit quantization path for low-memory experimentation.
- Initial training-loss evidence from constrained data and compute.

Model card: [ZETAGRID_25B_CARD.md](ZETAGRID_25B_CARD.md)

### RTH-Code 25B

RTH-Code 25B is a code-specialist Soul for the RTH-LM / ZetaGrid architecture. Its Hugging Face card is maintained separately so the GitHub README stays focused on the repository.

Code model card draft: [RTH_CODE_25B_CARD.md](RTH_CODE_25B_CARD.md)

### SwarmLM v2

SwarmLM v2 evaluates RTH-LM as a modular Genome/Soul system with centralized routing. A shared frozen Genome hosts multiple rank-512 specialist Souls, while `orchestrator_v2` routes requests to text, code, math, planning, or orchestration behavior.

Current controlled cascade result:

```text
Route accuracy: 0.875
Cascade success rate: 0.750
Specialist marker score average: 0.750
Peak eval VRAM per loaded Soul: 18.62 GB
```

The result supports centralized orchestration over specialist Souls. It does not claim robust general-assistant quality or universal self-routing by every Soul.

Targeted Orchestrator v3b routing update:

```text
Route accuracy: 1.000
Cascade success rate: 0.875
Specialist marker score average: 0.875
Previous code_prime routing failure: corrected
```

The v3b result keeps the same frozen Genome and v2 specialist Souls, while replacing the central routing checkpoint with `ORCHESTRATOR_V3B.pt`. The remaining observed cascade failure is marker-related on `text_fro`, not a routing failure.

Next controller direction:

```text
SwarmLM v4 = Orchestrator v3b + FRO-LM Controller
```

FRO-LM is planned as a lightweight resonance/control Soul for route confidence, ambiguity, safety risk, fallback, multi-Soul composition, and post-output validation. It is a controller layer, not a general assistant or autonomous tool executor.

The preferred FRO-LM path is now a small standalone byte-level controller trained from scratch with the FRO optimizer, rather than another full-size ZetaGrid Soul initialized from `orchestrator_v3b`.

Controller plan: [FRO_LM_CONTROLLER_V1_PLAN.md](FRO_LM_CONTROLLER_V1_PLAN.md)

Updated SwarmLM v2 model-card text: [HF_RTH_LM_SWARMLM_V2_CARD.md](HF_RTH_LM_SWARMLM_V2_CARD.md)

## Quick Start

The reference Python script expects model artifacts to be present locally. Download the required files from the Hugging Face model repositories before running inference.

```bash
python ZETAGRID_INFERENCE.py
```

The current script is a research/demo entry point, not a packaged Python library. Paths and hardware assumptions may need adjustment for your machine.

## Citation

```bibtex
@techreport{deluca2026rthlm,
  author      = {De Luca, Christian Quintino},
  title       = {RTH-LM: A Fractal Temporal Convolutional Language Model},
  institution = {RTH Italia (Research & Technology Hub)},
  year        = {2026},
  url         = {https://github.com/rthgit/ZetaGrid},
  doi         = {10.6084/m9.figshare.31376560},
  note        = {Non-commercial license. Contact RTH Italia for commercial use.}
}
```

## License

Model artifacts are released for research and non-commercial use under CC BY-NC 4.0 unless a separate commercial license is granted by RTH Italia.

See [LICENSE.md](LICENSE.md) for the current project licensing terms.
