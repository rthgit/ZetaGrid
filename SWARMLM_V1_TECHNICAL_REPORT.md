# SwarmLM v1 Technical Report

## Abstract

SwarmLM v1 evaluates the RTH-LM Genome/Soul architecture as a modular system rather than a monolithic assistant. The experiment uses one shared frozen Genome and multiple high-rank trainable Souls to test whether specialized behavior can be induced, measured, and routed. Six rank-512 Souls were trained or aligned on a single A40 GPU with Fractal Resonant Optimization (FRO): text, code, math, instruction, agentic planning, and orchestration.

The core result is that a common frozen Genome can host behaviorally distinct Souls. Target-domain marker score was 0.848 versus 0.218 off-target, while the Orchestrator Soul reached 1.000 route accuracy on controlled route-format prompts. The experiment includes checkpoint hashes, Genome hash, runtime telemetry, raw JSONL generations, and an interpreted evaluation report.

## Research Claim

RTH-LM demonstrates a modular Genome/Soul architecture in which a shared frozen Genome supports multiple behaviorally distinct high-rank Souls. SwarmLM extends this into an orchestration layer that routes tasks among specialized Souls.

This report does not claim that RTH-LM is already a competitive general assistant, nor that SwarmLM composes multiple Souls into a single end-to-end answer. The current result supports controlled modular specialization and controlled routing over a common frozen substrate.

## Architecture

The system separates long-lived frozen structure from task-specific trainable behavior:

| Component | Role |
| --- | --- |
| Genome | Shared frozen parameter substrate loaded from `zetagrid_25b_production.npy`. |
| Soul | High-rank trainable specialization checkpoint. |
| FRO | Optimizer used to train or align Souls while logging resonance telemetry. |
| SwarmLM | Experimental orchestration layer for selecting specialized Souls. |

The experimental Genome was held fixed across all runs:

```text
Genome path: /workspace/zetagrid_50b/zetagrid_25b_production.npy
Genome SHA256: 09dcebf875ec9f9a3b8f1da17536b42f09bc50ec7334afb6426a1dd41f1762e5
Loaded shape/dtype: (6979321856,) int8
```

Only Soul parameters were trained:

```text
LoRA matrices
normalization weights
scale parameters
token embedding
positional embedding
final normalization
```

## Training Setup

All six v1 Souls used the same architecture and rank:

```text
rank: 512
trainable parameters: ~949.1M
checkpoint size: ~3.6 GB each
dtype: bfloat16 on CUDA
GPU: NVIDIA A40
```

Common training configuration:

```text
steps: 500
sequence length: 384
batch size: 1
gradient accumulation: 4
learning rate: 2e-6
save policy: overwrite latest.pt, then preserve named checkpoint
```

Optimizer:

```text
optimizer: Fractal Resonant Optimization (FRO)
fro_alpha: 0.25
fro_gamma: 0.6 for text/code/instruction/agentic/orchestrator
fro_gamma: 0.7 for math
scales: (0.1, 0.01, 0.001)
betas: (0.9, 0.98)
```

FRO telemetry remained stable across all runs, with positive resonance/coherence signals on controlled alignment datasets.

## Soul Inventory

| Soul | Checkpoint | Step | Loss | SHA prefix |
| --- | --- | ---: | ---: | --- |
| Text Align | `TEXT_V2_ALIGN.pt` | 500 | 0.0972349741 | `86433832d064` |
| Code Align | `CODE_V2_ALIGN.pt` | 500 | 0.0393827808 | `6ce2a778033a` |
| Math Align | `MATH_V1_ALIGN.pt` | 500 | 0.0482873699 | `266f979b7d92` |
| Instruction | `INSTRUCTION_V1_SMOKE.pt` | 500 | 0.0324842194 | `291d054b47c5` |
| Agentic | `AGENTIC_V1_SMOKE.pt` | 500 | 0.0473323831 | `b04cbce03110` |
| Orchestrator | `ORCHESTRATOR_V1_SMOKE.pt` | 500 | 0.0418650329 | `2cbb48a49066` |

## Evaluation Protocol

The scientific smoke suite is implemented in:

```text
EVAL_SWARMLM_SUITE.py
```

It evaluates:

- same Genome / different Soul behavior
- target prompts for each Soul
- off-target prompts for each Soul
- route-format prompts for SwarmLM
- runtime telemetry
- checkpoint and Genome provenance

Evaluation settings:

```text
max_new: 120
temperature: 0.25
top_k: 10
hash_files: enabled
```

Generated artifacts:

```text
/workspace/zetagrid_50b/reports/swarmlm_v1_suite_hashed/eval_swarmlm_v1_suite.jsonl
/workspace/zetagrid_50b/reports/swarmlm_v1_suite_hashed/manifest.json
/workspace/zetagrid_50b/reports/swarmlm_v1_suite_hashed/SWARMLM_V1_EVAL_REPORT.md
/workspace/zetagrid_50b/reports/swarmlm_v1_suite_hashed/SWARMLM_V1_EVAL_REPORT_INTERPRETED.md
```

## Results

Summary metrics:

```text
Generation rows: 66
Target-only marker score average: 0.848
Off-target marker score average: 0.218
Global route accuracy: 0.167
Orchestrator-only route accuracy: 1.000
Non-orchestrator route accuracy: 0.000
Average generation speed: 16.43 tokens/sec
Peak eval VRAM: 18.61 GB
```

The target marker score is substantially higher than the off-target score. This supports behavioral differentiation of Souls over the same frozen Genome.

The global route accuracy is intentionally low because only one Soul was trained for routing. The relevant decomposition is:

```text
orchestrator_v1 route accuracy: 1.000
non-orchestrator route accuracy: 0.000
```

This supports the functional specialization claim: routing is localized in `orchestrator_v1`, while other Souls specialize in text, code, math, instruction following, or planning.

## Qualitative Findings

Text Align:

```text
Responds in instruction/text format and can explain the modular Genome/Soul concept in Italian.
```

Code Align:

```text
Generates Python code on code-format prompts, including Fibonacci and primality function structure.
```

Math Align:

```text
Solves controlled algebra and arithmetic prompts in Problem/Solution format.
```

Instruction:

```text
Learns answer-format behavior such as RESULT/WHY and concise bullet responses.
```

Agentic:

```text
Produces procedural plans for evaluation, dataset preparation, and demo workflows.
```

Orchestrator:

```text
Produces route decisions in ROUTE/REASON format.
```

Observed controlled route behavior:

```text
route_code    -> ROUTE: code_v2
route_math    -> ROUTE: math_v1
route_agentic -> ROUTE: agentic_v1
route_complex -> ROUTE: orchestrator_v1
```

## Limitations

The experiment should not be overclaimed.

Current limitations:

- The system is not yet a robust general assistant.
- SwarmLM v1 does not yet compose multiple Souls into a single synthesized final answer.
- The alignment datasets are small and controlled, so template overfitting is expected.
- Off-domain prompts often cause a specialized Soul to continue its own learned behavior.
- FRO has not yet been compared against AdamW or Adafactor in a controlled ablation.

These limitations are useful because they clarify the next experimental steps. In particular, off-domain behavior motivates orchestration rather than direct use of every Soul on every prompt.

## Interpretation

The result supports three separable conclusions:

1. Domain specialization is possible over a shared frozen Genome.
2. Functional specialization is possible over the same shared frozen Genome.
3. A dedicated Orchestrator Soul can route controlled requests toward specialized Souls.

The experiment therefore demonstrates a modular system:

```text
shared Genome
specialized Souls
functional routing
telemetry
hashes
evaluation
reproducible reports
```

## Recommended Next Work

1. Upload the full hashed evaluation report to Hugging Face.
2. Upload the six new v1 Soul checkpoints.
3. Update model cards with the SwarmLM v1 result and limitations.
4. Build `align_v2` with broader prompt variation and explicit off-domain behavior.
5. Evaluate v1 versus v2 with the same suite.
6. Add a small FRO versus AdamW/Adafactor ablation.
7. Only after consolidation, revisit 60B/70B scaling.

## Suggested SPRIND Wording

RTH-LM is not evaluated here as a monolithic model. It is evaluated as a modular cognitive substrate: a shared frozen Genome supporting specialized, high-rank Souls. SwarmLM demonstrates controlled routing among these Souls, providing an initial experimental basis for coordinated modular specialization.
