# RTH-LM 25B / SwarmLM v2 Model Card

## Overview

RTH-LM is an experimental non-Transformer language-model research line based on the ZetaGrid Fractal Gated Causal TCN architecture.

The current public research direction evaluates RTH-LM as a modular Genome/Soul system:

```text
Genome = shared frozen substrate
Soul = trainable specialization module
SwarmLM = orchestration layer for routing among Souls
```

SwarmLM v2 extends the previous v1 experiment with stronger v2 Souls and an end-to-end cascade evaluation:

```text
user request
-> orchestrator_v2 selects route
-> selected specialist Soul generates output
```

This card should be read as a research artifact, not as a claim of frontier general-assistant quality.

## Core Research Claim

RTH-LM demonstrates a modular Genome/Soul architecture in which a shared frozen Genome supports multiple behaviorally distinct high-rank Souls. SwarmLM v2 shows that centralized orchestration over this shared Genome can route user requests to specialized Souls with measurable end-to-end cascade success.

Recommended claim language:

> SwarmLM v2 demonstrates that centralized orchestration over a shared frozen Genome can route user requests to specialized Souls with 87.5% route accuracy and 75% end-to-end cascade success in a controlled scientific smoke evaluation.

## Shared Genome

All Souls in this release use the same frozen Genome:

```text
Genome file: zetagrid_25b_production.npy
SHA256: 09dcebf875ec9f9a3b8f1da17536b42f09bc50ec7334afb6426a1dd41f1762e5
Loaded shape/dtype: (6979321856,) int8
```

The Genome remains frozen during Soul training and evaluation. Only Soul-side trainable components are updated.

## Architecture

| Component | Role |
| --- | --- |
| Genome | Shared frozen parameter substrate. |
| Soul | Rank-512 trainable specialization checkpoint. |
| FRO | Fractal Resonant Optimization, used for Soul training with resonance telemetry. |
| SwarmLM | Experimental orchestration system for selecting specialist Souls. |

Each v2 Soul uses rank 512 and approximately 949.1M trainable parameters.

## SwarmLM v2 Soul Inventory

Main RTH-LM repository:

| Soul | File | Role |
| --- | --- | --- |
| Text Align v2 | `souls/text_align_v2/TEXT_ALIGN_V2.pt` | Natural-language explanation and text behavior |
| Instruction v2 | `souls/instruction_v2/INSTRUCTION_V2.pt` | Instruction-format behavior |
| Agentic v2 | `souls/agentic_v2/AGENTIC_V2.pt` | Step-by-step planning behavior |
| Orchestrator v2 | `souls/orchestrator_v2/ORCHESTRATOR_V2.pt` | Centralized route selection |

Companion specialist repositories:

| Repository | Soul | File | Role |
| --- | --- | --- | --- |
| `RthItalia/Rth-lm-code-25b` | Code Align v2 | `souls/code_align_v2/CODE_ALIGN_V2.pt` | Python/code-format generation |
| `RthItalia/Rth-lm-math-25b` | Math Align v2 | `souls/math_align_v2/MATH_ALIGN_V2.pt` | Math Problem/Solution behavior |

## Training Summary

All SwarmLM v2 Souls were trained on an A40-class GPU with bfloat16 execution and the shared frozen Genome.

Common configuration:

```text
rank: 512
layers: 32
sequence length: 384
batch size: 1
gradient accumulation: 4
steps: 1000
learning rate: 1.5e-6
optimizer: Fractal Resonant Optimization (FRO)
```

Final v2 checkpoint losses:

| Soul | Step | Best loss |
| --- | ---: | ---: |
| `text_align_v2` | 1000 | 0.0424 |
| `code_align_v2` | 1000 | 0.0354 |
| `math_align_v2` | 1000 | 0.0470 |
| `instruction_v2` | 1000 | 0.0507 |
| `agentic_v2` | 1000 | 0.0398 |
| `orchestrator_v2` | 1000 | 0.0375 |

## Evaluation Artifacts

SwarmLM v2 self-routing/specialization suite:

```text
reports/swarmlm_v2_suite_hashed/eval_swarmlm_v2_suite.jsonl
reports/swarmlm_v2_suite_hashed/manifest.json
reports/swarmlm_v2_suite_hashed/SWARMLM_V2_EVAL_REPORT.md
reports/swarmlm_v2_suite_hashed/SWARMLM_V2_EVAL_REPORT_INTERPRETED.md
```

SwarmLM v2 cascade suite:

```text
reports/swarmlm_v2_cascade_hashed/eval_swarmlm_v2_cascade.jsonl
reports/swarmlm_v2_cascade_hashed/manifest.json
reports/swarmlm_v2_cascade_hashed/SWARMLM_V2_CASCADE_REPORT.md
```

Interpretation note:

```text
reports/SWARMLM_V2_INTERPRETATION.md
```

## SwarmLM v2 Specialization Results

The v2 specialization suite tests six v2 Souls over the same frozen Genome.

```text
Generation rows: 72
Target-only marker score average: 0.889
Off-target marker score average: 0.297
Off-target leakage score average: 0.201
Global route accuracy: 0.167
Orchestrator-only route accuracy: 1.000
Non-orchestrator route accuracy: 0.000
ROUTE_REQUESTED accuracy: 0.000
Average tokens/sec: 16.30
Peak eval VRAM: 18.62 GB
```

Interpretation:

- Target specialization improved over v1: `0.848 -> 0.889`.
- Orchestrator route accuracy remained perfect on controlled route prompts: `1.000`.
- Non-orchestrator Souls did not reliably self-delegate outside their domain.
- This supports centralized routing rather than universal self-routing.

## SwarmLM v2 Cascade Results

The cascade suite evaluates the intended SwarmLM architecture:

```text
orchestrator_v2 -> selected specialist Soul -> output
```

Summary:

```text
Tasks: 8
Route accuracy: 0.875
Specialist marker score average: 0.750
Cascade success rate: 0.750
Average cascade latency: 43.46s
Average route tokens/sec: 17.62
Average specialist tokens/sec: 15.70
Peak route VRAM: 18.60 GB
Peak specialist VRAM: 18.62 GB
```

Task-level results:

| Task | Expected route | Got | Specialist | Marker | Success |
| --- | --- | --- | --- | ---: | --- |
| `text_genome_soul` | `text_v2` | `text_v2` | `text_align_v2` | 1.000 | yes |
| `text_fro` | `text_v2` | `text_v2` | `text_align_v2` | 0.000 | no |
| `code_fibonacci` | `code_v2` | `code_v2` | `code_align_v2` | 1.000 | yes |
| `code_prime` | `code_v2` | `text_v2` | `text_align_v2` | 0.000 | no |
| `math_linear` | `math_v1` | `math_v1` | `math_align_v2` | 1.000 | yes |
| `math_speed` | `math_v1` | `math_v1` | `math_align_v2` | 1.000 | yes |
| `agentic_eval_plan` | `agentic_v1` | `agentic_v1` | `agentic_v2` | 1.000 | yes |
| `complex_multisoul` | `orchestrator_v1` | `orchestrator_v1` | `orchestrator_v2` | 1.000 | yes |

Interpretation:

SwarmLM v2 cascade works materially better than asking every specialist Soul to self-route.

```text
Self-delegation ROUTE_REQUESTED accuracy: 0.000
Cascade route accuracy: 0.875
Cascade success rate: 0.750
```

This supports the architecture:

```text
central route -> specialist execution
```

## Known Failure Modes

Current observed limitations:

- `text_fro` routed to `text_v2`, but the specialist answer did not match the FRO-specific target markers.
- `code_prime` was misrouted to `text_v2` instead of `code_v2`.
- Non-orchestrator Souls are specialized executors, not reliable self-routing agents.
- The system is not yet a robust general assistant.
- The cascade currently reloads one model at a time on A40, so latency includes model switching.

These failures motivate targeted `orchestrator_v3` work, especially for code classification and FRO/text distinction.

## Orchestrator v3b Routing Update

`orchestrator_v3b` is a targeted routing update trained after the v2 cascade evaluation. It keeps the same frozen Genome and the same v2 specialist Souls, but replaces the central routing checkpoint:

```text
orchestrator_v3b -> selected v2 specialist Soul -> output
```

Artifact path:

```text
souls/orchestrator_v3b/ORCHESTRATOR_V3B.pt
```

Controlled cascade result:

```text
Tasks: 8
Route accuracy: 1.000
Specialist marker score average: 0.875
Cascade success rate: 0.875
Average cascade latency: 48.36s
Average route tokens/sec: 17.72
Average specialist tokens/sec: 15.72
Peak route VRAM: 18.60 GB
Peak specialist VRAM: 18.62 GB
```

Task-level comparison:

| Task | Expected route | v2 route | v3b route | v3b success |
| --- | --- | --- | --- | --- |
| `text_genome_soul` | `text_v2` | `text_v2` | `text_v2` | yes |
| `text_fro` | `text_v2` | `text_v2` | `text_v2` | no |
| `code_fibonacci` | `code_v2` | `code_v2` | `code_v2` | yes |
| `code_prime` | `code_v2` | `text_v2` | `code_v2` | yes |
| `math_linear` | `math_v1` | `math_v1` | `math_v1` | yes |
| `math_speed` | `math_v1` | `math_v1` | `math_v1` | yes |
| `agentic_eval_plan` | `agentic_v1` | `agentic_v1` | `agentic_v1` | yes |
| `complex_multisoul` | `orchestrator_v1` | `orchestrator_v1` | `orchestrator_v1` | yes |

Interpretation:

```text
SwarmLM Orchestrator v3b restores full route accuracy on the controlled 8-task cascade suite and corrects the previous code_prime routing failure.
```

The remaining `text_fro` failure is not routing-related: it routes correctly to `text_v2`, but the generated specialist output does not match the current FRO-specific marker set. This result should therefore be reported as a controlled cascade-suite result, not as evidence of general routing robustness.

## What This Release Supports

This release supports:

- same frozen Genome hosting multiple specialized Souls;
- stronger v2 target specialization compared with v1;
- centralized Orchestrator routing;
- end-to-end cascade behavior over selected Souls;
- reproducible evaluation artifacts with JSONL, manifests, checkpoint metadata, and runtime telemetry;
- A40-class operation with approximately 18.6 GB peak VRAM per loaded Soul during evaluation.

## What This Release Does Not Claim

This release does not claim:

- frontier general-assistant performance;
- broad benchmark superiority;
- universal self-routing by every Soul;
- autonomous multi-Soul composition into a single final response;
- superiority of FRO over AdamW without controlled optimizer ablations;
- production readiness.

## Recommended Usage

Use this release for research into:

- modular specialization over a shared frozen substrate;
- centralized routing among specialist adapters/Souls;
- non-Transformer sequence-model alternatives;
- controlled evaluation of modular model systems;
- FRO training telemetry and high-rank Soul dynamics.

## Citation Language

Recommended short description:

> RTH-LM / SwarmLM v2 is a modular Genome/Soul research system. A shared frozen Genome supports multiple rank-512 specialist Souls, while `orchestrator_v2` routes controlled requests to specialist executors. In a controlled cascade evaluation, SwarmLM v2 reached 87.5% route accuracy and 75% end-to-end cascade success.

Updated Orchestrator v3b description:

> SwarmLM Orchestrator v3b improves centralized routing over v2 while preserving the same frozen Genome and v2 specialist Souls. On the controlled 8-task cascade suite, v3b reached 100% route accuracy and 87.5% cascade success, correcting the previous `code_prime` routing failure.

## License

Model artifacts are released for research and non-commercial use under the project license unless a separate commercial license is granted by RTH Italia.
