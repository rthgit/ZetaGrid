# Hugging Face Model Card Update: SwarmLM v1

The following section can be appended to the main RTH-LM model card and adapted for the Code and Math repositories.

## SwarmLM v1: Modular Soul Evaluation

RTH-LM is being evaluated as a modular Genome/Soul architecture rather than a monolithic instruction model. A shared frozen Genome is used as the common parameter substrate, while rank-512 Souls provide task and behavior specialization.

SwarmLM v1 extends this setup with a first orchestration experiment: a dedicated Orchestrator Soul routes controlled requests toward specialized Souls.

### Core Claim

RTH-LM demonstrates a modular Genome/Soul architecture in which a shared frozen Genome supports multiple behaviorally distinct high-rank Souls. SwarmLM extends this into an orchestration layer that routes tasks among specialized Souls.

### Shared Genome

```text
Genome: zetagrid_25b_production.npy
SHA256: 09dcebf875ec9f9a3b8f1da17536b42f09bc50ec7334afb6426a1dd41f1762e5
Loaded shape/dtype: (6979321856,) int8
```

The Genome remained frozen during the Soul v1 alignment and SwarmLM experiments.

### Soul Checkpoints

| Soul | File | Role |
| --- | --- | --- |
| Text Align | `souls/text_align_v1/TEXT_V2_ALIGN.pt` | Natural-language/instruction text behavior |
| Code Align | `souls/code_align_v1/CODE_V2_ALIGN.pt` | Python/code-format behavior |
| Math Align | `souls/math_align_v1/MATH_V1_ALIGN.pt` | Short math Problem/Solution behavior |
| Instruction | `souls/instruction_v1/INSTRUCTION_V1_SMOKE.pt` | Response formatting and instruction behavior |
| Agentic | `souls/agentic_v1/AGENTIC_V1_SMOKE.pt` | Step-by-step planning behavior |
| Orchestrator | `souls/orchestrator_v1/ORCHESTRATOR_V1_SMOKE.pt` | ROUTE/REASON routing behavior |

Each Soul uses rank 512 and contains approximately 949.1M trainable parameters.

### Scientific Smoke Evaluation

Evaluation artifacts are available under:

```text
reports/swarmlm_v1_suite_hashed/
```

Key files:

```text
reports/swarmlm_v1_suite_hashed/eval_swarmlm_v1_suite.jsonl
reports/swarmlm_v1_suite_hashed/manifest.json
reports/swarmlm_v1_suite_hashed/SWARMLM_V1_EVAL_REPORT.md
reports/swarmlm_v1_suite_hashed/SWARMLM_V1_EVAL_REPORT_INTERPRETED.md
reports/SWARMLM_V1_TECHNICAL_REPORT.md
```

### Results

```text
Target-only marker score average: 0.848
Off-target marker score average: 0.218
Global route accuracy: 0.167
Orchestrator-only route accuracy: 1.000
Non-orchestrator route accuracy: 0.000
Average generation speed: 16.43 tokens/sec
Peak eval VRAM: 18.61 GB
```

The global route accuracy includes Souls that were not trained for routing. The relevant routing result is that `orchestrator_v1` reached 1.000 accuracy on controlled route-format prompts, while non-orchestrator Souls did not route. This supports functional specialization rather than universal routing.

### Interpretation

The experiment supports:

- same frozen Genome, different behavior by Soul;
- domain specialization for text/code/math;
- functional specialization for instruction/planning/routing;
- controlled Orchestrator routing over specialized Souls;
- stable FRO-based training on A40-class hardware.

The experiment does not claim:

- frontier chatbot quality;
- robust generalization outside the controlled evaluation format;
- end-to-end multi-Soul composition in a single final answer;
- superiority of FRO over AdamW without future ablation.

### Recommended Citation Language

RTH-LM is evaluated here as a modular Genome/Soul system. SwarmLM v1 demonstrates controlled modular specialization over a shared frozen Genome, with a dedicated Orchestrator Soul routing controlled requests among specialized Souls.
