# Orchestrator v3 Plan

## Goal

`orchestrator_v3` is a targeted routing update, not a full SwarmLM retrain.

It addresses the observed SwarmLM v2 cascade failures:

```text
text_fro: route was correct, but specialist output was weak for FRO-specific content.
code_prime: routed to text_v2 instead of code_v2.
```

The primary target is routing, especially code classification.

## Dataset

Builder:

```text
BUILD_ORCHESTRATOR_V3_DATASET.py
```

Output:

```text
/workspace/zetagrid_50b/data/swarmlm_v3/orchestrator_v3.bin
```

Dataset focus:

- code routing: primality, parsers, regex, SQL, JSONL, debugging;
- math routing: algebra, speed, percentages, averages;
- text routing: Genome/Soul, FRO, SwarmLM explanations;
- agentic routing: plans, checklists, evaluation design;
- multi-Soul routing: combined text/code/math/planning tasks;
- confusion pairs: explain vs implement, conceptual SQL vs SQL query, prime math vs prime code.

## Training

Continue from:

```text
/workspace/zetagrid_50b/checkpoints/orchestrator_v2/ORCHESTRATOR_V2.pt
```

Save to:

```text
/workspace/zetagrid_50b/checkpoints/orchestrator_v3/ORCHESTRATOR_V3.pt
```

Recommended first run:

```text
steps: 600
save_every: 600
seq_len: 384
batch_size: 1
grad_accum: 4
rank: 512
lr: 8e-7
fro_gamma: 0.6
```

Rationale: the v2 orchestrator is already strong. v3 should be a small targeted correction, not a broad behavioral rewrite.

## Evaluation

After training:

1. Run a route-only test containing known failures.
2. Run the SwarmLM cascade suite with `ORCHESTRATOR_V3.pt`.
3. Compare:

```text
v2 cascade route accuracy: 0.875
v2 cascade success rate: 0.750
```

Expected improvement:

```text
code_prime route: text_v2 -> code_v2
route accuracy target: >= 0.875, preferably 1.000 on the 8-task cascade
```

## Claim Boundary

`orchestrator_v3` should only be described as a targeted routing refinement. It should not be described as a general safety or agentic robustness solution.
