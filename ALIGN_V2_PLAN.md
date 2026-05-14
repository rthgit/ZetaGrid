# Align v2 Plan for RTH-LM / SwarmLM

## Goal

Align v2 is not intended to turn RTH-LM into a generic chatbot. Its purpose is to improve the scientific SwarmLM result by reducing template overfit, improving off-domain behavior, and testing whether modular specialization remains stable under broader prompt variation.

The v1 result established:

```text
shared frozen Genome
six behaviorally distinct rank-512 Souls
orchestrator-only route accuracy = 1.000 on controlled route prompts
target marker score = 0.848
off-target marker score = 0.218
```

Align v2 should preserve specialization while reducing uncontrolled domain invasion.

## Current v1 Limitations

Observed behavior:

- Code Soul tends to generate code even on non-code prompts.
- Math Soul tends to generate math examples even on non-math prompts.
- Agentic Soul tends to generate plans even when not asked for planning.
- Orchestrator Soul tends to route even outside route-format prompts.
- Prompt formats are still too narrow and template-driven.

Interpretation:

```text
v1 validates modular specialization.
v2 should improve routing discipline and prompt robustness.
```

## v2 Soul Targets

| Soul | v1 Checkpoint | v2 Goal |
| --- | --- | --- |
| `text_align_v2` | `TEXT_V2_ALIGN.pt` | Better natural explanations, IT/EN variation, less template leakage. |
| `code_align_v2` | `CODE_V2_ALIGN.pt` | More robust Python completions, syntax closure, less code on non-code prompts. |
| `math_align_v2` | `MATH_V1_ALIGN.pt` | More stable short reasoning, varied algebra/arithmetic prompts, less math leakage. |
| `instruction_v2` | `INSTRUCTION_V1_SMOKE.pt` | Better instruction following and concise formatting. |
| `agentic_v2` | `AGENTIC_V1_SMOKE.pt` | Cleaner multi-step planning with task boundaries. |
| `orchestrator_v2` | `ORCHESTRATOR_V1_SMOKE.pt` | Routing under varied natural-language requests, explicit abstain/defer behavior. |

Optional new Souls after v2:

| Soul | Purpose |
| --- | --- |
| `prompt_soul_v1` | Rewrite and decompose complex user requests. |
| `critic_soul_v1` | Validate outputs and identify likely errors. |
| `memory_soul_v1` | Select context/retrieval items. |

## Dataset Design

Each v2 dataset should include four classes of examples.

### 1. Positive Domain Examples

Examples where the Soul should answer directly in its own domain.

Text:

```text
User: Explain FRO in simple terms.
Assistant: ...
```

Code:

```text
<|file|> language=python
# Instruction: Write a function that validates an email address.
...
```

Math:

```text
<|math|>
Problem: If 4x - 8 = 20, solve for x.
Solution: ...
```

### 2. Prompt Variation

The same intent should be written in multiple ways:

```text
Solve 3x + 5 = 20.
Find x when 3x + 5 equals 20.
What value of x satisfies 3x + 5 = 20?
```

### 3. Off-Domain Discipline

The Soul should learn not to force its specialty when a request is out of scope.

Example for Code Soul:

```text
<|instruction|>
User: Explain the Genome/Soul architecture in Italian.
Assistant: ROUTE_REQUESTED: text_align_v2
<|endinstruction|>
```

Example for Math Soul:

```text
<|instruction|>
User: Write Python code for fibonacci.
Assistant: ROUTE_REQUESTED: code_align_v2
<|endinstruction|>
```

### 4. Stop and Boundary Control

Every dataset should include strict end tokens:

```text
<|endinstruction|>
<|endfile|>
<|endmath|>
<|endagentic|>
<|endroute|>
```

The goal is to reduce multi-sample continuation and format leakage.

## Proposed Training Settings

Start conservative:

```text
steps: 1000
seq_len: 384
batch_size: 1
grad_accum: 4
rank: 512
lr: 1e-6 to 1.5e-6
fro_alpha: 0.25
fro_gamma: 0.6
save_every: 250
```

For math:

```text
fro_gamma: 0.7
```

If loss collapses too quickly below 0.03 on a tiny dataset, stop and expand data rather than continuing.

## Evaluation Requirements

Use the same v1 suite for direct comparison:

```text
EVAL_SWARMLM_SUITE.py
```

Required metrics:

- target marker score
- off-target marker score
- orchestrator-only route accuracy
- non-orchestrator route accuracy
- average tokens/sec
- peak eval VRAM
- raw JSONL outputs
- manifest hashes

Success criteria:

```text
target marker score >= v1 target score or qualitatively cleaner
off-target marker score lower than v1 or better abstain behavior
orchestrator-only route accuracy remains high
non-orchestrator route accuracy remains low unless deliberate route-defer behavior is added
```

## Research Interpretation

If v2 improves off-domain discipline while preserving target behavior, it supports the stronger claim:

```text
SwarmLM specialization can be made more controllable through alignment, without modifying the shared frozen Genome.
```

This is the next research step before any 60B/70B scaling attempt.
