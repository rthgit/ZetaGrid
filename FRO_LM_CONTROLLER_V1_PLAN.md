# FRO-LM Controller v1

## Purpose

FRO-LM is a lightweight control model for SwarmLM. It is not intended to replace the Orchestrator or the specialist Souls.

The intended separation is:

```text
Orchestrator = chooses the primary route
FRO-LM       = evaluates confidence, ambiguity, risk, fallback, and validation
Soul         = executes the selected specialist behavior
```

## Cascade

```text
User prompt
-> Orchestrator v3b selects route
-> FRO-LM evaluates routing state
-> selected specialist Soul generates
-> FRO-LM validates output or requests fallback
```

## Controller Decisions

FRO-LM v1 emits a compact JSON-like control block:

```json
{
  "route_confidence": "high",
  "domain_ambiguity": "low",
  "safety_risk": "low",
  "agentic_risk": "none",
  "needs_fallback": false,
  "fallback_route": "",
  "needs_multisoul": false,
  "validation_action": "accept",
  "reason": "The route is direct and the requested domain is clear."
}
```

Allowed fields:

```text
route_confidence: high | medium | low
domain_ambiguity: low | medium | high
safety_risk: low | medium | high
agentic_risk: none | tool_intent | unsafe_delegation | external_action
needs_fallback: true | false
fallback_route: "" | text_v2 | code_v2 | math_v1 | agentic_v1 | orchestrator_v1
needs_multisoul: true | false
validation_action: accept | fallback | split | reject | revise
```

## Dataset Modes

### Pre-route Control

Input:

```text
<|fro_control|>
MODE: pre_route
USER_REQUEST: ...
ORCHESTRATOR_ROUTE: ...
```

Output:

```text
CONTROL: {...}
<|endfro|>
```

### Post-output Validation

Input:

```text
<|fro_control|>
MODE: post_output
USER_REQUEST: ...
ORCHESTRATOR_ROUTE: ...
SPECIALIST_SOUL: ...
SPECIALIST_OUTPUT: ...
```

Output:

```text
CONTROL: {...}
<|endfro|>
```

## Initial Scope

FRO-LM v1 should cover:

- clear route confirmation;
- route ambiguity detection;
- code-vs-text hard negatives;
- multi-Soul decomposition;
- agentic/tool-use risk detection;
- unsafe delegation rejection;
- specialist output marker failure;
- fallback recommendation.

## Evaluation

Compare:

```text
orchestrator_v3b only
vs
orchestrator_v3b + FRO-LM controller
```

Metrics:

```text
route confidence accuracy
fallback precision
fallback recall
unsafe delegation detection
false fallback rate
post-output validation accuracy
cascade success delta
```

## Scientific Claim

Conservative claim:

```text
FRO-LM is a lightweight resonance controller for SwarmLM. It monitors route confidence, domain ambiguity, safety risk, and cascade validity over a shared frozen Genome/Soul system.
```

Non-claims:

```text
FRO-LM is not a general assistant.
FRO-LM is not an autonomous tool executor.
FRO-LM does not prove production safety.
FRO-LM v1 does not replace external policy validation.
```

## Training Recommendation

Start from an Orchestrator-adjacent checkpoint:

```text
/workspace/zetagrid_50b/checkpoints/orchestrator_v3b/ORCHESTRATOR_V3B.pt
```

Train as a new controller Soul:

```text
/workspace/zetagrid_50b/checkpoints/fro_controller_v1/FRO_CONTROLLER_V1.pt
```

Recommended first run:

```text
steps: 300-600
seq_len: 512
lr: 1e-7 to 3e-7
rank: 512
dataset: 64-128 MB
```

Build dataset:

```bash
python BUILD_FRO_CONTROLLER_V1_DATASET.py \
  --base_dir /workspace/zetagrid_50b \
  --target_mb 64
```

Train first controller:

```bash
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode fro_controller_v1 \
  --base_dir /workspace/zetagrid_50b \
  --steps 300 \
  --save_every 300 \
  --seq_len 512 \
  --batch_size 1 \
  --grad_accum 4 \
  --rank 512 \
  --lr 1e-7 \
  --fro_gamma 0.6
```

If v1 falls back to Orchestrator-style `ROUTE/REASON` output, use v1b. v1b
switches from JSON-like output to line-oriented controller fields:

```text
CONFIDENCE: high
AMBIGUITY: low
SAFETY: low
AGENTIC_RISK: none
ACTION: accept
FALLBACK_ROUTE: none
MULTISOUL: false
REASON: ...
```

Build v1b dataset:

```bash
python BUILD_FRO_CONTROLLER_V1B_DATASET.py \
  --base_dir /workspace/zetagrid_50b \
  --target_mb 128
```

Train v1b:

```bash
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode fro_controller_v1b \
  --base_dir /workspace/zetagrid_50b \
  --steps 600 \
  --save_every 600 \
  --seq_len 512 \
  --batch_size 1 \
  --grad_accum 4 \
  --rank 512 \
  --lr 3e-7 \
  --fro_gamma 0.6
```

Release checkpoint path:

```text
/workspace/zetagrid_50b/checkpoints/fro_controller_v1/FRO_CONTROLLER_V1.pt
```
