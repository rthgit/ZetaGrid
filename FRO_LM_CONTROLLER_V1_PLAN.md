# FRO-LM Controller v1

## Current Result: FRO-LM Small v0

The preferred controller path is now validated as a small standalone model:

```text
FRO-LM Small v0
parameters: 44.5M
initialization: random
optimizer: FRO
Genome dependency: none
checkpoint size: ~178 MB
build VRAM: ~0.18 GB
training best loss: 0.0329
smoke eval: 5/5
```

Artifacts:

```text
controllers/fro_lm_small_v0/FRO_LM_SMALL_V0.pt
reports/fro_lm_small_v0/FRO_LM_SMALL_V0_SMOKE.md
```

Smoke behaviors covered:

```text
accept   -> confirm correct route
fallback -> correct wrong Orchestrator route
reject   -> flag unsafe agentic request
split    -> send multi-capability request to Orchestrator
revise   -> flag weak specialist output
```

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

Preferred FRO-LM direction:

```text
FRO-LM Small should be trained from scratch as a lightweight standalone controller,
not as another 949M-parameter Soul initialized from the Orchestrator.
```

The Soul-based v1/v1b line is useful as a negative control:

```text
Orchestrator-initialized controller Souls inherit a strong ROUTE/REASON prior
and tend to resist new control formats.
```

Build route-compatible controller dataset:

```bash
python BUILD_FRO_CONTROLLER_V2_DATASET.py \
  --base_dir /workspace/zetagrid_50b \
  --target_mb 128
```

Train small FRO-LM from scratch:

```bash
python TRAIN_FRO_LM_SMALL.py \
  --base_dir /workspace/zetagrid_50b \
  --data /workspace/zetagrid_50b/data/swarmlm_v4/fro_controller_v2.bin \
  --save_dir /workspace/zetagrid_50b/checkpoints/fro_lm_small_v0 \
  --steps 2000 \
  --save_every 500 \
  --seq_len 512 \
  --batch_size 8 \
  --grad_accum 4 \
  --layers 12 \
  --d_model 512 \
  --d_ff 2048 \
  --lr 3e-4 \
  --fro_gamma 0.6
```

Evaluate:

```bash
python EVAL_FRO_LM_SMALL.py \
  --ckpt /workspace/zetagrid_50b/checkpoints/fro_lm_small_v0/latest.pt
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

If v1b still emits Orchestrator-style routes instead of controller fields, use
v2. v2 is route-compatible by design:

```text
ROUTE: corrected_or_confirmed_route
ACTION: accept | fallback | split | reject | revise
CONFIDENCE: high | medium | low
RISK: low | high
REASON: ...
<|endfro|>
```

This uses the inherited Orchestrator `ROUTE/REASON` prior as an advantage:
FRO-LM becomes a critic that confirms or corrects the Orchestrator route and
adds an action.

Build v2 dataset:

```bash
python BUILD_FRO_CONTROLLER_V2_DATASET.py \
  --base_dir /workspace/zetagrid_50b \
  --target_mb 128
```

Train v2:

```bash
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode fro_controller_v2 \
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
