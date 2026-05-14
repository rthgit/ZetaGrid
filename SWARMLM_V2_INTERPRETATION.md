# SwarmLM v2 Interpretation

## Result

SwarmLM v2 is scientifically useful, but it is not an across-the-board improvement over v1.

The v2 result separates three properties that should not be conflated:

| Property | Result | Interpretation |
| --- | --- | --- |
| Target specialization | Improved | v2 Souls are stronger in their intended domain. |
| Central routing | Stable and strong | `orchestrator_v2` keeps perfect route accuracy on controlled route prompts. |
| Self-delegation by every Soul | Not solved | Non-orchestrator Souls remain specialists rather than reliable self-routing agents. |

## Metrics

```text
Target marker score:
v1 0.848 -> v2 0.889

Orchestrator-only route accuracy:
v1 1.000 -> v2 1.000

Off-target marker score:
v1 0.218 -> v2 0.297

ROUTE_REQUESTED accuracy:
v2 0.000
```

## Scientific Interpretation

Align v2 improved target specialization but did not solve off-domain delegation. The Orchestrator remains the correct routing mechanism. Non-orchestrator Souls still behave as specialized executors, not as self-routing agents.

This is not a failure of the Genome/Soul/SwarmLM architecture. It clarifies the architecture:

```text
User request
-> Orchestrator Soul selects route
-> Specialist Soul executes
-> Optional future composer merges multi-Soul outputs
```

The correct claim is:

> SwarmLM v2 confirms that routing should be centralized in the Orchestrator, while individual Souls remain specialized executors.

The incorrect claim would be:

> Every Soul can reliably delegate outside its domain.

## Next Evaluation

The next test is the cascade evaluation:

```text
User prompt
-> orchestrator_v2 chooses ROUTE
-> selected v2 Soul generates specialist output
```

The cascade suite measures:

- route accuracy
- selected Soul marker score
- full cascade success
- total latency
- VRAM during route and specialist generation

This is the primary end-to-end SwarmLM evaluation because it tests the intended centralized routing architecture rather than requiring every specialist to self-route.

## Claim Boundary

This result supports modular specialization and centralized orchestration over a shared frozen Genome. It does not prove general assistant quality, broad benchmark superiority, autonomous multi-Soul composition, or FRO superiority over AdamW.
