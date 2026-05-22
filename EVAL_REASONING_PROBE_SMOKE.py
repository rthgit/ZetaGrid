#!/usr/bin/env python3
"""
Smoke evaluation for reasoning probe Souls.

The scorer checks the first assistant answer for a required final answer marker.
It is intentionally small and deterministic: use it to decide whether a
reasoning/scaling run is worth extending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from EVAL_TEXT_ALIGN_SMOKE import first_assistant_answer, has_format_leak
from EVAL_SWARMLM_CASCADE import generate, load_model, marker_score


TASKS = [
    {
        "name": "linear_heldout",
        "prompt": "<|instruction|>\nUser: Solve step by step: 4x + 2 = 18.\nAssistant:",
        "markers": ["x = 4"],
    },
    {
        "name": "average_speed_heldout",
        "prompt": "<|instruction|>\nUser: A bike travels 90 km in 3 hours. What is the average speed?\nAssistant:",
        "markers": ["30", "km/h"],
    },
    {
        "name": "sequence_double_heldout",
        "prompt": "<|instruction|>\nUser: What is the next number: 5, 10, 20, 40, ?\nAssistant:",
        "markers": ["80"],
    },
    {
        "name": "sequence_add_heldout",
        "prompt": "<|instruction|>\nUser: What is the next number: 4, 9, 14, 19, ?\nAssistant:",
        "markers": ["24"],
    },
    {
        "name": "logic_universal_heldout",
        "prompt": "<|instruction|>\nUser: If all roses are flowers and this plant is a rose, is this plant a flower?\nAssistant:",
        "markers": ["Yes"],
    },
    {
        "name": "logic_some_not_all",
        "prompt": "<|instruction|>\nUser: If some cars are electric, does that prove all cars are electric?\nAssistant:",
        "markers": ["No"],
    },
    {
        "name": "prime_yes_heldout",
        "prompt": "<|instruction|>\nUser: Is 31 prime?\nAssistant:",
        "markers": ["Yes"],
    },
    {
        "name": "prime_no_heldout",
        "prompt": "<|instruction|>\nUser: Is 27 prime?\nAssistant:",
        "markers": ["No"],
    },
    {
        "name": "percent_heldout",
        "prompt": "<|instruction|>\nUser: What is 20% of 150?\nAssistant:",
        "markers": ["30"],
    },
    {
        "name": "factor_heldout",
        "prompt": "<|instruction|>\nUser: Factor 60 into primes.\nAssistant:",
        "markers": ["2", "3", "5"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--suite_name", default="reasoning_probe_smoke")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--max_new", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--success_threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    genome = args.genome or base_dir / "zetagrid_25b_production.npy"
    ckpt = args.ckpt if args.ckpt.is_absolute() else base_dir / args.ckpt
    out_dir = base_dir / "reports" / args.suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"eval_{args.suite_name}.jsonl"
    raw_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[RUN] suite={args.suite_name} device={device} dtype={dtype} rank={args.rank} layers={args.layers}")
    print(f"[RUN] genome={genome}")
    print(f"[RUN] ckpt={ckpt}")

    model, meta = load_model(genome, ckpt, device, dtype, args.layers, args.rank)
    rows = []
    for task in TASKS:
        print(f"\n--- {task['name']} ---")
        output, telemetry = generate(model, task["prompt"], device, dtype, args.max_new, args.temperature, args.top_k)
        answer = first_assistant_answer(task["prompt"], output)
        score = marker_score(answer, task["markers"])
        leak = has_format_leak(answer)
        success = score >= args.success_threshold and not leak
        row = {
            "task": task["name"],
            "markers": task["markers"],
            "marker_score": score,
            "format_leak": leak,
            "success": success,
            "answer": answer,
            "output": output,
            "telemetry": telemetry,
        }
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"marker={score:.3f} leak={leak} success={success}")
        print(answer.replace("\n", "\\n")[:500])

    acc = sum(1 for row in rows if row["success"]) / len(rows)
    report_path = out_dir / f"{args.suite_name.upper()}_REPORT.md"
    report = [
        f"# {args.suite_name} Evaluation",
        "",
        "## Summary Metrics",
        "",
        f"- Tasks: {len(rows)}",
        f"- Success rate: {acc:.3f}",
        f"- Checkpoint step: {meta.get('checkpoint_step')}",
        f"- Checkpoint loss: {meta.get('checkpoint_loss')}",
        "",
        "## Task Results",
        "",
    ]
    for row in rows:
        report.append(f"- {row['task']}: marker={row['marker_score']:.3f} leak={row['format_leak']} success={row['success']}")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n[DONE] raw={raw_path}")
    print(f"[DONE] report={report_path}")
    print(f"[DONE] success_rate={acc:.3f}")


if __name__ == "__main__":
    main()
