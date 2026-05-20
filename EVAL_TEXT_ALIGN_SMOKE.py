#!/usr/bin/env python3
"""
Text specialist smoke evaluation.

Use this before the full cascade to decide whether a text checkpoint is worth
promoting. It loads only the text Soul over the frozen Genome and checks direct
natural-language prompts, including the known hard case from FRO cascade eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from EVAL_SWARMLM_CASCADE import generate, load_model, marker_score


TASKS = [
    {
        "name": "genome_soul",
        "profile": "rth",
        "prompt": "<|instruction|>\nUser: Explain the Genome/Soul architecture in simple English.\nAssistant:",
        "markers": ["Genome", "Soul", "shared"],
    },
    {
        "name": "fro_simple",
        "profile": "rth",
        "prompt": "<|instruction|>\nUser: Summarize Fractal Resonant Optimization in simple English.\nAssistant:",
        "markers": ["gradient", "coherence", "stable"],
    },
    {
        "name": "prime_no_code",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Explain what a primality test is, no code.\nAssistant:",
        "markers": ["prime", "number", "divisible"],
    },
    {
        "name": "sql_group_by_no_query",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Describe what SQL GROUP BY means, no query.\nAssistant:",
        "markers": ["rows", "same", "group"],
    },
    {
        "name": "python_function_no_code",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Explain what a Python function is, no code.\nAssistant:",
        "markers": ["named", "reusable", "return"],
    },
    {
        "name": "parser_plain",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Explain what a parser does.\nAssistant:",
        "markers": ["structured", "program", "understand"],
    },
    {
        "name": "api_plain",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Explain what an API is.\nAssistant:",
        "markers": ["software", "request", "data"],
    },
    {
        "name": "no_benchmark_warning",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Explain why a benchmark is not enough.\nAssistant:",
        "markers": ["benchmark", "general", "fail"],
    },
    {
        "name": "italian_prime",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: Spiega che cos'e' un test di primalita', senza codice.\nAssistant:",
        "markers": ["numero", "primo", "divisibile"],
    },
    {
        "name": "no_genome_drift",
        "profile": "general",
        "prompt": "<|instruction|>\nUser: If the user asks about a normal concept, should the answer mention Genome/Soul?\nAssistant:",
        "markers": ["No", "only", "architecture"],
    },
]


def first_assistant_answer(prompt: str, output: str) -> str:
    if output.startswith(prompt):
        text = output[len(prompt) :]
    else:
        marker = "Assistant:"
        idx = output.find(marker)
        text = output[idx + len(marker) :] if idx >= 0 else output
    stops = ["<|endinstruction|>", "\n<|instruction|>", "\nUser:", "\nAssistant:"]
    cut = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def has_format_leak(answer: str) -> bool:
    leaked = ["ROUTE_REQUESTED:", "<|instruction|>", "<|route|>", "User:", "Assistant:"]
    return any(token in answer for token in leaked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--text_ckpt", type=Path, required=True)
    parser.add_argument("--suite_name", default="text_align_smoke")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--max_new", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--success_threshold", type=float, default=0.67)
    parser.add_argument("--profile", choices=["all", "rth", "general"], default="all")
    parser.add_argument("--score_scope", choices=["answer", "legacy_output"], default="answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    genome = args.genome or (base_dir / "zetagrid_25b_production.npy")
    text_ckpt = args.text_ckpt if args.text_ckpt.is_absolute() else base_dir / args.text_ckpt
    out_dir = base_dir / "reports" / args.suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"eval_{args.suite_name}.jsonl"
    raw_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[RUN] suite={args.suite_name} device={device} dtype={dtype}")
    print(f"[RUN] genome={genome}")
    print(f"[RUN] text={text_ckpt}")

    model, meta = load_model(genome, text_ckpt, device, dtype, args.layers, args.rank)
    rows = []
    selected_tasks = [task for task in TASKS if args.profile == "all" or task["profile"] == args.profile]
    for task in selected_tasks:
        print(f"\n--- {task['name']} ---")
        output, telemetry = generate(
            model,
            task["prompt"],
            device,
            dtype,
            args.max_new,
            args.temperature,
            args.top_k,
        )
        answer = first_assistant_answer(task["prompt"], output)
        scored_text = output if args.score_scope == "legacy_output" else answer
        score = marker_score(scored_text, task["markers"])
        format_leak = has_format_leak(answer)
        success = score >= args.success_threshold and (args.score_scope == "legacy_output" or not format_leak)
        row = {
            "task": task["name"],
            "profile": task["profile"],
            "score_scope": args.score_scope,
            "markers": task["markers"],
            "marker_score": score,
            "format_leak": format_leak,
            "success": success,
            "answer": answer,
            "output": output,
            "telemetry": telemetry,
        }
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"marker={score:.3f} scope={args.score_scope} leak={format_leak} success={success}")
        print(answer.replace("\n", "\\n")[:500])

    acc = sum(1 for row in rows if row["success"]) / len(rows)
    report_path = out_dir / f"{args.suite_name.upper()}_REPORT.md"
    report = [
        f"# {args.suite_name} Evaluation",
        "",
        "## Artifacts",
        "",
        f"- Raw JSONL: `{raw_path}`",
        f"- Genome: `{genome}`",
        f"- Text checkpoint: `{text_ckpt}`",
        "",
        "## Summary Metrics",
        "",
        f"- Tasks: {len(rows)}",
        f"- Profile: {args.profile}",
        f"- Score scope: {args.score_scope}",
        f"- Success rate: {acc:.3f}",
        f"- Checkpoint step: {meta.get('checkpoint_step')}",
        f"- Checkpoint loss: {meta.get('checkpoint_loss')}",
        "",
        "## Task Results",
        "",
    ]
    for row in rows:
        report.append(
            f"- {row['task']}: profile={row['profile']} marker={row['marker_score']:.3f} "
            f"scope={row['score_scope']} leak={row['format_leak']} success={row['success']}"
        )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n[DONE] raw={raw_path}")
    print(f"[DONE] report={report_path}")
    print(f"[DONE] success_rate={acc:.3f}")


if __name__ == "__main__":
    main()
