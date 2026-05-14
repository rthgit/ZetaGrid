#!/usr/bin/env python3
"""
End-to-end SwarmLM cascade evaluation.

Protocol:
1. Load the shared Genome and orchestrator_v2.
2. Ask the orchestrator to route a user request.
3. Load the selected v2 Soul over the same Genome.
4. Generate the specialist output.
5. Record routing accuracy, specialist marker score, cascade success, latency,
   VRAM, and artifact hashes.

This evaluates the intended SwarmLM architecture: centralized routing plus
specialized executor Souls. It does not require every non-orchestrator Soul to
self-route outside its domain.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch

from TRAIN_SOUL_V2_FRO_A40 import GenomeWeightBank, ZetaGridSoul, load_init_checkpoint


ORCHESTRATOR = "checkpoints/orchestrator_v2/ORCHESTRATOR_V2.pt"

ROUTE_TO_SOUL = {
    "text_v2": ("text_align_v2", "checkpoints/text_align_v2/TEXT_ALIGN_V2.pt"),
    "code_v2": ("code_align_v2", "checkpoints/code_align_v2/CODE_ALIGN_V2.pt"),
    "math_v1": ("math_align_v2", "checkpoints/math_align_v2/MATH_ALIGN_V2.pt"),
    "agentic_v1": ("agentic_v2", "checkpoints/agentic_v2/AGENTIC_V2.pt"),
    "orchestrator_v1": ("orchestrator_v2", "checkpoints/orchestrator_v2/ORCHESTRATOR_V2.pt"),
}

TASKS = [
    {
        "name": "text_genome_soul",
        "request": "Explain the Genome/Soul architecture in simple English.",
        "expected_route": "text_v2",
        "specialist_prompt": "<|instruction|>\nUser: Explain the Genome/Soul architecture in simple English.\nAssistant:",
        "markers": ["Genome", "Soul", "shared"],
    },
    {
        "name": "text_fro",
        "request": "Summarize Fractal Resonant Optimization in simple English.",
        "expected_route": "text_v2",
        "specialist_prompt": "<|instruction|>\nUser: Summarize Fractal Resonant Optimization in simple English.\nAssistant:",
        "markers": ["gradient", "coherence", "stable"],
    },
    {
        "name": "code_fibonacci",
        "request": "Write a Python function for fibonacci.",
        "expected_route": "code_v2",
        "specialist_prompt": "<|file|> language=python task=cascade\n# Instruction: Write a Python fibonacci function.\ndef fibonacci(n):\n",
        "markers": ["return", "for", "range"],
    },
    {
        "name": "code_prime",
        "request": "Write a Python primality test.",
        "expected_route": "code_v2",
        "specialist_prompt": "<|file|> language=python task=cascade\n# Instruction: Write a Python primality test.\ndef is_prime(n):\n    if n < 2:\n",
        "markers": ["return False", "while", "%"],
    },
    {
        "name": "math_linear",
        "request": "Solve 3x + 5 = 20.",
        "expected_route": "math_v1",
        "specialist_prompt": "<|math|>\nProblem:\nIf 3x + 5 = 20, solve for x.\n\nSolution:\n",
        "markers": ["3x", "15", "x = 5"],
    },
    {
        "name": "math_speed",
        "request": "A train travels 120 km in 2 hours. What is the average speed?",
        "expected_route": "math_v1",
        "specialist_prompt": "<|math|>\nProblem:\nA train travels 120 km in 2 hours. What is the average speed?\n\nSolution:\n",
        "markers": ["120", "2", "60"],
    },
    {
        "name": "agentic_eval_plan",
        "request": "Create a step-by-step plan to evaluate the model.",
        "expected_route": "agentic_v1",
        "specialist_prompt": "<|agentic|>\nTask: Create a step-by-step plan to evaluate the model.\nPlan:\n",
        "markers": ["1.", "2.", "evaluate"],
    },
    {
        "name": "complex_multisoul",
        "request": "Explain the idea, write pseudocode, and solve a small equation.",
        "expected_route": "orchestrator_v1",
        "specialist_prompt": "<|route|>\nUSER_REQUEST: Explain the idea, write pseudocode, and solve a small equation.\n",
        "markers": ["ROUTE:", "orchestrator_v1"],
    },
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def extract_route(text: str) -> str:
    match = re.search(r"ROUTE:\s*([A-Za-z0-9_\-]+)", text)
    return match.group(1).strip() if match else ""


def marker_score(text: str, markers: list[str]) -> float:
    if not markers:
        return 0.0
    lower = text.lower()
    return sum(1 for marker in markers if marker.lower() in lower) / len(markers)


@torch.no_grad()
def generate(
    model: ZetaGridSoul,
    prompt: str,
    device: str,
    dtype: torch.dtype,
    max_new: int,
    temperature: float,
    top_k: int,
) -> tuple[str, dict]:
    model.eval()
    idx = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    start_tokens = idx.shape[1]
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(max_new):
        idx_crop = idx[:, -1024:]
        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            logits, _ = model(idx_crop)
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)

    elapsed = time.time() - t0
    new_tokens = idx.shape[1] - start_tokens
    text = bytes(idx[0].detach().cpu().tolist()).decode("utf-8", errors="replace")
    telemetry = {
        "elapsed_sec": elapsed,
        "new_tokens": new_tokens,
        "tokens_per_sec": new_tokens / max(elapsed, 1e-9),
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
    }
    return text, telemetry


def load_model(
    genome: Path,
    ckpt: Path,
    device: str,
    dtype: torch.dtype,
    layers: int,
    rank: int,
) -> tuple[ZetaGridSoul, dict]:
    cleanup()
    t0 = time.time()
    bank = GenomeWeightBank(genome, dtype=dtype, device=device)
    model = ZetaGridSoul(bank, n_layers=layers, rank=rank, dtype=dtype).to(device)
    del bank.data
    del bank
    cleanup()
    step, loss = load_init_checkpoint(model, ckpt, device)
    return model, {
        "checkpoint": str(ckpt),
        "checkpoint_step": step,
        "checkpoint_loss": loss,
        "load_elapsed_sec": time.time() - t0,
        "post_load_vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_suite(args: argparse.Namespace) -> None:
    base_dir = args.base_dir
    out_dir = args.out_dir or (base_dir / "reports" / "swarmlm_v2_cascade")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "eval_swarmlm_v2_cascade.jsonl"
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "SWARMLM_V2_CASCADE_REPORT.md"
    raw_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    genome = args.genome or (base_dir / "zetagrid_25b_production.npy")

    manifest = {
        "suite": "swarmlm_v2_cascade",
        "base_dir": str(base_dir),
        "genome": str(genome),
        "genome_sha256": sha256_file(genome) if args.hash_files else None,
        "device": device,
        "dtype": str(dtype),
        "max_new_route": args.max_new_route,
        "max_new_specialist": args.max_new_specialist,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "checkpoints": {},
    }

    print("===== LOAD orchestrator_v2 =====")
    orchestrator_ckpt = base_dir / ORCHESTRATOR
    orchestrator, orch_meta = load_model(genome, orchestrator_ckpt, device, dtype, args.layers, args.rank)
    orch_meta["checkpoint_sha256"] = sha256_file(orchestrator_ckpt) if args.hash_files else None
    manifest["checkpoints"]["orchestrator_v2"] = orch_meta

    rows = []
    for task in TASKS:
        route_prompt = f"<|route|>\nUSER_REQUEST: {task['request']}\n"
        cascade_t0 = time.time()
        print(f"\n--- ROUTE {task['name']} ---")
        route_output, route_telemetry = generate(
            orchestrator,
            route_prompt,
            device,
            dtype,
            args.max_new_route,
            args.temperature,
            args.top_k,
        )
        route = extract_route(route_output)
        route_ok = route == task["expected_route"]
        print(route_output[:400].replace("\n", "\\n"))

        selected = ROUTE_TO_SOUL.get(route)
        if selected is None:
            row = {
                "task": task["name"],
                "request": task["request"],
                "expected_route": task["expected_route"],
                "route": route,
                "route_ok": route_ok,
                "selected_soul": "",
                "specialist_marker_score": 0.0,
                "cascade_success": False,
                "route_output": route_output,
                "specialist_output": "",
                "cascade_elapsed_sec": time.time() - cascade_t0,
                "route_telemetry": route_telemetry,
            }
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue

        soul_name, soul_rel = selected
        soul_ckpt = base_dir / soul_rel
        print(f"--- LOAD SPECIALIST {soul_name} ---")
        specialist, spec_meta = load_model(genome, soul_ckpt, device, dtype, args.layers, args.rank)
        if soul_name not in manifest["checkpoints"]:
            spec_meta["checkpoint_sha256"] = sha256_file(soul_ckpt) if args.hash_files else None
            manifest["checkpoints"][soul_name] = spec_meta

        print(f"--- GENERATE {soul_name} / {task['name']} ---")
        specialist_output, specialist_telemetry = generate(
            specialist,
            task["specialist_prompt"],
            device,
            dtype,
            args.max_new_specialist,
            args.temperature,
            args.top_k,
        )
        score = marker_score(specialist_output, task["markers"])
        cascade_success = route_ok and score >= args.success_threshold
        print(specialist_output[:500].replace("\n", "\\n"))

        row = {
            "task": task["name"],
            "request": task["request"],
            "expected_route": task["expected_route"],
            "route": route,
            "route_ok": route_ok,
            "selected_soul": soul_name,
            "specialist_checkpoint": str(soul_ckpt),
            "specialist_marker_score": score,
            "success_threshold": args.success_threshold,
            "cascade_success": cascade_success,
            "route_output": route_output,
            "specialist_output": specialist_output,
            "cascade_elapsed_sec": time.time() - cascade_t0,
            "route_telemetry": route_telemetry,
            "specialist_telemetry": specialist_telemetry,
        }
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        del specialist
        cleanup()

    del orchestrator
    cleanup()

    metrics = {
        "tasks": len(rows),
        "route_accuracy": mean([1.0 if r["route_ok"] else 0.0 for r in rows]),
        "specialist_marker_score_avg": mean([r["specialist_marker_score"] for r in rows]),
        "cascade_success_rate": mean([1.0 if r["cascade_success"] else 0.0 for r in rows]),
        "average_cascade_elapsed_sec": mean([r["cascade_elapsed_sec"] for r in rows]),
        "average_route_tokens_per_sec": mean([r["route_telemetry"]["tokens_per_sec"] for r in rows]),
        "average_specialist_tokens_per_sec": mean(
            [r["specialist_telemetry"]["tokens_per_sec"] for r in rows if "specialist_telemetry" in r]
        ),
        "peak_route_vram_gb": max([r["route_telemetry"]["vram_peak_gb"] for r in rows], default=0.0),
        "peak_specialist_vram_gb": max(
            [r["specialist_telemetry"]["vram_peak_gb"] for r in rows if "specialist_telemetry" in r],
            default=0.0,
        ),
    }
    manifest["summary_metrics"] = metrics

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = [
        "# SwarmLM v2 Cascade Evaluation",
        "",
        "## Scope",
        "",
        "This suite evaluates the intended SwarmLM cascade: centralized routing with `orchestrator_v2`, followed by specialist generation with the selected v2 Soul over the same frozen Genome.",
        "",
        "## Artifacts",
        "",
        f"- Raw JSONL: `{raw_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Genome: `{genome}`",
        "",
        "## Summary Metrics",
        "",
        f"- Tasks: {metrics['tasks']}",
        f"- Route accuracy: {metrics['route_accuracy']:.3f}",
        f"- Specialist marker score average: {metrics['specialist_marker_score_avg']:.3f}",
        f"- Cascade success rate: {metrics['cascade_success_rate']:.3f}",
        f"- Average cascade latency: {metrics['average_cascade_elapsed_sec']:.2f}s",
        f"- Average route tokens/sec: {metrics['average_route_tokens_per_sec']:.2f}",
        f"- Average specialist tokens/sec: {metrics['average_specialist_tokens_per_sec']:.2f}",
        f"- Peak route VRAM: {metrics['peak_route_vram_gb']:.2f} GB",
        f"- Peak specialist VRAM: {metrics['peak_specialist_vram_gb']:.2f} GB",
        "",
        "## Task Results",
        "",
    ]
    for row in rows:
        report.append(
            f"- {row['task']}: expected_route={row['expected_route']} got={row['route']} "
            f"route_ok={row['route_ok']} selected={row['selected_soul']} "
            f"marker={row['specialist_marker_score']:.3f} success={row['cascade_success']}"
        )

    report.extend(["", "## Interpretation", ""])
    report.extend(
        [
            "This cascade suite is the primary end-to-end SwarmLM test. The previous v2 suite tests harder non-orchestrator self-delegation behavior, while this suite tests the intended centralized routing architecture.",
            "",
            "A strong result here supports the claim that SwarmLM should route centrally and execute with specialized Souls.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\n[DONE] raw={raw_path}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] report={report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--max_new_route", type=int, default=80)
    parser.add_argument("--max_new_specialist", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument("--hash_files", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
