#!/usr/bin/env python3
"""
Scientific smoke-evaluation suite for RTH-LM Soul/SwarmLM experiments.

The suite is intentionally lightweight enough for a single A40, but records the
evidence needed for research iteration:
- same Genome, different Soul behavior
- fixed-format prompts matching alignment data
- SwarmLM route tests
- latency, VRAM, token counts, checkpoint metadata
- JSONL raw outputs and Markdown summary
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


SOUL_FILES = {
    "text_align_v1": "checkpoints/text_align_v1/TEXT_V2_ALIGN.pt",
    "code_align_v1": "checkpoints/code_align_v1/CODE_V2_ALIGN.pt",
    "math_align_v1": "checkpoints/math_align_v1/MATH_V1_ALIGN.pt",
    "instruction_v1": "checkpoints/instruction_v1/INSTRUCTION_V1_SMOKE.pt",
    "agentic_v1": "checkpoints/agentic_v1/AGENTIC_V1_SMOKE.pt",
    "orchestrator_v1": "checkpoints/orchestrator_v1/ORCHESTRATOR_V1_SMOKE.pt",
}

PROMPTS = {
    "text_it": {
        "target": "text_align_v1",
        "prompt": "<|instruction|>\nUser: Spiega in italiano che cos'e' un modello linguistico frattale.\nAssistant:",
        "markers": ["modello", "frattale", "strutture"],
    },
    "text_fro": {
        "target": "instruction_v1",
        "prompt": "<|instruction|>\nUser: Summarize Fractal Resonant Optimization in simple English.\nAssistant:",
        "markers": ["gradient", "coherent", "update"],
    },
    "code_fibonacci": {
        "target": "code_align_v1",
        "prompt": "<|file|> language=python task=alignment\n# Instruction: Complete a Python fibonacci function.\ndef fibonacci(n):\n",
        "markers": ["return", "for", "if"],
    },
    "code_prime": {
        "target": "code_align_v1",
        "prompt": "<|file|> language=python task=alignment\n# Instruction: Complete a Python primality test.\ndef is_prime(n):\n    if n < 2:\n",
        "markers": ["return False", "while", "%"],
    },
    "math_linear": {
        "target": "math_align_v1",
        "prompt": "<|math|>\nProblem:\nIf 3x + 5 = 20, solve for x.\n\nSolution:\n",
        "markers": ["3x", "15", "5"],
    },
    "math_speed": {
        "target": "math_align_v1",
        "prompt": "<|math|>\nProblem:\nA train travels 120 km in 2 hours. What is the average speed?\n\nSolution:\n",
        "markers": ["120", "2", "60"],
    },
    "agentic_plan": {
        "target": "agentic_v1",
        "prompt": "<|agentic|>\nTask: Plan a Soul evaluation run.\nPlan:\n",
        "markers": ["1.", "2.", "JSONL"],
    },
    "route_code": {
        "target": "orchestrator_v1",
        "prompt": "<|route|>\nUSER_REQUEST: Write a Python function for fibonacci.\n",
        "expected_route": "code_v2",
        "markers": ["ROUTE:", "code"],
    },
    "route_math": {
        "target": "orchestrator_v1",
        "prompt": "<|route|>\nUSER_REQUEST: Solve 3x + 5 = 20.\n",
        "expected_route": "math_v1",
        "markers": ["ROUTE:", "math"],
    },
    "route_agentic": {
        "target": "orchestrator_v1",
        "prompt": "<|route|>\nUSER_REQUEST: Create a step-by-step plan to evaluate the model.\n",
        "expected_route": "agentic_v1",
        "markers": ["ROUTE:", "agentic"],
    },
    "route_complex": {
        "target": "orchestrator_v1",
        "prompt": "<|route|>\nUSER_REQUEST: Explain the idea, write pseudocode, and solve a small equation.\n",
        "expected_route": "orchestrator_v1",
        "markers": ["ROUTE:", "orchestrator"],
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
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
        "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
    }
    return text, telemetry


def run_suite(args: argparse.Namespace) -> None:
    base_dir = args.base_dir
    out_dir = args.out_dir or (base_dir / "reports" / "swarmlm_v1_suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "eval_swarmlm_v1_suite.jsonl"
    report_path = out_dir / "SWARMLM_V1_EVAL_REPORT.md"
    manifest_path = out_dir / "manifest.json"
    raw_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    genome = args.genome or (base_dir / "zetagrid_25b_production.npy")

    manifest = {
        "base_dir": str(base_dir),
        "genome": str(genome),
        "genome_sha256": sha256_file(genome) if genome.exists() and args.hash_files else None,
        "device": device,
        "dtype": str(dtype),
        "max_new": args.max_new,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "souls": {},
    }

    rows = []
    load_rows = []
    for soul_name, rel_path in SOUL_FILES.items():
        ckpt = base_dir / rel_path
        if not ckpt.exists():
            print(f"[SKIP] missing {soul_name}: {ckpt}")
            continue

        print(f"\n===== LOAD {soul_name} =====")
        cleanup()
        load_t0 = time.time()
        bank = GenomeWeightBank(genome, dtype=dtype, device=device)
        model = ZetaGridSoul(bank, n_layers=args.layers, rank=args.rank, dtype=dtype).to(device)
        del bank.data
        del bank
        cleanup()
        step, loss = load_init_checkpoint(model, ckpt, device)
        load_elapsed = time.time() - load_t0
        load_row = {
            "soul": soul_name,
            "checkpoint": str(ckpt),
            "checkpoint_step": step,
            "checkpoint_loss": loss,
            "load_elapsed_sec": load_elapsed,
            "post_load_vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "checkpoint_sha256": sha256_file(ckpt) if args.hash_files else None,
        }
        load_rows.append(load_row)
        manifest["souls"][soul_name] = load_row

        for prompt_name, spec in PROMPTS.items():
            if args.target_only and spec["target"] != soul_name:
                continue
            print(f"--- {soul_name} / {prompt_name} ---")
            text, telemetry = generate(
                model,
                spec["prompt"],
                device=device,
                dtype=dtype,
                max_new=args.max_new,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            route = extract_route(text)
            expected_route = spec.get("expected_route")
            route_ok = (route == expected_route) if expected_route else None
            row = {
                "kind": "generation",
                "soul": soul_name,
                "prompt_name": prompt_name,
                "target_soul": spec["target"],
                "checkpoint": str(ckpt),
                "checkpoint_step": step,
                "checkpoint_loss": loss,
                "prompt": spec["prompt"],
                "output": text,
                "marker_score": marker_score(text, spec.get("markers", [])),
                "route": route,
                "expected_route": expected_route,
                "route_ok": route_ok,
                **telemetry,
            }
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(text[:500].replace("\n", "\\n"))

        del model
        cleanup()

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    route_rows = [r for r in rows if r["expected_route"] is not None]
    route_acc = None
    if route_rows:
        route_acc = sum(1 for r in route_rows if r["route_ok"]) / len(route_rows)

    target_rows = [r for r in rows if r["soul"] == r["target_soul"]]
    off_target_rows = [r for r in rows if r["soul"] != r["target_soul"]]
    avg_target_marker = statistics.mean([r["marker_score"] for r in target_rows]) if target_rows else 0.0
    avg_off_marker = statistics.mean([r["marker_score"] for r in off_target_rows]) if off_target_rows else 0.0
    avg_tps = statistics.mean([r["tokens_per_sec"] for r in rows]) if rows else 0.0
    peak_vram = max([r["vram_peak_gb"] for r in rows], default=0.0)

    report = [
        "# SwarmLM v1 Scientific Smoke Evaluation",
        "",
        "## Scope",
        "",
        "This suite evaluates Soul differentiation, SwarmLM routing behavior, and runtime telemetry over a shared frozen Genome.",
        "",
        "## Artifacts",
        "",
        f"- Raw JSONL: `{raw_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Genome: `{genome}`",
        "",
        "## Summary Metrics",
        "",
        f"- Generation rows: {len(rows)}",
        f"- Target-only marker score average: {avg_target_marker:.3f}",
        f"- Off-target marker score average: {avg_off_marker:.3f}",
        f"- Route accuracy: {route_acc:.3f}" if route_acc is not None else "- Route accuracy: n/a",
        f"- Average tokens/sec: {avg_tps:.2f}",
        f"- Peak eval VRAM: {peak_vram:.2f} GB",
        "",
        "## Soul Checkpoints",
        "",
    ]
    for row in load_rows:
        report.append(
            f"- {row['soul']}: `{row['checkpoint']}` step={row['checkpoint_step']} loss={row['checkpoint_loss']:.4f} load={row['load_elapsed_sec']:.1f}s"
        )

    report.extend(["", "## Route Tests", ""])
    for row in route_rows:
        report.append(
            f"- {row['soul']} / {row['prompt_name']}: expected={row['expected_route']} got={row['route']} ok={row['route_ok']}"
        )

    report.extend(["", "## Interpretation Template", ""])
    report.extend(
        [
            "- If target marker score exceeds off-target marker score, same-Genome/different-Soul specialization is supported.",
            "- If route accuracy is high for orchestrator_v1, SwarmLM routing behavior is supported.",
            "- Failures should be preserved as evidence for the next alignment dataset revision.",
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
    parser.add_argument("--max_new", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target_only", action="store_true")
    parser.add_argument("--hash_files", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
