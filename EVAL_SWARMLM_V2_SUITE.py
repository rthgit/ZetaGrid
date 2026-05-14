#!/usr/bin/env python3
"""
Scientific v2 evaluation suite for RTH-LM / SwarmLM.

This is separate from the v1 suite so published v1 artifacts remain stable.
It evaluates:
- same Genome with six v2 Souls
- target-domain specialization
- off-domain leakage
- ROUTE_REQUESTED behavior for non-orchestrator Souls
- orchestrator route accuracy
- telemetry, checkpoint metadata, hashes, and reproducible JSONL/Markdown reports
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
    "text_align_v2": "checkpoints/text_align_v2/TEXT_ALIGN_V2.pt",
    "code_align_v2": "checkpoints/code_align_v2/CODE_ALIGN_V2.pt",
    "math_align_v2": "checkpoints/math_align_v2/MATH_ALIGN_V2.pt",
    "instruction_v2": "checkpoints/instruction_v2/INSTRUCTION_V2.pt",
    "agentic_v2": "checkpoints/agentic_v2/AGENTIC_V2.pt",
    "orchestrator_v2": "checkpoints/orchestrator_v2/ORCHESTRATOR_V2.pt",
}

ROUTE_PROMPTS = {
    "route_code": {
        "target": "orchestrator_v2",
        "family": "route",
        "prompt": "<|route|>\nUSER_REQUEST: Write a Python function for fibonacci.\n",
        "expected_route": "code_v2",
        "markers": ["ROUTE:", "code_v2"],
    },
    "route_math": {
        "target": "orchestrator_v2",
        "family": "route",
        "prompt": "<|route|>\nUSER_REQUEST: Solve 3x + 5 = 20.\n",
        "expected_route": "math_v1",
        "markers": ["ROUTE:", "math_v1"],
    },
    "route_agentic": {
        "target": "orchestrator_v2",
        "family": "route",
        "prompt": "<|route|>\nUSER_REQUEST: Create a step-by-step plan to evaluate the model.\n",
        "expected_route": "agentic_v1",
        "markers": ["ROUTE:", "agentic_v1"],
    },
    "route_complex": {
        "target": "orchestrator_v2",
        "family": "route",
        "prompt": "<|route|>\nUSER_REQUEST: Explain the idea, write pseudocode, and solve a small equation.\n",
        "expected_route": "orchestrator_v1",
        "markers": ["ROUTE:", "orchestrator_v1"],
    },
}

PROMPTS = {
    "text_domain": {
        "target": "text_align_v2",
        "family": "text",
        "prompt": "<|instruction|>\nUser: Explain the Genome/Soul architecture in simple English.\nAssistant:",
        "markers": ["Genome", "Soul", "shared"],
    },
    "text_italian": {
        "target": "text_align_v2",
        "family": "text",
        "prompt": "<|instruction|>\nUser: Spiega in italiano perche' separare Genome e Soul e' utile.\nAssistant:",
        "markers": ["Genome", "Soul", "base"],
    },
    "code_fibonacci": {
        "target": "code_align_v2",
        "family": "code",
        "prompt": "<|file|> language=python task=alignment\n# Instruction: Complete a Python fibonacci function.\ndef fibonacci(n):\n",
        "markers": ["return", "for", "range"],
    },
    "code_prime": {
        "target": "code_align_v2",
        "family": "code",
        "prompt": "<|file|> language=python task=alignment\n# Instruction: Complete a Python primality test.\ndef is_prime(n):\n    if n < 2:\n",
        "markers": ["return False", "while", "%"],
    },
    "math_linear": {
        "target": "math_align_v2",
        "family": "math",
        "prompt": "<|math|>\nProblem:\nIf 3x + 5 = 20, solve for x.\n\nSolution:\n",
        "markers": ["3x", "15", "x = 5"],
    },
    "math_speed": {
        "target": "math_align_v2",
        "family": "math",
        "prompt": "<|math|>\nProblem:\nA train travels 120 km in 2 hours. What is the average speed?\n\nSolution:\n",
        "markers": ["120", "2", "60"],
    },
    "instruction_format": {
        "target": "instruction_v2",
        "family": "instruction",
        "prompt": "<|instruction|>\nUser: Reply exactly with RESULT then WHY. Explain that the modular Soul test succeeded.\nAssistant:",
        "markers": ["RESULT", "WHY"],
    },
    "agentic_plan": {
        "target": "agentic_v2",
        "family": "agentic",
        "prompt": "<|agentic|>\nTask: Plan a scientific Soul evaluation run.\nPlan:\n",
        "markers": ["1.", "2.", "JSONL"],
    },
    **ROUTE_PROMPTS,
}

EXPECTED_ROUTE_REQUESTED = {
    ("text_align_v2", "code_fibonacci"): "code_align_v2",
    ("text_align_v2", "math_linear"): "math_align_v2",
    ("text_align_v2", "agentic_plan"): "agentic_v2",
    ("instruction_v2", "code_fibonacci"): "code_align_v2",
    ("instruction_v2", "math_linear"): "math_align_v2",
    ("agentic_v2", "code_fibonacci"): "code_align_v2",
    ("agentic_v2", "math_linear"): "math_align_v2",
}

FAMILY_MARKERS = {
    "text": ["Genome", "Soul", "shared", "architecture", "modular"],
    "code": ["def ", "return", "for ", "while ", "if ", "%"],
    "math": ["Solution", "x =", "60", "3x", "Divide", "Subtract"],
    "agentic": ["Plan:", "1.", "2.", "step", "evaluate"],
    "route": ["ROUTE:", "REASON:", "code_v2", "math_v1", "agentic_v1", "orchestrator_v1"],
}


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


def extract_route_requested(text: str) -> str:
    match = re.search(r"ROUTE_REQUESTED:\s*([A-Za-z0-9_\-]+)", text)
    return match.group(1).strip() if match else ""


def marker_score(text: str, markers: list[str]) -> float:
    if not markers:
        return 0.0
    lower = text.lower()
    return sum(1 for marker in markers if marker.lower() in lower) / len(markers)


def family_scores(text: str) -> dict[str, float]:
    return {family: marker_score(text, markers) for family, markers in FAMILY_MARKERS.items()}


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


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def write_reports(
    rows: list[dict],
    load_rows: list[dict],
    manifest: dict,
    out_dir: Path,
    raw_path: Path,
    manifest_path: Path,
    genome: Path,
) -> None:
    route_rows = [r for r in rows if r["expected_route"] is not None]
    orchestrator_route_rows = [r for r in route_rows if r["soul"] == "orchestrator_v2"]
    non_orchestrator_route_rows = [r for r in route_rows if r["soul"] != "orchestrator_v2"]
    route_requested_rows = [r for r in rows if r["expected_route_requested"] is not None]

    target_rows = [r for r in rows if r["soul"] == r["target_soul"]]
    off_target_rows = [r for r in rows if r["soul"] != r["target_soul"]]
    leakage_rows = [r for r in rows if r["soul_family"] != r["prompt_family"]]

    metrics = {
        "generation_rows": len(rows),
        "target_marker_score_avg": mean([r["marker_score"] for r in target_rows]),
        "off_target_marker_score_avg": mean([r["marker_score"] for r in off_target_rows]),
        "off_target_leakage_score_avg": mean([r["prompt_family_score"] for r in leakage_rows]),
        "route_accuracy_global": mean([1.0 if r["route_ok"] else 0.0 for r in route_rows]),
        "route_accuracy_orchestrator": mean([1.0 if r["route_ok"] else 0.0 for r in orchestrator_route_rows]),
        "route_accuracy_non_orchestrator": mean([1.0 if r["route_ok"] else 0.0 for r in non_orchestrator_route_rows]),
        "route_requested_accuracy": mean([1.0 if r["route_requested_ok"] else 0.0 for r in route_requested_rows]),
        "average_tokens_per_sec": mean([r["tokens_per_sec"] for r in rows]),
        "peak_eval_vram_gb": max([r["vram_peak_gb"] for r in rows], default=0.0),
    }
    manifest["summary_metrics"] = metrics

    report_path = out_dir / "SWARMLM_V2_EVAL_REPORT.md"
    interpreted_path = out_dir / "SWARMLM_V2_EVAL_REPORT_INTERPRETED.md"

    report = [
        "# SwarmLM v2 Scientific Evaluation",
        "",
        "## Scope",
        "",
        "This suite evaluates Align v2 and SwarmLM v2 Souls over a shared frozen Genome.",
        "",
        "## Artifacts",
        "",
        f"- Raw JSONL: `{raw_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Genome: `{genome}`",
        "",
        "## Summary Metrics",
        "",
        f"- Generation rows: {metrics['generation_rows']}",
        f"- Target-only marker score average: {metrics['target_marker_score_avg']:.3f}",
        f"- Off-target marker score average: {metrics['off_target_marker_score_avg']:.3f}",
        f"- Off-target leakage score average: {metrics['off_target_leakage_score_avg']:.3f}",
        f"- Global route accuracy: {metrics['route_accuracy_global']:.3f}",
        f"- Orchestrator-only route accuracy: {metrics['route_accuracy_orchestrator']:.3f}",
        f"- Non-orchestrator route accuracy: {metrics['route_accuracy_non_orchestrator']:.3f}",
        f"- ROUTE_REQUESTED accuracy: {metrics['route_requested_accuracy']:.3f}",
        f"- Average tokens/sec: {metrics['average_tokens_per_sec']:.2f}",
        f"- Peak eval VRAM: {metrics['peak_eval_vram_gb']:.2f} GB",
        "",
        "## Soul Checkpoints",
        "",
    ]
    for row in load_rows:
        report.append(
            f"- {row['soul']}: `{row['checkpoint']}` step={row['checkpoint_step']} "
            f"loss={row['checkpoint_loss']:.4f} load={row['load_elapsed_sec']:.1f}s"
        )

    report.extend(["", "## Orchestrator Route Tests", ""])
    for row in orchestrator_route_rows:
        report.append(
            f"- {row['prompt_name']}: expected={row['expected_route']} got={row['route']} ok={row['route_ok']}"
        )

    report.extend(["", "## ROUTE_REQUESTED Tests", ""])
    for row in route_requested_rows:
        report.append(
            f"- {row['soul']} / {row['prompt_name']}: expected={row['expected_route_requested']} "
            f"got={row['route_requested']} ok={row['route_requested_ok']}"
        )

    report.extend(["", "## Interpretation Guardrails", ""])
    report.extend(
        [
            "- This suite is a scientific smoke evaluation, not a full downstream benchmark.",
            "- Strong route accuracy supports routing specialization only for orchestrator_v2.",
            "- ROUTE_REQUESTED behavior supports off-domain delegation discipline.",
            "- Leakage scores should be compared against v1 before making claims about improvement.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    interpreted = [
        "# SwarmLM v2 Evaluation - Interpreted Results",
        "",
        "## Core Result",
        "",
        "The suite tests whether Align v2 improves specialization and off-domain discipline while preserving SwarmLM routing.",
        "",
        "## Evidence",
        "",
        f"- Target marker score: {metrics['target_marker_score_avg']:.3f}",
        f"- Off-target marker score: {metrics['off_target_marker_score_avg']:.3f}",
        f"- Off-target leakage score: {metrics['off_target_leakage_score_avg']:.3f}",
        f"- Orchestrator-only route accuracy: {metrics['route_accuracy_orchestrator']:.3f}",
        f"- ROUTE_REQUESTED accuracy: {metrics['route_requested_accuracy']:.3f}",
        "",
        "## Scientific Interpretation",
        "",
        "A successful v2 run should show three properties: domain specialization, controlled delegation outside each Soul domain, and high orchestrator routing accuracy.",
        "",
        "## Claim Boundary",
        "",
        "This supports a modular Genome/Soul/SwarmLM architecture. It does not by itself prove general assistant quality, broad benchmark superiority, or optimizer superiority over AdamW.",
    ]
    interpreted_path.write_text("\n".join(interpreted) + "\n", encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] raw={raw_path}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] report={report_path}")
    print(f"[DONE] interpreted={interpreted_path}")


def run_suite(args: argparse.Namespace) -> None:
    base_dir = args.base_dir
    out_dir = args.out_dir or (base_dir / "reports" / "swarmlm_v2_suite_hashed")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "eval_swarmlm_v2_suite.jsonl"
    manifest_path = out_dir / "manifest.json"
    raw_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    genome = args.genome or (base_dir / "zetagrid_25b_production.npy")

    manifest = {
        "suite": "swarmlm_v2",
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
    soul_family = {
        "text_align_v2": "text",
        "code_align_v2": "code",
        "math_align_v2": "math",
        "instruction_v2": "instruction",
        "agentic_v2": "agentic",
        "orchestrator_v2": "route",
    }

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
            scores = family_scores(text)
            route = extract_route(text)
            route_requested = extract_route_requested(text)
            expected_route = spec.get("expected_route")
            expected_route_requested = EXPECTED_ROUTE_REQUESTED.get((soul_name, prompt_name))
            row = {
                "kind": "generation",
                "soul": soul_name,
                "soul_family": soul_family[soul_name],
                "prompt_name": prompt_name,
                "prompt_family": spec.get("family", "route"),
                "target_soul": spec["target"],
                "checkpoint": str(ckpt),
                "checkpoint_step": step,
                "checkpoint_loss": loss,
                "prompt": spec["prompt"],
                "output": text,
                "marker_score": marker_score(text, spec.get("markers", [])),
                "family_scores": scores,
                "prompt_family_score": scores.get(spec.get("family", "route"), 0.0),
                "route": route,
                "expected_route": expected_route,
                "route_ok": (route == expected_route) if expected_route else None,
                "route_requested": route_requested,
                "expected_route_requested": expected_route_requested,
                "route_requested_ok": (route_requested == expected_route_requested) if expected_route_requested else None,
                **telemetry,
            }
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(text[:500].replace("\n", "\\n"))

        del model
        cleanup()

    write_reports(rows, load_rows, manifest, out_dir, raw_path, manifest_path, genome)


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
