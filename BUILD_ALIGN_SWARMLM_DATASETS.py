#!/usr/bin/env python3
"""
Build small byte-level alignment datasets for Soul v2 and SwarmLM smoke runs.

These datasets are deliberately compact and controlled. They are intended to
teach response format, routing, task decomposition, and short task behavior, not
world knowledge.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


TEXT_ALIGN = [
    (
        "Explain in Italian what a fractal language model is.",
        "Un modello linguistico frattale e' un sistema che riusa strutture a piu' scale: pattern locali, sequenze piu' lunghe e memoria globale cooperano per generare testo coerente.",
    ),
    (
        "Summarize Fractal Resonant Optimization in simple English.",
        "Fractal Resonant Optimization tracks whether gradients keep pointing in a coherent direction across multiple time scales. When the signal is coherent it allows learning; when the signal is noisy it damps the update.",
    ),
    (
        "Write a concise project status update.",
        "Status: the shared Genome loads correctly, specialized Souls produce distinct behavior, and the next step is low-learning-rate alignment followed by fixed-prompt evaluation.",
    ),
    (
        "Spiega perche' separare Genome e Soul e' utile.",
        "Separare Genome e Soul permette di mantenere una base comune stabile e addestrare competenze diverse come moduli intercambiabili. Questo riduce il costo di sperimentazione e rende il sistema piu' componibile.",
    ),
]

CODE_ALIGN = [
    (
        "Complete a Python fibonacci function.",
        "def fibonacci(n):\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    ),
    (
        "Complete a Python primality test.",
        "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    d = 3\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 2\n    return True\n",
    ),
    (
        "Write a small JSONL logger.",
        "import json\n\ndef write_jsonl(path, row):\n    with open(path, \"a\", encoding=\"utf-8\") as f:\n        f.write(json.dumps(row, ensure_ascii=False) + \"\\n\")\n",
    ),
    (
        "Write a route parser.",
        "def parse_route(text):\n    for line in text.splitlines():\n        if line.startswith(\"ROUTE:\"):\n            return line.split(\":\", 1)[1].strip()\n    return \"text_v2\"\n",
    ),
]

MATH_ALIGN = [
    (
        "If 3x + 5 = 20, solve for x.",
        "Subtract 5 from both sides: 3x = 15. Divide by 3: x = 5.",
    ),
    (
        "A train travels 120 km in 2 hours. What is the average speed?",
        "Average speed is distance divided by time: 120 / 2 = 60 km/h.",
    ),
    (
        "If a rectangle has width 4 and height 7, what is its area?",
        "Area equals width times height: 4 * 7 = 28 square units.",
    ),
    (
        "Solve 2y - 4 = 10.",
        "Add 4 to both sides: 2y = 14. Divide by 2: y = 7.",
    ),
]

INSTRUCTION = [
    (
        "Give a direct answer and then one short explanation.",
        "Answer: Use the Math Soul.\nExplanation: The request asks for a numeric solution, so mathematical reasoning is the main capability.",
    ),
    (
        "Reply in Italian with three concise bullets.",
        "- Genome: base comune congelata.\n- Soul: competenza modulare addestrabile.\n- SwarmLM: coordinamento tra Soul specializzate.",
    ),
    (
        "Follow the requested format exactly: RESULT then WHY.",
        "RESULT: The experiment validates Soul differentiation.\nWHY: The same Genome produced text-like, code-like, and math-like outputs when different Souls were loaded.",
    ),
]

AGENTIC = [
    (
        "Plan a Soul evaluation run.",
        "Plan:\n1. Load the shared Genome once per Soul.\n2. Load the target Soul checkpoint.\n3. Run fixed prompts for text, code, math, and routing.\n4. Save outputs to JSONL.\n5. Compare behavior and failure modes.",
    ),
    (
        "Break down the task: build a code dataset and train Code v2.",
        "Plan:\n1. Create a permissive code byte dataset.\n2. Verify file size and path.\n3. Train Code v2 with FRO at low learning rate.\n4. Save one checkpoint only.\n5. Run code-format evaluation prompts.",
    ),
    (
        "Plan the next SPRIND demo.",
        "Plan:\n1. State the Genome/Soul hypothesis.\n2. Show three specialized Souls on one Genome.\n3. Run the Orchestrator routing example.\n4. Show logs, checkpoints, and failure cases.\n5. Explain the next scaling step.",
    ),
]

ORCHESTRATOR = [
    ("Explain FRO in simple terms.", "ROUTE: text_v2\nREASON: The user asks for a natural-language explanation."),
    ("Write a Python function for fibonacci.", "ROUTE: code_v2\nREASON: The user asks for code generation."),
    ("Solve 3x + 5 = 20.", "ROUTE: math_v1\nREASON: The user asks for algebraic reasoning."),
    ("Create a step-by-step plan to evaluate the model.", "ROUTE: agentic_v1\nREASON: The user asks for planning and task decomposition."),
    ("Explain the idea, write pseudocode, and solve a small equation.", "ROUTE: orchestrator_v1\nREASON: The user asks for a multi-Soul task that should be split across capabilities."),
]


def format_pair(kind: str, instruction: str, answer: str) -> str:
    if kind == "code_align_v1":
        return f"\n<|file|> language=python task=alignment\n# Instruction: {instruction}\n{answer}<|endfile|>\n"
    if kind == "math_align_v1":
        return f"\n<|math|>\nProblem:\n{instruction}\n\nSolution:\n{answer}\n<|endmath|>\n"
    if kind == "orchestrator_v1":
        return f"\n<|route|>\nUSER_REQUEST: {instruction}\n{answer}\n<|endroute|>\n"
    if kind == "agentic_v1":
        return f"\n<|agentic|>\nTask: {instruction}\n{answer}\n<|endagentic|>\n"
    return f"\n<|instruction|>\nUser: {instruction}\nAssistant: {answer}\n<|endinstruction|>\n"


def write_bin(path: Path, kind: str, pairs: list[tuple[str, str]], target_bytes: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [format_pair(kind, instruction, answer) for instruction, answer in pairs]
    written = 0
    with path.open("wb") as f:
        while written < target_bytes:
            rng.shuffle(records)
            for record in records:
                b = record.encode("utf-8", errors="ignore")
                n = min(len(b), target_bytes - written)
                f.write(b[:n])
                written += n
                if written >= target_bytes:
                    break
    print(f"[DONE] {path} {path.stat().st_size / 1024**2:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--target_mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024**2
    jobs = {
        "align_v1/text_align_v1.bin": ("text_align_v1", TEXT_ALIGN),
        "align_v1/code_align_v1.bin": ("code_align_v1", CODE_ALIGN),
        "align_v1/math_align_v1.bin": ("math_align_v1", MATH_ALIGN),
        "swarmlm_v1/instruction_v1.bin": ("instruction_v1", INSTRUCTION),
        "swarmlm_v1/agentic_v1.bin": ("agentic_v1", AGENTIC),
        "swarmlm_v1/orchestrator_v1.bin": ("orchestrator_v1", ORCHESTRATOR),
    }
    for rel_path, (kind, pairs) in jobs.items():
        write_bin(args.base_dir / "data" / rel_path, kind, pairs, target_bytes, args.seed)


if __name__ == "__main__":
    main()
