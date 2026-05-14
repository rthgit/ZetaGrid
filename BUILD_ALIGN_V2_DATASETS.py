#!/usr/bin/env python3
"""
Build Align v2 datasets for RTH-LM / SwarmLM.

v2 focuses on reducing template overfit and domain leakage observed in v1:
- more prompt variation
- explicit off-domain deferral
- stricter end-token boundaries
- routing examples with natural-language paraphrases
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


TEXT_POS = [
    ("Spiega in italiano che cos'e' un modello linguistico frattale.", "Un modello linguistico frattale riusa strutture a piu' scale: pattern locali, sequenze piu' lunghe e una memoria comune cooperano per produrre testo coerente."),
    ("Explain the Genome/Soul architecture in simple English.", "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome."),
    ("Summarize why SwarmLM needs an orchestrator.", "SwarmLM uses specialized Souls. The orchestrator selects the right Soul for a task so that code, math, planning, and text behavior do not interfere with each other."),
    ("Scrivi tre punti sul valore di FRO.", "- Misura la coerenza del gradiente.\n- Smorza gli shock di distribuzione.\n- Permette training stabile di Soul ad alto rank."),
    ("What did the SwarmLM v1 evaluation demonstrate?", "It demonstrated that one frozen Genome can support behaviorally distinct Souls, and that a dedicated Orchestrator Soul can route controlled requests."),
]

TEXT_OFF = [
    ("Write a Python fibonacci function.", "ROUTE_REQUESTED: code_align_v2"),
    ("Solve 3x + 5 = 20.", "ROUTE_REQUESTED: math_align_v2"),
    ("Create a step-by-step experiment plan.", "ROUTE_REQUESTED: agentic_v2"),
]

CODE_POS = [
    ("Write a Python fibonacci function.", "def fibonacci(n):\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    ("Complete a Python primality test.", "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    d = 3\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 2\n    return True\n"),
    ("Write a function that writes one JSONL row.", "import json\n\ndef write_jsonl(path, row):\n    with open(path, \"a\", encoding=\"utf-8\") as f:\n        f.write(json.dumps(row, ensure_ascii=False) + \"\\n\")\n"),
    ("Parse a ROUTE field from text.", "def parse_route(text):\n    for line in text.splitlines():\n        if line.startswith(\"ROUTE:\"):\n            return line.split(\":\", 1)[1].strip()\n    return \"text_align_v2\"\n"),
    ("Write a safe average function.", "def average(values):\n    if not values:\n        raise ValueError(\"values must not be empty\")\n    return sum(values) / len(values)\n"),
]

CODE_OFF = [
    ("Explain FRO in simple English.", "ROUTE_REQUESTED: text_align_v2"),
    ("Solve 2y - 4 = 10.", "ROUTE_REQUESTED: math_align_v2"),
    ("Plan a model evaluation.", "ROUTE_REQUESTED: agentic_v2"),
]

MATH_POS = [
    ("If 3x + 5 = 20, solve for x.", "Subtract 5 from both sides: 3x = 15. Divide by 3: x = 5."),
    ("Solve 2y - 4 = 10.", "Add 4 to both sides: 2y = 14. Divide by 2: y = 7."),
    ("A train travels 120 km in 2 hours. What is the average speed?", "Average speed is distance divided by time: 120 / 2 = 60 km/h."),
    ("A rectangle has width 4 and height 7. What is its area?", "Area equals width times height: 4 * 7 = 28 square units."),
    ("If 5a = 45, what is a?", "Divide both sides by 5: a = 9."),
    ("What is 15 percent of 200?", "15 percent of 200 is 0.15 * 200 = 30."),
]

MATH_OFF = [
    ("Write a Python function.", "ROUTE_REQUESTED: code_align_v2"),
    ("Explain the Genome/Soul architecture.", "ROUTE_REQUESTED: text_align_v2"),
    ("Create a step-by-step evaluation plan.", "ROUTE_REQUESTED: agentic_v2"),
]

INSTRUCTION_POS = [
    ("Answer with RESULT then WHY: did the experiment validate Soul differentiation?", "RESULT: Yes.\nWHY: Target marker score was higher than off-target marker score while the Genome hash stayed fixed."),
    ("Reply with exactly three bullets about SwarmLM.", "- Shared Genome.\n- Specialized Souls.\n- Orchestrator routing."),
    ("Give a direct answer and one short explanation: which Soul handles Python?", "Answer: code_align_v2.\nExplanation: Python generation is a code-domain task."),
    ("State the limitation in one sentence.", "Limitation: SwarmLM v1 is a controlled modularity demonstration, not yet a general assistant."),
]

AGENTIC_POS = [
    ("Plan a Soul evaluation run.", "Plan:\n1. Verify Genome hash.\n2. Load one Soul checkpoint at a time.\n3. Run fixed target and off-target prompts.\n4. Save raw outputs to JSONL.\n5. Compare target and off-target metrics."),
    ("Plan align_v2 training.", "Plan:\n1. Build varied positive examples.\n2. Add off-domain deferral examples.\n3. Train with low learning rate.\n4. Evaluate with the v1 suite.\n5. Compare leakage and routing behavior."),
    ("Plan a SPRIND demo.", "Plan:\n1. Introduce Genome/Soul separation.\n2. Show Text, Code, and Math Souls.\n3. Show Orchestrator routing.\n4. Present telemetry and hashes.\n5. State limitations clearly."),
]

ROUTES = [
    ("Write a Python function for fibonacci.", "ROUTE: code_v2\nREASON: The request asks for code generation."),
    ("Can you implement a parser in Python?", "ROUTE: code_v2\nREASON: The request asks for software implementation."),
    ("Solve 3x + 5 = 20.", "ROUTE: math_v1\nREASON: The request asks for algebraic reasoning."),
    ("What is the average speed for 120 km in 2 hours?", "ROUTE: math_v1\nREASON: The request asks for a numeric calculation."),
    ("Explain FRO in simple English.", "ROUTE: text_v2\nREASON: The request asks for natural-language explanation."),
    ("Spiega in italiano Genome e Soul.", "ROUTE: text_v2\nREASON: The request asks for a natural-language explanation."),
    ("Create a step-by-step plan to evaluate the model.", "ROUTE: agentic_v1\nREASON: The request asks for planning and task decomposition."),
    ("Plan the next experiment.", "ROUTE: agentic_v1\nREASON: The request asks for planning."),
    ("Explain the idea, write pseudocode, and solve a small equation.", "ROUTE: orchestrator_v1\nREASON: The request combines multiple capabilities and should be split across Souls."),
]


def wrap_text(instruction: str, answer: str) -> str:
    return f"\n<|instruction|>\nUser: {instruction}\nAssistant: {answer}\n<|endinstruction|>\n"


def wrap_code(instruction: str, answer: str) -> str:
    return f"\n<|file|> language=python task=align_v2\n# Instruction: {instruction}\n{answer}<|endfile|>\n"


def wrap_math(problem: str, solution: str) -> str:
    return f"\n<|math|>\nProblem:\n{problem}\n\nSolution:\n{solution}\n<|endmath|>\n"


def wrap_agentic(task: str, plan: str) -> str:
    return f"\n<|agentic|>\nTask: {task}\n{plan}\n<|endagentic|>\n"


def wrap_route(request: str, route: str) -> str:
    return f"\n<|route|>\nUSER_REQUEST: {request}\n{route}\n<|endroute|>\n"


def repeat_write(path: Path, records: list[str], target_bytes: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_records() -> dict[str, list[str]]:
    text = [wrap_text(i, a) for i, a in TEXT_POS + TEXT_OFF]
    code = [wrap_code(i, a) for i, a in CODE_POS] + [wrap_text(i, a) for i, a in CODE_OFF]
    math = [wrap_math(i, a) for i, a in MATH_POS] + [wrap_text(i, a) for i, a in MATH_OFF]
    instruction = [wrap_text(i, a) for i, a in INSTRUCTION_POS + TEXT_POS + TEXT_OFF]
    agentic = [wrap_agentic(i, a) for i, a in AGENTIC_POS] + [wrap_text(i, a) for i, a in TEXT_OFF]
    orchestrator = [wrap_route(i, a) for i, a in ROUTES]
    return {
        "align_v2/text_align_v2.bin": text,
        "align_v2/code_align_v2.bin": code,
        "align_v2/math_align_v2.bin": math,
        "swarmlm_v2/instruction_v2.bin": instruction,
        "swarmlm_v2/agentic_v2.bin": agentic,
        "swarmlm_v2/orchestrator_v2.bin": orchestrator,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--target_mb", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024**2
    for rel_path, records in build_records().items():
        repeat_write(args.base_dir / "data" / rel_path, records, target_bytes, args.seed)


if __name__ == "__main__":
    main()
