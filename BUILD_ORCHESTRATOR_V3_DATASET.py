#!/usr/bin/env python3
"""
Build a focused Orchestrator v3 routing dataset for SwarmLM.

v3 targets the observed SwarmLM v2 cascade failures:
- code-like requests such as primality tests being routed to text_v2
- FRO/text explanation distinction
- Italian route prompts
- debugging, SQL, regex, parser, and implementation requests
- multi-capability requests that should remain with the orchestrator

This dataset trains only the router. It should be used to continue from
ORCHESTRATOR_V2.pt, not to retrain all specialist Souls.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


CODE_REQUESTS = [
    "Write a Python function for fibonacci.",
    "Write a Python primality test.",
    "Implement is_prime in Python.",
    "Complete a function that checks whether a number is prime.",
    "Write code to parse a ROUTE field from text.",
    "Implement a regex extractor in Python.",
    "Write a JSONL writer function.",
    "Create a Python parser for log lines.",
    "Debug this traceback and suggest a code fix.",
    "Fix this Python function that returns the wrong value.",
    "Write SQL to count users by country.",
    "Implement a small HTTP client in Python.",
    "Write pseudocode for a sorting algorithm.",
    "Create a function that validates an email address.",
    "Implement a command line argument parser.",
    "Scrivi una funzione Python per Fibonacci.",
    "Scrivi un test di primalita' in Python.",
    "Correggi questo bug nel codice Python.",
    "Implementa un parser JSONL.",
    "Genera codice Python per leggere un file.",
]


MATH_REQUESTS = [
    "Solve 3x + 5 = 20.",
    "A train travels 120 km in 2 hours. What is the average speed?",
    "If 5a = 45, what is a?",
    "What is 15 percent of 200?",
    "Solve 2y - 4 = 10.",
    "Compute the area of a rectangle with width 4 and height 7.",
    "Find x when 4x = 28.",
    "Calculate 18 divided by 3.",
    "What is the mean of 4, 8, and 12?",
    "A product costs 80 and has a 25 percent discount. What is the final price?",
    "Risolvi 3x + 5 = 20.",
    "Calcola la velocita' media: 120 km in 2 ore.",
    "Quanto fa il 15 percento di 200?",
]


TEXT_REQUESTS = [
    "Explain the Genome/Soul architecture in simple English.",
    "Summarize Fractal Resonant Optimization in simple English.",
    "Explain FRO without equations.",
    "Describe why a shared frozen Genome is useful.",
    "Explain why SwarmLM needs an orchestrator.",
    "Summarize the cascade evaluation result.",
    "Write a short paragraph about modular specialization.",
    "Explain topology-preserving adaptation in plain language.",
    "What does target marker score mean?",
    "Explain the limitation of this experiment.",
    "Spiega in italiano Genome e Soul.",
    "Spiega FRO in italiano semplice.",
    "Riassumi SwarmLM v2 in tre frasi.",
]


AGENTIC_REQUESTS = [
    "Create a step-by-step plan to evaluate the model.",
    "Plan the next experiment.",
    "Design an adversarial test plan for agentic_v2.",
    "Plan the SPRIND submission tasks.",
    "Create a checklist for uploading Souls to Hugging Face.",
    "Plan a controlled benchmark run.",
    "Create a safety evaluation plan for an agentic Soul.",
    "Break this research roadmap into steps.",
    "Define a staged validation plan before production exposure.",
    "Prepara un piano di test per orchestrator_v3.",
    "Crea una checklist per la valutazione scientifica.",
]


ORCHESTRATOR_REQUESTS = [
    "Explain the idea, write pseudocode, and solve a small equation.",
    "Summarize SwarmLM, generate pseudocode, and solve 3x + 5 = 20.",
    "Write code, solve a math example, and explain the architecture.",
    "Plan an experiment, then route code and math subtasks.",
    "Create a multi-Soul workflow for text, code, and math.",
    "Split this task into text, code, math, and agentic parts.",
    "Explain FRO, write pseudocode, and propose an evaluation plan.",
    "Give an Italian explanation, a Python sketch, and a small equation solution.",
    "Design a cascade where different Souls handle different parts.",
    "Scomponi il task in spiegazione, codice, matematica e piano operativo.",
]


CONFUSION_PAIRS = [
    ("Explain what a primality test is, no code.", "ROUTE: text_v2\nREASON: The request asks for natural-language explanation, not code generation."),
    ("Write a primality test in Python.", "ROUTE: code_v2\nREASON: The request asks for code generation."),
    ("Explain SQL joins conceptually.", "ROUTE: text_v2\nREASON: The request asks for natural-language explanation."),
    ("Write an SQL query using a join.", "ROUTE: code_v2\nREASON: The request asks for query/code generation."),
    ("Explain FRO in simple English.", "ROUTE: text_v2\nREASON: The request asks for natural-language explanation."),
    ("Plan adversarial tests for FRO training.", "ROUTE: agentic_v1\nREASON: The request asks for planning and task decomposition."),
    ("Solve a prime factorization problem.", "ROUTE: math_v1\nREASON: The request asks for mathematical reasoning."),
    ("Implement prime factorization in Python.", "ROUTE: code_v2\nREASON: The request asks for software implementation."),
    ("Explain a Python traceback in words.", "ROUTE: text_v2\nREASON: The request asks for explanation."),
    ("Fix this Python traceback.", "ROUTE: code_v2\nREASON: The request asks for debugging and code repair."),
]


def route_record(request: str, route: str, reason: str) -> str:
    return f"\n<|route|>\nUSER_REQUEST: {request}\nROUTE: {route}\nREASON: {reason}\n<|endroute|>\n"


def build_records() -> list[str]:
    records: list[str] = []
    for req in CODE_REQUESTS:
        records.append(route_record(req, "code_v2", "The request asks for code generation, debugging, parsing, SQL, regex, or software implementation."))
    for req in MATH_REQUESTS:
        records.append(route_record(req, "math_v1", "The request asks for numeric calculation, algebraic reasoning, or mathematical problem solving."))
    for req in TEXT_REQUESTS:
        records.append(route_record(req, "text_v2", "The request asks for natural-language explanation or summarization."))
    for req in AGENTIC_REQUESTS:
        records.append(route_record(req, "agentic_v1", "The request asks for planning, task decomposition, evaluation design, or a checklist."))
    for req in ORCHESTRATOR_REQUESTS:
        records.append(route_record(req, "orchestrator_v1", "The request combines multiple capabilities and should be split across Souls."))
    for req, answer in CONFUSION_PAIRS:
        records.append(f"\n<|route|>\nUSER_REQUEST: {req}\n{answer}\n<|endroute|>\n")
    return records


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
    print(f"[DONE] {path} {path.stat().st_size / 1024**2:.1f} MB records={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--target_mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    records = build_records()
    out = args.base_dir / "data" / "swarmlm_v3" / "orchestrator_v3.bin"
    repeat_write(out, records, args.target_mb * 1024**2, args.seed)


if __name__ == "__main__":
    main()
