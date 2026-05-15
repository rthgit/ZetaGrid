#!/usr/bin/env python3
"""
Build an Orchestrator v3b hard-negative dataset.

v3 fixed the main cascade failure (primality code -> code_v2) but introduced
two new route confusions:
- "Explain what a primality test is, no code." -> math_v1 instead of text_v2
- "Write SQL to count users by country." -> orchestrator_v1 instead of code_v2

v3b is intentionally small and conservative. It should continue from
ORCHESTRATOR_V3.pt with a lower learning rate.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


CONFUSION_RECORDS = [
    ("Explain what a primality test is, no code.", "text_v2", "The request asks for a natural-language explanation, not code or math solving."),
    ("Describe what a prime number is, no calculation.", "text_v2", "The request asks for conceptual explanation."),
    ("Explain prime testing conceptually.", "text_v2", "The request asks for natural-language explanation."),
    ("What is a primality test?", "text_v2", "The request asks for a definition."),
    ("Explain SQL joins conceptually.", "text_v2", "The request asks for natural-language explanation."),
    ("Describe what SQL GROUP BY does.", "text_v2", "The request asks for explanation."),
    ("Explain a Python traceback in words.", "text_v2", "The request asks for explanation, not code repair."),
    ("Summarize Fractal Resonant Optimization in simple English.", "text_v2", "The request asks for natural-language explanation."),
    ("Explain FRO without equations.", "text_v2", "The request asks for natural-language explanation."),
    ("Spiega in italiano cos'e' un test di primalita'.", "text_v2", "The request asks for natural-language explanation."),
    ("Write SQL to count users by country.", "code_v2", "The request asks for SQL/query generation, which is code."),
    ("Write an SQL query using GROUP BY.", "code_v2", "The request asks for SQL/query generation."),
    ("Create a SQL query that joins users and orders.", "code_v2", "The request asks for SQL/query generation."),
    ("Write a Python primality test.", "code_v2", "The request asks for code generation."),
    ("Implement is_prime in Python.", "code_v2", "The request asks for software implementation."),
    ("Implement prime factorization in Python.", "code_v2", "The request asks for software implementation."),
    ("Fix this Python traceback.", "code_v2", "The request asks for debugging and code repair."),
    ("Write a regex extractor in Python.", "code_v2", "The request asks for code generation."),
    ("Scrivi una query SQL per contare utenti per paese.", "code_v2", "The request asks for SQL/query generation."),
    ("Scrivi un test di primalita' in Python.", "code_v2", "The request asks for code generation."),
    ("Solve a prime factorization problem for 84.", "math_v1", "The request asks for mathematical reasoning."),
    ("Factor 84 into primes.", "math_v1", "The request asks for mathematical reasoning."),
    ("Solve 3x + 5 = 20.", "math_v1", "The request asks for algebraic reasoning."),
    ("A train travels 120 km in 2 hours. What is the average speed?", "math_v1", "The request asks for numeric calculation."),
    ("What is 15 percent of 200?", "math_v1", "The request asks for numeric calculation."),
    ("Create a step-by-step plan to adversarially test agentic_v2.", "agentic_v1", "The request asks for planning and evaluation design."),
    ("Plan the next Orchestrator evaluation.", "agentic_v1", "The request asks for planning."),
    ("Create a checklist for safe tool-use testing.", "agentic_v1", "The request asks for planning and checklist creation."),
    ("Explain the idea, write pseudocode, and solve a small equation.", "orchestrator_v1", "The request combines multiple capabilities and should be split across Souls."),
    ("Explain SQL, write a query, and plan tests.", "orchestrator_v1", "The request combines explanation, code, and planning."),
    ("Describe prime numbers, implement is_prime, and solve 3x + 5 = 20.", "orchestrator_v1", "The request combines text, code, and math capabilities."),
]


def route_record(request: str, route: str, reason: str) -> str:
    return f"\n<|route|>\nUSER_REQUEST: {request}\nROUTE: {route}\nREASON: {reason}\n<|endroute|>\n"


def build_records() -> list[str]:
    return [route_record(*row) for row in CONFUSION_RECORDS]


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
    parser.add_argument("--target_mb", type=int, default=128)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    out = args.base_dir / "data" / "swarmlm_v3" / "orchestrator_v3b.bin"
    repeat_write(out, build_records(), args.target_mb * 1024**2, args.seed)


if __name__ == "__main__":
    main()
