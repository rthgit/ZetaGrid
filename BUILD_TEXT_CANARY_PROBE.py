#!/usr/bin/env python3
"""
Build a tiny text-instruction canary set.

This is not a release dataset. It is a diagnostic probe: a text Soul that cannot
overfit this small set should not be trained for days on a larger corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_ROWS = [
    (
        "Explain the Genome/Soul architecture in simple English.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    (
        "Summarize Fractal Resonant Optimization in simple English.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so training stays stable.",
    ),
    (
        "Explain what a primality test is, no code.",
        "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
    ),
    (
        "Describe what SQL GROUP BY means, no query.",
        "SQL GROUP BY collects rows that share the same value so aggregate functions can summarize each group.",
    ),
    (
        "Explain what a Python function is, no code.",
        "A Python function is a named, reusable block of logic. It can receive inputs, do work, and return a result.",
    ),
    (
        "Explain what a parser does.",
        "A parser turns text or tokens into structured information that a program can understand and use.",
    ),
    (
        "Explain what an API is.",
        "An API is a software interface that lets one program send requests and exchange data with another program.",
    ),
    (
        "Explain why a benchmark is not enough.",
        "A benchmark is useful, but it is not enough because a model can pass fixed tests and still fail on general prompts.",
    ),
    (
        "Spiega che cos'e' un test di primalita', senza codice.",
        "Un test di primalita' controlla se un numero e' primo. Un numero primo e' maggiore di 1 ed e' divisibile solo per 1 e per se stesso.",
    ),
    (
        "If the user asks about a normal concept, should the answer mention Genome/Soul?",
        "No. Mention Genome/Soul only when the user asks about that architecture or when it is directly relevant.",
    ),
]

VARIANTS = [
    ("In one sentence, ", ""),
    ("Briefly ", ""),
    ("In plain English, ", ""),
    ("For a non-technical reader, ", ""),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for prompt, answer in BASE_ROWS:
        out.append({"prompt": prompt, "answer": answer})
        for prefix, suffix in VARIANTS:
            out.append({"prompt": f"{prefix}{prompt[0].lower()}{prompt[1:]}{suffix}", "answer": answer})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.base_dir / "data" / "text_canary" / "text_canary_probe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = rows()
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.repeat):
            for row in data:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[DONE] {out} examples={len(data) * args.repeat} unique={len(data)}")


if __name__ == "__main__":
    main()
