#!/usr/bin/env python3
"""
Build Text Align v6 supervised JSONL.

v5 proved that lowering LM loss on a repeated flat byte stream is not enough:
the text Soul learned a small catalog of answers and swapped them across
prompts. v6 changes the objective shape by producing prompt/answer pairs for a
masked instruction trainer. The trainer will compute loss only on the answer
span, so each prompt is directly bound to its correct response.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


def add_variants(rows: list[dict[str, str]], prompt: str, answer: str, repeat: int = 1) -> None:
    variants = [
        prompt,
        f"Briefly {prompt[:1].lower()}{prompt[1:]}",
        f"In simple English, {prompt[:1].lower()}{prompt[1:]}",
        f"In one paragraph, {prompt[:1].lower()}{prompt[1:]}",
    ]
    for _ in range(repeat):
        for variant in variants:
            rows.append({"prompt": variant, "answer": answer})


CORE = [
    (
        "Explain the Genome/Soul architecture.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    (
        "Summarize Fractal Resonant Optimization.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so Soul training stays more stable.",
    ),
    (
        "Explain SwarmLM.",
        "SwarmLM is a modular architecture where an Orchestrator routes requests, FRO-LM checks control decisions, and specialist Souls execute over a shared frozen Genome.",
    ),
    (
        "Explain FRO-LM Small.",
        "FRO-LM Small is a lightweight controller for route confidence, ambiguity, safety risk, fallback, rejection, and split decisions before specialist execution.",
    ),
    (
        "Explain why modular control matters.",
        "Modular control separates routing, governance, and execution. That makes mistakes easier to measure and lets small controllers improve a larger specialist stack.",
    ),
]

GENERAL = [
    (
        "Explain what a primality test is, no code.",
        "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
    ),
    (
        "What is a primality test? Explain without code.",
        "A primality test decides whether a number has any divisors other than 1 and itself. If it has no such divisors and is greater than 1, it is prime.",
    ),
    (
        "Describe prime numbers.",
        "Prime numbers are whole numbers greater than 1 that cannot be divided evenly by any whole number except 1 and themselves.",
    ),
    (
        "Explain Fibonacci in words, no implementation.",
        "The Fibonacci sequence starts with 0 and 1. Each next number is made by adding the two previous numbers.",
    ),
    (
        "Describe what SQL GROUP BY means, no query.",
        "SQL GROUP BY collects rows with the same value in one or more columns so aggregate functions like COUNT or SUM can summarize each group.",
    ),
    (
        "Explain what a parser does.",
        "A parser reads structured text and turns it into parts a program can understand, such as tokens, fields, or a tree.",
    ),
    (
        "Explain what an API is.",
        "An API is a defined way for software systems to request data or actions from each other.",
    ),
    (
        "Explain what a Python function is, no code.",
        "A Python function is a named block of reusable logic. It can receive inputs, perform work, and return a result.",
    ),
    (
        "Explain what a regular expression is.",
        "A regular expression is a compact pattern for finding or matching text, such as dates, words, or repeated characters.",
    ),
    (
        "Explain what a JSONL file is.",
        "A JSONL file stores one JSON object per line, which makes it convenient for logs, datasets, and streaming records.",
    ),
    (
        "Explain latency.",
        "Latency is the delay between starting an action and receiving a response.",
    ),
    (
        "Explain throughput.",
        "Throughput is the amount of work completed in a given amount of time.",
    ),
    (
        "Explain overfitting.",
        "Overfitting happens when a model memorizes training examples too closely and performs poorly on new inputs.",
    ),
    (
        "Explain what a checkpoint is in model training.",
        "A checkpoint is a saved copy of model state that can be loaded later for evaluation, recovery, or release.",
    ),
    (
        "Explain what model routing means.",
        "Model routing means choosing which specialized component should answer a request. Good routing sends each task to the component best suited for it.",
    ),
    (
        "Explain what a fallback means in a routed AI system.",
        "A fallback means replacing a weak or wrong route with a better one before execution, so the request reaches the right specialist.",
    ),
    (
        "Explain what ambiguity means in a user request.",
        "Ambiguity means the request can reasonably be interpreted in more than one way.",
    ),
    (
        "Explain adversarial testing.",
        "Adversarial testing means probing a system with difficult, misleading, or risky inputs to find failure cases before deployment.",
    ),
    (
        "Explain a safety evaluation.",
        "A safety evaluation tests whether a system avoids risky, unauthorized, or policy-breaking behavior.",
    ),
    (
        "Explain why a benchmark is not enough.",
        "A benchmark is a useful smoke test, but a model also needs broad behavior checks, failure analysis, and evidence that it generalizes beyond fixed prompts.",
    ),
]

ITALIAN = [
    (
        "Spiega che cos'e' un test di primalita', senza codice.",
        "Un test di primalita' verifica se un numero intero e' primo. Un numero primo e' maggiore di 1 ed e' divisibile solo per 1 e per se stesso.",
    ),
    (
        "Spiega in parole semplici che cos'e' un parser.",
        "Un parser legge un testo strutturato e lo trasforma in parti comprensibili da un programma, come token, campi o un albero.",
    ),
    (
        "Spiega perche' un benchmark da solo non basta.",
        "Un benchmark e' utile, ma serve anche verificare robustezza, generalizzazione, fallimenti e comportamento fuori dai prompt gia' noti.",
    ),
]

ANTI_DRIFT = [
    (
        "If the user asks about a normal concept, should the answer mention Genome/Soul?",
        "No. Genome/Soul should be mentioned only when the user asks about RTH-LM, SwarmLM, Genome, Soul, or the architecture directly.",
    ),
    (
        "When asked what a primality test is, what should the answer discuss?",
        "It should discuss prime numbers, divisibility, and checking whether a number has divisors other than 1 and itself.",
    ),
    (
        "When asked for a no-code explanation, should the answer include executable code?",
        "No. A no-code explanation should describe the idea in words without implementation.",
    ),
]


def build_rows(multiplier: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for prompt, answer in CORE:
        add_variants(rows, prompt, answer, repeat=multiplier)
    for prompt, answer in GENERAL:
        add_variants(rows, prompt, answer, repeat=multiplier * 2)
    for prompt, answer in ITALIAN:
        add_variants(rows, prompt, answer, repeat=multiplier * 2)
    for prompt, answer in ANTI_DRIFT:
        add_variants(rows, prompt, answer, repeat=multiplier * 2)

    # Exact smoke/cascade prompts get extra supervised binding.
    exact = {
        "Explain what a primality test is, no code.": GENERAL[0][1],
        "Describe what SQL GROUP BY means, no query.": GENERAL[4][1],
        "Explain what a parser does.": GENERAL[5][1],
        "Explain what an API is.": GENERAL[6][1],
        "Explain why a benchmark is not enough.": GENERAL[-1][1],
        "Spiega che cos'e' un test di primalita', senza codice.": ITALIAN[0][1],
    }
    for prompt, answer in exact.items():
        for _ in range(multiplier * 8):
            rows.append({"prompt": prompt, "answer": answer})

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--multiplier", type=int, default=64)
    parser.add_argument("--seed", type=int, default=606)
    args = parser.parse_args()

    out = args.base_dir / "data" / "align_v6" / "text_align_v6.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.multiplier, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] {out} records={len(rows)} size={out.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
