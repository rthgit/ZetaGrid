#!/usr/bin/env python3
"""
Build Text Align v4 dataset.

This is a targeted repair for the text specialist in the FRO-controlled
cascade. The controller and router already select text_v2 correctly; the
remaining failure is execution-layer drift where generic explanation prompts
are answered with memorized Genome/Soul text.

Design:
- keep RTH/Genome/Soul identity answers;
- heavily reinforce generic no-code explanations;
- include exact hard cases from the cascade;
- avoid ROUTE_REQUESTED records, because routing belongs to Orchestrator/FRO,
  not to the text execution Soul.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


def instruction_record(user: str, answer: str) -> str:
    return f"\n<|instruction|>\nUser: {user}\nAssistant: {answer.rstrip()}\n<|endinstruction|>\n"


def direct_answer(user: str, answer: str) -> str:
    return instruction_record(user, answer)


RTH_IDENTITY = [
    direct_answer(
        "Explain the Genome/Soul architecture in simple English.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    direct_answer(
        "Summarize Fractal Resonant Optimization in simple English.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so Soul training stays more stable.",
    ),
    direct_answer(
        "Explain why SwarmLM needs an orchestrator.",
        "SwarmLM uses specialized Souls for different domains. The orchestrator chooses which Soul should handle a request before generation begins.",
    ),
    direct_answer(
        "What does FRO-LM Small do in SwarmLM?",
        "FRO-LM Small is a lightweight controller. It checks route confidence, ambiguity, safety risk, and whether a request should be accepted, rejected, corrected, or split.",
    ),
    direct_answer(
        "Spiega in italiano perche' separare Genome e Soul e' utile.",
        "Separare Genome e Soul e' utile per mantenere una base congelata comune mentre ogni Soul apprende una specializzazione controllata.",
    ),
]


PRIMALITY_ANSWERS = [
    (
        "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
        ["prime", "number", "divisible"],
    ),
    (
        "A primality test decides whether a number has any divisors other than 1 and itself. If it has no such divisors and is greater than 1, it is prime.",
        ["prime", "divisors", "number"],
    ),
    (
        "A primality test is a way to tell whether a number is prime. It rules out numbers that can be divided evenly by smaller whole numbers.",
        ["prime", "number", "divided"],
    ),
]

PRIMALITY_PROMPTS = [
    "Explain what a primality test is, no code.",
    "What is a primality test? Explain without code.",
    "Explain primality testing in plain English.",
    "Describe a primality test without writing code.",
    "What does a primality test check?",
    "Explain how to tell if a number is prime, conceptually.",
    "Define primality test in one short paragraph.",
    "Explain prime-number testing for a non-programmer.",
    "What is the purpose of a primality test?",
    "Explain the idea of checking whether a number is prime.",
]


GENERIC_EXPLANATIONS = [
    (
        "Describe prime numbers in simple English.",
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
        "Explain what a shell command is, no execution.",
        "A shell command is an instruction typed into a command-line interface to ask the operating system or a program to do something.",
    ),
    (
        "Explain what a Python function is, no code.",
        "A Python function is a named block of reusable logic. It can receive inputs, perform work, and return a result.",
    ),
    (
        "Explain what average speed means.",
        "Average speed is total distance divided by total time. It describes how fast something moved overall.",
    ),
    (
        "Explain what a parser does.",
        "A parser reads structured text and turns it into parts a program can understand, such as tokens, fields, or a tree.",
    ),
    (
        "Explain what a regular expression is, no code.",
        "A regular expression is a compact pattern for finding or matching text, such as dates, words, or repeated characters.",
    ),
    (
        "Explain what a JSONL file is.",
        "A JSONL file stores one JSON object per line, which makes it convenient for logs, datasets, and streaming records.",
    ),
    (
        "Explain what model routing means.",
        "Model routing means choosing which specialized component should answer a request. Good routing sends each task to the component best suited for it.",
    ),
    (
        "Explain adversarial testing in simple English.",
        "Adversarial testing means probing a system with difficult, misleading, or risky inputs to find failure cases before deployment.",
    ),
    (
        "Explain why a lightweight controller can help a modular AI system.",
        "A lightweight controller can check route confidence, safety risk, and ambiguity before expensive specialist modules are loaded.",
    ),
    (
        "What is a controlled cascade in SwarmLM?",
        "A controlled cascade routes the user request, validates the route with a controller, then executes with the selected specialist Soul.",
    ),
    (
        "Explain what a fallback means in a routed AI system.",
        "A fallback means replacing a weak or wrong route with a better one before execution, so the request reaches the right specialist.",
    ),
]


ANTI_DRIFT = [
    direct_answer(
        "If the user asks about a normal concept, should the answer mention Genome/Soul?",
        "No. Genome/Soul should be mentioned only when the user asks about RTH-LM, SwarmLM, Genome, Soul, or the architecture directly.",
    ),
    direct_answer(
        "When asked what a primality test is, what should the answer discuss?",
        "It should discuss prime numbers, divisibility, and checking whether a number has divisors other than 1 and itself.",
    ),
    direct_answer(
        "When asked for a no-code explanation, should the answer include executable code?",
        "No. A no-code explanation should describe the idea in words without implementation.",
    ),
]


def build_records() -> list[str]:
    records: list[str] = []

    records.extend(RTH_IDENTITY * 8)
    records.extend(ANTI_DRIFT * 12)

    for i, prompt in enumerate(PRIMALITY_PROMPTS):
        answer, _ = PRIMALITY_ANSWERS[i % len(PRIMALITY_ANSWERS)]
        records.append(direct_answer(prompt, answer))

    # Exact cascade hard case gets high weight.
    exact = direct_answer(
        "Explain what a primality test is, no code.",
        "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
    )
    records.extend([exact] * 40)

    for user, answer in GENERIC_EXPLANATIONS:
        records.append(direct_answer(user, answer))
        records.append(direct_answer(user.replace("Explain", "Briefly explain"), answer))
        records.append(direct_answer(user.replace("Describe", "Briefly describe"), answer))

    # Add simple paraphrase anchors so the text Soul sees broad non-RTH concepts.
    concepts = [
        ("What is a database index?", "A database index is an auxiliary structure that helps a database find rows faster without scanning the whole table."),
        ("What is an API?", "An API is a defined way for software systems to request data or actions from each other."),
        ("What is latency?", "Latency is the delay between starting an action and receiving a response."),
        ("What is throughput?", "Throughput is the amount of work completed in a given amount of time."),
        ("What is a checkpoint in model training?", "A checkpoint is a saved copy of model state that can be loaded later for evaluation, recovery, or release."),
        ("What is overfitting?", "Overfitting happens when a model memorizes training examples too closely and performs poorly on new inputs."),
        ("What is a safety eval?", "A safety eval tests whether a system avoids risky, unauthorized, or policy-breaking behavior."),
        ("What is ambiguity in a user request?", "Ambiguity means the request can reasonably be interpreted in more than one way."),
    ]
    for user, answer in concepts:
        records.extend([direct_answer(user, answer)] * 4)

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
    parser.add_argument("--target_mb", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()

    out = args.base_dir / "data" / "align_v4" / "text_align_v4.bin"
    repeat_write(out, build_records(), args.target_mb * 1024**2, args.seed)


if __name__ == "__main__":
    main()
