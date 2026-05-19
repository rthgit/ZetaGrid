#!/usr/bin/env python3
"""
Build Text Align v5 dataset.

Goal:
  Train a serious text specialist, not a benchmark patch.

The v4 repair was intentionally narrow. v5 is broader:
- mixes real text bytes when an existing text corpus bin is available;
- adds broad instruction-style explanations across software, math concepts,
  safety, orchestration, evaluation, and plain-English definitions;
- preserves RTH/Genome/Soul/FRO identity;
- reinforces "no code" behavior without making the whole Soul benchmark-shaped;
- avoids route/delegation records because routing belongs to Orchestrator/FRO.

Output:
  /workspace/zetagrid_50b/data/align_v5/text_align_v5.bin
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


def instruction_record(user: str, answer: str) -> str:
    return f"\n<|instruction|>\nUser: {user}\nAssistant: {answer.rstrip()}\n<|endinstruction|>\n"


def expand_templates(items: list[tuple[str, str]], repeat: int) -> list[str]:
    records: list[str] = []
    prefixes = ["", "Briefly ", "In simple English, ", "In one paragraph, "]
    for user, answer in items:
        for prefix in prefixes:
            if prefix and user[:1].islower():
                prompt = prefix + user
            elif prefix:
                prompt = prefix + user[:1].lower() + user[1:]
            else:
                prompt = user
            records.extend([instruction_record(prompt, answer)] * repeat)
    return records


RTH_IDENTITY = [
    (
        "Explain the Genome/Soul architecture.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    (
        "Explain SwarmLM.",
        "SwarmLM is a modular system where an Orchestrator routes a request, FRO-LM checks control decisions, and specialist Souls execute over a shared frozen Genome.",
    ),
    (
        "Summarize Fractal Resonant Optimization.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so Soul training stays more stable.",
    ),
    (
        "Explain why FRO-LM Small matters.",
        "FRO-LM Small is a lightweight controller that checks route confidence, ambiguity, safety risk, fallback, split, and rejection decisions before specialist execution.",
    ),
    (
        "Why is modular control useful in SwarmLM?",
        "Modular control separates routing, governance, and execution. That makes mistakes easier to measure and lets small controllers improve a larger specialist stack.",
    ),
    (
        "Spiega in italiano che cosa fa FRO-LM Small.",
        "FRO-LM Small e' un controller leggero: valuta confidenza della route, ambiguita', rischio e fallback prima di caricare una Soul specialistica.",
    ),
]


GENERAL_CONCEPTS = [
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
        "Explain average speed.",
        "Average speed is total distance divided by total time. It describes how fast something moved overall.",
    ),
    (
        "Explain what a Python function is, no code.",
        "A Python function is a named block of reusable logic. It can receive inputs, perform work, and return a result.",
    ),
    (
        "Explain what SQL GROUP BY means, no query.",
        "SQL GROUP BY collects rows with the same value in one or more columns so aggregate functions like COUNT or SUM can summarize each group.",
    ),
    (
        "Explain what a parser does.",
        "A parser reads structured text and turns it into parts a program can understand, such as tokens, fields, or a tree.",
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
        "Explain what an API is.",
        "An API is a defined way for software systems to request data or actions from each other.",
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
]


EXPLANATION_TASKS = [
    (
        "Summarize why a small controller can improve a modular model.",
        "A small controller can catch routing mistakes, unsafe requests, and ambiguous prompts before expensive specialist modules run.",
    ),
    (
        "Explain the difference between routing and execution.",
        "Routing chooses which component should handle a request. Execution is the actual generation performed by the selected specialist.",
    ),
    (
        "Explain why a benchmark is not enough.",
        "A benchmark is a useful smoke test, but a model also needs broad behavior checks, failure analysis, and evidence that it generalizes beyond fixed prompts.",
    ),
    (
        "Explain why false fallback matters.",
        "A false fallback sends a clear request away from the right route, which can reduce quality even when the system is trying to be careful.",
    ),
    (
        "Explain why unsafe delegation matters.",
        "Unsafe delegation matters because an agentic system can cause harm if it follows requests to bypass boundaries, access secrets, or execute uncontrolled actions.",
    ),
    (
        "Explain what specialist coherence means.",
        "Specialist coherence means the selected module behaves consistently with its intended domain instead of drifting into unrelated behavior.",
    ),
    (
        "Explain why raw outputs should be saved during evaluation.",
        "Raw outputs make failures auditable. They let researchers inspect what the model actually generated instead of relying only on summary scores.",
    ),
    (
        "Explain why deterministic smoke tests are useful.",
        "Deterministic smoke tests reduce randomness so routing, control, and specialist behavior can be compared across checkpoints.",
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
    (
        "When asked to explain SQL GROUP BY without a query, what should the answer do?",
        "It should describe grouping rows and aggregates in words, without writing SQL.",
    ),
]


ITALIAN_TEXT = [
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


def build_instruction_records() -> list[str]:
    records: list[str] = []
    records.extend(expand_templates(RTH_IDENTITY, repeat=6))
    records.extend(expand_templates(GENERAL_CONCEPTS, repeat=8))
    records.extend(expand_templates(EXPLANATION_TASKS, repeat=7))
    records.extend(expand_templates(ANTI_DRIFT, repeat=10))
    records.extend(expand_templates(ITALIAN_TEXT, repeat=6))

    # High-weight exact hard case, but not enough to dominate the whole corpus.
    exact = instruction_record(
        "Explain what a primality test is, no code.",
        "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
    )
    records.extend([exact] * 60)
    return records


def sample_real_chunk(real_bytes: bytes, rng: random.Random, min_len: int = 2048, max_len: int = 16384) -> bytes:
    if len(real_bytes) <= min_len + 2:
        return real_bytes
    size = rng.randint(min_len, min(max_len, len(real_bytes) - 1))
    start = rng.randint(0, len(real_bytes) - size - 1)
    return b"\n" + real_bytes[start : start + size] + b"\n"


def read_real_sources(paths: list[Path]) -> bytes:
    chunks: list[bytes] = []
    for path in paths:
        if not path.exists():
            print(f"[SKIP] real source not found: {path}")
            continue
        data = path.read_bytes()
        if data:
            print(f"[REAL] {path} {len(data) / 1024**2:.1f} MB")
            chunks.append(data)
    return b"\n".join(chunks)


def write_mixed_dataset(
    path: Path,
    records: list[str],
    target_bytes: int,
    seed: int,
    real_bytes: bytes,
    real_ratio: float,
) -> None:
    rng = random.Random(seed)
    encoded = [record.encode("utf-8", errors="ignore") for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    instruction_written = 0
    real_written = 0

    with path.open("wb") as f:
        while written < target_bytes:
            use_real = bool(real_bytes) and rng.random() < real_ratio
            if use_real:
                block = sample_real_chunk(real_bytes, rng)
                real_written += len(block)
            else:
                block = rng.choice(encoded)
                instruction_written += len(block)
            n = min(len(block), target_bytes - written)
            f.write(block[:n])
            written += n

    print(f"[DONE] {path} {path.stat().st_size / 1024**2:.1f} MB records={len(records)}")
    print(f"[MIX] instruction={instruction_written / 1024**2:.1f} MB real={real_written / 1024**2:.1f} MB real_ratio={real_ratio}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--target_mb", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--real_ratio", type=float, default=0.35)
    parser.add_argument(
        "--real_source",
        type=Path,
        action="append",
        default=[],
        help="Optional real text/bin source. Can be passed multiple times.",
    )
    args = parser.parse_args()

    default_sources = [
        args.base_dir / "data" / "text_v2" / "fineweb_text_v2.bin",
        args.base_dir / "data" / "align_v3" / "text_align_v3.bin",
    ]
    sources = args.real_source or default_sources
    real_bytes = read_real_sources(sources)
    records = build_instruction_records()
    out = args.base_dir / "data" / "align_v5" / "text_align_v5.bin"
    write_mixed_dataset(out, records, args.target_mb * 1024**2, args.seed, real_bytes, args.real_ratio)


if __name__ == "__main__":
    main()
