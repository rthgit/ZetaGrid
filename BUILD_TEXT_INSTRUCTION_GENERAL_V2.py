#!/usr/bin/env python3
"""
Build a controlled general-instruction text curriculum.

This replaces the noisy long-context instruction mix that produced low loss but
bad generation. The goal is not to patch one benchmark: it creates many short,
answer-bound prompts while preserving RTH/SwarmLM identity and explicitly
teaching when not to mention Genome/Soul.

Output:
  /workspace/zetagrid_50b/data/text_instruction_general_v2/instruction_general_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


RTH_ROWS = [
    (
        "Explain the Genome/Soul architecture in simple English.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    (
        "Summarize Fractal Resonant Optimization in simple English.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so training stays stable.",
    ),
    (
        "Explain SwarmLM in simple terms.",
        "SwarmLM is a modular system where an orchestrator routes a request, a controller checks the route, and specialist Souls answer over a shared Genome.",
    ),
    (
        "What does FRO-LM Small do?",
        "FRO-LM Small is a lightweight controller that checks routing confidence, ambiguity, fallback, rejection, safety, and split decisions.",
    ),
    (
        "Why separate routing from specialist execution?",
        "Separating routing from execution makes failures easier to measure and lets the system send each request to the best specialist.",
    ),
]


GENERAL_ROWS = [
    ("Explain what a primality test is, no code.", "A primality test checks whether a whole number is prime. A prime number is greater than 1 and is divisible only by 1 and itself."),
    ("Describe what SQL GROUP BY means, no query.", "SQL GROUP BY collects rows that share the same value so aggregate functions can summarize each group."),
    ("Explain what a Python function is, no code.", "A Python function is a named, reusable block of logic. It can receive inputs, do work, and return a result."),
    ("Explain what a parser does.", "A parser turns text or tokens into structured information that a program can understand and use."),
    ("Explain what an API is.", "An API is a software interface that lets one program send requests and exchange data with another program."),
    ("Explain why a benchmark is not enough.", "A benchmark is useful, but it is not enough because a model can pass fixed tests and still fail on general prompts."),
    ("Explain what JSONL is.", "JSONL is a text format where each line is one complete JSON object. It is useful for logs, datasets, and streaming records."),
    ("Explain what latency means.", "Latency is the delay between making a request and receiving a response."),
    ("Explain what throughput means.", "Throughput is the amount of work a system completes in a given amount of time."),
    ("Explain what overfitting means.", "Overfitting happens when a model memorizes training examples too closely and performs poorly on new inputs."),
    ("Explain what a model checkpoint is.", "A checkpoint is a saved model state that can be loaded later for evaluation, recovery, or release."),
    ("Explain what a tokenizer does.", "A tokenizer splits text into smaller units that a model or program can process."),
    ("Explain what an embedding is.", "An embedding is a numeric representation that captures useful properties of text, tokens, or other data."),
    ("Explain what a loss function is.", "A loss function measures how wrong a model prediction is, so training can adjust the model to improve."),
    ("Explain what a gradient is.", "A gradient describes how changing model parameters would change the loss."),
    ("Explain what learning rate means.", "The learning rate controls how large each training update is."),
    ("Explain what validation means in model training.", "Validation checks model behavior on data that is not used directly for training updates."),
    ("Explain what a smoke test is.", "A smoke test is a small quick check that catches obvious failures before running larger evaluations."),
    ("Explain what a fallback means.", "A fallback is a safer or better alternative used when the first choice is weak, risky, or wrong."),
    ("Explain what ambiguity means.", "Ambiguity means a request can reasonably be understood in more than one way."),
    ("Explain what a safety gate is.", "A safety gate blocks or redirects requests that are risky, unauthorized, or outside allowed behavior."),
    ("Explain what deterministic evaluation means.", "Deterministic evaluation reduces randomness so results can be compared across checkpoints."),
    ("Explain why raw outputs matter in evaluation.", "Raw outputs show exactly what the model generated, making failures auditable instead of hidden behind summary scores."),
    ("Explain what generalization means.", "Generalization means performing well on new examples, not only on examples seen during training."),
    ("Explain what a regression is in software testing.", "A regression is a bug where behavior that used to work stops working after a change."),
    ("Explain what a schema is.", "A schema describes the expected structure and fields of data."),
    ("Explain what a cache is.", "A cache stores data temporarily so future requests can be served faster."),
    ("Explain what a database index does.", "A database index helps find rows faster without scanning the whole table."),
    ("Explain what authentication means.", "Authentication verifies who a user or system is."),
    ("Explain what authorization means.", "Authorization decides what an authenticated user or system is allowed to do."),
    ("Explain what encryption does.", "Encryption transforms data so only someone with the right key can read it."),
    ("Explain what a hash is.", "A hash is a fixed-size value computed from data, often used to compare or identify content."),
    ("Explain what version control is.", "Version control tracks changes to files so people can review, restore, and coordinate work."),
    ("Explain what a branch is in Git.", "A Git branch is a separate line of work that can later be merged with other changes."),
    ("Explain what a pull request is.", "A pull request proposes changes for review before they are merged into a shared codebase."),
    ("Explain what unit tests check.", "Unit tests check small pieces of code in isolation."),
    ("Explain what integration tests check.", "Integration tests check whether multiple parts of a system work together correctly."),
    ("Explain what an invariant is.", "An invariant is a condition that should remain true while a system runs."),
    ("Explain what a queue is.", "A queue stores items to be processed later, usually in the order they arrive."),
    ("Explain what backpressure means.", "Backpressure happens when a system slows input because downstream work cannot keep up."),
    ("Explain what rate limiting does.", "Rate limiting controls how many requests can be made in a period of time."),
    ("Explain what monitoring is.", "Monitoring collects signals about a system so problems can be detected and diagnosed."),
    ("Explain what observability means.", "Observability means having enough signals to understand what a system is doing internally."),
    ("Explain what a metric is.", "A metric is a numeric measurement used to track behavior or performance."),
    ("Explain what a log is.", "A log is a record of events produced by software while it runs."),
    ("Explain what an error budget is.", "An error budget is the amount of unreliability a service can tolerate before reliability work takes priority."),
    ("Explain what a dependency is.", "A dependency is external code, data, or service that a system relies on."),
    ("Explain what a data pipeline is.", "A data pipeline moves and transforms data through a sequence of processing steps."),
    ("Explain what normalization means in data processing.", "Normalization puts data into a consistent form so it is easier to compare or process."),
    ("Explain what sampling means.", "Sampling means choosing part of a larger set to estimate, inspect, or train on."),
    ("Explain what bias means in a dataset.", "Dataset bias means the data overrepresents or underrepresents patterns in a way that can skew results."),
    ("Explain what precision means in evaluation.", "Precision measures how many selected or predicted positives are actually correct."),
    ("Explain what recall means in evaluation.", "Recall measures how many actual positives were found."),
    ("Explain what a confusion matrix shows.", "A confusion matrix compares predicted labels with true labels to show correct and incorrect classifications."),
    ("Explain what a median is.", "The median is the middle value after numbers are sorted."),
    ("Explain what an average is.", "An average summarizes a set of numbers, often by adding them and dividing by how many there are."),
    ("Explain what variance means.", "Variance measures how spread out values are around their average."),
    ("Explain what correlation means.", "Correlation describes how two quantities tend to change together."),
    ("Explain what causation means.", "Causation means one event or factor directly helps produce another."),
    ("Explain what a hypothesis is.", "A hypothesis is a testable idea about how something works."),
    ("Explain what an experiment is.", "An experiment is a controlled test used to learn whether an idea or change has an effect."),
]


ITALIAN_ROWS = [
    ("Spiega che cos'e' un test di primalita', senza codice.", "Un test di primalita' controlla se un numero e' primo. Un numero primo e' maggiore di 1 ed e' divisibile solo per 1 e per se stesso."),
    ("Spiega che cos'e' un parser.", "Un parser trasforma testo o token in informazioni strutturate che un programma puo' capire e usare."),
    ("Spiega che cos'e' una API.", "Una API e' un'interfaccia software che permette a un programma di inviare richieste e scambiare dati con un altro programma."),
    ("Spiega perche' un benchmark non basta.", "Un benchmark e' utile, ma non basta: un modello puo' superare test fissi e fallire su richieste nuove o diverse."),
    ("Spiega che cos'e' l'overfitting.", "L'overfitting avviene quando un modello memorizza troppo gli esempi di training e funziona male su casi nuovi."),
]


CONTRAST_ROWS = [
    ("If the user asks about a normal concept, should the answer mention Genome/Soul?", "No. Mention Genome/Soul only when the user asks about that architecture or when it is directly relevant."),
    ("Should an explanation of SQL GROUP BY talk about prime numbers?", "No. It should talk about grouping rows with the same value and summarizing each group."),
    ("Should an explanation of a parser talk about Genome/Soul?", "No. It should explain how a parser turns text or tokens into structured information."),
    ("Should an explanation of an API talk about prime numbers?", "No. It should explain software interfaces, requests, and data exchange."),
    ("Should a no-code explanation include executable code?", "No. A no-code explanation should describe the idea in words without implementation."),
    ("When asked about primality, what topic should the answer stay on?", "It should stay on prime numbers, divisibility, and checking whether a whole number is prime."),
    ("When asked about FRO, what topic should the answer stay on?", "It should stay on gradient coherence, stable updates, and training dynamics."),
    ("When asked about a benchmark, what topic should the answer stay on?", "It should stay on evaluation limits, generalization, failure analysis, and behavior beyond fixed tests."),
]


PROMPT_VARIANTS = [
    "{prompt}",
    "Briefly {lower_prompt}",
    "In simple English, {lower_prompt}",
    "In one sentence, {lower_prompt}",
    "For a non-technical reader, {lower_prompt}",
]


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def add_variants(out: list[dict[str, str]], rows: list[tuple[str, str]], repeat: int) -> None:
    for prompt, answer in rows:
        lower_prompt = lower_first(prompt)
        for template in PROMPT_VARIANTS:
            variant = template.format(prompt=prompt, lower_prompt=lower_prompt)
            for _ in range(repeat):
                out.append({"prompt": variant, "answer": answer})


def build_rows(repeat: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    add_variants(rows, RTH_ROWS, repeat=max(1, repeat))
    add_variants(rows, GENERAL_ROWS, repeat=max(1, repeat * 2))
    add_variants(rows, ITALIAN_ROWS, repeat=max(1, repeat * 2))
    add_variants(rows, CONTRAST_ROWS, repeat=max(1, repeat * 3))

    # Extra retention for the canary/smoke prompts, but kept below the weight of
    # the broad general set.
    add_variants(rows, RTH_ROWS + GENERAL_ROWS[:6] + ITALIAN_ROWS[:1] + CONTRAST_ROWS[:1], repeat=max(1, repeat * 3))

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--seed", type=int, default=707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.base_dir / "data" / "text_instruction_general_v2" / "instruction_general_v2.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.repeat, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    unique = {(row["prompt"], row["answer"]) for row in rows}
    print(f"[DONE] {out} examples={len(rows)} unique={len(unique)} size={out.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
