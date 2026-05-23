#!/usr/bin/env python3
"""
Build a parametric reasoning probe dataset.

The first r1024 probe showed that repeating a tiny fixed set teaches format and
memorized answer fragments, not heldout reasoning. This builder generates many
short, verifiable variants across arithmetic, equations, sequences, logic,
primes, factors, percentages, speed, area, and perimeter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


HELDOUT_PROMPTS = {
    "Solve step by step: 4x + 2 = 18.",
    "A bike travels 90 km in 3 hours. What is the average speed?",
    "What is the next number: 5, 10, 20, 40, ?",
    "What is the next number: 4, 9, 14, 19, ?",
    "If all roses are flowers and this plant is a rose, is this plant a flower?",
    "If some cars are electric, does that prove all cars are electric?",
    "Is 31 prime?",
    "Is 27 prime?",
    "What is 20% of 150?",
    "Factor 60 into primes.",
}

PROMPT_VARIANTS = [
    "{prompt}",
    "Solve step by step: {lower_prompt}",
    "Give a short reasoning answer: {lower_prompt}",
    "Use the format Reasoning then Answer. {prompt}",
]


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def base_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    for a in range(2, 30):
        for b in range(2, 30):
            rows.append((f"If {a} + {b} = x, what is x?", f"Reasoning: Add {a} and {b}.\nAnswer: {a + b}"))
            if a + b <= 60:
                rows.append((f"If {a + b} - {a} = x, what is x?", f"Reasoning: Subtract {a} from {a + b}.\nAnswer: {b}"))
            if a <= 12 and b <= 12:
                rows.append((f"If {a} * {b} = x, what is x?", f"Reasoning: Multiply {a} by {b}.\nAnswer: {a * b}"))
                rows.append((f"If {a * b} / {a} = x, what is x?", f"Reasoning: Divide {a * b} by {a}.\nAnswer: {b}"))

    for coef in range(2, 10):
        for x in range(2, 16):
            for offset in range(1, 12):
                total = coef * x + offset
                rows.append((
                    f"Solve {coef}x + {offset} = {total}.",
                    f"Reasoning: Subtract {offset} to get {coef}x = {coef * x}. Divide by {coef}.\nAnswer: x = {x}",
                ))
                total2 = coef * x - offset
                if total2 > 0:
                    rows.append((
                        f"Solve {coef}x - {offset} = {total2}.",
                        f"Reasoning: Add {offset} to get {coef}x = {coef * x}. Divide by {coef}.\nAnswer: x = {x}",
                    ))

    for start in range(1, 12):
        for step in range(2, 8):
            seq = [start + i * step for i in range(4)]
            rows.append((
                f"What is the next number: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?",
                f"Reasoning: The sequence adds {step} each time.\nAnswer: {seq[-1] + step}",
            ))
    for start in range(2, 9):
        seq = [start * (2**i) for i in range(4)]
        rows.append((
            f"What is the next number: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?",
            f"Reasoning: Each number doubles.\nAnswer: {seq[-1] * 2}",
        ))

    categories = [
        ("cats", "mammals", "Milo", "cat"),
        ("squares", "rectangles", "this shape", "square"),
        ("robins", "birds", "this animal", "robin"),
        ("oaks", "trees", "this plant", "oak"),
        ("sedans", "cars", "this vehicle", "sedan"),
    ]
    for group, superset, item, singular in categories:
        rows.append((
            f"If all {group} are {superset} and {item} is a {singular}, is {item} a {superset[:-1] if superset.endswith('s') else superset}?",
            f"Reasoning: {item} belongs to the {group} group, and all {group} are {superset}.\nAnswer: Yes",
        ))
    some_groups = [("birds", "fly"), ("students", "left-handed"), ("cars", "electric"), ("books", "long")]
    for group, prop in some_groups:
        rows.append((
            f"If some {group} are {prop}, does that prove all {group} are {prop}?",
            "Reasoning: Some examples do not prove a universal rule.\nAnswer: No",
        ))

    for n in range(11, 80):
        if n in {27, 31}:
            continue
        if is_prime(n):
            rows.append((f"Is {n} prime?", f"Reasoning: {n} is greater than 1 and has no divisors other than 1 and {n}.\nAnswer: Yes"))
        else:
            divisor = next(d for d in range(2, n) if n % d == 0)
            rows.append((f"Is {n} prime?", f"Reasoning: {n} is divisible by {divisor}.\nAnswer: No"))

    for n in range(24, 121):
        if n == 60:
            continue
        factors = prime_factors(n)
        if len(factors) >= 2:
            rendered = " * ".join(str(f) for f in factors)
            rows.append((f"Factor {n} into primes.", f"Reasoning: Break {n} into prime factors.\nAnswer: {rendered}"))

    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        for base in range(40, 241, 10):
            if pct == 20 and base == 150:
                continue
            value = pct * base // 100
            if pct * base % 100 == 0:
                rows.append((f"What is {pct}% of {base}?", f"Reasoning: {pct}% means {pct} per 100, so compute {base} * {pct} / 100.\nAnswer: {value}"))

    for hours in range(2, 7):
        for speed in range(20, 91, 10):
            distance = hours * speed
            if distance == 90 and hours == 3:
                continue
            rows.append((
                f"A bike travels {distance} km in {hours} hours. What is the average speed?",
                f"Reasoning: Speed is distance divided by time, {distance} / {hours}.\nAnswer: {speed} km/h",
            ))

    for width in range(2, 13):
        for height in range(2, 13):
            rows.append((
                f"A rectangle has width {width} and height {height}. What is its area?",
                f"Reasoning: Area is width times height, {width} * {height}.\nAnswer: {width * height}",
            ))
    for side in range(2, 21):
        rows.append((
            f"A square has side length {side}. What is its perimeter?",
            f"Reasoning: A square has four equal sides, so 4 * {side}.\nAnswer: {4 * side}",
        ))

    unique: dict[str, str] = {}
    for prompt, answer in rows:
        if prompt not in HELDOUT_PROMPTS:
            unique[prompt] = answer
    return sorted(unique.items())


def build_rows(target_examples: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    base = base_rows()
    expanded: list[dict[str, str]] = []
    for prompt, answer in base:
        lower_prompt = lower_first(prompt)
        for template in PROMPT_VARIANTS:
            variant = template.format(prompt=prompt, lower_prompt=lower_prompt)
            expanded.append({"prompt": variant, "answer": answer})
    rng.shuffle(expanded)
    if target_examples > 0 and len(expanded) > target_examples:
        expanded = expanded[:target_examples]
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--examples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.base_dir / "data" / "reasoning_probe" / "reasoning_probe_v2_r1024.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.examples, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    unique = {(row["prompt"], row["answer"]) for row in rows}
    print(f"[DONE] {out} examples={len(rows)} unique={len(unique)} size={out.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
