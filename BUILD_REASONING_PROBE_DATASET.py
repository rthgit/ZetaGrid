#!/usr/bin/env python3
"""
Build a controlled reasoning probe dataset.

This is for post-submission scaling research, not for release claims. It keeps
examples short and verifiable so a 70B-class Genome/Soul reasoning run can be
judged by exact answers, format adherence, loss stability, VRAM, and speed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TRAIN_ROWS = [
    ("If 7 + 5 = x, what is x?", "Reasoning: Add 7 and 5.\nAnswer: 12"),
    ("If 18 - 9 = x, what is x?", "Reasoning: Subtract 9 from 18.\nAnswer: 9"),
    ("If 6 * 7 = x, what is x?", "Reasoning: Multiply 6 by 7.\nAnswer: 42"),
    ("If 48 / 6 = x, what is x?", "Reasoning: Divide 48 by 6.\nAnswer: 8"),
    ("Solve 3x + 5 = 20.", "Reasoning: Subtract 5 to get 3x = 15. Divide by 3.\nAnswer: x = 5"),
    ("Solve 2x - 4 = 10.", "Reasoning: Add 4 to get 2x = 14. Divide by 2.\nAnswer: x = 7"),
    ("Solve 5x = 45.", "Reasoning: Divide both sides by 5.\nAnswer: x = 9"),
    ("What is the next number: 2, 4, 8, 16, ?", "Reasoning: Each number doubles.\nAnswer: 32"),
    ("What is the next number: 3, 6, 9, 12, ?", "Reasoning: The sequence adds 3 each time.\nAnswer: 15"),
    ("What is the next number: 1, 1, 2, 3, 5, ?", "Reasoning: This is Fibonacci; add the previous two terms.\nAnswer: 8"),
    ("A box has 3 red balls and 2 blue balls. How many balls are there?", "Reasoning: Add 3 and 2.\nAnswer: 5"),
    ("Ana has 4 apples and buys 6 more. How many apples does she have?", "Reasoning: Add 4 and 6.\nAnswer: 10"),
    ("A train travels 120 km in 2 hours. What is the average speed?", "Reasoning: Speed is distance divided by time, 120 / 2.\nAnswer: 60 km/h"),
    ("A rectangle has width 4 and height 6. What is its area?", "Reasoning: Area is width times height, 4 * 6.\nAnswer: 24"),
    ("A square has side length 9. What is its perimeter?", "Reasoning: A square has four equal sides, so 4 * 9.\nAnswer: 36"),
    ("If all cats are mammals and Milo is a cat, is Milo a mammal?", "Reasoning: Milo belongs to the cat group, and all cats are mammals.\nAnswer: Yes"),
    ("If all squares are rectangles, is every square a rectangle?", "Reasoning: The statement directly says all squares are rectangles.\nAnswer: Yes"),
    ("If some birds can fly, does that prove all birds can fly?", "Reasoning: Some examples do not prove a universal rule.\nAnswer: No"),
    ("Which is larger: 17 or 23?", "Reasoning: Compare the two numbers; 23 is greater than 17.\nAnswer: 23"),
    ("Which is smaller: 0.3 or 0.8?", "Reasoning: Compare tenths; 0.3 is less than 0.8.\nAnswer: 0.3"),
    ("Is 29 prime?", "Reasoning: 29 is greater than 1 and has no divisors other than 1 and 29.\nAnswer: Yes"),
    ("Is 21 prime?", "Reasoning: 21 is divisible by 3 and 7.\nAnswer: No"),
    ("Factor 84 into primes.", "Reasoning: 84 = 2 * 42 = 2 * 2 * 21 = 2 * 2 * 3 * 7.\nAnswer: 2 * 2 * 3 * 7"),
    ("What is 15% of 200?", "Reasoning: 10% of 200 is 20 and 5% is 10, so 15% is 30.\nAnswer: 30"),
    ("If a price goes from 50 to 60, what is the increase?", "Reasoning: Subtract 50 from 60.\nAnswer: 10"),
]


PROMPT_VARIANTS = [
    "{prompt}",
    "Solve step by step: {lower_prompt}",
    "Give a short reasoning answer: {lower_prompt}",
    "Use the format Reasoning then Answer. {prompt}",
]


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def build_rows(repeat: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for prompt, answer in TRAIN_ROWS:
        lower_prompt = lower_first(prompt)
        for template in PROMPT_VARIANTS:
            variant = template.format(prompt=prompt, lower_prompt=lower_prompt)
            for _ in range(repeat):
                rows.append({"prompt": variant, "answer": answer})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--seed", type=int, default=909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.base_dir / "data" / "reasoning_probe" / "reasoning_probe_r1024.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.repeat, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    unique = {(row["prompt"], row["answer"]) for row in rows}
    print(f"[DONE] {out} examples={len(rows)} unique={len(unique)} size={out.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
