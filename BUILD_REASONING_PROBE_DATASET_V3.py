#!/usr/bin/env python3
"""Build a balanced parametric reasoning probe dataset.

V2 generated many unique examples, but arithmetic/equation rows dominated the
mix. This builder samples the same number of examples from each task family so
the model cannot minimize loss by always emitting an equation template.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from BUILD_REASONING_PROBE_DATASET_V2 import PROMPT_VARIANTS, base_rows, lower_first


def category_for(prompt: str) -> str:
    if prompt.startswith("Solve "):
        return "equation"
    if prompt.startswith("If ") and "= x, what is x?" in prompt:
        return "arithmetic"
    if prompt.startswith("What is the next number:"):
        return "sequence"
    if prompt.startswith("If all ") or prompt.startswith("If some "):
        return "logic"
    if prompt.startswith("Is "):
        return "prime"
    if prompt.startswith("Factor "):
        return "factor"
    if prompt.startswith("What is ") and "% of " in prompt:
        return "percent"
    if prompt.startswith("A bike travels "):
        return "speed"
    if prompt.startswith("A rectangle ") or prompt.startswith("A square "):
        return "geometry"
    return "other"


def expand_prompt(prompt: str, answer: str) -> list[dict[str, str]]:
    lower_prompt = lower_first(prompt)
    return [
        {"prompt": template.format(prompt=prompt, lower_prompt=lower_prompt), "answer": answer}
        for template in PROMPT_VARIANTS
    ]


def build_rows(per_category: int, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt, answer in base_rows():
        grouped[category_for(prompt)].extend(expand_prompt(prompt, answer))

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category in sorted(grouped):
        if category == "other":
            continue
        bucket = grouped[category]
        if not bucket:
            continue
        if len(bucket) >= per_category:
            picked = rng.sample(bucket, per_category)
        else:
            picked = [rng.choice(bucket) for _ in range(per_category)]
        rows.extend(picked)
        print(f"[CATEGORY] {category}: source={len(bucket)} picked={len(picked)}")
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--per_category", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=912)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.base_dir / "data" / "reasoning_probe" / "reasoning_probe_v3_balanced_r1024.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.per_category, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    unique = {(row["prompt"], row["answer"]) for row in rows}
    print(f"[DONE] {out} examples={len(rows)} unique={len(unique)} size={out.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
