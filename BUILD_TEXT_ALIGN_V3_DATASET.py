#!/usr/bin/env python3
"""
Build Text Align v3 dataset.

Purpose:
- fix the remaining FRO-controlled cascade failure:
  "Explain what a primality test is, no code.";
- preserve Genome/Soul and FRO explanations;
- improve general no-code explanations;
- keep off-domain delegation for explicit code/math/agentic requests.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


def instruction_record(user: str, answer: str) -> str:
    return f"\n<|instruction|>\nUser: {user}\nAssistant: {answer.rstrip()}\n<|endinstruction|>\n"


def route_request(user: str, target: str) -> str:
    return instruction_record(user, f"ROUTE_REQUESTED: {target}")


CORE_TEXT = [
    instruction_record(
        "Explain the Genome/Soul architecture in simple English.",
        "The Genome is a shared frozen substrate. A Soul is a trainable specialization layer that changes behavior without replacing the Genome.",
    ),
    instruction_record(
        "Summarize Fractal Resonant Optimization in simple English.",
        "Fractal Resonant Optimization monitors gradient coherence across scales and dampens unstable update shocks so Soul training stays more stable.",
    ),
    instruction_record(
        "Explain why SwarmLM needs an orchestrator.",
        "SwarmLM uses specialized Souls for different domains. The orchestrator chooses which Soul should handle a request before generation begins.",
    ),
    instruction_record(
        "Spiega in italiano perche' separare Genome e Soul e' utile.",
        "Separare Genome e Soul e' utile per mantenere una base congelata comune mentre ogni Soul apprende una specializzazione controllata.",
    ),
]


NO_CODE_EXPLANATIONS = [
    instruction_record(
        "Explain what a primality test is, no code.",
        "A primality test checks whether a number is prime. A prime number is greater than 1 and is divisible only by 1 and itself.",
    ),
    instruction_record(
        "What is a primality test? Explain without code.",
        "A primality test is a method for deciding whether a number has any divisors other than 1 and itself.",
    ),
    instruction_record(
        "Describe prime numbers in simple English.",
        "Prime numbers are whole numbers greater than 1 that cannot be divided evenly by any whole number except 1 and themselves.",
    ),
    instruction_record(
        "Explain Fibonacci in words, no implementation.",
        "The Fibonacci sequence starts with 0 and 1. Each next number is made by adding the two previous numbers.",
    ),
    instruction_record(
        "Describe what SQL GROUP BY means, no query.",
        "SQL GROUP BY collects rows with the same value in one or more columns so aggregate functions like COUNT or SUM can summarize each group.",
    ),
    instruction_record(
        "Explain what a shell command is, no execution.",
        "A shell command is an instruction typed into a command-line interface to ask the operating system or a program to do something.",
    ),
    instruction_record(
        "Explain what a Python function is, no code.",
        "A Python function is a named block of reusable logic. It can receive inputs, perform work, and return a result.",
    ),
    instruction_record(
        "Explain what average speed means.",
        "Average speed is total distance divided by total time. It describes how fast something moved overall.",
    ),
]


GENERAL_EXPLANATIONS = [
    instruction_record(
        "Explain model routing in one paragraph.",
        "Model routing means choosing which specialized component should answer a request. Good routing keeps tasks in the domain where the selected component is strongest.",
    ),
    instruction_record(
        "Explain adversarial testing in simple English.",
        "Adversarial testing means probing a system with difficult, misleading, or risky inputs to find where it fails before deployment.",
    ),
    instruction_record(
        "Explain why a lightweight controller can help a modular AI system.",
        "A lightweight controller can check route confidence, safety risk, and ambiguity before expensive specialist modules are loaded.",
    ),
    instruction_record(
        "What is a controlled cascade in SwarmLM?",
        "A controlled cascade routes the user request, validates the route with a controller, then executes with the selected specialist Soul.",
    ),
]


OFF_DOMAIN = [
    route_request("Write a Python fibonacci function.", "code_align_v3"),
    route_request("Write a Python primality test.", "code_align_v3"),
    route_request("Write SQL to count users by country.", "code_align_v3"),
    route_request("Solve 3x + 5 = 20.", "math_align_v2"),
    route_request("A train travels 120 km in 2 hours. What is the average speed?", "math_align_v2"),
    route_request("Create a step-by-step plan to evaluate the model.", "agentic_v2"),
    route_request("Design an adversarial test plan for agentic_v2.", "agentic_v2"),
]


def build_records() -> list[str]:
    records: list[str] = []
    records.extend(CORE_TEXT * 6)
    records.extend(NO_CODE_EXPLANATIONS * 10)
    records.extend(GENERAL_EXPLANATIONS * 5)
    records.extend(OFF_DOMAIN * 3)
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
    parser.add_argument("--target_mb", type=int, default=512)
    parser.add_argument("--seed", type=int, default=56)
    args = parser.parse_args()

    out = args.base_dir / "data" / "align_v3" / "text_align_v3.bin"
    repeat_write(out, build_records(), args.target_mb * 1024**2, args.seed)


if __name__ == "__main__":
    main()
