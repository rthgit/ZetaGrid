#!/usr/bin/env python3
"""
Build Code Align v3 dataset.

Purpose:
- fix code_align_v2 regressions observed in the FRO-controlled full cascade;
- preserve working primality generation;
- recover fibonacci;
- add SQL generation under the code_v2 route;
- avoid average-function collapse across unrelated code prompts.

The dataset is intentionally prompt-compatible with the cascade prompts used in
EVAL_SWARMLM_FRO_CASCADE.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


def file_record(language: str, task: str, instruction: str, body: str) -> str:
    return (
        f"\n<|file|> language={language} task={task}\n"
        f"# Instruction: {instruction}\n"
        f"{body.rstrip()}\n"
        "<|endfile|>\n"
    )


def sql_record(instruction: str, body: str) -> str:
    return (
        "\n<|file|> language=sql task=align_v3\n"
        f"-- Instruction: {instruction}\n"
        f"{body.rstrip()}\n"
        "<|endfile|>\n"
    )


def continuation_record(language: str, task: str, prefix: str, continuation: str) -> str:
    return f"\n<|file|> language={language} task={task}\n{prefix}{continuation.rstrip()}\n<|endfile|>\n"


def route_request(instruction: str, target: str) -> str:
    return f"\n<|instruction|>\nUser: {instruction}\nAssistant: ROUTE_REQUESTED: {target}\n<|endinstruction|>\n"


PYTHON_RECORDS = [
    file_record(
        "python",
        "align_v3",
        "Write a Python fibonacci function.",
        """def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a""",
    ),
    continuation_record(
        "python",
        "fro_cascade",
        "# Instruction: Write a Python fibonacci function.\ndef fibonacci(n):\n",
        """    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a recursive fibonacci function.",
        """def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a Python factorial function.",
        """def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a Python primality test.",
        """def is_prime(n):
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
    return True""",
    ),
    continuation_record(
        "python",
        "fro_cascade",
        "# Instruction: Write a Python primality test.\ndef is_prime(n):\n    if n < 2:\n",
        """        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a safe average function.",
        """def average(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a function to parse ROUTE from text.",
        '''def parse_route(text):
    for line in text.splitlines():
        if line.startswith("ROUTE:"):
            return line.split(":", 1)[1].strip()
    return ""''',
    ),
    file_record(
        "python",
        "align_v3",
        "Write a JSONL append helper.",
        """import json

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a function that counts words in a string.",
        """def count_words(text):
    words = [part for part in text.split() if part]
    return len(words)""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a function that returns even numbers from a list.",
        """def even_numbers(values):
    return [value for value in values if value % 2 == 0]""",
    ),
    file_record(
        "python",
        "align_v3",
        "Write a small CSV reader function.",
        """import csv

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))""",
    ),
]


SQL_RECORDS = [
    sql_record(
        "Write SQL to count users by country.",
        """SELECT country, COUNT(*) AS user_count
FROM users
GROUP BY country
ORDER BY user_count DESC;""",
    ),
    continuation_record(
        "sql",
        "fro_cascade",
        "-- Instruction: Write SQL to count users by country.\n",
        """SELECT country, COUNT(*) AS user_count
FROM users
GROUP BY country
ORDER BY user_count DESC;""",
    ),
    sql_record(
        "Write SQL to count orders by status.",
        """SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY order_count DESC;""",
    ),
    sql_record(
        "Write SQL to select active users.",
        """SELECT id, email, country
FROM users
WHERE active = TRUE;""",
    ),
    sql_record(
        "Write SQL to calculate revenue by month.",
        """SELECT DATE_TRUNC('month', created_at) AS month, SUM(total) AS revenue
FROM orders
GROUP BY month
ORDER BY month;""",
    ),
    sql_record(
        "Write SQL to join users and orders.",
        """SELECT users.id, users.email, orders.id AS order_id, orders.total
FROM users
JOIN orders ON orders.user_id = users.id;""",
    ),
    sql_record(
        "Write SQL to find duplicate emails.",
        """SELECT email, COUNT(*) AS duplicate_count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;""",
    ),
]


OFF_DOMAIN = [
    route_request("Explain FRO in simple English.", "text_align_v2"),
    route_request("Explain what a primality test is, no code.", "text_align_v2"),
    route_request("Solve 3x + 5 = 20.", "math_align_v2"),
    route_request("A train travels 120 km in 2 hours. What is the average speed?", "math_align_v2"),
    route_request("Create a step-by-step plan to evaluate the model.", "agentic_v2"),
]


def build_records() -> list[str]:
    records: list[str] = []

    # Oversample the exact failures and desired preserves.
    exact_fixes = [
        PYTHON_RECORDS[0],
        PYTHON_RECORDS[1],
        PYTHON_RECORDS[4],
        PYTHON_RECORDS[5],
        SQL_RECORDS[0],
        SQL_RECORDS[1],
    ]
    records.extend(exact_fixes * 8)
    records.extend(PYTHON_RECORDS * 4)
    records.extend(SQL_RECORDS * 5)
    records.extend(OFF_DOMAIN * 2)
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
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()

    out = args.base_dir / "data" / "align_v3" / "code_align_v3.bin"
    repeat_write(out, build_records(), args.target_mb * 1024**2, args.seed)


if __name__ == "__main__":
    main()
