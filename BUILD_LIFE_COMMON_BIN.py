#!/usr/bin/env python3
"""
Convert life_common JSONL chat rows to byte-level .bin corpus.

This is compatible with training scripts that read uint8 tokens directly.
"""

import argparse
import json
from pathlib import Path
from typing import List


def row_to_text(obj: dict) -> str:
    if "messages" in obj and isinstance(obj["messages"], list):
        user = ""
        assistant = ""
        for m in obj["messages"]:
            role = str(m.get("role", "")).lower()
            content = str(m.get("content", ""))
            if role == "user":
                user = content
            elif role == "assistant":
                assistant = content
        return f"User: {user}\nAssistant: {assistant}\n\n"

    text = str(obj.get("text", "")).strip()
    if text:
        return text + "\n\n"
    return ""


def convert_jsonl_to_bin(input_path: Path, output_path: Path, repeat: int = 1) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    blocks: List[bytes] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = row_to_text(obj)
            if not text:
                continue
            blob = text.encode("utf-8", errors="ignore")
            if blob:
                blocks.append(blob)

    with output_path.open("wb") as out:
        for _ in range(max(1, repeat)):
            for blob in blocks:
                out.write(blob)
                total_bytes += len(blob)

    return total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert life_common_it.jsonl to uint8 .bin.")
    parser.add_argument("--input", default="datasets/life_common_it.jsonl", help="Input JSONL.")
    parser.add_argument("--output", default="datasets/life_common_it.bin", help="Output .bin file.")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat corpus N times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    total_bytes = convert_jsonl_to_bin(in_path, out_path, repeat=args.repeat)
    print(f"Bin dataset: {out_path}")
    print(f"Total bytes: {total_bytes}")
    print(f"Approx tokens (uint8): {total_bytes}")


if __name__ == "__main__":
    main()
