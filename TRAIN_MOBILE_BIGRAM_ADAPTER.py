#!/usr/bin/env python3
"""
Train a lightweight bigram adapter for byte-level mobile generation.

The adapter learns P(next_byte | prev_byte) from a byte corpus and stores
log-probabilities used as an additive bias during decoding.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def train_bigram(byte_data: np.ndarray, smoothing: float) -> np.ndarray:
    counts = np.zeros((256, 256), dtype=np.float64)
    prev = byte_data[:-1]
    nxt = byte_data[1:]
    np.add.at(counts, (prev, nxt), 1.0)

    counts += float(smoothing)
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(row_sum, 1e-12)
    log_probs = np.log(np.maximum(probs, 1e-12)).astype(np.float32)
    return log_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bigram adapter for mobile byte-level decoding.")
    parser.add_argument(
        "--input-bin",
        default="datasets/life_common_it.bin",
        help="Input byte corpus (.bin, uint8 stream).",
    )
    parser.add_argument(
        "--output",
        default="adapters/life_common_bigram_adapter.npz",
        help="Output adapter path (.npz).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.2,
        help="Additive smoothing value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_bin)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input bin not found: {in_path}")

    data = np.fromfile(in_path, dtype=np.uint8)
    if data.size < 2:
        raise RuntimeError("Input bin is too small to train bigram adapter.")

    log_bigram = train_bigram(data, smoothing=float(args.smoothing))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        log_bigram=log_bigram,
        smoothing=np.float32(args.smoothing),
        total_bytes=np.int64(data.size),
        source=os.fspath(in_path),
    )

    print(f"Adapter saved: {out_path}")
    print(f"Source bytes: {int(data.size)}")
    print(f"Smoothing: {float(args.smoothing):.3f}")


if __name__ == "__main__":
    main()
