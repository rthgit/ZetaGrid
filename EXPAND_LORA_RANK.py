#!/usr/bin/env python3
"""Expand a Soul checkpoint from one LoRA rank to a larger rank.

Existing channels are copied exactly. New A channels are initialized with a
small random value and new B channels are initialized to zero. This preserves
the initial function while keeping the new channels trainable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--old_rank", type=int, default=512)
    parser.add_argument("--new_rank", type=int, default=1024)
    parser.add_argument("--extra_std", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def rand_like(shape: tuple[int, ...], dtype: torch.dtype, seed: int, std: float) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * std).to(dtype=dtype)


def main() -> None:
    args = parse_args()
    if args.new_rank <= args.old_rank:
        raise ValueError("--new_rank must be greater than --old_rank")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    expanded = {}
    copied = 0
    expanded_a = 0
    expanded_b = 0
    for idx, (key, value) in enumerate(state.items()):
        if not torch.is_tensor(value):
            expanded[key] = value
            continue

        lower = key.lower()
        is_lora = "lora" in lower
        if is_lora and value.ndim == 2 and value.shape[0] == args.old_rank:
            new_value = torch.zeros((args.new_rank, value.shape[1]), dtype=value.dtype)
            new_value[: args.old_rank, :] = value
            new_value[args.old_rank :, :] = rand_like(
                (args.new_rank - args.old_rank, value.shape[1]),
                value.dtype,
                args.seed + idx,
                args.extra_std,
            )
            expanded[key] = new_value
            expanded_a += 1
        elif is_lora and value.ndim == 2 and value.shape[1] == args.old_rank:
            new_value = torch.zeros((value.shape[0], args.new_rank), dtype=value.dtype)
            new_value[:, : args.old_rank] = value
            expanded[key] = new_value
            expanded_b += 1
        else:
            expanded[key] = value
            copied += 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "loss": 99.0,
            "mode": f"expand_lora_rank_{args.old_rank}_to_{args.new_rank}",
            "source": str(args.src),
            "model": expanded,
        },
        args.dst,
    )
    print(f"[OK] {args.src} -> {args.dst}")
    print(f"[OK] copied={copied} expanded_A={expanded_a} expanded_B={expanded_b}")
    print(f"[OK] size_gb={args.dst.stat().st_size / 1024**3:.3f}")


if __name__ == "__main__":
    main()
