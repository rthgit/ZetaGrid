#!/usr/bin/env python3
"""
Masked instruction trainer for Text Align v6.

This uses the same Genome/Soul model and FRO optimizer, but computes loss only
on the answer span. It is meant to repair the text specialist after v5 showed
low flat-LM loss but poor prompt/answer binding.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fro_optimizer import FRO
from TRAIN_SOUL_V2_FRO_A40 import GenomeWeightBank, ZetaGridSoul, load_init_checkpoint, trainable_state_dict


def encode_example(prompt: str, answer: str, seq_len: int) -> tuple[list[int], list[int]]:
    prefix = f"<|instruction|>\nUser: {prompt}\nAssistant: "
    suffix = f"{answer.rstrip()}\n<|endinstruction|>\n"
    prefix_ids = list(prefix.encode("utf-8", errors="ignore"))
    suffix_ids = list(suffix.encode("utf-8", errors="ignore"))
    ids = prefix_ids + suffix_ids
    if len(ids) > seq_len + 1:
        ids = ids[: seq_len + 1]
    x = ids[:-1]
    y = ids[1:]
    labels = [-100] * len(y)
    answer_start = max(len(prefix_ids) - 1, 0)
    for i in range(answer_start, len(y)):
        labels[i] = y[i]
    return x, labels


def load_examples(path: Path, seq_len: int) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(encode_example(row["prompt"], row["answer"], seq_len))
    if not examples:
        raise ValueError(f"empty dataset: {path}")
    return examples


def batch_examples(examples: list[tuple[list[int], list[int]]], batch_size: int, seq_len: int, device: str):
    picked = random.choices(examples, k=batch_size)
    x = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(picked):
        n = min(len(ids), seq_len)
        x[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(labels[:n], dtype=torch.long)
    return x.to(device), y.to(device)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/workspace/zetagrid_50b"))
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--init_ckpt", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fro_alpha", type=float, default=0.25)
    parser.add_argument("--fro_gamma", type=float, default=0.5)
    parser.add_argument("--train_last_layers", type=int, default=8)
    parser.add_argument("--train_embeddings", action="store_true")
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--seed", type=int, default=606)
    return parser.parse_args()


def configure_trainable(model: ZetaGridSoul, n_layers: int, train_last_layers: int, train_embeddings: bool, train_all: bool):
    if train_all:
        for param in model.parameters():
            param.requires_grad = True
        return

    cutoff = max(0, n_layers - train_last_layers)
    for name, param in model.named_parameters():
        param.requires_grad = False
        if train_embeddings and (name.startswith("emb.") or name.startswith("pos_emb.")):
            param.requires_grad = True
            continue
        if name.startswith("norm_f."):
            param.requires_grad = True
            continue
        if not name.startswith("layers."):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if layer_idx < cutoff:
            continue
        if ".lora_" in name or ".norm." in name or name.endswith(".scale"):
            param.requires_grad = True


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    genome = args.genome or (base_dir / "zetagrid_25b_production.npy")
    data_path = args.data or (base_dir / "data" / "align_v6" / "text_align_v6.jsonl")
    init_ckpt = args.init_ckpt or (base_dir / "checkpoints" / "text_align_v2" / "TEXT_ALIGN_V2.pt")
    save_dir = args.save_dir or (base_dir / "checkpoints" / "text_align_v6")
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[RUN] masked_text_align_v6 device={device} dtype={dtype} rank={args.rank} layers={args.layers}")
    print(f"[RUN] genome={genome}")
    print(f"[RUN] init={init_ckpt}")
    print(f"[RUN] data={data_path}")
    print(f"[RUN] save_dir={save_dir}")

    examples = load_examples(data_path, args.seq_len)
    print(f"[DATA] examples={len(examples)}")

    bank = GenomeWeightBank(genome, dtype=dtype, device=device)
    model = ZetaGridSoul(bank, n_layers=args.layers, rank=args.rank, dtype=dtype).to(device)
    del bank.data
    del bank
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"[MODEL] VRAM after build: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    latest_ckpt = save_dir / "latest.pt"
    resume_source = latest_ckpt if latest_ckpt.exists() else init_ckpt
    resuming_latest = latest_ckpt.exists()
    if resuming_latest:
        print(f"[RESUME] using overwrite checkpoint {latest_ckpt}")
    start_step, best_loss = load_init_checkpoint(model, resume_source, device)
    if not resuming_latest:
        start_step = 0
        best_loss = 99.0
        print("[INIT] reset step/best_loss for masked run")

    configure_trainable(model, args.layers, args.train_last_layers, args.train_embeddings, args.train_all)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    print(
        f"[MODEL] trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M "
        f"train_all={args.train_all} train_last_layers={args.train_last_layers} "
        f"train_embeddings={args.train_embeddings}"
    )
    optimizer = FRO(
        params,
        lr=args.lr,
        betas=(0.9, 0.98),
        scales=(0.1, 0.01, 0.001),
        alpha=args.fro_alpha,
        gamma=args.fro_gamma,
        weight_decay=0.0,
    )

    def scheduled_lr(step: int) -> float:
        if step < args.warmup:
            return args.lr * max(step, 1) / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    metrics_path = save_dir / "fro_metrics.jsonl"
    t0 = time.time()
    rolling_loss = 0.0
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        lr = scheduled_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, labels = batch_examples(examples, args.batch_size, args.seq_len, device)
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            accum_loss += float(scaled_loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        rolling_loss += accum_loss

        if step % args.log_every == 0:
            summary = optimizer.resonance_summary()
            avg_loss = rolling_loss / args.log_every
            best_loss = min(best_loss, avg_loss)
            elapsed = time.time() - t0
            print(
                f"step={step} loss={avg_loss:.4f} best={best_loss:.4f} lr={lr:.2e} "
                f"R={summary['resonance']:.3f} rho={summary['rho']:.3f} elapsed={elapsed/60:.1f}m"
            )
            write_jsonl(
                metrics_path,
                {
                    "step": step,
                    "mode": "text_align_v6_masked",
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "lr": lr,
                    "resonance": summary["resonance"],
                    "rho": summary["rho"],
                    "elapsed_sec": elapsed,
                },
            )
            rolling_loss = 0.0

        if step % args.save_every == 0:
            ckpt_path = save_dir / "latest.pt"
            torch.save(
                {
                    "step": step,
                    "loss": best_loss,
                    "mode": "text_align_v6_masked",
                    "data": str(data_path),
                    "model": trainable_state_dict(model),
                },
                ckpt_path,
            )
            print(f"[SAVE] {ckpt_path}")

    latest_path = save_dir / "latest.pt"
    torch.save(
        {
            "step": args.steps,
            "loss": best_loss,
            "mode": "text_align_v6_masked",
            "data": str(data_path),
            "model": trainable_state_dict(model),
        },
        latest_path,
    )
    print(f"[DONE] latest={latest_path} best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
