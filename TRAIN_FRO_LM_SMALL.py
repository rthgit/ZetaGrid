#!/usr/bin/env python3
"""
Train FRO-LM Small from scratch.

This is intentionally not a ZetaGrid Soul. It is a lightweight byte-level
controller model trained with the FRO optimizer. Its job is to sit beside
SwarmLM and emit route-compatible control decisions:

ROUTE: ...
ACTION: accept | fallback | split | reject | revise
CONFIDENCE: high | medium | low
RISK: low | high
REASON: ...
<|endfro|>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fro_optimizer import FRO


VOCAB_SIZE = 256


def default_base_dir() -> Path:
    return Path("/workspace/zetagrid_50b")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class CausalTCNBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(dim, 2 * dim, kernel_size, dilation=dilation)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h_conv = h.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation
        h_conv = F.pad(h_conv, (pad, 0))
        gate, value = self.conv(h_conv).transpose(1, 2).chunk(2, dim=-1)
        h = torch.tanh(value) * torch.sigmoid(gate)
        h = self.down(F.silu(self.up(h)))
        return residual + self.dropout(h)


class FROSmallLM(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        d_ff: int,
        seq_len: int,
        kernel_size: int,
        dropout: float,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = {
            "d_model": d_model,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "seq_len": seq_len,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "vocab_size": vocab_size,
        }
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    d_model,
                    d_ff,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, time_steps = idx.shape
        if time_steps > self.seq_len:
            idx = idx[:, -self.seq_len :]
            if targets is not None:
                targets = targets[:, -self.seq_len :]
            time_steps = self.seq_len
        pos = torch.arange(time_steps, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int, device: str, temperature: float = 0.2, top_k: int = 8) -> str:
        self.eval()
        idx = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.seq_len :])
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            text = "".join(chr(int(x)) for x in idx[0].tolist())
            if "<|endfro|>" in text:
                break
        return "".join(chr(int(x)) for x in idx[0].tolist())


def load_data(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"dataset bin not found: {path}")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    print(f"[DATA] {path} | {path.stat().st_size / 1024**2:.1f} MB")
    return data


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i : i + seq_len], dtype=np.uint8).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1 : i + seq_len + 1], dtype=np.uint8).astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_checkpoint(path: Path, model: FROSmallLM, step: int, best_loss: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_loss": best_loss,
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> tuple[FROSmallLM, int, float]:
    ckpt = torch.load(path, map_location=device)
    model = FROSmallLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, int(ckpt.get("step", 0)), float(ckpt.get("best_loss", 99.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--data", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fro_alpha", type=float, default=0.25)
    parser.add_argument("--fro_gamma", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--eval_on_save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    data_path = args.data or (args.base_dir / "data" / "swarmlm_v4" / "fro_controller_v2.bin")
    save_dir = args.save_dir or (args.base_dir / "checkpoints" / "fro_lm_small_v0")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] device={device} dtype={dtype} layers={args.layers} d_model={args.d_model} d_ff={args.d_ff}")
    print(f"[RUN] data={data_path}")
    print(f"[RUN] save_dir={save_dir}")

    if args.resume:
        model, start_step, best_loss = load_checkpoint(args.resume, device)
        print(f"[RESUME] {args.resume} step={start_step} best_loss={best_loss:.4f}")
    else:
        model = FROSmallLM(
            d_model=args.d_model,
            n_layers=args.layers,
            d_ff=args.d_ff,
            seq_len=args.seq_len,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
        start_step = 0
        best_loss = 99.0
        print("[INIT] random FRO-LM Small")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable params: {trainable / 1e6:.1f}M")
    if device == "cuda":
        print(f"[MODEL] VRAM after build: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    data = load_data(data_path)
    params = [p for p in model.parameters() if p.requires_grad]
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
    rolling = 0.0
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        lr = scheduled_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(data, args.batch_size, args.seq_len, device)
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                scaled = loss / args.grad_accum
            scaled.backward()
            accum += float(scaled.detach().cpu())
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        rolling += accum

        if step % args.log_every == 0:
            avg = rolling / args.log_every
            best_loss = min(best_loss, avg)
            elapsed = time.time() - t0
            summary = optimizer.resonance_summary()
            print(
                f"step={step} loss={avg:.4f} best={best_loss:.4f} lr={lr:.2e} "
                f"R={summary['resonance']:.3f} rho={summary['rho']:.3f} elapsed={elapsed/60:.1f}m"
            )
            write_jsonl(
                metrics_path,
                {
                    "step": step,
                    "loss": avg,
                    "best_loss": best_loss,
                    "lr": lr,
                    "resonance": summary["resonance"],
                    "rho": summary["rho"],
                    "elapsed_sec": elapsed,
                },
            )
            rolling = 0.0

        if step % args.save_every == 0:
            ckpt_path = save_dir / "latest.pt"
            save_checkpoint(ckpt_path, model, step, best_loss, args)
            print(f"[SAVE] {ckpt_path}")
            if args.eval_on_save:
                prompt = (
                    "<|fro_control|>\nMODE: pre_route\nUSER_REQUEST: Explain what a primality test is, no code.\n"
                    "ORCHESTRATOR_ROUTE: code_v2\n"
                )
                print(model.generate(prompt, max_new=160, device=device)[:600])
                model.train()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    latest = save_dir / "latest.pt"
    save_checkpoint(latest, model, args.steps, best_loss, args)
    print(f"[DONE] latest={latest} best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
