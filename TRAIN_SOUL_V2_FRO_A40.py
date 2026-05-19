#!/usr/bin/env python3
"""
ZetaGrid Soul v2 trainer for A40-class GPUs.

Targets:
- text_v2: resume from existing text Soul and continue on FineWeb/FineWeb-Edu bins.
- code_v2: resume from existing code Soul and continue on larger permissive code bins.
- math_v1: fork from text Soul and train on math/reasoning bins.
- text_align_v1/code_align_v1/math_align_v1: low-LR alignment passes.
- instruction_v1/agentic_v1/orchestrator_v1: functional Souls for SwarmLM routing experiments.
- fro_controller_v1: lightweight controller Soul for route confidence, safety, fallback, and validation.

The model stays in the Genome/Soul regime: Genome weights are frozen buffers,
while LoRA, norms, scales, and embeddings are trainable. FRO is used instead of
AdamW to reduce optimizer memory and expose resonance/shock logs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fro_optimizer import FRO


VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
KERNEL_SIZE = 3
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]
SOUL_MODES = [
    "text_v2",
    "code_v2",
    "math_v1",
    "text_align_v1",
    "text_align_v3",
    "text_align_v4",
    "text_align_v5",
    "code_align_v1",
    "code_align_v3",
    "math_align_v1",
    "instruction_v1",
    "agentic_v1",
    "orchestrator_v1",
    "fro_controller_v1",
    "fro_controller_v1b",
    "fro_controller_v2",
]


def default_base_dir() -> Path:
    if os.name == "nt":
        return Path("E:/ZETAGRID")
    return Path("/workspace/zetagrid_50b")


def resolve_defaults(mode: str, base_dir: Path) -> dict[str, Path | str]:
    data_names = {
        "text_v2": "data/text_v2/fineweb_text_v2.bin",
        "code_v2": "data/code_v2/code_v2.bin",
        "math_v1": "data/math_v1/math_v1.bin",
        "text_align_v1": "data/align_v1/text_align_v1.bin",
        "text_align_v3": "data/align_v3/text_align_v3.bin",
        "text_align_v4": "data/align_v4/text_align_v4.bin",
        "text_align_v5": "data/align_v5/text_align_v5.bin",
        "code_align_v1": "data/align_v1/code_align_v1.bin",
        "code_align_v3": "data/align_v3/code_align_v3.bin",
        "math_align_v1": "data/align_v1/math_align_v1.bin",
        "instruction_v1": "data/swarmlm_v1/instruction_v1.bin",
        "agentic_v1": "data/swarmlm_v1/agentic_v1.bin",
        "orchestrator_v1": "data/swarmlm_v1/orchestrator_v1.bin",
        "fro_controller_v1": "data/swarmlm_v4/fro_controller_v1.bin",
        "fro_controller_v1b": "data/swarmlm_v4/fro_controller_v1b.bin",
        "fro_controller_v2": "data/swarmlm_v4/fro_controller_v2.bin",
    }
    init_ckpts = {
        "text_v2": "zeta25b_v4_expanded_FINAL.pt",
        "code_v2": "zeta25b_code_FINAL.pt",
        "math_v1": "zeta25b_v4_expanded_FINAL.pt",
        "text_align_v1": "checkpoints/text_v2/TEXT_V2_BEST_0p9111.pt",
        "text_align_v3": "checkpoints/text_align_v2/TEXT_ALIGN_V2.pt",
        "text_align_v4": "checkpoints/text_align_v2/TEXT_ALIGN_V2.pt",
        "text_align_v5": "checkpoints/text_align_v2/TEXT_ALIGN_V2.pt",
        "code_align_v1": "checkpoints/code_v2/CODE_V2_SMOKE.pt",
        "code_align_v3": "checkpoints/code_align_v2/CODE_ALIGN_V2.pt",
        "math_align_v1": "checkpoints/math_v1/MATH_V1_SMOKE.pt",
        "instruction_v1": "checkpoints/text_align_v1/TEXT_V2_ALIGN.pt",
        "agentic_v1": "checkpoints/instruction_v1/INSTRUCTION_V1_SMOKE.pt",
        "orchestrator_v1": "checkpoints/instruction_v1/INSTRUCTION_V1_SMOKE.pt",
        "fro_controller_v1": "checkpoints/orchestrator_v3b/ORCHESTRATOR_V3B.pt",
        "fro_controller_v1b": "checkpoints/orchestrator_v3b/ORCHESTRATOR_V3B.pt",
        "fro_controller_v2": "checkpoints/orchestrator_v3b/ORCHESTRATOR_V3B.pt",
    }
    return {
        "genome": base_dir / "zetagrid_25b_production.npy",
        "data": base_dir / data_names[mode],
        "init": base_dir / init_ckpts[mode],
        "save_dir": base_dir / "checkpoints" / mode,
    }


class GenomeWeightBank:
    def __init__(self, genome_path: Path, dtype: torch.dtype, device: str):
        print(f"[GENOME] loading {genome_path}")
        raw = np.load(genome_path, mmap_mode="r")
        print(f"[GENOME] mmap opened: {raw.shape} {raw.dtype}")
        print("[GENOME] copying int8 to GPU, then converting on GPU...")
        raw_cpu = torch.from_numpy(raw)
        self.data = raw_cpu.to(device=device, non_blocking=False).to(dtype=dtype)
        self.offset = 0
        del raw_cpu
        del raw
        gc.collect()
        if device == "cuda":
            print(f"[GENOME] VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def get_weight(self, out_features: int, in_features: int) -> torch.Tensor:
        n = out_features * in_features
        if self.offset + n > len(self.data):
            self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(out_features, in_features)
        self.offset += n
        scale = 1.0 / math.sqrt(in_features * 0.1)
        return (chunk * scale).contiguous()

    def get_conv_weight(self, channels: int, kernel_size: int) -> torch.Tensor:
        n = channels * kernel_size
        if self.offset + n > len(self.data):
            self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(channels, 1, kernel_size)
        self.offset += n
        return (chunk * (1.0 / math.sqrt(kernel_size))).contiguous()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x.float() * rms).to(x.dtype) * self.w.to(x.dtype)).to(x.dtype)


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


class TCNLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, kernel_size: int, dilation: int, rank: int, bank: GenomeWeightBank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.norm = RMSNorm(d_model)
        self.register_buffer("w_in", bank.get_weight(2 * d_ff, d_model), persistent=False)
        self.register_buffer("w_dw", bank.get_conv_weight(d_ff, kernel_size), persistent=False)
        self.register_buffer("w_out", bank.get_weight(d_model, d_ff), persistent=False)
        self.lora_in = LoRA(d_model, 2 * d_ff, rank)
        self.lora_out = LoRA(d_ff, d_model, rank)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        ag = F.linear(x_norm, self.w_in) + self.lora_in(x_norm)
        a, gate = ag.chunk(2, dim=-1)
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, self.w_dw, groups=D_FF, dilation=self.dilation)
        a = a.transpose(1, 2)
        y = F.silu(a) * torch.sigmoid(gate)
        out = F.linear(y, self.w_out) + self.lora_out(y)
        return residual + out * self.scale


class ZetaGridSoul(nn.Module):
    def __init__(self, bank: GenomeWeightBank, n_layers: int, rank: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        self.layers = nn.ModuleList(
            [
                TCNLayer(D_MODEL, D_FF, KERNEL_SIZE, DILATION_CYCLE[i % len(DILATION_CYCLE)], rank, bank)
                for i in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(D_MODEL)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(self.dtype)
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm_f(x)
        logits = F.linear(x.float(), self.emb.weight.float())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 220, temperature: float = 0.7, top_k: int = 40) -> str:
        self.eval()
        device = next(self.parameters()).device
        idx = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
        for _ in range(max_new):
            idx_crop = idx[:, -1024:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return bytes(idx[0].detach().cpu().tolist()).decode("utf-8", errors="replace")


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    keep = {}
    for key, value in model.state_dict().items():
        if (
            "lora" in key
            or "norm" in key
            or "scale" in key
            or key.startswith("emb.")
            or key.startswith("pos_emb.")
            or key.startswith("norm_f.")
        ):
            keep[key] = value.detach().cpu()
    return keep


def load_init_checkpoint(model: nn.Module, path: Path, device: str) -> tuple[int, float]:
    if not path.exists():
        print(f"[INIT] no checkpoint found at {path}; starting from initialized Soul")
        return 0, 99.0
    print(f"[INIT] loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INIT] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    return int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0, float(ckpt.get("loss", 99.0)) if isinstance(ckpt, dict) else 99.0


def load_data(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"dataset bin not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    if len(data) < 4096:
        raise ValueError(f"dataset too small: {path}")
    print(f"[DATA] {path} | {len(data) / 1e9:.2f} GB")
    return data


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: str):
    starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=SOUL_MODES, default="text_v2")
    parser.add_argument("--base_dir", type=Path, default=default_base_dir())
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--init_ckpt", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fro_alpha", type=float, default=0.25)
    parser.add_argument("--fro_gamma", type=float)
    parser.add_argument("--eval_on_save", action="store_true")
    parser.add_argument("--write_final", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = resolve_defaults(args.mode, args.base_dir)
    genome = args.genome or defaults["genome"]
    data_path = args.data or defaults["data"]
    init_ckpt = args.init_ckpt or defaults["init"]
    save_dir = args.save_dir or defaults["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[RUN] mode={args.mode} device={device} dtype={dtype} rank={args.rank} layers={args.layers}")
    print(f"[RUN] genome={genome}")
    print(f"[RUN] init={init_ckpt}")
    print(f"[RUN] data={data_path}")
    print(f"[RUN] save_dir={save_dir}")

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
        print("[INIT] reset step/best_loss for a fresh mode run")
    data = load_data(data_path)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    print(f"[MODEL] trainable params: {trainable / 1e6:.1f}M")

    optimizer = FRO(
        params,
        lr=args.lr,
        betas=(0.9, 0.98),
        scales=(0.1, 0.01, 0.001),
        alpha=args.fro_alpha,
        gamma=args.fro_gamma if args.fro_gamma is not None else (0.7 if "math" in args.mode else 0.5),
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
            x, y = get_batch(data, args.batch_size, args.seq_len, device)
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
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
                    "mode": args.mode,
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
                    "mode": args.mode,
                    "data": str(data_path),
                    "model": trainable_state_dict(model),
                },
                ckpt_path,
            )
            print(f"[SAVE] {ckpt_path}")
            if args.eval_on_save:
                model.eval()
                prompt = "The future of efficient AI is" if "text" in args.mode or "instruction" in args.mode else "def fibonacci(n):\n"
                if "math" in args.mode:
                    prompt = "Problem: If x + 3 = 7, then x ="
                if "agentic" in args.mode:
                    prompt = "Task: Build a small evaluation plan.\nPlan:\n"
                if "orchestrator" in args.mode:
                    prompt = "USER_REQUEST: Write Python code to solve 3x+5=20.\nROUTE:"
                if "fro_controller" in args.mode:
                    prompt = "<|fro_control|>\nMODE: pre_route\nUSER_REQUEST: Write a Python function.\nORCHESTRATOR_ROUTE: code_v2\nCONTROL:"
                print(model.generate(prompt, max_new=180)[:400])
                model.train()

    latest_path = save_dir / "latest.pt"
    torch.save(
        {
            "step": args.steps,
            "loss": best_loss,
            "mode": args.mode,
            "data": str(data_path),
            "model": trainable_state_dict(model),
        },
        latest_path,
    )
    print(f"[DONE] latest={latest_path} best_loss={best_loss:.4f}")
    if args.write_final:
        final_path = save_dir / "FINAL.pt"
        torch.save(
            {
                "step": args.steps,
                "loss": best_loss,
                "mode": args.mode,
                "data": str(data_path),
                "model": trainable_state_dict(model),
            },
            final_path,
        )
        print(f"[DONE] final={final_path}")
    else:
        print("[DONE] FINAL.pt not written automatically; copy latest.pt to FINAL.pt only for release.")


if __name__ == "__main__":
    main()
