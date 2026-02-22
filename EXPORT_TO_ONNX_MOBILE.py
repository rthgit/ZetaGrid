#!/usr/bin/env python3
"""
EXPORT_TO_ONNX_MOBILE.py
========================
Exports RTH-LM QULP checkpoints to ONNX with a mobile-first profile.

The exporter supports "superfast" mode by selecting a reduced subset of
layers and a short sequence length, which is much more practical on phones.

Example:
    python EXPORT_TO_ONNX_MOBILE.py --model v4 --input E:/ZETAGRID/rth_lm_25b_v4.qulp --profile superfast
"""

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# MODEL SHAPE DEFAULTS (must match QULP model layout)
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
KERNEL_SIZE = 3
GROUP_SIZE = 128

PROFILE_PRESETS = {
    "superfast": {"layers": 8, "seq_len": 64},
    "mid3b": {"layers": 15, "seq_len": 96},
    "balanced": {"layers": 16, "seq_len": 128},
    "quality": {"layers": 32, "seq_len": 256},
}


def _prod(shape: Iterable[int]) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n


def _select_layer_indices(total_layers: int, wanted_layers: int) -> List[int]:
    if wanted_layers >= total_layers:
        return list(range(total_layers))

    if wanted_layers == 1:
        return [0]

    indices: List[int] = []
    for i in range(wanted_layers):
        # Even sampling across depth (e.g. 128 -> 8 gives [0,18,36,...,127]).
        idx = round(i * (total_layers - 1) / (wanted_layers - 1))
        indices.append(int(idx))
    return indices


def _estimate_params(layers: int, seq_len: int) -> int:
    per_layer = (2 * D_FF * D_MODEL) + (D_FF * KERNEL_SIZE) + (D_MODEL * D_FF) + D_MODEL + 1
    base = (VOCAB_SIZE * D_MODEL) + (seq_len * D_MODEL) + D_MODEL
    return int(base + layers * per_layer)


def _summarize_external_tensors(onnx_path: str) -> Tuple[int, int, List[str]]:
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    root = Path(onnx_path).resolve().parent

    files = set()
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = None
        for entry in init.external_data:
            if entry.key == "location":
                location = entry.value
                break
        if location:
            files.add(location)

    total_bytes = 0
    missing: List[str] = []
    for rel in sorted(files):
        p = root / rel
        if p.is_file():
            total_bytes += p.stat().st_size
        else:
            missing.append(rel)

    return len(files), total_bytes, missing


class Dequantize2Bit(nn.Module):
    """
    Reconstructs float weights from packed 2-bit QULP tensors.
    QULP stores:
      - q: packed uint8 (4 values per byte)
      - s: fp16 scales, typically per group of 128 values
      - z: fp16 zero points, per group
      - sh: original tensor shape
      - p: padding count
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        shape: Iterable[int],
        pad: int = 0,
        out_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.register_buffer("packed", packed_weight.to(torch.uint8).flatten())
        self.register_buffer("scales", scales.to(torch.float32).flatten())
        self.register_buffer("zeros", zeros.to(torch.float32).flatten())
        self.single_scale = bool(self.scales.numel() == 1)
        self.target_shape = tuple(int(v) for v in shape)
        self.pad = int(pad)
        self.group_size = GROUP_SIZE
        self.out_dtype = out_dtype
        self.logical_numel = _prod(self.target_shape)

    def _unpack_2bit(self) -> torch.Tensor:
        # Use arithmetic decomposition instead of bitwise ops.
        # Legacy torch->ONNX exporter fails on int bitwise AND.
        b = self.packed.to(torch.float32).unsqueeze(-1)
        d0 = torch.remainder(torch.floor(b / 64.0), 4.0)
        d1 = torch.remainder(torch.floor(b / 16.0), 4.0)
        d2 = torch.remainder(torch.floor(b / 4.0), 4.0)
        d3 = torch.remainder(b, 4.0)
        unpacked = torch.cat([d0, d1, d2, d3], dim=-1).reshape(-1)

        needed = self.logical_numel + self.pad
        unpacked = unpacked[:needed]
        if self.pad > 0:
            unpacked = unpacked[:-self.pad]
        return unpacked

    def forward(self) -> torch.Tensor:
        q = self._unpack_2bit().to(torch.float32)

        if self.single_scale:
            w = q * self.scales[0] + self.zeros[0]
            return w.view(self.target_shape).to(self.out_dtype)

        # Group-wise dequantization (group 128) used by QULP writer.
        grouped = q.view(-1, self.group_size)
        scales = self.scales.view(-1, 1)
        zeros = self.zeros.view(-1, 1)
        w = grouped * scales + zeros
        return w.reshape(self.target_shape).to(self.out_dtype)


class StaticWeight(nn.Module):
    """Uniform callable wrapper for non-quantized tensors."""

    def __init__(self, tensor: torch.Tensor, out_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.register_buffer("value", tensor.to(torch.float32))
        self.out_dtype = out_dtype

    def forward(self) -> torch.Tensor:
        return self.value.to(self.out_dtype)


def _make_weight_module(
    qulp: Dict[str, object],
    key: str,
    fallback_shape: Tuple[int, ...],
    out_dtype: torch.dtype,
) -> nn.Module:
    item = qulp.get(key)
    if item is None:
        return StaticWeight(torch.zeros(fallback_shape, dtype=torch.float32), out_dtype=out_dtype)

    if isinstance(item, dict) and "q" in item and "s" in item and "z" in item:
        shape = item.get("sh", fallback_shape)
        pad = int(item.get("p", 0))
        return Dequantize2Bit(
            packed_weight=item["q"],
            scales=item["s"],
            zeros=item["z"],
            shape=shape,
            pad=pad,
            out_dtype=out_dtype,
        )

    if isinstance(item, torch.Tensor):
        return StaticWeight(item, out_dtype=out_dtype)

    return StaticWeight(torch.zeros(fallback_shape, dtype=torch.float32), out_dtype=out_dtype)


def _rms_norm_channels_first(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # x: [B, D, T], normalize over D.
    var = x.float().pow(2).mean(dim=1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(var + eps)
    return x_norm.to(x.dtype) * weight.to(x.dtype).view(1, -1, 1)


class RTHMobileBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        qulp: Dict[str, object],
        d_model: int,
        d_ff: int,
        kernel_size: int,
        out_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}"
        self.kernel_size = kernel_size

        self.w_in = _make_weight_module(qulp, f"{prefix}.w_in", (2 * d_ff, d_model), out_dtype=out_dtype)
        self.w_dw = _make_weight_module(qulp, f"{prefix}.w_dw", (d_ff, 1, kernel_size), out_dtype=out_dtype)
        self.w_out = _make_weight_module(qulp, f"{prefix}.w_out", (d_model, d_ff), out_dtype=out_dtype)
        self.norm_w = _make_weight_module(qulp, f"{prefix}.norm.w", (d_model,), out_dtype=out_dtype)
        self.scale = _make_weight_module(qulp, f"{prefix}.scale", (1,), out_dtype=out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = _rms_norm_channels_first(x, self.norm_w())

        w_in = self.w_in().unsqueeze(-1)  # [2*FF, D, 1]
        ag = F.conv1d(x, w_in)
        a, g = ag.chunk(2, dim=1)

        w_dw = self.w_dw()  # [FF, 1, K]
        a = F.pad(a, (self.kernel_size - 1, 0))
        a = F.conv1d(a, w_dw, groups=w_dw.shape[0])

        y = F.silu(a) * torch.sigmoid(g)

        w_out = self.w_out().unsqueeze(-1)  # [D, FF, 1]
        out = F.conv1d(y, w_out)

        # Works for scalar scale ([] -> [1,1,1]) and channel scale ([D] -> [1,D,1]).
        scale = self.scale().reshape(1, -1, 1)
        scaled = out * scale

        return res + scaled


class RTHMobileModel(nn.Module):
    def __init__(
        self,
        qulp: Dict[str, object],
        layer_indices: List[int],
        seq_len: int,
        out_dtype: torch.dtype = torch.float16,
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        vocab_size: int = VOCAB_SIZE,
        kernel_size: int = KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        emb_weight = qulp.get("emb.weight")
        pos_weight = qulp.get("pos_emb.weight")
        norm_f = qulp.get("norm_f.w")
        if not isinstance(emb_weight, torch.Tensor):
            raise ValueError("Missing emb.weight in QULP model.")
        if not isinstance(pos_weight, torch.Tensor):
            raise ValueError("Missing pos_emb.weight in QULP model.")
        if not isinstance(norm_f, torch.Tensor):
            raise ValueError("Missing norm_f.w in QULP model.")

        emb_weight = emb_weight.to(torch.float32)
        pos_weight = pos_weight.to(torch.float32)
        if pos_weight.shape[0] < seq_len:
            raise ValueError(
                f"Requested seq_len={seq_len}, but pos_emb has only {pos_weight.shape[0]} positions."
            )

        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb.weight = nn.Parameter(emb_weight, requires_grad=False)

        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.pos_emb.weight = nn.Parameter(pos_weight[:seq_len], requires_grad=False)

        self.layers = nn.ModuleList(
            [
                RTHMobileBlock(
                    layer_idx=src_idx,
                    qulp=qulp,
                    d_model=d_model,
                    d_ff=d_ff,
                    kernel_size=kernel_size,
                    out_dtype=out_dtype,
                )
                for src_idx in layer_indices
            ]
        )

        self.norm_f = nn.Parameter(norm_f.to(torch.float32), requires_grad=False)
        self.compute_dtype = out_dtype

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        bsz, t = idx.shape
        idx = idx[:, -self.seq_len :]
        t = idx.shape[1]

        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(bsz, t)
        x = (self.emb(idx) + self.pos_emb(pos)).to(self.compute_dtype)
        x = x.transpose(1, 2)  # [B, D, T]

        for layer in self.layers:
            x = layer(x)

        x = _rms_norm_channels_first(x, self.norm_f)
        x_last = x[:, :, -1]
        logits = F.linear(x_last.to(torch.float32), self.emb.weight.to(torch.float32))
        return logits


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    return torch.float16


def build_export_config(args: argparse.Namespace, total_layers: int) -> Tuple[int, int]:
    preset = PROFILE_PRESETS[args.profile]
    layers = args.layers if args.layers is not None else preset["layers"]
    seq_len = args.seq_len if args.seq_len is not None else preset["seq_len"]

    layers = max(1, min(int(layers), total_layers))
    seq_len = max(8, int(seq_len))
    return layers, seq_len


def export_onnx(args: argparse.Namespace) -> None:
    # torch.onnx exporter still requires the `onnx` package even on legacy path.
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "Missing dependency: onnx. Install with: python -m pip install onnx"
        )

    print(f"Loading QULP file: {args.input}")
    data = torch.load(args.input, map_location="cpu")
    qulp = data.get("model", data)
    metadata = data.get("metadata", {})

    total_layers = int(metadata.get("layers", 128))
    layers, seq_len = build_export_config(args, total_layers=total_layers)
    layer_indices = _select_layer_indices(total_layers, layers)
    out_dtype = _dtype_from_arg(args.dtype)

    if args.output:
        output_onnx = args.output
    else:
        stem, _ = os.path.splitext(args.input)
        output_onnx = f"{stem}.{args.profile}.onnx"

    print("Build mobile model")
    print(f"  model type: {args.model}")
    print(f"  profile: {args.profile}")
    print(f"  selected layers: {layers}/{total_layers}")
    print(f"  layer indices: {layer_indices}")
    print(f"  seq_len: {seq_len}")
    print(f"  output dtype: {args.dtype}")
    print(f"  approx params: {_estimate_params(layers, seq_len) / 1e9:.2f}B")
    print(f"  output: {output_onnx}")

    model = RTHMobileModel(
        qulp=qulp,
        layer_indices=layer_indices,
        seq_len=seq_len,
        out_dtype=out_dtype,
    )
    model.eval()

    dummy_input = torch.randint(0, VOCAB_SIZE, (1, seq_len), dtype=torch.long)
    dynamic_axes = None
    if not args.fixed_shape:
        dynamic_axes = {"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}}

    print(f"Exporting ONNX (opset={args.opset})...")
    export_kwargs = {
        "opset_version": args.opset,
        "input_names": ["input_ids"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
    }

    # Prefer legacy exporter path first to avoid onnxscript dependency.
    try:
        torch.onnx.export(model, dummy_input, output_onnx, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions may not expose the `dynamo` flag.
        torch.onnx.export(model, dummy_input, output_onnx, **export_kwargs)

    size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"Done. ONNX size: {size_mb:.2f} MB")
    ext_count, ext_total_bytes, missing = _summarize_external_tensors(output_onnx)
    if ext_count > 0:
        ext_gb = ext_total_bytes / (1024 * 1024 * 1024)
        print(f"  external tensors: {ext_count} files, total {ext_gb:.2f} GiB")
        if missing:
            print(f"  WARNING: missing {len(missing)} external tensor files.")
            print("  The ONNX graph will not load correctly until all files are present.")
        else:
            print("  Keep the .onnx and all external tensor files in the same directory.")
    print("Next: convert ONNX to mobile runtime (QNN / LiteRT / ORT Mobile).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RTH-LM QULP to mobile ONNX.")
    parser.add_argument("--model", required=True, choices=["v4", "code"], help="Model variant label.")
    parser.add_argument("--input", required=True, help="Input .qulp file.")
    parser.add_argument("--output", default=None, help="Output ONNX path.")
    parser.add_argument(
        "--profile",
        choices=["superfast", "mid3b", "balanced", "quality"],
        default="superfast",
        help="Mobile export profile.",
    )
    parser.add_argument("--layers", type=int, default=None, help="Override selected layer count.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override export sequence length.")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Internal export dtype.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--fixed-shape",
        action="store_true",
        help="Export fixed shape graph (no dynamic axes). Better for some mobile delegates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
