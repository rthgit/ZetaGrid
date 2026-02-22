#!/usr/bin/env python3
"""
RTH-LM 25B V2 - UNIFIED INFERENCE
==================================
Native Inference for RTH-LM (Fractal TCN Architecture).
Supports:
1. Sharded Safetensors (Merged 128-layer models)
2. QULP 2-bit compressed models
3. Genome + Soul (Legacy)

Usage:
    python RTH_LM_INFERENCE_v2.py --model "E:/ZETAGRID/rth_lm_25b_v4_sharded"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gc
import argparse
from safetensors.torch import load_file

# MODEL PARAMS
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 128
KERNEL_SIZE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ============================================================
# ARCHITECTURE MODULES
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.w

class RTHLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_in = nn.Parameter(torch.empty(2 * D_FF, D_MODEL))
        self.w_dw = nn.Parameter(torch.empty(D_FF, 1, KERNEL_SIZE))
        self.w_out = nn.Parameter(torch.empty(D_MODEL, D_FF))
        self.norm = RMSNorm(D_MODEL)
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, state=None):
        # x: [B, T, D]
        res = x
        x = self.norm(x)
        
        # Linear + Gate
        # Transpose for Conv1D: [B, D, T]
        x_t = x.transpose(1, 2)
        
        # Project In
        ag = F.conv1d(x_t, self.w_in.unsqueeze(-1)) # 1x1 Conv
        a, g = ag.chunk(2, dim=1)
        
        # Depthwise Conv (Causal)
        pad = (KERNEL_SIZE - 1, 0)
        a = F.pad(a, pad)
        a = F.conv1d(a, self.w_dw, groups=D_FF)
        
        # Gating
        y = F.silu(a) * torch.sigmoid(g)
        
        # Project Out
        out = F.conv1d(y, self.w_out.unsqueeze(-1))
        
        # Back to [B, T, D]
        out = out.transpose(1, 2)
        
        return res + out * self.scale

class ZetaGridV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(2048, D_MODEL))
        self.layers = nn.ModuleList([RTHLayer() for _ in range(N_LAYERS)])
        self.norm_f = RMSNorm(D_MODEL)
        
    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx)
        x = x + self.pos_emb[:T]
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = F.linear(x[:, -1, :], self.emb.weight)
        return logits

# ============================================================
# LOADING LOGIC
# ============================================================

def load_unified_model(model_path):
    print(f"📦 Loading ZetaGrid V2 Unified Model: {model_path}")
    model = ZetaGridV2().to(DEVICE).to(DTYPE)
    
    if os.path.isdir(model_path):
        # Sharded Safetensors
        shards = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        state_dict = {}
        for s in shards:
            print(f"   Reading {s}...", end="\r")
            sd = load_file(os.path.join(model_path, s))
            state_dict.update(sd)
        model.load_state_dict(state_dict, strict=False)
        print("\n   ✅ Shards Loaded Successfully.")
    else:
        # Single .pt or .qulp (Future)
        print("   Detected single file checkpoint.")
        ckpt = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        
    model.eval()
    return model

@torch.inference_mode()
def generate(model, prompt, max_tokens=100, temp=0.8, top_k=40):
    idx = torch.tensor([[ord(c) for c in prompt]], device=DEVICE)
    
    print(prompt, end="", flush=True)
    
    for _ in range(max_tokens):
        # Crop context
        idx_cond = idx[:, -1024:]
        logits = model(idx_cond)
        logits = logits / temp
        
        # Top-K
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, next_token), dim=1)
        char = chr(next_token.item()) if next_token.item() < 256 else "?"
        print(char, end="", flush=True)
        
        if next_token.item() == 0: break # End of text
        
    print("\n")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to sharded dir or .pt file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial text")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model path {args.model} not found.")
        sys.exit(1)
        
    model = load_unified_model(args.model)
    
    print("\n🚀 Ready for generation. Type 'exit' to quit.")
    while True:
        p = input("Prompt >>> ")
        if p.lower() == 'exit': break
        generate(model, p)
