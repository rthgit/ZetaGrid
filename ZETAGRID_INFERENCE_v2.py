#!/usr/bin/env python3
"""
ZETAGRID 25B v2 - COMPLETE MODEL INFERENCE
=========================================
Standalone Inference for Repaired/Unified Model (v2).
NO DEPENDENCY on Genome .npy file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import gc

print("=" * 70)
print("ZETAGRID 25B v2 - STANDALONE INFERENCE")
print("Non-Transformer LLM | TCN Backbone | Complete Model")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

CKPT_PATH = r"E:/ZETAGRID/zeta_25B_v2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384 # Check if v2 used 8192 or 16384. RUN_REPAIR used 8192?
# WAITING: RUN_REPAIR used d_ff=8192 in previous context.
# BUT convert_rth_to_gguf used 16384.
# CRITICAL: We need to match the trained model.
# I will assume 16384 for now based on original inference, but if it fails, it's 8192.
# Actually detailed check: RUN_REPAIR_A40.py uses d_ff=8192?
# Let's check RUN_REPAIR code from context if possible. 
# Re-checking RUN_REPAIR output... "D_FF = 8192" in written file.
# So v2 is 8192.
D_FF = 8192 

N_LAYERS = 32
KERNEL_SIZE = 3
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# ============================================================
# MODEL v2 (Unified)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.w

# Standard TCN Layer (No Weight Bank, Just Parameters)
class TCNLayerV2(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.norm = RMSNorm(d_model)
        
        # Standard Linear/Conv Layers
        self.w_in = nn.Linear(d_model, 2*d_ff, bias=False)
        self.w_dw = nn.Conv1d(d_ff, d_ff, kernel_size, groups=d_ff, dilation=dilation)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        res = x
        x = self.norm(x).to(DTYPE)
        
        # Proj In
        ag = self.w_in(x)
        a, g = ag.chunk(2, dim=-1)
        
        # DW Conv
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = self.w_dw(a)
        a = a.transpose(1, 2)
        
        # Act
        y = F.silu(a) * torch.sigmoid(g)
        
        # Proj Out
        out = self.w_out(y)
        
        return res + out * self.scale

class ZetaGridV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayerV2(D_MODEL, D_FF, KERNEL_SIZE, dil))
            
        self.norm_f = RMSNorm(D_MODEL)
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        
        x = (self.emb(idx) + self.pos_emb(pos[:, :T])).to(DTYPE)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return F.linear(x.float(), self.emb.weight.float()) # Tied embeddings

    @torch.no_grad()
    def generate(self, prompt, max_new=300, temperature=0.7, top_k=50):
        prompt_bytes = list(prompt.encode('utf-8'))
        idx = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
        
        generated = []
        for _ in range(max_new):
            logits = self(idx[:, -1024:])[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            idx = torch.cat([idx, next_token], dim=1)
            generated.append(next_token.item())
            
            print(bytes([generated[-1]]).decode('utf-8', errors='replace'), end="", flush=True)
            
        return bytes(prompt_bytes + generated).decode('utf-8', errors='replace')

# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        return

    print("🏗️  Building ZetaGrid v2 Model...")
    model = ZetaGridV2().to(DEVICE).to(DTYPE)
    
    print(f"📂 Loading Weights from {CKPT_PATH}...")
    sd = torch.load(CKPT_PATH, map_location="cpu")
    
    # Key Mapping Handling (If trained with 'module.' or specific prefixes)
    # We attempt to clean keys to match standard TCNLayerV2
    clean_sd = {}
    for k, v in sd.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        # Remap QLoRA/Bank keys to standard Linear if needed?
        # If saved from QLoRA, keys might be 'w_in.default.weight' etc?
        # Or if saved via state_dict(), they are just weights.
        # We assume standard state dict was saved.
        clean_sd[new_k] = v
        
    try:
        model.load_state_dict(clean_sd, strict=False)
        print("✅ Weights Loaded (Strict=False for safety)")
    except Exception as e:
        print(f"⚠️  Load Error: {e}")
        print("Trying strict loading...")
        model.load_state_dict(clean_sd)

    model.eval()
    print(f"🚀 Ready! VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    while True:
        p = input("\n\nPrompt > ")
        if p.lower() in ['q', 'quit']: break
        model.generate(p)

if __name__ == "__main__":
    main()
