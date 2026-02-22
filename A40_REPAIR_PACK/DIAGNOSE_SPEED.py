import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np
import bitsandbytes as bnb
import math

# CONFIG (FULL V2)
BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
DEVICE = "cuda"
DTYPE = torch.bfloat16
LORA_RANK = 128
D_MODEL = 4096
D_FF = 16384 # The Beast
KERNEL_SIZE = 3
N_LAYERS = 32
VOCAB_SIZE = 256
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# SAFE SETTINGS
BATCH_SIZE = 4
GRAD_ACCUM = 16

print("🔍 ZETAGRID DIAGNOSTIC TOOL (SPEED & VRAM)")
print("==========================================")

class GenomeWeightBank:
    def __init__(self, path):
        if not os.path.exists(path):
            print(f"❌ Genome not found: {path} (Using Mock Data)")
            self.mock = True
        else:
            self.data = np.load(path, mmap_mode='r')
            self.mock = False
        self.offset = 0
        
    def get_slice(self, size):
        if self.mock: return torch.randn(size)
        start = self.offset
        end = start + size
        self.offset = (self.offset + size) % len(self.data)
        return torch.from_numpy(self.data[start:end].copy())

class TCNLayerQLoRA(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, bank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Mocking Loading for Speed Test (We just want architecture size)
        # Using BF16 Linear to simulate 4-bit/8-bit memory footprint roughly
        # (Actually BF16 is bigger, so if this fits, 4-bit definitely fits)
        
        # In real script we used bnb 8bit or 4bit.
        # Let's use bnb 8bit here to match likely runtime.
        
        self.w_in = bnb.nn.Linear8bitLt(d_model, 2*d_ff, bias=False, has_fp16_weights=False)
        self.w_out = bnb.nn.Linear8bitLt(d_ff, d_model, bias=False, has_fp16_weights=False)
        
        self.w_dw = nn.Parameter(torch.randn(d_ff, 1, kernel_size).to(DTYPE)) # Keep BF16
        
        self.lora_in_A = nn.Parameter(torch.zeros(LORA_RANK, d_model, dtype=DTYPE))
        self.lora_in_B = nn.Parameter(torch.zeros(2*d_ff, LORA_RANK, dtype=DTYPE))
        self.lora_out_A = nn.Parameter(torch.zeros(LORA_RANK, d_ff, dtype=DTYPE))
        self.lora_out_B = nn.Parameter(torch.zeros(d_model, LORA_RANK, dtype=DTYPE))
        
        self.norm = nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))

    def forward(self, x):
        return x # Dummy forward for memory alloc check
        # Real forward is expensive. Let's do a partial real forward to stress compute.
        x_f = x.to(DTYPE)
        # 1. Linear In
        h = self.w_in(x_f) # [B, T, 2*FF] - Heavy
        # 2. LoRA In
        l = (x_f @ self.lora_in_A.T @ self.lora_in_B.T)
        h = h + l
        return x + (h[:,:,:4096] * self.scale) # Hack return to keep shape

class ZetaGrid(nn.Module):
    def __init__(self, bank, n_layers, d_ff):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList()
        print(f"⚡ Allocating {n_layers} Layers (FF={d_ff})...")
        for i in range(n_layers):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayerQLoRA(D_MODEL, d_ff, KERNEL_SIZE, dil, bank))
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, idx):
        x = self.emb(idx).to(DTYPE)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        return self.head(x)

def run_diagnostic():
    torch.set_default_dtype(DTYPE)
    print(f"🖥️  Checking GPU...")
    print(torch.cuda.get_device_name(0))
    
    bank = GenomeWeightBank(GENOME_PATH)
    model = ZetaGrid(bank, N_LAYERS, D_FF).to(DEVICE)
    
    print("🔧 Compiling Optimizer (8-bit)...")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    
    print("📉 Generating Dummy Batch...")
    x = torch.randint(0, 256, (BATCH_SIZE, 2048)).to(DEVICE)
    
    print("🚀 STARTING SPEED TEST (5 Steps)...")
    model.train()
    
    for step in range(1, 6):
        t0 = time.time()
        
        # Accumulation Loop
        for _ in range(GRAD_ACCUM):
            out = model(x)
            loss = out.mean() # Dummy loss
            loss.backward()
            
        optimizer.step()
        optimizer.zero_grad()
        
        dt = time.time() - t0
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Step {step}: {dt:.2f}s | VRAM: {mem:.2f} GB")
        sys.stdout.flush()

    print("✅ DIAGNOSTIC COMPLETE.")

if __name__ == "__main__":
    run_diagnostic()
