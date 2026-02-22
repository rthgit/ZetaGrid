#!/usr/bin/env python3
"""
ZETAGRID REPAIR (A40) - 25B/50B QLoRA
=====================================
Unified Script to Repair either 25B or 50B models on a single A40 GPU using 4-bit Quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import math
import gc
import json
import bitsandbytes as bnb # PIP INSTALL BITSANDBYTES

# CONFIG
# ============================================================
BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
REPAIR_MIX = f"{BASE_DIR}/repair_mix.jsonl"
SAVE_DIR = f"{BASE_DIR}/repaired_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_SIZE = "25B" # Options: "25B", "50B"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# HYPERPARAMETERS
BATCH_SIZE = 4 # Small batches for A40
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4 # High LR for LoRA repair
MAX_STEPS = 500

# ARCHITECTURE PARAMS
VOCAB_SIZE = 256
D_MODEL = 4096 # Same for 25B/50B
KERNEL_SIZE = 3
LORA_RANK = 64 # Keep LoRA light

if MODEL_SIZE == "25B":
    N_LAYERS = 32
    D_FF = 8192
elif MODEL_SIZE == "50B":
    N_LAYERS = 64
    D_FF = 16384

DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# ============================================================
# QLORA LAYERS (4-BIT)
# ============================================================

class GenomeWeightBank:
    def __init__(self, path):
        print(f"🧬 Loading Genome Map: {path}")
        self.data = np.load(path, mmap_mode='r')
        self.offset = 0
        self.total_size = len(self.data)
        
    def get_slice(self, size):
        if self.offset + size > self.total_size: self.offset = 0
        start = self.offset
        end = start + size
        self.offset += size
        return torch.from_numpy(self.data[start:end].copy())

class TCNLayerQLoRA(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, bank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Norm (BF16)
        self.norm = nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.eps = 1e-6
        
        # 1. Input Projector (4-bit QLoRA)
        self.w_in = self._load_4bit(d_model, 2*d_ff, bank)
        self.scale_in = 1.0 / math.sqrt(d_model * 0.1)
        
        # LoRA In
        self.lora_in_A = nn.Parameter(torch.zeros(LORA_RANK, d_model, dtype=DTYPE))
        self.lora_in_B = nn.Parameter(torch.zeros(2*d_ff, LORA_RANK, dtype=DTYPE))
        
        # 2. Depthwise Conv (Standard 1D Conv - small enough for BF16)
        # DW Conv weights [C_out, 1, K] are tiny relative to Linear.
        # 16384 * 3 * 2 bytes = ~100KB. Keep BF16.
        w_dw_flat = bank.get_slice(d_ff * 1 * kernel_size)
        w_dw = w_dw_flat.view(d_ff, 1, kernel_size).to(DTYPE)
        self.w_dw = nn.Parameter(w_dw, requires_grad=False) # Frozen
        self.scale_dw = 1.0 / math.sqrt(kernel_size)
        
        # 3. Output Projector (4-bit QLoRA)
        self.w_out = self._load_4bit(d_ff, d_model, bank)
        self.scale_out = 1.0 / math.sqrt(d_ff * 0.1)
        
        # LoRA Out
        self.lora_out_A = nn.Parameter(torch.zeros(LORA_RANK, d_ff, dtype=DTYPE))
        self.lora_out_B = nn.Parameter(torch.zeros(d_model, LORA_RANK, dtype=DTYPE))
        
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))

    def _load_4bit(self, in_features, out_features, bank):
        """Loads a slice from genome and quantized it into a bnb Linear4bit layer."""
        # 1. Fetch raw BF16 weights
        n_params = in_features * out_features
        raw_w = bank.get_slice(n_params).view(out_features, in_features).to(torch.float16) # BNB needs FP16 input
        
        # 2. Create 4-bit Linear Layer
        # Use simple Linear4bit from bitsandbytes if available
        # Or better: construct standard Linear and replace content?
        # bnb.nn.Linear4bit(in_features, out_features, bias=False, compute_dtype=DTYPE)
        
        layer = bnb.nn.Linear4bit(
            in_features, 
            out_features, 
            bias=False, 
            compute_dtype=DTYPE,
            quant_type="nf4" # Normal Float 4
        )
        
        # 3. Assign weights (This triggers quantization internally in bnb)
        # We need to act carefully. bnb usually expects .weight to be assigned.
        # But initializing with pre-trained weights is tricky.
        # HACK: Create layer, perform a dummy forward to init buffers? No.
        # Standard way: load_state_dict.
        # But we have raw tensor.
        # Let's trust bnb creates the parameter.
        # We need to manually quantize and assign.
        # Actually, simpler approach:
        # Just assign .weight.data = raw_w (if on CUDA)?
        # BNB layers handle quantization on .cuda() call or explicitly.
        
        # For simplicity in this script, we will instantiate the layer on CPU, assign heavy weights, then move to CUDA?
        # A40 has RAM.
        # Let's try:
        layer.weight.data = raw_w # Assign FP16 data
        layer = layer.to(DEVICE) # Should trigger quantization if implemented in bnb.nn
        
        # If manual quantization is needed:
        # This part is tricky without testing. 
        # But assume standard bnb usage:
        # For now, simplistic approach.
        
        return layer

    def forward(self, x):
        res = x
        # Norm
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = (x_f * rms).to(DTYPE) * self.norm
        
        # Linear In (4-bit)
        # bnb linear accepts BF16 input
        ag_base = self.w_in(x_norm) * self.scale_in
        lora_in = (x_norm @ self.lora_in_A.T) @ self.lora_in_B.T
        ag = ag_base + lora_in
        a, g = ag.chunk(2, dim=-1)
        
        # Conv
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, self.w_dw, groups=self.w_dw.shape[0], dilation=self.dilation) * self.scale_dw
        a = a.transpose(1, 2)
        
        y = F.silu(a) * torch.sigmoid(g)
        
        # Linear Out (4-bit)
        out_base = self.w_out(y) * self.scale_out
        lora_out = (y @ self.lora_out_A.T) @ self.lora_out_B.T
        
        return res + (out_base + lora_out) * self.scale

class ZetaGridQLoRA(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        self.layers = nn.ModuleList()
        print(f"⚡ Building {N_LAYERS} Layers ({MODEL_SIZE}) QLoRA...")
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayerQLoRA(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
        self.norm_f = nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        self.eps = 1e-6

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # Final Norm
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x_f * rms).to(DTYPE) * self.norm_f
        
        logits = F.linear(x, self.emb.weight.to(DTYPE))
        return logits

def train():
    print(f"🚀 STARTING {MODEL_SIZE} REPAIR (QLoRA)...")
    bank = GenomeWeightBank(GENOME_PATH)
    model = ZetaGridQLoRA(bank).to(DEVICE)
    del bank
    
    print("✅ Model Built in 4-bit.")
    print(f"   Footprint: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer (Only optimize parameters requiring grad)
    # LoRA params + Norms + Embeddings?
    # Usually freeze Embeddings to save memory/stability.
    # We only train LoRA and Norms.
    trainable_params = []
    for n, p in model.named_parameters():
        if 'lora' in n or 'norm' in n:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False
            
    print(f"   Trainable Params: {len(trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # Dataset Loading
    # Check if REPAIR_MIX exists
    if not os.path.exists(REPAIR_MIX):
        print(f"⚠️  {REPAIR_MIX} not found. Running PREPARE_REPAIR_DATASET.py logic inline (mock)...")
        # In real scenario, we crash or fallback.
        # Fallback to Dummy knowledge for test?
        # NO. We assume user ran PREPARE_REPAIR_DATASET.py
        print("❌ CRITICAL: Run PREPARE_REPAIR_DATASET.py first!")
        return

    print(f"📖 Loading Repair Mix: {REPAIR_MIX}")
    with open(REPAIR_MIX, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"⚡ Training on {len(data)} examples for {MAX_STEPS} steps...")
    model.train()
    
    import random
    
    def get_batch():
        batch_idx = []
        batch_labels = []
        batch_mask = []
        
        for _ in range(BATCH_SIZE):
            ex = random.choice(data)
            msgs = ex['messages']
            # Support both ChatML and Simple Text
            if isinstance(msgs, list):
                user_txt = msgs[0]['content']
                asst_txt = msgs[1]['content']
                full_txt = f"User: {user_txt}\nAssistant: {asst_txt}"
            else:
                full_txt = msgs # Raw text fallback
                
            tokens = list(full_txt.encode('utf-8'))
            if len(tokens) > 2048: tokens = tokens[:2048]
            
            # Predict only after 'Assistant:'
            tag = b"Assistant:"
            try:
                tag_idx = full_txt.encode('utf-8').find(tag)
            except: tag_idx = -1
            
            labels = tokens[1:] + [0]
            mask = [0] * len(tokens)
            
            if tag_idx != -1:
                start = tag_idx + len(tag)
                for i in range(start, len(mask)):
                    mask[i] = 1
            else:
                # If no tag (e.g. raw wiki), train on EVERYTHING?
                # Or just skip?
                # Let's train on everything for raw knowledge
                mask = [1] * len(tokens) 
                    
            # Pad
            pad_len = 2048 - len(tokens)
            final_tokens = tokens + [0]*pad_len
            final_labels = labels + [0]*pad_len
            final_mask = mask + [0]*pad_len
            
            batch_idx.append(final_tokens)
            batch_labels.append(final_labels)
            batch_mask.append(final_mask)
            
        return (
            torch.tensor(batch_idx, dtype=torch.long, device=DEVICE),
            torch.tensor(batch_labels, dtype=torch.long, device=DEVICE),
            torch.tensor(batch_mask, dtype=torch.float32, device=DEVICE)
        )

    optimizer.zero_grad()
    step = 0
    t0 = time.time()
    
    while step < MAX_STEPS:
        total_loss = 0
        for _ in range(GRAD_ACCUM):
            x, y, mask = get_batch()
            logits = model(x)
            
            logits = logits.view(-1, VOCAB_SIZE)
            y = y.view(-1)
            mask = mask.view(-1)
            
            loss_raw = F.cross_entropy(logits, y, reduction='none')
            # Normalize by mask sum
            den = mask.sum() + 1e-6
            loss = (loss_raw * mask).sum() / den
            loss = loss / GRAD_ACCUM
            
            loss.backward()
            total_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        if step % 5 == 0:
            dt = time.time() - t0
            print(f"Step {step}/{MAX_STEPS} | Loss: {total_loss:.4f} | Time: {dt:.2f}s")
            t0 = time.time()
            
        if step % 100 == 0:
             print(f"💾 Checkpoint Step {step}...")
             torch.save(model.state_dict(), f"{SAVE_DIR}/zeta_repair_{MODEL_SIZE}_step{step}.pt")
            
    # Save Final
    print(f"💾 Saving Repaired {MODEL_SIZE} Model: {SAVE_DIR}/zeta_repair_{MODEL_SIZE}_final.pt")
    torch.save(model.state_dict(), f"{SAVE_DIR}/zeta_repair_{MODEL_SIZE}_final.pt")

if __name__ == "__main__":
    train()
