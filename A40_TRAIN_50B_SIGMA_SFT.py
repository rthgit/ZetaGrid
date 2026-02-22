#!/usr/bin/env python3
"""
ZETAGRID 50B - PHASE 4 (SIGMA SFT) - STANDALONE MASTER
======================================================
Supervised Fine-Tuning for Fractal TCN.
- Instruction Masking (Loss only on Assistant response)
- UTF-8 Byte Tokenization
- Linear-time context handling
- 50B Architecture (64 Layers)
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
import glob

print("=" * 70)
print("ZETAGRID 50B - PHASE 4 (SIGMA SFT)")
print("INSTRUCTION TUNING | MASKED CROSS-ENTROPY | BF16")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

# Paths - ADJUST THESE FOR YOUR A40 CLUSTER
BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
PHASE3_CKPT = f"{BASE_DIR}/zeta50b_phase3_final.pt" # Points to your recovered 12k step model
SFT_DATA_PATH = f"{BASE_DIR}/data/sft/merged_finetune_data.jsonl"

SAVE_DIR = f"{BASE_DIR}/phase4_sft_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model - 50B Specs
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 64
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# SFT Hyperparameters
SEQ_LEN = 1024        
BATCH_SIZE = 2        
GRAD_ACCUM = 16       
LR = 5e-6             
GRAD_CLIP = 0.5       
TOTAL_STEPS = 5000    
SAVE_EVERY = 200      # More frequent saves for safety
LOG_EVERY = 10

# ============================================================
# GENOME BANK
# ============================================================

class GenomeWeightBank:
    def __init__(self, genome_path):
        print(f"\n[GENOME] Loading {genome_path}...")
        raw = np.load(genome_path)
        self.data = torch.from_numpy(raw.astype(np.float32)).to(DTYPE).to(DEVICE)
        del raw; gc.collect()
        self.offset = 0
        
    def get_weight(self, out_features, in_features):
        n = out_features * in_features
        if self.offset + n > len(self.data): self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(out_features, in_features)
        self.offset += n
        scale = 1.0 / math.sqrt(in_features * 0.1)
        return (chunk * scale).contiguous()
    
    def get_conv_weight(self, channels, kernel_size):
        n = channels * kernel_size
        if self.offset + n > len(self.data): self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(channels, 1, kernel_size)
        self.offset += n
        scale = 1.0 / math.sqrt(kernel_size)
        return (chunk * scale).contiguous()

# ============================================================
# LAYERS
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.w

class LoRA(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

class TCNLayer50B(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, bank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.norm = RMSNorm(d_model)
        
        self.register_buffer('w_in', bank.get_weight(2 * d_ff, d_model))
        self.register_buffer('w_dw', bank.get_conv_weight(d_ff, kernel_size))
        self.register_buffer('w_out', bank.get_weight(d_model, d_ff))
        
        self.lora_in = LoRA(d_model, 2 * d_ff, LORA_RANK)
        self.lora_out = LoRA(d_ff, d_model, LORA_RANK)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        res = x
        x = self.norm(x).to(DTYPE)
        
        ag = F.linear(x, self.w_in) + self.lora_in(x)
        a, g = ag.chunk(2, dim=-1)
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = a.contiguous()
        a = F.conv1d(a, self.w_dw, groups=D_FF, dilation=self.dilation)
        a = a.transpose(1, 2)
        
        y = F.silu(a) * torch.sigmoid(g)
        out = F.linear(y, self.w_out) + self.lora_out(y)
        return res + out * self.scale

class ZetaGrid50B(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer50B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
        self.norm_f = RMSNorm(D_MODEL)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
        for layer in self.layers:
            # Use checkpointing for VRAM efficiency during SFT
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.norm_f(x)
        logits = F.linear(x.float(), self.emb.weight.float())
        return logits

# ============================================================
# SFT DATA & WRAPPER
# ============================================================

class SFTDataset:
    def __init__(self, path, seq_len):
        self.path = path
        self.seq_len = seq_len
        self.assistant_tag = b"Assistant:"
        
    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if not text: continue
                    
                    tokens = list(text.encode('utf-8'))
                    if len(tokens) < 10: continue
                    
                    labels = tokens[1:] + [0]
                    mask = [0] * len(tokens)
                    
                    tag_pos = text.encode('utf-8').find(self.assistant_tag)
                    if tag_pos != -1:
                        start_idx = tag_pos + len(self.assistant_tag)
                        for i in range(start_idx, len(mask)):
                            mask[i] = 1
                    
                    for i in range(0, len(tokens), self.seq_len):
                        chunk_tokens = tokens[i : i + self.seq_len]
                        chunk_labels = labels[i : i + self.seq_len]
                        chunk_mask = mask[i : i + self.seq_len]
                        
                        n = len(chunk_tokens)
                        if n < 10: continue
                        if n < self.seq_len:
                            chunk_tokens += [0] * (self.seq_len - n)
                            chunk_labels += [0] * (self.seq_len - n)
                            chunk_mask += [0] * (self.seq_len - n)
                            
                        yield (
                            torch.tensor(chunk_tokens, dtype=torch.long),
                            torch.tensor(chunk_labels, dtype=torch.long),
                            torch.tensor(chunk_mask, dtype=torch.float32)
                        )
                except: continue

class ZetaGrid50B_SFT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
    def forward(self, x, y, mask):
        logits = self.base(x)
        logits = logits.view(-1, VOCAB_SIZE)
        y = y.view(-1)
        mask = mask.view(-1)
        
        raw_loss = F.cross_entropy(logits, y, reduction='none')
        weighted_loss = raw_loss * mask
        active_tokens = mask.sum()
        
        if active_tokens > 0:
            loss = weighted_loss.sum() / active_tokens
        else:
            loss = raw_loss.mean() * 0.0 
            
        return loss

# ============================================================
# MAIN TRAIN
# ============================================================

def train():
    # 1. Build Model
    bank = GenomeWeightBank(GENOME_PATH)
    base_model = ZetaGrid50B(bank).to(DEVICE)
    del bank.data; del bank; gc.collect()
    
    # 2. Load Phase 3 Checkpoint
    if os.path.exists(PHASE3_CKPT):
        print(f"🌱 Loading Phase 3 Checkpoint: {PHASE3_CKPT}")
        ckpt = torch.load(PHASE3_CKPT, map_location=DEVICE)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        base_model.load_state_dict(state, strict=False)
        del ckpt; del state; gc.collect()
    
    model = ZetaGrid50B_SFT(base_model).to(DEVICE)
 
    # 2b. Attempt Resume from SFT Checkpoint
    start_step = 1
    sft_ckpts = sorted(glob.glob(f"{SAVE_DIR}/zeta50b_sft_step*.pt"), key=os.path.getmtime)
    
    # Try the latest, then fallback
    while len(sft_ckpts) > 0:
        latest = sft_ckpts[-1]
        try:
            print(f"🔄 Attempting Resume from SFT: {latest}")
            # Use safe weights_only=False for checkpoints
            ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
            
            # Extract state and step
            # If checkpoint has 'model' key, use it. Else assume it's the state dict.
            state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            
            # Extract step number from dict or filename
            if isinstance(ckpt, dict) and 'step' in ckpt:
                step_num = ckpt['step']
            else:
                base_name = os.path.basename(latest)
                step_num = int(base_name.split('step')[-1].replace('.pt',''))
            
            # Load into base model
            model.base.load_state_dict(state, strict=False)
            start_step = step_num + 1
            print(f"✅ SUCCESS! Resuming from Step {step_num}")
            del ckpt; del state; gc.collect()
            break
        except Exception as e:
            print(f"⚠️ Failed to resume from {latest}: {e}")
            print("   (Trying older checkpoint if available...)")
            sft_ckpts.pop()

    
    # 3. Data & Optimizer
    dataset = SFTDataset(SFT_DATA_PATH, SEQ_LEN)
    data_iter = iter(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
    
    print(f"\n🚀 STARTING SIGMA SFT (50B)")
    t0 = time.time()
    model.train()
    
    for step in range(start_step, TOTAL_STEPS + 1):
        optimizer.zero_grad()
        loss_acc = 0.0
        
        for _ in range(GRAD_ACCUM):
            try:
                x, y, m = next(data_iter)
            except StopIteration:
                data_iter = iter(dataset)
                x, y, m = next(data_iter)
            
            x, y, m = x.unsqueeze(0).to(DEVICE), y.unsqueeze(0).to(DEVICE), m.unsqueeze(0).to(DEVICE)
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y, m)
                loss = loss / GRAD_ACCUM
            
            loss.backward()
            loss_acc += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        if step % LOG_EVERY == 0:
            print(f"Step {step}/{TOTAL_STEPS} | SFT Loss: {loss_acc:.4f} | Time: {time.time()-t0:.1f}s")
            
        if step % SAVE_EVERY == 0:
            p = f"{SAVE_DIR}/zeta50b_sft_step{step}.pt"
            s = {k:v for k,v in base_model.state_dict().items() if 'w_' not in k}
            torch.save({'step': step, 'model': s, 'loss': loss_acc}, p)
            print(f"💾 Saved {p}")

if __name__ == "__main__":
    train()
