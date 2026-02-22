#!/usr/bin/env python3
"""
ZETAGRID 50B - PHASE 3 (FIXED)
==============================
- 64 Layers (Doubled from 25B)
- Seed: zetagrid_50b_seed.pt
- Fixes: Reduced LR, Tighter Clip, NaN Check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import math
import gc

print("=" * 70)
print("ZETAGRID 50B - PHASE 3 (FIXED)")
print("64 Layers | TCN + Genome Backbone | Reduced LR")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
SEED_PATH = f"{BASE_DIR}/zetagrid_50b_seed.pt" 
DATA_PATH = f"{BASE_DIR}/data/pretrain/clean_text_utf8.bin"

SAVE_DIR = f"{BASE_DIR}/phase3_checkpoints"
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

# Training - Tuned for Stability
SEQ_LEN = 256
BATCH_SIZE = 4        
GRAD_ACCUM = 8        
LR = 5e-5             # REDUCED (was 1.5e-4) -> Safer start
WARMUP_STEPS = 500
TOTAL_STEPS = 20000   
SAVE_EVERY = 2000
LOG_EVERY = 20
GRAD_CLIP = 0.5       # TIGHTER CLIP (was 1.0)

# ============================================================
# GENOME BANK
# ============================================================

class GenomeWeightBank:
    def __init__(self, genome_path):
        print(f"\n[GENOME] Loading {genome_path}...")
        raw = np.load(genome_path)
        print(f"   Raw size: {len(raw)/1e9:.2f}GB")
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
        
        # Frozen weights
        self.register_buffer('w_in', bank.get_weight(2 * d_ff, d_model))
        self.register_buffer('w_dw', bank.get_conv_weight(d_ff, kernel_size))
        self.register_buffer('w_out', bank.get_weight(d_model, d_ff))
        
        # LoRA
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

# ============================================================
# MODEL
# ============================================================

class ZetaGrid50B(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer50B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
            if (i+1) % 8 == 0:
                print(f"   Built Layer {i+1}/{N_LAYERS}")
        self.norm_f = RMSNorm(D_MODEL)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
        
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.norm_f(x)
        logits = F.linear(x.float(), self.emb.weight.float())
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new=100):
        self.eval()
        idx = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)
        for _ in range(max_new):
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits, _ = self(idx[:, -1024:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return bytes(idx[0].cpu().tolist()).decode('utf-8', errors='ignore')

# ============================================================
# MAIN
# ============================================================

def train():
    bank = GenomeWeightBank(GENOME_PATH)
    model = ZetaGrid50B(bank).to(DEVICE)
    
    # Load Seed
    if os.path.exists(SEED_PATH):
        print(f"\n🌱 Loading 50B Fractal Seed: {SEED_PATH}")
        ckpt = torch.load(SEED_PATH, map_location=DEVICE)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        # Filter strictly
        # Filter out frozen weights if they exist in checkpoint somehow
        s = {k:v for k,v in state_dict.items() if 'w_' not in k}
        msg = model.load_state_dict(s, strict=False)
        print(f"   Load status: {msg}")
    else:
        print(f"\n⚠️ SEED NOT FOUND: {SEED_PATH} - Starting scratch (NOT RECOMMENDED)")

    # Resume if checkpoint
    ckpts = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith('zeta50b_')])
    start_step = 0
    if ckpts:
        latest = os.path.join(SAVE_DIR, ckpts[-1])
        print(f"   Resuming from {latest}")
        c = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(c['model'], strict=False)
        start_step = c.get('step', 0)

    # Free bank
    del bank.data
    del bank
    gc.collect(); torch.cuda.empty_cache()

    # Data
    raw = np.fromfile(DATA_PATH, dtype=np.uint8)
    print(f"\n📚 Data: {len(raw)/1e9:.2f}GB")
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    
    t0 = time.time()
    model.train()
    
    print(f"\n🚀 STARTING PHASE 3 TRAINING (50B) - FIXED LR")
    skipped = 0
    
    for step in range(start_step+1, TOTAL_STEPS+1):
        optimizer.zero_grad()
        loss_acc = 0.0
        
        # Check for NaN gradients
        valid_batch = True
        
        for _ in range(GRAD_ACCUM):
            ix = np.random.randint(0, len(raw)-SEQ_LEN-1, BATCH_SIZE)
            x = torch.stack([torch.tensor(raw[i:i+SEQ_LEN]) for i in ix]).long().to(DEVICE)
            y = torch.stack([torch.tensor(raw[i+1:i+1+SEQ_LEN]) for i in ix]).long().to(DEVICE)
            with torch.amp.autocast('cuda', dtype=DTYPE):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            
            if torch.isnan(loss):
                print(f"⚠️ NaN Loss detected at step {step} batch!")
                valid_batch = False
                break
                
            loss.backward()
            loss_acc += loss.item()
        
        if not valid_batch:
            optimizer.zero_grad()
            skipped += 1
            print(f"Skipping step {step} due to NaN loss.")
            continue
            
        # CLIP GRADIENTS
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        if step % LOG_EVERY == 0:
            avg = loss_acc * GRAD_ACCUM # Back to per-batch scale approximation? No, loss.item() is already scaled?
            # loss was / GRAD_ACCUM. act_loss = loss_acc * GRAD_ACCUM? 
            # Actually loss_acc is sum of (loss / GRAD_ACCUM). So loss_acc IS the average loss if accum logic is correct.
            # No, if loss = total/accum, then sum(loss) = total. 
            # Wait. loss.backward() handles gradients.
            # loss value for logging should be the average over the batch.
            # loss_acc is roughly correct.
            
            print(f"Step {step}/{TOTAL_STEPS} | Loss: {loss_acc:.4f} | Skipped: {skipped} | Time: {(time.time()-t0):.1f}s")
        
        if step % SAVE_EVERY == 0:
            p = f"{SAVE_DIR}/zeta50b_step{step}.pt"
            s = {k:v for k,v in model.state_dict().items() if 'w_' not in k}
            torch.save({'step': step, 'model': s}, p)
            print(f"💾 Saved {p}")
            print(f"   Gen: {model.generate('The future ')}")
            model.train()

if __name__ == "__main__":
    train()
