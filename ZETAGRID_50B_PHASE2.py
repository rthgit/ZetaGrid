#!/usr/bin/env python3
"""
ZETAGRID 50B - PHASE 2 GRADIENT TRAINING
=========================================
50B parameter NON-TRANSFORMER LLM

Steps:
1. Expand 25B genome → 50B via fractal replication
2. Build 64-layer TCN with frozen genome backbone
3. Optionally load trained 25B LoRA adapters for first 32 layers
4. Train with cross-entropy

VRAM: ~25GB on A40 (genome stays int8, converts per-layer)
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
print("ZETAGRID 50B - PHASE 2 GRADIENT TRAINING")
print("Non-Transformer LLM | 64 TCN Layers | Genome Backbone")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_25B_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
GENOME_50B_PATH = f"{BASE_DIR}/zetagrid_50b_expanded.npy"
CKPT_25B_PATH = f"{BASE_DIR}/phase2_checkpoints/zeta25b_FINAL.pt"
DATA_PATHS = [
    f"{BASE_DIR}/data/pretrain/KAM_SFT_MASTER.bin",
    f"{BASE_DIR}/data/pretrain/amazon_reviews_10M.bin",
    f"{BASE_DIR}/data/pretrain/training_data.bin",
]
SAVE_DIR = f"{BASE_DIR}/phase2_50b_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# 50B Model (doubled from 25B)
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 64          # 2x the 25B model
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# Training
SEQ_LEN = 256
BATCH_SIZE = 4         # Smaller due to more layers
GRAD_ACCUM = 8         # Effective = 32
LR = 1e-4              # Lower LR for larger model
WARMUP_STEPS = 300
TOTAL_STEPS = 10000
SAVE_EVERY = 1000
LOG_EVERY = 25
GRAD_CLIP = 1.0

# ============================================================
# STEP 1: EXPAND GENOME 25B → 50B
# ============================================================

def expand_genome():
    """Fractal replication: 25B → 50B"""
    
    if os.path.exists(GENOME_50B_PATH):
        print(f"\n[GENOME] 50B already exists: {GENOME_50B_PATH}")
        genome = np.load(GENOME_50B_PATH)
        print(f"   Size: {len(genome)/1e9:.2f}GB")
        return genome
    
    print(f"\n[GENOME] Expanding 25B → 50B...")
    genome_25b = np.load(GENOME_25B_PATH)
    size_25b = len(genome_25b)
    size_50b = size_25b * 2
    
    print(f"   25B: {size_25b/1e9:.2f}GB")
    print(f"   50B target: {size_50b/1e9:.2f}GB")
    
    # Fractal replication
    genome_50b = np.zeros(size_50b, dtype=np.int8)
    genome_50b[:size_25b] = genome_25b
    genome_50b[size_25b:] = genome_25b
    
    # Add diversity noise (5% on second half)
    print("   Adding diversity noise (5%)...")
    noise_region = genome_50b[size_25b:]
    n_noise = int(len(noise_region) * 0.05)
    noise_idx = np.random.randint(0, len(noise_region), size=n_noise)
    noise_region[noise_idx] = np.random.randint(-1, 2, size=n_noise, dtype=np.int8)
    
    # Save
    print(f"   Saving {GENOME_50B_PATH}...")
    np.save(GENOME_50B_PATH, genome_50b)
    print(f"   ✅ 50B genome: {size_50b/1e9:.2f}GB")
    
    del genome_25b
    gc.collect()
    
    return genome_50b

# ============================================================
# MODEL COMPONENTS (same as 25B but int8 forward for VRAM)
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
    """TCN layer with int8 frozen weights (converts on-the-fly to save VRAM)"""
    
    def __init__(self, d_model, d_ff, kernel_size, dilation, genome_slice):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.d_ff = d_ff
        
        self.norm = RMSNorm(d_model)
        
        # Store frozen weights as INT8 buffers (saves 2x VRAM vs BF16)
        offset = 0
        n_in = d_model * 2 * d_ff
        raw_in = genome_slice[offset:offset+n_in].reshape(2*d_ff, d_model)
        self.register_buffer('w_in', torch.from_numpy(raw_in.copy()).to(torch.int8))
        self.in_scale = 1.0 / math.sqrt(d_model * 0.1)
        offset += n_in
        
        n_dw = d_ff * kernel_size
        raw_dw = genome_slice[offset:offset+n_dw].reshape(d_ff, 1, kernel_size)
        self.register_buffer('w_dw', torch.from_numpy(raw_dw.copy()).to(torch.int8))
        self.dw_scale = 1.0 / math.sqrt(kernel_size)
        offset += n_dw
        
        n_out = d_ff * d_model
        raw_out = genome_slice[offset:offset+n_out].reshape(d_model, d_ff)
        self.register_buffer('w_out', torch.from_numpy(raw_out.copy()).to(torch.int8))
        self.out_scale = 1.0 / math.sqrt(d_ff * 0.1)
        offset += n_out
        
        self.genome_bytes = offset
        
        # Trainable adapters
        self.lora_in = LoRA(d_model, 2*d_ff, LORA_RANK)
        self.lora_out = LoRA(d_ff, d_model, LORA_RANK)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        res = x
        x = self.norm(x)
        
        # In-projection (int8 → bf16 on-the-fly)
        w = self.w_in.to(DTYPE) * self.in_scale
        ag = F.linear(x, w) + self.lora_in(x)
        del w
        a, g = ag.chunk(2, dim=-1)
        
        # Causal depthwise conv
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        dw = self.w_dw.to(DTYPE) * self.dw_scale
        a = F.conv1d(a, dw, groups=self.d_ff, dilation=self.dilation)
        del dw
        a = a.transpose(1, 2)
        
        # Gate
        y = F.silu(a) * torch.sigmoid(g)
        
        # Out-projection
        w = self.w_out.to(DTYPE) * self.out_scale
        out = F.linear(y, w) + self.lora_out(y)
        del w
        
        return res + out * self.scale

# ============================================================
# ZETAGRID 50B MODEL
# ============================================================

class ZetaGrid50B(nn.Module):
    def __init__(self, genome):
        super().__init__()
        
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        
        bytes_per_layer = D_MODEL*2*D_FF + D_FF*KERNEL_SIZE + D_FF*D_MODEL
        print(f"   Bytes per layer: {bytes_per_layer/1e6:.0f}M")
        print(f"   Total needed: {bytes_per_layer*N_LAYERS/1e9:.2f}GB")
        print(f"   Genome available: {len(genome)/1e9:.2f}GB")
        
        self.layers = nn.ModuleList()
        offset = 0
        
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            
            if offset + bytes_per_layer > len(genome):
                offset = offset % (len(genome) - bytes_per_layer)
            
            layer = TCNLayer50B(D_MODEL, D_FF, KERNEL_SIZE, dil, genome[offset:])
            self.layers.append(layer)
            offset += bytes_per_layer
            
            if (i+1) % 16 == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"   Layer {i+1}/{N_LAYERS} | VRAM: {vram:.1f}GB")
        
        self.norm_f = RMSNorm(D_MODEL)
        
        frozen_b = sum(b.numel() for b in self.buffers())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n   📊 ZETAGRID 50B:")
        print(f"   Frozen backbone: {frozen_b/1e9:.1f}B params (int8)")
        print(f"   Trainable: {train_p/1e6:.0f}M params")
        print(f"   Total: {(frozen_b+train_p)/1e9:.1f}B")
    
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
    def generate(self, prompt_bytes, max_new=200, temperature=0.8, top_k=50):
        self.eval()
        idx = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
        for _ in range(max_new):
            idx_crop = idx[:, -1024:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return bytes(idx[0].cpu().tolist()).decode('utf-8', errors='replace')

# ============================================================
# TRANSFER 25B LORA → 50B (first 32 layers)
# ============================================================

def transfer_25b_lora(model):
    """Load trained 25B LoRA weights into first 32 layers of 50B"""
    if not os.path.exists(CKPT_25B_PATH):
        print(f"\n[TRANSFER] No 25B checkpoint found, training from scratch")
        return
    
    print(f"\n[TRANSFER] Loading 25B LoRA: {CKPT_25B_PATH}")
    ckpt = torch.load(CKPT_25B_PATH, map_location='cpu')
    state_25b = ckpt.get('model', ckpt.get('model_state_dict', {}))
    
    transferred = 0
    for key, value in state_25b.items():
        # Map 25B keys to first 32 layers of 50B
        if key in model.state_dict():
            try:
                model.state_dict()[key].copy_(value)
                transferred += 1
            except:
                pass
    
    del ckpt, state_25b
    gc.collect()
    print(f"   ✅ Transferred {transferred} tensors from 25B → 50B")

# ============================================================
# DATA
# ============================================================

def load_data():
    print("\n[DATA] Loading...")
    parts = []
    for p in DATA_PATHS:
        if os.path.exists(p):
            d = np.fromfile(p, dtype=np.uint8)
            parts.append(d)
            print(f"   ✅ {os.path.basename(p)}: {len(d)/1e6:.0f}M bytes")
    combined = np.concatenate(parts)
    print(f"   Total: {len(combined)/1e9:.2f}GB")
    return combined

def get_batch(data):
    starts = np.random.randint(0, len(data)-SEQ_LEN-1, BATCH_SIZE)
    x = np.stack([data[s:s+SEQ_LEN] for s in starts]).astype(np.int64)
    y = np.stack([data[s+1:s+1+SEQ_LEN] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ============================================================
# TRAIN
# ============================================================

def train():
    # Step 1: Expand genome
    genome = expand_genome()
    
    # Step 2: Build model
    print(f"\n{'='*70}")
    print(f"BUILDING ZETAGRID 50B")
    print(f"{'='*70}")
    print(f"   D={D_MODEL}, FF={D_FF}, L={N_LAYERS}")
    print(f"   LoRA rank: {LORA_RANK}")
    
    model = ZetaGrid50B(genome).to(DEVICE)
    del genome
    gc.collect()
    torch.cuda.empty_cache()
    
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n   💾 VRAM: {vram:.1f}GB / 48GB")
    
    # Step 3: Transfer 25B LoRA (if available)
    transfer_25b_lora(model)
    
    # Step 4: Load data
    data = load_data()
    
    # Step 5: Train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        r = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        return LR * 0.1 + 0.5 * (LR - LR*0.1) * (1 + math.cos(math.pi * r))
    
    print(f"\n{'='*70}")
    print(f"TRAINING 50B (frozen backbone + adapters)")
    print(f"{'='*70}")
    print(f"   Steps: {TOTAL_STEPS:,} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"   Random baseline: {math.log(VOCAB_SIZE):.2f}")
    
    t0 = time.time()
    rloss = 0.0
    best = 99.0
    model.train()
    
    for step in range(1, TOTAL_STEPS+1):
        lr = get_lr(step)
        for pg in optimizer.param_groups: pg['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)
        al = 0.0
        
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(data)
            with torch.amp.autocast('cuda', dtype=DTYPE):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            loss.backward()
            al += loss.item()
        
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        rloss += al
        
        if step % LOG_EVERY == 0:
            avg = rloss / LOG_EVERY
            el = time.time() - t0
            sps = step / el
            eta = (TOTAL_STEPS - step) / sps / 60
            ppl = math.exp(min(avg, 20))
            print(f"Step {step:>6,}/{TOTAL_STEPS:,} | Loss: {avg:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | {sps:.2f} s/s | ETA: {eta:.0f}m")
            if avg < best: best = avg
            rloss = 0.0
        
        if step % SAVE_EVERY == 0:
            p = f"{SAVE_DIR}/zeta50b_step{step}.pt"
            print(f"\n💾 {p}")
            torch.save({'step': step, 'model': model.state_dict(), 'loss': best}, p)
            
            model.eval()
            for pr in ["The ", "Hello ", "In the "]:
                out = model.generate(list(pr.encode('utf-8')), max_new=80)
                print(f"   \"{pr}\" → {out[:200]}")
            model.train()
            print()
    
    final = f"{SAVE_DIR}/zeta50b_FINAL.pt"
    torch.save({'model': model.state_dict(), 'loss': best}, final)
    
    print(f"\n{'='*70}")
    print(f"50B DONE! Best: {best:.4f} | Time: {(time.time()-t0)/3600:.1f}h")
    print(f"{'='*70}")
    
    model.eval()
    for p in ["The future of AI", "Once upon a time", "Python is"]:
        out = model.generate(list(p.encode('utf-8')), max_new=150, temperature=0.7)
        print(f"\n   \"{p}\" → {out[:300]}")

if __name__ == "__main__":
    train()
