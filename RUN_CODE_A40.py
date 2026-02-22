#!/usr/bin/env python3
"""
ZETAGRID 25B - CODE SPECIALIST SOUL
=====================================
Fork from V4 Expanded checkpoint (rank 512).
Trains on 5GB code-only dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import math
import gc
import sys

print("=" * 70)
print("ZETAGRID 25B - CODE SPECIALIST SOUL")
print("Fork from V4 | TCN + Genome + LoRA Rank 512")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
DATA_BIN = f"{BASE_DIR}/data/pretrain/code_5gb.bin"
SAVE_DIR = f"{BASE_DIR}/code_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Resume from V4 Expanded
V4_CHECKPOINT = f"{BASE_DIR}/v4_checkpoints/zeta25b_v4_expanded_FINAL.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model (same as V4)
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 32
KERNEL_SIZE = 3
LORA_RANK = 512
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# Training (tuned for code specialization)
SEQ_LEN = 512          # Longer context for code
BATCH_SIZE = 4          # Reduced for longer SEQ_LEN
GRAD_ACCUM = 8          # Effective batch = 32
LR = 2e-5               # Gentle: preserving V4 knowledge
WARMUP_STEPS = 150
TOTAL_STEPS = 8000
SAVE_EVERY = 1000
LOG_EVERY = 25
GRAD_CLIP = 1.0

# ============================================================
# GENOME WEIGHT BANK
# ============================================================

class GenomeWeightBank:
    def __init__(self, genome_path):
        print(f"\n[GENOME] Loading {genome_path}...")
        raw = np.load(genome_path)
        print(f"   Raw size: {len(raw)/1e9:.2f}GB")
        self.data = torch.from_numpy(raw.astype(np.float32)).to(DTYPE).to(DEVICE)
        del raw; gc.collect()
        self.offset = 0
        print(f"   ✅ Genome on GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    def get_weight(self, out_features, in_features):
        n = out_features * in_features
        if self.offset + n > len(self.data):
            self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(out_features, in_features)
        self.offset += n
        scale = 1.0 / math.sqrt(in_features * 0.1)
        return (chunk * scale).contiguous()
    
    def get_conv_weight(self, channels, kernel_size):
        n = channels * kernel_size
        if self.offset + n > len(self.data):
            self.offset = 0
        chunk = self.data[self.offset : self.offset + n].reshape(channels, 1, kernel_size)
        self.offset += n
        scale = 1.0 / math.sqrt(kernel_size)
        return (chunk * scale).contiguous()

# ============================================================
# MODEL
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

class TCNLayer25B(nn.Module):
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
        a = F.conv1d(a, self.w_dw, groups=D_FF, dilation=self.dilation)
        a = a.transpose(1, 2)
        y = F.silu(a) * torch.sigmoid(g)
        out = F.linear(y, self.w_out) + self.lora_out(y)
        return res + out * self.scale

class ZetaGrid25B(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer25B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
            if (i+1) % 8 == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"   Layer {i+1}/{N_LAYERS} | VRAM: {vram:.1f}GB")
        self.norm_f = RMSNorm(D_MODEL)
        frozen_b = sum(b.numel() for b in self.buffers())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n   📊 CODE SPECIALIST SOUL:")
        print(f"   Frozen backbone: {frozen_b/1e9:.1f}B params")
        print(f"   Trainable: {train_p/1e6:.0f}M params")
    
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
    def generate(self, prompt_bytes, max_new=300, temperature=0.6, top_k=30):
        self.eval()
        idx = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
        for _ in range(max_new):
            idx_crop = idx[:, -1024:]
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return bytes(idx[0].cpu().tolist()).decode('utf-8', errors='replace')

# ============================================================
# DATA
# ============================================================

def load_data():
    print(f"\n[DATA] Loading {DATA_BIN}...")
    if not os.path.exists(DATA_BIN):
        print(f"   ❌ NOT FOUND: {DATA_BIN}")
        sys.exit(1)
    d = np.fromfile(DATA_BIN, dtype=np.uint8)
    print(f"   ✅ {os.path.basename(DATA_BIN)}: {len(d)/1e9:.2f}GB")
    return d

def get_batch(data):
    starts = np.random.randint(0, len(data)-SEQ_LEN-1, BATCH_SIZE)
    x = np.stack([data[s:s+SEQ_LEN] for s in starts]).astype(np.int64)
    y = np.stack([data[s+1:s+1+SEQ_LEN] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ============================================================
# TRAIN
# ============================================================

def train():
    bank = GenomeWeightBank(GENOME_PATH)
    
    print(f"\n{'='*70}")
    print(f"BUILDING CODE SPECIALIST (RANK {LORA_RANK})")
    print(f"{'='*70}")
    
    model = ZetaGrid25B(bank)
    del bank.data; del bank; gc.collect(); torch.cuda.empty_cache()
    model = model.to(DEVICE)
    
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n   💾 VRAM: {vram:.1f}GB / 48GB")
    
    data = load_data()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        r = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        return LR * 0.1 + 0.5 * (LR - LR*0.1) * (1 + math.cos(math.pi * r))
    
    # Load V4 checkpoint
    start_step = 0
    best = 99.0
    
    # Check for code resume first
    ckpt_files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith('zeta25b_code')]) if os.path.exists(SAVE_DIR) else []
    if ckpt_files:
        latest = os.path.join(SAVE_DIR, ckpt_files[-1])
        print(f"\n[RESUME] Loading code checkpoint: {latest}...")
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        start_step = ckpt.get('step', 0)
        best = ckpt.get('loss', 99.0)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt; gc.collect(); torch.cuda.empty_cache()
        print(f"   ✅ Resuming code training from step {start_step}")
    elif os.path.exists(V4_CHECKPOINT):
        print(f"\n[INIT] Loading V4 Expanded: {V4_CHECKPOINT}...")
        ckpt = torch.load(V4_CHECKPOINT, map_location=DEVICE, weights_only=False)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        best = ckpt.get('loss', 99.0)
        print(f"   ✅ V4 loaded (loss: {best:.4f})")
        del ckpt, state; gc.collect(); torch.cuda.empty_cache()
    else:
        print("   ⚠️ No V4 checkpoint! Training from genome only.")
    
    print(f"\n{'='*70}")
    print(f"CODE SPECIALIST TRAINING")
    print(f"{'='*70}")
    print(f"   Steps: {start_step+1} → {TOTAL_STEPS:,} | SEQ_LEN: {SEQ_LEN}")
    print(f"   Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM} | LR: {LR}")
    print(f"   Data: {len(data)/1e9:.1f}GB")
    sys.stdout.flush()
    
    t0 = time.time()
    rloss = 0.0
    model.train()
    
    for step in range(start_step+1, TOTAL_STEPS+1):
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
            sps = (step - start_step) / el if el > 0 else 0
            eta = (TOTAL_STEPS - step) / sps / 60 if sps > 0 else 0
            ppl = math.exp(min(avg, 20))
            print(f"Step {step:>6,}/{TOTAL_STEPS:,} | Loss: {avg:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | {sps:.2f} s/s | ETA: {eta:.0f}m")
            sys.stdout.flush()
            if avg < best: best = avg
            rloss = 0.0
        
        if step % SAVE_EVERY == 0:
            p = f"{SAVE_DIR}/zeta25b_code_step{step}.pt"
            print(f"\n💾 Saving → {p}")
            sys.stdout.flush()
            try:
                trainable_state = {k: v for k, v in model.state_dict().items() 
                                   if not k.startswith('layers.') or 'lora' in k or 'scale' in k or 'norm' in k}
                torch.save({'step': step, 'model': trainable_state, 'loss': best,
                           'optimizer': optimizer.state_dict()}, p)
                
                # Strict cleanup: keep ONLY current and FINAL
                for f in os.listdir(SAVE_DIR):
                    if f.startswith('zeta25b_code_step') and f.endswith('.pt'):
                        fp = os.path.join(SAVE_DIR, f)
                        if fp != p and 'FINAL' not in f:
                            try:
                                os.remove(fp)
                                print(f"   🗑️ Deleted old: {f}")
                            except Exception as e:
                                print(f"   ⚠️ Could not delete {f}: {e}")
            except Exception as e:
                print(f"   ⚠️ Save failed: {e}")
            
            # Code-specific generation tests
            model.eval()
            try:
                code_prompts = [
                    "def fibonacci(n):\n",
                    "# Sort a list of integers\ndef sort_list(",
                    "class Database:\n    def __init__(self",
                    "import os\nimport sys\n\ndef main():\n",
                    "function fetchData(url) {\n",
                    "# Calculate the factorial\ndef factorial(",
                ]
                for pr in code_prompts:
                    out = model.generate(list(pr.encode('utf-8')), max_new=200, temperature=0.6)
                    print(f"   ```\n   {out[:350]}\n   ```")
            except Exception as e:
                print(f"   ⚠️ Generate failed: {e}")
            model.train()
            print()
            sys.stdout.flush()
    
    # FINAL
    final = f"{SAVE_DIR}/zeta25b_code_FINAL.pt"
    print(f"\n💾 Saving FINAL → {final}")
    trainable_state = {k: v for k, v in model.state_dict().items() 
                       if not k.startswith('layers.') or 'lora' in k or 'scale' in k or 'norm' in k}
    torch.save({'model': trainable_state, 'loss': best}, final)
    
    print(f"\n{'='*70}")
    print(f"CODE SPECIALIST DONE! Best loss: {best:.4f} | Time: {(time.time()-t0)/3600:.1f}h")
    print(f"{'='*70}")
    
    model.eval()
    final_prompts = [
        "def binary_search(arr, target):\n    ",
        "class LinkedList:\n    def __init__(self):\n        ",
        "# Read a CSV file and calculate average\nimport csv\n\ndef process_csv(filename):\n    ",
        "async function getData() {\n    try {\n        ",
    ]
    for p in final_prompts:
        out = model.generate(list(p.encode('utf-8')), max_new=300, temperature=0.5)
        print(f"\n   ```\n   {out[:500]}\n   ```")

if __name__ == "__main__":
    train()
