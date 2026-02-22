#!/usr/bin/env python3
"""
ZETAGRID 25B - V2 REPAIR
========================
IDENTICAL to ZETAGRID_PHASE2_GRADIENT.py (the proven V1 script)
with ONLY config changes:
  - TOTAL_STEPS: 15000 → 500
  - LR: 3e-4 → 1e-4 (fine-tuning, not from-scratch)
  - SAVE_DIR: repaired_checkpoints
  - Resume from zeta25b_step15000.pt (V1 checkpoint)
  - DATA_PATHS: uses existing .bin data

ALL architecture code is VERBATIM from V1.
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
print("ZETAGRID 25B - V2 REPAIR (V1-Clone)")
print("Non-Transformer LLM | TCN + Genome Backbone")
print("=" * 70)

# ============================================================
# CONFIG (only these lines differ from V1)
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
DATA_PATHS_BIN = [
    f"{BASE_DIR}/data/pretrain/clean_text_utf8.bin",   # V1 data (preferred)
]
DATA_PATHS_JSONL = [
    f"{BASE_DIR}/repair_mix.jsonl",                    # Fallback (1.4GB)
]
SAVE_DIR = f"{BASE_DIR}/repaired_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Resume from V1 checkpoint
V1_CHECKPOINT = f"{BASE_DIR}/zeta25b_step15000.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model (IDENTICAL to V1)
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 32
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# Training (CHANGED for fine-tuning)
SEQ_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4        # Effective batch = 32
LR = 1e-4             # Lower than V1 (3e-4) for fine-tuning
WARMUP_STEPS = 50     # Shorter warmup (resuming, not cold start)
TOTAL_STEPS = 500     # Repair is shorter
SAVE_EVERY = 100
LOG_EVERY = 10
GRAD_CLIP = 1.0

# ============================================================
# GENOME → BF16 WEIGHT CONVERTER (VERBATIM from V1)
# ============================================================

class GenomeWeightBank:
    """Pre-converts genome int8 → BF16 with proper Xavier scaling"""
    
    def __init__(self, genome_path):
        print(f"\n[GENOME] Loading {genome_path}...")
        raw = np.load(genome_path)
        print(f"   Raw size: {len(raw)/1e9:.2f}GB ({len(raw):,} bytes)")
        
        # Convert entire genome to float32 once
        print("   Converting genome → BF16 (one-time)...")
        self.data = torch.from_numpy(raw.astype(np.float32)).to(DTYPE).to(DEVICE)
        del raw
        gc.collect()
        
        self.offset = 0
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"   ✅ Genome on GPU as BF16: {vram:.1f}GB VRAM")
    
    def get_weight(self, out_features, in_features):
        """Extract a weight matrix with proper Xavier scaling"""
        n = out_features * in_features
        
        # Wrap around if needed
        if self.offset + n > len(self.data):
            self.offset = 0
        
        # Extract and reshape
        chunk = self.data[self.offset : self.offset + n].reshape(out_features, in_features)
        self.offset += n
        
        # Xavier scaling for ternary weights
        density = 0.1
        scale = 1.0 / math.sqrt(in_features * density)
        
        return (chunk * scale).contiguous()
    
    def get_conv_weight(self, channels, kernel_size):
        """Extract conv weight with proper scaling"""
        n = channels * kernel_size
        if self.offset + n > len(self.data):
            self.offset = 0
        
        chunk = self.data[self.offset : self.offset + n].reshape(channels, 1, kernel_size)
        self.offset += n
        
        scale = 1.0 / math.sqrt(kernel_size)
        return (chunk * scale).contiguous()

# ============================================================
# MODEL COMPONENTS (VERBATIM from V1)
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
    """Low-rank trainable adapter"""
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
    
    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

# ============================================================
# TCN LAYER (VERBATIM from V1)
# ============================================================

class TCNLayer25B(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, bank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Trainable norm
        self.norm = RMSNorm(d_model)
        
        # FROZEN weights (pre-converted BF16, no grad)
        self.register_buffer('w_in', bank.get_weight(2 * d_ff, d_model))
        self.register_buffer('w_dw', bank.get_conv_weight(d_ff, kernel_size))
        self.register_buffer('w_out', bank.get_weight(d_model, d_ff))
        
        # Trainable LoRA adapters
        self.lora_in = LoRA(d_model, 2 * d_ff, LORA_RANK)
        self.lora_out = LoRA(d_ff, d_model, LORA_RANK)
        
        # Trainable gate scale (starts small for stability)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        res = x
        x = self.norm(x).to(DTYPE)  # Ensure BF16 after norm
        
        # In-projection: frozen + LoRA
        ag = F.linear(x, self.w_in) + self.lora_in(x)
        a, g = ag.chunk(2, dim=-1)
        
        # Causal depthwise conv
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, self.w_dw, groups=D_FF, dilation=self.dilation)
        a = a.transpose(1, 2)
        
        # Gate
        y = F.silu(a) * torch.sigmoid(g)
        
        # Out-projection: frozen + LoRA
        out = F.linear(y, self.w_out) + self.lora_out(y)
        
        return res + out * self.scale

# ============================================================
# ZETAGRID 25B MODEL (VERBATIM from V1)
# ============================================================

class ZetaGrid25B(nn.Module):
    def __init__(self, bank):
        super().__init__()
        
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer25B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
            if (i+1) % 8 == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"   Layer {i+1}/{N_LAYERS} | VRAM: {vram:.1f}GB")
        
        self.norm_f = RMSNorm(D_MODEL)
        
        # Count
        frozen_b = sum(b.numel() for b in self.buffers())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n   📊 ZETAGRID 25B:")
        print(f"   Frozen backbone: {frozen_b/1e9:.1f}B params")
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
# DATA (VERBATIM from V1)
# ============================================================

def load_data():
    import json as json_mod
    print("\n[DATA] Loading...")
    parts = []
    
    # Try .bin first (V1 format, preferred)
    for p in DATA_PATHS_BIN:
        if os.path.exists(p):
            d = np.fromfile(p, dtype=np.uint8)
            parts.append(d)
            print(f"   ✅ {os.path.basename(p)}: {len(d)/1e6:.0f}M bytes (.bin)")
    
    # Fallback to JSONL if no .bin found
    if not parts:
        print("   ⚠️ No .bin files found. Falling back to JSONL...")
        for p in DATA_PATHS_JSONL:
            if os.path.exists(p):
                print(f"   📖 Reading {os.path.basename(p)}...")
                text_chunks = []
                count = 0
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json_mod.loads(line)
                            # Handle SFT chat format: {"messages": [{"role":..., "content":...}, ...]}
                            if 'messages' in obj:
                                for msg in obj['messages']:
                                    c = msg.get('content', '')
                                    if isinstance(c, str) and len(c) > 5:
                                        text_chunks.append(c)
                                        count += 1
                            else:
                                # Flat format: {"text": ...} or {"content": ...}
                                text = obj.get('text', obj.get('content', obj.get('output', '')))
                                if isinstance(text, str) and len(text) > 10:
                                    text_chunks.append(text)
                                    count += 1
                        except:
                            # Raw text line
                            if len(line) > 10:
                                text_chunks.append(line)
                                count += 1
                
                # Convert text → raw UTF-8 bytes (matching V1 format)
                combined_text = "\n".join(text_chunks)
                raw_bytes = np.frombuffer(combined_text.encode('utf-8'), dtype=np.uint8)
                parts.append(raw_bytes)
                print(f"   ✅ {os.path.basename(p)}: {count:,} entries → {len(raw_bytes)/1e6:.0f}M bytes")
                del text_chunks, combined_text
                gc.collect()
    
    if not parts:
        print("   ❌ No data found! Cannot train.")
        exit(1)
    
    combined = np.concatenate(parts)
    print(f"   Total: {len(combined)/1e9:.2f}GB")
    return combined

def get_batch(data):
    starts = np.random.randint(0, len(data)-SEQ_LEN-1, BATCH_SIZE)
    x = np.stack([data[s:s+SEQ_LEN] for s in starts]).astype(np.int64)
    y = np.stack([data[s+1:s+1+SEQ_LEN] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ============================================================
# TRAIN (VERBATIM from V1, with V2 config)
# ============================================================

def train():
    bank = GenomeWeightBank(GENOME_PATH)
    
    print(f"\n{'='*70}")
    print(f"BUILDING ZETAGRID 25B")
    print(f"{'='*70}")
    
    model = ZetaGrid25B(bank)
    
    # Free genome bank (weights are now in model buffers)
    del bank.data
    del bank
    gc.collect()
    torch.cuda.empty_cache()
    
    model = model.to(DEVICE)
    
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n   💾 VRAM after model: {vram:.1f}GB / 48GB")
    
    data = load_data()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        r = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        return LR * 0.1 + 0.5 * (LR - LR*0.1) * (1 + math.cos(math.pi * r))
    
    # ============================
    # LOAD V1 CHECKPOINT (KEY CHANGE)
    # ============================
    start_step = 0
    best = 99.0
    
    if os.path.exists(V1_CHECKPOINT):
        print(f"\n[RESUME] Loading V1 checkpoint: {V1_CHECKPOINT}...")
        ckpt = torch.load(V1_CHECKPOINT, map_location=DEVICE)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        best = ckpt.get('loss', 99.0)
        print(f"   ✅ V1 checkpoint loaded (loss: {best:.4f})")
        del ckpt, state; gc.collect(); torch.cuda.empty_cache()
    else:
        # Fallback: check repaired_checkpoints
        ckpt_files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith('zeta25b')]) if os.path.exists(SAVE_DIR) else []
        if ckpt_files:
            latest = os.path.join(SAVE_DIR, ckpt_files[-1])
            print(f"\n[RESUME] Loading {latest}...")
            ckpt = torch.load(latest, map_location=DEVICE)
            model.load_state_dict(ckpt['model'], strict=False)
            start_step = ckpt.get('step', 0)
            best = ckpt.get('loss', 99.0)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            del ckpt; gc.collect(); torch.cuda.empty_cache()
            print(f"   ✅ Resuming from step {start_step}, best loss: {best:.4f}")
        else:
            print(f"\n   ⚠️ No checkpoint found. Training from Genome only.")
    
    print(f"\n{'='*70}")
    print(f"V2 REPAIR TRAINING (25B frozen + adapters)")
    print(f"{'='*70}")
    print(f"   Steps: {start_step+1} → {TOTAL_STEPS:,} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"   Random baseline loss: {math.log(VOCAB_SIZE):.2f}")
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
            sps = step / el if el > 0 else 0
            eta = (TOTAL_STEPS - step) / sps / 60 if sps > 0 else 0
            ppl = math.exp(min(avg, 20))
            print(f"Step {step:>6,}/{TOTAL_STEPS:,} | Loss: {avg:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | {sps:.2f} s/s | ETA: {eta:.0f}m")
            sys.stdout.flush()
            if avg < best: best = avg
            rloss = 0.0
        
        if step % SAVE_EVERY == 0:
            p = f"{SAVE_DIR}/zeta25b_v2_step{step}.pt"
            print(f"\n💾 Saving trainable params only → {p}")
            sys.stdout.flush()
            try:
                # Save ONLY trainable params (~500MB-1GB, not 13GB!)
                trainable_state = {k: v for k, v in model.state_dict().items() 
                                   if not k.startswith('layers.') or 'lora' in k or 'scale' in k or 'norm' in k}
                torch.save({'step': step, 'model': trainable_state, 'loss': best}, p)
                
                # Delete old checkpoints to save space
                for old in os.listdir(SAVE_DIR):
                    old_path = os.path.join(SAVE_DIR, old)
                    if old_path != p and 'FINAL' not in old and 'v2' in old:
                        os.remove(old_path)
                        print(f"   🗑️ Deleted old: {old}")
            except Exception as e:
                print(f"   ⚠️ Save failed (disk full?): {e}")
            
            model.eval()
            try:
                for pr in ["The ", "Hello ", "In the "]:
                    out = model.generate(list(pr.encode('utf-8')), max_new=80)
                    print(f"   \"{pr}\" → {out[:200]}")
            except Exception as e:
                print(f"   ⚠️ Generate failed: {e}")
            model.train()
            print()
            sys.stdout.flush()
    
    final = f"{SAVE_DIR}/zeta25b_v2_FINAL.pt"
    print(f"\n💾 Saving FINAL → {final}")
    trainable_state = {k: v for k, v in model.state_dict().items() 
                       if not k.startswith('layers.') or 'lora' in k or 'scale' in k or 'norm' in k}
    torch.save({'model': trainable_state, 'loss': best}, final)
    
    print(f"\n{'='*70}")
    print(f"V2 REPAIR DONE! Best loss: {best:.4f} | Time: {(time.time()-t0)/3600:.1f}h")
    print(f"{'='*70}")
    
    model.eval()
    for p in ["The future of AI", "Once upon a time", "Python is", "Hello world"]:
        out = model.generate(list(p.encode('utf-8')), max_new=150, temperature=0.7)
        print(f"\n   \"{p}\" → {out[:300]}")

if __name__ == "__main__":
    train()
