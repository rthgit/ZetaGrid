#!/usr/bin/env python3
"""
ZETAGRID 25B - INTERACTIVE INFERENCE
Load trained checkpoint and generate text interactively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import gc

print("=" * 70)
print("ZETAGRID 25B - INTERACTIVE INFERENCE")
print("Non-Transformer LLM | TCN Backbone")
print("=" * 70)

# ============================================================
# CONFIG (must match training)
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
DTYPE = torch.bfloat16
DEVICE = "cuda"

VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 32
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# Find best checkpoint
CKPT_DIR = f"{BASE_DIR}/phase2_checkpoints"

# ============================================================
# MODEL (same as training)
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

class GenomeWeightBank:
    def __init__(self, genome_path):
        print(f"[GENOME] Loading {genome_path}...")
        raw = np.load(genome_path)
        self.data = torch.from_numpy(raw.astype(np.float32)).to(DTYPE).to(DEVICE)
        del raw; gc.collect()
        self.offset = 0
        print(f"   ✅ Genome on GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    def get_weight(self, out_f, in_f):
        n = out_f * in_f
        if self.offset + n > len(self.data): self.offset = 0
        chunk = self.data[self.offset:self.offset+n].reshape(out_f, in_f)
        self.offset += n
        scale = 1.0 / math.sqrt(in_f * 0.1)
        return (chunk * scale).contiguous()
    
    def get_conv_weight(self, channels, ks):
        n = channels * ks
        if self.offset + n > len(self.data): self.offset = 0
        chunk = self.data[self.offset:self.offset+n].reshape(channels, 1, ks)
        self.offset += n
        return (chunk * (1.0/math.sqrt(ks))).contiguous()

class TCNLayer25B(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, bank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.d_ff = d_ff
        self.norm = RMSNorm(d_model)
        self.register_buffer('w_in', bank.get_weight(2*d_ff, d_model))
        self.register_buffer('w_dw', bank.get_conv_weight(d_ff, kernel_size))
        self.register_buffer('w_out', bank.get_weight(d_model, d_ff))
        self.lora_in = LoRA(d_model, 2*d_ff, LORA_RANK)
        self.lora_out = LoRA(d_ff, d_model, LORA_RANK)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        res = x
        x = self.norm(x).to(DTYPE)
        ag = F.linear(x, self.w_in) + self.lora_in(x)
        a, g = ag.chunk(2, dim=-1)
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, self.w_dw, groups=self.d_ff, dilation=self.dilation)
        a = a.transpose(1, 2)
        y = F.silu(a) * torch.sigmoid(g)
        out = F.linear(y, self.w_out) + self.lora_out(y)
        return res + out * self.scale

class ZetaGrid25B(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer25B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
            if (i+1) % 8 == 0:
                print(f"   Layer {i+1}/{N_LAYERS}")
        self.norm_f = RMSNorm(D_MODEL)
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
            for layer in self.layers:
                x = layer(x)
            x = self.norm_f(x)
            return F.linear(x.float(), self.emb.weight.float())
    
    @torch.no_grad()
    def generate(self, prompt, max_new=300, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text from prompt string"""
        prompt_bytes = list(prompt.encode('utf-8'))
        idx = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
        
        generated_bytes = []
        
        for i in range(max_new):
            idx_crop = idx[:, -1024:]
            logits = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k:
                v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = -float('Inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_byte], dim=1)
            generated_bytes.append(next_byte.item())
            
            # Print progress every 50 bytes
            if (i+1) % 50 == 0:
                partial = bytes(generated_bytes).decode('utf-8', errors='replace')
                print(f"   [{i+1} bytes] ...{partial[-80:]}")
        
        # Decode full output
        all_bytes = prompt_bytes + generated_bytes
        text = bytes(all_bytes).decode('utf-8', errors='replace')
        
        # Debug: show byte distribution
        from collections import Counter
        top5 = Counter(generated_bytes).most_common(5)
        print(f"   [DEBUG] Top bytes: {[(b, chr(b) if 32<=b<127 else f'0x{b:02x}', c) for b,c in top5]}")
        
        return text

# ============================================================
# LOAD & INTERACT
# ============================================================

def find_best_checkpoint():
    """Find latest/best checkpoint"""
    if not os.path.exists(CKPT_DIR):
        return None
    
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')]
    if not files:
        return None
    
    # Prefer FINAL, otherwise latest step
    for f in files:
        if 'FINAL' in f:
            return os.path.join(CKPT_DIR, f)
    
    # Sort by step number
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or '0'))
    return os.path.join(CKPT_DIR, files[-1])

def main():
    # Build model
    print("\n[1/3] Building model...")
    bank = GenomeWeightBank(GENOME_PATH)
    model = ZetaGrid25B(bank).to(DEVICE)
    del bank.data; del bank; gc.collect()
    
    # Load checkpoint
    ckpt_path = find_best_checkpoint()
    if ckpt_path:
        print(f"\n[2/3] Loading checkpoint: {os.path.basename(ckpt_path)}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get('model', ckpt.get('model_state_dict', {}))
        model.load_state_dict(state, strict=False)
        loss = ckpt.get('loss', ckpt.get('best_loss', '?'))
        step = ckpt.get('step', '?')
        print(f"   ✅ Loaded step {step}, best loss: {loss}")
        del ckpt, state; gc.collect(); torch.cuda.empty_cache()
    else:
        print("\n[2/3] ⚠️ No checkpoint found! Using untrained model.")
    
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n[3/3] Ready! VRAM: {vram:.1f}GB")
    
    # Preset prompts
    print(f"\n{'='*70}")
    print("SAMPLE GENERATIONS")
    print(f"{'='*70}")
    
    presets = [
        "The future of artificial intelligence is",
        "Once upon a time, there was a",
        "Python programming is",
        "In a world where technology",
        "The most important thing in life is",
    ]
    
    for p in presets:
        print(f"\n📝 Prompt: \"{p}\"")
        out = model.generate(p, max_new=200, temperature=0.7, top_k=40)
        print(f"   → {out[:400]}")
        print("-" * 50)
    
    # Interactive mode
    print(f"\n{'='*70}")
    print("INTERACTIVE MODE")
    print("Commands: /temp 0.5 | /topk 30 | /topp 0.8 | /len 200 | quit")
    print(f"{'='*70}")
    
    temp = 0.7
    top_k = 40
    top_p = 0.9
    max_len = 300
    
    while True:
        try:
            prompt = input(f"\n🔵 ZetaGrid [T={temp} K={top_k}] > ")
        except (EOFError, KeyboardInterrupt):
            break
        
        if not prompt:
            continue
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        
        # Settings commands
        if prompt.startswith('/temp '):
            temp = float(prompt.split()[1])
            print(f"   Temperature → {temp}")
            continue
        if prompt.startswith('/topk '):
            top_k = int(prompt.split()[1])
            print(f"   Top-K → {top_k}")
            continue
        if prompt.startswith('/topp '):
            top_p = float(prompt.split()[1])
            print(f"   Top-P → {top_p}")
            continue
        if prompt.startswith('/len '):
            max_len = int(prompt.split()[1])
            print(f"   Max length → {max_len}")
            continue
        
        # Generate
        print(f"\n🟢 Generating...")
        out = model.generate(prompt, max_new=max_len, temperature=temp, top_k=top_k, top_p=top_p)
        print(f"\n{out}")
    
    print("\n👋 Bye!")

if __name__ == "__main__":
    main()
