import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import gc
from tqdm import tqdm

print("=" * 70)
print("ZETAGRID 50B - SFT INFERENCE TEST (RUNPOD)")
print("Fractal TCN | Local Genome Reuse")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/workspace/zetagrid_50b"
GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
# Point to your NEW SFT checkpoint (Step 2000)
CKPT_PATH = f"{BASE_DIR}/phase4_sft_checkpoints/zeta50b_sft_step2000.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16

# 50B Architecture Spec
VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 64  # <--- 50B has 64 layers! (25B had 32)
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# ============================================================
# MODEL DEFINITION (Copied from Training Script)
# ============================================================

class GenomeWeightBank:
    def __init__(self, genome_path):
        print(f"[GENOME] Loading {genome_path}...")
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
        # No 'groups' arg here because bank gives [C, 1, K] weights which implies depthwise if groups=C
        # But let's check training script usage. 
        # Train script: F.conv1d(a, self.w_dw, groups=D_FF, dilation=self.dilation)
        # Here D_FF is d_ff from init.
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
        print(f"🏗️ Building 64 Layers (This may take ~2-3 mins)...")
        for i in tqdm(range(N_LAYERS), desc="Loading Layers"):
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(TCNLayer50B(D_MODEL, D_FF, KERNEL_SIZE, dil, bank))
        self.norm_f = RMSNorm(D_MODEL)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = F.linear(x.float(), self.emb.weight.float())
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new=200, temperature=0.7):
        """Simple greedy/sampling generation"""
        idx = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)
        
        print(f"\n📝 Prompt: {prompt}", end="", flush=True)
        
        for _ in range(max_new):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            # Print last char
            char = bytes([next_token.item()]).decode('utf-8', errors='replace')
            print(char, end="", flush=True)
            
        print("\n" + "-"*50)

# ============================================================
# MAIN
# ============================================================

def chat(model):
    print("\n🟢 CHAT MODE (Type 'quit' to exit)")
    print("Format: <USER>: [your input]")
    
    while True:
        user_input = input("\n👤 YOU: ")
        if user_input.lower() in ["quit", "exit"]: break
        
        # Format for SFT
        prompt = f"<USER>: {user_input}\n<ASSISTANT>:"
        
        model.generate(prompt, max_new=300, temperature=0.6)

if __name__ == "__main__":
    # 1. Load Genome
    bank = GenomeWeightBank(GENOME_PATH)
    
    # 2. Init Model
    print("🏗️ Building 50B Model...")
    model = ZetaGrid50B(bank).to(DEVICE)
    del bank.data; del bank; gc.collect()
    
    # 3. Load SFT Checkpoint
    if os.path.exists(CKPT_PATH):
        print(f"📥 Loading SFT Checkpoint: {CKPT_PATH}")
        # weights_only=False because we trust our own file
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        
        # Handle if it wrapped in 'model' key (likely yes from training script)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint Loaded Successfully!")
    else:
        print(f"⚠️ Checkpoint not found at {CKPT_PATH}")
        print("   Using random weights (Output will be garbage)...")

    model.eval()
    
    # 4. Start Chat
    chat(model)
