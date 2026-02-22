import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import gc
import sys
from tqdm import tqdm

print("=" * 70)
print("ZETAGRID 50B - FAST INFERENCE (JIT) - V2 (EXACT OFFSETS)")
print("Starts in seconds. Uses Virtual Weights (Zero-Copy).")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

if os.name == 'nt':
    print("🖥️ Windows Environment Detected")
    BASE_DIR = "C:/Users/PC/Desktop/cpu-da" 
    GENOME_PATH = "C:/Users/PC/Desktop/Glifosv2/kam_llm_3b/zetagrid_25b_production.npy"
    CKPT_PATH = f"{BASE_DIR}/zeta50b_sft_step2000.pt"
else:
    print("🐧 Linux/RunPod Environment Detected")
    BASE_DIR = "/workspace/zetagrid_50b"
    GENOME_PATH = f"{BASE_DIR}/zetagrid_25b_production.npy"
    CKPT_PATH = f"{BASE_DIR}/phase4_sft_checkpoints/zeta50b_sft_step2000.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16

VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 64
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

# ============================================================
# SIMULATED BANK (For correct offset calculation)
# ============================================================

class SimulatedBank:
    """Mimics the sequential allocator without loading data."""
    def __init__(self, total_size):
        self.total_size = total_size
        self.offset = 0
        
    def get_offset(self, size):
        # Match training logic exactly:
        # if self.offset + n > len(self.data): self.offset = 0
        if self.offset + size > self.total_size:
            self.offset = 0
        
        start = self.offset
        end = start + size
        self.offset += size
        return (start, end)

# ============================================================
# FAST VIRTUAL LAYER (ZERO-COPY)
# ============================================================

class VirtualTCNLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, genome_tensor, offsets):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.d_model = d_model
        self.d_ff = d_ff
        self.kernel_size = kernel_size
        
        # Norm
        self.norm = nn.Parameter(torch.ones(d_model)) 
        self.eps = 1e-6
        
        # LoRA
        self.lora_in = nn.Module()
        self.lora_in.A = nn.Parameter(torch.zeros(LORA_RANK, d_model))
        self.lora_in.B = nn.Parameter(torch.zeros(2*d_ff, LORA_RANK))
        
        self.lora_out = nn.Module()
        self.lora_out.A = nn.Parameter(torch.zeros(LORA_RANK, d_ff))
        self.lora_out.B = nn.Parameter(torch.zeros(d_model, LORA_RANK))
        
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        # Genome Reference
        self.genome = genome_tensor
        
        # Exact Offsets from Simulation
        self.idx_in = offsets['w_in']
        self.idx_dw = offsets['w_dw']
        self.idx_out = offsets['w_out']
        
        # Scaling Factors
        self.scale_in = 1.0 / math.sqrt(d_model * 0.1)
        self.scale_dw = 1.0 / math.sqrt(kernel_size)
        self.scale_out = 1.0 / math.sqrt(d_ff * 0.1)

    def forward(self, x):
        # 0. RMSNorm (Compute in float32 for stability, cast output to BF16)
        res = x
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = (x.float() * rms).to(DTYPE) * self.norm.to(DTYPE)
        
        # 1. Linear IN (Virtual)
        # Slicing from genome (BF16)
        w_in_flat = self.genome[self.idx_in[0] : self.idx_in[1]]
        w_in = w_in_flat.view(2*self.d_ff, self.d_model)
        
        # Apply Linear + Scale
        # Force inputs to be same dtype
        ag_base = F.linear(x_norm, w_in) * self.scale_in
        
        # Apply LoRA (Cast weights to BF16)
        lora_in_A = self.lora_in.A.to(DTYPE)
        lora_in_B = self.lora_in.B.to(DTYPE)
        lora_in_out = (x_norm @ lora_in_A.T) @ lora_in_B.T
        
        ag = ag_base + lora_in_out
        a, g = ag.chunk(2, dim=-1)
        
        # 2. Conv DW (Virtual)
        w_dw_flat = self.genome[self.idx_dw[0] : self.idx_dw[1]]
        w_dw = w_dw_flat.view(self.d_ff, 1, self.kernel_size)
        
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        
        # Conv1d
        a = F.conv1d(a, w_dw, groups=self.d_ff, dilation=self.dilation) * self.scale_dw
        a = a.transpose(1, 2)
        
        y = F.silu(a) * torch.sigmoid(g)
        
        # 3. Linear OUT (Virtual)
        w_out_flat = self.genome[self.idx_out[0] : self.idx_out[1]]
        w_out = w_out_flat.view(self.d_model, self.d_ff)
        
        out_base = F.linear(y, w_out) * self.scale_out
        
        # LoRA Out
        lora_out_A = self.lora_out.A.to(DTYPE)
        lora_out_B = self.lora_out.B.to(DTYPE)
        lora_out_out = (y @ lora_out_A.T) @ lora_out_B.T
        
        return res + (out_base + lora_out_out) * self.scale.to(DTYPE)

class ZetaGrid50B_Fast(nn.Module):
    def __init__(self, genome_tensor):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL) # Just weights
        self.layers = nn.ModuleList()
        self.norm_f = nn.Parameter(torch.ones(D_MODEL))
        self.eps = 1e-6
        
        # Pre-calculate EXACT offsets
        print("🧮 Simulating Allocation Layout...")
        sim_bank = SimulatedBank(len(genome_tensor))
        
        n_w_in = 2 * D_FF * D_MODEL
        n_w_dw = D_FF * KERNEL_SIZE
        n_w_out = D_MODEL * D_FF
        
        print("⚡ Building 64 Virtual Layers...")
        for i in tqdm(range(N_LAYERS)):
            offsets = {}
            offsets['w_in'] = sim_bank.get_offset(n_w_in)
            offsets['w_dw'] = sim_bank.get_offset(n_w_dw)
            offsets['w_out'] = sim_bank.get_offset(n_w_out)
            
            dil = DILATION_CYCLE[i % len(DILATION_CYCLE)]
            self.layers.append(VirtualTCNLayer(D_MODEL, D_FF, KERNEL_SIZE, dil, genome_tensor, offsets))

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = (self.emb(idx) + self.pos_emb(pos)).to(DTYPE)
        for layer in self.layers:
            x = layer(x)
        
        # Final Norm
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x.float() * rms).to(DTYPE) * self.norm_f.to(DTYPE)
        
        # Head (tied embeddings)
        logits = F.linear(x, self.emb.weight.to(DTYPE))
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new=200, temperature=0.7, top_k=50):
        idx = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)
        print(f"\n📝 Prompt: {prompt}", end="", flush=True)
        
        for _ in range(max_new):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            char = bytes([next_token.item()]).decode('utf-8', errors='replace')
            print(char, end="", flush=True)
        print("\n" + "-"*50)

# ============================================================
# LOADER
# ============================================================

def load_checkpoint_into_fast_model(model, ckpt_path):
    print(f"📥 Loading Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    
    new_state = {}
    for k, v in state.items():
        name = k.replace('base.', '')
        if 'norm.w' in name:
            name = name.replace('norm.w', 'norm')
        if 'norm_f.w' in name:
            name = name.replace('norm_f.w', 'norm_f')
        new_state[name] = v
        
    print(f"   Checkpoint keys (first 10): {list(new_state.keys())[:10]}")
    print(f"   Model keys (first 10): {list(model.state_dict().keys())[:10]}")
    
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"   Missing: {missing[:10]}")
    print(f"   Unexpected: {unexpected[:10]}")
    
    # Verify critical weight
    if torch.all(model.layers[0].lora_in.A == 0):
        print("⚠️ WARNING: LoRA weights are still ZERO (Load Failed or Initialized)!")
    else:
        print("✅ LoRA weights loaded correctly (Non-Zero).")

def chat(model):
    print("\n🟢 CHAT MODE (FAST JIT) - Type 'quit' to exit")
    print("Commands: /temp 0.X | /topk XX | /tmpl <format> | /raw <text>")
    
    # Defaults
    temp = 0.6
    top_k = 50
    # Templates: 0=User/Assistant, 1=Human/Bot, 2=Raw
    template = "User: {input}\nAssistant:"
    
    while True:
        user_input = input(f"\n👤 YOU (T={temp}): ")
        if not user_input: continue
        if user_input.lower() in ["quit", "exit"]: break
        
        # Commands
        if user_input.startswith("/temp "):
            try:
                temp = float(user_input.split()[1])
                print(f"   Temperature set to {temp}")
            except: print("   Invalid temperature")
            continue
            
        if user_input.startswith("/topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"   Top-K set to {top_k}")
            except: print("   Invalid Top-K")
            continue
            
        if user_input.startswith("/tmpl "):
            # Allow user to set custom template like "User: {input}\nAssistant:"
            parts = user_input.split(" ", 1)
            if len(parts) > 1:
                template = parts[1].replace("\\n", "\n")
                print(f"   Template set to: {template}")
            else:
                print(f"   Current Template: {template}")
            continue
            
        # Generating
        if user_input.startswith("/raw "):
            prompt = user_input[5:]
        else:
            prompt = template.replace("{input}", user_input)
            
        model.generate(prompt, max_new=300, temperature=temp, top_k=top_k)

if __name__ == "__main__":
    # 1. Load Genome (FAST)
    print(f"🧬 Loading Genome: {GENOME_PATH}")
    # Use mmap_mode='r' to avoid loading 100GB into RAM!
    if os.path.exists(GENOME_PATH):
        try:
            genome_np = np.load(GENOME_PATH, mmap_mode='r')
            genome = torch.from_numpy(genome_np) # Zero-copy wrapper
        except Exception as e:
            print(f"❌ Error loading genome: {e}")
            return
    else:
        print(f"❌ GENOME NOT FOUND AT: {GENOME_PATH}")
        return
    del genome_np; gc.collect()
    print(f"   Shape: {genome.shape} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # 2. Build Virtual Model (INSTANT)
    model = ZetaGrid50B_Fast(genome).to(DEVICE)
    print("⚡ Model Built.")
    
    # 3. Load Checkpoint
    load_checkpoint_into_fast_model(model, CKPT_PATH)
    model.eval()
    
    # 4. Chat
    chat(model)
