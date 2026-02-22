import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# DEBUG: 50B SFT LOCAL GENOME CHECK
# We suspect the Genome on RunPod != Genome Local.

BASE_DIR = "C:/Users/PC/Desktop/cpu-da"
GENOME_PATH = "C:/Users/PC/Desktop/Glifosv2/kam_llm_3b/zetagrid_25b_production.npy"
CKPT_PATH = f"{BASE_DIR}/zeta50b_sft_step2000.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

VOCAB_SIZE = 256
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 64
KERNEL_SIZE = 3
LORA_RANK = 128
DILATION_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128]

class SimulatedBank:
    def __init__(self, total_size):
        self.total_size = total_size
        self.offset = 0
    def get_offset(self, size):
        if self.offset + size > self.total_size: self.offset = 0
        start = self.offset
        end = start + size
        self.offset += size
        return (start, end)

class VirtualTCNLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, genome_tensor, offsets):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.d_model = d_model
        self.d_ff = d_ff
        self.kernel_size = kernel_size
        self.norm = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-6
        self.lora_in = nn.Module()
        self.lora_in.A = nn.Parameter(torch.zeros(LORA_RANK, d_model))
        self.lora_in.B = nn.Parameter(torch.zeros(2*d_ff, LORA_RANK))
        self.lora_out = nn.Module()
        self.lora_out.A = nn.Parameter(torch.zeros(LORA_RANK, d_ff))
        self.lora_out.B = nn.Parameter(torch.zeros(d_model, LORA_RANK))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.genome = genome_tensor
        self.idx_in = offsets['w_in']
        self.idx_dw = offsets['w_dw']
        self.idx_out = offsets['w_out']
        self.scale_in = 1.0 / math.sqrt(d_model * 0.1)
        self.scale_dw = 1.0 / math.sqrt(kernel_size)
        self.scale_out = 1.0 / math.sqrt(d_ff * 0.1)

    def forward(self, x):
        res = x
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = (x.float() * rms).to(DTYPE) * self.norm.to(DTYPE)
        
        # Virtual Weights
        w_in_flat = self.genome[self.idx_in[0] : self.idx_in[1]]
        w_in = w_in_flat.view(2*self.d_ff, self.d_model)
        ag_base = F.linear(x_norm, w_in) * self.scale_in
        
        lora_in_A = self.lora_in.A.to(DTYPE)
        lora_in_B = self.lora_in.B.to(DTYPE)
        lora_in_out = (x_norm @ lora_in_A.T) @ lora_in_B.T
        
        ag = ag_base + lora_in_out
        a, g = ag.chunk(2, dim=-1)
        
        w_dw_flat = self.genome[self.idx_dw[0] : self.idx_dw[1]]
        w_dw = w_dw_flat.view(self.d_ff, 1, self.kernel_size)
        
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, w_dw, groups=self.d_ff, dilation=self.dilation) * self.scale_dw
        a = a.transpose(1, 2)
        
        y = F.silu(a) * torch.sigmoid(g)
        
        w_out_flat = self.genome[self.idx_out[0] : self.idx_out[1]]
        w_out = w_out_flat.view(self.d_model, self.d_ff)
        out_base = F.linear(y, w_out) * self.scale_out
        
        lora_out_A = self.lora_out.A.to(DTYPE)
        lora_out_B = self.lora_out.B.to(DTYPE)
        lora_out_out = (y @ lora_out_A.T) @ lora_out_B.T
        
        return res + (out_base + lora_out_out) * self.scale.to(DTYPE)

class ZetaGrid50B_Fast(nn.Module):
    def __init__(self, genome_tensor):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(2048, D_MODEL)
        self.layers = nn.ModuleList()
        self.norm_f = nn.Parameter(torch.ones(D_MODEL))
        self.eps = 1e-6
        
        print("🧮 Simulating Allocation Layout...")
        sim_bank = SimulatedBank(len(genome_tensor))
        n_w_in = 2 * D_FF * D_MODEL
        n_w_dw = D_FF * KERNEL_SIZE
        n_w_out = D_MODEL * D_FF
        
        print("⚡ Building 64 Virtual Layers...")
        for i in range(N_LAYERS):
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
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x.float() * rms).to(DTYPE) * self.norm_f.to(DTYPE)
        logits = F.linear(x, self.emb.weight.to(DTYPE))
        return logits

def load_ckpt(model, path):
    print(f"📥 Loading Checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    new_state = {}
    for k, v in state.items():
        name = k.replace('base.', '')
        if 'norm.w' in name: name = name.replace('norm.w', 'norm')
        if 'norm_f.w' in name: name = name.replace('norm_f.w', 'norm_f')
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)
    print("✅ Weights Loaded.")

def test_raw_completion(model, prompt):
    print(f"\n🧪 TEST: RAW COMPLETION ('{prompt}')")
    model.eval()
    idx = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(idx)
        # Take the LAST logits (to predict next token)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 10)
        
        print("   Top 10 Predictions:")
        for i in range(10):
            token_idx = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            try:
                char = bytes([token_idx]).decode('utf-8', errors='replace')
                # Represent unprintable chars
                if not char.isprintable(): char = f"Maybe({token_idx})"
            except: char = "?"
            print(f"   #{i+1}: '{char}' (Prob: {prob:.4f})")

def main():
    if not os.path.exists(GENOME_PATH):
        print(f"❌ GENOME NOT FOUND: {GENOME_PATH}")
        return
        
    print(f"🧬 Loading Genome: {GENOME_PATH}")
    genome_np = np.load(GENOME_PATH)
    genome = torch.from_numpy(genome_np).to(DTYPE).to(DEVICE)
    
    # Check Genome MD5/Sum or Sample?
    # Simple check: print mean/std
    print(f"   Genome Mean: {genome.float().mean():.4f}, Std: {genome.float().std():.4f}")
    
    model = ZetaGrid50B_Fast(genome).to(DEVICE)
    load_ckpt(model, CKPT_PATH)
    
    # Test 1: "The capital of France is" -> Expect " Paris" or " P"
    test_raw_completion(model, "The capital of France is")
    
    # Test 2: SFT specific prompt
    test_raw_completion(model, "User: What is the capital of France?\nAssistant:")

if __name__ == "__main__":
    main()
