# ================================================================================
# 🦍 SOUL KAGGLE ALIGNMENT v1.0 (SlimOrca Fine-Tuning)
# ================================================================================
# OBJECTIVE: Align "Soul" model to follow instructions using SlimOrca dataset.
# ================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import json, os, time, gc, glob
from transformers import GPT2Tokenizer

# --- HARDWARE ACCELERATION FLAGS ---
torch.backends.cuda.matmul.allow_tf32 = True  # 🚀 TF32 for matrix multiplications
torch.backends.cudnn.allow_tf32 = True        # 🚀 TF32 for convolutions
torch.backends.cudnn.benchmark = True         # 🚀 Auto-tuner

# ==============================================================================
# CONFIG
# ==============================================================================
# KAGGLE PATHS
CHECKPOINT_INPUT = "/kaggle/input/spuol-restart/SOUL_PAUSE_SAFE.PT" 
# Alternative: SOUL_GENERALIST_FINAL.pt if available
SAVE_DIR = "/kaggle/working"
DATASET_PATH = "/kaggle/input/slimorca-deduped/slimorca_repair_100k.jsonl"
SAVE_EVERY = 250  # More frequent saves for fine-tuning

class ZETAGRID_CONFIG:
    def __init__(self):
        self.max_steps = 5000    # Reduced for Alignment (Instruction Tuning is fast)
        self.batch_size = 1      
        self.accum_steps = 16    # High gradient accumulation for stability
        self.seq_len = 1024
        self.lr = 5e-6           # 📉 ULTRA-LOW LR for Fine-Tuning (Prevent Catastrophic Forgetting)
        self.n_embd = 4096      
        self.n_layer = 12       
        self.n_inner = 16384    
        self.vocab_size = 50257 

# ==============================================================================
# MODEL (Same Architecture as Pre-training)
# ==============================================================================
class EchoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = 32
        self.head_dim = config.n_embd // self.n_head
        self.c_qk = nn.Linear(config.n_embd, config.n_embd) 
        self.c_v = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qk = self.c_qk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # 🚀 SDPA: Using PyTorch Native Flash Attention
        y = F.scaled_dot_product_attention(qk, qk, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MirrorFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.act = nn.GELU()
        self.bias_out = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x):
        h = self.act(self.c_fc(x))
        # Mirror Weight Tying
        return (F.linear(h, self.c_fc.weight.t()) * 0.9) + self.bias_out

class EchoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = EchoAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MirrorFFN(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class EchoTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(2048, config.n_embd)
        self.blocks = nn.ModuleList([EchoBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        t = idx.size(1)
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
            
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==============================================================================
# DATASET
# ==============================================================================
class AlignmentDataset(Dataset):
    def __init__(self, path, tokenizer):
        print("📦 Indexing SlimOrca dataset...")
        self.lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f: 
                try:
                    # Dataset 'text' field already formatted as instruction/response
                    data = json.loads(line)
                    if 'text' in data and len(data['text']) > 10:
                        self.lines.append(data['text'])
                except: pass
        print(f"✅ Indexed {len(self.lines)} instructions.")
        self.tokenizer = tokenizer

    def __len__(self): return len(self.lines)
    def __getitem__(self, idx):
        text = self.lines[idx]
        t = self.tokenizer.encode(text)
        # Dynamic truncation/padding happens in collate
        t = t[:1024] # Hard clip for safety
        return torch.tensor(t if t else [50256])

# ==============================================================================
# CHECKPOINT UTILS
# ==============================================================================
def find_latest_checkpoint(save_dir):
    pattern = os.path.join(save_dir, "SOUL_ALIGN_step*.pt")
    files = glob.glob(pattern)
    if not files: return None, 0
    def get_step(f):
        try: return int(os.path.basename(f).replace("SOUL_ALIGN_step", "").replace(".pt", ""))
        except: return 0
    latest = max(files, key=get_step)
    return latest, get_step(latest)

def save_checkpoint(model, optimizer, step, save_dir):
    path = os.path.join(save_dir, f"SOUL_ALIGN_step{step}.pt")
    for old_ckpt in glob.glob(os.path.join(save_dir, "SOUL_ALIGN_step*.pt")):
        try: os.remove(old_ckpt)
        except: pass
        
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"💾 Saved Alignment Checkpoint: {path}")

# ==============================================================================
# PIPELINE PARALLEL SETUP (GPU 0 + GPU 1)
# ==============================================================================
def setup_pipeline(model):
    if torch.cuda.device_count() < 2:
        print("⚠️ Single GPU mode (Potentially OOM)")
        return model.to(device='cuda', dtype=torch.float16), 'cuda:0', 'cuda:0'
    
    print("🚀 Pipeline Parallel Enabled: GPU 1 (Embeds + Head + Layers 8-11) | GPU 0 (Layers 0-7)")
    model.wte = model.wte.to(device='cuda:1', dtype=torch.float16)
    model.wpe = model.wpe.to(device='cuda:1', dtype=torch.float16)
    for i in range(8):
        model.blocks[i] = model.blocks[i].to(device='cuda:0', dtype=torch.float16)
    for i in range(8, 12):
        model.blocks[i] = model.blocks[i].to(device='cuda:1', dtype=torch.float16)
    model.ln_f = model.ln_f.to(device='cuda:1', dtype=torch.float16)
    model.lm_head = model.lm_head.to(device='cuda:1', dtype=torch.float16)
    return model, 'cuda:0', 'cuda:1'

def forward_pipeline(model, idx, targets, dev0, dev1):
    # Shift logic for Next Token Prediction
    if idx.size(1) > 1:
        input_seq = idx[:, :-1]
        target_seq = targets[:, 1:]
    else: return None, None # Skip

    # 1. Start on GPU 1
    input_seq = input_seq.to(dev1)
    t = input_seq.size(1)
    pos = torch.arange(0, t, device=dev1).unsqueeze(0)
    x = model.wte(input_seq) + model.wpe(pos)
    
    # 2. Transfer to GPU 0
    x = x.to(dev0)
    for i in range(8):
        x = checkpoint(model.blocks[i], x, use_reentrant=False)
    
    # 3. Transfer back to GPU 1
    x = x.to(dev1)
    for i in range(8, 12):
        x = checkpoint(model.blocks[i], x, use_reentrant=False)
    
    logits = model.lm_head(model.ln_f(x))
    
    loss = None
    if target_seq is not None:
        target_seq = target_seq.to(dev1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        
    return logits, loss

# ==============================================================================
# TRAINING
# ==============================================================================
def train():
    print("🦍 SOUL ALIGNMENT (Phase 14)")
    print(f"🔥 HW Accel: TF32={torch.backends.cuda.matmul.allow_tf32}")
    
    config = ZETAGRID_CONFIG()
    model = EchoTransformer(config)
    
    # Checkpoint Loading logic
    latest_ckpt, start_step = find_latest_checkpoint(SAVE_DIR)
    
    if latest_ckpt:
        print(f"🔄 RESUMING ALIGNMENT from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif os.path.exists(CHECKPOINT_INPUT):
        print(f"📦 Loading BASE MODEL for Fine-Tuning: {CHECKPOINT_INPUT}")
        ckpt = torch.load(CHECKPOINT_INPUT, map_location='cpu')
        model.load_state_dict(ckpt, strict=False) # Load weights from pre-training
    else:
        print(f"❌ Base Checkpoint not found at {CHECKPOINT_INPUT}. Cannot fine-tune.")
        return

    model, dev0, dev1 = setup_pipeline(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = AlignmentDataset(DATASET_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda b: torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=50256),
                        num_workers=4, pin_memory=True)
    
    step = 0 # Reset steps for alignment phase (or continue if you prefer)
    accum_loss = 0.0
    it_start = time.time()
    scaler = torch.amp.GradScaler('cuda') 
    
    model.train()
    optimizer.zero_grad()
    
    print("🚀 Alignment Started...")
    
    for batch_idx, batch in enumerate(loader):
        if batch.size(1) < 2: continue
        
        _, loss = forward_pipeline(model, batch, batch, dev0, dev1)
        
        loss = loss / config.accum_steps
        scaler.scale(loss).backward()
        accum_loss += loss.item()
        
        if (batch_idx + 1) % config.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            
            if step % 5 == 0: # Print more often for fine-tuning
                print(f"Step {step} | Loss: {accum_loss:.4f}")
                accum_loss = 0.0
            
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, step, SAVE_DIR)
            
            if step >= config.max_steps:
                save_checkpoint(model, optimizer, step, SAVE_DIR)
                print("🏁 Alignment Complete!")
                break

if __name__ == "__main__":
    train()
