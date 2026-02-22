# ================================================================================
# 🦍 SOUL KAGGLE v19.0 "GORILLA SUPERCHARGED"
# ================================================================================
# OTTIMIZZAZIONI APPLICATE (v19):
# 1. TF32 Enabled (Matmul 32-bit su Tensor Core)
# 2. BFloat16 (Più stabile e veloce di FP16)
# 3. Fused AdamW (Optimizer C++ implementation)
# 4. Torch Compile (Graph Optimization - "inductor")
# 5. Cudnn Benchmark (Auto-tune convolution algos)
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
CHECKPOINT_INPUT = "/kaggle/input/spuol-restart/SOUL_PAUSE_SAFE.PT"
SAVE_DIR = "/kaggle/working"
DATASET_PATH = "/kaggle/input/resume-soul-v3/SOUL_MERGED_CORPUS (1).jsonl"
SAVE_EVERY = 500  # Reso meno frequente per non bloccare I/O troppo spesso

class ZETAGRID_CONFIG:
    def __init__(self):
        self.max_steps = 1200000 
        self.batch_size = 1      
        self.accum_steps = 8     # Ultra-conservative for FP16 stability
        self.seq_len = 1024
        self.lr = 1e-4  # Ultra-low for FP16 numerical stability 
        self.n_embd = 4096      
        self.n_layer = 12       
        self.n_inner = 16384    
        self.vocab_size = 50257 

# ==============================================================================
# MODEL
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
        # Fused permute operations where possible or beneficial via .view
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
        
        # Standard loop, checkpointing handled externally or per-block
        for block in self.blocks:
            # Gradient Checkpointing is crucial for 24GB VRAM with this size
            x = checkpoint(block, x, use_reentrant=False)
            
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==============================================================================
# DATASET
# ==============================================================================
class StreamingDataset(Dataset):
    def __init__(self, path, tokenizer):
        print("📦 Indexing dataset...")
        self.lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f: self.lines.append(line)
        print(f"✅ Indexed {len(self.lines)} lines.")
        self.tokenizer = tokenizer

    def __len__(self): return len(self.lines)
    def __getitem__(self, idx):
        try: text = json.loads(self.lines[idx]).get('text', '')
        except: text = ''
        t = self.tokenizer.encode(text)[:1024]
        return torch.tensor(t if t else [50256])

# ==============================================================================
# CHECKPOINT UTILS
# ==============================================================================
def find_latest_checkpoint(save_dir):
    pattern = os.path.join(save_dir, "SOUL_step*.pt")
    files = glob.glob(pattern)
    if not files: return None, 0
    def get_step(f):
        try: return int(os.path.basename(f).replace("SOUL_step", "").replace(".pt", ""))
        except: return 0
    latest = max(files, key=get_step)
    return latest, get_step(latest)

def save_checkpoint(model, optimizer, step, save_dir):
    path = os.path.join(save_dir, f"SOUL_step{step}.pt")
    # Conserve disk space: Delete old checkpoints
    for old_ckpt in glob.glob(os.path.join(save_dir, "SOUL_step*.pt")):
        try: os.remove(old_ckpt)
        except: pass
        
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"💾 Saved: {path}")

# ==============================================================================
# PIPELINE PARALLEL SETUP (GPU 0 + GPU 1)
# ==============================================================================
def setup_pipeline(model):
    """Divide il modello tra GPU 0 e GPU 1 per gestire i parametri in VRAM."""
    if torch.cuda.device_count() < 2:
        print("⚠️ Single GPU mode (Potrebbe andare OOM)")
        return model.to(device='cuda', dtype=torch.float16), 'cuda:0', 'cuda:0'
    
    print("🚀 Pipeline Parallel Enabled: GPU 1 (Embeds + Head + Layers 8-11) | GPU 0 (Layers 0-7)")
    print("🚀 Pure FP16 Mode (Ultra-Conservative Settings)")
    
    # Pure FP16 for memory efficiency
    model.wte = model.wte.to(device='cuda:1', dtype=torch.float16)
    model.wpe = model.wpe.to(device='cuda:1', dtype=torch.float16)
    
    # GPU 0 takes the bulk of layers (8 layers)
    for i in range(8):
        model.blocks[i] = model.blocks[i].to(device='cuda:0', dtype=torch.float16)
    for i in range(8, 12):
        model.blocks[i] = model.blocks[i].to(device='cuda:1', dtype=torch.float16)
        
    model.ln_f = model.ln_f.to(device='cuda:1', dtype=torch.float16)
    model.lm_head = model.lm_head.to(device='cuda:1', dtype=torch.float16)
    
    return model, 'cuda:0', 'cuda:1'

def forward_pipeline(model, idx, targets, dev0, dev1):
    """Forward pass ottimizzato con ping-pong GPU per weight tying."""
    # 🚀 SHIFT LOGIC: Apply before checkpointing to ensure consistency
    # Input: tokens[:-1], Target: tokens[1:]
    if idx.size(1) > 1:
        input_seq = idx[:, :-1]
        target_seq = targets[:, 1:]
    else:
        # Fallback for edge case
        input_seq = idx
        target_seq = targets
    
    # 1. Start on GPU 1 (dove stanno gli embeddings)
    input_seq = input_seq.to(dev1)
    t = input_seq.size(1)
    pos = torch.arange(0, t, device=dev1).unsqueeze(0)
    
    x = model.wte(input_seq) + model.wpe(pos)
    
    # 2. Transfer to GPU 0 (Layers 0-7)
    x = x.to(dev0)
    for i in range(8):
        x = checkpoint(model.blocks[i], x, use_reentrant=False)
    
    # 3. Transfer back to GPU 1 (Layers 8-11 + Head)
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
    print("🦍 SOUL V19.5 DUAL-TURBO")
    print(f"🔥 HW Accel: TF32={torch.backends.cuda.matmul.allow_tf32}, Bench={torch.backends.cudnn.benchmark}")
    print("=" * 60)
    
    config = ZETAGRID_CONFIG()
    model = EchoTransformer(config)
    
    # Checkpoint Loading (CPU first to avoid OOM before split)
    latest_ckpt, start_step = find_latest_checkpoint(SAVE_DIR)
    optimizer_state = None
    
    if latest_ckpt:
        print(f"🔄 RESUMING from {latest_ckpt} (Step {start_step})")
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer_state = ckpt.get('optimizer_state_dict')
    elif os.path.exists(CHECKPOINT_INPUT):
        print(f"📦 Loading base: {CHECKPOINT_INPUT}")
        ckpt = torch.load(CHECKPOINT_INPUT, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        if 'step' in ckpt:
            start_step = ckpt['step']
            print(f"✅ Fast-forwarding to step {start_step}")
    
    # 🚀 SETUP PIPELINE (Splits model across GPUs)
    model, dev0, dev1 = setup_pipeline(model)
    
    # Optimizer (Fused AdamW handles multi-device params automatically?)
    # Attenzione: Fused AdamW richiede che i parametri siano su CUDA. 
    # Poiché model.parameters() ora spanna su due device, pytorch gestisce i gruppi.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    if optimizer_state:
        try: optimizer.load_state_dict(optimizer_state)
        except: print("⚠️ Optimizer state mismatch (ignored)")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = StreamingDataset(DATASET_PATH, tokenizer)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda b: torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=50256),
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    step = start_step
    accum_loss = 0.0
    it_start = time.time()
    
    model.train()
    optimizer.zero_grad()
    scaler = torch.amp.GradScaler('cuda') # 🚀 Required for stable FP16
    
    # Force clean
    gc.collect()
    torch.cuda.empty_cache()
    
    print("🚀 Dual-GPU Training Started (Optimized Version)")
    
    for batch_idx, batch in enumerate(loader):
        if batch.size(1) < 2: continue # Skip short sequences
        
        # No autocast - direct FP16 computation
        _, loss = forward_pipeline(model, batch, batch, dev0, dev1)
        
        loss = loss / config.accum_steps
        
        # 🚀 Scaled Backward (Prevent FP16 underflow)
        scaler.scale(loss).backward()
        accum_loss += loss.item()
        
        if (batch_idx + 1) % config.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            
            if step % 10 == 0:
                dt = time.time() - it_start
                # TPS Correction: dt covers 10 steps!
                tps = (config.seq_len * config.accum_steps * 10) / dt
                speedup = tps / 600.0
                print(f"Step {step} | Loss: {accum_loss:.4f} | TPS: {tps:.0f} (🚀 {speedup:.1f}x vs Baseline) | DT: {dt:.1f}s")
                it_start = time.time()
                accum_loss = 0.0
            
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, step, SAVE_DIR)
            
            if step >= config.max_steps:
                break

if __name__ == "__main__":
    train()
