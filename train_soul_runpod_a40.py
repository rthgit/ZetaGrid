# ================================================================================
# 🦍 SOUL RUNPOD A40 "UNLEASHED"
# ================================================================================
# OPTIMIZATIONS FOR A40 (48GB VRAM):
# 1. Single GPU (no pipeline parallelism needed)
# 2. BFloat16 Native (Ampere architecture)
# 3. TF32 Enabled
# 4. Fused AdamW
# 5. Torch Compile
# 6. Larger Batch Size (accum_steps=64)
# ================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import json, os, time, gc, glob
from transformers import GPT2Tokenizer

# --- HARDWARE ACCELERATION FLAGS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ==============================================================================
# CONFIG
# ==============================================================================
CHECKPOINT_INPUT = "/workspace/SOUL_KAGGLE_step270000.pt"  # Update this path
SAVE_DIR = "/workspace/checkpoints"
DATASET_PATH = "/workspace/SOUL_MERGED_CORPUS.jsonl"  # Update this path
SAVE_EVERY = 500

class ZETAGRID_CONFIG:
    def __init__(self):
        self.max_steps = 1200000
        self.batch_size = 1
        self.accum_steps = 64  # Large batch for A40
        self.seq_len = 1024
        self.lr = 5e-5  # Low LR for resuming training
        self.n_embd = 4096
        self.n_layer = 12
        self.n_inner = 16384
        self.vocab_size = 50257

# ==============================================================================
# MODEL (Same architecture as before)
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
        # Shift for next-token prediction
        if idx.size(1) > 1 and targets is not None:
            input_seq = idx[:, :-1]
            target_seq = targets[:, 1:]
        else:
            input_seq = idx
            target_seq = targets
        
        t = input_seq.size(1)
        pos = torch.arange(0, t, device=input_seq.device).unsqueeze(0)
        x = self.wte(input_seq) + self.wpe(pos)
        
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if target_seq is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        
        return logits, loss

# ==============================================================================
# DATASET
# ==============================================================================
class StreamingDataset(Dataset):
    def __init__(self, path, tokenizer):
        print("📦 Indexing dataset...")
        self.lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f: 
                self.lines.append(line)
        print(f"✅ Indexed {len(self.lines)} lines.")
        self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.lines)
    
    def __getitem__(self, idx):
        try: 
            text = json.loads(self.lines[idx]).get('text', '')
        except: 
            text = ''
        t = self.tokenizer.encode(text)[:1024]
        return torch.tensor(t if t else [50256])

# ==============================================================================
# CHECKPOINT UTILS
# ==============================================================================
def find_latest_checkpoint(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pattern = os.path.join(save_dir, "SOUL_step*.pt")
    files = glob.glob(pattern)
    if not files: 
        return None, 0
    
    def get_step(f):
        try: 
            return int(os.path.basename(f).replace("SOUL_step", "").replace(".pt", ""))
        except: 
            return 0
    
    latest = max(files, key=get_step)
    return latest, get_step(latest)

def save_checkpoint(model, optimizer, step, save_dir):
    # DISK SAFE MODE: Delete ALL previous checkpoints before saving to prevent OOM
    # A40 Pod has only 20GB. Checkpoint is ~9GB. We cannot fit two.
    
    # 1. Aggressive Cleanup
    files = glob.glob(os.path.join(save_dir, "SOUL_step*.pt"))
    for f in files:
        try:
            print(f"🗑️ Deleting old checkpoint to free space: {f}")
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Failed to remove {f}: {e}")
            
    # 2. Save New
    path = os.path.join(save_dir, f"SOUL_step{step}.pt")
    print(f"💾 Saving new checkpoint: {path} ...")
    
    try:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"✅ Saved successfully.")
    except Exception as e:
        print(f"❌ SAVE FAILED (Disk Full?): {e}")

# ==============================================================================
# TRAINING
# ==============================================================================
def train():
    print("🦍 SOUL A40 UNLEASHED")
    print(f"🔥 HW: TF32={torch.backends.cuda.matmul.allow_tf32}, BF16=Native, Bench={torch.backends.cudnn.benchmark}")
    print("=" * 60)
    
    config = ZETAGRID_CONFIG()
    model = EchoTransformer(config)
    
    # 🚀 BFLOAT16: Native on A40 (Ampere)
    model = model.to(dtype=torch.bfloat16, device='cuda')
    
    # 🚀 TORCH COMPILE
    try:
        print("⚡ Compiling model...")
        model = torch.compile(model, mode='max-autotune')
        print("⚡ Compilation complete.")
    except Exception as e:
        print(f"⚠️ Compile skipped: {e}")
    
    # Checkpoint Loading
    latest_ckpt, start_step = find_latest_checkpoint(SAVE_DIR)
    optimizer_state = None
    
    if latest_ckpt:
        print(f"🔄 RESUMING from {latest_ckpt} (Step {start_step})")
        ckpt = torch.load(latest_ckpt, map_location='cuda')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer_state = ckpt.get('optimizer_state_dict')
    elif os.path.exists(CHECKPOINT_INPUT):
        print(f"📦 Loading base: {CHECKPOINT_INPUT}")
        try:
            ckpt = torch.load(CHECKPOINT_INPUT, map_location='cuda', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    if 'step' in ckpt:
                        start_step = ckpt['step']
                        print(f"✅ Fast-forwarding to step {start_step}")
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'], strict=False)
                else:
                    # Assume it's a direct state_dict
                    model.load_state_dict(ckpt, strict=False)
            else:
                # Direct state_dict (not wrapped in dict)
                model.load_state_dict(ckpt, strict=False)
            
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("⚠️ Starting from scratch")
            start_step = 0
    
    # 🚀 FUSED ADAMW
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = StreamingDataset(DATASET_PATH, tokenizer)
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True,
        collate_fn=lambda b: torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=50256),
        num_workers=8,  # More workers for A40
        pin_memory=True,
        persistent_workers=True
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    step = start_step
    accum_loss = 0.0
    it_start = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    print("🚀 Training Started")
    
    for batch_idx, batch in enumerate(loader):
        if batch.size(1) < 2: 
            continue
        
        batch = batch.to('cuda', non_blocking=True)
        
        # 🚀 Autocast BFloat16
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(batch, batch)
        
        loss = loss / config.accum_steps
        loss.backward()
        accum_loss += loss.item()
        
        if (batch_idx + 1) % config.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            if step % 10 == 0:
                dt = time.time() - it_start
                avg_loss = accum_loss / 10.0
                tps = (config.seq_len * config.accum_steps * 10) / dt
                speedup = tps / 600.0
                print(f"Step {step} | Loss: {avg_loss:.4f} | TPS: {tps:.0f} (🚀 {speedup:.1f}x) | DT: {dt:.1f}s")
                it_start = time.time()
                accum_loss = 0.0
            
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, step, SAVE_DIR)
            
            if step >= config.max_steps:
                break

if __name__ == "__main__":
    train()
