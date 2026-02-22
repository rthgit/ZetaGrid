# ================================================================================
# 🦍 SOUL KAGGLE v18.0 "GORILLA PERSISTENT"
# ================================================================================
# FATTO PER SOPRAVVIVERE AL TIMEOUT DI 8 ORE DI KAGGLE.
# → Salva su Google Drive ogni 100 step.
# → Auto-resume dal checkpoint più recente.
# → Pipeline Parallel stabile (già testato: 26s/step, 1262 TPS).
# ================================================================================
# ISTRUZIONI:
# 1. Monta Google Drive su Kaggle (vedi cella sotto).
# 2. Copia questo script in una cella.
# 3. Lancia.
# 4. Quando Kaggle resetta, ri-lancia lo STESSO script: riprenderà da dove era.
# ================================================================================

# --- CELLA 0: MONTA GOOGLE DRIVE (ESEGUI PRIMA DI TUTTO) ---
# from google.colab import drive  # Su Colab
# drive.mount('/content/drive')
# SAVE_DIR = "/content/drive/MyDrive/SOUL_CHECKPOINTS"

# Su Kaggle, usa Kaggle Secrets per leggere un path o salva su /kaggle/working 
# che persiste fino alla fine della sessione (ma non tra sessioni).
# Per persistenza VERA, usa Kaggle Datasets output.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import json, os, time, gc, glob
from transformers import GPT2Tokenizer

# ==============================================================================
# CONFIG
# ==============================================================================
CHECKPOINT_INPUT = "/kaggle/input/spuol-restart/SOUL_KAGGLE_step270000.pt"
SAVE_DIR = "/kaggle/working"  # Persiste durante la sessione
DATASET_PATH = "/kaggle/input/resume/SOUL_MERGED_CORPUS.jsonl"
SAVE_EVERY = 100  # ⚡ Salva ogni 100 step per minimizzare perdite

class ZETAGRID_CONFIG:
    def __init__(self):
        self.max_steps = 1200000 
        self.batch_size = 1      
        self.accum_steps = 32    
        self.seq_len = 1024
        self.lr = 6e-4 
        self.n_embd = 4096      
        self.n_layer = 12       
        self.n_inner = 16384    
        self.vocab_size = 50257 

# ==============================================================================
# MODEL (Identico a GORILLA V15 - Stabile)
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
        t = idx.size(1)
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ==============================================================================
# DATASET (Streaming - Memoria Efficiente)
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
    """Trova il checkpoint più recente salvato."""
    pattern = os.path.join(save_dir, "SOUL_step*.pt")
    files = glob.glob(pattern)
    if not files:
        return None, 0
    # Estrai step number dal nome file
    def get_step(f):
        try: return int(os.path.basename(f).replace("SOUL_step", "").replace(".pt", ""))
        except: return 0
    latest = max(files, key=get_step)
    return latest, get_step(latest)

def save_checkpoint(model, optimizer, step, save_dir):
    """Salva checkpoint con step nel nome."""
    path = os.path.join(save_dir, f"SOUL_step{step}.pt")
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
    """Divide il modello tra GPU 0 e GPU 1."""
    if torch.cuda.device_count() < 2:
        print("⚠️ Single GPU mode")
        return model.cuda().half(), 'cuda:0', 'cuda:0'
    
    print("🚀 Pipeline Parallel: GPU 0 (layers 0-5) + GPU 1 (layers 6-11 + head)")
    model.wte = model.wte.to('cuda:0')
    model.wpe = model.wpe.to('cuda:0')
    for i in range(6):
        model.blocks[i] = model.blocks[i].to('cuda:0')
    for i in range(6, 12):
        model.blocks[i] = model.blocks[i].to('cuda:1')
    model.ln_f = model.ln_f.to('cuda:1')
    model.lm_head = model.lm_head.to('cuda:1')
    return model.half(), 'cuda:0', 'cuda:1'

def forward_pipeline(model, idx, targets, dev0, dev1):
    """Forward pass attraverso la pipeline."""
    idx = idx.to(dev0)
    t = idx.size(1)
    pos = torch.arange(0, t, device=dev0).unsqueeze(0)
    x = model.wte(idx) + model.wpe(pos)
    
    # GPU 0: Layers 0-5
    for i in range(6):
        x = checkpoint(model.blocks[i], x, use_reentrant=False)
    
    # Transfer to GPU 1
    x = x.to(dev1)
    
    # GPU 1: Layers 6-11 + Head
    for i in range(6, 12):
        x = checkpoint(model.blocks[i], x, use_reentrant=False)
    
    logits = model.lm_head(model.ln_f(x))
    
    if targets is not None:
        targets = targets.to(dev1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    return logits, None

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def train():
    print("🦍 SOUL V18 GORILLA PERSISTENT")
    print("=" * 60)
    
    config = ZETAGRID_CONFIG()
    model = EchoTransformer(config)
    
    # Cerca checkpoint esistente
    latest_ckpt, start_step = find_latest_checkpoint(SAVE_DIR)
    
    if latest_ckpt:
        print(f"🔄 RESUMING from {latest_ckpt} (Step {start_step})")
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif os.path.exists(CHECKPOINT_INPUT):
        print(f"📦 Loading initial checkpoint: {CHECKPOINT_INPUT}")
        model.load_state_dict(torch.load(CHECKPOINT_INPUT, map_location='cpu'), strict=False)
        start_step = 270000
    else:
        print("⚠️ No checkpoint found, starting from scratch")
        start_step = 0
    
    # Setup pipeline
    model, dev0, dev1 = setup_pipeline(model)
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    if latest_ckpt and 'optimizer_state_dict' in ckpt:
        try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except: pass
    
    # Dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = StreamingDataset(DATASET_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda b: torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=50256),
                        num_workers=2, pin_memory=True)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"🚀 Starting from Step {start_step}")
    print(f"💾 Saving every {SAVE_EVERY} steps to {SAVE_DIR}")
    print("=" * 60)
    
    step = start_step
    accum_loss = 0.0
    it_start = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        with torch.cuda.amp.autocast():
            _, loss = forward_pipeline(model, batch, batch, dev0, dev1)
        
        loss = loss / config.accum_steps
        loss.backward()
        accum_loss += loss.item()
        
        if (batch_idx + 1) % config.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            dt = time.time() - it_start
            tps = (config.seq_len * config.accum_steps) / dt
            it_start = time.time()
            
            print(f"Step {step} | Loss: {accum_loss:.4f} | TPS: {tps:.0f} | DT: {dt:.1f}s")
            accum_loss = 0.0
            
            # SALVATAGGIO FREQUENTE
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, step, SAVE_DIR)
            
            if step >= config.max_steps:
                print("✅ TRAINING COMPLETE!")
                save_checkpoint(model, optimizer, step, SAVE_DIR)
                break

if __name__ == "__main__":
    train()
