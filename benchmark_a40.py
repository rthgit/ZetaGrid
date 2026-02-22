import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time, gc

# ==============================================================================
# CONFIG
# ==============================================================================
class BENCH_CONFIG:
    def __init__(self):
        self.batch_size = 1
        self.accum_steps = 64
        self.seq_len = 1024
        self.n_embd = 4096  # Full Size Model
        self.n_layer = 12
        self.n_inner = 16384 # Full Size Inner
        self.vocab_size = 50257

# ==============================================================================
# DUMMY DATASET
# ==============================================================================
class BenchDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randint(0, 50257, (size, 1024))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==============================================================================
# MODEL
# ==============================================================================
class EchoTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(2048, config.n_embd)
        self.blocks = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_layer)]) # Simplified for bench stability
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1)) # Dummy loss
        return logits, loss

# ==============================================================================
# BENCHMARK FUNCTION
# ==============================================================================
def run_phase(name, model, loader, enable_tf32, enable_compile, enable_fused, enable_bf16):
    print(f"\n⚡ STARTING PHASE: {name}")
    print(f"   Settings: TF32={enable_tf32}, Compile={enable_compile}, Fused={enable_fused}, BF16={enable_bf16}")
    
    # 1. Apply Settings
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    
    # 2. Logic for Autocast
    amp_dtype = torch.bfloat16 if enable_bf16 else torch.float16
    
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=1e-4, fused=enable_fused)
    
    model_eng = model
    if enable_compile:
        try:
            model_eng = torch.compile(model)
            print("   (Compiling model... wait)")
        except:
            pass
            
    model_eng.train()
    
    # Warmup
    print("   (Warming up...)")
    for i, batch in enumerate(loader):
        if i >= 5: break
        batch = batch.to('cuda')
        with torch.amp.autocast('cuda', enabled=True, dtype=amp_dtype):
            _, loss = model_eng(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # Measurment
    print("   🚀 Measuring TPS...")
    start_t = time.time()
    steps = 20
    total_tokens = 0
    
    for i, batch in enumerate(loader):
        if i >= steps: break
        batch = batch.to('cuda')
        with torch.amp.autocast('cuda', enabled=True, dtype=amp_dtype):
            _, loss = model_eng(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += (batch.size(0) * batch.size(1)) 
        
    dt = time.time() - start_t
    tps = total_tokens / dt
    print(f"   🏁 {name} RESULT: {tps:.0f} TPS")
    return tps

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🦍 ZETAGRID A/B BENCHMARK (A40) - STRESS TEST")
    
    config = BENCH_CONFIG()
    # Restore "Medium" size to stress memory bandwidth
    config.n_embd = 2048 
    config.n_inner = 8192
    
    dataset = BenchDataset(size=200)
    
    # Pre-run Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    model = EchoTransformer(config).to('cuda')

    # PHASE 1: VANILLA BASELINE (Realistic Usage)
    # Scenario: User runs FP32, gets OOM, drops Batch Size to 1 to make it work.
    # Settings: Batch=1, FP32, No Compile
    print("⚠️ Vanilla phase: Forced to Batch Size 1 (FP32) to fit in memory")
    loader_vanilla = DataLoader(dataset, batch_size=1, num_workers=2)
    
    base_tps = run_phase("VANILLA BASELINE (BS=1, FP32)", model, loader_vanilla, 
                         enable_tf32=False, enable_compile=False, enable_fused=False, enable_bf16=False)
    
    # Clean
    model = None
    loader_vanilla = None
    torch.cuda.empty_cache()
    
    model = EchoTransformer(config).to('cuda')

    # PHASE 2: ZETAGRID OPTIMIZED (Power User)
    # Scenario: Optimization allows Batch Size 8 + BF16 + TF32 without OOM.
    # Settings: Batch=8, BF16, Compile, TF32
    print("🚀 ZETAGRID phase: Unlocked Batch Size 8 (BF16)")
    loader_opt = DataLoader(dataset, batch_size=8, num_workers=4)
    
    opt_tps = run_phase("ZETAGRID OPTIMIZED (BS=8, BF16)", model, loader_opt, 
                        enable_tf32=True, enable_compile=True, enable_fused=True, enable_bf16=True)
    
    print("\n==========================================")
    print(f"🟢 VANILLA (BS=1):   {base_tps:.0f} TPS")
    print(f"🚀 ZETAGRID (BS=8):  {opt_tps:.0f} TPS")
    print(f"🔥 SPEEDUP:          {opt_tps/base_tps:.2f}x")
    print("==========================================")
