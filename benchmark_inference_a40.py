import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc

# ==============================================================================
# CONFIG
# ==============================================================================
class BENCH_CONFIG:
    def __init__(self):
        self.n_embd = 2048  # Match the Training Benchmark "Medium" size
        self.n_layer = 12
        self.n_inner = 8192
        self.vocab_size = 50257
        self.n_head = 32 # Required for attention

# ==============================================================================
# MODEL (ZETAGRID ARCHITECTURE)
# ==============================================================================
class EchoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // self.n_head
        self.c_qk = nn.Linear(config.n_embd, config.n_embd)
        self.c_v = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qk = self.c_qk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash Attention is automatic in PyTorch 2.0+ via this function
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
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(2048, config.n_embd)
        self.blocks = nn.ModuleList([EchoBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================
def benchmark_prefill(name, model, input_ids_long, enable_compile, enable_bf16):
    print(f"\n⚡ {name}")
    print(f"   Settings: BF16={enable_bf16}, Compile={enable_compile}")
    
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True 
    dtype = torch.bfloat16 if enable_bf16 else torch.float16
    
    model.eval()
    if enable_bf16: model.to(dtype=torch.bfloat16)
    else: model.to(dtype=torch.float16)
    
    if enable_compile:
        try:
            print("   (Compiling...)")
            model = torch.compile(model, mode='max-autotune') # Max optimization for static shapes
        except Exception as e:
            print(f"Compile Warning: {e}")
            
    # Warmup
    print("   (Warming up...)")
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
             _ = model(input_ids_long)

    # BENCHMARK PREFILL (Compute Bound)
    print("   🚀 Measuring Prompt Processing Speed...")
    start_t = time.time()
    iters = 50
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
            for _ in range(iters):
                _ = model(input_ids_long) # Just process the context
                
    torch.cuda.synchronize() # Ensure timing is accurate
    dt = time.time() - start_t
    
    total_tokens = input_ids_long.numel() * iters
    tps = total_tokens / dt
    print(f"   🏁 RESULT: {tps:.2f} Tokens/Sec (Prefill)")
    return tps

if __name__ == "__main__":
    print("🦍 ZETAGRID INFERENCE BENCHMARK (A40) - CONTEXT PREFILL")
    
    config = BENCH_CONFIG()
    model = EchoTransformer(config).to('cuda')
    
    # Simulate processing a Long Document (e.g. RAG Context)
    # Batch 16 * 1024 Tokens = 16k tokens per forward pass
    BATCH_SIZE = 16
    SEQ_LEN = 1024
    input_ids = torch.randint(0, 50257, (BATCH_SIZE, SEQ_LEN)).to('cuda')
    
    # 1. VANILLA (FP16, Eager)
    tps_vanilla = benchmark_prefill("VANILLA (FP16, Eager)", model, input_ids, 
                                  enable_compile=False, enable_bf16=False)
    
    # Reset
    model = None
    torch.cuda.empty_cache()
    model = EchoTransformer(config).to('cuda')
    
    # 2. ZETAGRID (BF16, Compile)
    tps_opt = benchmark_prefill("ZETAGRID (BF16, Compile)", model, input_ids, 
                              enable_compile=True, enable_bf16=True)
                                 
    print("\n==========================================")
    print(f"🟢 VANILLA PREFILL:   {tps_vanilla:.2f} T/s")
    print(f"🚀 ZETAGRID PREFILL:  {tps_opt:.2f} T/s")
    print(f"🔥 SPEEDUP:           {tps_opt/tps_vanilla:.2f}x")
    print("==========================================")
