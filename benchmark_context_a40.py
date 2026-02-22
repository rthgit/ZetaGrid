import torch
import torch.nn as nn
import torch.nn.functional as F
import time, gc, sys

# ==============================================================================
# CONFIG
# ==============================================================================
class BENCH_CONFIG:
    def __init__(self):
        self.n_embd = 4096      # Full Size
        self.n_layer = 12       # Standard Depth
        self.n_inner = 16384    # Full FF
        self.vocab_size = 50257
        self.n_head = 32
        self.seq_len = 1024     # Placeholder, will be overwritten

# ==============================================================================
# MODEL (ZETAGRID STACK)
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
        # Flash Attention is critical for Long Context
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
        self.wpe = nn.Embedding(32768, config.n_embd) # Large Pos Embedding support
        self.blocks = nn.ModuleList([EchoBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.size()
        if T > 32768: raise ValueError("Seq len too long for embedding")
        
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        
        # Checkpointing is usually needed for extreme lengths, but let's test RAW first
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss to stress backward graph
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
        return logits, loss

# ==============================================================================
# TEST ENGINE
# ==============================================================================
def run_context_test(seq_len):
    print(f"\n🧪 Testing Context: {seq_len} tokens")
    
    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        config = BENCH_CONFIG()
        config.seq_len = seq_len
        
        # 1. Init Model (BF16 Native)
        model = EchoTransformer(config).to('cuda', dtype=torch.bfloat16)
        
        # 2. Dummy Input
        inputs = torch.randint(0, 50257, (1, seq_len)).to('cuda')
        
        print("   Running Forward...")
        # 3. Forward (Autocast)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            _, loss = model(inputs)
            
        print("   Running Backward...")
        # 4. Backward (Where OOM usually happens)
        loss.backward()
        
        # 5. Report Memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   ✅ SUCCESS! Peak Memory: {peak_mem:.2f} GB")
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"   ❌ OOM (Out Of Memory) at {seq_len}")
        else:
            print(f"   ❌ Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    print("🦍 ZETAGRID LONG CONTEXT SCALING (A40 48GB)")
    print("Settings: BF16 Native, Flash Attention")
    
    lengths = [2048, 4096, 8192, 16384, 24000, 32768]
    
    results = {}
    
    for seq in lengths:
        success = run_context_test(seq)
        results[seq] = success
        if not success:
            print("\n⚠️ Stopping early due to failure.")
            break
            
    print("\n==========================================")
    print("📊 MAX CONTEXT REPORT")
    print("==========================================")
    for seq, ok in results.items():
        status = "✅ PASSED" if ok else "❌ FAILED"
        print(f"Length {seq:5d}: {status}")
    print("==========================================")
