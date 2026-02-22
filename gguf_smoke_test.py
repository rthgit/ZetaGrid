import torch
import numpy as np
import os
from convert_rth_to_gguf import convert_rth_to_gguf

def create_dummy_weights():
    print("🛠️ Creating dummy weights for smoke test...")
    # Matches ZETAGRID_INFERENCE.py config
    D_MODEL = 4096
    D_FF = 16384
    N_LAYERS = 32
    KERNEL_SIZE = 3
    
    # 1. Genome Dummy (Total size pool)
    # We only need enough for a few layers to test logic, or full if we want a real test
    # Full genome size calculation: N_LAYERS * (2*D_FF*D_MODEL + D_FF*KERNEL_SIZE + D_MODEL*D_FF)
    # To keep test fast, we'll simulate just 1 block for the dummy run
    n = 2 * D_FF * D_MODEL + D_FF * KERNEL_SIZE + D_MODEL * D_FF
    dummy_genome = np.random.randn(n + 1000).astype(np.float32)
    np.save("dummy_genome.npy", dummy_genome)
    
    # 2. Soul Dummy (State Dict)
    state = {
        'emb.weight': torch.randn(256, D_MODEL),
        'pos_emb.weight': torch.randn(2048, D_MODEL),
        'norm_f.w': torch.randn(D_MODEL),
    }
    
    for i in range(32):
        prefix = f"layers.{i}"
        state[f"{prefix}.norm.w"] = torch.randn(D_MODEL)
        state[f"{prefix}.scale"] = torch.tensor(0.1)
        state[f"{prefix}.lora_in.A"] = torch.randn(128, D_MODEL)
        state[f"{prefix}.lora_in.B"] = torch.randn(2*D_FF, 128)
        state[f"{prefix}.lora_out.A"] = torch.randn(128, D_FF)
        state[f"{prefix}.lora_out.B"] = torch.randn(D_MODEL, 128)
        
    torch.save({'model': state}, "dummy_soul.pt")
    print("✅ Dummy weights created.")

def run_smoke_test():
    create_dummy_weights()
    print("\n📦 Starting conversion smoke test...")
    try:
        # Test with just 1 block
        convert_rth_to_gguf("dummy_genome.npy", "dummy_soul.pt", "rth_test_v1.gguf", n_layers=1)
        print("\n🏆 TEST SUCCESS: GGUF file generated without crashes.")
        if os.path.exists("rth_test_v1.gguf"):
            size = os.path.getsize("rth_test_v1.gguf") / 1e6
            print(f"📄 Produced File: rth_test_v1.gguf ({size:.2f} MB)")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")

if __name__ == "__main__":
    run_smoke_test()
