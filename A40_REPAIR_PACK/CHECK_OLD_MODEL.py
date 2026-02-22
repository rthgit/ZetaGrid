import torch
import os
import sys

# CONFIG A40
OLD_CKPT = "/workspace/zetagrid_50b/zeta25b_step15000.pt"

def analyze_v1():
    print(f"🧬 ZETAGRID V1 ANALYSIS")
    print(f"Loading Checkpoint: {OLD_CKPT}")
    
    if not os.path.exists(OLD_CKPT):
        print("❌ File not found.")
        return

    # Load Full Checkpoint
    try:
        full_ckpt = torch.load(OLD_CKPT, map_location="cpu")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # Handle State Dict
    state = full_ckpt.get('model', full_ckpt.get('model_state_dict', full_ckpt))
    
    print(f"✅ Loaded. Keys: {len(state)}")
    
    # Analyze Dimensions
    # Look for 'layers.0.w_in_lora.A' or similar
    # V1 key format? Let's print first 10 keys
    print("Sample Keys:")
    for k in list(state.keys())[:5]:
        shape = state[k].shape
        print(f"   {k}: {shape}")
        
    # Analyze LoRA Rank
    # Assuming 'layers.0.lora_in.A' -> [rank, d_model]
    rank = "?"
    d_model = "?"
    d_ff = "?"
    
    for k, v in state.items():
        if "lora_in.A" in k:
            rank = v.shape[0]
            d_model = v.shape[1]
        if "lora_in.B" in k:
            # [2*d_ff, rank]
            d_ff_x2 = v.shape[0]
            d_ff = d_ff_x2 // 2
            
    print(f"\n📊 INFERRED CONFIG (V1):")
    print(f"   LORA_RANK: {rank}")
    print(f"   D_MODEL: {d_model}")
    print(f"   D_FF: {d_ff}")

if __name__ == "__main__":
    analyze_v1()
