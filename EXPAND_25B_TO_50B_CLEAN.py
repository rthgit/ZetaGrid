import torch
import torch.nn as nn
import os
import gc
import shutil
import math

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_25B_PATH = "zeta25b_step15000.pt" 
OUTPUT_50B_PATH = "zetagrid_50b_seed_clean.pt" # Cleaner name

def expand_model_clean():
    print(f"🚀 ZETAGRID CLEAN EXPANSION: 25B → 50B (NO NOISE)")
    print(f"   Source: {MODEL_25B_PATH}")
    
    if not os.path.exists(MODEL_25B_PATH):
        print(f"❌ Error: {MODEL_25B_PATH} not found.")
        return

    print("   Loading 25B Model...")
    try:
        ckpt = torch.load(MODEL_25B_PATH, map_location="cpu")
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    print(f"   Original Parameters: {len(state_dict)} keys")
    
    new_state_dict = {}
    
    print("   Processing Layers (Exact Duplication)...")
    
    # Separate layers from others
    layer_params = {}
    other_params = {}
    
    for k, v in state_dict.items():
        if k.startswith("layers."):
            layer_params[k] = v
        else:
            other_params[k] = v
            new_state_dict[k] = v.clone()
            
    print(f"   Base Parameters (Non-Layer): {len(other_params)} keys copied.")
    
    # Expand Layers 0-31 to 0-63
    for i in range(32):
        prefix = f"layers.{i}."
        current_layer_keys = [k for k in layer_params.keys() if k.startswith(prefix)]
        
        for k in current_layer_keys:
            suffix = k[len(prefix):] # e.g. "norm.w"
            
            # --- LOWER FRACTAL (0-31) ---
            new_key_lower = k
            new_state_dict[new_key_lower] = layer_params[k].clone()
            
            # --- UPPER FRACTAL (32-63) ---
            # EXACT COPY - No Noise
            new_layer_idx = i + 32
            new_key_upper = f"layers.{new_layer_idx}.{suffix}"
            new_state_dict[new_key_upper] = layer_params[k].clone()

        if i % 4 == 0:
            print(f"   ✨ Expanded Layer {i} → {i} & {i+32} (Exact Copy)")

    total_keys = len(new_state_dict)
    print(f"   Total New Parameters: {total_keys} keys")
    
    print(f"   Saving 50B CLEAN Seed to {OUTPUT_50B_PATH}...")
    torch.save({'model': new_state_dict, 'step': 0, 'fractional_epoch': 0}, OUTPUT_50B_PATH)
    
    size_gb = os.path.getsize(OUTPUT_50B_PATH) / 1e9 if os.path.exists(OUTPUT_50B_PATH) else 0
    print(f"✅ CLEAN EXPANSION COMPLETE!")
    print(f"   New Model Size: {size_gb:.2f} GB")
    print(f"   Structure: 64 Layers (Exact Duplication)")
    print(f"   Ready for Phase 3 Training (Stable).")

if __name__ == "__main__":
    expand_model_clean()
