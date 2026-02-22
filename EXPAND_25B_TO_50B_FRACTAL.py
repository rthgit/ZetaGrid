import torch
import torch.nn as nn
import os
import gc
import shutil
import math

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_25B_PATH = "zeta25b_step15000.pt"  # Put your downloaded model here (or adjust path)
OUTPUT_50B_PATH = "zetagrid_50b_seed.pt"

# Expansion Config (Fractal Scaling)
# 25B was: 32 Layers, 4096 Dim, 16384 FF
# 50B will be: 64 Layers, 4096 Dim, 16384 FF
# Strategy: Stack two 25B blocks, with "Fractal Noise" on the second block
# to break symmetry for further evolution.

TARGET_LAYERS = 64  
NOISE_SCALE = 0.02

def expand_model():
    print(f"🚀 ZETAGRID FRACTAL EXPANSION: 25B → 50B")
    print(f"   Source: {MODEL_25B_PATH}")
    
    if not os.path.exists(MODEL_25B_PATH):
        print(f"❌ Error: {MODEL_25B_PATH} not found.")
        print("   Please ensure the 25B checkpoint is in this folder or update the path.")
        return

    print("   Loading 25B Model...")
    try:
        ckpt = torch.load(MODEL_25B_PATH, map_location="cpu")
        # Handle if checkpoint is wrapped in 'model' key or direct state_dict
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        # If wrapped again (some code does 'state_dict': model.state_dict())
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    print(f"   Original Parameters: {len(state_dict)} keys")
    
    new_state_dict = {}
    
    print("   Processing Layers (Fractal Duplication)...")
    
    # 1. Copy Non-Layer Params (Embeddings, Norms, etc.)
    # We need to be careful. 'layers.0.xxx' -> 'layers.0.xxx' AND 'layers.32.xxx'
    
    # Separate layers from others
    layer_params = {}
    other_params = {}
    
    for k, v in state_dict.items():
        if k.startswith("layers."):
            layer_params[k] = v
        else:
            other_params[k] = v
            new_state_dict[k] = v.clone() # Copy base params exactly
            
    print(f"   Base Parameters (Non-Layer): {len(other_params)} keys copied.")
    
    # 2. Expand Layers 0-31 to 0-63
    # We iterate 0 to 31.
    # New dict will have 0..31 (Original) and 32..63 (Fractal Copy)
    
    for i in range(32):
        prefix = f"layers.{i}."
        
        # Find all keys for this layer idx
        current_layer_keys = [k for k in layer_params.keys() if k.startswith(prefix)]
        
        for k in current_layer_keys:
            # Suffix is the part after "layers.i."
            suffix = k[len(prefix):] # e.g. "norm.w", "lora_in.A"
            
            # --- LOWER FRACTAL (0-31) ---
            # Keep identical to trained 25B
            new_key_lower = k
            new_state_dict[new_key_lower] = layer_params[k].clone()
            
            # --- UPPER FRACTAL (32-63) ---
            # Copy + Noise
            new_layer_idx = i + 32
            new_key_upper = f"layers.{new_layer_idx}.{suffix}"
            
            tensor = layer_params[k].clone()
            
            # Add noise only to floating point tensors (weights), not boolean/long
            if tensor.is_floating_point():
                 noise = torch.randn_like(tensor) * tensor.std() * NOISE_SCALE
                 tensor += noise
            
            new_state_dict[new_key_upper] = tensor

        if i % 4 == 0:
            print(f"   ✨ Expanded Layer {i} → {i} & {i+32}")

    # Verify we have 64 layers worth of params
    total_keys = len(new_state_dict)
    print(f"   Total New Parameters: {total_keys} keys")
    
    print(f"   Saving 50B Seed Model to {OUTPUT_50B_PATH}...")
    torch.save({'model': new_state_dict, 'step': 0, 'fractional_epoch': 0}, OUTPUT_50B_PATH)
    
    size_gb = os.path.getsize(OUTPUT_50B_PATH) / 1e9 if os.path.exists(OUTPUT_50B_PATH) else 0
    print(f"✅ EXPANSION COMPLETE!")
    print(f"   New Model Size: {size_gb:.2f} GB")
    print(f"   Structure: 64 Layers (Doubled from 32)")
    print(f"   Ready for Phase 3 Training.")

if __name__ == "__main__":
    expand_model()
