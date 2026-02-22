import torch
import os
import sys

# CONFIG A40
BASE_DIR = "/workspace/zetagrid_50b"
MODEL_PATH = f"{BASE_DIR}/repaired_checkpoints/zeta_25B_v2.pt"
OUTPUT_PATH = f"{BASE_DIR}/zeta_25B_v2_soul.pt" # The Small Adapter

def extract_soul():
    print(f"🧬 ZETAGRID SOUL EXTRACTION (A40 REMOTE)")
    print(f"Loading Checkpoint: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ File not found.")
        return

    # Load Full Checkpoint
    try:
        full_ckpt = torch.load(MODEL_PATH, map_location="cpu")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # Handle State Dict
    state = full_ckpt.get('model', full_ckpt.get('model_state_dict', full_ckpt))
    
    soul_state = {}
    genome_count = 0
    soul_count = 0
    
    print("💎 Filtering Weights (Keeping LoRA, Norm, Emb)...")
    
    for k, v in state.items():
        # Clean Key if needed
        key = k # Keep raw key to preserve structure? Or clean? 
        # Ideally clean prefixes like 'module.' or 'base.'
        clean_key = k.replace("module.", "").replace("_orig_mod.", "").replace("base.", "")
        
        # Heuristic: Match 'lora', 'norm', 'scale', 'emb'
        if "lora" in clean_key or "norm" in clean_key or "scale" in clean_key or "emb" in clean_key:
            # Check if it's NOT Genome
            if "w_in" in clean_key or "w_dw" in clean_key or "w_out" in clean_key:
                # Tricky case: is it lora_in? or w_in?
                # 'w_in' is usually heavy weight. 'lora_in' is adapter.
                # If key contains 'lora', SAFE.
                if "lora" in clean_key:
                    soul_state[clean_key] = v
                    soul_count += 1
                else: 
                     # Norm/Scale/Emb safe?
                     # scale_in is scalar. w_in is matrix.
                     # CAREFUL: 'w_in.weight' vs 'scale_in'
                     if "w_" not in clean_key: # If no w_in, w_dw, w_out
                          soul_state[clean_key] = v
                          soul_count += 1
                     else:
                          genome_count += 1
            else:
                 soul_state[clean_key] = v
                 soul_count += 1
                 
        elif "w_in" in clean_key or "w_dw" in clean_key or "w_out" in clean_key:
            genome_count += 1
        else:
            soul_state[clean_key] = v
            # print(f"⚠️  Keeping unknown key: {clean_key}")

    print(f"✅ Extracted {soul_count} Soul Tensors.")
    print(f"🗑️  Dropped {genome_count} Genome Tensors.")
    
    # Save
    print(f"💾 Saving to {OUTPUT_PATH}...")
    torch.save(soul_state, OUTPUT_PATH)
    
    # Verify Size
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"📦 Final Soul Size: {size_mb:.2f} MB")
    print("DONE.")

if __name__ == "__main__":
    extract_soul()
