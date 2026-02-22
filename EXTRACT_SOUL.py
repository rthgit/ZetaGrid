import torch
import os
import sys

# CONFIG
MODEL_PATH = "E:/ZETAGRID/zeta_25B_v2.pt"
OUTPUT_PATH = "E:/ZETAGRID/zeta_25B_v2_soul.pt"
GENOME_PREFIXES = ["w_in", "w_dw", "w_out"]
SOUL_PREFIXES = ["lora_", "norm", "scale", "emb", "pos_emb"]

def extract_soul():
    print(f"🧬 ZETAGRID SOUL EXTRACTION")
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
    
    print("💎 Filtering Weights...")
    for k, v in state.items():
        # Clean Key
        key = k.replace("module.", "").replace("_orig_mod.", "").replace("base.", "")
        
        # Check if Genome (Frozen)
        is_genome = any(p in key for p in GENOME_ বিদ্ব) # Wait, simple string check
        # Specifically: lines like 'layers.0.w_in.weight' are Genome.
        # But 'layers.0.lora_in.A' are Soul.
        
        # Heuristic:
        # If key contains 'lora', 'norm', 'scale', 'emb', KEEP IT.
        # If key contains 'w_in', 'w_out', 'w_dw' AND NOT 'lora', DROP IT.
        
        if "lora" in key or "norm" in key or "scale" in key or "emb" in key:
            soul_state[key] = v
            soul_count += 1
        elif "w_in" in key or "w_dw" in key or "w_out" in key:
            # Drop Genome
            genome_count += 1
        else:
            # Unknown? Keep to be safe (e.g. bias?)
            soul_state[key] = v
            print(f"⚠️  Keeping unknown key: {key}")

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
