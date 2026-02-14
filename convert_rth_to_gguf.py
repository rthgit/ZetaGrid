import numpy as np
import torch
import math
import gguf
import os

# CONFIG (Matches ZETAGRID_INFERENCE.py)
D_MODEL = 4096
D_FF = 16384
N_LAYERS = 32
KERNEL_SIZE = 3
LORA_RANK = 128

def convert_rth_to_gguf(weights_path, ckpt_path, output_gguf, n_layers=N_LAYERS):
    # 1. LOAD DATA
    print(f"📂 Loading Genome from {weights_path}...")
    genome_data = np.load(weights_path).astype(np.float32)
    print(f"   [DEBUG] Genome data size: {len(genome_data)}")
    print(f"📂 Loading Soul Checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get('model', ckpt.get('model_state_dict', {}))
    
    # 2. INITIALIZE GGUF WRITER
    writer = gguf.GGUFWriter(output_gguf, "rth_tcn")
    
    # Metadata
    writer.add_name("RTH-LM 25B (Fractal TCN)")
    writer.add_uint32("rth_tcn.block_count", n_layers)
    writer.add_uint32("rth_tcn.embedding_length", D_MODEL)
    writer.add_uint32("rth_tcn.feed_forward_length", D_FF)
    writer.add_uint32("rth_tcn.kernel_size", KERNEL_SIZE)
    
    # 3. GLOBAL TENSORS
    print("💎 Mapping Global Tensors...")
    writer.add_tensor("token_embd.weight", state['emb.weight'].numpy())
    writer.add_tensor("pos_embd.weight", state['pos_emb.weight'].numpy())
    writer.add_tensor("output_norm.weight", state['norm_f.w'].numpy())
    
    # 4. LAYER TENSORS (GENOME + SOUL)
    offset = 0
    print("💎 Mapping Layer Tensors (Genome + Soul)...")
    
    for i in range(n_layers):
        prefix = f"blk.{i}"
        pytorch_prefix = f"layers.{i}"
        
        # --- GENOME EXTRACTION (Buffers) ---
        # w_in
        n = 2 * D_FF * D_MODEL
        w_in = genome_data[offset:offset+n].reshape(2 * D_FF, D_MODEL)
        offset += n
        writer.add_tensor(f"{prefix}.tcn_in.weight", w_in)
        
        # w_dw
        n = D_FF * KERNEL_SIZE
        w_dw = genome_data[offset:offset+n].reshape(D_FF, 1, KERNEL_SIZE)
        offset += n
        writer.add_tensor(f"{prefix}.tcn_dw.weight", w_dw)
        
        # w_out
        n = D_MODEL * D_FF
        w_out = genome_data[offset:offset+n].reshape(D_MODEL, D_FF)
        offset += n
        writer.add_tensor(f"{prefix}.tcn_out.weight", w_out)
        
        # --- SOUL MAPPING (LoRA + Scales) ---
        writer.add_tensor(f"{prefix}.attn_norm.weight", state[f"{pytorch_prefix}.norm.w"].numpy())
        writer.add_tensor(f"{prefix}.scale", state[f"{pytorch_prefix}.scale"].numpy())
        
        writer.add_tensor(f"{prefix}.lora_in_a.weight", state[f"{pytorch_prefix}.lora_in.A"].numpy())
        writer.add_tensor(f"{prefix}.lora_in_b.weight", state[f"{pytorch_prefix}.lora_in.B"].numpy())
        writer.add_tensor(f"{prefix}.lora_out_a.weight", state[f"{pytorch_prefix}.lora_out.A"].numpy())
        writer.add_tensor(f"{prefix}.lora_out_b.weight", state[f"{pytorch_prefix}.lora_out.B"].numpy())
        
        if (i+1) % 8 == 0 or (i+1) == n_layers:
            print(f"   Processed Layer {i+1}/{n_layers}")

    # 5. WRITE FILE
    print(f"🚀 Serializing to {output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("✅ RTH-LM GGUF Release Prepared.")

if __name__ == "__main__":
    GENOME = "E:/ZETAGRID/zetagrid_25b_production.npy"
    CKPT = "E:/ZETAGRID/zeta25b_step15000.pt"
    OUT = "E:/ZETAGRID/rth_lm_25b_v1.gguf"
    
    if os.path.exists(GENOME) and os.path.exists(CKPT):
        convert_rth_to_gguf(GENOME, CKPT, OUT)
    else:
        print("⚠️ Production files not found. Check E:/ZETAGRID paths.")
