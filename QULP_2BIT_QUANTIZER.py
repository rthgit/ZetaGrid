import torch
import torch.nn as nn
import os
import gc
import math
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = "zeta25b_step15000.pt"       # Input 25B model
OUTPUT_QULP_PATH = "zeta25b_2bit.qulp"    # Output Quantized Model

# QULP Config
BITS = 2
GROUP_SIZE = 128 # Quantize in groups of 128 for accuracy

def quantize_to_2bit(tensor):
    """
    Simulates 2-bit quantization (4 values: -1.5, -0.5, 0.5, 1.5)
    In practice, we map to indices [0, 1, 2, 3] and store scale factors.
    """
    if tensor.numel() == 0: return tensor, None
    
    # Reshape into groups
    shape = tensor.shape
    numel = tensor.numel()
    
    # Pad if needed
    pad = 0
    if numel % GROUP_SIZE != 0:
        pad = GROUP_SIZE - (numel % GROUP_SIZE)
        tensor = torch.nn.functional.pad(tensor.flatten(), (0, pad))
    
    t_flat = tensor.view(-1, GROUP_SIZE)
    
    # Calculate min/max per group
    # We use absmax quantization usually, or min-max
    # For 2-bit, we want to capture the dynamic range efficiently.
    
    # Simple Symmetric quantization:
    # Max value per group
    absmax = t_flat.abs().max(dim=1, keepdim=True)[0] + 1e-6
    scale = absmax / 1.5 # 2-bit range is [-2, 1] usually or centers? 
    # 2-bit values: -2, -1, 0, 1 ??? Or -1.5, -0.5, 0.5, 1.5 (if centered)?
    # Standard: 0, 1, 2, 3 mapped to real values.
    
    t_scaled = t_flat / scale
    
    # Round to nearest integer [-2, -1, 0, 1] ??
    # Let's use 4 levels around 0: -1.5, -0.5, 0.5, 1.5
    # t_scaled should be in range [-1.5, 1.5] roughly?
    
    # Actually, simplistic 2-bit is often:
    # 00 -> -1
    # 01 -> -0.33
    # 10 -> +0.33
    # 11 -> +1
    # But let's stick to standard integer quantization for simplicity in this script
    # Map float to [0, 3]
    
    min_val = t_flat.min(dim=1, keepdim=True)[0]
    max_val = t_flat.max(dim=1, keepdim=True)[0]
    scale = (max_val - min_val) / 3.0 + 1e-8
    zero_point = min_val
    
    t_quant = ((t_flat - zero_point) / scale).round().clamp(0, 3).to(torch.uint8)
    
    # Pack 4 values (2 bits each) into 1 byte (uint8)
    # This reduces size by 4x vs int8, 16x vs fp32
    # [0, 1, 2, 3] -> 2 bits
    
    # We won't implement bitpacking in Python for speed in this demo, 
    # but we save the quantized tensor as uint8 (4x compression vs float32)
    # Real 2-bit packing requires bitwise ops
    
    # Return uint8 tensor and scales check
    return t_quant, (scale, zero_point), shape, pad

def run_quantization():
    print(f"🧊 ZETAGRID QULP 2-BIT QUANTIZER")
    print(f"   Input: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    print("   Loading Model (BF16)...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict'] # Handle wrapper
    
    quantized_dict = {}
    total_params = 0
    total_compressed_bytes = 0
    
    print("   Quantizing Layers...")
    
    for k, v in state_dict.items():
        # Only quantize weights (dim > 1), keep biases/norms in float16/32
        if v.ndim > 1 and v.is_floating_point() and "norm" not in k:
            # Quantize
            q_tensor, scales, original_shape, pad = quantize_to_2bit(v.float())
            
            # Pack simulates the size reduction
            # 2 bits per element. 
            # q_tensor is uint8 (8 bits). 
            # So real compressed size is numel * 2 bits = numel / 4 bytes
            compressed_bytes = q_tensor.numel() // 4
            
            quantized_dict[k] = {
                'q_data': q_tensor, # Stored as uint8 for now (in simulation)
                'scales': scales[0].half(),
                'zeros': scales[1].half(),
                'shape': original_shape,
                'pad': pad
            }
            total_compressed_bytes += compressed_bytes
        else:
            # Keep as is (convert to half for compact)
            quantized_dict[k] = v.half()
            total_compressed_bytes += v.numel() * 2 # FP16 bytes
            
        total_params += v.numel()
        
    print(f"   Saving QULP Model to {OUTPUT_QULP_PATH}...")
    torch.save(quantized_dict, OUTPUT_QULP_PATH)
    
    original_size = os.path.getsize(MODEL_PATH) / 1e9
    new_size = os.path.getsize(OUTPUT_QULP_PATH) / 1e9
    
    # Note: torch.save overhead might mask true 2-bit gains without bitpacking lib
    # But theoretically:
    theoretical_size = total_compressed_bytes / 1e9
    
    print(f"✅ QUANTIZATION COMPLETE!")
    print(f"   Original Size: {original_size:.2f} GB")
    print(f"   QULP 2-bit Size (Theoretical): {theoretical_size:.2f} GB")
    print(f"   Saved to: {OUTPUT_QULP_PATH}")

if __name__ == "__main__":
    run_quantization()
