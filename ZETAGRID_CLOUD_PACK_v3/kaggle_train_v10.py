import ctypes
import time
import os
import subprocess
import threading

# ==========================================================
# ZETAGRID v10: KAGGLE FLASH TRAINER
# ==========================================================

def compile_engine():
    print("🔨 Compiling ZetaGrid Engine...")
    cmd = "g++ -shared -fPIC -O3 zeta_v10_lib.cpp -o libzeta_v10.so -lOpenCL"
    subprocess.check_call(cmd, shell=True)
    print("✅ Compilation Complete.")

# Thread Worker for one GPU Island
def train_island(gpu_id, params_billions, duration_hours):
    print(f"🏝️ [Island {gpu_id}] loading LibZeta...")
    
    # Load Library (Unique instance per thread requires care, but ctypes is thread-safe usually)
    # Actually, global vars in C++ might conflict if shared lib is loaded once.
    # Hack: We rely on single process for now handling one GPU or separate scripts.
    # For Kaggle Dual T4, simpler to run this script twice with an argument.
    
    lib = ctypes.CDLL("./libzeta_v10.so")
    
    lib.InitGPU.argtypes = [ctypes.c_int]
    lib.AllocModel.argtypes = [ctypes.c_int]
    lib.EvolveStep.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_long]
    lib.AcceptMutation.argtypes = [ctypes.c_long]
    lib.Sync.argtypes = []

    # Init
    print(f"🏝️ [Island {gpu_id}] Init GPU...")
    if lib.InitGPU(gpu_id) != 0:
        print(f"❌ GPU {gpu_id} Init Failed.")
        return

    # Alloc: 12GB provides 48 Billion Params (2-bit)
    # T4 has 16GB, 12GB is safe.
    model_size_gb = 12 
    total_bytes = model_size_gb * 1024*1024*1024
    
    print(f"🏝️ [Island {gpu_id}] Allocating {model_size_gb}GB VRAM for 48 Billion Params...")
    if lib.AllocModel(model_size_gb) != 0:
        print("❌ Alloc Failed.")
        return

    print(f"🚀 [Island {gpu_id}] STARTING EVOLUTION ({duration_hours} Hours)...")
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    gen = 0
    mutation_rate = 0.0001
    seed = 0.123
    
    while time.time() < end_time:
        gen += 1
        
        # 1. Mutate
        seed = (seed * 1.5) % 100.0
        lib.EvolveStep(mutation_rate, seed, total_bytes)
        
        # 2. Evaluate (Simulated in this loop, would be Forward Pass)
        # Assuming improvement found periodically
        if gen % 10 == 0:
            lib.AcceptMutation(total_bytes)
            
        lib.Sync()
        
        if gen % 1000 == 0:
            elapsed = time.time() - start_time
            gens_sec = gen / elapsed
            print(f"🏝️ [Island {gpu_id}] Gen {gen} | Speed: {gens_sec:.2f} Hz | Rate: {(total_bytes*gens_sec*4)/1e9:.2f} G-Params/s")

    print(f"🏁 [Island {gpu_id}] Finished.")

if __name__ == "__main__":
    if not os.path.exists("./libzeta_v10.so"):
        compile_engine()
    
    # Launch for GPU 0
    # For Dual T4, user can duplicate this or use threading.
    # Let's run Single Island huge model for safety.
    train_island(0, 48, 10.0) 
