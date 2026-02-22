import os
import platform
import multiprocessing
import subprocess
import sys

def get_cpu_info_linux():
    """Parses lscpu on Linux/WSL"""
    info = {}
    try:
        output = subprocess.check_output("lscpu", shell=True).decode()
        for line in output.split('\n'):
            if ":" in line:
                key, val = line.split(':', 1)
                info[key.strip()] = val.strip()
    except:
        pass
    return info

def detect_extensions_linux():
    """Checks for AVX2, AVX512, VNNI"""
    flags = []
    try:
        output = subprocess.check_output("lscpu | grep Flags", shell=True).decode()
        if "avx2" in output: flags.append("AVX2")
        if "avx512" in output: flags.append("AVX-512")
        if "avx_vnni" in output or "avx512_vnni" in output: flags.append("VNNI (DL Boost)")
        if "f16c" in output: flags.append("FP16 (F16C)")
    except:
        pass
    return flags

def calculate_potential(cores_phys, freq_ghz, flags):
    """
    Calculates Theoretical Peaks.
    Haswell/Skylake/AlderLake (AVX2): 16 SP FLOPS / cycle / core (2 FMA ports * 8 width).
    VNNI (Int8): 4x throughput of FP32 (roughly).
    """
    
    # 1. FP32 Potential (ZetaGrid)
    # 2 FMA units * 8 elements (256bit) = 16 ops/cycle
    peak_gflops_fp32 = cores_phys * freq_ghz * 16
    
    # 2. FP16 Potential (Storage/Bandwidth Boost)
    # Usually bandwidth limited, but compute is same as FP32 on AVX2 (conversion overhead)
    # Effective throughput ~1.5x due to cache.
    estimated_gflops_fp16 = peak_gflops_fp32 * 1.5
    
    # 3. Int8 Potential (VNNI)
    # If VNNI present: 64 ops/cycle? (Assuming 4x int8 packing per lane)
    peak_tops_int8 = 0
    if "VNNI (DL Boost)" in flags:
        peak_tops_int8 = cores_phys * freq_ghz * 64 / 1000.0 # TOPS
    else:
        # Fallback to standard Int8 SIMD (roughly 2x-3x FP32)
        peak_tops_int8 = peak_gflops_fp32 * 2 / 1000.0
        
    return peak_gflops_fp32, estimated_gflops_fp16, peak_tops_int8

def get_gpu_info():
    """Detects NVIDIA GPU info via nvidia-smi"""
    gpu_name = "None"
    gpu_mem = 0
    try:
        # Simple query for name and total memory
        output = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True).decode().strip()
        if output:
            parts = output.split(',')
            gpu_name = parts[0].strip()
            gpu_mem = parts[1].strip()
    except:
        pass
    return gpu_name, gpu_mem

def main():
    print(">>> ZETAGRID HARDWARE POTENTIAL PROBE <<<")
    print("Analyzing Compute Node...")
    
    # 1. OS Check
    sys_os = platform.system()
    print(f"OS: {sys_os}")
    
    # 2. CPU Analysis
    info = {}
    flags = []
    cores = multiprocessing.cpu_count()
    freq = 2.3 # GHz (Colab Xeon default)
    
    if sys_os == "Linux":
        info = get_cpu_info_linux()
        flags = detect_extensions_linux()
        if 'CPU(s)' in info: cores = int(info['CPU(s)'])
        if 'Model name' in info:
            print(f"CPU Model: {info['Model name']}")
            if "2.30GHz" in info['Model name']: freq = 2.3
            if "2.20GHz" in info['Model name']: freq = 2.2
            
    print(f"Detected CPU Cores: {cores}")
    print(f"CPU Extensions: {', '.join(flags)}")
    
    # 3. GPU Analysis
    gpu_name, gpu_mem = get_gpu_info()
    print(f"Detected GPU: {gpu_name} ({gpu_mem})")
    
    # 4. Roofline Calculation
    # CPU (AVX2): 16 ops/cycle * cores * freq
    cpu_gflops = cores * freq * 16
    
    # GPU (T4 Spec): 8.1 TFLOPS FP32, 65 TFLOPS FP16 (Tensor)
    gpu_tflops = 0
    gpu_tensor_tflops = 0
    if "T4" in gpu_name:
        gpu_tflops = 8.1
        gpu_tensor_tflops = 65.0
    elif "V100" in gpu_name:
        gpu_tflops = 15.7
        gpu_tensor_tflops = 125.0
    elif "A100" in gpu_name:
        gpu_tflops = 19.5
        gpu_tensor_tflops = 312.0 # FP16 Tensor
    
    print("\n>>> THEORETICAL PEAK PERFORMANCE <<<")
    print(f"1. CPU ENGINE (AVX2):       {cpu_gflops:.2f} GFLOPS")
    print(f"2. GPU ENGINE (FP32):       {gpu_tflops:.2f} TFLOPS ({(gpu_tflops*1000/cpu_gflops):.1f}x CPU)")
    print(f"3. GPU TENSOR (FP16):       {gpu_tensor_tflops:.2f} TFLOPS ({(gpu_tensor_tflops*1000/cpu_gflops):.1f}x CPU)")
    
    print("\n[ANALYSIS]")
    if gpu_tflops > 0:
        print("This node is ACCELERATED. The ZetaGrid 3D Kernel should target the TENSOR limit.")
        print(f"Potential Upside: Up to {gpu_tensor_tflops:.1f} Trillion Operations/Sec.")
    else:
        print("This node is CPU-ONLY. Performance is limited to ~{cpu_gflops:.0f} GFLOPS.")

if __name__ == "__main__":
    main()
