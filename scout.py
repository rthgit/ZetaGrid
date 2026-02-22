"""
🧬 ZETAGRID SCOUT v2.0 - COMPREHENSIVE HARDWARE RECON
======================================================
Run this on any cloud environment to discover:
- GPU specs (CUDA, compute capability, memory, Flash Attention support)
- CPU specs (cores, architecture, frequency)
- RAM and SWAP
- Disk space
- TPU/NPU/Special Accelerators
- Key ML libraries and their versions
"""

import os
import sys
import platform
import subprocess

def run_cmd(cmd):
    """Safely run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return "N/A"

def scout_gpu():
    print("\n" + "="*60)
    print("🎮 GPU RECON")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"| CUDA Available: ✅ YES")
            print(f"| CUDA Version: {torch.version.cuda}")
            print(f"| cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"| GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n| GPU {i}: {props.name}")
                print(f"|   VRAM: {props.total_memory / (1024**3):.2f} GB")
                print(f"|   Compute Capability: {props.major}.{props.minor}")
                print(f"|   SM Count: {props.multi_processor_count}")
                print(f"|   L2 Cache: {props.l2_cache_size / (1024**2):.1f} MB" if hasattr(props, 'l2_cache_size') else "")
                
                # Flash Attention 2 support
                if props.major >= 8:
                    print(f"|   ⚡ Flash Attention 2: SUPPORTED (CC >= 8.0)")
                    print(f"|   ⚡ BF16: SUPPORTED")
                else:
                    print(f"|   ⚠️ Flash Attention 2: NOT SUPPORTED (Need CC >= 8.0)")
                    print(f"|   ⚠️ BF16: Limited support, use FP16")
                    
                # Tensor Cores
                if props.major >= 7:
                    print(f"|   ⚡ Tensor Cores: YES")
                else:
                    print(f"|   ⚠️ Tensor Cores: NO")
                    
            # Memory bandwidth estimate
            print(f"\n| Current GPU Memory:")
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"|   GPU {i}: {free/(1024**3):.2f} GB free / {total/(1024**3):.2f} GB total")
        else:
            print("| CUDA Available: ❌ NO")
    except ImportError:
        print("| PyTorch: NOT INSTALLED")
    except Exception as e:
        print(f"| Error: {e}")

def scout_cpu():
    print("\n" + "="*60)
    print("🖥️ CPU RECON")
    print("="*60)
    
    print(f"| Logical Cores: {os.cpu_count()}")
    print(f"| Architecture: {platform.machine()}")
    
    # Get processor info
    proc_cmd = 'cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2'
    proc_info = platform.processor() or run_cmd(proc_cmd)
    print(f"| Processor: {proc_info}")
    
    # Try to get physical cores
    try:
        import psutil
        print(f"| Physical Cores: {psutil.cpu_count(logical=False)}")
        freq = psutil.cpu_freq()
        if freq:
            print(f"| CPU Frequency: {freq.current:.0f} MHz (max: {freq.max:.0f} MHz)")
    except:
        pass
    
    # Check for specific CPU features
    cpu_info = run_cmd("cat /proc/cpuinfo 2>/dev/null || echo 'N/A'")
    if "avx512" in cpu_info.lower():
        print(f"| ⚡ AVX-512: SUPPORTED")
    elif "avx2" in cpu_info.lower():
        print(f"| ⚡ AVX2: SUPPORTED")
    elif "avx" in cpu_info.lower():
        print(f"| ⚡ AVX: SUPPORTED")

def scout_memory():
    print("\n" + "="*60)
    print("💾 MEMORY RECON")
    print("="*60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"| Total RAM: {mem.total / (1024**3):.2f} GB")
        print(f"| Available RAM: {mem.available / (1024**3):.2f} GB")
        print(f"| RAM Usage: {mem.percent}%")
        print(f"| Swap Total: {swap.total / (1024**3):.2f} GB")
        print(f"| Swap Used: {swap.used / (1024**3):.2f} GB")
    except ImportError:
        # Fallback
        mem_info = run_cmd("free -g")
        print(f"| Memory Info:\n{mem_info}")

def scout_disk():
    print("\n" + "="*60)
    print("💿 DISK RECON")
    print("="*60)
    
    try:
        import psutil
        for part in psutil.disk_partitions():
            if 'loop' not in part.device:
                usage = psutil.disk_usage(part.mountpoint)
                print(f"| {part.mountpoint}: {usage.free/(1024**3):.1f} GB free / {usage.total/(1024**3):.1f} GB total")
    except:
        disk_info = run_cmd("df -h / /workspace 2>/dev/null || df -h /")
        print(f"| {disk_info}")

def scout_accelerators():
    print("\n" + "="*60)
    print("🚀 SPECIAL ACCELERATORS RECON")
    print("="*60)
    
    # TPU Check
    try:
        import torch_xla
        print("| ⚡ TPU (XLA): DETECTED")
        import torch_xla.core.xla_model as xm
        print(f"|   Devices: {xm.get_xla_supported_devices()}")
    except ImportError:
        print("| TPU (XLA): Not available")
    
    # Intel NPU / Gaudi Check
    try:
        import habana_frameworks
        print("| ⚡ Intel Gaudi/Habana: DETECTED")
    except ImportError:
        pass
    
    # AMD ROCm Check
    rocm = run_cmd("which rocminfo 2>/dev/null")
    if rocm and "not found" not in rocm.lower():
        print("| ⚡ AMD ROCm: DETECTED")
        rocm_info = run_cmd("rocminfo | grep 'Name:' | head -3")
        print(f"|   {rocm_info}")
    
    # Check for Apple Silicon MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("| ⚡ Apple MPS: DETECTED")
    except:
        pass

def scout_libraries():
    print("\n" + "="*60)
    print("📚 ML LIBRARIES")
    print("="*60)
    
    libs = [
        'torch', 'transformers', 'accelerate', 'bitsandbytes', 
        'triton', 'flash_attn', 'xformers', 'deepspeed',
        'datasets', 'tokenizers', 'safetensors'
    ]
    for lib in libs:
        try:
            mod = __import__(lib.replace('-', '_'))
            ver = getattr(mod, '__version__', '✓')
            print(f"| {lib}: {ver}")
        except ImportError:
            print(f"| {lib}: ❌ NOT INSTALLED")

def scout_environment():
    print("\n" + "="*60)
    print("🌍 ENVIRONMENT")
    print("="*60)
    print(f"| Python: {sys.version.split()[0]}")
    print(f"| Platform: {platform.system()} {platform.release()}")
    print(f"| Hostname: {platform.node()}")
    
    # Cloud detection
    if os.path.exists('/kaggle'):
        print("| ☁️ Environment: KAGGLE")
    elif os.path.exists('/content'):
        print("| ☁️ Environment: GOOGLE COLAB")
    elif os.environ.get('RUNPOD_POD_ID'):
        print("| ☁️ Environment: RUNPOD")
    elif os.environ.get('AWS_EXECUTION_ENV'):
        print("| ☁️ Environment: AWS")
    else:
        print("| ☁️ Environment: Unknown/Local")

def main():
    print("🧬 ZETAGRID SCOUT v2.0")
    print("=" * 60)
    
    scout_environment()
    scout_gpu()
    scout_cpu()
    scout_memory()
    scout_disk()
    scout_accelerators()
    scout_libraries()
    
    print("\n" + "="*60)
    print("✅ SCOUT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
