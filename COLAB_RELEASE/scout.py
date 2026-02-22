import os
import platform
import subprocess
import json
import sys

def get_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return None

def scout_hardware():
    print("🧬 ZETAGRID HARDWARE FINGERPRINT v1.0")
    print("======================================")
    
    info = {
        "os": platform.system() + " " + platform.release(),
        "cpu_model": platform.processor(),
        "arch": platform.machine(),
        "cpu_flags": [],
        "gpu_devices": [],
        "npu_detected": False,
        "cl_platforms": []
    }

    # 1. CPU FLAGS (Crucial for AVX-512 o AVX2)
    if platform.system() == "Windows":
        cmd = "wmic cpu get Caption, name, L2CacheSize, L3CacheSize, NumberOfCores, NumberOfLogicalProcessors"
        info["cpu_details"] = get_output(cmd)
        # Check for AVX-512 via instruction detection
        try:
            import ctypes
            is_avx512 = ctypes.windll.kernel32.IsProcessorFeaturePresent(38) # PF_AVX512F_INSTRUCTIONS_AVAILABLE
            if is_avx512: info["cpu_flags"].append("AVX-512 (Extreme Potential)")
        except:
            pass
    else:
        info["cpu_flags"] = get_output("grep flags /proc/cpuinfo | head -1")

    # 2. DISCOVERY GPU & RUNTIME (OpenCL / ROCm)
    print("🔍 Looking for Compute Accelerators...")
    cl_info = get_output("clinfo -s")
    if cl_info:
        info["cl_platforms"] = cl_info.split('\n')
    
    # 3. NPU DETECTION (AMD XDNA / Ryzen AI)
    if platform.system() == "Windows":
        npu_check = get_output('powershell "Get-PnpDevice | Where-Object { $_.FriendlyName -match \'NPU\' -or $_.FriendlyName -match \'XDNA\' }"')
        if npu_check:
            info["npu_detected"] = True
            info["npu_info"] = npu_check
    
    # 4. GPU SPECS (Vulkan info)
    vulkan = get_output("vulkaninfo --summary")
    if vulkan:
        info["vulkan_summary"] = "Detected"

    # OUTPUT FINALE PER IL TECNICO
    print(f"\n🌍 SISTEMA: {info['os']}")
    print(f"🧠 CPU: {info['cpu_model']}")
    print(f"🚀 ISTRUZIONI: {', '.join(info['cpu_flags']) if info['cpu_flags'] else 'AVX2 Standard'}")
    
    print(f"\n🎮 GPU/ACCEL STATUS:")
    if cl_info:
        print(f" ├─ OpenCL: DISPONIBILE")
        for p in info["cl_platforms"]: print(f" │  └─ {p}")
    else:
        print(f" ├─ OpenCL: NON RILEVATO (Installare Driver Hardware)")

    if info["npu_detected"]:
        print(f" ├─ NPU (Neural Processor): RILEVATA ✅")
    else:
        print(f" └─ NPU (Neural Processor): Non visibile (Verificare driver)")

    print("\n======================================")
    print("💡 ANALISI PER ZETAGRID:")
    if "AVX-512" in str(info["cpu_flags"]):
        print("- CPU: Target AVX-512 sbloccato. Possibile raddoppio GFLOPS.")
    print("- GPU: Target Ryzen AI Max (RX 8060S / 890M) con Memoria Unificata.")
    print("- STATUS: Pronto per l'iniezione dei kernel residenti.")

if __name__ == "__main__":
    scout_hardware()
