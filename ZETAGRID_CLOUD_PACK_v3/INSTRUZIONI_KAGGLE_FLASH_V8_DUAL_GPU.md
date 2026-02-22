# 🦅 KAGGLE V8 (DUAL GPU - 48 MILIARDI)

## 🔥 ORA USIAMO TUTTO
Hai confermato **2 GPU**.
Il codice V7 ne usava solo una (24B Parametri).
Il codice V8 lancia **DUE PROCESSI PARALLELI**:
1.  **GPU 0 (Adam):** 24 Miliardi di Parametri.
2.  **GPU 1 (Eve):** 24 Miliardi di Parametri.
**TOTALE: 48 MILIARDI DI PARAMETRI IN EVOLUZIONE SIMULTANEA.**

## 1. Copia-Incolla QUESTO (E sovrascrivi tutto):

```python
# ==========================================
# 🦅 ZETAGRID v10: KAGGLE DUAL-GPU SWARM (V8)
# ==========================================
import os
import time
import ctypes
import subprocess
import numpy as np
import urllib.request
import random
import multiprocessing

# --- 1. SETUP AMBIENTE ---
if __name__ == "__main__":
    print("📦 INSTALLAZIONE LIBRERIE...")
    os.system("apt-get update && apt-get install -y opencl-headers ocl-icd-opencl-dev")

    # --- DOWNLOAD DATASET ---
    print("📚 Downloading WikiText-2...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    urllib.request.urlretrieve(url, "wikitext.txt")
    print("✅ Dataset Ready.")

# --- 2. C++ ENGINE ---
cpp_source = r"""
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

const char* kernel_src = R"(
inline float unpack_weight(uchar packed, int idx) {
    uchar bits = (packed >> (idx * 2)) & 0x3;
    if (bits == 1) return 1.0f;
    if (bits == 2) return -1.0f;
    return 0.0f;
}

// INIT KERNEL
__kernel void init_genome(__global uchar* Genome, const float chaos_seed, const ulong n_bytes) {
    ulong gid = get_global_id(0);
    if(gid >= n_bytes) return;
    uint seed = (uint)gid ^ as_uint(chaos_seed);
    seed = (seed * 1664525u) + 1013904223u;
    Genome[gid] = (uchar)(seed & 0xFF);
}

__kernel void mutate_gene(__global uchar* Genome, __global const uchar* BestGenome, const float mutation_rate, const float chaos_seed, const ulong n_bytes) {
    ulong gid = get_global_id(0);
    if(gid >= n_bytes) return;
    uchar best_byte = BestGenome[gid];
    uint seed = (uint)gid ^ as_uint(chaos_seed);
    seed = (seed * 1664525u) + 1013904223u;
    float rng = (float)(seed & 0xFFFFFF) / 16777216.0f;
    if (rng < mutation_rate) {
        uint noise = (seed >> 8) & 0x3; 
        if(noise == 3) noise = 0; 
        uint shift = (seed % 4) * 2;
        uchar mask = ~(0x3 << shift);
        Genome[gid] = (best_byte & mask) | (noise << shift);
    } else {
        Genome[gid] = best_byte;
    }
}

__kernel void evaluate_loss(__global const uchar* Genome, __global const uchar* Data, __global float* LossBuffer, const int offset, const ulong n_bytes) {
    int gid = get_global_id(0); 
    int seq_len = 64; 
    float prediction = 0.0f;
    for(int i=0; i<seq_len; i++) {
        ulong w_idx = ((ulong)gid * seq_len + i) % n_bytes; 
        float w = unpack_weight(Genome[w_idx / 4], w_idx % 4);
        float input_val = (float)Data[offset + gid - i] / 255.0f; 
        prediction += input_val * w;
    }
    prediction = tanh(prediction);
    float target = (float)Data[offset + gid + 1] / 255.0f;
    float diff = prediction - target;
    LossBuffer[gid] = diff * diff; 
}
)";

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel k_init, k_mutate, k_eval;
cl_mem d_genome_best, d_genome_trial, d_data, d_loss;

extern "C" {
    __attribute__((visibility("default"))) int InitGPU(int device_idx) {
        cl_int err;
        cl_uint num_platforms; 
        clGetPlatformIDs(0, NULL, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), NULL);
        if(num_platforms == 0) return -1;
        
        cl_platform_id platform = platforms[0]; 
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
        
        if (device_idx >= num_devices) return -2; // Device not found

        context = clCreateContext(NULL, 1, &devices[device_idx], NULL, NULL, &err);
        queue = clCreateCommandQueue(context, devices[device_idx], 0, &err);
        program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, &err);
        clBuildProgram(program, 1, &devices[device_idx], NULL, NULL, NULL);
        
        k_init = clCreateKernel(program, "init_genome", &err);
        k_mutate = clCreateKernel(program, "mutate_gene", &err);
        k_eval = clCreateKernel(program, "evaluate_loss", &err);
        return 0;
    }

    __attribute__((visibility("default"))) int AllocModelAndData(int size_gb, void* host_data_void, int data_len) {
        unsigned char* host_data = (unsigned char*)host_data_void;
        cl_int err;
        size_t n_bytes = (size_t)size_gb * 1024 * 1024 * 1024;
        
        d_genome_best = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bytes, NULL, &err);
        d_genome_trial = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bytes, NULL, &err);
        d_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_len, host_data, &err);
        d_loss = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &err); 
        
        // INIT GENOME
        float seed = 0.5f;
        clSetKernelArg(k_init, 0, sizeof(cl_mem), &d_genome_best);
        clSetKernelArg(k_init, 1, sizeof(float), &seed);
        clSetKernelArg(k_init, 2, sizeof(cl_ulong), &n_bytes);
        size_t global = n_bytes;
        size_t chunk = 1024*1024*64;
        for(size_t off=0; off<global; off+=chunk) {
            size_t sz = (global-off > chunk) ? chunk : (global-off);
            clEnqueueNDRangeKernel(queue, k_init, 1, NULL, &sz, NULL, 0, NULL, NULL);
        }
        clFinish(queue);
        clEnqueueCopyBuffer(queue, d_genome_best, d_genome_trial, 0, 0, n_bytes, 0, NULL, NULL);
        clFinish(queue);
        
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    __attribute__((visibility("default"))) float StepAndEval(float mutation_rate, float seed, long n_bytes, int data_len) {
        // Mutate
        clSetKernelArg(k_mutate, 0, sizeof(cl_mem), &d_genome_trial);
        clSetKernelArg(k_mutate, 1, sizeof(cl_mem), &d_genome_best);
        clSetKernelArg(k_mutate, 2, sizeof(float), &mutation_rate);
        clSetKernelArg(k_mutate, 3, sizeof(float), &seed);
        clSetKernelArg(k_mutate, 4, sizeof(cl_ulong), &n_bytes); 
        size_t global = n_bytes;
        size_t chunk = 1024*1024*64;
        for(size_t off=0; off<global; off+=chunk) {
            size_t sz = (global-off > chunk) ? chunk : (global-off);
            clEnqueueNDRangeKernel(queue, k_mutate, 1, NULL, &sz, NULL, 0, NULL, NULL);
        }
        
        // Evaluate
        int batch_size = 1024;
        int offset = (int)((long)seed * 12345 % (data_len - batch_size - 128));
        if (offset < 0) offset = 0;
        if (offset > data_len - 2000) offset = 0;
        clSetKernelArg(k_eval, 0, sizeof(cl_mem), &d_genome_trial); 
        clSetKernelArg(k_eval, 1, sizeof(cl_mem), &d_data);
        clSetKernelArg(k_eval, 2, sizeof(cl_mem), &d_loss);
        clSetKernelArg(k_eval, 3, sizeof(int), &offset);
        clSetKernelArg(k_eval, 4, sizeof(cl_ulong), &n_bytes); 
        size_t batch_global = batch_size;
        clEnqueueNDRangeKernel(queue, k_eval, 1, NULL, &batch_global, NULL, 0, NULL, NULL);
        
        // Read Loss
        float host_loss[1024];
        clEnqueueReadBuffer(queue, d_loss, CL_TRUE, 0, batch_size * sizeof(float), host_loss, 0, NULL, NULL);
        float sum = 0.0f;
        for(int i=0; i<batch_size; i++) sum += host_loss[i];
        return sum / batch_size;
    }
    
    __attribute__((visibility("default"))) void AcceptMutation(long n_bytes) {
         clEnqueueCopyBuffer(queue, d_genome_trial, d_genome_best, 0, 0, n_bytes, 0, NULL, NULL);
         clFinish(queue);
    }
}
"""

def train_island(gpu_id):
    # Compile PER PROCESS to ensure isolation (Hack but safe)
    lib_name = f"libzeta_v10_gpu{gpu_id}_{random.randint(1000,9999)}.so"
    cpp_name = f"zeta_v10_gpu{gpu_id}.cpp"
    with open(cpp_name, "w") as f:
        f.write(cpp_source)
    
    print(f"🔨 [GPU {gpu_id}] Compiling...")
    cmd = f"g++ -shared -fPIC -O3 {cpp_name} -o {lib_name} -lOpenCL"
    subprocess.check_call(cmd, shell=True)
    
    lib = ctypes.CDLL(f"./{lib_name}")
    lib.InitGPU.argtypes = [ctypes.c_int]
    lib.AllocModelAndData.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.StepAndEval.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_long, ctypes.c_int]
    lib.StepAndEval.restype = ctypes.c_float
    lib.AcceptMutation.argtypes = [ctypes.c_long]

    print(f"🚀 [GPU {gpu_id}] Init...")
    if lib.InitGPU(gpu_id) != 0:
        print(f"❌ [GPU {gpu_id}] Failed Init. (Maybe only 1 GPU available?)")
        return

    # Load Data AGAIN per process (Memory isolation)
    with open("wikitext.txt", "rb") as f:
        text_data = f.read()
    tokens = np.frombuffer(text_data, dtype=np.uint8)
    data_ptr = tokens.ctypes.data_as(ctypes.c_void_p)

    GB = 6 # 6GB per GPU = 12GB Total VRAM Usage per card (2 buffers)
    BYTES = GB * 1024**3

    print(f"🧠 [GPU {gpu_id}] Allocating {GB*4} Billion Params (24B)...")
    if lib.AllocModelAndData(GB, data_ptr, len(tokens)) != 0:
        print(f"❌ [GPU {gpu_id}] Alloc Failed.")
        return

    start = time.time()
    gen = 0
    best_loss = 9999.0
    
    print(f"🧬 [GPU {gpu_id}] EVOLUTION STARTED.")
    
    while True:
        gen += 1
        seed = gen * 0.123 + time.time() + gpu_id*1000
        
        current_loss = lib.StepAndEval(0.005, seed, BYTES, len(tokens))
        
        if current_loss > 0.000001:
            if current_loss < best_loss:
                best_loss = current_loss
                lib.AcceptMutation(BYTES)
        
        if gen % 100 == 0:
            dt = time.time() - start
            hz = gen / dt
            print(f"🏝️ [GPU {gpu_id}] Gen {gen} | Speed: {hz:.2f} Hz | BEST LOSS: {best_loss:.6f}")

if __name__ == "__main__":
    # Launch 2 Parallel Processes
    p1 = multiprocessing.Process(target=train_island, args=(0,))
    p2 = multiprocessing.Process(target=train_island, args=(1,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
```
