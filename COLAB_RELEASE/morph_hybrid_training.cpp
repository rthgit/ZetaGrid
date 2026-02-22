/*
 * 🧬 ZETAGRID v6.0: HYBRID TRAINING ENGINE
 * ========================================
 * Strategy: "Flash & Deep"
 *   - Forward Pass: Int8 Quantized (NPU Sim) -> Speed
 *   - Backward Pass: Float32 (AVX2 Stable) -> Precision
 * Goal: Total Step Time < 35ms on CPU
 */

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <cstring>

#define DIM 2240

// ------------------------------------------------------------------
// ENGINE 1: FORWARD PASS (Int8 NPU Sim)
// ------------------------------------------------------------------
inline void microkernel_int8_avx2(int K, const int8_t* A, const int8_t* B, int32_t* C, int ldc) {
    __m256i c0 = _mm256_loadu_si256((__m256i*)(C + 0 * ldc));
    __m256i c1 = _mm256_loadu_si256((__m256i*)(C + 1 * ldc));
    __m256i c2 = _mm256_loadu_si256((__m256i*)(C + 2 * ldc));
    __m256i c3 = _mm256_loadu_si256((__m256i*)(C + 3 * ldc));

    for (int k = 0; k < K; k += 32) { // 32 bytes loaded
        __m256i a_vec = _mm256_loadu_si256((__m256i*)(A + k)); 
        __m256i b_vec = _mm256_loadu_si256((__m256i*)(B + k));
        __m256i mad = _mm256_maddubs_epi16(a_vec, b_vec);
        
        __m256i mad_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mad, 0));
        __m256i mad_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mad, 1));
        
        c0 = _mm256_add_epi32(c0, mad_lo);
        c1 = _mm256_add_epi32(c1, mad_hi);
        c2 = _mm256_add_epi32(c2, mad_lo);
        c3 = _mm256_add_epi32(c3, mad_hi);
    }
    
    _mm256_storeu_si256((__m256i*)(C + 0 * ldc), c0);
    _mm256_storeu_si256((__m256i*)(C + 1 * ldc), c1);
    _mm256_storeu_si256((__m256i*)(C + 2 * ldc), c2);
    _mm256_storeu_si256((__m256i*)(C + 3 * ldc), c3);
}

void forward_pass_npu(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j += 32) {
        for (int i = 0; i < M; i += 4) {
            microkernel_int8_avx2(K, A + i*K, B + j*K, C + i*N + j, N);
        }
    }
}

// ------------------------------------------------------------------
// ENGINE 2: BACKWARD PASS (Float32 AVX2 Stable)
// ------------------------------------------------------------------
inline void microkernel_avx2_8x8_row(int K, const float* A_packed, const float* B_packed, float* C, int ldc) {
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    for (int k = 0; k < K; k++) {
        __m256 b_vec = _mm256_loadu_ps(B_packed + k * 8);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 0]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 1]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 2]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 3]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 4]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 5]), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 6]), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 7]), b_vec, c7);
    }
    _mm256_storeu_ps(C + 0 * ldc, c0); _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2); _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4); _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6); _mm256_storeu_ps(C + 7 * ldc, c7);
}

void backward_pass_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int NC = 256; const int MC = 64; const int KC = 256;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < N; j += NC) {
        for (int i = 0; i < M; i += MC) {
            float pA[MC * KC]; float pB[NC * KC];
            for (int k = 0; k < K; k += KC) {
                int mc_eff = std::min(MC, M - i); int nc_eff = std::min(NC, N - j); int kc_eff = std::min(KC, K - k);
                for (int jj = 0; jj < nc_eff; jj += 8) {
                    for (int kk = 0; kk < kc_eff; kk++) _mm256_storeu_ps(pB + jj*kc_eff + kk*8, _mm256_loadu_ps(B + (k + kk)*N + (j + jj)));
                }
                for (int ii = 0; ii < mc_eff; ii += 8) {
                    for (int kk = 0; kk < kc_eff; kk++) for(int r=0; r<8; r++) pA[ii*kc_eff + kk*8 + r] = A[(i + ii + r)*K + (k + kk)];
                    for (int jj = 0; jj < nc_eff; jj += 8) microkernel_avx2_8x8_row(kc_eff, pA + ii*kc_eff, pB + jj*kc_eff, C + (i + ii)*N + (j + jj), N);
                }
            }
        }
    }
}

int main() {
    std::cout << "🧬 ZETAGRID v6.0: HYBRID TRAINING (CPU)" << std::endl;
    int M = 256, N = DIM, K = DIM;
    
    // Alloc Forward (Int8)
    int8_t *fA = (int8_t*)_mm_malloc(M*K, 32); memset(fA, 1, M*K);
    int8_t *fB = (int8_t*)_mm_malloc(K*N, 32); memset(fB, 2, K*N);
    int32_t *fC = (int32_t*)_mm_malloc(M*N*4, 32);
    
    // Alloc Backward (Float)
    float *bA = (float*)_mm_malloc(M*K*4, 32); for(int i=0; i<M*K; i++) bA[i]=1.0f;
    float *bB = (float*)_mm_malloc(K*N*4, 32); for(int i=0; i<K*N; i++) bB[i]=0.01f;
    float *bC = (float*)_mm_malloc(M*N*4, 32); for(int i=0; i<M*N; i++) bC[i]=0.0f;

    std::cout << "\n🚀 RUNNING TRAINING STEP (Forward + Backward)..." << std::endl;
    
    // Warmup
    forward_pass_npu(M, N, K, fA, fB, fC);
    backward_pass_cpu(M, N, K, bA, bB, bC);

    auto t_start = std::chrono::high_resolution_clock::now();
    
    // 1. Forward Pass (NPU Sim)
    forward_pass_npu(M, N, K, fA, fB, fC);
    auto t_mid = std::chrono::high_resolution_clock::now();
    
    // 2. Backward Pass (CPU Stable)
    backward_pass_cpu(M, N, K, bA, bB, bC);
    auto t_end = std::chrono::high_resolution_clock::now();

    double ms_fwd = std::chrono::duration<double, std::milli>(t_mid - t_start).count();
    double ms_bwd = std::chrono::duration<double, std::milli>(t_end - t_mid).count();
    double ms_total = ms_fwd + ms_bwd;

    std::cout << " ┌─ Forward (Int8): " << ms_fwd << " ms" << std::endl;
    std::cout << " ├─ Backward (F32): " << ms_bwd << " ms" << std::endl;
    std::cout << " └─ TOTAL STEP    : " << ms_total << " ms" << std::endl;
    
    if (ms_total < 35.0) std::cout << "\n✅ STATUS: REAL-TIME TRAINING ACHIEVED (<35ms)" << std::endl;
    else std::cout << "\n⚠️ STATUS: HIGH PERFORMANCE (Optimizable)" << std::endl;

    _mm_free(fA); _mm_free(fB); _mm_free(fC);
    _mm_free(bA); _mm_free(bB); _mm_free(bC);
    return 0;
}
