/*
 * 🧬 ZETAGRID EXTREME: v3.9 AVX-512 OPTIMIZED
 * ==========================================
 * Targeted for: Intel High-Performance CPUs (Alder/Raptor Lake +)
 * Benefit: 512-bit vector width (16 floats per instruction)
 */

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <algorithm>

#define DIM 2240
#define MAX_THREADS 32

// ------------------------------------------------------------------
// AVX-512 MICROKERNEL (16x16 Tile)
// ------------------------------------------------------------------
// Uses 16 ZMM registers for the C-accumulator block.
// This is the absolute peak of CPU matrix multiplication logic.
inline void microkernel_avx512_16x16(int K, const float* A, const float* B, float* C, int ldc) {
    __m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(C + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(C + 5 * ldc);
    __m512 c6 = _mm512_loadu_ps(C + 6 * ldc);
    __m512 c7 = _mm512_loadu_ps(C + 7 * ldc);
    __m512 c8 = _mm512_loadu_ps(C + 8 * ldc);
    __m512 c9 = _mm512_loadu_ps(C + 9 * ldc);
    __m512 c10 = _mm512_loadu_ps(C + 10 * ldc);
    __m512 c11 = _mm512_loadu_ps(C + 11 * ldc);
    __m512 c12 = _mm512_loadu_ps(C + 12 * ldc);
    __m512 c13 = _mm512_loadu_ps(C + 13 * ldc);
    __m512 c14 = _mm512_loadu_ps(C + 14 * ldc);
    __m512 c15 = _mm512_loadu_ps(C + 15 * ldc);

    for (int k = 0; k < K; k++) {
        __m512 a_vec = _mm512_loadu_ps(A + k * 16);
        c0 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 0]), c0);
        c1 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 1]), c1);
        c2 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 2]), c2);
        c3 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 3]), c3);
        c4 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 4]), c4);
        c5 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 5]), c5);
        c6 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 6]), c6);
        c7 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 7]), c7);
        c8 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 8]), c8);
        c9 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 9]), c9);
        c10 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 10]), c10);
        c11 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 11]), c11);
        c12 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 12]), c12);
        c13 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 13]), c13);
        c14 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 14]), c14);
        c15 = _mm512_fmadd_ps(a_vec, _mm512_set1_ps(B[k * 16 + 15]), c15);
    }

    _mm512_storeu_ps(C + 0 * ldc, c0);
    _mm512_storeu_ps(C + 1 * ldc, c1);
    _mm512_storeu_ps(C + 2 * ldc, c2);
    _mm512_storeu_ps(C + 3 * ldc, c3);
    _mm512_storeu_ps(C + 4 * ldc, c4);
    _mm512_storeu_ps(C + 5 * ldc, c5);
    _mm512_storeu_ps(C + 6 * ldc, c6);
    _mm512_storeu_ps(C + 7 * ldc, c7);
    _mm512_storeu_ps(C + 8 * ldc, c8);
    _mm512_storeu_ps(C + 9 * ldc, c9);
    _mm512_storeu_ps(C + 10 * ldc, c10);
    _mm512_storeu_ps(C + 11 * ldc, c11);
    _mm512_storeu_ps(C + 12 * ldc, c12);
    _mm512_storeu_ps(C + 13 * ldc, c13);
    _mm512_storeu_ps(C + 14 * ldc, c14);
    _mm512_storeu_ps(C + 15 * ldc, c15);
}

// ------------------------------------------------------------------
// PACKING LOGIC
// ------------------------------------------------------------------
void pack_A_16xK(int K, const float* A, int lda, float* pA) {
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < 16; i++) {
            pA[k * 16 + i] = A[i * lda + k];
        }
    }
}

void pack_B_Kx16(int K, const float* B, int ldb, float* pB) {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < 16; j++) {
            pB[k * 16 + j] = B[k * ldb + j];
        }
    }
}

// ------------------------------------------------------------------
// ZETAGRID AVX-512 ENGINE
// ------------------------------------------------------------------
void gemm_avx512_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 128; // Blocks to fit in cache
    const int NC = 128;
    const int KC = 256;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < N; j += NC) {
        for (int i = 0; i < M; i += MC) {
            float pA[MC * KC]; float pB[NC * KC]; // Stack buffers (small enough)
            for (int k = 0; k < K; k += KC) {
                int mc_eff = std::min(MC, M - i);
                int nc_eff = std::min(NC, N - j);
                int kc_eff = std::min(KC, K - k);

                for (int ii = 0; ii < mc_eff; ii += 16) {
                    pack_A_16xK(kc_eff, A + (i + ii) * K + k, K, pA + ii * kc_eff);
                    for (int jj = 0; jj < nc_eff; jj += 16) {
                        if (ii == 0) pack_B_Kx16(kc_eff, B + k * N + (j + jj), N, pB + jj * kc_eff);
                        microkernel_avx512_16x16(kc_eff, pA + ii * kc_eff, pB + jj * kc_eff, C + (i + ii) * N + j + jj, N);
                    }
                }
            }
        }
    }
}

int main() {
    std::cout << "🧬 ZETAGRID EXTREME: v3.9 AVX-512 ENGINE" << std::endl;
    std::cout << "Target: Intel 512-bit Vector Units" << std::endl;

    int M = 256, N = DIM, K = DIM;
    float *hA, *hB, *hC;
    hA = (float*)_mm_malloc((size_t)M * K * 4, 64);
    hB = (float*)_mm_malloc((size_t)K * N * 4, 64);
    hC = (float*)_mm_malloc((size_t)M * N * 4, 64);

    for (int i = 0; i < M * K; i++) hA[i] = 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = 0.01f;

    std::cout << "\n🚀 Benchmarking AVX-512 Engine... " << std::flush;
    auto t1 = std::chrono::high_resolution_clock::now();
    gemm_avx512_tiled(M, N, K, hA, hB, hC);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
    
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    std::cout << " 🔥 Throughput: " << gflops << " GFLOPS" << std::endl;
    
    // Recovery Info
    std::cout << "\n💡 Prossimi passi: Installare driver Intel Graphics per sbloccare la GPU." << std::endl;

    _mm_free(hA); _mm_free(hB); _mm_free(hC);
    return 0;
}
