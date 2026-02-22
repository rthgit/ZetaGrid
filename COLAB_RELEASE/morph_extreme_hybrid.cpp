/*
 * 🧬 ZETAGRID EXTREME: v4.1 HYBRID-SAFE ENGINE (STABLE)
 * ====================================================
 * Architecture: Optimized AVX2 (Row-Accumulation)
 * Strategy: 8x8 Register Blocking (Safe for M=256)
 * Fix: Removed Out-Of-Bounds writes.
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
// EXTREME AVX2 MICROKERNEL (8x8 Tile)
// ------------------------------------------------------------------
// Uses 8 YMM registers (c0..c7) to accumulate 8 rows x 8 cols of C.
// Regs:
// c0 = C[row0][0..7]
// ...
// c7 = C[row7][0..7]
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
        __m256 b_vec = _mm256_loadu_ps(B_packed + k * 8); // Load Row of B

        // Broadcast Scalars from A (Panel) and FMA
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 0]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 1]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 2]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 3]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 4]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 5]), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 6]), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[k * 8 + 7]), b_vec, c7);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}


// ------------------------------------------------------------------
// SAFE STACK-PACKING ENGINE
// ------------------------------------------------------------------
void gemm_hybrid_stable(int M, int N, int K, const float* A, const float* B, float* C) {
    const int NC = 256; // Smaller blocks for stability
    const int MC = 64;  
    const int KC = 256;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < N; j += NC) {
        for (int i = 0; i < M; i += MC) {
            float pA[MC * KC]; float pB[NC * KC];

            for (int k = 0; k < K; k += KC) {
                int mc_eff = std::min(MC, M - i);
                int nc_eff = std::min(NC, N - j);
                int kc_eff = std::min(KC, K - k);

                // --- PACKING B (Panel K x N) ---
                // We need pB[k][col] to be contiguous row vectors
                for (int jj = 0; jj < nc_eff; jj += 8) {
                    if (i == 0) { // Only first thread block packs B (optimization)
                         // But since we are parallel over i, this is risky.
                         // SIMPLER: Pack Locally.
                         for (int kk = 0; kk < kc_eff; kk++) {
                             _mm256_storeu_ps(pB + jj*kc_eff + kk*8, _mm256_loadu_ps(B + (k + kk)*N + (j + jj)));
                         }
                    } else {
                         // Redundant Pack for simplicity (or use barrier/shared)
                         // Local stack packing is fast enough.
                         for (int kk = 0; kk < kc_eff; kk++) {
                             _mm256_storeu_ps(pB + jj*kc_eff + kk*8, _mm256_loadu_ps(B + (k + kk)*N + (j + jj)));
                         }
                    }
                }

                // --- PACKING A (Panel M x K) ---
                // We need pA[row][k] such that we can read pA[k] easily?
                // Actually we read scalar pA[k*8 + row].
                // So we transpose 8xK block to Kx8.
                for (int ii = 0; ii < mc_eff; ii += 8) {
                    // Transpose Packing A: 8 rows -> Kx8 block
                    for (int kk = 0; kk < kc_eff; kk++) {
                        for(int r=0; r<8; r++) {
                            pA[ii*kc_eff + kk*8 + r] = A[(i + ii + r)*K + (k + kk)];
                        }
                    }

                    // --- COMPUTE ---
                    for (int jj = 0; jj < nc_eff; jj += 8) {
                         // pA ptr: start of block ii
                         // pB ptr: start of block jj.
                         // But pB layout above: jj outer loop. pB + jj*kc_eff.
                         // Inside, kk*8.
                         // So pB pointer: pB + jj*kc_eff.
                         
                         microkernel_avx2_8x8_row(kc_eff, 
                                                pA + ii*kc_eff, 
                                                pB + jj*kc_eff, 
                                                C + (i + ii)*N + (j + jj), 
                                                N);
                    }
                }
            }
        }
    }
}

int main() {
    std::cout << "🧬 ZETAGRID v4.1: STABLE AVX2 ENGINE" << std::endl;
    std::cout << "Strategy: 8x8 RegBlock | Safe Memory" << std::endl;

    int M = 256, N = DIM, K = DIM;
    float *hA = (float*)_mm_malloc((size_t)M * K * 4, 32);
    float *hB = (float*)_mm_malloc((size_t)K * N * 4, 32);
    float *hC = (float*)_mm_malloc((size_t)M * N * 4, 32);

    for (int i = 0; i < M * K; i++) hA[i] = 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = 0.01f;
    // Init C to zero?? The kernel accumulates!
    // But since we tile K, we accumulate.
    // However, the outer loops (j, i) are separate blocks of C.
    // Inside, k += KC. We load C, add, store C.
    // We must init C to 0.
    for (int i = 0; i < M * N; i++) hC[i] = 0.0f;

    std::cout << "\n🚀 Benchmarking Stable Engine... " << std::flush;
    auto t1 = std::chrono::high_resolution_clock::now();
    gemm_hybrid_stable(M, N, K, hA, hB, hC);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
    std::cout << " 🔥 Throughput: " << (2.0 * M * N * K) / (ms * 1e6) << " GFLOPS" << std::endl;

    _mm_free(hA); _mm_free(hB); _mm_free(hC);
    return 0;
}
