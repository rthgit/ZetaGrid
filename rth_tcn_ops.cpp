/**
 * @file rth_tcn_ops.cpp
 * @brief Implementation of custom TCN operators for llama.cpp.
 */

#include "ggml.h"
#include <cmath>
#include <algorithm>

// --- CAUSAL CONV1D (CPU REFERENCE) ---
// This is the reference implementation for the causal convolution.
// In production, this would be optimized with AVX/NEON or CUDA.

void ggml_compute_forward_causal_conv1d(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0, // Input: [N, C, T]
    const struct ggml_tensor * src1, // Kernel: [C, 1, K]
    struct ggml_tensor * dst) {
    
    const int nc = src0->ne[1]; // Channels
    const int nt = src0->ne[0]; // Time steps
    const int nk = src1->ne[0]; // Kernel size
    const int dilation = params->ith; // We use params->ith to pass dilation for simplicity in prototype

    // Causal Convolution Logic:
    // Output[c, t] = sum_{k=0}^{nk-1} Input[c, t - k * dilation] * Kernel[c, k]
    // If t - k * dilation < 0, Input is 0 (causal padding).

    for (int c = 0; c < nc; c++) {
        for (int t = 0; t < nt; t++) {
            float sum = 0.0f;
            for (int k = 0; k < nk; k++) {
                int src_t = t - k * dilation;
                if (src_t >= 0) {
                    float val = ((float*)src0->data)[c * nt + src_t];
                    float weight = ((float*)src1->data)[c * nk + k];
                    sum += val * weight;
                }
            }
            ((float*)dst->data)[c * nt + t] = sum;
        }
    }
}

// --- FRACTAL GATE (CPU REFERENCE) ---
// Y = SiLU(path_a) * Sigmoid(path_b)
// Note: In ZetaGrid, gating is applied after convolution.

void ggml_compute_forward_fractal_gate(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0, // Path A (Mixed)
    const struct ggml_tensor * src1, // Path B (Gate)
    struct ggml_tensor * dst) {
    
    const int ne = ggml_nelements(src0);
    const float * a = (const float *)src0->data;
    const float * b = (const float *)src1->data;
    float * out = (float *)dst->data;

    for (int i = 0; i < ne; i++) {
        // SiLU(x) = x * sigmoid(x)
        float silu_a = a[i] * (1.0f / (1.0f + std::exp(-a[i])));
        // Sigmoid(g)
        float sig_b = 1.0f / (1.0f + std::exp(-b[i]));
        out[i] = silu_a * sig_b;
    }
}
