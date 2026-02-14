/**
 * @file rth_tcn_ops.h
 * @brief Custom C++ operators for llama.cpp to support Fractal TCN architectures.
 * 
 * This header defines the interface for Causal Convolutions and Fractal Gating
 * that will be integrated into the ggml/llama.cpp backend.
 */

#pragma once
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Causal 1D Convolution operator.
 * Performs a dilated convolution looking only at past time-steps.
 * 
 * @param ctx       GGML context
 * @param a         Input tensor (SequenceLength x D_in)
 * @param b         Kernel tensor (KernelSize x D_in x D_out)
 * @param dilation  Dilation factor for exponential receptive field
 * @return struct ggml_tensor* Resulting mixed sequence
 */
struct ggml_tensor * ggml_causal_conv1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   dilation);

/**
 * @brief Fractal Gating operator.
 * Implements the gated sum for fractal block expansion.
 * 
 * @param ctx       GGML context
 * @param path_a    First fractal path
 * @param path_b    Second fractal path
 * @param gate      Gate tensor (normalized)
 * @return struct ggml_tensor* Gated output
 */
struct ggml_tensor * ggml_fractal_gate(
        struct ggml_context * ctx,
        struct ggml_tensor  * path_a,
        struct ggml_tensor  * path_b,
        struct ggml_tensor  * gate);

#ifdef __cplusplus
}
#endif
