#pragma once

#include "common.cuh"

// Prefill path of ggml_escha_moe (routed trellis experts), batches of more than 8 tokens: the exllamav3 GEMV
// arithmetic of the routed GEMV path (escha-gemv.cu) with each decoded weight tile applied to up to 64 rows of its
// expert. Every row is the same bits as through the 1-token path. Returns false if it does not take the call.
bool ggml_cuda_escha_moe_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
