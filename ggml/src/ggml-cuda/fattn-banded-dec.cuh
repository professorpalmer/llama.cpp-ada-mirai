// Banded relative-position flash attention (Inkling 2), decode path: 1-32 query rows, K/V f16 / q8_0 / q4_0.
// See fattn-banded-dec.cu for the origin and credits.
#pragma once

#include "common.cuh"

// largest number of query rows the decode kernel can take
#define GGML_CUDA_BANDED_DEC_MAX_Q 32

#include <cstdlib>

// Query rows up to which the decode kernel takes a banded op with K/V of the given type; more rows go to the
// tensor-core kernel of fattn-banded.cu, which uses this same function for its lower bound. Quantized K/V:
// GGML_CUDA_BANDED_DEC_MAX_Q_DEFAULT, GGML_CUDA_BANDED_DEC_MAX_Q=n overrides (1-32). The tensor-core kernel costs about
// the same from 1 to 32 rows (one 32-row block per Q head) and is faster from about 5 rows (RTX 4070, q4_0, 32k cells:
// decode kernel 288 / 563 / 1,121 us at 2 / 4 / 8 rows, tensor-core 540 us at 8), so the default becomes 4 once the
// tensor-core kernel takes quantized K/V from 5 rows. f16: 32 (the tensor-core kernel needs 16 rows). Batch-invariant
// mode: 32 for every type, so that a query gives the same bits alone and inside a verify batch (one kernel for both).
#ifndef GGML_CUDA_BANDED_DEC_MAX_Q_DEFAULT
#define GGML_CUDA_BANDED_DEC_MAX_Q_DEFAULT GGML_CUDA_BANDED_DEC_MAX_Q
#endif
static inline int64_t ggml_cuda_banded_dec_max_q(const ggml_type type_kv) {
    static const int64_t max_q_quant = [] {
        const char * e = getenv("GGML_CUDA_BANDED_DEC_MAX_Q");
        const int64_t v = e ? atoll(e) : (int64_t) GGML_CUDA_BANDED_DEC_MAX_Q_DEFAULT;
        return v < 1 ? (int64_t) 1 : (v > GGML_CUDA_BANDED_DEC_MAX_Q ? (int64_t) GGML_CUDA_BANDED_DEC_MAX_Q : v);
    }();
    if (ggml_cuda_batch_invariant() || type_kv == GGML_TYPE_F16) {
        return GGML_CUDA_BANDED_DEC_MAX_Q;
    }
    return max_q_quant;
}

bool ggml_cuda_flash_attn_ext_banded_dec_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_banded_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
