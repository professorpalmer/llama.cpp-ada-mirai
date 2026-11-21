// Banded relative-position flash attention (Inkling 2), decode path: 1-32 query rows, K/V f16 / q8_0 / q4_0.
// See fattn-banded-dec.cu for the origin and credits.
#pragma once

#include "common.cuh"

// largest number of query rows the decode path takes; larger batches go to the prefill kernel (fattn-banded.cu)
#define GGML_CUDA_BANDED_DEC_MAX_Q 32

bool ggml_cuda_flash_attn_ext_banded_dec_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_banded_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
