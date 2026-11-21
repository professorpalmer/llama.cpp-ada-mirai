// Banded relative-position flash attention (Inkling 2). Ported from Unsloth's llama.cpp mix b12588-mix-c0c3de8
// (MIT, the ggml authors and Unsloth; https://github.com/unslothai/llama.cpp), unchanged except: local warp-mask
// and shuffle helpers (our common.cuh predates ggml_cuda_shfl) and ggml_cuda_flash_attn_ext_banded_use_mma kept in
// this file (Unsloth routes a sink-free banded op to its newer generic MMA FA kernel; here it uses this file's own).
#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst);
