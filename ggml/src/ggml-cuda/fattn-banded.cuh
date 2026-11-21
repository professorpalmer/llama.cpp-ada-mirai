// Banded relative-position flash attention (Inkling 2). Ported from Unsloth's llama.cpp mix b12588-mix-c0c3de8
// (MIT, the ggml authors and Unsloth; https://github.com/unslothai/llama.cpp), unchanged except: local warp-mask
// and shuffle helpers (our common.cuh predates ggml_cuda_shfl) and ggml_cuda_flash_attn_ext_banded_use_mma kept in
// this file (Unsloth routes a sink-free banded op to its newer generic MMA FA kernel; here it uses this file's own).
#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst);

// The distance of a K/V cell from a query and whether the cell is visible, for every banded kernel (prefill, warp and
// decode kernels use this one definition). With token positions (ggml_flash_attn_ext_banded_set_pos): pq = pos_q[query],
// pk = pos_kv[cell]; else the tail rule: pq = iq + (n_kv - n_q), pk = ik. An empty cell (pk < 0) is never visible.
// window <= 0: no implicit window (a mask tensor decides). The band bias rel[dist] applies to a visible cell iff
// 0 <= dist < rel_extent. int32 throughout (registers in the inner loops): positions are I32 by the API and the
// tail rule needs n_kv <= INT32_MAX/2, which _supported() of the kernels checks.
static __device__ __forceinline__ bool fattn_banded_visible(const int pq, const int pk, const int window, int & dist) {
    dist = pq - pk;
    return pk >= 0 && (window <= 0 || (dist >= 0 && dist < window));
}
