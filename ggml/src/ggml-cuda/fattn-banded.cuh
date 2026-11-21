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
// The K / V type pairs the banded kernels take (prefill and decode use this one list): the same type for K and V, f16 /
// q8_0 / q4_0, or K q8_0 with V q4_0 (the opt-in "more context in VRAM" cache).
static inline bool fattn_banded_kv_pair_ok(const ggml_type tk, const ggml_type tv) {
    return tk == tv ? (tk == GGML_TYPE_F16 || tk == GGML_TYPE_Q8_0 || tk == GGML_TYPE_Q4_0)
                    : (tk == GGML_TYPE_Q8_0 && tv == GGML_TYPE_Q4_0);
}

// The band bias from its factors (ggml_flash_attn_ext_banded_set_rel_factors), one definition for every banded kernel:
// sum over c = 0 .. d_rel-1 of table[c, d] * rcur_row[c] as one fp32 fma chain in c order from 0.0f. table: the F32 or
// F16 table [d_rel, rel_extent] (t_nb1 bytes per distance d), an F16 value read as its exact fp32 value; rcur_row: the
// d_rel factors of one (head, query).
static __device__ __forceinline__ float fattn_banded_rel_bias(const char * table, const int table_type, const int64_t t_nb1,
        const float * rcur_row, const int d_rel, const int d) {
    const char * t = table + d*t_nb1;
    float sum = 0.0f;
    if (table_type == GGML_TYPE_F16) {
        for (int c = 0; c < d_rel; ++c) {
            sum = fmaf(__half2float(((const half *) t)[c]), rcur_row[c], sum);
        }
    } else {
        for (int c = 0; c < d_rel; ++c) {
            sum = fmaf(((const float *) t)[c], rcur_row[c], sum);
        }
    }
    return sum;
}

// The same chain on a table row already read into registers as fp32 (DR = d_rel known at compile time): the same
// operands in the same order as fattn_banded_rel_bias, so the same bits.
template <int DR>
static __device__ __forceinline__ float fattn_banded_rel_bias_reg(const float (&t)[DR], const float * rcur_row) {
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < DR; ++c) {
        sum = fmaf(t[c], rcur_row[c], sum);
    }
    return sum;
}

static __device__ __forceinline__ bool fattn_banded_visible(const int pq, const int pk, const int window, int & dist) {
    dist = pq - pk;
    return pk >= 0 && (window <= 0 || (dist >= 0 && dist < window));
}
