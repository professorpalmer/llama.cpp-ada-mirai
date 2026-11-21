// Banded relative-position flash attention (Inkling 2). Ported from Unsloth's llama.cpp mix b12588-mix-c0c3de8
// (MIT, the ggml authors and Unsloth; https://github.com/unslothai/llama.cpp), unchanged except: local warp-mask
// and shuffle helpers (our common.cuh predates ggml_cuda_shfl) and ggml_cuda_flash_attn_ext_banded_use_mma kept in
// this file (Unsloth routes a sink-free banded op to its newer generic MMA FA kernel; here it uses this file's own).
//
// Our changes for prefill (ubatch > 32; Inkling 2a): the tensor-core kernel reads q4_0 / q8_0 K/V natively with the
// quantized tile loaders of our tensor-core FA kernel (fattn-mma-f16.cuh: fetch into registers, dequantize into the
// same f16 shared tile, bit-identical to to_fp16), with the next tiles fetched under the tensor-core work. Unsloth's
// full-cache f16 copy of a quantized K/V (n_kv*D*n_head_kv*2 bytes each, 1 GiB per op at 262k) is then not needed
// (it stays behind GGML_CUDA_BANDED_F16_COPY=1, for checks). The entry stages host rows of a tiered K/V cache into
// VRAM as ggml_cuda_flash_attn_ext does. Every prefill op (with or without sinks) takes the tensor-core kernel, so the
// ggml_cuda_flash_attn_ext_banded_use_mma stub is gone.
#include "common.cuh"
#include "convert.cuh"
#include "fattn-banded.cuh"
#include "fattn-banded-dec.cuh"
#include "fattn.cuh"
#include "fattn-mma-f16.cuh"

#include <algorithm>
#include <cstdint>

static __device__ __forceinline__ float fattn_banded_load(
        const char * ptr, const int type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16: {
            // Read BF16 as raw bits so this kernel needs no native BF16 support; all math stays FP32.
            const uint32_t bits = uint32_t(*(const uint16_t *) ptr) << 16;
            return __uint_as_float(bits);
        }
        default:
            return 0.0f;
    }
}

template<int D, int WARPS_PER_BLOCK>
static __global__ void flash_attn_ext_banded_f32(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ dst,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_batch,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        int32_t window,
        const int32_t * __restrict__ pos_q,
        const int32_t * __restrict__ pos_kv,
        int64_t pos_q_ne1,
        int64_t pos_kv_ne1) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    static_assert(D % WARP_SIZE == 0, "head dimension must be divisible by warp size");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t iq = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (iq >= n_q || ih >= n_head_q || ib >= n_batch) {
        return;
    }

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;

    // token positions (ggml_flash_attn_ext_banded_set_pos): the distance from them, an empty cell (-1) never visible
    const int32_t * pkv = pos_kv ? pos_kv + (ib % pos_kv_ne1)*n_kv : nullptr;
    const int64_t   pq  = pos_kv ? pos_q[(ib % pos_q_ne1)*n_q + iq] : 0;

    for (int64_t ik = 0; ik < n_kv; ++ik) {
        const int64_t dist = pkv ? (pkv[ik] < 0 ? -1 : pq - pkv[ik]) : iq + q_offset - ik;
        if (window > 0) {
            // implicit mask (ggml_flash_attn_ext_banded_set_window): visible iff 0 <= dist < window
            if (dist < 0 || dist >= window) {
                continue;
            }
        }
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = dist;
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = __shfl_sync(0xffffffffu, score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    // attention sink: an extra logit with a zero value
    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + ((ib*n_q + iq)*n_head_q + ih)*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        dst_row[d] = out[j]*inv_sum;
    }
}

template<int D, int NW>
static __global__ void flash_attn_ext_banded_f32_split(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ part,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        int32_t window,
        const int32_t * __restrict__ pos_q,
        const int32_t * __restrict__ pos_kv,
        int64_t pos_q_ne1,
        int64_t pos_kv_ne1) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t ic = blockIdx.x;
    const int64_t ih = blockIdx.y;
    const int64_t iq = int64_t(blockIdx.z) % n_q;
    const int64_t ib = int64_t(blockIdx.z) / n_q;

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;
    const int64_t ik0 = ic*chunk;
    const int64_t ik1 = ik0 + chunk < n_kv ? ik0 + chunk : n_kv;

    // token positions (ggml_flash_attn_ext_banded_set_pos): the distance from them, an empty cell (-1) never visible
    const int32_t * pkv = pos_kv ? pos_kv + (ib % pos_kv_ne1)*n_kv : nullptr;
    const int64_t   pq  = pos_kv ? pos_q[(ib % pos_q_ne1)*n_q + iq] : 0;

    for (int64_t ik = ik0 + warp; ik < ik1; ik += NW) {
        const int64_t dist = pkv ? (pkv[ik] < 0 ? -1 : pq - pkv[ik]) : iq + q_offset - ik;
        if (window > 0) {
            // implicit mask (ggml_flash_attn_ext_banded_set_window): visible iff 0 <= dist < window
            if (dist < 0 || dist >= window) {
                continue;
            }
        }
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = dist;
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = __shfl_sync(0xffffffffu, score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    __shared__ float s_max[NW];
    __shared__ float s_sum[NW];
    __shared__ float s_out[NW][D];
    if (lane == 0) {
        s_max[warp] = row_max;
        s_sum[warp] = row_sum;
    }
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        s_out[warp][lane + j*WARP_SIZE] = out[j];
    }
    __syncthreads();

    if (warp == 0) {
        float m = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            m = fmaxf(m, s_max[w]);
        }
        float acc[values_per_lane];
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            acc[j] = 0.0f;
        }
        float sum = 0.0f;
        if (m != -INFINITY) {
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                if (s_max[w] == -INFINITY) {
                    continue;
                }
                const float f = expf(s_max[w] - m);
                sum += s_sum[w]*f;
#pragma unroll
                for (int j = 0; j < values_per_lane; ++j) {
                    acc[j] += s_out[w][lane + j*WARP_SIZE]*f;
                }
            }
        }
        float * p = part + ((((ib*n_q + iq)*n_head_q + ih)*n_chunks) + ic)*(D + 2);
        if (lane == 0) {
            p[0] = m;
            p[1] = sum;
        }
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            p[2 + lane + j*WARP_SIZE] = acc[j];
        }
    }
}

template<int D>
static __global__ void flash_attn_ext_banded_f32_merge(
        const float * __restrict__ part,
        float       * __restrict__ dst,
        int64_t n_rows,
        int64_t n_head_q,
        int64_t n_q,
        int64_t n_chunks,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2) {
    constexpr int values_per_lane = D / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = int64_t(blockIdx.x)*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const int64_t ih = row % n_head_q;
    const int64_t iq = (row / n_head_q) % n_q;
    const int64_t ib = row / (n_head_q*n_q);
    const float * p = part + row*n_chunks*(D + 2);

    float m = -INFINITY;
    for (int64_t c = 0; c < n_chunks; ++c) {
        m = fmaxf(m, p[c*(D + 2)]);
    }
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        out[j] = 0.0f;
    }
    float row_sum = 0.0f;
    if (m != -INFINITY) {
        for (int64_t c = 0; c < n_chunks; ++c) {
            const float * pc = p + c*(D + 2);
            if (pc[0] == -INFINITY) {
                continue;
            }
            const float f = expf(pc[0] - m);
            row_sum += pc[1]*f;
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] += pc[2 + lane + j*WARP_SIZE]*f;
            }
        }
    }
    float row_max = m;

    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + row*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        dst_row[lane + j*WARP_SIZE] = out[j]*inv_sum;
    }
}

#if defined(TURING_MMA_AVAILABLE)
#define FATTN_BANDED_MMA_AVAILABLE
#endif

static constexpr int fattn_banded_mma_bk = 64;

#ifdef FATTN_BANDED_MMA_AVAILABLE
static __device__ __forceinline__ void fattn_banded_mma_16816(
        float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a2), "r"(a3), "r"(b1));
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4_trans(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ uint32_t fattn_banded_pack_h2(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

template<int D, int NW>
static __device__ __forceinline__ void fattn_banded_mma_load_tile(
        half * __restrict__ tile, const char * __restrict__ src, const uint64_t nb1, const int64_t kv0, const int64_t n_kv) {
    constexpr int stride   = D + 8;
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
#pragma unroll
    for (int c0 = 0; c0 < fattn_banded_mma_bk*chunks; c0 += nthreads) {
        const int c   = c0 + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        half * dst = tile + row*stride + col;
        if (kv0 + row < n_kv) {
            const char * s = src + uint64_t(kv0 + row)*nb1 + uint64_t(col)*sizeof(half);
#ifdef CP_ASYNC_AVAILABLE
            const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(d), "l"(s));
#else
            *(int4 *) dst = *(const int4 *) s;
#endif // CP_ASYNC_AVAILABLE
        } else {
            *(int4 *) dst = make_int4(0, 0, 0, 0);
        }
    }
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template<int n>
static __device__ __forceinline__ void fattn_banded_mma_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}
#endif // FATTN_BANDED_MMA_AVAILABLE

static __global__ void flash_attn_ext_banded_mma_tile_flags(
        const char * __restrict__ mask,
        uint8_t    * __restrict__ flags,
        int64_t n_q,
        int64_t n_kv,
        int bq,
        int64_t mask_ne2,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3) {
    const int64_t kt   = blockIdx.x;
    const int64_t qt   = blockIdx.y;
    const int64_t z    = blockIdx.z;
    const int64_t z2   = z % mask_ne2;
    const int64_t z3   = z / mask_ne2;
    const char  * mrow = mask + z2*m_nb2 + z3*m_nb3;

    bool any_visible = false;
    bool any_nonzero = false;
    for (int e = threadIdx.x; e < bq*fattn_banded_mma_bk; e += blockDim.x) {
        const int64_t iq = qt*bq + e / fattn_banded_mma_bk;
        const int64_t ik = kt*fattn_banded_mma_bk + e % fattn_banded_mma_bk;
        if (iq < n_q && ik < n_kv) {
            const float m = __half2float(*(const half *) (mrow + uint64_t(iq)*m_nb1 + uint64_t(ik)*sizeof(half)));
            any_visible |= m != -INFINITY;
            any_nonzero |= m != 0.0f;
        }
    }
    any_visible = __syncthreads_or(any_visible);
    any_nonzero = __syncthreads_or(any_nonzero);
    if (threadIdx.x == 0) {
        const int64_t n_kt = gridDim.x;
        const int64_t n_qt = gridDim.y;
        flags[(z*n_qt + qt)*n_kt + kt] = !any_visible ? 0 : (any_nonzero ? 2 : 1);
    }
}

// type_KV: F16 reads the K/V tiles with cp.async (Unsloth's kernel); Q4_0 / Q8_0 read the quantized cache directly
// (our change, see the file header).
template<int D, int NW, ggml_type type_KV>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void flash_attn_ext_banded_mma(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const uint8_t * __restrict__ flags,
        const char * __restrict__ rel,
        const float * __restrict__ sinks,
        float      * __restrict__ dst,
        float      * __restrict__ part,
        float scale,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t tiles_per_chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        int64_t sinks_s1,
        int64_t sinks_s2,
        int32_t window,
        const int32_t * __restrict__ pos_q,
        const int32_t * __restrict__ pos_kv,
        int64_t pos_q_ne1,
        int64_t pos_kv_ne1) {
#ifdef FATTN_BANDED_MMA_AVAILABLE
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    static_assert(type_KV == GGML_TYPE_F16 || type_KV == GGML_TYPE_Q4_0 || type_KV == GGML_TYPE_Q8_0, "unsupported K/V type");
    constexpr int BK     = fattn_banded_mma_bk;
    constexpr int BQ     = NW*16;
    constexpr int stride = D + 8; // padding keeps ldmatrix bank-conflict free
    constexpr int ND     = D/8;
    constexpr int NK     = D/16;

    // quantized K/V: block pairs (64 weights) per thread per tile, and 32-bit words per pair (fattn-mma-f16.cuh)
    constexpr bool kv_q    = type_KV != GGML_TYPE_F16;
    constexpr int  q_slots = kv_q ? flash_attn_ext_f16_pair_slots<type_KV, NW, BK, D>() : 1;
    constexpr int  q_words = kv_q ? flash_attn_ext_f16_pair_words<type_KV>() : 1;

    __shared__ __align__(16) half K_s[BK*stride];
    __shared__ __align__(16) half V_s[BK*stride];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int64_t n_qt = (n_q + BQ - 1)/BQ;
    const int64_t qt   = int64_t(blockIdx.x) / n_chunks;
    const int64_t ic   = int64_t(blockIdx.x) % n_chunks;
    const int64_t ih   = blockIdx.y;
    const int64_t ib   = blockIdx.z;
    const int64_t ih_kv = ih / (n_head_q / n_head_kv);

    const int64_t n_kt     = (n_kv + BK - 1)/BK;
    const int64_t q_offset = n_kv - n_q;

    // implicit mask (window > 0, ggml_flash_attn_ext_banded_set_window): the block's queries [iq_b0, iq_b1] see the
    // cells [iq_b0 + q_offset - window + 1, iq_b1 + q_offset]; K/V tiles outside that range are not visited
    const int64_t iq_b0 = qt*BQ;
    const int64_t iq_b1 = (qt*BQ + BQ < n_q ? qt*BQ + BQ : n_q) - 1;
    // token positions (ggml_flash_attn_ext_banded_set_pos): the cells are in any order, every tile is visited and
    // each cell is tested by its position
    const int32_t * pkv = pos_kv ? pos_kv + (ib % pos_kv_ne1)*n_kv : nullptr;
    int64_t kt_lo = 0;
    int64_t kt_hi = n_kt;
    if (window > 0 && !pkv) {
        const int64_t c_lo = iq_b0 + q_offset - int64_t(window) + 1;
        const int64_t c_hi = iq_b1 + q_offset;
        kt_lo = c_lo > 0 ? c_lo/BK : 0;
        kt_hi = c_hi < 0 ? 0 : (c_hi/BK + 1 < n_kt ? c_hi/BK + 1 : n_kt);
    }
    const int64_t kt_c0    = ic*tiles_per_chunk;
    const int64_t kt_c1    = kt_c0 + tiles_per_chunk < n_kt ? kt_c0 + tiles_per_chunk : n_kt;
    const int64_t kt_begin = kt_c0 > kt_lo ? kt_c0 : kt_lo;
    const int64_t kt_end   = kt_c1 < kt_hi ? kt_c1 : kt_hi;

    const int64_t iq_w = qt*BQ + warp*16;
    const int64_t iq_r[2] = {iq_w + g, iq_w + g + 8};
    // the positions of this lane's two query rows (positions only)
    int64_t pq_r[2] = {0, 0};
    if (pkv) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            pq_r[r] = iq_r[r] < n_q ? pos_q[(ib % pos_q_ne1)*n_q + iq_r[r]] : 0;
        }
    }

    const char * k_base = k + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
    const char * v_base = v + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
    const char * m_base = mask ? mask + uint64_t(ih % mask_ne2)*m_nb2 + uint64_t(ib % mask_ne3)*m_nb3 : nullptr;
    const uint8_t * f_base = flags ? flags + ((ih % mask_ne2) + mask_ne2*(ib % mask_ne3))*n_qt*n_kt + qt*n_kt : nullptr;
    const char * r_base = rel + uint64_t(ih)*r_nb1 + uint64_t(ib % rel_ne3)*r_nb3;

    uint32_t Q_a[NK][4];
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float lo0 = 0.0f, hi0 = 0.0f, lo1 = 0.0f, hi1 = 0.0f;
            if (iq_r[r] < n_q) {
                const float * qr = (const float *) (q + uint64_t(iq_r[r])*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3);
                lo0 = qr[kk*16 + 2*t + 0];
                hi0 = qr[kk*16 + 2*t + 1];
                lo1 = qr[kk*16 + 2*t + 8];
                hi1 = qr[kk*16 + 2*t + 9];
            }
            Q_a[kk][r + 0] = fattn_banded_pack_h2(lo0, hi0);
            Q_a[kk][r + 2] = fattn_banded_pack_h2(lo1, hi1);
        }
    }

    float O[ND][4];
#pragma unroll
    for (int n = 0; n < ND; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            O[n][e] = 0.0f;
        }
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    uint32_t P_a[BK/8][2];

    auto next_tile = [&](int64_t kt) {
        if (f_base) {
            while (kt < kt_end && f_base[kt] == 0) {
                ++kt;
            }
        }
        return kt;
    };

    // 0: no visible cell (skipped by next_tile), 1: all visible, 2: read the mask tensor, 3: implicit window per cell,
    // 4: token positions per cell
    auto tile_flag = [&](const int64_t kt) {
        if (f_base) {
            return int(f_base[kt]);
        }
        if (pkv) {
            return 4;
        }
        if (window <= 0) {
            return 1;
        }
        const int64_t d_min = iq_b0 + q_offset - (kt*BK + BK - 1);
        const int64_t d_max = iq_b1 + q_offset - kt*BK;
        return d_min >= 0 && d_max < window ? 1 : 3;
    };

    // S = Q K^T of the tile in K_s (cells kv0 ...), band bias, mask, online softmax (fp32); P to f16 in P_a, O rescaled
    auto score_tile = [&](const int64_t kv0, const int flag) {
        float S[BK/8][4];
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                S[j][e] = 0.0f;
            }
        }
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
            for (int j = 0; j < BK/8; j += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4(b, K_s + ((j + mi/2)*8 + lane % 8)*stride + kk*16 + (mi % 2)*8);
                fattn_banded_mma_16816(S[j + 0], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[0], b[1]);
                fattn_banded_mma_16816(S[j + 1], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[2], b[3]);
            }
        }

        const int64_t dist_hi = iq_w + 15 + q_offset - kv0;
        const int64_t dist_lo = iq_w + q_offset - (kv0 + BK - 1);
        const bool in_band = dist_hi >= 0 && dist_lo < rel_extent;

#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int     r  = e / 2;
                const int64_t ik = kv0 + j*8 + 2*t + (e % 2);
                float s = S[j][e]*scale;
                if (ik >= n_kv) {
                    s = -INFINITY;
                } else if (flag == 4) {
                    if (iq_r[r] < n_q) {
                        const int32_t pk   = pkv[ik];
                        const int64_t dist = pq_r[r] - pk;
                        if (pk < 0 || dist < 0 || dist >= window) {
                            s = -INFINITY;
                        } else if (dist < rel_extent) {
                            s += fattn_banded_load(r_base + uint64_t(dist)*r_nb0 + uint64_t(iq_r[r])*r_nb2, type_rel);
                        }
                    }
                } else if (iq_r[r] < n_q) {
                    if (in_band) {
                        const int64_t dist = iq_r[r] + q_offset - ik;
                        if (dist >= 0 && dist < rel_extent) {
                            s += fattn_banded_load(r_base + uint64_t(dist)*r_nb0 + uint64_t(iq_r[r])*r_nb2, type_rel);
                        }
                    }
                    if (flag == 2) {
                        s += __half2float(*(const half *) (m_base + uint64_t(iq_r[r])*m_nb1 + uint64_t(ik)*sizeof(half)));
                    } else if (flag == 3) {
                        const int64_t dist = iq_r[r] + q_offset - ik;
                        if (dist < 0 || dist >= window) {
                            s = -INFINITY;
                        }
                    }
                }
                S[j][e] = s;
            }
        }

        float alpha[2];
        float msub[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float m = row_max[r];
#pragma unroll
            for (int j = 0; j < BK/8; ++j) {
                m = fmaxf(m, fmaxf(S[j][2*r + 0], S[j][2*r + 1]));
            }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1, WARP_SIZE));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2, WARP_SIZE));
            msub[r]    = m == -INFINITY ? 0.0f : m;
            alpha[r]   = expf(row_max[r] - msub[r]);
            row_max[r] = m;
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            row_sum[r] *= alpha[r];
        }
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float p0 = expf(S[j][2*r + 0] - msub[r]);
                const float p1 = expf(S[j][2*r + 1] - msub[r]);
                row_sum[r] += p0 + p1;
                P_a[j][r] = fattn_banded_pack_h2(p0, p1);
            }
        }
#pragma unroll
        for (int n = 0; n < ND; ++n) {
            O[n][0] *= alpha[0];
            O[n][1] *= alpha[0];
            O[n][2] *= alpha[1];
            O[n][3] *= alpha[1];
        }
    };

    // O += P V for the tile in V_s (fp32 accumulators)
    auto pv_tile = [&]() {
#pragma unroll
        for (int s = 0; s < BK/16; ++s) {
#pragma unroll
            for (int n = 0; n < ND; n += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4_trans(b, V_s + (s*16 + (mi % 2)*8 + lane % 8)*stride + (n + mi/2)*8);
                fattn_banded_mma_16816(O[n + 0], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[0], b[1]);
                fattn_banded_mma_16816(O[n + 1], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[2], b[3]);
            }
        }
    };

    if constexpr (kv_q) {
        // Quantized K/V: each thread fetches its block pairs of a tile into registers (w*), then dequantizes them into
        // the f16 tile (fattn-mma-f16.cuh loaders, the same values as to_fp16). The next K tile is fetched before the
        // scores of this one and the next V tile before this P.V, so both loads are in flight under tensor-core work.
        uint32_t wK[q_slots][q_words];
        uint32_t wV[q_slots][q_words];
        auto fetch_q = [&](uint32_t (&w)[q_slots][q_words], const char * base, const uint64_t nb1, const int64_t kv0) {
            const char * src = base + uint64_t(kv0)*nb1;
            if (kv0 + BK <= n_kv) {
                flash_attn_ext_f16_fetch_tile_q<type_KV, NW, BK, D, false, q_slots, q_words>(src, w, int(nb1), BK);
            } else {
                flash_attn_ext_f16_fetch_tile_q<type_KV, NW, BK, D, true,  q_slots, q_words>(src, w, int(nb1), int(n_kv - kv0));
            }
        };
        auto store_q = [&](half * tile, const uint32_t (&w)[q_slots][q_words], const int64_t kv0) {
            if (kv0 + BK <= n_kv) {
                flash_attn_ext_f16_store_tile_q<type_KV, stride/2, NW, BK, D, false, q_slots, q_words>((half2 *) tile, w, BK);
            } else {
                flash_attn_ext_f16_store_tile_q<type_KV, stride/2, NW, BK, D, true,  q_slots, q_words>((half2 *) tile, w, int(n_kv - kv0));
            }
        };

        int64_t kt = next_tile(kt_begin);
        if (kt < kt_end) {
            fetch_q(wK, k_base, k_nb1, kt*BK);
            fetch_q(wV, v_base, v_nb1, kt*BK);
            store_q(K_s, wK, kt*BK);
        }
        while (kt < kt_end) {
            const int64_t kv0     = kt*BK;
            const int     flag    = tile_flag(kt);
            const int64_t kt_next = next_tile(kt + 1);

            __syncthreads(); // K_s holds tile kt; every warp is done with V_s of the previous tile
            if (kt_next < kt_end) {
                fetch_q(wK, k_base, k_nb1, kt_next*BK);
            }
            score_tile(kv0, flag);
            store_q(V_s, wV, kv0);
            if (kt_next < kt_end) {
                fetch_q(wV, v_base, v_nb1, kt_next*BK);
            }
            __syncthreads(); // V_s holds tile kt; every warp is done with K_s
            pv_tile();
            if (kt_next < kt_end) {
                store_q(K_s, wK, kt_next*BK);
            }
            kt = kt_next;
        }
    } else {
        int64_t kt = next_tile(kt_begin);
        if (kt < kt_end) {
            fattn_banded_mma_load_tile<D, NW>(K_s, k_base, k_nb1, kt*BK, n_kv);
            fattn_banded_mma_load_tile<D, NW>(V_s, v_base, v_nb1, kt*BK, n_kv);
        }

        while (kt < kt_end) {
            const int64_t kv0  = kt*BK;
            const int     flag = tile_flag(kt);

            fattn_banded_mma_wait<1>(); // K tile landed, V may still be in flight
            __syncthreads();

            score_tile(kv0, flag);

            __syncthreads();
            const int64_t kt_next = next_tile(kt + 1);
            if (kt_next < kt_end) {
                fattn_banded_mma_load_tile<D, NW>(K_s, k_base, k_nb1, kt_next*BK, n_kv);
                fattn_banded_mma_wait<1>(); // V tile of this iteration landed
            } else {
                fattn_banded_mma_wait<0>();
            }
            __syncthreads();

            pv_tile();

            __syncthreads();
            if (kt_next < kt_end) {
                fattn_banded_mma_load_tile<D, NW>(V_s, v_base, v_nb1, kt_next*BK, n_kv);
            }
            kt = kt_next;
        }
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffffu, row_sum[r], 1, WARP_SIZE);
        row_sum[r] += __shfl_xor_sync(0xffffffffu, row_sum[r], 2, WARP_SIZE);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int64_t iq = iq_r[r];
        if (iq >= n_q) {
            continue;
        }
        const int64_t row = (ib*n_q + iq)*n_head_q + ih;
        if (n_chunks == 1) {
            float m   = row_max[r];
            float sum = row_sum[r];
            float f   = 1.0f;
            if (sinks) {
                const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
                const float m_new = fmaxf(m, sink);
                f   = expf(m - m_new);
                sum = sum*f + expf(sink - m_new);
            }
            const float inv = sum == 0.0f ? 0.0f : f/sum;
            float2 * d = (float2 *) (dst + row*D);
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                d[n*4 + t] = make_float2(O[n][2*r + 0]*inv, O[n][2*r + 1]*inv);
            }
        } else {
            float * p = part + (row*n_chunks + ic)*(D + 2);
            if (t == 0) {
                p[0] = row_max[r];
                p[1] = row_sum[r];
            }
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                p[2 + n*8 + 2*t + 0] = O[n][2*r + 0];
                p[2 + n*8 + 2*t + 1] = O[n][2*r + 1];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, flags, rel, sinks, dst, part, scale, type_rel, n_q, n_kv, n_head_q, n_head_kv,
        n_chunks, tiles_per_chunk, rel_extent, mask_ne2, mask_ne3, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3,
        v_nb1, v_nb2, v_nb3, m_nb1, m_nb2, m_nb3, r_nb0, r_nb1, r_nb2, r_nb3, rel_ne3, sinks_s1, sinks_s2, window,
        pos_q, pos_kv, pos_q_ne1, pos_kv_ne1);
    NO_DEVICE_CODE;
#endif // FATTN_BANDED_MMA_AVAILABLE
}

// n_tok <= 32 (decode, verify batches) with f16 / q8_0 / q4_0 K/V is the decode kernel's (fattn-banded-dec.cu). Unsloth's
// small-batch kernels below keep n_tok <= 32 with f32 / bf16 K/V, and f16 when the decode kernel is switched off
// (GGML_CUDA_BANDED_DEC=0); quantized K/V with n_tok <= 32 is then unsupported.
static constexpr int64_t fattn_banded_dec_max_q = GGML_CUDA_BANDED_DEC_MAX_Q;

static bool fattn_banded_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static bool fattn_banded_kv_type_native(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0;
}

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }
    // 1-32 query rows with f16 / q8_0 / q4_0 K/V: the decode kernel
    if (ggml_cuda_flash_attn_ext_banded_dec_supported(device, dst)) {
        return true;
    }

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    if (!q || !k || !v || !rel || q->type != GGML_TYPE_F32) {
        return false;
    }
    if (!fattn_banded_type_supported(rel->type)) {
        return false;
    }
    if (fattn_banded_kv_type_native(k->type) || fattn_banded_kv_type_native(v->type)) {
        // quantized K/V: the tensor-core prefill kernel, the same type for K and V
        if (k->type != v->type || q->ne[1] <= fattn_banded_dec_max_q ||
                !turing_mma_available(ggml_cuda_info().devices[device].cc)) {
            return false;
        }
    } else if (!fattn_banded_type_supported(k->type) || !fattn_banded_type_supported(v->type)) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 128) || v->ne[0] != q->ne[0] || k->ne[0] != q->ne[0]) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || rel->nb[0] != ggml_type_size(rel->type)) {
        return false;
    }
    if (rel->ne[0] <= 0 || rel->ne[1] != q->ne[2] || rel->ne[2] != q->ne[1] ||
        (rel->ne[3] != 1 && rel->ne[3] != q->ne[3])) {
        return false;
    }
    const ggml_tensor * sinks = dst->src[4];
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float))) {
        return false;
    }
    // implicit window (op_params[4] > 0) and a mask tensor together: not supported
    if (ggml_get_op_params_i32(dst, 4) > 0 && m) {
        return false;
    }
    // token positions: I32, contiguous, one per query / cell, with the implicit window
    const ggml_tensor * pos_kv = dst->src[5];
    const ggml_tensor * pos_q  = dst->src[7];
    if ((pos_kv != nullptr) != (pos_q != nullptr)) {
        return false;
    }
    if (pos_kv && (pos_kv->type != GGML_TYPE_I32 || pos_q->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos_kv) ||
            !ggml_is_contiguous(pos_q) || pos_kv->ne[0] != k->ne[1] || pos_q->ne[0] != q->ne[1] || m ||
            ggml_get_op_params_i32(dst, 4) <= 0)) {
        return false;
    }
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0);
#endif
}

static constexpr int64_t fattn_banded_mma_min_q = 16;

static bool fattn_banded_mma_aligned(const ggml_tensor * t) {
    return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

// the quantized tile loaders read each block pair (36 / 68 bytes) as 32-bit words
static bool fattn_banded_mma_q_aligned(const ggml_tensor * t) {
    return (uintptr_t) t->data % 4 == 0 && t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && t->nb[3] % 4 == 0 &&
        t->ne[0] % 64 == 0 && t->nb[1] <= INT32_MAX;
}

// GGML_CUDA_BANDED_F16_COPY=1 (checks): quantized K/V take Unsloth's full f16 copy and the f16 kernel
static bool fattn_banded_f16_copy() {
    static const bool on = [] {
        const char * env = getenv("GGML_CUDA_BANDED_F16_COPY");
        return env != nullptr && atoi(env) != 0;
    }();
    return on;
}

static bool fattn_banded_mma_usable(const int device, const ggml_tensor * dst) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_DISABLE_BANDED_MMA");
        return env != nullptr && atoi(env) != 0;
    }();
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    if (!turing_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    if (fattn_banded_kv_type_native(k->type)) {
        return true; // the only kernel for quantized K/V (supported() asks for n_tok > 32)
    }
    if (disabled) {
        return false;
    }
    if (q->ne[1] < fattn_banded_mma_min_q) {
        return false;
    }
    for (const ggml_tensor * t : {k, v}) {
        if (t->type == GGML_TYPE_F16 ? !fattn_banded_mma_aligned(t) : t->type != GGML_TYPE_BF16) {
            return false;
        }
    }
    return true;
}

#define FATTN_BANDED_MMA_NW(F, DD, T) \
    switch (nw) { \
        case 4:  F(DD, 4, T); break; \
        case 2:  F(DD, 2, T); break; \
        default: F(DD, 1, T); break; \
    }
// quantized K/V run only with n_tok > 32, so with 2 or 4 warps
#define FATTN_BANDED_MMA_NW_Q(F, DD, T) \
    switch (nw) { \
        case 4:  F(DD, 4, T); break; \
        default: F(DD, 2, T); break; \
    }
#define FATTN_BANDED_MMA_SWITCH(F) \
    switch (type_KV) { \
        case GGML_TYPE_Q4_0: \
            if (D == 64) { FATTN_BANDED_MMA_NW_Q(F, 64, GGML_TYPE_Q4_0); } else { FATTN_BANDED_MMA_NW_Q(F, 128, GGML_TYPE_Q4_0); } \
            break; \
        case GGML_TYPE_Q8_0: \
            if (D == 64) { FATTN_BANDED_MMA_NW_Q(F, 64, GGML_TYPE_Q8_0); } else { FATTN_BANDED_MMA_NW_Q(F, 128, GGML_TYPE_Q8_0); } \
            break; \
        default: \
            if (D == 64) { FATTN_BANDED_MMA_NW(F, 64, GGML_TYPE_F16); } else { FATTN_BANDED_MMA_NW(F, 128, GGML_TYPE_F16); } \
            break; \
    }

static void fattn_banded_mma_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale, const int64_t rel_extent) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * m     = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];
    cudaStream_t stream = ctx.stream();

    const int64_t D    = q->ne[0];
    const int64_t n_q  = q->ne[1];
    const int64_t n_kv = k->ne[1];

    const char * k_data = (const char *) k->data;
    const char * v_data = (const char *) v->data;
    uint64_t k_nb[4] = {k->nb[0], k->nb[1], k->nb[2], k->nb[3]};
    uint64_t v_nb[4] = {v->nb[0], v->nb[1], v->nb[2], v->nb[3]};
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool());
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool());
    auto to_f16 = [&](const ggml_tensor * t, ggml_cuda_pool_alloc<half> & buf, const char *& data, uint64_t (&nb)[4]) {
        if (t->type == GGML_TYPE_F16) {
            return;
        }
        const size_t ts = ggml_type_size(t->type);
        buf.alloc(ggml_nelements(t));
        ggml_get_to_fp16_nc_cuda(t->type)(t->data, buf.get(), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[1]/ts, t->nb[2]/ts, t->nb[3]/ts, stream);
        data  = (const char *) buf.get();
        nb[0] = sizeof(half);
        nb[1] = t->ne[0]*nb[0];
        nb[2] = t->ne[1]*nb[1];
        nb[3] = t->ne[2]*nb[2];
    };
    // q4_0 / q8_0: read natively unless asked for the f16 copy (or not word aligned)
    ggml_type type_KV = GGML_TYPE_F16;
    if (k->type == v->type && fattn_banded_kv_type_native(k->type) && !fattn_banded_f16_copy() &&
            fattn_banded_mma_q_aligned(k) && fattn_banded_mma_q_aligned(v)) {
        type_KV = k->type;
    } else {
        to_f16(k, k_f16, k_data, k_nb);
        to_f16(v, v_f16, v_data, v_nb);
    }

    // implicit mask (ggml_flash_attn_ext_banded_set_window): no mask tensor, no tile flags
    const int32_t window = ggml_get_op_params_i32(dst, 4);
    const ggml_tensor * pos_kv_t = dst->src[5];   // ggml_flash_attn_ext_banded_set_pos
    const ggml_tensor * pos_q_t  = dst->src[7];
    GGML_ASSERT(window <= 0 || m == nullptr);

    const int nw = n_q >= 64 ? 4 : (n_q >= 32 ? 2 : 1);
    const int bq = nw*16;
    const int64_t n_qt = (n_q + bq - 1)/bq;
    const int64_t n_kt = (n_kv + fattn_banded_mma_bk - 1)/fattn_banded_mma_bk;

    ggml_cuda_pool_alloc<uint8_t> flags(ctx.pool());
    if (m) {
        const int64_t n_z = m->ne[2]*m->ne[3];
        flags.alloc(n_z*n_qt*n_kt);
        flash_attn_ext_banded_mma_tile_flags<<<dim3(n_kt, n_qt, n_z), 256, 0, stream>>>(
            (const char *) m->data, flags.get(), n_q, n_kv, bq, m->ne[2], m->nb[1], m->nb[2], m->nb[3]);
    }

    int occ = 0;
#define OCC_BANDED_MMA(DD, NW, T) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, flash_attn_ext_banded_mma<DD, NW, T>, NW*WARP_SIZE, 0))
    FATTN_BANDED_MMA_SWITCH(OCC_BANDED_MMA)
#undef OCC_BANDED_MMA
    const int64_t resident    = int64_t(std::max(occ, 1))*ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t base_blocks = n_qt*q->ne[2]*q->ne[3];
    constexpr int64_t min_tiles_per_chunk = 4;
    const int64_t max_chunks  = std::max<int64_t>(1, std::min<int64_t>(n_kt/min_tiles_per_chunk, (4*resident + base_blocks - 1)/base_blocks));
    int64_t n_chunks = 1;
    double  best_eff = 0.0;
    for (int64_t c = 1; c <= max_chunks; ++c) {
        const int64_t blocks = base_blocks*c;
        const double  eff    = double(blocks) / double(((blocks + resident - 1)/resident)*resident);
        if (eff > best_eff + 1e-3) {
            best_eff = eff;
            n_chunks = c;
        }
    }
    const int64_t tiles_per_chunk = (n_kt + n_chunks - 1)/n_chunks;
    n_chunks = (n_kt + tiles_per_chunk - 1)/tiles_per_chunk;

    const int64_t n_rows = n_q*q->ne[2]*q->ne[3];
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (n_chunks > 1) {
        part.alloc(n_rows*n_chunks*(D + 2));
    }

    const float * sinks_d  = sinks ? (const float *) sinks->data : nullptr;
    const int64_t sinks_s1 = sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0;
    const int64_t sinks_s2 = sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0;

    const dim3 blocks(n_qt*n_chunks, q->ne[2], q->ne[3]);
#define LAUNCH_BANDED_MMA(DD, NW, T) \
    flash_attn_ext_banded_mma<DD, NW, T><<<blocks, NW*WARP_SIZE, 0, stream>>>( \
        (const char *) q->data, k_data, v_data, m ? (const char *) m->data : nullptr, m ? flags.get() : nullptr, \
        (const char *) rel->data, n_chunks == 1 ? sinks_d : nullptr, (float *) dst->data, part.get(), \
        scale, rel->type, n_q, n_kv, q->ne[2], k->ne[2], n_chunks, tiles_per_chunk, rel_extent, \
        m ? m->ne[2] : 1, m ? m->ne[3] : 1, q->nb[1], q->nb[2], q->nb[3], \
        k_nb[1], k_nb[2], k_nb[3], v_nb[1], v_nb[2], v_nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], sinks_s1, sinks_s2, window, \
        pos_q_t ? (const int32_t *) pos_q_t->data : nullptr, pos_kv_t ? (const int32_t *) pos_kv_t->data : nullptr, \
        pos_q_t ? pos_q_t->ne[1] : 1, pos_kv_t ? pos_kv_t->ne[1] : 1)
    FATTN_BANDED_MMA_SWITCH(LAUNCH_BANDED_MMA)
#undef LAUNCH_BANDED_MMA
    CUDA_CHECK(cudaGetLastError());

    if (n_chunks > 1) {
        constexpr int merge_warps = 4;
        const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
        const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
        if (D == 64) {
            flash_attn_ext_banded_f32_merge<64><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        } else {
            flash_attn_ext_banded_f32_merge<128><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        }
    }
}
#undef FATTN_BANDED_MMA_SWITCH
#undef FATTN_BANDED_MMA_NW_Q
#undef FATTN_BANDED_MMA_NW

// Unsloth's warp-per-query kernels (f32 / f16 / bf16 K/V): split over KV chunks when there are few rows
static void fattn_banded_warp_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale, const int64_t rel_extent) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    const ggml_tensor * sinks = dst->src[4];
    const int32_t window = ggml_get_op_params_i32(dst, 4);
    const ggml_tensor * pos_kv_t = dst->src[5];   // ggml_flash_attn_ext_banded_set_pos
    const ggml_tensor * pos_q_t  = dst->src[7];

    {
        constexpr int split_warps   = 4;
        constexpr int min_chunk     = 64;
        const int64_t n_rows        = q->ne[1]*q->ne[2]*q->ne[3];
        const int64_t n_kv          = k->ne[1];
        const int64_t target_blocks = int64_t(ggml_cuda_info().devices[ctx.device].nsm)*16;
        int64_t n_chunks = n_rows < target_blocks ? (target_blocks + n_rows - 1)/n_rows : 1;
        n_chunks = std::min<int64_t>(n_chunks, (n_kv + min_chunk - 1)/min_chunk);
        if (n_chunks > 1 && q->ne[1]*q->ne[3] <= 65535) {
            const int64_t chunk = (n_kv + n_chunks - 1)/n_chunks;
            n_chunks = (n_kv + chunk - 1)/chunk;
            const int64_t D = q->ne[0];
            ggml_cuda_pool_alloc<float> part(ctx.pool(), n_rows*n_chunks*(D + 2));
            const dim3 sblocks(n_chunks, q->ne[2], q->ne[1]*q->ne[3]);
            const dim3 sthreads(split_warps*WARP_SIZE, 1, 1);
            constexpr int merge_warps = 4;
            const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
            const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
            cudaStream_t stream = ctx.stream();
#define LAUNCH_SPLIT(DD) \
            flash_attn_ext_banded_f32_split<DD, split_warps><<<sblocks, sthreads, 0, stream>>>( \
                (const char *) q->data, (const char *) k->data, (const char *) v->data, \
                m ? (const char *) m->data : nullptr, (const char *) rel->data, part.get(), \
                scale, k->type, v->type, rel->type, q->ne[1], n_kv, q->ne[2], k->ne[2], n_chunks, chunk, \
                rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
                q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
                v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
                m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
                rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], window, \
                pos_q_t ? (const int32_t *) pos_q_t->data : nullptr, pos_kv_t ? (const int32_t *) pos_kv_t->data : nullptr, \
        pos_q_t ? pos_q_t->ne[1] : 1, pos_kv_t ? pos_kv_t->ne[1] : 1); \
            flash_attn_ext_banded_f32_merge<DD><<<mblocks, mthreads, 0, stream>>>( \
                part.get(), (float *) dst->data, n_rows, q->ne[2], q->ne[1], n_chunks, \
                sinks ? (const float *) sinks->data : nullptr, \
                sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
                sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0)
            if (D == 64) {
                LAUNCH_SPLIT(64);
            } else {
                LAUNCH_SPLIT(128);
            }
#undef LAUNCH_SPLIT
            return;
        }
    }

    constexpr int warps_per_block = 4;
    const dim3 blocks((q->ne[1] + warps_per_block - 1) / warps_per_block, q->ne[2], q->ne[3]);
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

#define LAUNCH_BANDED(D) \
    flash_attn_ext_banded_f32<D, warps_per_block><<<blocks, threads, 0, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        m ? (const char *) m->data : nullptr, (const char *) rel->data, (float *) dst->data, \
        scale, k->type, v->type, rel->type, q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3], \
        rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
        q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        sinks ? (const float *) sinks->data : nullptr, \
        sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
        sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], window, \
        pos_q_t ? (const int32_t *) pos_q_t->data : nullptr, pos_kv_t ? (const int32_t *) pos_kv_t->data : nullptr, \
        pos_q_t ? pos_q_t->ne[1] : 1, pos_kv_t ? pos_kv_t->ne[1] : 1)

    if (q->ne[0] == 64) {
        LAUNCH_BANDED(64);
    } else {
        LAUNCH_BANDED(128);
    }
#undef LAUNCH_BANDED
}

// ggml-cuda.cu: tiered K/V cache staging, as used by ggml_cuda_flash_attn_ext (fattn.cu)
void *  ggml_cuda_tier_stage(const void * ptr, size_t nbytes, cudaStream_t stream);
bool    ggml_cuda_tier_fa_begin(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, void ** K_alias, void ** V_alias);
void    ggml_cuda_tier_fa_end(ggml_backend_cuda_context & ctx);
int64_t ggml_cuda_tier_first_host_cell(const ggml_tensor * t);

// GGML_CUDA_KV_TIER_STAGE_MIN_Q=q: the same rule as for ggml_cuda_flash_attn_ext (fattn.cu): ops with fewer than q
// queries read the host tail in place. Unset: 0 (every op that reaches the tail stages).
static int64_t fattn_banded_tier_stage_min_q() {
    static const int64_t q = [] {
        const char * e = getenv("GGML_CUDA_KV_TIER_STAGE_MIN_Q");
        return e ? (int64_t) atoll(e) : (int64_t) 0;
    }();
    return q;
}

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_supported(ctx.device, dst));

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * rel   = dst->src[6];
    const ggml_tensor * sinks = dst->src[4];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    // the tensor extent (not op_params) is authoritative after graph cloning
    const int64_t rel_extent = rel->ne[0];

    // K/V in a tiered buffer whose host tail this op reaches (the global layers at depth): the host rows go to the
    // VRAM staging alias first, from the prefetch pipeline (ggml_cuda_tier_graph_begin lists these ops) or by a
    // synchronous copy, exactly as for ggml_cuda_flash_attn_ext. Only such ops call ggml_cuda_tier_fa_begin: an op
    // outside the pipeline's list (an SWA layer's banded op) would stop the pipeline. The bytes are the same.
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    void * K_data = K->data;
    void * V_data = V->data;
    bool tier_piped = false;
    if (V != K && (ggml_cuda_tier_first_host_cell(K) >= 0 || ggml_cuda_tier_first_host_cell(V) >= 0)) {
        void * K_alias = nullptr;
        void * V_alias = nullptr;
        tier_piped = ggml_cuda_tier_fa_begin(ctx, dst, &K_alias, &V_alias);
        if (tier_piped) {
            if (K_alias) {
                K->data = K_alias;
            }
            if (V_alias) {
                V->data = V_alias;
            }
        } else if (q->ne[1] >= fattn_banded_tier_stage_min_q()) {
            for (ggml_tensor * t : { K, V }) {
                void * a = ggml_cuda_tier_stage(t->data, ggml_nbytes(t), ctx.stream());
                if (a) {
                    t->data = a;
                }
            }
        }
    }

    if (ggml_cuda_flash_attn_ext_banded_dec_supported(ctx.device, dst)) {
        // decode and verify batches (n_tok <= 32) with f16 / q8_0 / q4_0 K/V
        ggml_cuda_flash_attn_ext_banded_dec(ctx, dst);
    } else if (q->ne[1] <= fattn_banded_dec_max_q) {
        // n_tok <= 32 with f32 / bf16 K/V (or the decode kernel switched off): Unsloth's small-batch kernels
        if (sinks && fattn_banded_mma_usable(ctx.device, dst)) {
            fattn_banded_mma_launch(ctx, dst, scale, rel_extent);
        } else {
            fattn_banded_warp_launch(ctx, dst, scale, rel_extent);
        }
    } else if (fattn_banded_mma_usable(ctx.device, dst)) {
        // prefill: the tensor-core kernel, with or without sinks
        fattn_banded_mma_launch(ctx, dst, scale, rel_extent);
    } else {
        fattn_banded_warp_launch(ctx, dst, scale, rel_extent);
    }

    if (tier_piped) {
        ggml_cuda_tier_fa_end(ctx);
    }
    K->data = K_data;
    V->data = V_data;
}
