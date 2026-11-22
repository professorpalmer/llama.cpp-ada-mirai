// Banded relative-position flash attention (Inkling 2), decode path: 1-32 query rows with K/V f16 / q8_0 / q4_0.
//
// Origin and credits (MIT):
// - the vector flash-attention kernel of ggml / llama.cpp (fattn-vec.cuh, the ggml authors), whose structure this
//   kernel follows (128 threads, 128-cell tiles, one warp per 32 cells, online softmax, KV split + merge) and whose
//   q4_0 / q8_0 K/V arithmetic (fattn-common.cuh: Q as q8_1, the dp4a dot, the dequantize_V formulas) it reuses;
// - the banded op of Unsloth's llama.cpp mix b12588-mix-c0c3de8 (the ggml authors and Unsloth): the band offset
//   (d = iq + (n_kv - n_q) - ik), the per-query sinks and the rule for rows without a visible cell.
//
// Differences to the vector kernel: the columns of a block are ncols1 queries x ncols2 Q heads of one K/V head, so
// each K and V row is read and unpacked once for all heads of a GQA group; scores, softmax state, P.V and the merge
// are fp32 (no half2 accumulators); n_kv need not be a multiple of the tile (cells past n_kv are never read); the band
// bias and per-query sinks are added in the kernel.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-banded-dec.cuh"
#include "fattn-banded.cuh"

#include <algorithm>
#include <cstdint>

// load-group sizes (compile-time tuning knobs, 0 = the defaults in the kernel)
#ifndef GGML_CUDA_BANDED_DEC_VG
#define GGML_CUDA_BANDED_DEC_VG 0
#endif
#ifndef GGML_CUDA_BANDED_DEC_KPRE
#define GGML_CUDA_BANDED_DEC_KPRE 0
#endif

struct banded_dec_params {
    const char  * Q;
    const char  * K;
    const char  * V;
    const char  * mask;
    const char  * rel;
    const float * sinks;
    float       * dst;       // final output, or the per-split partial outputs when gridDim.y > 1
    float2      * dst_meta;  // (max, sum) per row and split when gridDim.y > 1

    float   scale;
    int     n_q;
    int     n_head;
    int     gqa;
    int     n_kv;

    int64_t nb01, nb02, nb03;
    int64_t nb11, nb12, nb13;
    int64_t nb21, nb22, nb23;
    int64_t nb31, nb32, nb33;
    int     ne32, ne33;

    int64_t r_nb0, r_nb1, r_nb2, r_nb3;
    int     r_ne3;
    int     rel_extent;
    int     type_rel;

    int64_t sink_s1, sink_s2; // in floats, 0 for broadcast dimensions

    int     window;      // > 0: implicit mask (no mask tensor), cell visible iff 0 <= d < window
    int     window_skip; // start each block's loop at the first tile the window can see (off in batch-invariant mode)

    // token positions (ggml_flash_attn_ext_banded_set_pos, ring caches): d = pos_q[iq] - pos_kv[cell] replaces the tail
    // rule, a cell with pos_kv < 0 is empty. nullptr: the tail rule. Stream of batch b: b % pos_*_ns.
    const int32_t * pos_q;
    const int32_t * pos_kv;
    int     pos_q_ns;
    int     pos_kv_ns;

    // band bias from its factors (ggml_flash_attn_ext_banded_set_rel_factors): rel is then the table [d_rel, rel_extent]
    // (r_nb1 bytes per distance, type_rel F32 / F16) and rcur [d_rel, n_head, n_q] F32; nullptr: rel holds rel_logits
    const float * rcur;
    int     d_rel;
};

static __device__ __forceinline__ float banded_dec_load_rel(const char * ptr, const int type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16:
            return __uint_as_float(uint32_t(*(const uint16_t *) ptr) << 16);
        default:
            return 0.0f;
    }
}

// The K row piece of one thread: 4 consecutive values of the row at 4*(lane % nthreads_KQ), in the form the dot with
// each column's Q needs. Loaded and unpacked once per cell, used by every column of the block.
// 4 bytes at a 2-byte aligned address, as two 16-bit loads joined in a register (no address of a register variable is
// taken, which could force the K / V pieces into local memory)
static __device__ __forceinline__ int banded_dec_ld4(const void * p) {
    const uint16_t * h = (const uint16_t *) p;
    return (int) ((uint32_t) h[0] | ((uint32_t) h[1] << 16));
}

// 2 half2 at a 4-byte aligned address
static __device__ __forceinline__ void banded_dec_ld_h2x2(const half2 * p, half2 & a, half2 & b) {
    a = p[0];
    b = p[1];
}

template <ggml_type type_K, int VPL>
struct banded_dec_k_piece;

// f16 K: 4 values per thread, Q in fp32
template <>
struct banded_dec_k_piece<GGML_TYPE_F16, 4> {
    float2 a, b;
    template <int nthreads_KQ>
    __device__ __forceinline__ void load(const char * K_row) {
        half2 ta, tb;
        banded_dec_ld_h2x2((const half2 *) K_row + 2*(threadIdx.x % nthreads_KQ), ta, tb);
        a = __half22float2(ta);
        b = __half22float2(tb);
    }
    __device__ __forceinline__ float dot(const float2 * Qf, const int * Qi, const float2 Qds) const {
        GGML_UNUSED(Qi);
        GGML_UNUSED(Qds);
        return a.x*Qf[0].x + a.y*Qf[0].y + b.x*Qf[1].x + b.y*Qf[1].y;
    }
};

// q4_0 K, 4 values per thread (head size 64): one int of a block, Q as q8_1
template <>
struct banded_dec_k_piece<GGML_TYPE_Q4_0, 4> {
    int   v;
    float d;
    template <int nthreads_KQ>
    __device__ __forceinline__ void load(const char * K_row) {
        const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_row;
        const int k_KQ  = threadIdx.x % nthreads_KQ;
        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);
        v = banded_dec_ld4(K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        d = __half2float(K_q4_0[ib].d);
    }
    __device__ __forceinline__ float dot(const float2 * Qf, const int * Qi, const float2 Qds) const {
        GGML_UNUSED(Qf);
        const int sumi = ggml_cuda_dp4a(v, Qi[0], 0);
        return d * (sumi*Qds.x - (8/QI8_1)*Qds.y);
    }
};

// q8_0 K, 4 values per thread (head size 64)
template <>
struct banded_dec_k_piece<GGML_TYPE_Q8_0, 4> {
    int   v;
    float d;
    template <int nthreads_KQ>
    __device__ __forceinline__ void load(const char * K_row) {
        const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_row;
        const int k_KQ = threadIdx.x % nthreads_KQ;
        const int ib   = k_KQ / QI8_0;
        const int iqs  = k_KQ % QI8_0;
        v = banded_dec_ld4(K_q8_0[ib].qs + sizeof(int)*iqs);
        d = __half2float(K_q8_0[ib].d);
    }
    __device__ __forceinline__ float dot(const float2 * Qf, const int * Qi, const float2 Qds) const {
        GGML_UNUSED(Qf);
        const int sumi = ggml_cuda_dp4a(v, Qi[0], 0);
        return d * Qds.x * float(sumi);
    }
};

// q4_0 K, 16 values per thread (head size 128, 8 threads per row): half a block, the low (even thread) or the high
// (odd thread) nibbles of its 16 bytes. Two threads share a block, so each subtracts half the q8_1 offset term.
template <>
struct banded_dec_k_piece<GGML_TYPE_Q4_0, 16> {
    int   v[4];
    float d;
    template <int nthreads_KQ>
    __device__ __forceinline__ void load(const char * K_row) {
        const int t     = threadIdx.x % nthreads_KQ;
        const block_q4_0 * x = (const block_q4_0 *) K_row + t/2;
        const int shift = 4*(t % 2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = (banded_dec_ld4(x->qs + 4*j) >> shift) & 0x0F0F0F0F;
        }
        d = __half2float(x->d);
    }
    __device__ __forceinline__ float dot(const float2 * Qf, const int * Qi, const float2 Qds) const {
        GGML_UNUSED(Qf);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(v[j], Qi[j], sumi);
        }
        return d * (sumi*Qds.x - (8/2)*Qds.y);
    }
};

// q8_0 K, 16 values per thread (head size 128, 8 threads per row): half a block
template <>
struct banded_dec_k_piece<GGML_TYPE_Q8_0, 16> {
    int   v[4];
    float d;
    template <int nthreads_KQ>
    __device__ __forceinline__ void load(const char * K_row) {
        const int t = threadIdx.x % nthreads_KQ;
        const block_q8_0 * x = (const block_q8_0 *) K_row + t/2;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = banded_dec_ld4(x->qs + 16*(t % 2) + 4*j);
        }
        d = __half2float(x->d);
    }
    __device__ __forceinline__ float dot(const float2 * Qf, const int * Qi, const float2 Qds) const {
        GGML_UNUSED(Qf);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(v[j], Qi[j], sumi);
        }
        return d * Qds.x * float(sumi);
    }
};

// signed byte i of x as float, by shifts (taking the address of a register value can force it into local memory)
static __device__ __forceinline__ float banded_dec_s8(const int x, const int i) {
    return (float) (((int) ((unsigned int) x << (24 - 8*i))) >> 24);
}

// The V row piece of one thread: 4 consecutive values of the row at 4*(lane % nthreads_V), loaded raw first (so that
// the loads of several cells are in flight together) and converted to fp32 when used.
template <ggml_type type_V>
struct banded_dec_v_piece;

template <>
struct banded_dec_v_piece<GGML_TYPE_F16> {
    half2 h[2];
    template <int nthreads_V>
    __device__ __forceinline__ void load(const char * V_row) {
        banded_dec_ld_h2x2((const half2 *) V_row + 2*(threadIdx.x % nthreads_V), h[0], h[1]);
    }
    __device__ __forceinline__ void get(float2 * out) const {
        out[0] = __half22float2(h[0]);
        out[1] = __half22float2(h[1]);
    }
};

template <>
struct banded_dec_v_piece<GGML_TYPE_Q4_0> {
    int   q;
    float d;
    template <int nthreads_V>
    __device__ __forceinline__ void load(const char * V_row) {
        const block_q4_0 * x = (const block_q4_0 *) V_row;
        const int i0    = 4*(threadIdx.x % nthreads_V);
        const int ib    = i0 / QK4_0;
        const int iqs   = i0 % (QK4_0/2);
        const int shift = (i0 % QK4_0) / (QK4_0/2);
        q = banded_dec_ld4(x[ib].qs + iqs);
        q = (q >> (4*shift)) & 0x0F0F0F0F;
        d = __half2float(x[ib].d);
    }
    __device__ __forceinline__ void get(float2 * out) const {
        const int qs = __vsubss4(q, 0x08080808);
        out[0] = make_float2(d*banded_dec_s8(qs, 0), d*banded_dec_s8(qs, 1));
        out[1] = make_float2(d*banded_dec_s8(qs, 2), d*banded_dec_s8(qs, 3));
    }
};

template <>
struct banded_dec_v_piece<GGML_TYPE_Q8_0> {
    int   q;
    float d;
    template <int nthreads_V>
    __device__ __forceinline__ void load(const char * V_row) {
        const block_q8_0 * x = (const block_q8_0 *) V_row;
        const int i0 = 4*(threadIdx.x % nthreads_V);
        const int ib = i0 / QK8_0;
        q = banded_dec_ld4(x[ib].qs + i0 % QK8_0);
        d = __half2float(x[ib].d);
    }
    __device__ __forceinline__ void get(float2 * out) const {
        out[0] = make_float2(d*banded_dec_s8(q, 0), d*banded_dec_s8(q, 1));
        out[1] = make_float2(d*banded_dec_s8(q, 2), d*banded_dec_s8(q, 3));
    }
};

template<int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
__launch_bounds__(4*WARP_SIZE, 1)
static __global__ void flash_attn_ext_banded_vec(const banded_dec_params p) {
#ifdef FLASH_ATTN_AVAILABLE
    static_assert(D == 64 || D == 128, "banded decode: head size 64 or 128");
    constexpr int ncols       = ncols1*ncols2;
    constexpr int nwarps      = 4;
    constexpr int nthreads    = nwarps*WARP_SIZE;          // = cells per tile
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    // threads per K row: quantized K at head size 128 takes 8 threads x 16 values (4 cells per warp step, 3 shuffles
    // per reduction); otherwise D/4 threads x 4 values
    constexpr int nthreads_KQ = Q_q8_1 && D == 128 ? 8 : (D/4 < WARP_SIZE ? D/4 : WARP_SIZE);
    constexpr int VPL         = D/nthreads_KQ;             // K values per thread
    constexpr int nthreads_V  = D/4 < WARP_SIZE ? D/4 : WARP_SIZE; // threads per V row, 4 values each
    constexpr int V_cols_per_iter = WARP_SIZE/nthreads_V;  // V rows a warp handles at once

    constexpr int ne_Qtmp    = ncols*D;
    constexpr int ne_KQ      = ncols*nthreads;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
    constexpr int ne_max2    = ne_KQ > ne_combine ? ne_KQ : ne_combine;
    constexpr int ne_shared  = ne_Qtmp > ne_max2 ? ne_Qtmp : ne_max2;
    __shared__ float KQ[ne_shared];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int ic0      = blockIdx.x*ncols1;
    const int n_groups = p.n_head/ncols2;
    const int sequence = blockIdx.z / n_groups;
    const int head0    = (blockIdx.z - sequence*n_groups)*ncols2;
    const int head_kv  = head0 / p.gqa;

    const char * K = p.K + sequence*p.nb13 + head_kv*p.nb12;
    const char * V = p.V + sequence*p.nb23 + head_kv*p.nb22;

    // Q of every column: q8_1 (quantized K) or 4 fp32 values per thread (f16 K), the scale folded in
    float2 Q_f [ncols][2];
    int    Q_i [ncols][VPL/4];
    float2 Q_ds[ncols];

    if constexpr (Q_q8_1) {
#pragma unroll
        for (int c0 = 0; c0 < ncols; c0 += nwarps) {
            const int c = c0 + warp;
            if (c0 + nwarps > ncols && c >= ncols) {
                break;
            }
            const int jq = c / ncols2;
            const int jh = c % ncols2;

            int    * tmp_q_i32 = (int    *) &KQ[c*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            if (ncols1 > 1 && ic0 + jq >= p.n_q) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + lane;
                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (lane < D/QK8_1) {
                    tmp_q_ds[lane] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_row = (const float *) (p.Q + sequence*p.nb03 + (head0 + jh)*p.nb02 + (ic0 + jq)*p.nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_row + i0*sizeof(int), p.scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const int    * tmp_q_i32 = (const int    *) &KQ[c*D];
            const float2 * tmp_q_ds  = (const float2 *) (tmp_q_i32 + D/sizeof(int));
            const int t = lane % nthreads_KQ; // this thread holds Q ints t*VPL/4 .. +VPL/4-1, within one q8_1 block
#pragma unroll
            for (int j = 0; j < VPL/4; ++j) {
                Q_i[c][j] = tmp_q_i32[t*(VPL/4) + j];
            }
            Q_ds[c] = tmp_q_ds[(t*VPL)/QK8_1];
            Q_f [c][0] = make_float2(0.0f, 0.0f);
            Q_f [c][1] = make_float2(0.0f, 0.0f);
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const int jq = c / ncols2;
            const int jh = c % ncols2;
#pragma unroll
            for (int j = 0; j < VPL/4; ++j) {
                Q_i[c][j] = 0;
            }
            Q_ds[c] = make_float2(0.0f, 0.0f);
            if (ncols1 > 1 && ic0 + jq >= p.n_q) {
                Q_f[c][0] = make_float2(0.0f, 0.0f);
                Q_f[c][1] = make_float2(0.0f, 0.0f);
            } else {
                const float2 * Q_row = (const float2 *) (p.Q + sequence*p.nb03 + (head0 + jh)*p.nb02 + (ic0 + jq)*p.nb01);
                const int i = 2*(lane % nthreads_KQ);
                const float2 q0 = Q_row[i + 0];
                const float2 q1 = Q_row[i + 1];
                Q_f[c][0] = make_float2(q0.x*p.scale, q0.y*p.scale);
                Q_f[c][1] = make_float2(q1.x*p.scale, q1.y*p.scale);
            }
        }
    }

    float  KQ_max[ncols];
    float  KQ_sum[ncols];
    float2 VKQ[ncols][2];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        KQ_max[c] = -FLT_MAX/2.0f;
        KQ_sum[c] = 0.0f;
        VKQ[c][0] = make_float2(0.0f, 0.0f);
        VKQ[c][1] = make_float2(0.0f, 0.0f);
    }

    const int q_offset = p.n_kv - p.n_q; // the queries are the last n_q cells of the view

    // token positions of this block's queries and of the cells (ring caches); without them the tail rule
    const int32_t * pos_kv = p.pos_kv ? p.pos_kv + int64_t(sequence % p.pos_kv_ns)*p.n_kv : nullptr;
    int pos_q[ncols1];
#pragma unroll
    for (int jq = 0; jq < ncols1; ++jq) {
        pos_q[jq] = p.pos_q && (ncols1 == 1 || ic0 + jq < p.n_q) ? p.pos_q[int64_t(sequence % p.pos_q_ns)*p.n_q + ic0 + jq] : 0;
    }

    // implicit window (tail rule only): no query of this block sees a cell below ic0 + q_offset - (window - 1)
    int tile_lo = 0;
    if (p.window > 0 && p.window_skip && !pos_kv) {
        const int lo = ic0 + q_offset - (p.window - 1);
        tile_lo = lo > 0 ? lo / nthreads : 0;
    }

    for (int k_VKQ_0 = (tile_lo + blockIdx.y)*nthreads; k_VKQ_0 < p.n_kv; k_VKQ_0 += gridDim.y*nthreads) {
        // Each lane owns one cell of its warp's 32 (cell_t = k_VKQ_0 + tid) for the bias, the mask and the softmax: the
        // per-cell loop below only computes the dots. bias_t / mask_t hold this lane's band bias and mask per column.
        const int  cell_t  = k_VKQ_0 + tid;
        const bool valid_t = cell_t < p.n_kv; // a cell past the view is never visible
        // the cell's position: pos_kv (ring caches; < 0 = empty cell) or the tail rule (pk = cell index)
        const int  pk_t    = pos_kv ? (valid_t ? pos_kv[cell_t] : -1) : cell_t;
        float bias_t[ncols];
        float mask_t[ncols1];
        bool visible = false;
#pragma unroll
        for (int jq = 0; jq < ncols1; ++jq) {
            const int iq = ic0 + jq;
            // shared definition of distance and visibility (fattn-banded.cuh), as in the prefill kernels
            int d;
            const bool vis = fattn_banded_visible(pos_kv ? pos_q[jq] : iq + q_offset, pk_t, p.window, d);
            float mv = -INFINITY;
            if (valid_t && vis && (ncols1 == 1 || iq < p.n_q)) {
                if (p.window > 0) {
                    mv = 0.0f;
                } else if (p.mask && p.ne32 == 1) {
                    mv = __half2float(((const half *) (p.mask + (sequence % p.ne33)*p.nb33 + int64_t(iq)*p.nb31))[cell_t]);
                } else {
                    mv = 0.0f; // no mask, or a mask with a head dimension (added per head below)
                }
            }
            mask_t[jq] = mv;
#pragma unroll
            for (int jh = 0; jh < ncols2; ++jh) {
                const int c = jq*ncols2 + jh;
                float b = 0.0f;
                if (mv != -INFINITY) {
                    if (d >= 0 && d < p.rel_extent) {
                        b = p.rcur
                            // the shared definition of the bias from its factors (fattn-banded.cuh), as in the prefill path
                            ? fattn_banded_rel_bias(p.rel, p.type_rel, p.r_nb1, p.rcur + (int64_t(iq)*p.n_head + head0 + jh)*p.d_rel,
                                p.d_rel, d)
                            : banded_dec_load_rel(p.rel + (sequence % p.r_ne3)*p.r_nb3 + int64_t(iq)*p.r_nb2 +
                                int64_t(head0 + jh)*p.r_nb1 + int64_t(d)*p.r_nb0, p.type_rel);
                    }
                    if (p.window == 0 && p.mask && p.ne32 > 1) {
                        b += __half2float(((const half *) (p.mask + (sequence % p.ne33)*p.nb33 +
                            int64_t((head0 + jh) % p.ne32)*p.nb32 + int64_t(iq)*p.nb31))[cell_t]);
                    }
                    visible = visible || b != -INFINITY;
                }
                bias_t[c] = b;
            }
        }
        // a warp whose 32 cells are all invisible (masked or past n_kv) reads none of their K/V rows; the result is
        // the same as processing them (they add exp(-inf) = 0 and do not move the maximum)
        if (!__any_sync(0xFFFFFFFF, visible)) {
            continue;
        }

        // P.V operands: the V rows of the warp's first cell group are loaded now, so that these loads overlap the Q.K work
        // below; each next group is loaded before the current one is used. Many columns (more registers per cell):
        // smaller groups, so that the per-column state does not spill.
        constexpr int cells_vg  = GGML_CUDA_BANDED_DEC_VG > 0 ? GGML_CUDA_BANDED_DEC_VG : 8; // cells per V group
        constexpr int n_pre_V   = cells_vg/V_cols_per_iter; // V pieces per group
        constexpr int n_vgroups = WARP_SIZE/cells_vg;
        banded_dec_v_piece<type_V> vp[n_vgroups][n_pre_V];
#pragma unroll
        for (int i1 = 0; i1 < n_pre_V; ++i1) {
            const int k = warp*WARP_SIZE + i1*V_cols_per_iter + (nthreads_V == WARP_SIZE ? 0 : lane / nthreads_V);
            if (k_VKQ_0 + k < p.n_kv) {
                vp[0][i1].template load<nthreads_V>(V + int64_t(k_VKQ_0 + k)*p.nb21);
            }
        }

        // Q.K: the dot of each cell is reduced over its nthreads_KQ threads and the owning lane keeps it. The K rows of
        // n_pre cells are loaded before any of them is used, so that n_pre loads per warp are in flight together.
        constexpr int n_pre = GGML_CUDA_BANDED_DEC_KPRE > 0 ? GGML_CUDA_BANDED_DEC_KPRE : (ncols > 3 ? 4 : 8);
        float KQ_reg[ncols];
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            KQ_reg[c] = 0.0f;
        }
        const int cell_w = k_VKQ_0 + warp*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (lane & ~(nthreads_KQ - 1)));
#pragma unroll
        for (int i0 = 0; i0 < nthreads_KQ; i0 += n_pre) {
            banded_dec_k_piece<type_K, VPL> kp[n_pre];
#pragma unroll
            for (int i1 = 0; i1 < n_pre; ++i1) {
                if (cell_w + i0 + i1 < p.n_kv) { // uniform over the nthreads_KQ threads of the cell
                    kp[i1].template load<nthreads_KQ>(K + int64_t(cell_w + i0 + i1)*p.nb11);
                }
            }
#pragma unroll
            for (int i1 = 0; i1 < n_pre; ++i1) {
                const bool valid = cell_w + i0 + i1 < p.n_kv;
#pragma unroll
                for (int c = 0; c < ncols; ++c) {
                    float sum = valid ? kp[i1].dot(Q_f[c], Q_i[c], Q_ds[c]) : 0.0f;
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                    if ((lane % nthreads_KQ) == i0 + i1) {
                        KQ_reg[c] = sum;
                    }
                }
            }
        }

        // online softmax, per warp; this lane holds the KQ value of cell_t. vis_t: the cell is visible for a column.
        bool vis_t = false;
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float m = mask_t[c / ncols2];
            const float s = m == -INFINITY ? -INFINITY : KQ_reg[c] + bias_t[c] + m;
            vis_t = vis_t || s != -INFINITY;
            const float KQ_max_new = fmaxf(KQ_max[c], warp_reduce_max(s) + FATTN_KQ_MAX_OFFSET);
            const float KQ_max_scale = expf(KQ_max[c] - KQ_max_new);
            KQ_max[c] = KQ_max_new;

            const float w = expf(s - KQ_max[c]);
            KQ_sum[c] = KQ_sum[c]*KQ_max_scale + w;
            KQ[c*nthreads + tid] = w;

            VKQ[c][0].x *= KQ_max_scale;
            VKQ[c][0].y *= KQ_max_scale;
            VKQ[c][1].x *= KQ_max_scale;
            VKQ[c][1].y *= KQ_max_scale;
        }

        __syncwarp();

        // the cells of this warp that some column sees (the others add weight 0 and are skipped, see below)
        const uint32_t vis_mask = __ballot_sync(0xFFFFFFFF, vis_t);

        // P.V
#pragma unroll
        for (int g = 0; g < n_vgroups; ++g) {
            if (g + 1 < n_vgroups) {
#pragma unroll
                for (int i1 = 0; i1 < n_pre_V; ++i1) {
                    const int k = warp*WARP_SIZE + (g + 1)*cells_vg + i1*V_cols_per_iter + (nthreads_V == WARP_SIZE ? 0 : lane / nthreads_V);
                    if (k_VKQ_0 + k < p.n_kv) {
                        vp[g + 1][i1].template load<nthreads_V>(V + int64_t(k_VKQ_0 + k)*p.nb21);
                    }
                }
            }
#pragma unroll
            for (int i1 = 0; i1 < n_pre_V; ++i1) {
                const int k = warp*WARP_SIZE + g*cells_vg + i1*V_cols_per_iter + (nthreads_V == WARP_SIZE ? 0 : lane / nthreads_V);
                // a cell no column sees (masked, empty ring cell, past the view) has weight 0 in every column; its V values
                // are replaced by 0 (a select, no branch), so the bytes of an empty cell of a padded view need not be finite
                const bool vis_k = vis_mask == 0xFFFFFFFF || ((vis_mask >> (k % WARP_SIZE)) & 1);
                float2 tmp[2];
                vp[g][i1].get(tmp);
                if (!vis_k) {
                    tmp[0] = make_float2(0.0f, 0.0f);
                    tmp[1] = make_float2(0.0f, 0.0f);
                }
#pragma unroll
                for (int c = 0; c < ncols; ++c) {
                    const float w = KQ[c*nthreads + k];
                    VKQ[c][0].x += tmp[0].x*w;
                    VKQ[c][0].y += tmp[0].y*w;
                    VKQ[c][1].x += tmp[1].x*w;
                    VKQ[c][1].y += tmp[1].y*w;
                }
            }
        }

        __syncwarp();
    }

    // per-query sinks (KV split 0 only): an extra logit with a zero value, in the denominator only. Warp jh % nwarps
    // takes column c; the loop runs over compile-time column indices so the per-column state stays in registers.
    if (p.sinks && blockIdx.y == 0) {
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const int jq = c / ncols2;
            const int jh = c % ncols2;
            const int iq = ic0 + jq;
            // the warp that adds the sink of column c depends only on its Q head in the group (jh), never on the query's
            // place in the tile (jq): a query must get the same arithmetic alone and inside a verify batch (the sink
            // lands in one warp's partial sum, and the cross-warp sum below rounds by warp)
            if ((jh % nwarps) != warp || (ncols1 > 1 && iq >= p.n_q)) {
                continue;
            }
            const float sink = p.sinks[(head0 + jh) + iq*p.sink_s1 + sequence*p.sink_s2];

            const float kqmax_new = fmaxf(sink, KQ_max[c]);
            const float KQ_max_scale = expf(KQ_max[c] - kqmax_new);
            KQ_max[c] = kqmax_new;
            KQ_sum[c] = KQ_sum[c]*KQ_max_scale + (lane == 0 ? expf(sink - kqmax_new) : 0.0f);

            VKQ[c][0].x *= KQ_max_scale;
            VKQ[c][0].y *= KQ_max_scale;
            VKQ[c][1].x *= KQ_max_scale;
            VKQ[c][1].y *= KQ_max_scale;
        }
    }

    // merge the 4 warps (and the V_cols_per_iter row groups of a warp) per column
    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        if (warp == 0) {
            KQ_max_shared[c][lane] = -FLT_MAX/2.0f;
            KQ_sum_shared[c][lane] = 0.0f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        if (lane == 0) {
            KQ_max_shared[c][warp] = KQ_max[c];
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const int jq = c / ncols2;
        const int jh = c % ncols2;
        const int iq = ic0 + jq;
        if (ncols1 > 1 && iq >= p.n_q) {
            break;
        }

        const float kqmax_new   = warp_reduce_max(KQ_max_shared[c][lane]);
        const float kqmax_scale = expf(KQ_max[c] - kqmax_new);

        float * comb = KQ + (warp*V_cols_per_iter + (nthreads_V == WARP_SIZE ? 0 : lane / nthreads_V))*D;
        const int i = 4*(lane % nthreads_V);
        comb[i + 0] = VKQ[c][0].x*kqmax_scale;
        comb[i + 1] = VKQ[c][0].y*kqmax_scale;
        comb[i + 2] = VKQ[c][1].x*kqmax_scale;
        comb[i + 3] = VKQ[c][1].y*kqmax_scale;

        float s = warp_reduce_sum(KQ_sum[c]*kqmax_scale);
        if (lane == 0) {
            KQ_sum_shared[c][warp] = s;
        }

        __syncthreads();

        if (tid < D) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                sum += KQ_sum_shared[c][w];
            }
            float val = 0.0f;
#pragma unroll
            for (int l = 0; l < nwarps*V_cols_per_iter; ++l) {
                val += KQ[l*D + tid];
            }
            const int64_t row = (int64_t(sequence)*p.n_q + iq)*p.n_head + head0 + jh;
            if (gridDim.y == 1) {
                p.dst[row*D + tid] = sum == 0.0f ? 0.0f : val/sum;
            } else {
                p.dst[(row*gridDim.y + blockIdx.y)*D + tid] = val;
                if (tid == 0) {
                    p.dst_meta[row*gridDim.y + blockIdx.y] = make_float2(kqmax_new, sum);
                }
            }
        }

        __syncthreads();
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// merge the KV splits of one output row: fp32, 0 when no split saw a visible cell (and no sink)
template<int D>
__launch_bounds__(D, 1)
static __global__ void flash_attn_ext_banded_vec_merge(
        const float * __restrict__ parts, const float2 * __restrict__ meta, float * __restrict__ dst, const int n_split) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    parts += row*n_split*D;
    meta  += row*n_split;

    float m = -FLT_MAX/2.0f;
    for (int l = 0; l < n_split; ++l) {
        m = fmaxf(m, meta[l].x);
    }
    float num = 0.0f;
    float den = 0.0f;
    for (int l = 0; l < n_split; ++l) {
        const float2 ml = meta[l];
        const float  f  = expf(ml.x - m);
        num += f*parts[l*D + tid];
        den += f*ml.y;
    }
    dst[row*D + tid] = den == 0.0f ? 0.0f : num/den;
}

template<int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
static void banded_dec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];

    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    banded_dec_params p = {};
    p.Q     = (const char *) Q->data;
    p.K     = (const char *) K->data;
    p.V     = (const char *) V->data;
    p.mask  = mask ? (const char *) mask->data : nullptr;
    p.rel   = (const char *) rel->data;
    p.sinks = sinks ? (const float *) sinks->data : nullptr;
    memcpy(&p.scale, dst->op_params, sizeof(float));
    p.n_q    = Q->ne[1];
    p.n_head = Q->ne[2];
    p.gqa    = Q->ne[2] / K->ne[2];
    p.n_kv   = K->ne[1];
    p.nb01 = Q->nb[1]; p.nb02 = Q->nb[2]; p.nb03 = Q->nb[3];
    // K / V with one stream for all query streams (a unified cache, llama-server -np > 1): stride 0 over the streams
    p.nb11 = K->nb[1]; p.nb12 = K->nb[2]; p.nb13 = K->ne[3] == 1 ? 0 : K->nb[3];
    p.nb21 = V->nb[1]; p.nb22 = V->nb[2]; p.nb23 = V->ne[3] == 1 ? 0 : V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb32 = mask ? mask->nb[2] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;
    p.ne32 = mask ? mask->ne[2] : 1;
    p.ne33 = mask ? mask->ne[3] : 1;
    const ggml_tensor * rcur = dst->src[8];
    p.r_nb0 = rel->nb[0]; p.r_nb1 = rel->nb[1]; p.r_nb2 = rel->nb[2]; p.r_nb3 = rel->nb[3];
    p.r_ne3      = rcur ? 1 : rel->ne[3];
    // the tensor extent (not op_params) is authoritative after graph cloning: rel_logits [extent, ...] or the factor
    // table [d_rel, extent]
    p.rel_extent = (int) (rcur ? rel->ne[1] : rel->ne[0]);
    p.type_rel   = rel->type;
    p.rcur       = rcur ? (const float *) rcur->data : nullptr;
    p.d_rel      = rcur ? (int) rcur->ne[0] : 0;
    p.sink_s1 = sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0;
    p.sink_s2 = sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0;
    p.window  = ggml_get_op_params_i32(dst, 4);
    const ggml_tensor * pos_kv = dst->src[5];
    const ggml_tensor * pos_q  = dst->src[7];
    p.pos_q     = pos_q  ? (const int32_t *) pos_q->data  : nullptr;
    p.pos_kv    = pos_kv ? (const int32_t *) pos_kv->data : nullptr;
    p.pos_q_ns  = pos_q  ? (int) pos_q->ne[1]  : 1;
    p.pos_kv_ns = pos_kv ? (int) pos_kv->ne[1] : 1;

    const int ntiles_x  = (p.n_q + ncols1 - 1) / ncols1;
    const int n_groups  = p.n_head / ncols2;
    // tiles a block's loop can visit: with an implicit window only those the window reaches
    const int64_t n_kv_vis = p.window > 0 && !p.pos_kv ? std::min<int64_t>(p.n_kv, int64_t(p.window) + ncols1 - 1 + 4*WARP_SIZE) : p.n_kv;
    const int ntiles_KV = (int) ((n_kv_vis + 4*WARP_SIZE - 1) / (4*WARP_SIZE));

    const dim3 block_dim(WARP_SIZE, 4, 1);
    const auto kernel = flash_attn_ext_banded_vec<D, ncols1, ncols2, type_K, type_V>;

    // KV split as launch_fattn (fattn-common.cuh). Batch-invariant mode: the split depends on neither the number of
    // queries nor the KV length, so a query gives the same bits alone and in a verify batch with a longer view.
    const bool bi = ggml_cuda_batch_invariant();
    p.window_skip = bi ? 0 : 1;
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_dim.x*block_dim.y, 0));
    GGML_ASSERT(max_blocks_per_sm > 0);
    if (bi) {
        max_blocks_per_sm = 4;
    }
    const int ntiles_dst      = ntiles_x * n_groups * Q->ne[3];
    // batch-invariant: the KV split of a sequence does not depend on the query rows nor on the other sequences of the
    // batch (Q->ne[3] streams, llama-server -np > 1; with one stream this is n_groups * Q->ne[3] as before)
    const int ntiles_dst_eff  = bi ? n_groups : ntiles_dst;
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int parallel_blocks       = max_blocks_per_sm;
    const int parallel_blocks_max = bi ? std::max(parallel_blocks, 64) : std::max(1, ntiles_KV);
    parallel_blocks = std::min(parallel_blocks, parallel_blocks_max);
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int pb = parallel_blocks; pb <= parallel_blocks_max; ++pb) {
        const int nblocks_total = ntiles_dst_eff * pb;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = pb;
        }
    }

    ggml_cuda_pool_alloc<float>  dst_tmp(ctx.pool());
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(ctx.pool());
    const int64_t n_rows = int64_t(p.n_q) * p.n_head * Q->ne[3];
    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks * n_rows * D);
        dst_tmp_meta.alloc(parallel_blocks * n_rows);
        p.dst      = dst_tmp.ptr;
        p.dst_meta = dst_tmp_meta.ptr;
    } else {
        p.dst      = (float *) dst->data;
        p.dst_meta = nullptr;
    }

    const dim3 blocks_num(ntiles_x, parallel_blocks, n_groups * Q->ne[3]);
    kernel<<<blocks_num, block_dim, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        flash_attn_ext_banded_vec_merge<D><<<n_rows, D, 0, stream>>>(dst_tmp.ptr, dst_tmp_meta.ptr, (float *) dst->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<int D, int ncols2, ggml_type type_K, ggml_type type_V>
static void banded_dec_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n_q = dst->src[0]->ne[1];
    if (n_q == 1) {
        banded_dec_launch<D, 1, ncols2, type_K, type_V>(ctx, dst);
    } else if (n_q == 2) {
        banded_dec_launch<D, 2, ncols2, type_K, type_V>(ctx, dst);
    } else {
        banded_dec_launch<D, 4, ncols2, type_K, type_V>(ctx, dst);
    }
}

template<int D, ggml_type type_K, ggml_type type_V>
static void banded_dec_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    if (gqa % 3 == 0) {
        banded_dec_switch_ncols1<D, 3, type_K, type_V>(ctx, dst);
    } else {
        banded_dec_switch_ncols1<D, 1, type_K, type_V>(ctx, dst);
    }
}

template<int D>
static void banded_dec_switch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type tk = dst->src[1]->type;
    const ggml_type tv = dst->src[2]->type;
    if (tk == tv) {
        switch (tk) {
            case GGML_TYPE_F16:  banded_dec_switch_ncols2<D, GGML_TYPE_F16,  GGML_TYPE_F16 >(ctx, dst); return;
            case GGML_TYPE_Q8_0: banded_dec_switch_ncols2<D, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst); return;
            case GGML_TYPE_Q4_0: banded_dec_switch_ncols2<D, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst); return;
            default: break;
        }
    } else if constexpr (D == 128) {
        // mixed K/V (the "more context in VRAM" cache): K q8_0, V q4_0, head size 128 only
        if (tk == GGML_TYPE_Q8_0 && tv == GGML_TYPE_Q4_0) {
            banded_dec_switch_ncols2<D, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>(ctx, dst);
            return;
        }
    }
    GGML_ABORT("banded decode: unsupported K/V types");
}

// GGML_CUDA_BANDED_DEC=0: banded ops of 1-32 query rows take the f16/bf16/f32-only kernels of fattn-banded.cu instead
// (debugging and A/B timing; quantized K/V is then unsupported for those ops)
static bool ggml_cuda_banded_dec_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_BANDED_DEC");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

bool ggml_cuda_flash_attn_ext_banded_dec_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (!ggml_cuda_banded_dec_enabled()) {
        return false;
    }
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * m     = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];
    if (!q || !k || !v || !rel || q->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->ne[1] < 1 || q->ne[1] > ggml_cuda_banded_dec_max_q(k->type)) {
        return false;
    }
    // K/V type pairs: the list shared with the prefill kernel (fattn-banded.cuh); a mixed pair only at head size 128
    if (!fattn_banded_kv_pair_ok(k->type, v->type) || (k->type != v->type && q->ne[0] != 128)) {
        return false;
    }
    if (rel->type != GGML_TYPE_F32 && rel->type != GGML_TYPE_F16 && rel->type != GGML_TYPE_BF16) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 128) || k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0]) {
        return false;
    }
    if (k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0 || k->ne[1] != v->ne[1]) {
        return false;
    }
    if (k->ne[3] != v->ne[3] || (q->ne[3] != k->ne[3] && k->ne[3] != 1)) {
        return false;
    }
    if (k->ne[1] > INT32_MAX/2) {
        return false;
    }
    // row layout: contiguous rows; f16 V rows are read 8 bytes at a time (dequantize_V_f16), Q 8 bytes at a time
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) ||
        rel->nb[0] != ggml_type_size(rel->type)) {
        return false;
    }
    if ((uintptr_t) q->data % 8 != 0 || q->nb[1] % 8 != 0 || q->nb[2] % 8 != 0 || q->nb[3] % 8 != 0) {
        return false;
    }
    for (const ggml_tensor * t : { k, v }) {
        const size_t align = t->type == GGML_TYPE_F16 ? 8 : 2;
        if ((uintptr_t) t->data % align != 0 || t->nb[1] % align != 0 || t->nb[2] % align != 0 || t->nb[3] % align != 0) {
            return false;
        }
    }
    const ggml_tensor * rcur = dst->src[8];
    if (rcur) {
        // band bias from its factors: table F32 / F16 [d_rel, extent], rcur F32 [d_rel, n_head, n_q], both contiguous
        if ((rel->type != GGML_TYPE_F32 && rel->type != GGML_TYPE_F16) || !ggml_is_contiguous(rel) ||
            rel->ne[1] <= 0 || rel->ne[1] > INT32_MAX/2 || rel->ne[2] != 1 || rel->ne[3] != 1 ||
            rcur->type != GGML_TYPE_F32 || !ggml_is_contiguous(rcur) || rcur->ne[0] != rel->ne[0] ||
            rcur->ne[1] != q->ne[2] || rcur->ne[2] != q->ne[1] || rcur->ne[3] != 1 || q->ne[3] != 1) {
            return false;
        }
    } else if (rel->ne[0] <= 0 || rel->ne[0] > INT32_MAX/2 || rel->ne[1] != q->ne[2] || rel->ne[2] != q->ne[1] ||
        (rel->ne[3] != 1 && rel->ne[3] != q->ne[3])) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float) || sinks->ne[0] != q->ne[2] ||
                  (sinks->ne[1] != 1 && sinks->ne[1] != q->ne[1]) || (sinks->ne[2] != 1 && sinks->ne[2] != q->ne[3]))) {
        return false;
    }
    const int32_t window = ggml_get_op_params_i32(dst, 4);
    if (window < 0 || (window > 0 && m)) {
        return false;
    }
    // token positions (ring caches): both or none, with the implicit window
    const ggml_tensor * pos_kv = dst->src[5];
    const ggml_tensor * pos_q  = dst->src[7];
    if ((pos_kv != nullptr) != (pos_q != nullptr)) {
        return false;
    }
    if (pos_kv && (window <= 0 || m || pos_kv->type != GGML_TYPE_I32 || pos_q->type != GGML_TYPE_I32 ||
                   !ggml_is_contiguous(pos_kv) || !ggml_is_contiguous(pos_q) ||
                   pos_kv->ne[0] != k->ne[1] || pos_q->ne[0] != q->ne[1] ||
                   q->ne[3] % pos_kv->ne[1] != 0 || q->ne[3] % pos_q->ne[1] != 0)) {
        return false;
    }
    if (m && (m->type != GGML_TYPE_F16 || m->nb[0] != sizeof(half) || m->ne[0] < k->ne[1] || m->ne[1] < q->ne[1] ||
              q->ne[2] % m->ne[2] != 0 || q->ne[3] % m->ne[3] != 0)) {
        return false;
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

void ggml_cuda_flash_attn_ext_banded_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_dec_supported(ctx.device, dst));
    if (dst->src[0]->ne[0] == 64) {
        banded_dec_switch_type<64>(ctx, dst);
    } else {
        banded_dec_switch_type<128>(ctx, dst);
    }
}
