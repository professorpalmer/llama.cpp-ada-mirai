#include "escha-moe-prefill.cuh"
#include "mmid.cuh"

// Prefill path of ggml_escha_moe: routed trellis experts, batches of more than 8 tokens.
//
//   y = T128(T128(x * rin[e]) @ decode(code[e])) * rout[e]   for the rows routed to expert e
//
// The arithmetic is the one of the routed GEMV path (escha_gemv_kernel<bits, 1, 2, 8, true, true> in escha-gemv.cu, the
// main loop of exllamav3's exl3_gemv_kernel, MIT, Copyright (c) 2025 Turboderp): a 16x16 trellis tile is decoded into
// fp16 fragments, multiplied with m16n8k16 fp16-accumulate MMAs, folded to fp32 every 4 k-slices; k is cut into 8
// splits of ceil(k_slices / 8) slices; a row's result is 0 + p0 + ... + p7 over the splits in order, then T128 and rout.
// That path runs 8 rows per block (A rows 8-15 of the MMA are zero) and so decodes every tile once per 8 rows.
//
// Here a block has one expert, 128 output columns and up to 64 rows of that expert (4 MMA row fragments of 16 rows,
// both A halves live). A warp has 32 columns (2 n-tiles) and walks the 8 k-splits one after the other with the same
// slice ranges, the same fold cadence and the same split order, decodes each tile once and runs it on all row
// fragments. A row's sums do not depend on the other rows of an MMA, so every row is the same bits as through the
// 1-token path (test-escha-moe --invariant and --prefill check it). Decode work drops by up to 8x.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// The exllamav3 headers take these from their util.cuh and util.h (as in escha-gemv.cu).
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))

typedef struct __align__(8) half4 {
    half2 x;
    half2 y;
} half4;

union half2_uint32 {
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
};

union half_uint16 {
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
};

#include "exl3/quant/exl3_gemv_kernel.cuh"

#define ESCHA_PREFILL_RS 0.088388347648f   // 1/sqrt(128), the fp32 constant of the format (ESCHA_GEMV_RS)

// the routed GEMV configuration; the bits of a row depend on these, so they must stay equal to escha-gemv.cu's
static constexpr int ESCHA_PF_S    = 8;   // k-splits
static constexpr int ESCHA_PF_WNT  = 2;   // n-tiles per warp
static constexpr int ESCHA_PF_FOLD = 4;   // fp16 -> fp32 fold cadence of WNT 2
static constexpr int ESCHA_PF_PF   = 4;   // prefetch ring depth of WNT 2 (latency only)
static constexpr int ESCHA_PF_RF   = 4;   // row fragments of 16 rows per block
static constexpr int ESCHA_PF_ROWS = 16*ESCHA_PF_RF;

// xh[row] = f16(T128(x[row] * rin[expert of row])), row = token*n_ids + slot: escha_gemv_had_in_routed of escha-gemv.cu
static __global__ void escha_prefill_had_in_routed(
        const float   * __restrict__ x,
        const half    * __restrict__ rin,
        const int32_t * __restrict__ ids,
        half          * __restrict__ xh,
        const int IC, const int n_rows, const int n_x, const int n_ids,
        const int64_t nb_x1, const int64_t nb_x2, const int64_t nb_i0, const int64_t nb_i1) {
    const int nb = IC/128;
    const int w  = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    if (w >= n_rows*nb) {
        return;
    }
    const int row = w/nb;
    const int blk = w%nb;
    const int it  = row/n_ids;
    const int is  = row%n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);

    const float * x_row = (const float *)((const char *) x + (int64_t)(is%n_x)*nb_x1 + (int64_t) it*nb_x2);
    had_fh_r_128_inner<true, false>(x_row + blk*128, xh + (int64_t) row*IC + blk*128, rin + (int64_t) e*IC + blk*128, ESCHA_PREFILL_RS);
}

// one work item for each chunk of up to ESCHA_PF_ROWS compact rows of an expert (the items write different rows)
static __global__ void escha_prefill_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += ESCHA_PF_ROWS) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(ESCHA_PF_ROWS, hi - s), 0);
        }
    }
}

// grid: [items][size_n/128] in one dimension, 128 threads = 4 warps = 4 column groups of 32 columns
template <int bits>
static __global__ void __launch_bounds__(128, 3) escha_moe_prefill_kernel(
        const half     * __restrict__ xh,
        const uint16_t * __restrict__ B_all,
        const half     * __restrict__ rout_all,
        float          * __restrict__ dst,
        const int size_k, const int size_n, const int ne1,
        const int64_t nb_d1, const int64_t nb_d2,
        const int32_t * __restrict__ ids_dst,
        const int4    * __restrict__ work,
        const int32_t * __restrict__ n_work) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int S    = ESCHA_PF_S;
    constexpr int WNT  = ESCHA_PF_WNT;
    constexpr int PF   = ESCHA_PF_PF;
    constexpr int FOLD = ESCHA_PF_FOLD;
    constexpr int RF   = ESCHA_PF_RF;
    constexpr int ROWS = ESCHA_PF_ROWS;
    constexpr int G    = 4;
    constexpr int cb   = 1;   // mcg

    constexpr int TWORDS = 8*bits;                               // uint32 per 16x16 tile
    constexpr bool TWO_PER_LOAD = bits == 2;                     // two tiles per warp load
    constexpr int LOADS   = TWO_PER_LOAD ? WNT/2 : WNT;          // warp loads per k-slice
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2*TWORDS : TWORDS;    // uint32 per load (lanes < LSTRIDE load)
    static_assert(bits == 2 || bits == 3, "2 and 3 bpw only");
    static_assert(G*WNT*16 == 128, "a block has one Hadamard block of the output");

    const int n_cb = size_n/128;
    const int item = blockIdx.x/n_cb;
    const int bx   = blockIdx.x%n_cb;
    // the grid has an upper bound of the item count. The test is the same for all threads of the block
    if (item >= *n_work) {
        return;
    }
    const int4 wi     = work[item];
    const int  e      = wi.x;
    const int  r_s    = wi.y;
    const int  size_m = wi.z;
    const int  n_rf   = (size_m + 15)/16;

    const uint16_t * B    = B_all    + (size_t) e*(size_k/16)*(size_n/16)*(16*bits);
    const half     * rout = rout_all + (size_t) e*size_n;

    const int warp = threadIdx.x/32;   // column group
    const int lane = threadIdx.x%32;

    const int ntiles  = size_n/16;
    const int kslices = size_k/16;
    const int chunk   = CEIL_DIVIDE(kslices, S);

    const uint32_t * B32 = (const uint32_t *) B;
    const size_t slice_stride = (size_t) ntiles*TWORDS;   // uint32 per k-slice row
    const half2 * A2 = (const half2 *) xh;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A rows of this lane: fragment rf has rows rf*16 + lane/4 (MMA a0 / a2) and rf*16 + lane/4 + 8 (a1 / a3)
    const int g = lane >> 2;
    int  a_row[RF][2];
    bool a_ok[RF][2];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = rf*16 + g + 8*h;
            a_ok[rf][h]  = r < size_m;
            a_row[rf][h] = a_ok[rf][h] ? ids_dst[r_s + r] : 0;
        }
    }

    // per-lane extraction constants (as escha_gemv_kernel)
    int x_src_a = 0, x_src_b = 0;
    [[maybe_unused]] int x_s2 = 0;
    if constexpr (bits == 2) {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257)*3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16)/32;
        const int i2 = (b2 - 1)/32;
        x_s2 = (i2 + 1)*32 - b2;
        x_src_a = i0%24;
        x_src_b = i2%24;
    }

    // the running row sums over the splits, in split order (the GEMV's sum = 0 + sh_red[0] + ... + sh_red[S-1])
    __shared__ __align__(16) float sh_sum[ROWS][128];

    const int group = bx*G + warp;
    const int c0    = warp*WNT*16 + 2*(lane & 3);   // first column of this lane in the block's 128

    for (int ksp = 0; ksp < S; ++ksp) {
        const int ks0 = ksp*chunk;
        const int myn = max(0, min(chunk, kslices - ks0));

        const uint32_t * bp = B32 + (size_t) ks0*slice_stride + group*WNT*TWORDS + lane;
        auto ld_b = [&] (int i, int l) -> uint32_t {
            if constexpr (LSTRIDE < 32) {
                return lane < LSTRIDE ? __ldcs(bp + (size_t) i*slice_stride + l*LSTRIDE) : 0;
            } else {
                return __ldcs(bp + (size_t) i*slice_stride + l*LSTRIDE);
            }
        };

        uint32_t pf[PF][LOADS];
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            if (d < myn) {
#pragma unroll
                for (int l = 0; l < LOADS; ++l) {
                    pf[d][l] = ld_b(d, l);
                }
            }
        }

        FragC_h ch[RF][WNT][2];
        float4  acc[RF][WNT][2];   // x, y: row g; z, w: row g + 8
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
#pragma unroll
            for (int t = 0; t < WNT; ++t) {
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    ch[rf][t][f][0] = hzero;
                    ch[rf][t][f][1] = hzero;
                    acc[rf][t][f] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }

        for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
            for (int d = 0; d < PF; ++d) {
                const int i = ib + d;
                if (i >= myn) {
                    break;
                }

                uint32_t bw[LOADS];
#pragma unroll
                for (int l = 0; l < LOADS; ++l) {
                    bw[l] = pf[d][l];
                }
                if (i + PF < myn) {
#pragma unroll
                    for (int l = 0; l < LOADS; ++l) {
                        pf[d][l] = ld_b(i + PF, l);
                    }
                }

                // decode the WNT tiles of this k-slice once
                FragB fb[WNT][2];
#pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    if constexpr (bits == 2) {
                        const uint32_t w = bw[t >> 1];
                        const int base = (t & 1) << 4;
                        const uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                        const uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                        exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, fb[t][0], fb[t][1]);
                    } else {
                        const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                        const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                        exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, fb[t][0], fb[t][1]);
                    }
                }

                // A fragments: lane covers rows g and g + 8 of each fragment, k pairs (2(lane%4), +1) and (+8, +9)
                const size_t a_col = (size_t) (ks0 + i)*8 + (lane & 3);
#pragma unroll
                for (int rf = 0; rf < RF; ++rf) {
                    if (rf < n_rf) {
                        FragB a01, a23;
                        a01[0] = a_ok[rf][0] ? A2[(size_t) a_row[rf][0]*(size_k/2) + a_col]     : hzero;
                        a01[1] = a_ok[rf][1] ? A2[(size_t) a_row[rf][1]*(size_k/2) + a_col]     : hzero;
                        a23[0] = a_ok[rf][0] ? A2[(size_t) a_row[rf][0]*(size_k/2) + a_col + 4] : hzero;
                        a23[1] = a_ok[rf][1] ? A2[(size_t) a_row[rf][1]*(size_k/2) + a_col + 4] : hzero;
#pragma unroll
                        for (int t = 0; t < WNT; ++t) {
                            exl3_gemv_ns::mma_ab_h(a01, a23, fb[t][0], ch[rf][t][0]);
                            exl3_gemv_ns::mma_ab_h(a01, a23, fb[t][1], ch[rf][t][1]);
                        }
                    }
                }

                if ((d + 1)%FOLD == 0 || i + 1 == myn) {
#pragma unroll
                    for (int rf = 0; rf < RF; ++rf) {
#pragma unroll
                        for (int t = 0; t < WNT; ++t) {
#pragma unroll
                            for (int f = 0; f < 2; ++f) {
                                acc[rf][t][f].x += __low2float (ch[rf][t][f][0]);
                                acc[rf][t][f].y += __high2float(ch[rf][t][f][0]);
                                acc[rf][t][f].z += __low2float (ch[rf][t][f][1]);
                                acc[rf][t][f].w += __high2float(ch[rf][t][f][1]);
                                ch[rf][t][f][0] = hzero;
                                ch[rf][t][f][1] = hzero;
                            }
                        }
                    }
                }
            }
        }

        // add this split's partial to the row sums (a lane owns its elements: no barrier between the splits)
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
            if (rf < n_rf) {
#pragma unroll
                for (int t = 0; t < WNT; ++t) {
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int col = c0 + t*16 + f*8;
                        float * s0 = &sh_sum[rf*16 + g    ][col];
                        float * s1 = &sh_sum[rf*16 + g + 8][col];
                        const float4 p = acc[rf][t][f];
                        s0[0] = (ksp == 0 ? 0.0f : s0[0]) + p.x;
                        s0[1] = (ksp == 0 ? 0.0f : s0[1]) + p.y;
                        s1[0] = (ksp == 0 ? 0.0f : s1[0]) + p.z;
                        s1[1] = (ksp == 0 ? 0.0f : s1[1]) + p.w;
                    }
                }
            }
        }
    }
    __syncthreads();

    // T128 and rout of each row, one warp per row (as the GEMV's epilogue)
    for (int r = warp; r < size_m; r += G) {
        const int orow = ids_dst[r_s + r];
        float * dst_row = (float *)((char *) dst + (int64_t)(orow%ne1)*nb_d1 + (int64_t)(orow/ne1)*nb_d2);
        had_ff_r_128_inner<false, true>(sh_sum[r], dst_row + bx*128, rout + bx*128, ESCHA_PREFILL_RS);
    }
#else
    GGML_UNUSED_VARS(xh, B_all, rout_all, dst, size_k, size_n, ne1, nb_d1, nb_d2, ids_dst, work, n_work);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static bool escha_prefill_rows_aligned(const ggml_tensor * t) {
    return (uintptr_t) t->data%16 == 0 && t->nb[1]%16 == 0 && t->nb[2]%16 == 0;
}

// GGML_ESCHA_MOE_PREFILL=0: batches go to the routed GEMV path (8-row chunks). Read at every call, so a test can
// compare the two paths in one process.
static bool escha_prefill_enabled() {
    const char * e = getenv("GGML_ESCHA_MOE_PREFILL");
    return e == nullptr || atoi(e) != 0;
}

bool ggml_cuda_escha_moe_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * x    = dst->src[5];
    const ggml_tensor * ids  = dst->src[6];

    const int K  = code->ne[0]/16;
    const int OC = code->ne[1]*16;
    const int IC = code->ne[2]*16;

    const int n_expert = code->ne[3];
    const int n_ids    = ids->ne[0];
    const int n_tokens = ids->ne[1];
    const int n_rows   = n_ids*n_tokens;
    const int n_x      = x->ne[1];

    if (!escha_prefill_enabled() || n_tokens <= 8 || (K != 2 && K != 3) || IC%128 != 0 || OC%128 != 0) {
        return false;
    }
    // the GEMV environment switches keep their meaning: GGML_ESCHA_MOE_GEMV=0 means the block kernels for every batch
    static const bool gemv_on = [] {
        const char * e = getenv("GGML_ESCHA_MOE_GEMV");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!gemv_on || !ampere_mma_available(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    if (!escha_prefill_rows_aligned(x) || !escha_prefill_rows_aligned(dst) || (uintptr_t) rin->data%8 != 0 || (uintptr_t) rout->data%8 != 0 ||
            ids->nb[0] != ggml_element_size(ids) || ids->nb[1]%ggml_element_size(ids) != 0 || (n_x != 1 && n_x != n_ids)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> xh(ctx.pool(), (size_t) n_rows*IC);
    escha_prefill_had_in_routed<<<(n_rows*(IC/128) + 7)/8, 256, 0, stream>>>(
        (const float *) x->data, (const half *) rin->data, (const int32_t *) ids->data, xh.get(),
        IC, n_rows, n_x, n_ids, x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);
    CUDA_CHECK(cudaGetLastError());

    // compact rows grouped by expert (top-k routing: an expert at most once per token)
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        n_expert, n_tokens, n_ids, n_x, (int) (ids->nb[1]/ggml_element_size(ids)), 1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    // an expert can end with a part chunk, so the item count has a bound only
    const int n_items = (n_rows + ESCHA_PF_ROWS - 1)/ESCHA_PF_ROWS + n_expert;
    ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_items);
    ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
    escha_prefill_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert);
    CUDA_CHECK(cudaGetLastError());

    auto launch = [&](auto kernel) {
        kernel<<<n_items*(OC/128), 128, 0, stream>>>(
            xh.get(), (const uint16_t *) code->data, (const half *) rout->data, (float *) dst->data,
            IC, OC, n_ids, dst->nb[1], dst->nb[2], ids_dst.get(), work.get(), n_work.get());
    };
    if (K == 2) {
        launch(escha_moe_prefill_kernel<2>);
    } else {
        launch(escha_moe_prefill_kernel<3>);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_escha_moe_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
