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
// Here a block has one expert, 128 output columns and up to 16*RF rows of that expert (RF MMA row fragments of 16 rows,
// both A halves live). A warp has 32 columns (2 n-tiles) and walks the 8 k-splits one after the other with the same
// slice ranges, the same fold cadence and the same split order, decodes each tile once and runs it on all row
// fragments. The A slices of the block go through a cp.async ring in shared memory, shared by the 4 warps, and reach
// the MMAs by ldmatrix: the same fp16 values as the GEMV's direct loads. A row's sums do not depend on the other rows
// of an MMA, so every row is the same bits as through the 1-token path (test-escha-moe --invariant and --prefill).

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
// latency only
static constexpr int ESCHA_PF_PF   = 4;   // prefetch ring depth (B in registers, A in shared memory)
static constexpr int ESCHA_PF_AST  = 24;  // halves per A row in shared memory (16 used): ldmatrix without bank conflicts

// GGML_ESCHA_MOE_PREFILL_RF=2|4: row fragments of 16 rows per block (default 4)
static int escha_prefill_rf() {
    static const int rf = [] {
        const char * e = getenv("GGML_ESCHA_MOE_PREFILL_RF");
        return e != nullptr && atoi(e) == 2 ? 2 : 4;
    }();
    return rf;
}

// GGML_ESCHA_MOE_PREFILL_SUMS=smem|dst: where the running row sums live (default dst)
static bool escha_prefill_sums_dst() {
    static const bool d = [] {
        const char * e = getenv("GGML_ESCHA_MOE_PREFILL_SUMS");
        return e == nullptr || strcmp(e, "smem") != 0;
    }();
    return d;
}

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

// one work item for each chunk of up to `rows` compact rows of an expert (the items write different rows)
static __global__ void escha_prefill_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert, const int rows) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += rows) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(rows, hi - s), 0);
        }
    }
}

static __device__ __forceinline__ void escha_pf_cp_async_16(void * dst, const void * src, const bool valid) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned int d = (unsigned int) __cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;   // 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" : : "r"(d), "l"(src), "r"(n));
#else
    *(int4 *) dst = valid ? *(const int4 *) src : make_int4(0, 0, 0, 0);
#endif
}

static __device__ __forceinline__ void escha_pf_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif
}

template <int n>
static __device__ __forceinline__ void escha_pf_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif
}

// grid: [items][size_n/128] in one dimension, 128 threads = 4 warps = 4 column groups of 32 columns
// SUMS_DST: the running row sums live in the output rows (no 128-column sum buffer in shared memory: more blocks per SM)
template <int bits, int RF, bool SUMS_DST>
static __global__ void __launch_bounds__(128, SUMS_DST ? (RF == 4 ? 3 : 4) : (RF == 4 ? 2 : 4)) escha_moe_prefill_kernel(
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
    constexpr int AST  = ESCHA_PF_AST;
    constexpr int ROWS = 16*RF;
    constexpr int G    = 4;
    constexpr int cb   = 1;   // mcg

    constexpr int TWORDS = 8*bits;                               // uint32 per 16x16 tile
    constexpr bool TWO_PER_LOAD = bits == 2;                     // two tiles per warp load
    constexpr int LOADS   = TWO_PER_LOAD ? WNT/2 : WNT;          // warp loads per k-slice
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2*TWORDS : TWORDS;    // uint32 per load (lanes < LSTRIDE load)
    static_assert(bits == 2 || bits == 3, "2 and 3 bpw only");
    static_assert(G*WNT*16 == 128, "a block has one Hadamard block of the output");
    static_assert(RF == 2 || RF == 4, "2 or 4 row fragments");
    static_assert(2*ROWS <= 32*G, "one 16-byte copy per thread per A slice");

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
    const int g    = lane >> 2;

    const int ntiles  = size_n/16;
    const int kslices = size_k/16;
    const int chunk   = CEIL_DIVIDE(kslices, S);

    const uint32_t * B32 = (const uint32_t *) B;
    const size_t slice_stride = (size_t) ntiles*TWORDS;   // uint32 per k-slice row
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A copies: thread t < 2*ROWS copies 16 bytes (8 k values) of row t/2 for each k-slice; rows past size_m are zeros
    const bool   cp_on  = threadIdx.x < 2*ROWS;
    const int    cp_row = threadIdx.x/2;
    const int    cp_h   = threadIdx.x%2;
    const bool   cp_ok  = cp_on && cp_row < size_m;
    const half * cp_src = xh + (cp_ok ? (size_t) ids_dst[r_s + cp_row]*size_k : 0) + cp_h*8;

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
    __shared__ __align__(16) float sh_sum[SUMS_DST ? 1 : ROWS][128];
    // the A ring: PF k-slices of ROWS rows x 16 k
    __shared__ __align__(16) half  sh_a[PF][ROWS][AST];

    // ldmatrix x4 address of this lane in a 16-row fragment: matrices (rows 0-7, k 0-7), (rows 8-15, k 0-7),
    // (rows 0-7, k 8-15), (rows 8-15, k 8-15) = the m16n8k16 A registers a0, a1, a2, a3
    const int lm_row = (lane & 7) + ((lane >> 3) & 1)*8;
    const int lm_col = (lane >> 4)*8;

    const int group = bx*G + warp;
    const int c0    = warp*WNT*16 + 2*(lane & 3);   // first column of this lane in the block's 128

    // the 128 output columns of the block in compact row r of the item
    auto out_row = [&](const int r) -> float * {
        const int orow = ids_dst[r_s + r];
        return (float *)((char *) dst + (int64_t)(orow%ne1)*nb_d1 + (int64_t)(orow/ne1)*nb_d2) + bx*128;
    };

    for (int ksp = 0; ksp < S; ++ksp) {
        const int ks0 = ksp*chunk;
        const int myn = max(0, min(chunk, kslices - ks0));

        // every warp is done with the A ring of the last split
        __syncthreads();

        auto a_issue = [&](const int i, const int slot) {
            if (cp_on && i < myn) {
                escha_pf_cp_async_16(&sh_a[slot][cp_row][cp_h*8], cp_ok ? cp_src + (size_t) (ks0 + i)*16 : xh, cp_ok);
            }
            escha_pf_cp_async_commit();
        };
#pragma unroll
        for (int d = 0; d < PF - 1; ++d) {
            a_issue(d, d);
        }

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

                // A slice i (slot d) has landed for every thread; every warp is done with slice i - 1 (slot d - 1)
                escha_pf_cp_async_wait<PF - 2>();
                __syncthreads();
                a_issue(i + PF - 1, (d + PF - 1)%PF);

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

#pragma unroll
                for (int rf = 0; rf < RF; ++rf) {
                    if (rf < n_rf) {
                        uint32_t a[4];
                        const unsigned int addr = (unsigned int) __cvta_generic_to_shared(&sh_a[d][rf*16 + lm_row][lm_col]);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                            : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
                        FragB a01, a23;
                        a01[0] = *(const half2 *) &a[0];
                        a01[1] = *(const half2 *) &a[1];
                        a23[0] = *(const half2 *) &a[2];
                        a23[1] = *(const half2 *) &a[3];
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
        escha_pf_cp_async_wait<0>();

        // add this split's partial to the row sums (a lane owns its elements)
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
            if (rf < n_rf) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = rf*16 + g + 8*h;
                    float * srow;
                    if constexpr (SUMS_DST) {
                        // rows of other items are not ours to write
                        if (r >= size_m) {
                            continue;
                        }
                        srow = out_row(r);
                    } else {
                        srow = sh_sum[r];
                    }
#pragma unroll
                    for (int t = 0; t < WNT; ++t) {
#pragma unroll
                        for (int f = 0; f < 2; ++f) {
                            float2 * s = (float2 *) (srow + c0 + t*16 + f*8);
                            const float4 p = acc[rf][t][f];
                            float2 v = ksp == 0 ? make_float2(0.0f, 0.0f) : *s;
                            v.x += h == 0 ? p.x : p.z;
                            v.y += h == 0 ? p.y : p.w;
                            *s = v;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // T128 and rout of each row, one warp per row (as the GEMV's epilogue; in place when the sums are in dst)
    for (int r = warp; r < size_m; r += G) {
        float * o = out_row(r);
        if constexpr (SUMS_DST) {
            had_ff_r_128_inner<false, true>(o, o, rout + bx*128, ESCHA_PREFILL_RS);
        } else {
            had_ff_r_128_inner<false, true>(sh_sum[r], o, rout + bx*128, ESCHA_PREFILL_RS);
        }
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

// GGML_ESCHA_MOE_PREFILL_MIN=n: batches of fewer than n tokens stay on the GEMV path (more warps per row there).
// Read at every call (tests). Default: see ggml_cuda_escha_moe_prefill.
static int escha_prefill_min_tokens(const int def) {
    const char * e = getenv("GGML_ESCHA_MOE_PREFILL_MIN");
    return e != nullptr ? atoi(e) : def;
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

    if (!escha_prefill_enabled() || n_tokens <= 8 || n_tokens < escha_prefill_min_tokens(9) ||
            (K != 2 && K != 3) || IC%128 != 0 || OC%128 != 0) {
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

    const int  rf       = escha_prefill_rf();
    const int  rows     = 16*rf;
    const bool sums_dst = escha_prefill_sums_dst();

    // an expert can end with a part chunk, so the item count has a bound only
    const int n_items = (n_rows + rows - 1)/rows + n_expert;
    ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_items);
    ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
    escha_prefill_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert, rows);
    CUDA_CHECK(cudaGetLastError());

    auto launch = [&](auto kernel) {
        kernel<<<n_items*(OC/128), 128, 0, stream>>>(
            xh.get(), (const uint16_t *) code->data, (const half *) rout->data, (float *) dst->data,
            IC, OC, n_ids, dst->nb[1], dst->nb[2], ids_dst.get(), work.get(), n_work.get());
    };
#define ESCHA_PF_LAUNCH(bits) \
    if (sums_dst) { \
        if (rf == 4) { launch(escha_moe_prefill_kernel<bits, 4, true>);  } else { launch(escha_moe_prefill_kernel<bits, 2, true>);  } \
    } else { \
        if (rf == 4) { launch(escha_moe_prefill_kernel<bits, 4, false>); } else { launch(escha_moe_prefill_kernel<bits, 2, false>); } \
    }
    if (K == 2) {
        ESCHA_PF_LAUNCH(2)
    } else {
        ESCHA_PF_LAUNCH(3)
    }
#undef ESCHA_PF_LAUNCH
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
