#include "head2.cuh"
#include "mmvq.cuh"

#include <atomic>
#include <mutex>
#include <vector>

#define HEAD2_BINS     4096   // 12 bits for each radix pass
#define HEAD2_SLACK    64     // room for tokens whose stage-1 logit equals that of candidate N

static std::mutex                         g_head2_mutex;
static std::vector<ggml_cuda_head2_entry> g_head2;
static std::atomic<int>                   g_head2_n{0};

void ggml_backend_cuda_head2_register(const void * q8_data, const void * q4_data, int n_candidates, int q1_type, const float * err2, int64_t n_rows) {
    float * err2_d = nullptr;
    if (err2 != nullptr && n_rows > 0) {
        CUDA_CHECK(cudaMalloc(&err2_d, (size_t) n_rows*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(err2_d, err2, (size_t) n_rows*sizeof(float), cudaMemcpyHostToDevice));
    }
    std::lock_guard<std::mutex> lock(g_head2_mutex);
    for (auto & e : g_head2) {
        if (e.q8 == q8_data) {
            e.q4     = q4_data;
            e.n_cand = n_candidates;
            e.type1  = (ggml_type) q1_type;
            if (e.err2 != nullptr) {
                cudaFree((void *) e.err2);
            }
            e.err2   = err2_d;
            return;
        }
    }
    g_head2.push_back({ q8_data, q4_data, (ggml_type) q1_type, err2_d, n_candidates });
    g_head2_n = (int) g_head2.size();
    GGML_LOG_INFO("%s: two-stage head: exact rows at %p (system RAM), stage 1 %s at %p (VRAM), %d candidates for each row\n",
        __func__, q8_data, ggml_type_name((ggml_type) q1_type), q4_data, n_candidates);
}

void ggml_backend_cuda_head2_unregister(const void * q8_data) {
    std::lock_guard<std::mutex> lock(g_head2_mutex);
    for (size_t i = 0; i < g_head2.size(); ++i) {
        if (g_head2[i].q8 == q8_data) {
            if (g_head2[i].err2 != nullptr) {
                cudaFree((void *) g_head2[i].err2);
            }
            g_head2.erase(g_head2.begin() + i);
            break;
        }
    }
    g_head2_n = (int) g_head2.size();
}

bool ggml_cuda_head2_find(const ggml_tensor * src0, ggml_cuda_head2_entry * out) {
    if (g_head2_n.load(std::memory_order_relaxed) == 0 || src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_head2_mutex);
    for (auto & e : g_head2) {
        if (e.q8 == src0->data) {
            *out = e;
            return true;
        }
    }
    return false;
}

bool ggml_cuda_head2_check_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_HEAD2_CHECK");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// larger float = larger key (the order of the stage-1 logits)
static __device__ __forceinline__ uint32_t head2_key(const float x) {
    const uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

static __global__ void k_head2_init(uint32_t * prefix, int32_t * need, int32_t * count, const int n_cand) {
    prefix[blockIdx.x] = 0;
    need[blockIdx.x]   = n_cand;
    count[blockIdx.x]  = 0;
}

// one radix pass: the histogram of the digit (key >> shift) of the keys that match the prefix found so far
static __global__ void k_head2_hist(const float * __restrict__ x, const int64_t stride_col, const int n,
        const uint32_t * __restrict__ prefix, uint32_t * __restrict__ hist, const int shift, const uint32_t hi_mask) {
    const int row = blockIdx.y;
    const int i   = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint32_t k = head2_key(x[row*stride_col + i]);
    if ((k & hi_mask) != (prefix[row] & hi_mask)) {
        return;
    }
    atomicAdd(&hist[(size_t) row*HEAD2_BINS + ((k >> shift) & (HEAD2_BINS - 1))], 1u);
}

// the digit of the key with rank need[row] (1 = the largest) among the keys that match the prefix; 64 threads
static __global__ void k_head2_scan(const uint32_t * __restrict__ hist, uint32_t * __restrict__ prefix, int32_t * __restrict__ need,
        uint32_t * __restrict__ eq, const int shift) {
    const int row = blockIdx.x;
    const uint32_t * h = hist + (size_t) row*HEAD2_BINS;
    __shared__ uint32_t s[64];
    uint32_t sum = 0;
    for (int j = 0; j < HEAD2_BINS/64; ++j) {
        sum += h[threadIdx.x*(HEAD2_BINS/64) + j];
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    int rem = need[row];
    int c   = 63;
    while (c > 0 && rem > (int) s[c]) {
        rem -= (int) s[c];
        --c;
    }
    int d = c*(HEAD2_BINS/64) + HEAD2_BINS/64 - 1;
    const int d_lo = c*(HEAD2_BINS/64);
    while (d > d_lo && rem > (int) h[d]) {
        rem -= (int) h[d];
        --d;
    }
    prefix[row] |= (uint32_t) d << shift;
    need[row]    = rem;      // the rank among the keys with this digit
    eq[row]      = h[d];     // after the last pass: the number of keys equal to the threshold key
}

// candidates of a row: keys above the threshold, and the keys equal to it when all of them fit in n_cap
static __global__ void k_head2_compact(const float * __restrict__ x, const int64_t stride_col, const int n,
        const uint32_t * __restrict__ thr, const int32_t * __restrict__ need, const uint32_t * __restrict__ eq,
        const int n_cand, const int n_cap, int32_t * __restrict__ count, int32_t * __restrict__ idx) {
    const int row = blockIdx.y;
    const int i   = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint32_t k = head2_key(x[row*stride_col + i]);
    const uint32_t t = thr[row];
    const bool take_eq = (n_cand - need[row]) + (int) eq[row] <= n_cap;
    if (k > t || (k == t && take_eq)) {
        const int slot = atomicAdd(&count[row], 1);
        idx[(size_t) row*n_cap + slot] = i;
    }
}

// tail correction, step 1: hn[row] = |h|^2 / n of one hidden row. One block of 256 threads; the order of the sums is fixed.
static __global__ void k_head2_hnorm(const float * __restrict__ h, const int64_t stride_col, const int n, float * __restrict__ hn) {
    const float * hr = h + blockIdx.x*stride_col;
    __shared__ float s[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += hr[i]*hr[i];
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if ((int) threadIdx.x < w) {
            s[threadIdx.x] += s[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        hn[blockIdx.x] = s[0]/n;
    }
}

// tail correction, step 2: logit -= 0.5 * (|h|^2 / n) * |w1 - w|^2 for all tokens (the candidates get exact values after it)
static __global__ void k_head2_tail(float * __restrict__ x, const int64_t stride_col, const int n, const float * __restrict__ hn,
        const float * __restrict__ err2) {
    const int row = blockIdx.y;
    const int i   = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    x[row*stride_col + i] -= 0.5f*hn[row]*err2[i];
}

// one block for each scratch row: copy the Q8_0 row of candidate c (rows past the count repeat a valid row)
static __global__ void k_head2_gather(const char * __restrict__ w, const size_t row_bytes, const int32_t * __restrict__ idx,
        const int32_t * __restrict__ count, char * __restrict__ out) {
    const int c   = blockIdx.x;
    const int n   = *count;
    const int src = c < n ? idx[c] : (n > 0 ? idx[0] : 0);
    const uint4 * s = (const uint4 *) (w + (size_t) src*row_bytes);
    uint4       * d = (uint4 *) (out + (size_t) c*row_bytes);
    for (size_t k = threadIdx.x; k < row_bytes/16; k += blockDim.x) {
        d[k] = s[k];
    }
}

static __global__ void k_head2_scatter(const float * __restrict__ v, const int32_t * __restrict__ idx, const int32_t * __restrict__ count,
        float * __restrict__ dst_col) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < *count) {
        dst_col[idx[i]] = v[i];
    }
}

// check mode, one block of 1,024 threads for each row: top token of both logit rows, and the bits of the candidates
static __global__ void k_head2_check(const float * __restrict__ a, const float * __restrict__ e, const int64_t stride_col, const int n,
        const int32_t * __restrict__ idx, const int32_t * __restrict__ count, const int n_cap, uint32_t * __restrict__ stats) {
    const int row = blockIdx.x;
    const float * ar = a + row*stride_col;
    const float * er = e + row*stride_col;
    __shared__ float sva[1024];
    __shared__ float sve[1024];
    __shared__ int   sia[1024];
    __shared__ int   sie[1024];
    float va = -INFINITY, ve = -INFINITY;
    int   ia = 0x7fffffff, ie = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (ar[i] > va) { va = ar[i]; ia = i; }
        if (er[i] > ve) { ve = er[i]; ie = i; }
    }
    sva[threadIdx.x] = va; sia[threadIdx.x] = ia;
    sve[threadIdx.x] = ve; sie[threadIdx.x] = ie;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if ((int) threadIdx.x < w) {
            const int o = threadIdx.x + w;
            if (sva[o] > sva[threadIdx.x] || (sva[o] == sva[threadIdx.x] && sia[o] < sia[threadIdx.x])) { sva[threadIdx.x] = sva[o]; sia[threadIdx.x] = sia[o]; }
            if (sve[o] > sve[threadIdx.x] || (sve[o] == sve[threadIdx.x] && sie[o] < sie[threadIdx.x])) { sve[threadIdx.x] = sve[o]; sie[threadIdx.x] = sie[o]; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats[0], 1u);
        if (sia[0] != sie[0]) {
            atomicAdd(&stats[1], 1u);
        }
    }
    const int nc = count[row];
    for (int i = threadIdx.x; i < nc; i += blockDim.x) {
        const int t = idx[(size_t) row*n_cap + i];
        atomicAdd(&stats[3], 1u);
        if (__float_as_uint(ar[t]) != __float_as_uint(er[t])) {
            atomicAdd(&stats[2], 1u);
        }
    }
}

// check mode, before stage 2: the rank of the exact top token in the stage-1 order (0 = stage 1 has the same top token),
// counted in buckets: rank < 1, 4, 16, 64, 128, 256, 512, 1024, 4096, and the rest. One block of 1,024 threads for each row.
static __global__ void k_head2_rank(const float * __restrict__ x, const float * __restrict__ e, const int64_t stride_col, const int n,
        uint32_t * __restrict__ buckets) {
    const int row = blockIdx.x;
    const float * xr = x + row*stride_col;
    const float * er = e + row*stride_col;
    __shared__ float sv[1024];
    __shared__ int   si[1024];
    float v = -INFINITY;
    int   iv = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (er[i] > v) { v = er[i]; iv = i; }
    }
    sv[threadIdx.x] = v; si[threadIdx.x] = iv;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if ((int) threadIdx.x < w) {
            const int o = threadIdx.x + w;
            if (sv[o] > sv[threadIdx.x] || (sv[o] == sv[threadIdx.x] && si[o] < si[threadIdx.x])) { sv[threadIdx.x] = sv[o]; si[threadIdx.x] = si[o]; }
        }
        __syncthreads();
    }
    const float t = xr[si[0]];
    __syncthreads();
    int c = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c += xr[i] > t;
    }
    si[threadIdx.x] = c;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if ((int) threadIdx.x < w) {
            si[threadIdx.x] += si[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int rank = si[0];
        const int lim[9] = { 1, 4, 16, 64, 128, 256, 512, 1024, 4096 };
        int b = 0;
        while (b < 9 && rank >= lim[b]) {
            ++b;
        }
        atomicAdd(&buckets[b], 1u);
    }
}

void ggml_cuda_head2_refine(ggml_backend_cuda_context & ctx, const ggml_cuda_head2_entry & h, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst, const float * exact) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

    cudaStream_t stream = ctx.stream();

    const int     n         = (int) src0->ne[1];       // tokens
    const int     rows      = (int) src1->ne[1];       // output rows
    const size_t  row_bytes = src0->nb[1];
    const int     n_cand    = std::max(1, std::min(h.n_cand, n));
    const int     n_cap     = n_cand + HEAD2_SLACK;
    const int64_t stride    = dst->nb[1]/sizeof(float);
    GGML_ASSERT(row_bytes % 16 == 0 && (uintptr_t) src0->data % 16 == 0);

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<uint32_t> hist(pool, (size_t) rows*HEAD2_BINS);
    ggml_cuda_pool_alloc<uint32_t> prefix(pool, rows);
    ggml_cuda_pool_alloc<int32_t>  need(pool, rows);
    ggml_cuda_pool_alloc<uint32_t> eq(pool, rows);
    ggml_cuda_pool_alloc<int32_t>  count(pool, rows);
    ggml_cuda_pool_alloc<int32_t>  idx(pool, (size_t) rows*n_cap);
    ggml_cuda_pool_alloc<char>     scratch(pool, (size_t) n_cap*row_bytes);
    ggml_cuda_pool_alloc<float>    exact_c(pool, n_cap);

    const float * x = (const float *) dst->data;

    static uint32_t * buckets_d = nullptr;
    if (exact != nullptr) {
        if (buckets_d == nullptr) {
            CUDA_CHECK(cudaMalloc(&buckets_d, 10*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(buckets_d, 0, 10*sizeof(uint32_t)));
        }
        k_head2_rank<<<rows, 1024, 0, stream>>>(x, exact, stride, n, buckets_d);
    }

    k_head2_init<<<rows, 1, 0, stream>>>(prefix.get(), need.get(), count.get(), n_cand);
    const dim3 grid_all((n + 255)/256, rows);
    static const int      shifts[3] = { 20, 8, 0 };
    static const uint32_t hi_mask[3] = { 0u, 0xFFF00000u, 0xFFFFFF00u };
    for (int p = 0; p < 3; ++p) {
        CUDA_CHECK(cudaMemsetAsync(hist.get(), 0, (size_t) rows*HEAD2_BINS*sizeof(uint32_t), stream));
        k_head2_hist<<<grid_all, 256, 0, stream>>>(x, stride, n, prefix.get(), hist.get(), shifts[p], hi_mask[p]);
        k_head2_scan<<<rows, 64, 0, stream>>>(hist.get(), prefix.get(), need.get(), eq.get(), shifts[p]);
    }
    k_head2_compact<<<grid_all, 256, 0, stream>>>(x, stride, n, prefix.get(), need.get(), eq.get(), n_cand, n_cap, count.get(), idx.get());
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<float> hn(pool);
    if (h.err2 != nullptr) {
        hn.alloc(rows);
        k_head2_hnorm<<<rows, 256, 0, stream>>>((const float *) src1->data, (int64_t) (src1->nb[1]/sizeof(float)), (int) src1->ne[0], hn.get());
        k_head2_tail<<<grid_all, 256, 0, stream>>>((float *) dst->data, stride, n, hn.get(), h.err2);
        CUDA_CHECK(cudaGetLastError());
    }

    // the scratch matrix and one column of src1 / one column of exact logits, as tensors for the stock kernel
    ggml_tensor w_s = *src0;
    w_s.ne[1]    = n_cap;
    w_s.nb[2]    = w_s.nb[1]*n_cap;
    w_s.nb[3]    = w_s.nb[2];
    w_s.data     = scratch.get();
    w_s.view_src = nullptr;

    ggml_tensor t_c = *dst;
    t_c.ne[0] = n_cap;
    t_c.ne[1] = 1;
    t_c.nb[0] = sizeof(float);
    t_c.nb[1] = (size_t) n_cap*sizeof(float);
    t_c.nb[2] = t_c.nb[1];
    t_c.nb[3] = t_c.nb[1];
    t_c.data  = exact_c.get();

    for (int r = 0; r < rows; ++r) {
        ggml_tensor y_r = *src1;
        y_r.ne[1] = 1;
        y_r.nb[2] = y_r.nb[1];
        y_r.nb[3] = y_r.nb[1];
        y_r.data  = (char *) src1->data + (size_t) r*src1->nb[1];

        t_c.src[0] = &w_s;
        t_c.src[1] = &y_r;

        k_head2_gather<<<n_cap, 128, 0, stream>>>((const char *) src0->data, row_bytes, idx.get() + (size_t) r*n_cap, count.get() + r, scratch.get());
        ggml_cuda_mul_mat_vec_q(ctx, &w_s, &y_r, nullptr, &t_c);
        k_head2_scatter<<<(n_cap + 255)/256, 256, 0, stream>>>(exact_c.get(), idx.get() + (size_t) r*n_cap, count.get() + r,
            (float *) dst->data + (size_t) r*stride);
    }
    CUDA_CHECK(cudaGetLastError());

    if (exact != nullptr) {
        static uint32_t * stats_d = nullptr;
        static uint32_t   seen_rows = 0, seen_top = 0, seen_bits = 0;
        if (stats_d == nullptr) {
            CUDA_CHECK(cudaMalloc(&stats_d, 4*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(stats_d, 0, 4*sizeof(uint32_t)));
        }
        k_head2_check<<<rows, 1024, 0, stream>>>((const float *) dst->data, exact, stride, n, idx.get(), count.get(), n_cap, stats_d);
        CUDA_CHECK(cudaGetLastError());
        uint32_t st[4];
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(st, stats_d, sizeof(st), cudaMemcpyDeviceToHost));
        if (seen_rows == 0 || st[0]/1024 != seen_rows/1024 || st[1] != seen_top || st[2] != seen_bits) {
            uint32_t bk[10];
            CUDA_CHECK(cudaMemcpy(bk, buckets_d, sizeof(bk), cudaMemcpyDeviceToHost));
            GGML_LOG_WARN("head2 check: rows %u, top token different %u, candidates %u, candidate logits with different bits %u; "
                "rank of the exact top token in stage 1: <1 %u, <4 %u, <16 %u, <64 %u, <128 %u, <256 %u, <512 %u, <1024 %u, <4096 %u, more %u\n",
                st[0], st[1], st[3], st[2], bk[0], bk[1], bk[2], bk[3], bk[4], bk[5], bk[6], bk[7], bk[8], bk[9]);
        }
        seen_rows = st[0];
        seen_top  = st[1];
        seen_bits = st[2];
    }
}

// one thread for each 16 bytes: chunk k of row r comes from src + start[r] + 16 k
static __global__ void k_gather_bench(const char * __restrict__ src, const int64_t * __restrict__ start, const int chunks_per_row,
        const int64_t n_chunks, uint4 * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_chunks) {
        return;
    }
    const int64_t r = i / chunks_per_row;
    const int     k = (int) (i % chunks_per_row);
    dst[i] = *(const uint4 *) (src + start[r] + (int64_t) k*16);
}

double ggml_backend_cuda_gather_bench(const void * src, size_t src_bytes, int row_bytes, int n_rows, int random, int repeat) {
    if (src == nullptr || row_bytes < 16 || row_bytes % 16 != 0 || n_rows <= 0 || repeat <= 0 || (size_t) row_bytes*(size_t) n_rows > src_bytes) {
        return 0.0;
    }
    const int64_t rows_in_src = (int64_t) (src_bytes/(size_t) row_bytes);
    std::vector<int64_t> start((size_t) n_rows);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (int r = 0; r < n_rows; ++r) {
        x = x*6364136223846793005ull + 1442695040888963407ull;
        const int64_t row = random ? (int64_t) ((x >> 16) % (uint64_t) rows_in_src) : r;
        start[(size_t) r] = row*row_bytes;
    }
    const int     cpr      = row_bytes/16;
    const int64_t n_chunks = (int64_t) n_rows*cpr;
    int64_t * start_d = nullptr;
    uint4   * dst_d   = nullptr;
    CUDA_CHECK(cudaMalloc(&start_d, (size_t) n_rows*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dst_d, (size_t) n_chunks*16));
    CUDA_CHECK(cudaMemcpy(start_d, start.data(), (size_t) n_rows*sizeof(int64_t), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const unsigned blocks = (unsigned) ((n_chunks + 255)/256);
    k_gather_bench<<<blocks, 256>>>((const char *) src, start_d, cpr, n_chunks, dst_d);   // first use
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < repeat; ++i) {
        k_gather_bench<<<blocks, 256>>>((const char *) src, start_d, cpr, n_chunks, dst_d);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(start_d));
    CUDA_CHECK(cudaFree(dst_d));
    return ms > 0.0f ? (double) n_chunks*16.0*repeat/(ms*1e-3)/1e9 : 0.0;
}
