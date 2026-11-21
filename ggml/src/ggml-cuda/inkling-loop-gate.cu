#include "inkling-loop-gate.cuh"

// Inkling 2 loop message controller (control()) in two launches instead of 11:
//   h   = down @ [rms(cur); rms(msg)]                      (rank R, kernel 1)
//   out = (1 - tanh(forget @ h))*cur + tanh(gain @ h)*msg  (kernel 2)
// The weights are F16 or F32. Every sum runs in a fixed order, so a token gives the same bits alone and in a batch.

template <typename W>
static __device__ __forceinline__ float w2f(const W v) {
    if constexpr (std::is_same_v<W, half>) {
        return __half2float(v);
    } else {
        return v;
    }
}

// one block per (rank row r, token t)
template <typename W, int NT>
static __global__ void inkling_loop_down(
        const float * __restrict__ cur, const float * __restrict__ msg, const W * __restrict__ down,
        float * __restrict__ h, const int E, const int R, const int64_t nb_cur1, const int64_t nb_msg1, const float eps) {
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;

    const float * c = (const float *) ((const char *) cur + (int64_t) t*nb_cur1);
    const float * m = (const float *) ((const char *) msg + (int64_t) t*nb_msg1);
    const W * wr = down + (int64_t) r*2*E;

    __shared__ float s_red[NT/WARP_SIZE][4];
    float sc = 0.0f, sm = 0.0f, dc = 0.0f, dm = 0.0f;
    for (int e = tid; e < E; e += NT) {
        const float a = c[e];
        const float b = m[e];
        sc += a*a;
        sm += b*b;
        dc += w2f(wr[e])*a;
        dm += w2f(wr[E + e])*b;
    }
    sc = warp_reduce_sum(sc);
    sm = warp_reduce_sum(sm);
    dc = warp_reduce_sum(dc);
    dm = warp_reduce_sum(dm);
    if (tid % WARP_SIZE == 0) {
        s_red[tid/WARP_SIZE][0] = sc;
        s_red[tid/WARP_SIZE][1] = sm;
        s_red[tid/WARP_SIZE][2] = dc;
        s_red[tid/WARP_SIZE][3] = dm;
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f, b = 0.0f, x = 0.0f, y = 0.0f;
        for (int w = 0; w < NT/WARP_SIZE; ++w) {
            a += s_red[w][0];
            b += s_red[w][1];
            x += s_red[w][2];
            y += s_red[w][3];
        }
        const float rc = rsqrtf(a/E + eps);
        const float rm = rsqrtf(b/E + eps);
        h[(int64_t) t*R + r] = x*rc + y*rm;
    }
}

// one warp per output channel e of a token; R <= 64
template <typename W>
static __global__ void inkling_loop_apply(
        const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ h,
        const W * __restrict__ gain, const W * __restrict__ forget, float * __restrict__ dst,
        const int E, const int R, const int64_t nb_cur1, const int64_t nb_msg1) {
    const int e = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int t = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    if (e >= E) {
        return;
    }
    const float * ht = h + (int64_t) t*R;
    float g = 0.0f, f = 0.0f;
    for (int k = lane; k < R; k += WARP_SIZE) {
        g += w2f(gain[(int64_t) e*R + k])*ht[k];
        f += w2f(forget[(int64_t) e*R + k])*ht[k];
    }
    g = warp_reduce_sum(g);
    f = warp_reduce_sum(f);
    if (lane == 0) {
        const float c = *(const float *) ((const char *) cur + (int64_t) t*nb_cur1 + (int64_t) e*sizeof(float));
        const float m = *(const float *) ((const char *) msg + (int64_t) t*nb_msg1 + (int64_t) e*sizeof(float));
        const float keep = __fmaf_rn(-1.0f, tanhf(f), 1.0f);
        dst[(int64_t) t*E + e] = __fadd_rn(__fmul_rn(keep, c), __fmul_rn(tanhf(g), m));
    }
}

template <typename W>
static void inkling_loop_gate_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * msg    = dst->src[1];
    const ggml_tensor * down   = dst->src[2];
    const ggml_tensor * gain   = dst->src[3];
    const ggml_tensor * forget = dst->src[4];

    const int E = cur->ne[0];
    const int T = cur->ne[1];
    const int R = down->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> h(ctx.pool(), (size_t) R*T);

    constexpr int NT = 256;
    inkling_loop_down<W, NT><<<dim3(R, T), NT, 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, (const W *) down->data, h.get(), E, R, cur->nb[1], msg->nb[1], eps);
    const int warps = 8;
    inkling_loop_apply<W><<<dim3((E + warps - 1)/warps, T), warps*WARP_SIZE, 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, h.get(), (const W *) gain->data, (const W *) forget->data,
        (float *) dst->data, E, R, cur->nb[1], msg->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_inkling_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * down = dst->src[2];
    if (down->type == GGML_TYPE_F16) {
        inkling_loop_gate_cuda<half>(ctx, dst);
    } else {
        GGML_ASSERT(down->type == GGML_TYPE_F32);
        inkling_loop_gate_cuda<float>(ctx, dst);
    }
}
