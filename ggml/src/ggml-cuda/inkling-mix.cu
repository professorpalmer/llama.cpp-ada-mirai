#include "inkling-mix.cuh"

// Inkling 2 depth mix in one launch: out[:, q, t] = sum_i p[i, q, t] * rms(s_i[:, t]),
// p[:, q, t] = softmax_i(scale * rms(s_i[:, t]) . query[:, q]). One block per token; the sums run in a fixed order,
// so a token gives the same bits alone and in a batch.
#define INKLING_MIX_MAX_Q 4

template <int NT>
static __global__ void inkling_mix_f32(
        const inkling_mix_srcs srcs, const float * __restrict__ query, float * __restrict__ dst,
        const int n, const int nq, const int E, const float eps, const float scale) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_red[NT/WARP_SIZE][1 + INKLING_MIX_MAX_Q];
    __shared__ float s_rs[GGML_INKLING_MIX_MAX_STATES];
    __shared__ float s_p[GGML_INKLING_MIX_MAX_STATES][INKLING_MIX_MAX_Q];

    // pass 1: for each state its sum of squares and its dot products with the queries
    float dots[GGML_INKLING_MIX_MAX_STATES][INKLING_MIX_MAX_Q];
    for (int i = 0; i < n; ++i) {
        const float * x = (const float *) (srcs.data[i] + (int64_t) t*srcs.nb1[i]);
        float ss = 0.0f;
        float d[INKLING_MIX_MAX_Q] = { 0.0f };
        for (int e = tid; e < E; e += NT) {
            const float v = x[e];
            ss += v*v;
#pragma unroll
            for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
                if (q < nq) {
                    d[q] += v*query[(int64_t) q*E + e];
                }
            }
        }
        ss = warp_reduce_sum(ss);
#pragma unroll
        for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
            d[q] = warp_reduce_sum(d[q]);
        }
        if (tid % WARP_SIZE == 0) {
            s_red[tid/WARP_SIZE][0] = ss;
#pragma unroll
            for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
                s_red[tid/WARP_SIZE][1 + q] = d[q];
            }
        }
        __syncthreads();
        if (tid == 0) {
            float a = 0.0f;
            float b[INKLING_MIX_MAX_Q] = { 0.0f };
            for (int w = 0; w < NT/WARP_SIZE; ++w) {
                a += s_red[w][0];
#pragma unroll
                for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
                    b[q] += s_red[w][1 + q];
                }
            }
            const float rs = rsqrtf(a/E + eps);
            s_rs[i] = rs;
#pragma unroll
            for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
                dots[i][q] = b[q]*rs*scale;
            }
        }
        __syncthreads();
    }

    // softmax over the states, for each query
    if (tid == 0) {
        for (int q = 0; q < nq; ++q) {
            float m = -INFINITY;
            for (int i = 0; i < n; ++i) {
                m = fmaxf(m, dots[i][q]);
            }
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float e = expf(dots[i][q] - m);
                s_p[i][q] = e;
                sum += e;
            }
            const float inv = 1.0f/sum;
            for (int i = 0; i < n; ++i) {
                s_p[i][q] *= inv;
            }
        }
    }
    __syncthreads();

    // pass 2: the weighted sum of the normalized states
    for (int e = tid; e < E; e += NT) {
        float acc[INKLING_MIX_MAX_Q] = { 0.0f };
        for (int i = 0; i < n; ++i) {
            const float * x = (const float *) (srcs.data[i] + (int64_t) t*srcs.nb1[i]);
            const float v = x[e]*s_rs[i];
#pragma unroll
            for (int q = 0; q < INKLING_MIX_MAX_Q; ++q) {
                acc[q] += v*s_p[i][q];
            }
        }
        for (int q = 0; q < nq; ++q) {
            dst[((int64_t) t*nq + q)*E + e] = acc[q];
        }
    }
}

void ggml_cuda_op_inkling_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];
    const int n  = ggml_get_op_params_i32(dst, 0);
    const float eps   = ggml_get_op_params_f32(dst, 1);
    const float scale = ggml_get_op_params_f32(dst, 2);
    const int E  = query->ne[0];
    const int nq = query->ne[1];
    const int T  = dst->ne[2];

    GGML_ASSERT(n >= 1 && n <= GGML_INKLING_MIX_MAX_STATES && nq >= 1 && nq <= INKLING_MIX_MAX_Q);
    GGML_ASSERT(ggml_is_contiguous(query) && ggml_is_contiguous(dst));

    inkling_mix_srcs srcs = {};
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * s = dst->src[1 + i];
        GGML_ASSERT(s->type == GGML_TYPE_F32 && s->nb[0] == sizeof(float) && s->ne[0] == E);
        srcs.data[i] = (const char *) s->data;
        srcs.nb1[i]  = s->nb[1];
    }

    constexpr int NT = 256;
    inkling_mix_f32<NT><<<T, NT, 0, ctx.stream()>>>(srcs, (const float *) query->data, (float *) dst->data, n, nq, E, eps, scale);
    CUDA_CHECK(cudaGetLastError());
}
