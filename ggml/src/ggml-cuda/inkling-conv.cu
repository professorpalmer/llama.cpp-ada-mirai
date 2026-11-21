#include "inkling-conv.cuh"

// Inkling 2 short conv, one launch for the chain concat(past, cur^T) -> ssm_conv [-> 4 x (dynamic tap: mul, add)]
// -> add cur -> scale, and the new history (the last 3 time steps of [past, cur]) for the state store.
// dst = [out: C x T x S | history: 3 x C x S]. A thread has one channel of one sequence; the time loop is in order,
// so a token gives the same bits alone and in a batch.
static __global__ void inkling_conv_f32(
        const float * __restrict__ cur, const char * __restrict__ past, const float * __restrict__ w,
        const float * __restrict__ dyn, float * __restrict__ dst,
        const int C, const int T, const int S, const int H,
        const int64_t cur_nb1, const int64_t cur_nb2,
        const int64_t past_nb1, const int64_t past_nb2,
        const int64_t dyn_nb1, const int64_t dyn_nb2,
        const float scale) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= C) {
        return;
    }

    const float * pc = (const float *) (past + (int64_t) c*past_nb1 + (int64_t) s*past_nb2);
    const float * wc = w + (int64_t) c*4;

    float x[4] = { pc[0], pc[1], pc[2], 0.0f };
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wr[j] = wc[j];
    }
    const int head = H > 0 ? c / (C/H) : 0;

    float * out = dst;
    float * hist = dst + (int64_t) C*T*S;

    for (int t = 0; t < T; ++t) {
        const float xt = *(const float *) ((const char *) cur + (int64_t) t*cur_nb1 + (int64_t) s*cur_nb2 + (int64_t) c*sizeof(float));
        x[3] = xt;

        // ssm_conv: the same expression as ssm_conv_f32
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += x[j]*wr[j];
        }
        sumf += 0.0f;

        if (dyn) {
            const float * dt = (const float *) ((const char *) dyn + (int64_t) t*dyn_nb1 + (int64_t) s*dyn_nb2) + (int64_t) head*4;
            // lag l reads the window x[t + 3 - l]: the separate mul and add of the chain, no contraction
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                sumf = __fadd_rn(sumf, __fmul_rn(x[3 - l], dt[l]));
            }
        }

        out[((int64_t) s*T + t)*C + c] = __fmul_rn(scale, __fadd_rn(xt, sumf));

        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
    }

    // history: the last 3 steps of [past, cur], time fastest
    float * hc = hist + ((int64_t) s*C + c)*3;
    hc[0] = x[0];
    hc[1] = x[1];
    hc[2] = x[2];
}

void ggml_cuda_op_inkling_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur  = dst->src[0];
    const ggml_tensor * past = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * dyn  = dst->src[3];

    GGML_ASSERT(cur->type == GGML_TYPE_F32 && past->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(cur->nb[0] == sizeof(float) && past->nb[0] == sizeof(float) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));

    const int C = cur->ne[0];
    const int T = cur->ne[1];
    const int S = cur->ne[2];
    const int H = dyn ? dyn->ne[0]/4 : 0;

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int block = 128;
    const dim3 grid((C + block - 1)/block, S, 1);
    inkling_conv_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) cur->data, (const char *) past->data, (const float *) w->data,
        dyn ? (const float *) dyn->data : nullptr, (float *) dst->data,
        C, T, S, H, cur->nb[1], cur->nb[2], past->nb[1], past->nb[2],
        dyn ? dyn->nb[1] : 0, dyn ? dyn->nb[2] : 0, scale);
    CUDA_CHECK(cudaGetLastError());
}
