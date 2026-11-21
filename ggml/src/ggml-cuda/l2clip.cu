#include "l2clip.cuh"

// y = v * clamp(th / (sqrt(sum(v*v)) + 1e-6), 0, 1), v = x * scale: one warp per row.
// The steps of the chain mul, sqr, sum_rows, sqrt, scale_bias, fill, div, clamp, mul (Inkling 2 clips of q, k and r).
// The sum order does not depend on the row count, so a row gives the same bits alone and in a batch.
static __global__ void l2_clip_f32(
        const char * __restrict__ x, const float * __restrict__ scale, float * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t nrows,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t sne1, const int64_t sne2, const int64_t sne3,
        const int64_t snb1, const int64_t snb2, const int64_t snb3,
        const float th) {
    const int64_t row = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / (ne1*ne2);

    const float * xr = (const float *) (x + i1*nb1 + i2*nb2 + i3*nb3);
    const float * sr = scale ? (const float *) ((const char *) scale + (i1 % sne1)*snb1 + (i2 % sne2)*snb2 + (i3 % sne3)*snb3) : nullptr;
    float       * yr = dst + row*ne0;

    float sum = 0.0f;
    for (int64_t i = lane; i < ne0; i += WARP_SIZE) {
        const float v = sr ? xr[i]*sr[i] : xr[i];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    const float norm = 1.0f*sqrtf(sum) + 1e-6f;
    const float f    = fminf(fmaxf(th/norm, 0.0f), 1.0f);

    for (int64_t i = lane; i < ne0; i += WARP_SIZE) {
        const float v = sr ? xr[i]*sr[i] : xr[i];
        yr[i] = v*f;
    }
}

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0  = dst->src[0];
    const ggml_tensor * scale = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));
    if (scale) {
        GGML_ASSERT(scale->type == GGML_TYPE_F32 && scale->nb[0] == sizeof(float) && scale->ne[0] == src0->ne[0]);
    }

    const float th = ggml_get_op_params_f32(dst, 0);

    const int64_t nrows = ggml_nrows(src0);
    const int warps_per_block = 4;
    const int64_t nblocks = (nrows + warps_per_block - 1)/warps_per_block;

    l2_clip_f32<<<nblocks, warps_per_block*WARP_SIZE, 0, ctx.stream()>>>(
        (const char *) src0->data, scale ? (const float *) scale->data : nullptr, (float *) dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], nrows,
        src0->nb[1], src0->nb[2], src0->nb[3],
        scale ? scale->ne[1] : 1, scale ? scale->ne[2] : 1, scale ? scale->ne[3] : 1,
        scale ? scale->nb[1] : 0, scale ? scale->nb[2] : 0, scale ? scale->nb[3] : 0,
        th);
    CUDA_CHECK(cudaGetLastError());
}
