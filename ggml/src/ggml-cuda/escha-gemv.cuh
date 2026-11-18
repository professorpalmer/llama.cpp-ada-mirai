#include "common.cuh"

// small-batch path of ggml_escha_mul_mat (rows <= 8). Returns false if it does not take the call.
bool ggml_cuda_escha_gemv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The two rotations alone, one warp per 128 channels, for the prefill path in escha-moe.cu. Each returns false if it does not take the call (rows not 16-byte aligned).
//   xh[row]  = f16(T128(x[row] * rin))
//   dst[row] = T128(mid[row]) * rout, mid is [rows][OC] and can be the memory of dst
bool ggml_cuda_escha_had_in(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * rin, half * xh);
bool ggml_cuda_escha_had_out(ggml_backend_cuda_context & ctx, const float * mid, const ggml_tensor * rout, ggml_tensor * dst);
