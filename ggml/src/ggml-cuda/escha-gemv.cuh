#include "common.cuh"

// small-batch path of ggml_escha_mul_mat (rows <= 8). Returns false if it does not take the call.
bool ggml_cuda_escha_gemv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
