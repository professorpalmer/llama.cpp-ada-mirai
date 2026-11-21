#include "common.cuh"

struct inkling_mix_srcs {
    const char * data[GGML_INKLING_MIX_MAX_STATES];
    int64_t      nb1[GGML_INKLING_MIX_MAX_STATES];
};

void ggml_cuda_op_inkling_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
