// ROCm stand-in for <cuda_fp16.h>, which exl3/quant/exl3_kernel_map.cuh includes. On the include path of
// escha-gemv.cu only, in the HIP build (ggml/src/ggml-hip/CMakeLists.txt).
#pragma once
#include <hip/hip_fp16.h>
