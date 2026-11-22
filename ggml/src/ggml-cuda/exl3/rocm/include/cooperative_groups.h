// ROCm stand-in for <cooperative_groups.h>, which exl3/quant/exl3_gemv_kernel.cuh includes (escha-gemv.cu does not
// use the cooperative kernel: it runs the three stages as plain launches). On the include path of escha-gemv.cu only,
// in the HIP build (ggml/src/ggml-hip/CMakeLists.txt).
#pragma once
#include <hip/hip_cooperative_groups.h>
