#pragma once

// ROCm shims for the exllamav3 headers in exl3/, for escha-gemv.cu in the HIP build. The functions below are copied from
// exllamav3 v1.6.7, exllamav3/exllamav3_ext/rocm/compat.h (commit 8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0, MIT License,
// Copyright (c) 2025 Turboderp, see ../LICENSE); only the ones that the headers used by escha-gemv.cu need.
// The warp-sync shuffle macros of that file are not copied: ggml-cuda/vendors/hip.h has the same forms.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// __nanosleep only paces spin-waits. s_sleep takes an immediate in units of 64 clocks, so it cannot follow a
// runtime duration; the shortest sleep keeps the back-off close to the CUDA one instead of overshooting it.
// Declared for the host pass too, which parses device function bodies

__device__ __forceinline__ void __nanosleep(unsigned int)
{
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#endif
}

// __dp4a, unsigned form only (the only one the kernels use: byte sums against 0x01010101). A signed call
// fails to resolve rather than getting the wrong extension

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
#if !defined(EXL3_FORCE_SCALAR_DOT)
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_udot4))
        return __builtin_amdgcn_udot4(a, b, c, false);
#endif
    #pragma unroll
    for (int i = 0; i < 4; ++i) c += ((a >> (8 * i)) & 0xffu) * ((b >> (8 * i)) & 0xffu);
    return c;
}

// HIP has __hmax2/__hmin2 for bfloat162 only

__device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)), __hmin(__high2half(a), __high2half(b)));
}

// __ldcs (streaming, evict-first) is only a cache hint; a nontemporal load is the RDNA counterpart.
// __ldcg (cache at L2, bypassing L1) is load-bearing: the kernels use it to read values other blocks wrote,
// and RDNA's per-CU L0 is just as incoherent as NVIDIA's L1. A relaxed agent-scope atomic load reads at the
// coherent level

template <typename T>
__device__ __forceinline__ T __ldcs(const T* p)
{
    return __builtin_nontemporal_load(p);
}

