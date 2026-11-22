# exllamav3 kernel headers

Unmodified copies from [exllamav3](https://github.com/turboderp-org/exllamav3), path `exllamav3/exllamav3_ext/`, commit 8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0 (v1.6.7). MIT License, Copyright (c) 2025 Turboderp, see `LICENSE`.

Used by `../escha-gemv.cu` (small-batch GEMV of the Escha op). The directory layout is the same as upstream because the headers include each other with relative paths.

`rocm/` holds the ROCm (HIP) part, used by the same file in the HIP build:

- `rocm/ptx_rocm.cuh`: unmodified copy of exllamav3 `exllamav3_ext/rocm/ptx_rocm.cuh` (same commit). It emulates the tensor-core fragment operations (`mma.sync m16n8k16`, `ldmatrix.x4`) with lane shuffles and FMAs, so the kernels run on RDNA (wave32) unchanged.
- `rocm/compat_ggml.cuh`: the functions from exllamav3 `exllamav3_ext/rocm/compat.h` (same commit) that these headers need (`__nanosleep`, `__dp4a`, `__hmax2` / `__hmin2` for `__half2`, `__ldcs`).
- `rocm/include/`: stand-ins for `<cuda_fp16.h>` and `<cooperative_groups.h>`, on the include path of `escha-gemv.cu` only (`ggml/src/ggml-hip/CMakeLists.txt`).
