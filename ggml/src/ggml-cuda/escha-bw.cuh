#pragma once

#include <cstddef>

// Bandwidth meter of the AMD self-test (amd-test-5; ggml_backend_reg_get_proc_address "ggml_backend_cuda_bw_probe"):
// the device memory bandwidth that plain kernels reach on this card, as the line for the "% of measured bandwidth" of the
// Escha GEMV. On `device` with two buffers of `bytes` each, best of `reps` timed runs (after 2 warm-up runs), GB/s =
// 1e9 bytes per second:
//   res[0] read:   a read-only kernel (16-byte loads, 4 in flight per thread, grid-stride; the best of 3 grid sizes)
//   res[1] copy:   a copy kernel, read + write bytes counted (the same loads, 16-byte stores)
//   res[2] memcpy: cudaMemcpyAsync device to device, read + write bytes counted
//   res[3] the grid size of res[0] (blocks of 256 threads)
// Returns false (res unchanged) when the buffers cannot be allocated.
bool ggml_backend_cuda_bw_probe(int device, size_t bytes, int reps, double * res);
