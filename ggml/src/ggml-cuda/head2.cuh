#pragma once

#include "common.cuh"

// Two-stage output head (ggml_backend_cuda_head2_register).
//
// The exact head W (Q8_0, [n_embd, n_vocab]) stays in pinned system RAM that is mapped into the device address space
// (a tiered buffer with no VRAM part). A small copy W1 of it (Q4_0, or a lower-bit type) is in VRAM. A MUL_MAT with W as its matrix runs as:
//   stage 1  logits = W1 . h for all tokens (the stock kernels)
//   stage 2  for each output row: the N tokens with the largest stage-1 logits are found on the device, their rows of W
//            are copied from system RAM into a VRAM scratch matrix, the stock Q8_0 matrix-vector kernel (one column)
//            gives their exact logits, and those replace the stage-1 values. All other tokens keep stage-1 logits,
//            less a bias term (tail correction): a stage-1 logit has the error e = h . (w1 - w), about normal with
//            variance |h|^2 |w1 - w|^2 / n, and exp() of it is too large by exp(variance / 2) on average. Without
//            the term the tokens outside the candidates get too much probability (KL +0.3 % with Q2_K, N = 1024).
// The candidates of a row and their exact logits depend on that row only. The exact values have the bits of the
// one-column Q8_0 kernel, which is what the single-pass head gives for 1 to 4 columns, and for 1 to 48 columns in
// batch-invariant mode.
// GGML_CUDA_HEAD2_CHECK=1: also runs the single-pass head from system RAM (slow) and counts rows with a different top
// token and candidates with different bits (log lines; CUDA graphs are off in this mode).

struct ggml_cuda_head2_entry {
    const void * q8;       // data of W (the registered MUL_MAT matrix)
    const void * q4;       // data of W1 (stage 1; any quantized type the stock kernels read)
    ggml_type    type1;    // type of W1
    const float * err2;    // device, [n_vocab]: squared error of each W1 row against W (nullptr: no tail correction)
    int          n_cand;   // N
};

// reached through ggml_backend_reg_get_proc_address
// err2 (host, n_rows floats, may be nullptr): the squared error of each stage-1 row; the backend keeps a device copy
void ggml_backend_cuda_head2_register(const void * q8_data, const void * q1_data, int n_candidates, int q1_type, const float * err2, int64_t n_rows);
void ggml_backend_cuda_head2_unregister(const void * q8_data);

// Benchmark (ggml_backend_reg_get_proc_address "ggml_backend_cuda_gather_bench"): rows of row_bytes are copied by a
// kernel from src (device address; pinned system RAM of a tiered buffer, or VRAM) to a VRAM scratch buffer, n_rows rows
// at pseudo-random places (random != 0) or one after the other, `repeat` times. Returns GB/s (0 on bad arguments).
double ggml_backend_cuda_gather_bench(const void * src, size_t src_bytes, int row_bytes, int n_rows, int random, int repeat);

bool ggml_cuda_head2_find(const ggml_tensor * src0, ggml_cuda_head2_entry * out);
bool ggml_cuda_head2_check_enabled();

// Stage 2 on dst, which holds the stage-1 logits of src1's columns.
// exact != nullptr (check mode): the single-pass logits of the same columns, in the layout of dst.
void ggml_cuda_head2_refine(ggml_backend_cuda_context & ctx, const ggml_cuda_head2_entry & h, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst, const float * exact);
