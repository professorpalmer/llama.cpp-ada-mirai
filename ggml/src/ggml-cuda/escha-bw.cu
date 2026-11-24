#include "common.cuh"
#include "escha-bw.cuh"

// See escha-bw.cuh. The kernels read and write int4 (16 bytes) per thread and step; 4 loads are issued before their
// values are used, so each thread has 64 bytes in flight.

static __global__ void escha_bw_read_kernel(const int4 * __restrict__ src, const size_t n, int * __restrict__ sink) {
    const size_t stride = (size_t) gridDim.x*blockDim.x;
    size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    int acc = 0;
    for (; i + 3*stride < n; i += 4*stride) {
        const int4 a = src[i];
        const int4 b = src[i + stride];
        const int4 c = src[i + 2*stride];
        const int4 d = src[i + 3*stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) {
        const int4 a = src[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    // the buffer holds zeros: no thread writes, but the compiler cannot drop the loads
    if (acc == 0x5a5a5a5a) {
        sink[blockIdx.x] = acc;
    }
}

static __global__ void escha_bw_copy_kernel(const int4 * __restrict__ src, int4 * __restrict__ dst, const size_t n) {
    const size_t stride = (size_t) gridDim.x*blockDim.x;
    size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    for (; i + 3*stride < n; i += 4*stride) {
        const int4 a = src[i];
        const int4 b = src[i + stride];
        const int4 c = src[i + 2*stride];
        const int4 d = src[i + 3*stride];
        dst[i]            = a;
        dst[i + stride]   = b;
        dst[i + 2*stride] = c;
        dst[i + 3*stride] = d;
    }
    for (; i < n; i += stride) {
        dst[i] = src[i];
    }
}

bool ggml_backend_cuda_bw_probe(int device, size_t bytes, int reps, double * res) {
    ggml_cuda_set_device(device);
    bytes &= ~(size_t) 15;
    void * a = nullptr;
    void * b = nullptr;
    int  * sink = nullptr;
    if (cudaMalloc(&a, bytes) != cudaSuccess || cudaMalloc(&b, bytes) != cudaSuccess || cudaMalloc(&sink, 65536*sizeof(int)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (a) { CUDA_CHECK(cudaFree(a)); }
        if (b) { CUDA_CHECK(cudaFree(b)); }
        return false;
    }
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMemset(b, 0, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const size_t n = bytes/16;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    reps = reps < 1 ? 1 : reps;

    // best time of reps runs (seconds) of fn after 2 warm-up runs
    auto best = [&](auto fn) {
        double t_best = 1e30;
        for (int r = 0; r < reps + 2; ++r) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            fn();
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 2 && ms > 0.0f) {
                t_best = std::min(t_best, 1e-3*ms);
            }
        }
        return t_best;
    };

    double t_read = 1e30;
    int g_read = 0;
    for (const int mul : {4, 16, 64}) {
        const int grid = std::max(1, std::min(65535, nsm*mul));
        const double t = best([&] {
            escha_bw_read_kernel<<<grid, 256, 0, stream>>>((const int4 *) a, n, sink);
            CUDA_CHECK(cudaGetLastError());
        });
        if (t < t_read) {
            t_read = t;
            g_read = grid;
        }
    }
    const double t_copy = best([&] {
        escha_bw_copy_kernel<<<g_read, 256, 0, stream>>>((const int4 *) a, (int4 *) b, n);
        CUDA_CHECK(cudaGetLastError());
    });
    const double t_memcpy = best([&] {
        CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, stream));
    });

    res[0] = 1e-9*bytes/t_read;
    res[1] = 2e-9*bytes/t_copy;
    res[2] = 2e-9*bytes/t_memcpy;
    res[3] = g_read;

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(sink));
    return true;
}
