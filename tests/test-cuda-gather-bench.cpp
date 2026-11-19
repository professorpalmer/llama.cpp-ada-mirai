// Rate of a kernel that copies small rows from pinned system RAM (a tiered buffer with no VRAM part) into VRAM, against
// the same copy from VRAM. No model file. For the design of reads from a K/V cache or a head that is in system RAM.
//   test-cuda-gather-bench [buffer MiB = 2048] [MiB moved for each launch = 256]
// Prints GB/s for row sizes from 144 bytes (one K or V head of one cell at q4_0) to 1 MiB, rows in order and at random
// places. Without a GPU device or without tiered buffers it prints SKIP.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    const size_t buf_mib  = argc > 1 ? (size_t) atoll(argv[1]) : 2048;
    const size_t move_mib = argc > 2 ? (size_t) atoll(argv[2]) : 256;

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = ggml_backend_dev_get(i);
            break;
        }
    }
    if (!dev) {
        printf("SKIP: no GPU device\n");
        return 0;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    using tier_fn_t  = ggml_backend_buffer_type_t (*)(int, double, int, const char *);
    using bench_fn_t = double (*)(const void *, size_t, int, int, int, int);
    auto * tier  = reg ? (tier_fn_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_tier_buffer_type") : nullptr;
    auto * bench = reg ? (bench_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_gather_bench") : nullptr;
    if (!tier || !bench) {
        printf("SKIP: the backend of %s has no tiered buffers or no gather benchmark\n", ggml_backend_dev_name(dev));
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("SKIP: cannot start %s\n", ggml_backend_dev_name(dev));
        return 0;
    }

    const size_t bytes = buf_mib << 20;
    ggml_backend_buffer_type_t buft_host = tier(0, 0.0, 1, "nostage_bench");
    ggml_backend_buffer_t buf_host = buft_host ? ggml_backend_buft_alloc_buffer(buft_host, bytes) : nullptr;
    ggml_backend_buffer_t buf_vram = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), bytes);
    if (!buf_host || !buf_vram) {
        printf("SKIP: cannot allocate %zu MiB of pinned system RAM and of VRAM\n", buf_mib);
        return 0;
    }
    ggml_backend_buffer_clear(buf_host, 1);
    ggml_backend_buffer_clear(buf_vram, 1);

    printf("buffer %zu MiB, %zu MiB moved for each launch, 8 launches for each cell; GB/s\n", buf_mib, move_mib);
    printf("%10s | %14s %14s | %14s %14s\n", "row bytes", "RAM in order", "RAM random", "VRAM in order", "VRAM random");
    const int sizes[] = { 144, 576, 1152, 2304, 5440, 18432, 65536, 1048576 };
    for (int row : sizes) {
        const int n_rows = (int) ((move_mib << 20)/(size_t) row);
        double r[4];
        int k = 0;
        ggml_backend_buffer_t bufs[2] = { buf_host, buf_vram };
        for (int ib = 0; ib < 2; ++ib) {
            for (int random = 0; random < 2; ++random) {
                r[k++] = bench(ggml_backend_buffer_get_base(bufs[ib]), bytes, row, n_rows, random, 8);
            }
        }
        printf("%10d | %14.1f %14.1f | %14.1f %14.1f\n", row, r[0], r[1], r[2], r[3]);
        fflush(stdout);
    }
    ggml_backend_buffer_free(buf_host);
    ggml_backend_buffer_free(buf_vram);
    ggml_backend_free(backend);
    return 0;
}
