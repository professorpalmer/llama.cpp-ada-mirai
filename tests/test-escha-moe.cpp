// Checks ggml_escha_moe against numpy references built by dump_op_case.py.
//
// The reference path folds each expert to a dense matrix and does a plain matmul,
// so it shares no code with the op, which decodes tiles on the fly and rotates the
// activations instead. Agreement means the fused form is right, not just repeatable.
//
// Every case runs on the CPU and, when one is present, on the GPU. Both are scored
// against the same reference rather than against each other, so a shared bug in the
// two ggml paths cannot hide.
//
//   usage: test-escha-moe [path-to-escha-moe-cases.gguf]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * DEFAULT_CASES =
    "/media/aj-homeserver/GAMES1/llm_models/escha-gguf-port/escha-moe-cases.gguf";

struct escha_case {
    ggml_tensor * code;
    ggml_tensor * rin;
    ggml_tensor * rout;
    ggml_tensor * lut;
    ggml_tensor * dep;
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * yref;
};

static ggml_tensor * need(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "missing tensor '%s' in the case file\n", name.c_str());
        exit(1);
    }
    return t;
}

// run one case on `backend`, or on the CPU inline when backend is null
static std::vector<float> run_case(const escha_case & c, ggml_backend_t backend) {
    const int64_t n = ggml_nelements(c.yref);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32 + ggml_graph_overhead() + (backend ? 0 : 64u*1024*1024),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ backend != nullptr,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * src[7] = { c.code, c.rin, c.rout, c.lut, c.dep, c.x, c.ids };
    ggml_tensor * dev[7];
    for (int i = 0; i < 7; ++i) {
        dev[i] = backend ? ggml_dup_tensor(ctx, src[i]) : src[i];
    }

    ggml_tensor * y = ggml_escha_moe(ctx, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    std::vector<float> out(n);

    if (backend) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "failed to allocate backend buffer\n");
            exit(1);
        }
        for (int i = 0; i < 7; ++i) {
            ggml_backend_tensor_set(dev[i], src[i]->data, 0, ggml_nbytes(src[i]));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "backend compute failed\n");
            exit(1);
        }
        ggml_backend_tensor_get(y, out.data(), 0, n*sizeof(float));
        ggml_backend_buffer_free(buf);
    } else {
        ggml_graph_compute_with_ctx(ctx, gf, 12);
        std::copy((const float *) y->data, (const float *) y->data + n, out.begin());
    }

    ggml_free(ctx);
    return out;
}

// The op computes codebook A rather than reading the table, so the table it is handed
// has to BE codebook A. A checkpoint on one of escha's other codebooks would otherwise
// decode to plausible-looking garbage.
static bool check_codebook(const ggml_tensor * lut) {
    if (ggml_nelements(lut) != 65536 || lut->type != GGML_TYPE_F16) {
        printf("lut is not (65536,) f16\n");
        return false;
    }

    const ggml_fp16_t * t = (const ggml_fp16_t *) lut->data;
    int64_t bad = 0;

    for (uint32_t i = 0; i < 65536; ++i) {
        const uint32_t x = ((i*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
        ggml_fp16_t lo, hi;
        memcpy(&lo, (const char *) &x,     sizeof(lo));
        memcpy(&hi, (const char *) &x + 2, sizeof(hi));

        const float v = ggml_fp16_to_fp32(lo) + ggml_fp16_to_fp32(hi);
        if (ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) != ggml_fp16_to_fp32(t[i])) {
            bad++;
        }
    }

    printf("codebook: closed form vs shipped table  %s (%" PRId64 " of 65536 differ)\n\n",
           bad ? "FAIL" : "exact", bad);
    return bad == 0;
}

// The CUDA op takes the exllamav3 GEMV kernel (escha-gemv.cu) when the shape allows: it rounds the rotated
// activations to fp16, as Escha's own runtime does, and adds up to 64 products in fp16 before each fold to fp32.
// The line for it is the line of the dense GEMV test (test-escha-mul-mat.cpp): 1e-3.
// GGML_ESCHA_MOE_GEMV=0 selects the fp32 block kernels and the 1e-5 line.
static double tolerance_for(bool cuda) {
    static const bool gemv = getenv("GGML_ESCHA_MOE_GEMV") == nullptr || atoi(getenv("GGML_ESCHA_MOE_GEMV")) != 0;
    return cuda && gemv ? 1e-3 : 1e-5;
}

// a case with the tokens t0, t0 + 1, ... (modulo the token count of c) of case c; x and ids are new tensors in ctx
static escha_case sub_case(ggml_context * ctx, const escha_case & c, int64_t t0, int64_t n_tokens) {
    escha_case r = c;
    const int64_t nt_src = c.ids->ne[1];

    r.x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.x->ne[0], c.x->ne[1], n_tokens);
    r.ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.ids->ne[0], n_tokens);
    r.yref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.yref->ne[0], c.yref->ne[1], n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t ts = (t0 + t)%nt_src;
        memcpy((char *) r.x->data    + t*r.x->nb[2],    (const char *) c.x->data    + ts*c.x->nb[2],    c.x->nb[2]);
        memcpy((char *) r.ids->data  + t*r.ids->nb[1],  (const char *) c.ids->data  + ts*c.ids->nb[1],  c.ids->nb[1]);
        memcpy((char *) r.yref->data + t*r.yref->nb[2], (const char *) c.yref->data + ts*c.yref->nb[2], c.yref->nb[2]);
    }
    return r;
}

// max abs difference and relative RMS against the reference
static bool score(const char * label, uint32_t ic, int64_t K, int64_t IC, int64_t OC,
                  const std::vector<float> & got, const ggml_tensor * yref) {
    const float * b = (const float *) yref->data;
    const int64_t n = ggml_nelements(yref);

    double max_abs = 0.0, sse = 0.0, ssr = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) got[i] - (double) b[i];
        max_abs = std::max(max_abs, std::fabs(d));
        sse += d*d;
        ssr += (double) b[i] * (double) b[i];
    }
    const double rel = std::sqrt(sse/ssr);

    // f32 reassociation only: the op runs a butterfly Hadamard and accumulates per
    // tile, the reference multiplies by a dense folded matrix
    const bool ok = rel < tolerance_for(strcmp(label, "cuda") == 0);

    printf("case %u  %-4s  K=%" PRId64 "  %4" PRId64 " -> %-4" PRId64
           "  max_abs=%.3e  rel_rms=%.3e  %s\n",
           ic, label, K, IC, OC, max_abs, rel, ok ? "OK" : "FAIL");
    return ok;
}

// wall-clock per op, so kernel changes can be judged without a 4 minute model load
static void bench(const escha_case & c, ggml_backend_t backend, uint32_t ic) {
    const int64_t IC = c.code->ne[2]*16;
    const int64_t OC = c.code->ne[1]*16;
    const int64_t decodes = ggml_nelements(c.ids)*IC*OC;

    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * src[7] = { c.code, c.rin, c.rout, c.lut, c.dep, c.x, c.ids };
    ggml_tensor * dev[7];
    for (int i = 0; i < 7; ++i) {
        dev[i] = ggml_dup_tensor(ctx, src[i]);
    }
    ggml_tensor * y = ggml_escha_moe(ctx, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int i = 0; i < 7; ++i) {
        ggml_backend_tensor_set(dev[i], src[i]->data, 0, ggml_nbytes(src[i]));
    }

    for (int i = 0; i < 8; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    const int reps = 200;
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const double us = (double)(ggml_time_us() - t0)/reps;

    printf("bench %u  K=%" PRId64 "  %4" PRId64 " -> %-4" PRId64
           "  tokens %4" PRId64 "  %9.1f us/op   %7.1f Gdecode/s\n",
           ic, c.code->ne[0]/16, IC, OC, (int64_t) c.ids->ne[1], us, decodes/us*1e-3);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Batch-invariant rule: the rows of a token are the same bits when the token runs alone, in a batch of 3 and in the
// full batch of the case. Returns true if all compared rows are equal.
static bool check_invariant(const escha_case & c, ggml_backend_t backend, uint32_t ic) {
    const int64_t n_tokens = c.ids->ne[1];
    if (n_tokens < 2) {
        return true;
    }
    const int64_t row = c.yref->ne[0]*c.yref->ne[1];   // floats per token

    const std::vector<float> full = run_case(c, backend);

    ggml_init_params ip = { 64u*1024*1024 + (size_t) 16*ggml_nbytes(c.x)/n_tokens*3, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    int64_t bad = 0, n = 0;
    const int64_t picks[] = { 0, 1, n_tokens/2, n_tokens - 1 };
    for (const int64_t t : picks) {
        for (const int64_t nb : { (int64_t) 1, (int64_t) 3 }) {
            if (t + nb > n_tokens) {
                continue;
            }
            const escha_case sc = sub_case(ctx, c, t, nb);
            const std::vector<float> got = run_case(sc, backend);
            for (int64_t i = 0; i < nb*row; ++i) {
                bad += memcmp(&got[i], &full[t*row + i], sizeof(float)) != 0;
            }
            n += nb*row;
        }
    }
    ggml_free(ctx);

    printf("case %u  invariant  %" PRId64 " tokens: alone / batch of 3 against the full batch  %s (%" PRId64 " of %" PRId64 " values differ)\n",
           ic, n_tokens, bad ? "FAIL" : "bit-identical", bad, n);
    return bad == 0;
}

static void set_prefill_env(const char * v) {
#ifdef _WIN32
    _putenv_s("GGML_ESCHA_MOE_PREFILL", v);
#else
    setenv("GGML_ESCHA_MOE_PREFILL", v, 1);
#endif
}

// Prefill kernel (escha-moe-prefill.cu, batches of more than 8 tokens) against the routed GEMV path (8-row chunks,
// GGML_ESCHA_MOE_PREFILL=0), which --invariant ties to the 1-token path: the same bits for every value. With --bench
// also the time of both paths. The tokens of the case repeat up to the batch size.
static bool check_prefill(const escha_case & c, ggml_backend_t backend, uint32_t ic, bool timing) {
    const int64_t row = c.yref->ne[0]*c.yref->ne[1];   // floats per token
    bool ok = true;
    for (const int64_t nt : { (int64_t) 9, (int64_t) 64, (int64_t) 200, (int64_t) 512, (int64_t) 1024 }) {
        ggml_init_params ip = { 64u*1024*1024 + (size_t) (ggml_nbytes(c.x) + ggml_nbytes(c.yref))/c.ids->ne[1]*(nt + 8), nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        const escha_case sc = sub_case(ctx, c, 0, nt);

        set_prefill_env("0");
        const std::vector<float> ref = run_case(sc, backend);
        if (timing) {
            printf("  gemv    ");
            bench(sc, backend, ic);
        }
        set_prefill_env("1");
        const std::vector<float> got = run_case(sc, backend);
        if (timing) {
            printf("  prefill ");
            bench(sc, backend, ic);
        }

        int64_t bad = 0;
        for (int64_t i = 0; i < nt*row; ++i) {
            bad += memcmp(&got[i], &ref[i], sizeof(float)) != 0;
        }
        // rows per expert of this batch
        std::vector<int> cnt(c.code->ne[3], 0);
        for (int64_t t = 0; t < nt; ++t) {
            for (int64_t s = 0; s < sc.ids->ne[0]; ++s) {
                cnt[((const int32_t *) ((const char *) sc.ids->data + t*sc.ids->nb[1]))[s]]++;
            }
        }
        int used = 0, mx = 0;
        for (const int v : cnt) {
            used += v > 0;
            mx = std::max(mx, v);
        }
        printf("case %u  prefill  %4" PRId64 " tokens (experts used %d, rows per used expert mean %.1f max %d): against the GEMV path  %s (%" PRId64 " of %" PRId64 " values differ)\n",
               ic, nt, used, used ? (double) (nt*sc.ids->ne[0])/used : 0.0, mx, bad ? "FAIL" : "bit-identical", bad, nt*row);
        ok &= bad == 0;
        ggml_free(ctx);
    }
    set_prefill_env("1");
    return ok;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : DEFAULT_CASES;
    const std::string mode = argc > 2 ? argv[2] : "";
    const bool do_bench     = mode == "--bench";
    const bool do_invariant = mode == "--invariant";
    const bool do_prefill   = mode == "--prefill" || mode == "--prefill-bench";
    const bool prefill_time = mode == "--prefill-bench";

    ggml_context * ctx_data = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };

    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }

    const int64_t key = gguf_find_key(gctx, "escha_test.n_cases");
    const uint32_t n_cases = key < 0 ? 0 : gguf_get_val_u32(gctx, key);
    if (n_cases == 0) {
        fprintf(stderr, "no cases in %s\n", path);
        return 1;
    }

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    printf("gpu backend: %s\n\n", gpu ? ggml_backend_name(gpu) : "none, CPU only");

    ggml_tensor * lut    = need(ctx_data, "lut");
    ggml_tensor * dep_k2 = need(ctx_data, "dep.k2");
    ggml_tensor * dep_k3 = need(ctx_data, "dep.k3");

    bool all_ok = check_codebook(lut);

    for (uint32_t ic = 0; ic < n_cases; ++ic) {
        const std::string p = "c" + std::to_string(ic) + ".";

        escha_case c;
        c.code = need(ctx_data, p + "code");
        c.rin  = need(ctx_data, p + "rin");
        c.rout = need(ctx_data, p + "rout");
        c.x    = need(ctx_data, p + "x");
        c.ids  = need(ctx_data, p + "ids");
        c.yref = need(ctx_data, p + "yref");
        c.lut  = lut;

        const int64_t K = c.code->ne[0]/16;
        c.dep = K == 2 ? dep_k2 : dep_k3;

        const int64_t IC = c.code->ne[2]*16;
        const int64_t OC = c.code->ne[1]*16;

        if (!do_bench) {
            all_ok &= score("cpu", ic, K, IC, OC, run_case(c, nullptr), c.yref);
        }
        if (gpu) {
            all_ok &= score("cuda", ic, K, IC, OC, run_case(c, gpu), c.yref);
            if (do_invariant) {
                all_ok &= check_invariant(c, gpu, ic);
            }
            if (do_prefill) {
                all_ok &= check_prefill(c, gpu, ic, prefill_time);
            }
            if (do_bench) {
                bench(c, gpu, ic);
                // the same routing repeated up to a 512-token ubatch
                if (c.ids->ne[1] >= 100 && c.ids->ne[1] < 512) {
                    ggml_init_params ip = { 64u*1024*1024 + (size_t) (ggml_nbytes(c.x) + ggml_nbytes(c.yref))/c.ids->ne[1]*520, nullptr, false };
                    ggml_context * ctx = ggml_init(ip);
                    bench(sub_case(ctx, c, 0, 512), gpu, ic);
                    ggml_free(ctx);
                }
            }
        }
    }

    if (gpu) {
        ggml_backend_free(gpu);
    }
    gguf_free(gctx);
    ggml_free(ctx_data);

    printf("\n%s\n", all_ok ? "ESCHA MOE: PASS" : "ESCHA MOE: FAIL");
    return all_ok ? 0 : 1;
}
