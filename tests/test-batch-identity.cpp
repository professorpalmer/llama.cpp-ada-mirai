// Batch identity check with a real model: a token decoded alone and the same token decoded in a
// batch of 2 to 8 rows must give the same logits bit for bit (GGML_CUDA_BATCH_INVARIANT=1). That is
// what makes speculative decoding exact: the verify batch of the drafted tokens must reproduce the
// plain decode.
//
// The test decodes a fixed prefix (--prefix-step tokens for each call), then the next R tokens
//   (a) one at a time, and
//   (b) from the same state (memory cleared, prefix decoded again) as one batch of R rows,
// and compares the logits of every row. With --watch it also records the output of every graph node
// in both runs through the eval callback and reports the first nodes whose rows differ. Watching all
// nodes makes the scheduler compute one node per backend call, so the backend cannot fuse nodes; when
// the difference is visible only without --watch, the cause is in a fused path or in the graph plan.
// A prefix that ends 2 tokens before a multiple of 256 makes the batch cross a KV padding step: the
// batch then has a longer KV view than the first rows decoded alone.
//
//   usage: test-batch-identity <model.gguf> [--prefix 64[,254,...]] [--rows 3[,5,...]] [--kv f16|q8_0|q4_0]
//                              [--ctx 4096] [--packed-mask] [--prefix-step 1] [--watch] [--watch-match <substr>] [--show N]
//
// Needs a model file, so it is built only (not registered as a ctest).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * TEXT =
    "The lighthouse keeper counted the ships every evening and wrote the numbers in a green book. "
    "On the first day of March he saw seven fishing boats, two tankers and one white ferry that was late. "
    "A function that sorts a list of integers in Python can be written as: def sort(xs): return sorted(xs). "
    "The capital of France is Paris, the capital of Japan is Tokyo, and the capital of Kenya is Nairobi. "
    "When water freezes it expands by about nine percent, which is why ice floats and pipes can burst in winter. "
    "She opened the door, looked at the long table with the maps, and asked who had moved the red pins. "
    "In 1969 the first people walked on the Moon; the computer on board had less memory than a pocket calculator. "
    "To make bread you need flour, water, salt and yeast: mix, wait, fold, wait again, and bake in a hot oven. "
    "The quick brown fox jumps over the lazy dog while 3.14159 is not exactly pi and 2 + 2 = 4 in every base above four.";

struct node_rec {
    std::string name;
    ggml_op     op;
    ggml_type   type;
    int64_t     ne[GGML_MAX_DIMS];
    std::string srcs;
    std::vector<uint8_t> data;
};

struct watch_state {
    std::vector<node_rec> * sink = nullptr; // null: do not watch (the graph runs in one piece)
    const char * match = nullptr;           // watch only nodes whose name has this text
};

// ops that are not recorded: aliases of other nodes, and MIRAI_QUANTIZE (packed rows, then the statistics of all rows)
static bool is_alias_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE ||
           op == GGML_OP_MIRAI_QUANTIZE;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    watch_state * ws = (watch_state *) ud;
    if (ws->sink == nullptr) {
        return ask ? false : true;
    }
    const bool want = !is_alias_op(t->op) && ggml_is_contiguous(t) && (ws->match == nullptr || strstr(t->name, ws->match) != nullptr);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    node_rec r;
    r.name = t->name;
    r.op   = t->op;
    r.type = t->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        r.ne[i] = t->ne[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] == nullptr) {
            continue;
        }
        const ggml_tensor * s = t->src[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "\n        src%d %s op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu]%s",
            i, s->name, ggml_op_name(s->op), ggml_type_name(s->type), s->ne[0], s->ne[1], s->ne[2], s->ne[3],
            s->nb[0], s->nb[1], s->nb[2], s->nb[3], s->view_src ? " (view)" : "");
        r.srcs += buf;
    }
    r.data.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data.data(), 0, r.data.size());
    ws->sink->push_back(std::move(r));
    return true;
}

static int64_t nelem(const node_rec & r) {
    return r.ne[0]*r.ne[1]*r.ne[2]*r.ne[3];
}

// rows `first` .. `first + count - 1` of dimension d, in memory order
static std::vector<uint8_t> slice_dim(const node_rec & r, int d, int64_t first, int64_t count) {
    const size_t es = r.data.size()/(size_t) nelem(r);
    int64_t inner = 1;
    for (int i = 0; i < d; ++i) {
        inner *= r.ne[i];
    }
    int64_t outer = 1;
    for (int i = d + 1; i < GGML_MAX_DIMS; ++i) {
        outer *= r.ne[i];
    }
    std::vector<uint8_t> out;
    out.reserve((size_t) (outer*count*inner)*es);
    for (int64_t o = 0; o < outer; ++o) {
        const uint8_t * p = r.data.data() + (size_t) ((o*r.ne[d] + first)*inner)*es;
        out.insert(out.end(), p, p + (size_t) (count*inner)*es);
    }
    return out;
}

struct diff_stat {
    int64_t n_diff  = 0;
    int64_t n_total = 0;
    double  max_abs = 0.0;
    int64_t first   = -1;
};

static diff_stat diff_bytes(const uint8_t * a, const uint8_t * b, size_t nbytes, ggml_type type) {
    diff_stat d;
    if (type == GGML_TYPE_F32) {
        const size_t n = nbytes/4;
        d.n_total = (int64_t) n;
        for (size_t i = 0; i < n; ++i) {
            uint32_t ua, ub;
            memcpy(&ua, a + 4*i, 4);
            memcpy(&ub, b + 4*i, 4);
            if (ua != ub) {
                float fa, fb;
                memcpy(&fa, &ua, 4);
                memcpy(&fb, &ub, 4);
                d.n_diff++;
                if (d.first < 0) {
                    d.first = (int64_t) i;
                }
                const double ad = fabs((double) fa - (double) fb);
                if (ad > d.max_abs || std::isnan(ad)) {
                    d.max_abs = ad;
                }
            }
        }
    } else {
        d.n_total = (int64_t) nbytes;
        for (size_t i = 0; i < nbytes; ++i) {
            if (a[i] != b[i]) {
                d.n_diff++;
                if (d.first < 0) {
                    d.first = (int64_t) i;
                }
            }
        }
    }
    return d;
}

static bool decode_rows(llama_context * ctx, const std::vector<llama_token> & toks, int first, int count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i]     = toks[first + i];
        batch.pos[i]       = first + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return false;
    }
    return true;
}

static std::vector<float> get_logits(llama_context * ctx, int i, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, i);
    return std::vector<float>(p, p + n_vocab);
}

static int argmax(const std::vector<float> & v) {
    int b = 0;
    for (int i = 1; i < (int) v.size(); ++i) {
        if (v[i] > v[b]) {
            b = i;
        }
    }
    return b;
}

static void print_node(const char * tag, int idx, const node_rec & r) {
    printf("  %s node %d: name='%s' op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]%s\n",
        tag, idx, r.name.c_str(), ggml_op_name(r.op), ggml_type_name(r.type), r.ne[0], r.ne[1], r.ne[2], r.ne[3], r.srcs.c_str());
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--prefix 64[,254]] [--rows 3[,5]] [--kv f16|q8_0|q4_0] [--ctx 4096] [--packed-mask] [--prefix-step 1] [--watch] [--watch-match s] [--show N]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    int  n_prefix = 64;
    std::vector<int> prefix_list = { 64 };
    int  n_ctx    = 4096;
    int  n_show   = 6;
    bool watch    = false;
    bool packed   = false;
    int  prefix_step = 1; // tokens per decode call of the prefix (the same in every run)
    const char * watch_match = nullptr;
    std::vector<int> rows_list = { 3 };
    ggml_type kv = GGML_TYPE_F16;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--prefix") {
            prefix_list.clear();
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) {
                    q = s.size();
                }
                prefix_list.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else if (a == "--ctx") {
            n_ctx = atoi(next());
        } else if (a == "--show") {
            n_show = atoi(next());
        } else if (a == "--watch") {
            watch = true;
        } else if (a == "--watch-match") {
            watch = true;
            watch_match = next();
        } else if (a == "--packed-mask") {
            packed = true;
        } else if (a == "--prefix-step") {
            prefix_step = atoi(next());
        } else if (a == "--rows") {
            rows_list.clear();
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) {
                    q = s.size();
                }
                rows_list.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else if (a == "--kv") {
            const std::string v = next();
            if (v == "f16") {
                kv = GGML_TYPE_F16;
            } else if (v == "q8_0") {
                kv = GGML_TYPE_Q8_0;
            } else if (v == "q4_0") {
                kv = GGML_TYPE_Q4_0;
            } else {
                fprintf(stderr, "unknown --kv %s\n", v.c_str());
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    printf("GGML_CUDA_BATCH_INVARIANT=%s\n", getenv("GGML_CUDA_BATCH_INVARIANT") ? getenv("GGML_CUDA_BATCH_INVARIANT") : "(not set)");

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(1024);
    const int n_tok = llama_tokenize(vocab, TEXT, (int32_t) strlen(TEXT), toks.data(), (int32_t) toks.size(), true, false);
    if (n_tok <= 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    toks.resize(n_tok);

    int max_rows = 0;
    for (int r : rows_list) {
        max_rows = r > max_rows ? r : max_rows;
    }
    // a long prefix repeats the text
    int max_prefix = 0;
    for (int p : prefix_list) {
        max_prefix = p > max_prefix ? p : max_prefix;
    }
    for (int i = 0; (int) toks.size() < max_prefix + max_rows; ++i) {
        toks.push_back(toks[1 + i % (n_tok - 1)]);
    }

    watch_state ws;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_ctx;
    cp.n_batch           = 512;
    cp.n_ubatch          = 512;
    cp.n_seq_max         = 1;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k            = kv;
    cp.type_v            = kv;
    cp.kq_mask_packed    = packed;
    cp.cb_eval           = eval_cb;
    cp.cb_eval_user_data = &ws;
    cp.no_perf           = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    auto prefix = [&]() -> bool {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int i = 0; i < n_prefix; i += prefix_step) {
            if (!decode_rows(ctx, toks, i, prefix_step < n_prefix - i ? prefix_step : n_prefix - i)) {
                return false;
            }
        }
        return true;
    };

    int n_fail = 0;

    for (size_t ic = 0; ic < prefix_list.size()*rows_list.size(); ++ic) {
        n_prefix = prefix_list[ic/rows_list.size()];
        const int rows = rows_list[ic % rows_list.size()];
        printf("\n=== prefix %d tokens, %d rows, kv %s%s\n", n_prefix, rows, ggml_type_name(kv), packed ? ", packed mask" : "");

        // alone, alone again (determinism), batch: no watching, the graph runs in one piece
        std::vector<std::vector<float>> la[2], lb;
        for (int rep = 0; rep < 2; ++rep) {
            if (!prefix()) {
                return 1;
            }
            la[rep].resize(rows);
            for (int s = 0; s < rows; ++s) {
                if (!decode_rows(ctx, toks, n_prefix + s, 1)) {
                    return 1;
                }
                la[rep][s] = get_logits(ctx, 0, n_vocab);
            }
        }
        if (!prefix() || !decode_rows(ctx, toks, n_prefix, rows)) {
            return 1;
        }
        lb.resize(rows);
        for (int s = 0; s < rows; ++s) {
            lb[s] = get_logits(ctx, s, n_vocab);
        }

        for (int s = 0; s < rows; ++s) {
            const diff_stat d0 = diff_bytes((const uint8_t *) la[0][s].data(), (const uint8_t *) la[1][s].data(), (size_t) n_vocab*4, GGML_TYPE_F32);
            const diff_stat d1 = diff_bytes((const uint8_t *) la[0][s].data(), (const uint8_t *) lb[s].data(),    (size_t) n_vocab*4, GGML_TYPE_F32);
            printf("row %d: alone vs alone again: %" PRId64 " of %d logits differ; alone vs batch: %" PRId64 " differ (max abs %.3g), argmax %d / %d %s\n",
                s, d0.n_diff, n_vocab, d1.n_diff, d1.max_abs, argmax(la[0][s]), argmax(lb[s]), d1.n_diff == 0 ? "SAME" : "DIFFERENT");
            if (d0.n_diff != 0 || d1.n_diff != 0) {
                n_fail++;
            }
        }

        if (!watch) {
            continue;
        }

        // the same two runs with the node outputs recorded
        std::vector<std::vector<node_rec>> na(rows);
        std::vector<node_rec> nb;
        if (!prefix()) {
            return 1;
        }
        for (int s = 0; s < rows; ++s) {
            ws.sink  = &na[s];
            ws.match = watch_match;
            const bool ok = decode_rows(ctx, toks, n_prefix + s, 1);
            ws.sink = nullptr;
            if (!ok) {
                return 1;
            }
            const std::vector<float> l = get_logits(ctx, 0, n_vocab);
            const diff_stat d = diff_bytes((const uint8_t *) l.data(), (const uint8_t *) la[0][s].data(), (size_t) n_vocab*4, GGML_TYPE_F32);
            printf("watched alone row %d: %" PRId64 " logits differ from the run without watching\n", s, d.n_diff);
        }
        if (!prefix()) {
            return 1;
        }
        ws.sink  = &nb;
        ws.match = watch_match;
        const bool ok = decode_rows(ctx, toks, n_prefix, rows);
        ws.sink = nullptr;
        if (!ok) {
            return 1;
        }
        for (int s = 0; s < rows; ++s) {
            const std::vector<float> l = get_logits(ctx, s, n_vocab);
            const diff_stat d  = diff_bytes((const uint8_t *) l.data(), (const uint8_t *) lb[s].data(),    (size_t) n_vocab*4, GGML_TYPE_F32);
            const diff_stat d2 = diff_bytes((const uint8_t *) l.data(), (const uint8_t *) la[0][s].data(), (size_t) n_vocab*4, GGML_TYPE_F32);
            printf("watched batch row %d: %" PRId64 " logits differ from the batch without watching, %" PRId64 " from alone without watching\n", s, d.n_diff, d2.n_diff);
        }

        printf("nodes recorded: alone %zu, batch %zu\n", na[0].size(), nb.size());
        if (na[0].size() != nb.size()) {
            printf("  the graphs have a different number of recorded nodes: compare by position not possible\n");
            const size_t n = na[0].size() < nb.size() ? na[0].size() : nb.size();
            for (size_t i = 0; i < n; ++i) {
                if (na[0][i].op != nb[i].op || na[0][i].name != nb[i].name) {
                    print_node("first structural difference, alone", (int) i, na[0][i]);
                    print_node("first structural difference, batch", (int) i, nb[i]);
                    break;
                }
            }
            n_fail++;
            continue;
        }

        int n_shown = 0, n_skipped = 0, n_differ = 0;
        for (size_t i = 0; i < nb.size(); ++i) {
            const node_rec & B = nb[i];
            const node_rec & A0 = na[0][i];
            if (A0.op != B.op || A0.type != B.type) {
                print_node("op/type mismatch, alone", (int) i, A0);
                print_node("op/type mismatch, batch", (int) i, B);
                n_skipped++;
                continue;
            }
            // find the token dimension
            int d_tok = -1, n_other = 0;
            int kind = 0; // 1: rows x, 2: same shape, 3: rows then a tail of state rows
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (A0.ne[d] == B.ne[d]) {
                    continue;
                }
                n_other++;
                d_tok = d;
            }
            if (n_other == 0) {
                kind = 2;
            } else if (n_other == 1 && B.ne[d_tok] == rows*A0.ne[d_tok]) {
                kind = 1;
            } else if (n_other == 1 && B.ne[d_tok] - A0.ne[d_tok] == rows - 1) {
                kind = 3;
            }
            if (kind == 0) {
                if (n_skipped < 40) {
                    printf("  skipped (shape) node %zu '%s' op=%s alone ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] batch ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                        i, B.name.c_str(), ggml_op_name(B.op), A0.ne[0], A0.ne[1], A0.ne[2], A0.ne[3], B.ne[0], B.ne[1], B.ne[2], B.ne[3]);
                }
                n_skipped++;
                continue;
            }

            bool differ = false;
            std::string msg;
            auto cmp = [&](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const char * what, int s) {
                if (a.size() != b.size()) {
                    msg += "    size mismatch\n";
                    differ = true;
                    return;
                }
                const diff_stat d = diff_bytes(a.data(), b.data(), a.size(), B.type);
                if (d.n_diff != 0) {
                    char buf[256];
                    snprintf(buf, sizeof(buf), "    %s %d: %" PRId64 " of %" PRId64 " values differ, max abs %.3g, first at %" PRId64 "\n", what, s, d.n_diff, d.n_total, d.max_abs, d.first);
                    msg += buf;
                    differ = true;
                }
            };
            if (kind == 2) {
                // no token dimension: a state read is the state before the first row, a state write is the state after the last row
                const bool same_first = A0.data.size() == B.data.size() && memcmp(A0.data.data(), B.data.data(), B.data.size()) == 0;
                if (!same_first) {
                    cmp(na[rows - 1][i].data, B.data, "whole tensor (not equal to alone step 0) vs alone step", rows - 1);
                }
            } else if (kind == 1) {
                const int64_t w = A0.ne[d_tok];
                for (int s = 0; s < rows; ++s) {
                    cmp(na[s][i].data, slice_dim(B, d_tok, s*w, w), "row", s);
                }
            } else {
                // rows first, then the state rows (gated delta net: outputs, then the new state)
                for (int s = 0; s < rows; ++s) {
                    cmp(slice_dim(na[s][i], d_tok, 0, 1), slice_dim(B, d_tok, s, 1), "row (rows+state layout)", s);
                }
                cmp(slice_dim(na[rows - 1][i], d_tok, 1, A0.ne[d_tok] - 1), slice_dim(B, d_tok, rows, A0.ne[d_tok] - 1), "state tail vs alone step", rows - 1);
            }
            if (differ) {
                n_differ++;
                if (n_shown < n_show) {
                    print_node(n_shown == 0 ? "FIRST DIFFERENT" : "different", (int) i, B);
                    printf("%s", msg.c_str());
                    n_shown++;
                }
            }
        }
        printf("nodes: %zu compared, %d different, %d skipped\n", nb.size() - n_skipped, n_differ, n_skipped);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("\n%s\n", n_fail == 0 ? "PASS: batch = alone for every row" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
