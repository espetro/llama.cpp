#include "decision.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// browser entry point for the Kev decision path: one model, one decision context,
// JSON in and JSON out over ccall. The page owns the GGUF file (written to the
// emscripten FS before kev_init) and all of the UI.

static llama_model *          g_model = nullptr;
static llama_context *        g_ctx   = nullptr;
static common_decision_head   g_head;

static char * to_c_string(const std::string & str) {
    char * out = (char *) malloc(str.size() + 1);
    if (out != nullptr) {
        memcpy(out, str.c_str(), str.size() + 1);
    }
    return out;
}

extern "C" {

int    kev_init(const char * model_path, int row_cap, int n_threads);
char * kev_system_one(const char * request_json);

// row_cap 0 takes the model limit, which needs more memory than a tab can grow to;
// 1024 rows fit 0.8B q8_0 in about 1 GB
int kev_init(const char * model_path, int row_cap, int n_threads) {
    if (g_ctx != nullptr) {
        return 0;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (g_model == nullptr) {
        return 1;
    }

    std::string err;
    if (!common_decision_load(model_path, g_head, err)) {
        fprintf(stderr, "kev: %s\n", err.c_str());
        return 2;
    }

    const uint32_t rows = row_cap > 0 ? (uint32_t) row_cap : g_head.max_row;

    auto cparams = llama_context_default_params();
    cparams.n_ctx                 = 4 * rows;
    cparams.n_batch               = rows;
    cparams.n_ubatch              = std::min<uint32_t>(rows, 128);
    cparams.n_seq_max             = 4;
    cparams.n_outputs_max         = 0;
    cparams.n_outputs_max_per_seq = 1;
    cparams.embeddings            = true;
    cparams.pooling_type          = LLAMA_POOLING_TYPE_NONE;
    cparams.kv_unified            = false;
    cparams.n_threads             = n_threads > 0 ? n_threads : 1;
    cparams.n_threads_batch       = cparams.n_threads;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == nullptr) {
        return 3;
    }

    fprintf(stderr, "kev: head_dim %d, temperature %.4f, %u rows x %u tokens\n",
        g_head.head_dim, g_head.temperature, llama_n_seq_max(g_ctx), llama_n_ctx_seq(g_ctx));
    return 0;
}

// answers as JSON, or {"error": ...}; the caller frees the string
char * kev_system_one(const char * request_json) {
    if (g_ctx == nullptr) {
        return to_c_string("{\"error\":\"kev_init was not called\"}");
    }

    nlohmann::ordered_json body;
    try {
        body = nlohmann::ordered_json::parse(request_json);
    } catch (const std::exception & e) {
        return to_c_string(nlohmann::ordered_json{{"error", std::string("invalid JSON: ") + e.what()}}.dump());
    }

    std::string err;
    common_decision_request req;
    if (!common_decision_parse(body, req, err)) {
        return to_c_string(nlohmann::ordered_json{{"error", err}}.dump());
    }

    common_decision_result res;
    if (!common_decision_run(g_ctx, g_head, req, res, err)) {
        return to_c_string(nlohmann::ordered_json{{"error", err}}.dump());
    }

    nlohmann::ordered_json out = {
        {"answers",    common_decision_answers(req, res)},
        {"latency_ms", std::round((res.t_state_ms + res.t_branches_ms) * 10.0) / 10.0},
    };
    return to_c_string(out.dump());
}

}
