#include "server-decision.h"
#include "server-common.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <memory>

server_decision::~server_decision() {
    llama_free(ctx);
}

bool server_decision::init(const common_params & params, llama_model * model, llama_context * main_ctx, std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!params.kev_head_path.empty()) {
        if (!common_decision_load_json(params.kev_head_path, vocab, head, err)) {
            return false;
        }
        if (head.source.empty()) {
            head.source = "sidecar";
        }
    } else if (!common_decision_load(params.model.path, head, err)) {
        return false;
    }

    const uint32_t row_cap = std::min(head.max_row, llama_n_ctx_seq(main_ctx));
    if (row_cap == 0) {
        err = "invalid Kev decision context size";
        return false;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 4 * row_cap;
    cparams.n_batch = row_cap;
    const uint32_t ubatch = params.n_ubatch > 0 ? params.n_ubatch : 512;
    cparams.n_ubatch = std::min<uint32_t>(row_cap, std::min<uint32_t>(512, ubatch));
    cparams.n_seq_max = 4;
    cparams.n_outputs_max = 0;
    cparams.n_outputs_max_per_seq = 1;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.kv_unified = false;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        err = "failed to create Kev decision context";
        return false;
    }

    SRV_INF("Kev decision context: %u rows x %u tokens\n", llama_n_seq_max(ctx), llama_n_ctx_seq(ctx));
    return true;
}

bool server_decision::run(const nlohmann::ordered_json & body, nlohmann::ordered_json & response, std::string & err) {
    common_decision_request req;
    if (!common_decision_parse(body, req, err)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    common_decision_result result;
    if (!common_decision_run(ctx, head, req, result, err)) {
        return false;
    }

    response = {
        {"answers", common_decision_answers(req, result)},
        {"latency_ms", std::round((result.t_state_ms + result.t_branches_ms) * 10.0) / 10.0},
    };
    return true;
}
