#pragma once

#include "common.h"
#include "decision.h"

#include <memory>
#include <mutex>
#include <string>

struct server_decision {
    server_decision() = default;
    ~server_decision();

    bool init(const common_params & params, llama_model * model, llama_context * main_ctx, std::string & err);
    bool run(const nlohmann::ordered_json & body, nlohmann::ordered_json & response, std::string & err);

    const common_decision_head & get_head() const {
        return head;
    }

private:
    common_decision_head head;
    llama_context * ctx = nullptr;
    std::mutex mutex;
};
