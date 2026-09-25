#pragma once

#include "llama.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct common_decision_head {
    int32_t n_embd = 0;
    int32_t head_dim = 0;
    float temperature = 1.0f;
    std::vector<float> q_w;
    std::vector<float> q_b;
    std::vector<float> k_w;
    std::vector<float> k_b;
    llama_token tok_state = LLAMA_TOKEN_NULL;
    llama_token tok_question = LLAMA_TOKEN_NULL;
    llama_token tok_opt_start = LLAMA_TOKEN_NULL;
    llama_token tok_opt_end = LLAMA_TOKEN_NULL;
    llama_token tok_decide = LLAMA_TOKEN_NULL;
    uint32_t max_state = 8192;
    uint32_t max_row = 8192;
    std::string source;
};

bool common_decision_load(
        const std::string & gguf_path,
        common_decision_head & head,
        std::string & err);

bool common_decision_load_json(
        const std::string & head_json,
        const llama_vocab * vocab,
        common_decision_head & head,
        std::string & err);

struct common_decision_question {
    std::string id;
    std::string type;
    std::string instructions;
    std::vector<std::string> keys;
    std::vector<std::string> options;
};

struct common_decision_request {
    std::string state;
    std::vector<common_decision_question> questions;
};

bool common_decision_parse(
        const nlohmann::ordered_json & body,
        common_decision_request & req,
        std::string & err);

struct common_decision_result {
    std::vector<std::vector<float>> probs;
    int32_t n_state_tokens = 0;
    std::vector<int32_t> n_branch_tokens;
    double t_state_ms = 0;
    double t_branches_ms = 0;
};

bool common_decision_run(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_request & req,
        common_decision_result & res,
        std::string & err);

nlohmann::ordered_json common_decision_answers(
        const common_decision_request & req,
        const common_decision_result & res);
