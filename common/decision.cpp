#include "decision.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>

namespace {

using ordered_json = nlohmann::ordered_json;

static bool fail(std::string & err, const std::string & message) {
    err = message;
    return false;
}

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    const int32_t n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, parse_special);
    if (n == INT32_MIN) {
        throw std::runtime_error("tokenization size overflow");
    }
    const int32_t size = n < 0 ? -n : n;
    std::vector<llama_token> tokens(size);
    const int32_t got = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, add_special, parse_special);
    if (got < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(got);
    return tokens;
}

static llama_token delimiter(const llama_vocab * vocab, const char * text) {
    const auto tokens = tokenize(vocab, text, false, true);
    if (tokens.size() != 1) {
        throw std::runtime_error(std::string("delimiter did not resolve to one token: ") + text);
    }
    return tokens[0];
}

static std::string trim_left(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : value.substr(first);
}

static std::string render(const ordered_json & value, int indent = 0) {
    const std::string pad(static_cast<size_t>(indent) * 2, ' ');
    if (value.is_null()) {
        return "";
    }
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "True" : "False";
    }
    if (value.is_number()) {
        return value.dump();
    }
    if (value.is_array()) {
        std::string out;
        for (size_t i = 0; i < value.size(); ++i) {
            if (i != 0) {
                out += '\n';
            }
            out += pad + "- " + trim_left(render(value[i], indent + 1));
        }
        return out;
    }
    std::string out;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!out.empty()) {
            out += '\n';
        }
        if (it.value().is_object() || it.value().is_array()) {
            out += pad + it.key() + ":\n" + render(it.value(), indent + 1);
        } else {
            out += pad + it.key() + ": " + render(it.value());
        }
    }
    return out;
}

static std::string escape_user(const std::string & text) {
    static const std::regex special(R"(<\|([A-Za-z0-9_]+)\|>)");
    return std::regex_replace(text, special, "<\xC2\xA6$1\xC2\xA6>");
}

static std::string option_text(const std::string & name, const ordered_json & description) {
    if (description.is_null() || (description.is_string() && description.get<std::string>().empty())) {
        return name;
    }
    return name + ": " + render(description);
}

static bool read_tensor(
        const std::string & path,
        const gguf_context * gguf,
        int64_t tensor_id,
        int64_t n_embd,
        int64_t head_dim,
        std::vector<float> & values,
        std::string & err) {
    if (tensor_id < 0 || gguf_get_tensor_type(gguf, tensor_id) != GGML_TYPE_F32) {
        return fail(err, "Kev tensor is missing or not F32");
    }
    const int64_t * ne = gguf_get_tensor_ne(gguf, tensor_id);
    const int64_t expected = ne[1] == 1 ? head_dim : head_dim * n_embd;
    if (ne[2] != 1 ||
        (ne[1] == 1 && ne[0] != head_dim) ||
        (ne[1] != 1 && (ne[0] != n_embd || ne[1] != head_dim)) ||
        static_cast<int64_t>(gguf_get_tensor_size(gguf, tensor_id)) != expected * static_cast<int64_t>(sizeof(float))) {
        return fail(err, "invalid Kev tensor shape");
    }
    FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return fail(err, "failed to open " + path);
    }
    const uint64_t offset = static_cast<uint64_t>(gguf_get_data_offset(gguf)) + gguf_get_tensor_offset(gguf, tensor_id);
    values.resize(static_cast<size_t>(expected));
    if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0 ||
        std::fread(values.data(), sizeof(float), values.size(), file) != values.size()) {
        std::fclose(file);
        return fail(err, "failed to read Kev tensor");
    }
    std::fclose(file);
    return true;
}

static bool read_metadata_token(
        const gguf_context * gguf,
        const char * key,
        llama_token & token,
        std::string & err) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_INT32) {
        return fail(err, std::string("missing Kev metadata: ") + key);
    }
    token = static_cast<llama_token>(gguf_get_val_i32(gguf, id));
    return true;
}

static std::vector<float> project(
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        const float * hidden,
        int32_t head_dim,
        int32_t n_embd) {
    std::vector<float> out(head_dim);
    for (int32_t i = 0; i < head_dim; ++i) {
        float sum = bias[i];
        for (int32_t j = 0; j < n_embd; ++j) {
            sum += weight[static_cast<size_t>(i) * n_embd + j] * hidden[j];
        }
        out[i] = sum;
    }
    return out;
}

static std::vector<float> probabilities(
        const common_decision_head & head,
        const float * decide,
        const std::vector<const float *> & options) {
    const auto query = project(head.q_w, head.q_b, decide, head.head_dim, head.n_embd);
    std::vector<float> logits;
    logits.reserve(options.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head.head_dim)) / head.temperature;
    for (const float * option : options) {
        const auto key = project(head.k_w, head.k_b, option, head.head_dim, head.n_embd);
        float dot = 0;
        for (int32_t i = 0; i < head.head_dim; ++i) {
            dot += key[i] * query[i];
        }
        logits.push_back(dot * scale);
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (float & logit : logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    for (float & logit : logits) {
        logit /= sum;
    }
    return logits;
}

static void batch_add(
        llama_batch & batch,
        int32_t index,
        llama_token token,
        llama_pos position,
        llama_seq_id sequence) {
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence;
    batch.logits[index] = 1;
}

static void clear_sequences(llama_memory_t memory, uint32_t n_seq_max) {
    for (uint32_t i = 0; i < n_seq_max; ++i) {
        llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
    }
}

static double round_4(double value) {
    return std::round(value * 10000.0) / 10000.0;
}

}

bool common_decision_load(
        const std::string & gguf_path,
        common_decision_head & head,
        std::string & err) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (gguf == nullptr) {
        return fail(err, "failed to read model GGUF");
    }
    const auto cleanup = [&]() {
        ggml_free(meta_ctx);
        gguf_free(gguf);
    };
    const int64_t version = gguf_find_key(gguf, "kev.version");
    if (version < 0) {
        cleanup();
        return fail(err, "model has no Kev head (kev.* metadata)");
    }
    if (gguf_get_kv_type(gguf, version) != GGUF_TYPE_UINT32 || gguf_get_val_u32(gguf, version) != 1) {
        cleanup();
        return fail(err, "unsupported Kev metadata version");
    }
    const int64_t head_dim = gguf_find_key(gguf, "kev.head_dim");
    const int64_t temperature = gguf_find_key(gguf, "kev.temperature");
    if (head_dim < 0 || temperature < 0 ||
        gguf_get_kv_type(gguf, head_dim) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(gguf, temperature) != GGUF_TYPE_FLOAT32) {
        cleanup();
        return fail(err, "missing Kev head metadata");
    }
    head = common_decision_head();
    head.head_dim = static_cast<int32_t>(gguf_get_val_u32(gguf, head_dim));
    head.temperature = gguf_get_val_f32(gguf, temperature);
    const int64_t q_weight = gguf_find_tensor(gguf, "dec.head_q.weight");
    const int64_t k_weight = gguf_find_tensor(gguf, "dec.head_k.weight");
    const int64_t q_bias = gguf_find_tensor(gguf, "dec.head_q.bias");
    const int64_t k_bias = gguf_find_tensor(gguf, "dec.head_k.bias");
    if (q_weight < 0 || k_weight < 0 || q_bias < 0 || k_bias < 0) {
        cleanup();
        return fail(err, "missing Kev head tensor");
    }
    const int64_t * q_ne = gguf_get_tensor_ne(gguf, q_weight);
    const int64_t * k_ne = gguf_get_tensor_ne(gguf, k_weight);
    if (q_ne[1] != head.head_dim || k_ne[1] != head.head_dim || q_ne[0] != k_ne[0]) {
        cleanup();
        return fail(err, "invalid Kev head dimensions");
    }
    head.n_embd = static_cast<int32_t>(q_ne[0]);
    if (head.n_embd <= 0 || head.head_dim <= 0 ||
        !read_tensor(gguf_path, gguf, q_weight, head.n_embd, head.head_dim, head.q_w, err) ||
        !read_tensor(gguf_path, gguf, k_weight, head.n_embd, head.head_dim, head.k_w, err) ||
        !read_tensor(gguf_path, gguf, q_bias, head.n_embd, head.head_dim, head.q_b, err) ||
        !read_tensor(gguf_path, gguf, k_bias, head.n_embd, head.head_dim, head.k_b, err)) {
        cleanup();
        return false;
    }
    if (!read_metadata_token(gguf, "kev.tokens.state", head.tok_state, err) ||
        !read_metadata_token(gguf, "kev.tokens.question", head.tok_question, err) ||
        !read_metadata_token(gguf, "kev.tokens.opt_start", head.tok_opt_start, err) ||
        !read_metadata_token(gguf, "kev.tokens.opt_end", head.tok_opt_end, err) ||
        !read_metadata_token(gguf, "kev.tokens.decide", head.tok_decide, err)) {
        cleanup();
        return false;
    }
    const int64_t max_state = gguf_find_key(gguf, "kev.limits.max_state");
    const int64_t max_row = gguf_find_key(gguf, "kev.limits.max_row");
    if (max_state < 0 || max_row < 0 ||
        gguf_get_kv_type(gguf, max_state) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(gguf, max_row) != GGUF_TYPE_UINT32) {
        cleanup();
        return fail(err, "missing Kev token limits");
    }
    head.max_state = gguf_get_val_u32(gguf, max_state);
    head.max_row = gguf_get_val_u32(gguf, max_row);
    const int64_t source = gguf_find_key(gguf, "kev.source");
    if (source < 0 || gguf_get_kv_type(gguf, source) != GGUF_TYPE_STRING) {
        cleanup();
        return fail(err, "missing Kev source metadata");
    }
    head.source = gguf_get_val_str(gguf, source);
    cleanup();
    return true;
}

bool common_decision_load_json(
        const std::string & head_json,
        const llama_vocab * vocab,
        common_decision_head & head,
        std::string & err) {
    try {
        const auto json = ordered_json::parse(read_file(head_json));
        head = common_decision_head();
        if (vocab == nullptr) {
            return fail(err, "model vocabulary is required for head.json");
        }
        head.n_embd = json.at("hidden_size").get<int32_t>();
        head.head_dim = json.at("head_dim").get<int32_t>();
        head.temperature = json.at("temperature").get<float>();
        head.q_b = json.at("q_bias").get<std::vector<float>>();
        head.k_b = json.at("k_bias").get<std::vector<float>>();
        if (!json.at("q_weight").is_array() || !json.at("k_weight").is_array() ||
            json.at("q_weight").size() != static_cast<size_t>(head.head_dim) ||
            json.at("k_weight").size() != static_cast<size_t>(head.head_dim)) {
            return fail(err, "head.json weight dimensions do not match");
        }
        for (const auto & row : json.at("q_weight")) {
            const auto values = row.get<std::vector<float>>();
            if (values.size() != static_cast<size_t>(head.n_embd)) {
                return fail(err, "head.json q_weight row has invalid size");
            }
            head.q_w.insert(head.q_w.end(), values.begin(), values.end());
        }
        for (const auto & row : json.at("k_weight")) {
            const auto values = row.get<std::vector<float>>();
            if (values.size() != static_cast<size_t>(head.n_embd)) {
                return fail(err, "head.json k_weight row has invalid size");
            }
            head.k_w.insert(head.k_w.end(), values.begin(), values.end());
        }
        if (head.n_embd <= 0 || head.head_dim <= 0 ||
            static_cast<int64_t>(head.q_w.size()) != static_cast<int64_t>(head.n_embd) * head.head_dim ||
            static_cast<int64_t>(head.k_w.size()) != static_cast<int64_t>(head.n_embd) * head.head_dim ||
            head.q_b.size() != static_cast<size_t>(head.head_dim) ||
            head.k_b.size() != static_cast<size_t>(head.head_dim)) {
            return fail(err, "head.json dimensions do not match");
        }
        head.tok_state = delimiter(vocab, "<|fim_prefix|>");
        head.tok_question = delimiter(vocab, "<|fim_middle|>");
        head.tok_opt_start = delimiter(vocab, "<|box_start|>");
        head.tok_opt_end = delimiter(vocab, "<|box_end|>");
        head.tok_decide = delimiter(vocab, "<|fim_suffix|>");
        return true;
    } catch (const std::exception & e) {
        return fail(err, e.what());
    }
}

bool common_decision_parse(
        const ordered_json & body,
        common_decision_request & req,
        std::string & err) {
    try {
        if (!body.is_object() || !body.contains("state")) {
            return fail(err, "state is required");
        }
        if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
            return fail(err, "questions must be a non-empty object");
        }
        req = common_decision_request();
        req.state = render(body.at("state"));
        for (auto it = body.at("questions").begin(); it != body.at("questions").end(); ++it) {
            const auto & value = it.value();
            if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
                return fail(err, "question " + it.key() + " has an invalid type");
            }
            common_decision_question question;
            question.id = it.key();
            question.type = value.at("type").get<std::string>();
            question.instructions = render(value.contains("instructions") ? value.at("instructions") : ordered_json(nullptr));
            if (question.type == "noul") {
                if (value.contains("criteria") && !value.at("criteria").is_null() && !value.at("criteria").is_object()) {
                    return fail(err, "noul criteria must be an object");
                }
                const auto criteria = value.value("criteria", ordered_json::object());
                question.keys = {"false", "true"};
                question.options = {
                    option_text("no", criteria.contains("false") ? criteria.at("false") : ordered_json(nullptr)),
                    option_text("yes", criteria.contains("true") ? criteria.at("true") : ordered_json(nullptr)),
                };
            } else if (question.type == "choice") {
                if (!value.contains("criteria") || !value.at("criteria").is_object() || value.at("criteria").empty()) {
                    return fail(err, "choice criteria must be a non-empty object");
                }
                if (value.at("criteria").size() > 255) {
                    return fail(err, "choice criteria has too many options");
                }
                for (auto criterion = value.at("criteria").begin(); criterion != value.at("criteria").end(); ++criterion) {
                    question.keys.push_back(criterion.key());
                    question.options.push_back(option_text(criterion.key(), criterion.value()));
                }
            } else if (question.type == "score") {
                if (!value.contains("criteria") || !value.at("criteria").is_array() || value.at("criteria").empty()) {
                    return fail(err, "score criteria must be a non-empty array");
                }
                if (value.at("criteria").size() > 255) {
                    return fail(err, "score criteria has too many levels");
                }
                for (size_t i = 0; i < value.at("criteria").size(); ++i) {
                    question.keys.push_back(std::to_string(i));
                    question.options.push_back(render(value.at("criteria").at(i)));
                }
            } else {
                return fail(err, "question " + it.key() + " has an invalid type");
            }
            req.questions.push_back(std::move(question));
        }
        return true;
    } catch (const std::exception & e) {
        return fail(err, e.what());
    }
}

bool common_decision_run(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_request & req,
        common_decision_result & res,
        std::string & err) {
    try {
        if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            return fail(err, "Kev decision context must use pooling NONE");
        }
        if (head.n_embd != static_cast<int32_t>(llama_model_n_embd_out(llama_get_model(ctx)))) {
            return fail(err, "model hidden size does not match Kev head");
        }
        if (head.max_state < 1 || head.max_row < 1) {
            return fail(err, "invalid Kev token limits");
        }
        const uint32_t row_cap = std::min(head.max_row, llama_n_ctx_seq(ctx));
        if (row_cap < 1) {
            return fail(err, "invalid Kev decision context size");
        }
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        auto state_tokens = tokenize(vocab, escape_user(req.state), false, false);
        state_tokens.resize(std::min<size_t>(state_tokens.size(), std::min(head.max_state, row_cap) - 1));

        std::vector<std::vector<llama_token>> branches;
        std::vector<std::vector<int32_t>> option_ends;
        std::vector<int32_t> decides;
        size_t max_branch_tokens = 0;
        for (const auto & question : req.questions) {
            std::vector<llama_token> tokens = { head.tok_question };
            const auto instructions = tokenize(vocab, escape_user(question.instructions), false, false);
            tokens.insert(tokens.end(), instructions.begin(), instructions.end());
            std::vector<int32_t> ends;
            for (const auto & option : question.options) {
                tokens.push_back(head.tok_opt_start);
                const auto option_tokens = tokenize(vocab, escape_user(option), false, false);
                tokens.insert(tokens.end(), option_tokens.begin(), option_tokens.end());
                tokens.push_back(head.tok_opt_end);
                ends.push_back(static_cast<int32_t>(tokens.size() - 1));
            }
            tokens.push_back(head.tok_decide);
            if (tokens.size() > row_cap - 1) {
                return fail(err, "question " + question.id + " exceeds the context window (" +
                    std::to_string(row_cap) + " tokens)");
            }
            max_branch_tokens = std::max(max_branch_tokens, tokens.size());
            branches.push_back(std::move(tokens));
            option_ends.push_back(std::move(ends));
            decides.push_back(static_cast<int32_t>(branches.back().size() - 1));
        }

        const size_t max_state_tokens = row_cap - max_branch_tokens;
        state_tokens.resize(std::min(state_tokens.size() + 1, max_state_tokens) - 1);
        state_tokens.insert(state_tokens.begin(), head.tok_state);

        res = common_decision_result();
        res.n_state_tokens = static_cast<int32_t>(state_tokens.size());
        for (const auto & branch : branches) {
            res.n_branch_tokens.push_back(static_cast<int32_t>(branch.size()));
        }
        res.probs.resize(branches.size());
        llama_memory_t memory = llama_get_memory(ctx);
        const uint32_t n_seq_max = llama_n_seq_max(ctx);
        clear_sequences(memory, n_seq_max);
        llama_batch state_batch = llama_batch_init(static_cast<int32_t>(state_tokens.size()), 0, 1);
        for (size_t i = 0; i < state_tokens.size(); ++i) {
            batch_add(state_batch, static_cast<int32_t>(i), state_tokens[i], static_cast<llama_pos>(i), 0);
        }
        state_batch.n_tokens = static_cast<int32_t>(state_tokens.size());
        const auto state_start = std::chrono::steady_clock::now();
        const int state_rc = llama_decode(ctx, state_batch);
        const auto state_end = std::chrono::steady_clock::now();
        res.t_state_ms = std::chrono::duration<double, std::milli>(state_end - state_start).count();
        llama_batch_free(state_batch);
        if (state_rc != 0) {
            clear_sequences(memory, n_seq_max);
            return fail(err, "state decode failed");
        }

        if (n_seq_max < 2) {
            clear_sequences(memory, n_seq_max);
            return fail(err, "Kev decision context needs at least two sequences");
        }
        const uint32_t n_batch = llama_n_batch(ctx);
        const size_t chunk_size = n_seq_max - 1;
        const auto branch_start = std::chrono::steady_clock::now();
        for (size_t base = 0; base < branches.size();) {
            size_t count = 0;
            size_t total_tokens = 0;
            while (base + count < branches.size() && count < chunk_size) {
                const size_t branch_tokens = branches[base + count].size();
                if (branch_tokens > n_batch) {
                    clear_sequences(memory, n_seq_max);
                    return fail(err, "question " + req.questions[base + count].id +
                        " exceeds the batch size (" + std::to_string(n_batch) + " tokens)");
                }
                if (count > 0 && total_tokens + branch_tokens > n_batch) {
                    break;
                }
                total_tokens += branch_tokens;
                ++count;
            }
            for (size_t i = 0; i < count; ++i) {
                llama_memory_seq_cp(memory, 0, static_cast<llama_seq_id>(i + 1), -1, -1);
            }
            llama_batch batch = llama_batch_init(static_cast<int32_t>(total_tokens), 0, 1);
            size_t index = 0;
            std::vector<size_t> offsets(count);
            for (size_t i = 0; i < count; ++i) {
                offsets[i] = index;
                for (size_t j = 0; j < branches[base + i].size(); ++j) {
                    batch_add(batch, static_cast<int32_t>(index++), branches[base + i][j],
                        static_cast<llama_pos>(state_tokens.size() + j), static_cast<llama_seq_id>(i + 1));
                }
            }
            batch.n_tokens = static_cast<int32_t>(index);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                llama_batch_free(batch);
                clear_sequences(memory, n_seq_max);
                return fail(err, "branch decode failed");
            }
            for (size_t i = 0; i < count; ++i) {
                std::vector<const float *> options;
                for (const int32_t position : option_ends[base + i]) {
                    float * embedding = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + position));
                    if (embedding == nullptr) {
                        llama_batch_free(batch);
                        clear_sequences(memory, n_seq_max);
                        return fail(err, "decision context was not created with embeddings enabled");
                    }
                    options.push_back(embedding);
                }
                float * decide = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + decides[base + i]));
                if (decide == nullptr) {
                    llama_batch_free(batch);
                    clear_sequences(memory, n_seq_max);
                    return fail(err, "decision context was not created with embeddings enabled");
                }
                res.probs[base + i] = probabilities(head, decide, options);
            }
            llama_batch_free(batch);
            base += count;
        }
        const auto branch_end = std::chrono::steady_clock::now();
        res.t_branches_ms = std::chrono::duration<double, std::milli>(branch_end - branch_start).count();
        clear_sequences(memory, n_seq_max);
        return true;
    } catch (const std::exception & e) {
        clear_sequences(llama_get_memory(ctx), llama_n_seq_max(ctx));
        return fail(err, e.what());
    }
}

nlohmann::ordered_json common_decision_answers(
        const common_decision_request & req,
        const common_decision_result & res) {
    ordered_json answers = ordered_json::object();
    for (size_t q = 0; q < req.questions.size(); ++q) {
        const auto & question = req.questions[q];
        const auto & probs = res.probs[q];
        if (question.type == "noul") {
            answers[question.id] = {
                {"type", "noul"},
                {"noul", round_4(probs[1])},
            };
        } else if (question.type == "choice") {
            const size_t best = static_cast<size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            ordered_json distribution = ordered_json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                distribution[question.keys[i]] = round_4(probs[i]);
            }
            const double confidence = probs.size() == 1 ? 1.0 :
                (probs[best] - 1.0 / probs.size()) / (1.0 - 1.0 / probs.size());
            answers[question.id] = {
                {"type", "choice"},
                {"choice", question.keys[best]},
                {"confidence", round_4(confidence)},
                {"probabilities", distribution},
            };
        } else {
            const size_t best = static_cast<size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            double score = 0;
            double distance = 0;
            ordered_json distribution = ordered_json::object();
            ordered_json legend = ordered_json::object();
            for (size_t i = 0; i < probs.size(); ++i) {
                score += i * probs[i];
                distance += probs[i] * std::abs(static_cast<double>(i) - best);
                distribution[std::to_string(i)] = round_4(probs[i]);
                legend[question.keys[i]] = question.options[i];
            }
            const double confidence = probs.size() == 1 ? 1.0 : 1.0 - distance / (probs.size() - 1);
            answers[question.id] = {
                {"type", "score"},
                {"score", round_4(score)},
                {"legend", legend},
                {"probabilities", distribution},
                {"confidence", round_4(confidence)},
            };
        }
    }
    return answers;
}
