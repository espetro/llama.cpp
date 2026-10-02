#include "arg.h"
#include "common.h"
#include "decision.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    if (path == "-") {
        return std::string((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static void print_usage(int, char ** argv) {
    std::fprintf(stderr,
        "\nllama-decide usage:\n"
        "  %s -m model.gguf --json request.json [common model options]\n"
        "  %s -m model.gguf --check reference.json\n"
        "  %s -m model.gguf --kev-head head.json --check reference.json\n"
        "\n"
        "  --json FILE        TypeSafe request JSON, or - for stdin\n"
        "  --kev-head FILE    load head.json instead of embedded Kev metadata\n"
        "  --check FILE       compare all cases in a reference.json fixture\n"
        "  --bench N          repeat a JSON request N times after warmup\n",
        argv[0], argv[0], argv[0]);
}

static bool parse_custom_args(
        int argc,
        char ** argv,
        std::vector<char *> & llama_argv,
        std::string & json_path,
        std::string & head_path,
        std::string & check_path,
        int & bench,
        std::string & err) {
    llama_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json" || arg == "--kev-head" || arg == "--check" || arg == "--bench") {
            if (i + 1 >= argc) {
                err = arg + " expects a value";
                return false;
            }
            const std::string value = argv[++i];
            if (arg == "--json") {
                json_path = value;
            } else if (arg == "--kev-head") {
                head_path = value;
            } else if (arg == "--check") {
                check_path = value;
            } else {
                try {
                    bench = std::stoi(value);
                } catch (...) {
                    err = "--bench expects an integer";
                    return false;
                }
            }
        } else {
            llama_argv.push_back(argv[i]);
        }
    }
    llama_argv.push_back(nullptr);
    if (!json_path.empty() && !check_path.empty()) {
        err = "--json and --check cannot be used together";
        return false;
    }
    if (bench < 0) {
        err = "--bench must be non-negative";
        return false;
    }
    return true;
}

static bool run_check(
        llama_context * ctx,
        const common_decision_head & head,
        const ordered_json & fixture,
        std::string & err,
        double & max_diff,
        int & flips) {
    max_diff = 0;
    flips = 0;
    for (size_t r = 0; r < fixture.size(); ++r) {
        common_decision_request req;
        if (!common_decision_parse(fixture.at(r), req, err)) {
            return false;
        }
        common_decision_result result;
        if (!common_decision_run(ctx, head, req, result, err)) {
            return false;
        }
        const auto & expected = fixture.at(r).at("raw_probs");
        for (size_t q = 0; q < result.probs.size(); ++q) {
            const auto actual_best = std::max_element(result.probs[q].begin(), result.probs[q].end());
            const auto expected_probs = expected.at(q).get<std::vector<float>>();
            const auto expected_best = std::max_element(expected_probs.begin(), expected_probs.end());
            if (actual_best - result.probs[q].begin() != expected_best - expected_probs.begin()) {
                ++flips;
            }
            for (size_t i = 0; i < result.probs[q].size(); ++i) {
                max_diff = std::max(max_diff, std::abs(static_cast<double>(result.probs[q][i]) - expected_probs[i]));
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    try {
        std::vector<char *> llama_argv;
        std::string json_path;
        std::string head_path;
        std::string check_path;
        int bench = 0;
        std::string err;
        if (!parse_custom_args(argc, argv, llama_argv, json_path, head_path, check_path, bench, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            print_usage(argc, argv);
            return 2;
        }

        common_params params;
        params.n_ctx = 8192;
        params.n_batch = 8192;
        params.n_ubatch = 512;
        params.n_parallel = 4;
        common_init();
        if (!common_params_parse(static_cast<int>(llama_argv.size() - 1), llama_argv.data(), params,
                                 LLAMA_EXAMPLE_EMBEDDING, print_usage)) {
            return 1;
        }
        params.embedding = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        params.n_parallel = std::max(2, params.n_parallel);

        if (json_path.empty() && check_path.empty()) {
            err = "one of --json or --check is required";
            std::fprintf(stderr, "%s\n", err.c_str());
            print_usage(argc, argv);
            return 2;
        }

        ordered_json body;
        if (!json_path.empty()) {
            body = ordered_json::parse(read_file(json_path));
        }
        ordered_json fixture;
        if (!check_path.empty()) {
            fixture = ordered_json::parse(read_file(check_path));
            if (!fixture.is_array()) {
                throw std::runtime_error("reference.json must contain an array");
            }
            body = fixture.at(0);
        }

        llama_backend_init();
        llama_numa_init(params.numa);
        auto llama_init = common_init_from_params(params);
        llama_context * ctx = llama_init->context();
        llama_model * model = llama_init->model();
        if (ctx == nullptr || model == nullptr) {
            throw std::runtime_error("failed to initialize model context");
        }

        common_decision_head head;
        if (!head_path.empty()) {
            if (!common_decision_load_json(head_path, llama_model_get_vocab(model), head, err)) {
                throw std::runtime_error(err);
            }
        } else if (!common_decision_load(params.model.path, head, err)) {
            throw std::runtime_error(err + "; pack one with tools/kev/kev_pack.py or pass --kev-head");
        }

        if (!check_path.empty()) {
            double max_diff = 0;
            int flips = 0;
            if (!run_check(ctx, head, fixture, err, max_diff, flips)) {
                throw std::runtime_error(err);
            }
            std::printf("overall max |dp|: %.6f\n", max_diff);
            std::printf("overall argmax flips: %d\n", flips);
            return max_diff > 0.02 || flips != 0 ? 1 : 0;
        }

        common_decision_request req;
        if (!common_decision_parse(body, req, err)) {
            throw std::runtime_error(err);
        }
        common_decision_result result;
        if (!common_decision_run(ctx, head, req, result, err)) {
            throw std::runtime_error(err);
        }

        if (bench > 0) {
            std::vector<double> state_times;
            std::vector<double> branch_times;
            for (int i = 0; i < bench; ++i) {
                common_decision_result measured;
                if (!common_decision_run(ctx, head, req, measured, err)) {
                    throw std::runtime_error(err);
                }
                state_times.push_back(measured.t_state_ms);
                branch_times.push_back(measured.t_branches_ms);
            }
            std::sort(state_times.begin(), state_times.end());
            std::sort(branch_times.begin(), branch_times.end());
            const auto mean = [](const std::vector<double> & values) {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            };
            std::fprintf(stderr, "bench p50: state %.2f ms, branches %.2f ms; mean: state %.2f ms, branches %.2f ms\n",
                state_times[state_times.size() / 2], branch_times[branch_times.size() / 2],
                mean(state_times), mean(branch_times));
        }

        ordered_json answers = common_decision_answers(req, result);
        const std::string model_name = body.value("model", "kev-latest");
        const int32_t input_tokens = result.n_state_tokens +
            std::accumulate(result.n_branch_tokens.begin(), result.n_branch_tokens.end(), 0);
        ordered_json response = {
            {"model", model_name},
            {"answers", answers},
            {"usage", {
                {"input_tokens", input_tokens},
                {"output_tokens", 0},
            }},
            {"latency_ms", std::round((result.t_state_ms + result.t_branches_ms) * 10.0) / 10.0},
        };
        std::cout << response.dump() << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
