#include "llama-moe-stream.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <stdexcept>

static void check(bool ok) {
    if (!ok) {
        throw std::runtime_error("Markov cache test failed");
    }
}

static void test_ranks() {
    for (bool chain : {false, true}) {
        llama_moe_stream mgr(1, 3, 1, false);
        mgr.policy = chain ? llama_moe_stream_policy::chain : llama_moe_stream_policy::hotness;
        llama_moe_stream_layer sl;
        sl.il = 0;
        sl.n_expert = 5;
        sl.n_slots = 3;
        sl.slot_expert = {2, 0, 1};
        sl.slot_state.assign(3, LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl.slot_last_use = {1, 10, 20};
        sl.route_hotness = {5, 5, 9, 0, 0};
        sl.seen.assign(5, 1);
        sl.expert_slot = {{2, 0}, {0, 1}, {1, 2}};
        const int32_t ids[] = {0, 2, 4, 0, 1, 4};
        mgr.observe_locked(sl, ids, nullptr, 3, 2, false);
        auto & r = sl.rank_stats[0];
        check(r.hits[0] == 1 && r.hits[1] == 1 && r.hits[2] == 2 && r.miss == 2);
        check(sl.route_hotness[0] == 5 && sl.slot_last_use[1] == 10);
        sl.expert_slot.erase(2);
        sl.expert_slot[3] = 0;
        sl.slot_expert[0] = 3;
        sl.route_hotness[3] = 12;
        const int32_t replacement = 3;
        mgr.observe_locked(sl, &replacement, nullptr, 1, 1, false);
        check(r.hits[0] == 2);
        sl.slot_last_use[2] = 10;
        const int32_t tie = 0;
        mgr.observe_locked(sl, &tie, nullptr, 1, 1, false);
        check(r.hits[1] == 2);
        sl.slot_state[1] = LLAMA_MOE_STREAM_SLOT_LOADING;
        mgr.observe_locked(sl, &tie, nullptr, 1, 1, false);
        check(r.miss == 3 && r.loading == 1);
        for (size_t i = 3; i < r.hits.size(); i++) { check(r.hits[i] == 0); }
        check(sl.rank_stats[1].miss == 0 && sl.rank_stats[1].hits[0] == 0);
    }
    llama_moe_stream mgr(1, 181, 1, false);
    llama_moe_stream_layer sl;
    sl.n_slots = sl.n_expert = 181;
    sl.slot_state.assign(181, LLAMA_MOE_STREAM_SLOT_RESIDENT);
    sl.slot_last_use.assign(181, 0);
    sl.seen.assign(181, 1);
    for (int32_t e = 0; e < 181; e++) {
        sl.slot_expert.push_back(e);
        sl.route_hotness.push_back(181 - e);
        sl.expert_slot[e] = e;
    }
    const int32_t ids[] = {179, 180};
    mgr.observe_locked(sl, ids, nullptr, 2, 1, true);
    check(sl.rank_stats[1].hits[179] == 1 && sl.rank_stats[1].overflow == 1);
    check(sl.rank_stats[1].miss == 0);
}

static void test_eam() {
    llama_moe_stream mgr(2, 3, 1, false);
    bool rejected = false;
    try { mgr.configure("moe-infinity", nullptr, nullptr); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    mgr.configure("moe-infinity", nullptr, nullptr, "unused");
    mgr.eam_experts = 4;
    mgr.eam_history = {{4, 0, 0, 0, 0, 2, 1, 1}, {0, 4, 0, 0, 0, 1, 2, 1}};
    for (const auto & row : mgr.eam_history) {
        double norm = 0;
        for (auto v : row) { norm += double(v)*v; }
        mgr.eam_norms.push_back(std::sqrt(norm));
    }
    llama_moe_stream_layer sl;
    sl.il = 0;
    sl.n_expert = 4;
    sl.n_slots = 3;
    sl.slot_expert = {3, 1, 2};
    sl.slot_state.assign(3, LLAMA_MOE_STREAM_SLOT_RESIDENT);
    sl.slot_last_use = {0, 1, 2};
    sl.route_hotness = {1, 10, 5, 20};
    sl.seen.assign(4, 1);
    sl.expert_slot = {{3, 0}, {1, 1}, {2, 2}};
    mgr.begin_iteration();
    const int32_t first[] = {0, 0};
    mgr.observe_locked(sl, first, nullptr, 2, 1, true);
    check(mgr.eam_match == 0 && mgr.iteration_eam[0] == 2 && mgr.iteration_eam[4] == 0);
    check(sl.score[0] == 1 && mgr.pick_victim_locked(sl, nullptr) == 1);
    sl.il = 1;
    const int32_t second = 3;
    mgr.observe_locked(sl, &second, nullptr, 1, 1, true);
    check(mgr.eam_match == 0 && sl.score[1] == .5f && sl.score[2] == .25f);
    check(mgr.pick_victim_locked(sl, nullptr) == 2);
    uint8_t keep[] = {0, 0, 1};
    check(mgr.pick_victim_locked(sl, keep) == 0);
    sl.slot_state[0] = LLAMA_MOE_STREAM_SLOT_LOADING;
    check(mgr.pick_victim_locked(sl, keep) == 1);
    sl.slot_state[1] = LLAMA_MOE_STREAM_SLOT_EMPTY;
    check(mgr.pick_victim_locked(sl, keep) == 1);
    sl.slot_state.assign(3, LLAMA_MOE_STREAM_SLOT_RESIDENT);
    mgr.begin_iteration();
    check(mgr.iteration_eam[0] == 0 && mgr.iteration_eam[7] == 0);
    sl.il = 0;
    const int32_t unseen = 2;
    mgr.observe_locked(sl, &unseen, nullptr, 1, 1, true);
    check(mgr.eam_match == -1 && sl.score.empty() && mgr.pick_victim_locked(sl, nullptr) == 2);
    mgr.eam_history[0] = mgr.eam_history[1];
    mgr.begin_iteration();
    const int32_t one = 1;
    mgr.observe_locked(sl, &one, nullptr, 1, 1, true);
    check(mgr.eam_match == 0);
    mgr.observe_locked(sl, &one, nullptr, 1, 1, false);
    check(sl.score.empty());
    mgr.eam_history[0] = {0, 4, 0, 0, 0, 0, 0, 0};
    mgr.eam_history.resize(1);
    mgr.eam_norms = {4};
    mgr.begin_iteration();
    mgr.observe_locked(sl, &one, nullptr, 1, 1, true);
    sl.il = 1;
    mgr.observe_locked(sl, &unseen, nullptr, 1, 1, true);
    check(mgr.eam_match == 0 && sl.score.empty() && !strcmp(mgr.eam_fallback, "zero_row"));
}

static void test_cpu_policy_integration() {
    for (const char * policy : {"chain", "moe-infinity"}) {
        llama_moe_stream mgr(1, 2, 1, false);
        mgr.gpu_decode_requested = true;
        mgr.gpu_decode_continuous_requested = true;
        mgr.shadow = true;
        mgr.configure(policy, "matrix", nullptr, "eam");
        check(!mgr.gpu_decode_requested && !mgr.gpu_decode_continuous_requested && !mgr.shadow);
        llama_moe_stream_layer sl;
        sl.n_expert = 3;
        sl.n_slots = 2;
        sl.slot_expert = {0, 1};
        sl.slot_state.assign(2, LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl.slot_last_use = {0, 1};
        sl.route_hotness = {0, 10, 0};
        sl.score = {0, 1, 0};
        sl.expert_map.assign(3, -1);
        sl.expert_slot = {{0, 0}, {1, 1}};
        sl.keep.resize(2);
        sl.uniq = {2, 0};
        const int32_t ids[] = {2, 0};
        check(mgr.build_plan_locked(sl, ids, 2));
        check(sl.plan.loads.size() == 1 && sl.plan.loads[0].victim == 1);
        check(sl.plan.mapped_topk[0] == 1 && sl.plan.mapped_topk[1] == 0);
    }
    {
        llama_moe_stream mgr(1, 2, 1, false);
        mgr.gpu_decode_requested = mgr.gpu_decode_continuous_requested = mgr.shadow = true;
        mgr.configure("hotness", nullptr, nullptr);
        check(mgr.gpu_decode_requested && mgr.gpu_decode_continuous_requested && mgr.shadow);
    }
    const char * trace = "moe-policy-integration-trace.tmp";
    check(!std::ifstream(trace));
    {
        llama_moe_stream mgr(1, 2, 1, false);
        mgr.gpu_decode_requested = mgr.gpu_decode_continuous_requested = mgr.shadow = true;
        mgr.configure("hotness", nullptr, trace);
        check(mgr.trace.is_open());
        check(!mgr.gpu_decode_requested && !mgr.gpu_decode_continuous_requested && !mgr.shadow);
    }
    std::remove(trace);
}

int main() {
    test_cpu_policy_integration();
    ggml_time_init();
    test_ranks();
    test_eam();
    llama_moe_stream mgr(1, 3, 1, false);
    mgr.configure("chain", "unused", nullptr);
    llama_moe_stream_layer sl;
    sl.mgr = &mgr;
    sl.il = 0;
    sl.n_expert = 4;
    sl.n_slots = 3;
    sl.slot_expert = { 3, 1, 2 };
    sl.slot_state.assign(3, LLAMA_MOE_STREAM_SLOT_RESIDENT);
    sl.slot_last_use = { 0, 1, 2 };
    sl.route_hotness = { 1, 10, 5, 20 };
    sl.seen.assign(4, 1);
    sl.expert_slot = { {3, 0}, {1, 1}, {2, 2} };
    sl.markov = { .1f, .2f, .3f, .4f,
                  .4f, .3f, .2f, .1f,
                  .25f, .25f, .25f, .25f,
                  .25f, .25f, .25f, .25f };
    int32_t ids[] = {0, 1};
    float weights[] = {.75f, .25f};
    mgr.observe_locked(sl, ids, weights, 2, 1, true);
    check(std::abs(sl.score[3] - .325f) < 1e-6f);
    check(mgr.pick_victim_locked(sl, nullptr) == 1);
    uint8_t keep[] = {0, 1, 0};
    check(mgr.pick_victim_locked(sl, keep) == 2);
    sl.slot_state[2] = LLAMA_MOE_STREAM_SLOT_LOADING;
    check(mgr.pick_victim_locked(sl, keep) == 0);
    keep[0] = 1;
    check(mgr.pick_victim_locked(sl, keep) == -1);
    keep[0] = keep[1] = 0;
    sl.slot_state[2] = LLAMA_MOE_STREAM_SLOT_EMPTY;
    check(mgr.pick_victim_locked(sl, nullptr) == 2);
    sl.slot_state[2] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
    sl.score.assign(4, .25f);
    check(mgr.pick_victim_locked(sl, nullptr) == 1);
    mgr.observe_locked(sl, ids, weights, 2, 1, false);
    check(sl.score.empty());
    check(mgr.pick_victim_locked(sl, nullptr) == 2);
    check(mgr.phases[1].hit == 1 && mgr.phases[1].miss == 1);
    sl.slot_state[1] = LLAMA_MOE_STREAM_SLOT_LOADING;
    mgr.observe_locked(sl, ids, weights, 2, 1, true);
    check(mgr.phases[1].loading == 1 && mgr.phases[1].hit == 1 && mgr.phases[1].miss == 3);
    bool rejected = false;
    try { mgr.configure("invalid", nullptr, nullptr); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    rejected = false;
    try { mgr.configure("chain", nullptr, nullptr); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    mgr.configure("chain", "missing-markov-test.bin", nullptr);
    rejected = false;
    try { mgr.load_markov({"missing-model.gguf"}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    const char * model_path = "moe-markov-unit-model.tmp";
    const char * matrix_path = "moe-markov-unit-matrix.tmp";
    check(!std::ifstream(model_path) && !std::ifstream(matrix_path));
    { std::ofstream out(model_path, std::ios::binary); out << "abc"; }
    auto write_matrix = [&](int kind) {
        std::ofstream out(matrix_path, std::ios::binary | std::ios::trunc);
        out.write("MOEMKV01", 8);
        auto u32 = [&](uint32_t n) {
            char b[4] = {char(n), char(n >> 8), char(n >> 16), char(n >> 24)};
            out.write(b, 4);
        };
        u32(kind == 1 ? 2 : 1);
        u32(4);
        const uint8_t hash[32] = {0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,
            0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,
            0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
        out.write((const char *) hash, 32);
        if (kind == 2) { return; }
        for (int i = 0; i < 16; i++) {
            float p = kind == 3 ? -0.25f : kind == 4 ? 0.5f : kind == 6 ? NAN : kind == 7 ? INFINITY : 0.25f;
            uint32_t bits;
            memcpy(&bits, &p, 4);
            u32(bits);
        }
        if (kind == 5) { out.put('x'); }
    };
    mgr.configure("chain", matrix_path, nullptr);
    mgr.layers[0] = std::make_unique<llama_moe_stream_layer>();
    mgr.layers[0]->n_expert = 4;
    write_matrix(0);
    mgr.load_markov({model_path});
    check(mgr.layers[0]->markov.size() == 16);
    for (int kind = 1; kind <= 7; kind++) {
        write_matrix(kind);
        rejected = false;
        try { mgr.load_markov({model_path}); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected);
    }
    write_matrix(0);
    // Reuse the model fixture and digest for the independent EAM format.
    const char * eam_path = "moe-eam-unit.tmp";
    check(!std::ifstream(eam_path));
    auto write_eam = [&](int kind) {
        std::ifstream source(matrix_path, std::ios::binary);
        char header[48];
        source.read(header, 48);
        std::ofstream out(eam_path, std::ios::binary | std::ios::trunc);
        out.write(kind == 1 ? "BADVERSN" : "MOEEAM01", 8);
        auto u32 = [&](uint32_t value) {
            char b[4] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
            out.write(b, 4);
        };
        u32(kind == 2 ? 2 : 1); u32(kind == 3 ? 3 : 4); u32(kind == 4 ? 0 : 1);
        if (kind == 5) { header[16] ^= 1; }
        out.write(header + 16, 32);
        for (int i = 0; i < (kind == 6 ? 3 : 4); i++) { u32(kind == 7 ? 0 : i + 1); }
        if (kind == 8) { out.put('x'); }
    };
    llama_moe_stream infinity(1, 3, 1, false);
    infinity.layers[0] = std::make_unique<llama_moe_stream_layer>();
    infinity.layers[0]->n_expert = 4;
    infinity.configure("moe-infinity", nullptr, nullptr, eam_path);
    write_eam(0);
    infinity.load_eam({model_path});
    check(infinity.eam_history.size() == 1 && infinity.eam_history[0][3] == 4);
    for (int kind = 1; kind <= 8; kind++) {
        write_eam(kind);
        rejected = false;
        try { infinity.load_eam({model_path}); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected);
    }
    write_eam(0);
    rejected = false;
    try { infinity.load_eam({model_path, model_path}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    std::remove(eam_path);
    { std::ofstream out(model_path, std::ios::binary | std::ios::trunc); out << "abd"; }
    rejected = false;
    try { mgr.load_markov({model_path}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    std::remove(model_path);
    std::remove(matrix_path);
    puts("Markov scoring, residency, protection, ties, prefill fallback and configuration: PASS");
}
