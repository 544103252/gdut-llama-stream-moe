#include "llama-moe-stream.h"
#include "../vendor/nlohmann/json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <sstream>

static void check(bool ok) {
    if (!ok) {
        throw std::runtime_error("Markov cache test failed");
    }
}

static void test_fate() {
    const auto cap = llama_moe_stream_fate_capacity({10, 10, 0, 10, 10, 10, 10}, 10, 2, 2, 360);
    check(cap == std::vector<uint32_t>({10, 10, 0, 4, 4, 4, 4}));
    check(llama_moe_stream_fate_capacity({10, 20}, 10, 2, 1, 151) == std::vector<uint32_t>({10, 2}));
    check(llama_moe_stream_fate_capacity({10, 10}, 10, 2, 0, 55) == std::vector<uint32_t>({3, 2}));
    check(llama_moe_stream_fate_capacity({10, 10}, 10, 2, 4, 200) == std::vector<uint32_t>({10, 10}));
    bool rejected = false;
    try { llama_moe_stream_fate_capacity({10, 20}, 10, 2, 1, 59); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    llama_moe_stream_arc arc;
    std::vector<int32_t> slots(2, -1);
    std::vector<uint8_t> states(2, LLAMA_MOE_STREAM_SLOT_EMPTY);
    auto access = [&](int32_t e, bool demand = true) {
        const int32_t s = arc.access(e, slots, states, nullptr, demand);
        check(s >= 0);
        slots[s] = e;
        states[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        check(arc.t1.size() + arc.t2.size() <= 2 && arc.t1.size() + arc.b1.size() <= 2);
        check(arc.t1.size() + arc.t2.size() + arc.b1.size() + arc.b2.size() <= 4);
        return s;
    };
    access(0); access(1); access(0); access(2);
    check(arc.b1 == std::deque<int32_t>({1}) && arc.t2 == std::deque<int32_t>({0}));
    access(1);
    check(arc.p == 1 && arc.b2 == std::deque<int32_t>({0}));
    access(0);
    check(arc.p == 0 && arc.b1 == std::deque<int32_t>({2}));
    const uint8_t keep[] = {1, 1};
    const auto before = arc.t1;
    check(arc.access(3, slots, states, keep) == -1 && arc.t1 == before);
    states.assign(2, LLAMA_MOE_STREAM_SLOT_LOADING);
    check(arc.access(3, slots, states, nullptr) == -1);
    states.assign(2, LLAMA_MOE_STREAM_SLOT_RESIDENT);
    access(3, false);
    check(!arc.pending.empty());
    access(3);
    check(arc.pending.empty() && arc.t1.front() == 3);
    access(3);
    check(arc.t2.front() == 3);
    for (int i = 0; i < 1000; i++) { access((i*i + i/7)%9); }
    puts("Fate byte allocation, ARC adaptation, ghosts, preload and protection: PASS");
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

static void test_stmoe() {
    using predictor = llama_moe_stmoe;
    llama_moe_stmoe_pair pair{0, 2, 5, {{{0, 1}, {1, 0}}, {{0, 1}, {2, 3}}, {}, {}, {}}};
    check(predictor::predict(pair, {0, 1}, {}) == std::vector<int32_t>({0, 2}));
    check(predictor::predict(pair, {0, 1}, {1, 2}) == std::vector<int32_t>({0, 1, 2}));
    check(predictor::predict(pair, {2}, {}).empty());
    predictor::update(pair, {0, 1}, {2, 3});
    check(pair.rows[0][0].confidence == 0);
    check(pair.rows[0][1].expert == 2 && pair.rows[0][1].confidence == 2);
    check(pair.rows[1][1].confidence == 3);
    predictor::update(pair, {0}, {2, 3});
    check(pair.rows[0][0].expert == 3 && pair.rows[0][0].confidence == 2);
    check(pair.rows[0][1].expert == 2 && pair.rows[0][1].confidence == 3);
    llama_moe_stmoe_pair full{0, 2, 2, {{{0, 0}, {1, 0}}, {}}};
    predictor::update(full, {0}, {1});
    check(full.rows[0][0].expert == 0 && full.rows[0][0].confidence == 0);

    predictor p;
    p.k = 2;
    p.layers = {0, 2};
    p.pairs = {{0, 2, 4, {{{0, 2}, {1, 2}}, {}, {}, {}}}};
    p.reset();
    std::ostringstream trace;
    const int32_t source[] = {0, 1}, target[] = {2, 3};
    p.observe(0, source, 2, &trace);
    check(p.pending[0] == std::vector<int32_t>({0, 1}) && p.history[1].empty());
    p.observe(2, target, 2, &trace);
    check(p.step == 1 && p.next == 0 && p.stats[0].hit == 0 && p.stats[0].actual == 2);
    p.observe(0, source, 2, &trace);
    check(p.pending[0] == std::vector<int32_t>({2, 3}));
    p.observe(2, target, 2, &trace);
    check(p.stats[0].hit == 2 && p.stats[0].all_hit == 1 && p.step == 2);
    bool rejected = false;
    try { p.observe(2, target, 2, nullptr); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    const int32_t duplicate[] = {0, 0};
    rejected = false;
    try { p.observe(0, duplicate, 2, nullptr); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    p.reset();
    check(p.step == 0 && p.stats[0].actual == 0 && p.history[1].empty());

    const char * model_path = "test-stmoe-model.tmp", * cct_path = "test-stmoe-cct.tmp";
    { std::ofstream out(model_path, std::ios::binary); out << "abc"; }
    nlohmann::json valid = {
        {"version", 1}, {"layer_count", 3}, {"k", 2}, {"layers", {0, 2}},
        {"model_sha256", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"pairs", {{{"source", 0}, {"target", 2}, {"experts", 4},
                    {"rows", nlohmann::json::array({{{0, 2}, {1, 2}}, nlohmann::json::array(),
                                                  nlohmann::json::array(), nlohmann::json::array()})}}}}
    };
    llama_moe_stream mgr(3, 3, 1, false);
    mgr.configure("hotness", nullptr, nullptr, nullptr, cct_path);
    mgr.stmoe.k = 2;
    mgr.stmoe_layers = {0, 2};
    for (int il : mgr.stmoe_layers) {
        mgr.layers[il] = std::make_unique<llama_moe_stream_layer>();
        mgr.layers[il]->n_expert = 4;
    }
    auto write = [&](const nlohmann::json & doc) { std::ofstream out(cct_path); out << doc; };
    write(valid);
    mgr.load_stmoe({model_path});
    check(mgr.stmoe.pairs.size() == 1 && mgr.stmoe.pairs[0].rows[1].empty());
    mgr.layers[0]->seen.assign(4, 0);
    const int32_t prefill[] = {0, 1, 2, 3};
    mgr.observe_locked(*mgr.layers[0], prefill, nullptr, 2, 2, false);
    check(mgr.stmoe.step == 0 && mgr.stmoe.history[0].empty() &&
            mgr.stmoe.pairs[0].rows[0][0].confidence == 2);
    rejected = false;
    try { mgr.load_stmoe({model_path, model_path}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    for (int kind = 0; kind < 15; kind++) {
        auto doc = valid;
        if (kind == 0) { doc["version"] = 2; }
        if (kind == 1) { doc["model_sha256"] = std::string(64, '0'); }
        if (kind == 2) { doc["k"] = 1; }
        if (kind == 3) { doc["layer_count"] = 4; }
        if (kind == 4) { doc["layers"] = {0}; }
        if (kind == 5) { doc["pairs"][0]["target"] = 1; }
        if (kind == 6) { doc["pairs"][0]["experts"] = 5; }
        if (kind == 7) { doc["pairs"][0]["rows"][0][1][0] = 0; }
        if (kind == 8) { doc["pairs"][0]["rows"][0][0][0] = 4; }
        if (kind == 9) { doc["pairs"][0]["rows"][0][0][1] = 4; }
        if (kind == 10) { doc["pairs"][0]["rows"][0][0][1] = -1; }
        if (kind == 11) { doc["pairs"][0]["rows"][0][0][1] = 2.5; }
        if (kind == 12) { doc["pairs"][0]["rows"].erase(0); }
        if (kind == 13) { doc["pairs"][0]["rows"][0].erase(0); }
        if (kind == 14) { doc["layers"][0] = false; }
        write(doc);
        rejected = false;
        try { mgr.load_stmoe({model_path}); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected);
    }
    write(valid);
    mgr.layers[2].reset();
    rejected = false;
    try { mgr.load_stmoe({model_path}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    { std::ofstream out(cct_path); out << "{bad"; }
    rejected = false;
    try { mgr.load_stmoe({model_path}); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    std::remove(model_path);
    std::remove(cct_path);
    puts("ST-MoE scores, history, updates, ordering and calibration validation: PASS");
}

int main() {
    ggml_time_init();
    test_stmoe();
    test_fate();
    test_cpu_policy_integration();
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
