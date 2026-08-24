// Copyright (c) MoEPrefetch.
// SPDX-License-Identifier: Apache-2.0
//
// Bit-exactness of the SSD residency round trip against the operator's own packed weights.
//
// The reference here is not recomputed by the test: it is a byte-for-byte snapshot of
// gate_bb_raw_[e] taken after the operator's real load_weights() and before its real
// spill_offloaded_experts(). Every step in between is production code --
//
//     AMX_BF16_MOE_TP::load_weights()      packs the BF16 source into gate_bb_raw_
//     [snapshot taken here]                <- the reference bytes
//     spill_offloaded_experts()            writes them out and frees the DRAM
//     forward()                            calls LayerResidency::load_phase() at its real site
//     [compare]                            what each BufferB points at must equal the snapshot
//
// so a bug anywhere in that chain shows up, including in the two functions a hand-written
// mirror would have to reimplement and could therefore get wrong in the same way twice:
// spill's record keying and its buffer_b_required_size argument order, and load_phase's
// invocation from inside forward().
//
// Two layers with different offload sets run against one shared store, because with a single
// layer record_of() degenerates to the identity and a dropped layer term would go unnoticed.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "../../../cpu_backend/worker_pool.h"
#include "../../residency.hpp"
#include "../bf16-moe.hpp"

namespace {

using Kernel = amx::GemmKernel224BF16;
using Moe = AMX_BF16_MOE_TP<Kernel>;
using MoeBase = AMX_MOE_BASE<Kernel, Moe>;

// gate_bb_ and friends are public in AMX_MOE_BASE, but AMX_BF16_MOE_TP re-declares them with
// `using Base::gate_bb_` inside its default-private section, which makes the names private in
// the derived class. Reaching them through a base reference is the ordinary way around that.
MoeBase& base_of(Moe& moe) { return static_cast<MoeBase&>(moe); }

// Qwen3-30B-A3B expert dimensions.
constexpr int kHidden = 2048;
constexpr int kIntermediate = 768;

constexpr int kExpertNum = 6;
constexpr int kTopK = 2;
constexpr int kNumLayers = 2;
constexpr int kSlotCount = 4;   // peak offloaded experts activated in one layer
constexpr int kQlen = 32;       // >1 so forward() takes the prefill path; must also be a
                                // multiple of the kernel's M_STEP -- BufferA asserts on it

int g_failures = 0;

#define CHECK(cond, ...)                                   \
  do {                                                     \
    if (!(cond)) {                                         \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
      g_failures++;                                        \
    }                                                      \
  } while (0)

enum Which : int { kGate = 0, kUp = 1, kDown = 2 };
const char* which_name(int w) { return w == kGate ? "gate" : (w == kUp ? "up" : "down"); }

void* alloc64(size_t bytes) {
  void* p = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
  if (!p) { fprintf(stderr, "aligned_alloc(%zu) failed\n", bytes); std::exit(2); }
  return p;
}

// Weight-like values, distinct per (layer, expert, matrix) so that reading the wrong record
// produces a mismatch instead of a coincidence.
void fill_source(ggml_bf16_t* dst, size_t count, uint32_t seed) {
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < count; i++) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;  // xorshift32
    dst[i] = GGML_FP32_TO_BF16((static_cast<float>(x >> 8) / 8388608.0f - 1.0f) * 0.1f);
  }
}

// The BF16 source tensors load_weights() reads, laid out expert-major exactly as the operator
// indexes them: gate/up as (expert, intermediate, hidden), down as (expert, hidden, intermediate).
struct SourceWeights {
  std::vector<ggml_bf16_t> gate, up, down;

  explicit SourceWeights(int layer) {
    const size_t per_expert = static_cast<size_t>(kIntermediate) * kHidden;
    gate.resize(per_expert * kExpertNum);
    up.resize(per_expert * kExpertNum);
    down.resize(per_expert * kExpertNum);
    for (int e = 0; e < kExpertNum; e++) {
      const uint32_t base = (static_cast<uint32_t>(layer) * 1009u + e) * 3u + 1u;
      fill_source(gate.data() + per_expert * e, per_expert, base + kGate);
      fill_source(up.data() + per_expert * e, per_expert, base + kUp);
      fill_source(down.data() + per_expert * e, per_expert, base + kDown);
    }
  }
};

void compare(const void* got, const void* want, size_t len, int layer, int expert, int which) {
  if (got == nullptr) {
    fprintf(stderr, "FAIL layer %d expert %d %s: still null after load_phase\n", layer, expert,
            which_name(which));
    g_failures++;
    return;
  }
  if (std::memcmp(got, want, len) == 0) return;
  auto* g = static_cast<const uint8_t*>(got);
  auto* w = static_cast<const uint8_t*>(want);
  size_t i = 0, differing = 0;
  while (i < len && g[i] == w[i]) i++;
  for (size_t j = 0; j < len; j++) differing += (g[j] != w[j]);
  fprintf(stderr,
          "FAIL layer %d expert %d %s: first mismatch at byte %zu of %zu (got 0x%02x want 0x%02x), "
          "%zu bytes differ in total\n",
          layer, expert, which_name(which), i, len, g[i], w[i], differing);
  g_failures++;
}

}  // namespace

static int run(int argc, char** argv) {
  // Different offload sets per layer, so a dropped layer term in record_of() misaligns them.
  const std::vector<std::vector<uint8_t>> masks = {
      {0, 1, 1, 0, 1, 0},  // layer 0 offloads experts 1, 2, 4
      {1, 0, 0, 1, 1, 1},  // layer 1 offloads experts 0, 3, 4, 5
  };

  const std::string path = (argc > 1) ? argv[1] : "residency_official_test.bin";
  ::unlink(path.c_str());

  WorkerPool pool(std::min(16, std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) - 2)));

  std::vector<std::unique_ptr<Moe>> moes;
  // reference[layer][expert][which]: the packed bytes as load_weights() produced them.
  std::vector<std::vector<std::vector<void*>>> reference(
      kNumLayers, std::vector<std::vector<void*>>(kExpertNum, std::vector<void*>(3, nullptr)));

  size_t gate_bytes = 0, down_bytes = 0;

  for (int l = 0; l < kNumLayers; l++) {
    SourceWeights src(l);  // freed at the end of this scope: load_weights copies what it needs

    GeneralMOEConfig cfg(kExpertNum, kTopK, kHidden, kIntermediate);
    cfg.layer_idx = l;
    cfg.max_len = kQlen;
    cfg.pool = &pool;
    cfg.gate_proj = src.gate.data();
    cfg.up_proj = src.up.data();
    cfg.down_proj = src.down.data();
    // Residency must be configured before construction: the operator copies its config.
    cfg.ssd_experts_mask = const_cast<uint8_t*>(masks[l].data());
    cfg.ssd_store_path = path;
    cfg.ssd_num_layers = kNumLayers;
    cfg.ssd_slot_count = kSlotCount;

    auto moe = std::make_unique<Moe>(cfg, 0);
    moe->load_weights();

    // Same expression spill_offloaded_experts() uses to size a record.
    gate_bytes = Kernel::BufferB::required_size(kIntermediate, kHidden);
    down_bytes = Kernel::BufferB::required_size(kHidden, kIntermediate);

    // Snapshot before spilling: after spill_offloaded_experts() these allocations are gone.
    for (int e = 0; e < kExpertNum; e++) {
      if (!masks[l][e]) continue;
      const size_t sizes[3] = {gate_bytes, gate_bytes, down_bytes};
      void* const raws[3] = {moe->gate_bb_raw_[e], moe->up_bb_raw_[e], moe->down_bb_raw_[e]};
      for (int which = 0; which < 3; which++) {
        CHECK(raws[which] != nullptr, "layer %d expert %d %s raw buffer is null before spill", l, e,
              which_name(which));
        reference[l][e][which] = alloc64(sizes[which]);
        std::memcpy(reference[l][e][which], raws[which], sizes[which]);
      }
    }

    moe->spill_offloaded_experts();

    for (int e = 0; e < kExpertNum; e++) {
      if (!masks[l][e]) continue;
      CHECK(moe->gate_bb_raw_[e] == nullptr, "layer %d expert %d still holds DRAM after spill", l, e);
    }
    moes.push_back(std::move(moe));
  }

  auto& sh = kt_residency::SharedResidency::instance();
  printf("[test] packed sizes: gate=%zu down=%zu | store=%zu bytes, O_DIRECT=%s\n", gate_bytes,
         down_bytes, sh.store->file_bytes(), sh.store->direct() ? "yes" : "no");
  CHECK(sh.store->direct(), "O_DIRECT unavailable; this test would be measuring the page cache");

  // Route every token through every expert, so all offloaded experts of a layer are activated.
  std::vector<int64_t> expert_ids(kQlen * kTopK);
  std::vector<float> weights(kQlen * kTopK, 0.1f);
  for (int i = 0; i < kQlen * kTopK; i++) expert_ids[i] = i % kExpertNum;
  // The final reduction in forward_prefill stores to `output` with an aligned AVX512 store, so
  // these buffers have to be 64-byte aligned; a std::vector would not be.
  const size_t tokens = static_cast<size_t>(kQlen) * kHidden;
  auto* input = static_cast<ggml_bf16_t*>(alloc64(sizeof(ggml_bf16_t) * tokens));
  auto* output = static_cast<float*>(alloc64(sizeof(float) * tokens));
  fill_source(input, tokens, 7777u);

  for (int l = 0; l < kNumLayers; l++) {
    const uint64_t faults_before = sh.fault_count.load();

    // The real call site: forward() runs load_phase before its GEMM section.
    moes[l]->forward(kQlen, kTopK, expert_ids.data(), weights.data(), input, output);

    // forward() builds m_expert_id_map_ by scanning experts in ascending id order, so slots are
    // handed to the offloaded ones in that same order.
    int expected_slot = 0;
    for (int e = 0; e < kExpertNum; e++) {
      if (!masks[l][e]) continue;
      const auto& slot = sh.pool->at(expected_slot);
      CHECK(base_of(*moes[l]).gate_bb_[e]->b == slot.gate,
            "layer %d expert %d gate points at %p, expected slot %d (%p)", l, e,
            (void*)base_of(*moes[l]).gate_bb_[e]->b, expected_slot, slot.gate);
      CHECK(base_of(*moes[l]).up_bb_[e]->b == slot.up, "layer %d expert %d up not at slot %d", l, e, expected_slot);
      CHECK(base_of(*moes[l]).down_bb_[e]->b == slot.down, "layer %d expert %d down not at slot %d", l, e,
            expected_slot);

      compare(base_of(*moes[l]).gate_bb_[e]->b, reference[l][e][kGate], gate_bytes, l, e, kGate);
      compare(base_of(*moes[l]).up_bb_[e]->b, reference[l][e][kUp], gate_bytes, l, e, kUp);
      compare(base_of(*moes[l]).down_bb_[e]->b, reference[l][e][kDown], down_bytes, l, e, kDown);
      expected_slot++;
    }

    const uint64_t faults = sh.fault_count.load() - faults_before;
    CHECK(faults == static_cast<uint64_t>(expected_slot), "layer %d: %lu faults, expected %d", l,
          (unsigned long)faults, expected_slot);
    printf("[test] layer %d: %d offloaded experts faulted in by forward() and verified bit-exact\n", l,
           expected_slot);
  }

  for (auto& per_layer : reference)
    for (auto& per_expert : per_layer)
      for (void* p : per_expert) std::free(p);

  std::free(input);
  std::free(output);
  moes.clear();
  ::unlink(path.c_str());

  if (g_failures == 0) {
    printf("[test] PASS: packed weights survive load_weights -> spill -> forward/load_phase bit-exact\n");
    return 0;
  }
  fprintf(stderr, "[test] FAILED with %d error(s)\n", g_failures);
  return 1;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "[test] FAILED: uncaught exception: %s\n", e.what());
    return 1;
  }
}
