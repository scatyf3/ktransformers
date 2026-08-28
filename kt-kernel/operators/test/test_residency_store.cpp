// Copyright (c) MoEPrefetch.
// SPDX-License-Identifier: Apache-2.0
//
// Bit-exactness of the SSD expert store: bytes written by spill_offloaded_experts() must come
// back from read_expert() unchanged, for every (layer, expert) record.
//
// This is the cheap half of the residency test plan and needs neither a model nor the Python
// bindings: SsdExpertStore is header-only and talks to nothing but libc. What it can catch:
//   - stride / padding arithmetic in BlobLayout (sizes here are deliberately not 4K multiples)
//   - the three sub-offsets inside a record (a gate/up/down swap changes the payload tag)
//   - record indexing, i.e. layer_idx * expert_num + expert_id (patterns are keyed by record)
//   - the EOF path in pread_all, exercised by the last record's never-written tail padding
//
// Only the unpadded prefix of each matrix is compared. The padding is alignment filler that
// write_expert never writes, so its content is undefined by design -- comparing it would fail
// at random.

#include "../residency.hpp"

#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using kt_residency::align_up;
using kt_residency::BlobLayout;
using kt_residency::kIOAlign;
using kt_residency::SsdExpertStore;

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                            \
  do {                                              \
    if (!(cond)) {                                  \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                 \
      fprintf(stderr, "\n");                        \
      g_failures++;                                 \
    }                                               \
  } while (0)

// Matrix identity inside a record, so that a gate/up/down mixup is not silently symmetric.
enum Which : uint8_t { kGate = 0, kUp = 1, kDown = 2 };

// Deterministic, record- and matrix-specific payload. The first bytes carry the tags outright
// so a mismatch report can name the record it actually got.
void fill_pattern(void* buf, size_t len, int record, Which which) {
  auto* p = static_cast<uint8_t*>(buf);
  const uint32_t seed = static_cast<uint32_t>(record) * 3u + static_cast<uint32_t>(which);
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < len; i++) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;  // xorshift32: cheap and reproducible
    p[i] = static_cast<uint8_t>(x);
  }
  if (len >= 8) {
    p[0] = static_cast<uint8_t>(record & 0xff);
    p[1] = static_cast<uint8_t>((record >> 8) & 0xff);
    p[2] = static_cast<uint8_t>(which);
  }
}

// Aligned scratch, sized to the padded length because read_expert reads whole aligned extents.
struct Aligned {
  void* p = nullptr;
  explicit Aligned(size_t padded) {
    p = std::aligned_alloc(kIOAlign, padded);
    if (!p) { fprintf(stderr, "aligned_alloc(%zu) failed\n", padded); std::exit(2); }
  }
  ~Aligned() { std::free(p); }
  Aligned(const Aligned&) = delete;
  Aligned& operator=(const Aligned&) = delete;
};

// Reports the first differing byte: which record, which matrix, what offset, and -- when the
// payload tag survived -- which record the data actually came from.
void compare(const void* got, const void* want, size_t len, int record, Which which,
             const char* name) {
  if (std::memcmp(got, want, len) == 0) return;
  auto* g = static_cast<const uint8_t*>(got);
  auto* w = static_cast<const uint8_t*>(want);
  size_t i = 0;
  while (i < len && g[i] == w[i]) i++;
  int got_record = len >= 8 ? (g[0] | (g[1] << 8)) : -1;
  int got_which = len >= 8 ? g[2] : -1;
  fprintf(stderr,
          "FAIL record %d %s: first mismatch at byte %zu of %zu (got 0x%02x want 0x%02x); "
          "payload tag says record=%d matrix=%d\n",
          record, name, i, len, g[i], w[i], got_record, got_which);
  g_failures++;
}

}  // namespace

static int run(int argc, char** argv) {
  // Small enough to stay quick, large enough that a record spans several 4K extents. None of
  // the three sizes is a multiple of kIOAlign, and they differ from each other, so the padded
  // sub-offsets inside a record are all distinct.
  BlobLayout layout;
  layout.gate = 5000;
  layout.up = 5000;
  layout.down = 3000;

  const int num_layers = 3;   // >1, so record_of() is not the identity and a layer mixup shows
  const int expert_num = 4;
  const int records = num_layers * expert_num;

  CHECK(layout.padded_gate() == 8192, "padded_gate=%zu", layout.padded_gate());
  CHECK(layout.padded_down() == 4096, "padded_down=%zu", layout.padded_down());
  CHECK(layout.stride() == layout.padded_gate() + layout.padded_up() + layout.padded_down(),
        "stride mismatch");
  CHECK(layout.stride() % kIOAlign == 0, "stride %zu not 4K-aligned", layout.stride());

  const std::string path = (argc > 1) ? argv[1] : "residency_store_test.bin";
  ::unlink(path.c_str());  // a stale file from a previous run would mask write bugs

  // Reference copies of everything written, held in memory for the comparison below.
  std::vector<std::vector<uint8_t>> ref_gate(records), ref_up(records), ref_down(records);
  for (int r = 0; r < records; r++) {
    ref_gate[r].resize(layout.gate);
    ref_up[r].resize(layout.up);
    ref_down[r].resize(layout.down);
    fill_pattern(ref_gate[r].data(), layout.gate, r, kGate);
    fill_pattern(ref_up[r].data(), layout.up, r, kUp);
    fill_pattern(ref_down[r].data(), layout.down, r, kDown);
  }

  {
    SsdExpertStore store(path, records, layout);
    // Write out of order: sequential writes could hide an offset bug that only shows when a
    // record lands somewhere other than "wherever the file pointer happened to be".
    for (int i = 0; i < records; i++) {
      const int r = (i * 7 + 5) % records;
      store.write_expert(r, ref_gate[r].data(), ref_up[r].data(), ref_down[r].data());
    }
    store.fsync_all();
  }

  // Reopen: this must prove what is on the device, not what is still in this process's heap.
  {
    SsdExpertStore store(path, records, layout);
    printf("[test] store reopened, O_DIRECT=%s, file=%zu bytes\n", store.direct() ? "yes" : "no",
           store.file_bytes());
    CHECK(store.direct(), "O_DIRECT unavailable; this test is measuring the page cache");

    Aligned gate(layout.padded_gate()), up(layout.padded_up()), down(layout.padded_down());

    // Reverse order, so the last record -- whose trailing padding was never written, and which
    // therefore takes the EOF branch in pread_all -- is read first rather than last.
    for (int r = records - 1; r >= 0; r--) {
      std::memset(gate.p, 0xAA, layout.padded_gate());
      std::memset(up.p, 0xAA, layout.padded_up());
      std::memset(down.p, 0xAA, layout.padded_down());

      store.read_expert(r, gate.p, up.p, down.p);

      compare(gate.p, ref_gate[r].data(), layout.gate, r, kGate, "gate");
      compare(up.p, ref_up[r].data(), layout.up, r, kUp, "up");
      compare(down.p, ref_down[r].data(), layout.down, r, kDown, "down");
    }

    // Re-reading a record must be stable: two reads of the same offset returning different
    // bytes would mean the offset arithmetic depends on call history.
    Aligned gate2(layout.padded_gate()), up2(layout.padded_up()), down2(layout.padded_down());
    store.read_expert(0, gate.p, up.p, down.p);
    store.read_expert(0, gate2.p, up2.p, down2.p);
    CHECK(std::memcmp(gate.p, gate2.p, layout.gate) == 0, "record 0 gate not stable across reads");
  }

  ::unlink(path.c_str());

  if (g_failures == 0) {
    printf("[test] PASS: %d records bit-exact through write -> fsync -> reopen -> read\n", records);
    return 0;
  }
  fprintf(stderr, "[test] FAILED with %d error(s)\n", g_failures);
  return 1;
}

// The store reports every I/O problem by throwing. Catching here turns what would otherwise be
// a bare abort() into a message naming the failure, which matters most when the test is doing
// its job and something really is broken.
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "[test] FAILED: uncaught exception: %s\n", e.what());
    return 1;
  }
}
