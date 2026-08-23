// Copyright (c) MoEPrefetch.
// SPDX-License-Identifier: Apache-2.0
//
// SSD expert residency, stage 1: synchronous load, no prefetch, no cross-layer cache.
//
// Motivation: kt currently requires every expert to fit in CPU DRAM. `gate_bb_`/`up_bb_`/
// `down_bb_` are allocated per expert at construction and never change afterwards, so the
// resident set is fixed at `expert_num`. This header adds a third tier: a subset of experts
// keeps no permanent DRAM buffer and is faulted in from a backing file on demand.
//
// Key structural constraint that shapes this design: the GEMM section of a forward pass is
// parallel *across experts* (`do_work_stealing_job(nth * activated_expert * 2, ...)`), so a
// fault taken inside it would either need one scratch slot per concurrently-computed expert
// (no memory saved) or a lock (parallelism destroyed). Instead the fault is hoisted into a
// separate phase that runs *before* the GEMM section:
//
//     forward()
//       |- determine activated experts
//       |- [load phase]  ensure_resident(e) for each activated SSD expert -> BufferB::set_data
//       `- [GEMM phase]  unchanged; weight pointers are already fixed
//
// Because slot assignment only happens in the load phase, weight buffers stay immutable for
// the whole duration of the GEMM phase. The "weights never move after load" assumption that
// the rest of the operator tree relies on is therefore preserved, and no pinning or
// refcounting is required at this stage.
//
// Reads use O_DIRECT: on a machine with hundreds of GB of page cache, buffered reads would
// report cache hits rather than device behaviour and make the measurement meaningless.

#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace kt_residency {

constexpr size_t kIOAlign = 4096;  // O_DIRECT alignment for buffer, offset and length

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Byte sizes of one expert's three packed weight matrices, as produced by BufferB.
struct BlobLayout {
  size_t gate = 0;
  size_t up = 0;
  size_t down = 0;

  size_t padded_gate() const { return align_up(gate, kIOAlign); }
  size_t padded_up() const { return align_up(up, kIOAlign); }
  size_t padded_down() const { return align_up(down, kIOAlign); }
  size_t stride() const { return padded_gate() + padded_up() + padded_down(); }
};

// Backing file holding already-packed expert weights, one fixed-size record per expert.
// Records are padded so every read is O_DIRECT-legal.
class SsdExpertStore {
 public:
  // `record_count` spans every (layer, expert) pair, not just one layer: all layer
  // operators share a single store so the transient slot pool is amortised across layers.
  SsdExpertStore(const std::string& path, int record_count, BlobLayout layout)
      : path_(path), expert_num_(record_count), layout_(layout) {
    fd_write_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_write_ < 0) throw std::runtime_error("residency: cannot open " + path + ": " + strerror(errno));
    fd_read_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_read_ < 0) {
      // Not fatal: some filesystems reject O_DIRECT. Fall back, but say so loudly since it
      // silently turns every measurement into a page-cache measurement.
      fd_read_ = ::open(path.c_str(), O_RDONLY);
      direct_ = false;
      fprintf(stderr, "[residency] WARNING: O_DIRECT unavailable on %s, reads go through page cache\n",
              path.c_str());
    }
    if (fd_read_ < 0) throw std::runtime_error("residency: cannot open for read " + path);
  }

  ~SsdExpertStore() {
    if (fd_write_ >= 0) ::close(fd_write_);
    if (fd_read_ >= 0) ::close(fd_read_);
  }

  SsdExpertStore(const SsdExpertStore&) = delete;
  SsdExpertStore& operator=(const SsdExpertStore&) = delete;

  bool direct() const { return direct_; }
  const BlobLayout& layout() const { return layout_; }

  // Buffered write, used once at startup to spill packed weights.
  void write_expert(int record_idx, const void* gate, const void* up, const void* down) {
    const off_t base = static_cast<off_t>(record_idx) * layout_.stride();
    pwrite_all(gate, layout_.gate, base);
    pwrite_all(up, layout_.up, base + layout_.padded_gate());
    pwrite_all(down, layout_.down, base + layout_.padded_gate() + layout_.padded_up());
  }

  void fsync_all() { ::fsync(fd_write_); }

  // Blocking read into caller-supplied 4096-aligned buffers. Reads the padded length so the
  // request stays O_DIRECT-legal; the destination buffers must be padded accordingly.
  void read_expert(int record_idx, void* gate, void* up, void* down) {
    const off_t base = static_cast<off_t>(record_idx) * layout_.stride();
    pread_all(gate, layout_.padded_gate(), base);
    pread_all(up, layout_.padded_up(), base + layout_.padded_gate());
    pread_all(down, layout_.padded_down(), base + layout_.padded_gate() + layout_.padded_up());
  }

  size_t file_bytes() const { return static_cast<size_t>(expert_num_) * layout_.stride(); }

 private:
  void pwrite_all(const void* buf, size_t len, off_t off) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
      ssize_t n = ::pwrite(fd_write_, p, len, off);
      if (n <= 0) throw std::runtime_error(std::string("residency: pwrite failed: ") + strerror(errno));
      p += n;
      off += n;
      len -= static_cast<size_t>(n);
    }
  }

  void pread_all(void* buf, size_t len, off_t off) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
      ssize_t n = ::pread(fd_read_, p, len, off);
      if (n < 0) throw std::runtime_error(std::string("residency: pread failed: ") + strerror(errno));
      if (n == 0) break;  // short read at EOF: trailing padding of the last record
      p += n;
      off += n;
      len -= static_cast<size_t>(n);
    }
  }

  std::string path_;
  int expert_num_;
  BlobLayout layout_;
  int fd_write_ = -1;
  int fd_read_ = -1;
  bool direct_ = true;
};

// Fixed pool of DRAM slots, each holding one expert's three packed matrices.
class SlotPool {
 public:
  SlotPool(int slot_count, BlobLayout layout) : layout_(layout) {
    slots_.reserve(slot_count);
    for (int i = 0; i < slot_count; i++) {
      Slot s;
      s.gate = std::aligned_alloc(kIOAlign, layout.padded_gate());
      s.up = std::aligned_alloc(kIOAlign, layout.padded_up());
      s.down = std::aligned_alloc(kIOAlign, layout.padded_down());
      if (!s.gate || !s.up || !s.down) throw std::runtime_error("residency: slot allocation failed");
      slots_.push_back(s);
    }
  }

  ~SlotPool() {
    for (auto& s : slots_) {
      std::free(s.gate);
      std::free(s.up);
      std::free(s.down);
    }
  }

  SlotPool(const SlotPool&) = delete;
  SlotPool& operator=(const SlotPool&) = delete;

  struct Slot {
    void* gate = nullptr;
    void* up = nullptr;
    void* down = nullptr;
  };

  int size() const { return static_cast<int>(slots_.size()); }
  const Slot& at(int i) const { return slots_[i]; }
  size_t bytes() const { return slots_.size() * layout_.stride(); }

 private:
  BlobLayout layout_;
  std::vector<Slot> slots_;
};

// Zero-cost default. Every member is trivially inlined away, so an operator instantiated with
// this policy compiles to exactly what it compiled to before residency existed.
struct NullResidency {
  static constexpr bool kEnabled = false;
  bool is_offloaded(int64_t) const { return false; }
  int slot_of(int64_t) const { return -1; }
};

// Store and slot pool shared by every layer operator in the process.
//
// Sharing is not an optimisation, it is required for the design to save anything: layers each
// own their own operator instance, so a per-layer pool of N slots would cost num_layers * N
// slots of DRAM and cancel out exactly what offloading freed. Layers execute sequentially
// within a forward pass, so one pool suffices.
struct SharedResidency {
  std::unique_ptr<SsdExpertStore> store;
  std::unique_ptr<SlotPool> pool;
  int expert_num = 0;
  std::atomic<uint64_t> fault_count{0};
  std::atomic<uint64_t> fault_nanos{0};

  static SharedResidency& instance() {
    static SharedResidency s;
    return s;
  }

  // Idempotent: the first layer to reach this wins, the rest observe the built state.
  void init_once(const std::string& path, int num_layers, int expert_num_, BlobLayout layout,
                 int slot_count) {
    std::lock_guard<std::mutex> lk(mu_);
    if (store) return;
    expert_num = expert_num_;
    store = std::make_unique<SsdExpertStore>(path, num_layers * expert_num_, layout);
    pool = std::make_unique<SlotPool>(slot_count, layout);
    fprintf(stderr,
            "[residency] store=%s direct=%s records=%d (%d layers x %d experts) "
            "file=%.1f GiB | slots=%d (%.0f MiB, shared across layers)\n",
            path.c_str(), store->direct() ? "yes" : "NO", num_layers * expert_num_, num_layers,
            expert_num_, store->file_bytes() / 1073741824.0, slot_count, pool->bytes() / 1048576.0);
  }

  bool ready() const { return store != nullptr; }
  int record_of(int layer_idx, int64_t expert_id) const {
    return layer_idx * expert_num + static_cast<int>(expert_id);
  }

 private:
  std::mutex mu_;
};

// Per-layer view. Owns nothing but the offload mask; storage comes from SharedResidency.
//
// Stage 1 contract: `load_phase` must be called after the activated-expert list is built and
// before the GEMM section. It assigns one slot per activated offloaded expert and repoints the
// corresponding BufferB objects, after which weight pointers stay fixed for the whole GEMM
// section. No eviction happens while compute is running, so no pinning is needed.
class LayerResidency {
 public:
  LayerResidency(int layer_idx, int expert_num, const uint8_t* offload_mask)
      : layer_idx_(layer_idx), expert_num_(expert_num) {
    mask_.assign(expert_num, 0);
    if (offload_mask) {
      for (int i = 0; i < expert_num; i++) mask_[i] = offload_mask[i];
    }
  }

  bool is_offloaded(int64_t e) const {
    return e >= 0 && e < expert_num_ && mask_[static_cast<size_t>(e)];
  }
  int offloaded_count() const {
    int n = 0;
    for (auto v : mask_) n += v ? 1 : 0;
    return n;
  }

  // BufferPtrs is any container of shared_ptr<BufferB> supporting operator[].
  template <class BufferVec>
  void load_phase(const int* activated_ids, int activated_count, BufferVec& gate_bb, BufferVec& up_bb,
                  BufferVec& down_bb) {
    auto& sh = SharedResidency::instance();
    if (!sh.ready()) return;
    int next_slot = 0;
    for (int i = 0; i < activated_count; i++) {
      const int64_t e = activated_ids[i];
      if (!is_offloaded(e)) continue;
      if (next_slot >= sh.pool->size()) {
        throw std::runtime_error(
            "residency: slot pool exhausted (" + std::to_string(sh.pool->size()) +
            " slots); it must hold every offloaded expert activated in one layer. Raise "
            "ssd_slot_count or lower the offload fraction.");
      }
      const auto& slot = sh.pool->at(next_slot++);
      auto t0 = std::chrono::steady_clock::now();
      sh.store->read_expert(sh.record_of(layer_idx_, e), slot.gate, slot.up, slot.down);
      sh.fault_nanos.fetch_add(
          (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
      sh.fault_count.fetch_add(1, std::memory_order_relaxed);
      gate_bb[e]->set_data(slot.gate);
      up_bb[e]->set_data(slot.up);
      down_bb[e]->set_data(slot.down);
    }
  }

 private:
  int layer_idx_;
  int expert_num_;
  std::vector<uint8_t> mask_;
};

}  // namespace kt_residency
