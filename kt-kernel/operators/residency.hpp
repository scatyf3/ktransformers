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

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
  SsdExpertStore(const std::string& path, int expert_num, BlobLayout layout)
      : path_(path), expert_num_(expert_num), layout_(layout) {
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
  void write_expert(int expert_idx, const void* gate, const void* up, const void* down) {
    const off_t base = static_cast<off_t>(expert_idx) * layout_.stride();
    pwrite_all(gate, layout_.gate, base);
    pwrite_all(up, layout_.up, base + layout_.padded_gate());
    pwrite_all(down, layout_.down, base + layout_.padded_gate() + layout_.padded_up());
  }

  void fsync_all() { ::fsync(fd_write_); }

  // Blocking read into caller-supplied 4096-aligned buffers. Reads the padded length so the
  // request stays O_DIRECT-legal; the destination buffers must be padded accordingly.
  void read_expert(int expert_idx, void* gate, void* up, void* down) {
    const off_t base = static_cast<off_t>(expert_idx) * layout_.stride();
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

}  // namespace kt_residency
