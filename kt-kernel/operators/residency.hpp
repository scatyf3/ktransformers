// SSD expert residency, stage 1: synchronous load, no prefetch, no cross-layer cache.  The code contains
// BlobLayout for expert weight layout
// SsdExpertStore to hold handle for expert read and write, and perform expert read/write
// SlotPool pre-allocated fixed DRAM slots
// SharedResidency for SSD Expert state scedule
// LayerResidency for layer wise expert state view, hold expert offload mask only. call load_phase to load per layer expert to slotpool
// Corrness is check by expert weight bit exact with ktransformers' official loader

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
#include <type_traits>
#include <string>
#include <vector>

namespace kt_residency {

constexpr size_t kIOAlign = 4096;  // O_DIRECT alignment for buffer, offset and length

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/*
Byte sizes of one expert's three weight matrices in a single layer.
Hardcoded qwen3 layout
┌──────────── record 0 (layer0, expert0) ────────────┐
│  gate 3 MiB  │   up 3 MiB   │  down 3 MiB          │   ← AMX tile for GEMM
├──────────── record 1 (layer0, expert1) ────────────┤
│                    ...                              │
└─────────── record 6143 (layer47, expert127) ───────┘
*/
struct BlobLayout {
  size_t gate = 0;
  size_t up = 0;
  size_t down = 0;

  size_t padded_gate() const { return align_up(gate, kIOAlign); }
  size_t padded_up() const { return align_up(up, kIOAlign); }
  size_t padded_down() const { return align_up(down, kIOAlign); }
  size_t stride() const { return padded_gate() + padded_up() + padded_down(); }
};

class SsdExpertStore {
 public:

  // init 2 fd handle for read and write using give path and save the layout
  SsdExpertStore(const std::string& path, int record_count, BlobLayout layout)
      : path_(path), expert_num_(record_count), layout_(layout) {
    fd_write_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_write_ < 0) throw std::runtime_error("residency: cannot open " + path + ": " + strerror(errno));
    fd_read_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_read_ < 0) {
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

  // given the expert index, write the expert to gate/up/down buffer
  void write_expert(int record_idx, const void* gate, const void* up, const void* down) {
    const off_t base = static_cast<off_t>(record_idx) * layout_.stride();
    pwrite_all(gate, layout_.gate, base);
    pwrite_all(up, layout_.up, base + layout_.padded_gate());
    pwrite_all(down, layout_.down, base + layout_.padded_gate() + layout_.padded_up());
  }

  void fsync_all() { ::fsync(fd_write_); }

  // given the expert index, read the expert to gate/up/down buffer
  void read_expert(int record_idx, void* gate, void* up, void* down) {
    const off_t base = static_cast<off_t>(record_idx) * layout_.stride();
    pread_all(gate, layout_.padded_gate(), base);
    pread_all(up, layout_.padded_up(), base + layout_.padded_gate());
    pread_all(down, layout_.padded_down(), base + layout_.padded_gate() + layout_.padded_up());
  }

  size_t file_bytes() const { return static_cast<size_t>(expert_num_) * layout_.stride(); }

 private:
  // naive write to write len byte to offset off inside buffer
  void pwrite_all(const void* buf, size_t len, off_t off) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
      ssize_t n = ::pwrite(fd_write_, p, len, off);
      if (n < 0) {
        if (errno == EINTR) continue;  // signal mid-transfer: retry, do not fail
        throw std::runtime_error(std::string("residency: pwrite failed: ") + strerror(errno));
      }
      if (n == 0) throw std::runtime_error("residency: pwrite made no progress");
      p += n;
      off += n;
      len -= static_cast<size_t>(n);
    }
  }
  // naive read to write read byte to offset off inside buffer
  void pread_all(void* buf, size_t len, off_t off) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
      ssize_t n = ::pread(fd_read_, p, len, off);
      if (n < 0) {
        if (errno == EINTR) continue;  // signal mid-transfer: retry, do not fail
        throw std::runtime_error(std::string("residency: pread failed: ") + strerror(errno));
      }
      if (n == 0) break;  // legitimate EOF: the last record's trailing padding was never written
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
// which is reserver for expert load from ssd in dram
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

// Detects BufferB::set_data(void*). See LayerResidency::load_phase for why it is needed.
template <class B, class = void>
struct has_set_data : std::false_type {};
template <class B>
struct has_set_data<B, std::void_t<decltype(std::declval<B&>().set_data(std::declval<void*>()))>>
    : std::true_type {};

// Process-wide owner of the backing store and slot pool.
//
// Intended end state: each layer resolves its activated experts against this,
// getting DRAM hits where possible and SSD loads otherwise. 
// 
// Today only the second half exists — there is no residency map and no hit path, so every
// offloaded expert is re-read on every access (Stage 2 adds the cache).
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

// Aggregate fault counters, for reporting what a run actually paid to the device. Bytes are
// derived from the record stride rather than counted separately: every fault reads exactly one
// whole record.
struct ResidencyStats {
  uint64_t faults = 0;
  uint64_t nanos = 0;
  uint64_t bytes = 0;
  int slot_count = 0;
  bool enabled = false;
};

inline ResidencyStats residency_stats() {
  auto& sh = SharedResidency::instance();
  ResidencyStats out;
  if (!sh.ready()) return out;
  out.enabled = true;
  out.faults = sh.fault_count.load(std::memory_order_relaxed);
  out.nanos = sh.fault_nanos.load(std::memory_order_relaxed);
  out.bytes = out.faults * sh.store->layout().stride();
  out.slot_count = sh.pool->size();
  return out;
}

inline void reset_residency_stats() {
  auto& sh = SharedResidency::instance();
  sh.fault_count.store(0, std::memory_order_relaxed);
  sh.fault_nanos.store(0, std::memory_order_relaxed);
}

// Per-layer view. Owns nothing but the offload mask; storage comes from SharedResidency.
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

  /*
  
  load expert to slot pool before GEMM, core code is shown below
  for each activated expert e:
    if (!is_offloaded(e)) continue;
    slot = pool[next_slot++]           
    store->read_expert(layer*E+e, slot) 
    gate_bb[e]->set_data(slot.gate)
    gate_bb[e]->set_data(slot.gate);
    up_bb[e]->set_data(slot.up);
    down_bb[e]->set_data(slot.down);
  */
  template <class BufferVec>
  void load_phase(const int* activated_ids, int activated_count, BufferVec& gate_bb, BufferVec& up_bb,
                  BufferVec& down_bb) {
    using BufferB = typename BufferVec::value_type::element_type;
    if constexpr (!has_set_data<BufferB>::value) {
      (void)activated_ids; (void)activated_count; (void)gate_bb; (void)up_bb; (void)down_bb;
      throw std::runtime_error(
          "residency: this backend's BufferB has no set_data(); stage 1 supports --kt-method BF16 only");
    } else {
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
    }  // if constexpr has_set_data
  }

 private:
  int layer_idx_;
  int expert_num_;
  std::vector<uint8_t> mask_;
};

}  // namespace kt_residency
