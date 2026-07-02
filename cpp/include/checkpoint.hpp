// =============================================================================
// checkpoint.hpp — Crash-safe state persistence via memory-mapped files.
//
// Why mmap instead of write():
//   * The book/VWAP state is already POD (see order_book.hpp), so a
//     checkpoint is a memcpy into a page-cache-backed mapping — no
//     serialization, no allocation, ~single-digit microseconds for a full
//     4096-level dual-sided book (~128 KiB).
//   * The kernel flushes dirty pages asynchronously; an explicit
//     msync(MS_ASYNC) is issued per checkpoint and MS_SYNC on clean
//     shutdown. On crash we lose at most the last un-flushed interval,
//     never file consistency (see torn-write note below).
//
// Torn-write protection:
//   A crash mid-memcpy must not leave a half-written snapshot that restore()
//   would trust. We use a double-buffer ("A/B slot") layout with a
//   monotonically increasing sequence number written twice per slot
//   (header + trailer). A slot is valid iff header.seq == trailer.seq and
//   the CRC over the payload matches. Writers alternate slots; readers pick
//   the valid slot with the highest seq. This is the classic lock-free
//   seqlock-on-disk pattern used by exchange gateways.
// =============================================================================
#pragma once

#include "order_book.hpp"
#include "signal_generator.hpp"

#include <cstdint>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace microcore {

#pragma pack(push, 1)
struct SlotHeader {
    uint64_t magic;      // 'MRVCKPT1'
    uint64_t seq;        // checkpoint sequence (torn-write guard, part 1)
    uint32_t crc32;      // CRC over payload
    uint32_t payload_len;
};
struct SlotTrailer {
    uint64_t seq;        // torn-write guard, part 2 (must equal header.seq)
};
struct BookSnapshot {
    double   tick_size;
    double   last_trade;
    uint64_t book_seq;
    uint32_t n_bids;
    uint32_t n_asks;
    VwapTracker::Snapshot vwap;
    double   position;
    uint8_t  regime;
    uint8_t  _pad[7];
    Level    bids[BookSide::MAX_LEVELS];
    Level    asks[BookSide::MAX_LEVELS];
};
#pragma pack(pop)

static constexpr uint64_t CKPT_MAGIC = 0x3154504B4356524DULL;  // "MRVCKPT1"

// Slot = header + payload + trailer; file = two slots (A/B).
static constexpr size_t SLOT_SIZE =
    sizeof(SlotHeader) + sizeof(BookSnapshot) + sizeof(SlotTrailer);
static constexpr size_t CKPT_FILE_SIZE = 2 * SLOT_SIZE;

inline uint32_t crc32_sw(const uint8_t* data, size_t len) noexcept {
    // Small table-free CRC32 (reflected, poly 0xEDB88320). Checkpoints are
    // not on the tick hot path, so bitwise CRC is an acceptable trade for
    // zero static tables and zero dependencies.
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b)
            crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
    return ~crc;
}

class Checkpointer {
public:
    explicit Checkpointer(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) return;
        // Pre-size the file; mmap of a short file SIGBUSes on store.
        if (::ftruncate(fd_, static_cast<off_t>(CKPT_FILE_SIZE)) != 0) {
            ::close(fd_); fd_ = -1; return;
        }
        void* p = ::mmap(nullptr, CKPT_FILE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); fd_ = -1; return; }
        map_ = static_cast<uint8_t*>(p);
    }

    ~Checkpointer() {
        if (map_) { ::msync(map_, CKPT_FILE_SIZE, MS_SYNC); ::munmap(map_, CKPT_FILE_SIZE); }
        if (fd_ >= 0) ::close(fd_);
    }

    Checkpointer(const Checkpointer&) = delete;             // owns the mapping:
    Checkpointer& operator=(const Checkpointer&) = delete;  // non-copyable by design

    bool ok() const noexcept { return map_ != nullptr; }

    // Snapshot book + signal state into the next A/B slot. Total cost is
    // one memcpy (~128 KiB → ~10 µs) + CRC; safe to call from the data
    // thread every N seconds or every M book updates.
    bool save(const OrderBook& book, const SignalGenerator& sig) noexcept {
        if (!map_) return false;
        const uint64_t seq = ++seq_;
        uint8_t* slot = map_ + (seq % 2) * SLOT_SIZE;

        auto* hdr = reinterpret_cast<SlotHeader*>(slot);
        auto* snap = reinterpret_cast<BookSnapshot*>(slot + sizeof(SlotHeader));
        auto* trl = reinterpret_cast<SlotTrailer*>(
            slot + sizeof(SlotHeader) + sizeof(BookSnapshot));

        // Write payload first, then trailer, then header (with CRC) last —
        // header.seq becoming visible is the "commit" of the slot.
        snap->tick_size  = book.tick_size();
        snap->last_trade = book.last_trade();
        snap->book_seq   = book.seq();
        snap->n_bids     = static_cast<uint32_t>(book.bids().size());
        snap->n_asks     = static_cast<uint32_t>(book.asks().size());
        snap->vwap       = book.vwap_tracker().snapshot();
        snap->position   = sig.position();
        snap->regime     = static_cast<uint8_t>(sig.regime());
        std::memset(snap->_pad, 0, sizeof(snap->_pad));
        std::memcpy(snap->bids, book.bids().data(), snap->n_bids * sizeof(Level));
        std::memcpy(snap->asks, book.asks().data(), snap->n_asks * sizeof(Level));

        trl->seq = seq;

        hdr->magic = CKPT_MAGIC;
        hdr->payload_len = sizeof(BookSnapshot);
        hdr->crc32 = crc32_sw(reinterpret_cast<uint8_t*>(snap), sizeof(BookSnapshot));
        hdr->seq = seq;  // commit point

        ::msync(slot, SLOT_SIZE, MS_ASYNC);  // hint kernel; non-blocking
        return true;
    }

    // Restore from the newest valid slot. Returns false if no valid
    // checkpoint exists (fresh start) — caller then resyncs from exchange
    // snapshots instead.
    bool load(OrderBook& book, SignalGenerator& sig) noexcept {
        if (!map_) return false;
        const BookSnapshot* best = nullptr;
        uint64_t best_seq = 0;

        for (int i = 0; i < 2; ++i) {
            const uint8_t* slot = map_ + i * SLOT_SIZE;
            const auto* hdr = reinterpret_cast<const SlotHeader*>(slot);
            const auto* snap = reinterpret_cast<const BookSnapshot*>(slot + sizeof(SlotHeader));
            const auto* trl = reinterpret_cast<const SlotTrailer*>(
                slot + sizeof(SlotHeader) + sizeof(BookSnapshot));

            if (hdr->magic != CKPT_MAGIC) continue;
            if (hdr->seq == 0 || hdr->seq != trl->seq) continue;      // torn
            if (hdr->payload_len != sizeof(BookSnapshot)) continue;   // ABI drift
            if (crc32_sw(reinterpret_cast<const uint8_t*>(snap),
                         sizeof(BookSnapshot)) != hdr->crc32) continue;
            if (hdr->seq > best_seq) { best_seq = hdr->seq; best = snap; }
        }
        if (!best) return false;

        book.bids_mut().restore(best->bids, best->n_bids);
        book.asks_mut().restore(best->asks, best->n_asks);
        book.vwap_tracker_mut().restore(best->vwap);
        book.set_seq(best->book_seq);
        book.set_last_trade(best->last_trade);
        sig.set_position(best->position);
        sig.set_regime(static_cast<Regime>(best->regime));
        seq_ = best_seq;
        return true;
    }

private:
    int      fd_  = -1;
    uint8_t* map_ = nullptr;
    uint64_t seq_ = 0;
};

}  // namespace microcore
