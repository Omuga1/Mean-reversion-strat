// =============================================================================
// order_book.hpp — L2 Limit Order Book with microstructure analytics.
//
// Design goals (hot path = apply_delta / obi / vwap_zscore):
//   * ZERO heap allocation after construction. Both book sides live in
//     fixed-capacity contiguous arrays owned by the OrderBook object. Delta
//     updates mutate in place via binary-search + memmove. This eliminates
//     allocator jitter and heap fragmentation during high-throughput tick
//     bursts (Coinbase L2 can burst >10k deltas/sec on BTC-USD during
//     liquidation cascades) and makes memory usage a compile-time constant.
//   * Prices are stored as int64 ticks (price / tick_size). Floating-point
//     prices are never compared for equality; float keys in a book are a
//     classic source of phantom levels ("59999.999999" vs "60000.0").
//   * Contiguous arrays beat std::map<price, qty> for L2 books: the top of
//     book — where >95% of deltas land under a mean-reversion workload —
//     stays in L1/L2 cache, and the memmove cost of mid-array inserts is
//     bounded by MAX_LEVELS * sizeof(Level) ≈ 64 KiB per side.
//   * The entire book state is a POD blob, so crash-recovery checkpointing
//     is a single memcpy into a memory-mapped file (see checkpoint.hpp).
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace microcore {

enum class Side : uint8_t { BID = 0, ASK = 1 };

// One price level. 16 bytes, trivially copyable — 4 levels per cache line.
struct Level {
    int64_t price_ticks;  // price / tick_size, exact integer arithmetic
    double  qty;          // resting base-asset quantity at this level
};
static_assert(sizeof(Level) == 16, "Level must stay 16 bytes (POD, mmap-able)");

// -----------------------------------------------------------------------------
// BookSide: fixed-capacity sorted flat array.
//   Bids sorted descending (index 0 = best bid).
//   Asks sorted ascending  (index 0 = best ask).
// Memory management contract: `levels_` is a raw in-object array. No pointers,
// no dynamic allocation, no destructor logic — the side is trivially
// serializable and immune to leaks by construction.
// -----------------------------------------------------------------------------
class BookSide {
public:
    static constexpr size_t MAX_LEVELS = 4096;

    explicit BookSide(Side side = Side::BID) : side_(side), count_(0) {}

    // Apply an L2 delta: qty == 0 deletes the level, otherwise upsert.
    // Cost: O(log n) search + O(n) memmove worst case; in practice deltas
    // cluster at the touch so the memmove window is tiny.
    void apply_delta(int64_t price_ticks, double qty) noexcept {
        const size_t idx = lower_bound(price_ticks);
        const bool found = idx < count_ && levels_[idx].price_ticks == price_ticks;

        if (qty <= 0.0) {                       // -- delete --
            if (found) {
                std::memmove(&levels_[idx], &levels_[idx + 1],
                             (count_ - idx - 1) * sizeof(Level));
                --count_;
            }
            return;
        }
        if (found) {                            // -- replace --
            levels_[idx].qty = qty;             // L2 feeds send absolute qty
            return;
        }
        // -- insert --
        if (count_ == MAX_LEVELS) {
            // Book full: drop the level farthest from the touch (last index).
            // Bounded-memory guarantee: we sacrifice depth-4096 fidelity,
            // never correctness at the top of book where signals are computed.
            if (idx == MAX_LEVELS) return;      // new level is worse than all
            --count_;
        }
        std::memmove(&levels_[idx + 1], &levels_[idx],
                     (count_ - idx) * sizeof(Level));
        levels_[idx] = Level{price_ticks, qty};
        ++count_;
    }

    // Sum of resting qty over the top `depth` levels — the V_bid / V_ask
    // terms of the OBI. Sequential scan over contiguous memory: the
    // hardware prefetcher makes this effectively free for depth <= 64.
    double volume(size_t depth) const noexcept {
        const size_t n = std::min(depth, count_);
        double v = 0.0;
        for (size_t i = 0; i < n; ++i) v += levels_[i].qty;
        return v;
    }

    size_t       size()  const noexcept { return count_; }
    bool         empty() const noexcept { return count_ == 0; }
    const Level& best()  const noexcept { return levels_[0]; }
    const Level& at(size_t i) const noexcept { return levels_[i]; }
    void         clear() noexcept { count_ = 0; }

    // Raw access for mmap checkpointing (POD snapshot).
    const Level* data() const noexcept { return levels_; }
    void restore(const Level* src, size_t n) noexcept {
        count_ = std::min(n, MAX_LEVELS);
        std::memcpy(levels_, src, count_ * sizeof(Level));
    }

private:
    // Branch-light binary search honoring per-side sort order.
    size_t lower_bound(int64_t price_ticks) const noexcept {
        size_t lo = 0, hi = count_;
        while (lo < hi) {
            const size_t mid = (lo + hi) >> 1;
            const bool go_right = (side_ == Side::BID)
                ? levels_[mid].price_ticks > price_ticks   // bids descending
                : levels_[mid].price_ticks < price_ticks;  // asks ascending
            if (go_right) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    Side   side_;
    size_t count_;
    Level  levels_[MAX_LEVELS];   // in-object storage: no heap, no leaks
};

// -----------------------------------------------------------------------------
// VwapTracker: session VWAP + standard-deviation bands from trade prints.
//
// Maintains three running sums over executed trades (p_i, v_i):
//   S0 = Σ v_i          (total volume)
//   S1 = Σ p_i · v_i    (notional)
//   S2 = Σ p_i² · v_i   (second moment, volume-weighted)
//
//   VWAP     = S1 / S0
//   Var(p)   = S2/S0 − VWAP²         (volume-weighted population variance)
//   band(k)  = VWAP ± k·σ
//
// Numerical note: S2/S0 − VWAP² suffers catastrophic cancellation when
// σ/price is tiny over a long session, so sums are periodically re-anchored
// around the running VWAP (see fold_anchor). Doubles carry ~15.9 significant
// digits; re-anchoring keeps the cancellation error orders of magnitude
// below one tick even on 24h BTC sessions.
// -----------------------------------------------------------------------------
class VwapTracker {
public:
    void reset() noexcept { s0_ = s1_ = s2_ = 0.0; anchor_ = 0.0; n_ = 0; }

    void on_trade(double price, double qty) noexcept {
        if (qty <= 0.0) return;
        if (n_ == 0) anchor_ = price;         // first print anchors the sums
        const double d = price - anchor_;     // work in offset space
        s0_ += qty;
        s1_ += d * qty;
        s2_ += d * d * qty;
        if (++n_ % REANCHOR_EVERY == 0) fold_anchor();
    }

    double vwap() const noexcept {
        return s0_ > 0.0 ? anchor_ + s1_ / s0_ : 0.0;
    }

    // Volume-weighted std-dev of traded prices around VWAP.
    double sigma() const noexcept {
        if (s0_ <= 0.0) return 0.0;
        const double m = s1_ / s0_;
        const double var = s2_ / s0_ - m * m;  // offset space == price space var
        return var > 0.0 ? std::sqrt(var) : 0.0;
    }

    // z-score of an arbitrary price vs the VWAP distribution. This is the
    // primary mean-reversion coordinate: entries trigger at z ≤ −k.
    double zscore(double price) const noexcept {
        const double s = sigma();
        return s > 0.0 ? (price - vwap()) / s : 0.0;
    }

    double band(double k) const noexcept { return vwap() + k * sigma(); }
    double total_volume() const noexcept { return s0_; }

    // POD snapshot hooks for checkpointing.
    struct Snapshot { double s0, s1, s2, anchor; uint64_t n; };
    Snapshot snapshot() const noexcept { return {s0_, s1_, s2_, anchor_, n_}; }
    void restore(const Snapshot& s) noexcept {
        s0_ = s.s0; s1_ = s.s1; s2_ = s.s2; anchor_ = s.anchor; n_ = s.n;
    }

private:
    static constexpr uint64_t REANCHOR_EVERY = 65536;

    // Shift the anchor to the current VWAP. Algebraically exact:
    //   with a' = a + m,  d' = d − m:
    //   S1' = S1 − m·S0,  S2' = S2 − 2m·S1 + m²·S0
    void fold_anchor() noexcept {
        if (s0_ <= 0.0) return;
        const double m = s1_ / s0_;
        s2_ -= m * (2.0 * s1_ - m * s0_);
        s1_ = 0.0;
        anchor_ += m;
    }

    double   s0_ = 0.0, s1_ = 0.0, s2_ = 0.0;
    double   anchor_ = 0.0;
    uint64_t n_ = 0;
};

// -----------------------------------------------------------------------------
// OrderBook: full L2 book + trade tape analytics for one instrument.
// -----------------------------------------------------------------------------
class OrderBook {
public:
    OrderBook(double tick_size, size_t obi_depth = 20)
        : tick_size_(tick_size), obi_depth_(obi_depth),
          bids_(Side::BID), asks_(Side::ASK) {}

    // ---- market data ingestion (hot path) ----------------------------------
    void apply_delta(Side side, double price, double qty) noexcept {
        const int64_t ticks = to_ticks(price);
        (side == Side::BID ? bids_ : asks_).apply_delta(ticks, qty);
        ++seq_;
    }

    void on_trade(double price, double qty) noexcept {
        vwap_.on_trade(price, qty);
        last_trade_ = price;
        ++seq_;
    }

    void clear() noexcept { bids_.clear(); asks_.clear(); }

    // ---- microstructure signals ---------------------------------------------
    //
    // Order Book Imbalance over the top `obi_depth_` levels:
    //
    //           V_bid − V_ask
    //   OBI = ───────────────── ∈ [−1, +1]
    //           V_bid + V_ask
    //
    // OBI → +1: bid-side pressure dominates (buyers stacking, downside wick
    // likely to revert). OBI → −1: ask wall / seller pressure. A mean-
    // reversion long requires BOTH a stretched z-score (price dislocated
    // below VWAP) AND positive OBI (passive buyers actually absorbing the
    // flow) — z alone catches falling knives.
    double obi() const noexcept {
        const double vb = bids_.volume(obi_depth_);
        const double va = asks_.volume(obi_depth_);
        const double denom = vb + va;
        return denom > 0.0 ? (vb - va) / denom : 0.0;
    }

    double best_bid() const noexcept { return bids_.empty() ? 0.0 : from_ticks(bids_.best().price_ticks); }
    double best_ask() const noexcept { return asks_.empty() ? 0.0 : from_ticks(asks_.best().price_ticks); }

    double mid() const noexcept {
        return (bids_.empty() || asks_.empty()) ? last_trade_
             : 0.5 * (best_bid() + best_ask());
    }

    // Microprice: mid weighted by opposite-side top-of-book qty. A better
    // short-horizon fair value than mid — it leans toward where the thin
    // side will get lifted.
    double microprice() const noexcept {
        if (bids_.empty() || asks_.empty()) return mid();
        const double qb = bids_.best().qty, qa = asks_.best().qty;
        const double denom = qb + qa;
        if (denom <= 0.0) return mid();
        return (best_bid() * qa + best_ask() * qb) / denom;
    }

    double spread_bps() const noexcept {
        const double m = mid();
        return (m > 0.0 && !bids_.empty() && !asks_.empty())
             ? 1e4 * (best_ask() - best_bid()) / m : 0.0;
    }

    double vwap()   const noexcept { return vwap_.vwap(); }
    double sigma()  const noexcept { return vwap_.sigma(); }
    double vwap_zscore() const noexcept { return vwap_.zscore(mid()); }
    double vwap_band(double k) const noexcept { return vwap_.band(k); }

    uint64_t seq() const noexcept { return seq_; }
    double tick_size() const noexcept { return tick_size_; }

    // Checkpoint plumbing (see checkpoint.hpp).
    const BookSide& bids() const noexcept { return bids_; }
    const BookSide& asks() const noexcept { return asks_; }
    BookSide& bids_mut() noexcept { return bids_; }
    BookSide& asks_mut() noexcept { return asks_; }
    const VwapTracker& vwap_tracker() const noexcept { return vwap_; }
    VwapTracker& vwap_tracker_mut() noexcept { return vwap_; }
    void set_seq(uint64_t s) noexcept { seq_ = s; }
    void set_last_trade(double p) noexcept { last_trade_ = p; }
    double last_trade() const noexcept { return last_trade_; }

private:
    int64_t to_ticks(double price) const noexcept {
        // llround, not cast: truncation would map 60000.00 sent as
        // 59999.999999 to the wrong tick.
        return static_cast<int64_t>(std::llround(price / tick_size_));
    }
    double from_ticks(int64_t t) const noexcept { return t * tick_size_; }

    double      tick_size_;
    size_t      obi_depth_;
    BookSide    bids_;
    BookSide    asks_;
    VwapTracker vwap_;
    double      last_trade_ = 0.0;
    uint64_t    seq_ = 0;   // monotonic update counter (checkpoint ordering)
};

}  // namespace microcore
