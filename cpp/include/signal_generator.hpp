// =============================================================================
// signal_generator.hpp — Microstructure mean-reversion signal engine.
//
// Consumes an OrderBook (OBI + VWAP σ-bands) and emits discrete trading
// intents. All regime gating (night regime, correlation halt, drawdown
// breaker) is applied HERE in the C++ core rather than in Python: the gate
// must sit on the same critical path as the signal so a stale Python-side
// flag can never let an order slip through during a halt.
//
// Signal model
// ------------
//   z      = (mid − VWAP) / σ_vwap        (dislocation from fair value)
//   OBI    = (V_bid − V_ask)/(V_bid + V_ask)  over top-N levels
//
//   LONG entry   : z ≤ −entry_z  AND  OBI ≥ +obi_min
//                  (price stretched below VWAP AND passive bids absorbing —
//                   the imbalance term is the anti-falling-knife condition)
//   EXIT / short : z ≥ +exit_z  (reversion target reached)  OR
//                  OBI ≤ −obi_flip while in position (support evaporated)
//
// Regime overrides (NightMarketRegime, "Crypto Dead Zone" 05:00–11:00 UTC):
//   CEX_NIGHT_PASSIVE : entries only as deep passive quotes at
//                       VWAP − k_night·σ (k_night ∈ [3,4]) — emitted as
//                       Action::PLACE_DEEP_BID with an explicit limit price.
//   DEX_NIGHT_EXIT_ONLY : all buy intents suppressed; only EXIT /
//                       TRAIL_STOP actions pass. Night wicks on Pump.fun
//                       are statistically dev dumps, not anomalies.
//   HALTED            : circuit breaker / correlation filter — flat only.
// =============================================================================
#pragma once

#include "order_book.hpp"
#include <cstdint>

namespace microcore {

enum class Regime : uint8_t {
    NORMAL             = 0,
    CEX_NIGHT_PASSIVE  = 1,  // Coinbase: passive deep-limit posture only
    DEX_NIGHT_EXIT_ONLY= 2,  // Pump.fun: no dip buying, exits/trails only
    HALTED             = 3,  // circuit breaker / correlation halt: flat
};

enum class Action : uint8_t {
    NONE            = 0,
    ENTER_LONG      = 1,  // aggressive/marketable entry at fair dislocation
    EXIT_LONG       = 2,  // reversion complete or support evaporated
    PLACE_DEEP_BID  = 3,  // passive limit at VWAP − k_night·σ (night CEX)
    CANCEL_ALL_BIDS = 4,  // posture change → pull resting orders
    TRAIL_STOP      = 5,  // tighten stop on existing inventory (night DEX)
};

struct SignalConfig {
    double entry_z    = 2.5;   // enter when mid ≤ VWAP − 2.5σ ...
    double exit_z     = 0.0;   // ... exit at reversion to VWAP
    double obi_min    = 0.15;  // required bid-side imbalance to enter
    double obi_flip   = -0.30; // in-position support-collapse threshold
    double night_k    = 3.5;   // deep-bid depth in σ during night regime
    double min_sigma  = 0.0;   // ignore signals until σ is meaningful
    double min_volume = 0.0;   // ignore signals until session volume exists
};

struct Signal {
    Action  action;
    double  limit_price;   // populated for PLACE_DEEP_BID / TRAIL_STOP
    double  z;             // diagnostics: dislocation at decision time
    double  obi;
    double  mid;
    uint64_t seq;          // book sequence the decision was computed on
};

class SignalGenerator {
public:
    explicit SignalGenerator(SignalConfig cfg = SignalConfig{}) : cfg_(cfg) {}

    void set_regime(Regime r) noexcept {
        // Posture transition into a passive/halted regime must atomically
        // invalidate any working aggressive orders; the router treats the
        // CANCEL_ALL_BIDS emitted on the next evaluate() as mandatory.
        if (r != regime_) posture_dirty_ = true;
        regime_ = r;
    }
    Regime regime() const noexcept { return regime_; }

    void set_position(double qty) noexcept { position_ = qty; }
    double position() const noexcept { return position_; }

    // Pure function of (book state, regime, position) — no allocation,
    // no I/O, deterministic. Called on every book update or on a 5-min
    // candle close; cost is a handful of FLOPs on cached data.
    Signal evaluate(const OrderBook& book) noexcept {
        Signal s{Action::NONE, 0.0, 0.0, 0.0, 0.0, book.seq()};
        s.mid = book.mid();
        s.obi = book.obi();

        const double sigma = book.sigma();
        const bool stats_ready =
            sigma > cfg_.min_sigma &&
            book.vwap_tracker().total_volume() > cfg_.min_volume;
        s.z = stats_ready ? book.vwap_zscore() : 0.0;

        // One-shot posture flush after any regime transition.
        if (posture_dirty_) {
            posture_dirty_ = false;
            s.action = Action::CANCEL_ALL_BIDS;
            return s;
        }

        switch (regime_) {
        case Regime::HALTED:
            // Correlation breakdown / drawdown breaker: exit inventory,
            // never add. Falling-knife protection is absolute here.
            if (position_ > 0.0) s.action = Action::EXIT_LONG;
            return s;

        case Regime::DEX_NIGHT_EXIT_ONLY:
            // Pump.fun dead zone: a −4σ wick at 07:00 UTC is a dev dump
            // until proven otherwise. Only manage existing inventory.
            if (position_ > 0.0) {
                if (stats_ready && s.z >= cfg_.exit_z) {
                    s.action = Action::EXIT_LONG;
                } else {
                    s.action = Action::TRAIL_STOP;
                    // Trail one σ under mid, ratcheted by the router.
                    s.limit_price = s.mid - sigma;
                }
            }
            return s;

        case Regime::CEX_NIGHT_PASSIVE:
            // Coinbase dead zone: spreads widen, top-of-book thins. Do not
            // cross the spread; rest a deep bid at VWAP − k·σ to be the
            // resting liquidity that catches a cascade wick.
            if (position_ > 0.0 && stats_ready && s.z >= cfg_.exit_z) {
                s.action = Action::EXIT_LONG;
            } else if (stats_ready && position_ <= 0.0) {
                s.action = Action::PLACE_DEEP_BID;
                s.limit_price = book.vwap_band(-cfg_.night_k);
            }
            return s;

        case Regime::NORMAL:
        default:
            if (!stats_ready) return s;
            if (position_ <= 0.0) {
                // Entry: dislocation AND order-flow confirmation.
                if (s.z <= -cfg_.entry_z && s.obi >= cfg_.obi_min)
                    s.action = Action::ENTER_LONG;
            } else {
                // Exit: reversion target hit, or the bid support that
                // justified the entry has flipped to an ask wall.
                if (s.z >= cfg_.exit_z || s.obi <= cfg_.obi_flip)
                    s.action = Action::EXIT_LONG;
            }
            return s;
        }
    }

    const SignalConfig& config() const noexcept { return cfg_; }

private:
    SignalConfig cfg_;
    Regime  regime_ = Regime::NORMAL;
    double  position_ = 0.0;
    bool    posture_dirty_ = false;
};

}  // namespace microcore
