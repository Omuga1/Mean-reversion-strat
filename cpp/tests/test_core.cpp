// =============================================================================
// test_core.cpp — dependency-free unit tests for the C++ core.
// Validates: book delta semantics, OBI arithmetic, VWAP σ-band math,
// regime gating, and mmap checkpoint round-trip (incl. torn-slot rejection).
// =============================================================================
#include "order_book.hpp"
#include "signal_generator.hpp"
#include "checkpoint.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <unistd.h>

using namespace microcore;

static int g_failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                               \
        }                                                               \
    } while (0)

static bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

static void test_book_deltas() {
    OrderBook ob(0.01, 20);
    ob.apply_delta(Side::BID, 100.00, 2.0);
    ob.apply_delta(Side::BID, 99.50, 1.0);
    ob.apply_delta(Side::ASK, 100.10, 3.0);
    ob.apply_delta(Side::ASK, 100.50, 1.0);

    CHECK(approx(ob.best_bid(), 100.00));
    CHECK(approx(ob.best_ask(), 100.10));
    CHECK(approx(ob.mid(), 100.05));

    // Replace semantics: absolute qty, not additive.
    ob.apply_delta(Side::BID, 100.00, 5.0);
    CHECK(approx(ob.bids().best().qty, 5.0));

    // Delete via qty=0 promotes the next level.
    ob.apply_delta(Side::BID, 100.00, 0.0);
    CHECK(approx(ob.best_bid(), 99.50));

    // Float-price robustness: 99.4999999 must land on the 99.50 tick.
    ob.apply_delta(Side::BID, 99.4999999, 7.0);
    CHECK(approx(ob.bids().best().qty, 7.0));
    CHECK(ob.bids().size() == 1);
}

static void test_obi() {
    OrderBook ob(0.01, 2);  // OBI over top 2 levels only
    ob.apply_delta(Side::BID, 100.00, 6.0);
    ob.apply_delta(Side::BID, 99.90, 2.0);
    ob.apply_delta(Side::BID, 99.80, 100.0);   // beyond depth: ignored
    ob.apply_delta(Side::ASK, 100.10, 1.0);
    ob.apply_delta(Side::ASK, 100.20, 1.0);
    // V_bid = 8, V_ask = 2 → OBI = (8−2)/(8+2) = 0.6
    CHECK(approx(ob.obi(), 0.6));
}

static void test_vwap_bands() {
    VwapTracker v;
    // Trades: (100, 1), (102, 1), (98, 2) → S0=4, S1=398 → VWAP=99.5
    v.on_trade(100.0, 1.0);
    v.on_trade(102.0, 1.0);
    v.on_trade(98.0, 2.0);
    CHECK(approx(v.vwap(), 99.5));
    // Var = Σp²v/S0 − VWAP² = (10000+10404+2·9604)/4 − 9900.25 = 2.75
    CHECK(approx(v.sigma(), std::sqrt(2.75)));
    CHECK(approx(v.zscore(99.5 - 2.0 * v.sigma()), -2.0));
    CHECK(approx(v.band(-3.0), 99.5 - 3.0 * std::sqrt(2.75)));
}

static void test_signal_gating() {
    OrderBook ob(0.01, 20);
    SignalConfig cfg;
    cfg.entry_z = 2.0; cfg.obi_min = 0.2;
    SignalGenerator sg(cfg);

    // Build a VWAP distribution around 100 with σ≈1.
    for (int i = 0; i < 100; ++i) {
        ob.on_trade(99.0, 1.0);
        ob.on_trade(101.0, 1.0);
    }
    CHECK(approx(ob.vwap(), 100.0));
    CHECK(approx(ob.sigma(), 1.0));

    // Price dislocated to −3σ with strong bid imbalance → ENTER_LONG.
    ob.apply_delta(Side::BID, 96.99, 10.0);
    ob.apply_delta(Side::ASK, 97.01, 1.0);
    Signal s = sg.evaluate(ob);
    CHECK(s.action == Action::ENTER_LONG);
    CHECK(s.z < -2.0 && s.obi > 0.2);

    // Same dislocation but ask-heavy book (falling knife) → NONE.
    ob.apply_delta(Side::BID, 96.99, 1.0);
    ob.apply_delta(Side::ASK, 97.01, 50.0);
    s = sg.evaluate(ob);
    CHECK(s.action == Action::NONE);

    // Night DEX regime: dip buying suppressed even on perfect setup.
    ob.apply_delta(Side::BID, 96.99, 50.0);
    ob.apply_delta(Side::ASK, 97.01, 1.0);
    sg.set_regime(Regime::DEX_NIGHT_EXIT_ONLY);
    s = sg.evaluate(ob);                       // 1st call: posture flush
    CHECK(s.action == Action::CANCEL_ALL_BIDS);
    s = sg.evaluate(ob);                       // flat + exit-only → NONE
    CHECK(s.action == Action::NONE);

    // ...but existing inventory still gets trailed.
    sg.set_position(10.0);
    s = sg.evaluate(ob);
    CHECK(s.action == Action::TRAIL_STOP);
    CHECK(s.limit_price < ob.mid());

    // Night CEX regime: flat book → deep passive bid at VWAP − kσ.
    sg.set_position(0.0);
    sg.set_regime(Regime::CEX_NIGHT_PASSIVE);
    s = sg.evaluate(ob);
    CHECK(s.action == Action::CANCEL_ALL_BIDS);  // posture flush
    s = sg.evaluate(ob);
    CHECK(s.action == Action::PLACE_DEEP_BID);
    CHECK(approx(s.limit_price, ob.vwap_band(-3.5)));

    // HALTED with inventory → forced exit.
    sg.set_position(5.0);
    sg.set_regime(Regime::HALTED);
    s = sg.evaluate(ob);
    CHECK(s.action == Action::CANCEL_ALL_BIDS);
    s = sg.evaluate(ob);
    CHECK(s.action == Action::EXIT_LONG);
}

static void test_checkpoint_roundtrip() {
    const char* path = "/tmp/microcore_test.ckpt";
    ::unlink(path);

    OrderBook ob(0.01, 20);
    SignalGenerator sg;
    for (int i = 0; i < 50; ++i) ob.on_trade(100.0 + (i % 5), 1.0);
    ob.apply_delta(Side::BID, 100.00, 2.5);
    ob.apply_delta(Side::ASK, 100.10, 1.5);
    sg.set_position(3.25);
    sg.set_regime(Regime::CEX_NIGHT_PASSIVE);

    {
        Checkpointer ck(path);
        CHECK(ck.ok());
        CHECK(ck.save(ob, sg));
        CHECK(ck.save(ob, sg));  // exercise the A/B slot alternation
    }

    OrderBook ob2(0.01, 20);
    SignalGenerator sg2;
    {
        Checkpointer ck(path);
        CHECK(ck.load(ob2, sg2));
    }
    CHECK(approx(ob2.best_bid(), 100.00));
    CHECK(approx(ob2.best_ask(), 100.10));
    CHECK(approx(ob2.vwap(), ob.vwap()));
    CHECK(approx(ob2.sigma(), ob.sigma()));
    CHECK(ob2.seq() == ob.seq());
    CHECK(approx(sg2.position(), 3.25));
    CHECK(sg2.regime() == Regime::CEX_NIGHT_PASSIVE);

    // Fresh file (all zeros) must be rejected, not "restored".
    ::unlink(path);
    OrderBook ob3(0.01, 20);
    SignalGenerator sg3;
    Checkpointer ck(path);
    CHECK(!ck.load(ob3, sg3));
    ::unlink(path);
}

int main() {
    test_book_deltas();
    test_obi();
    test_vwap_bands();
    test_signal_gating();
    test_checkpoint_roundtrip();
    if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
