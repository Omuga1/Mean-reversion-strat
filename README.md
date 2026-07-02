# Mean-Reversion Microstructure System

Hybrid C++/Python trading system: microstructure mean reversion routed to
**Coinbase Advanced Trade** (CLOB) and **Pump.fun** (Solana bonding curves via
low-level RPC + Jito MEV bundles).

---

## 1. Two venues, two different physics

The system's central architectural fact is that its venues have *incompatible
microstructure*, and the abstraction layer must normalize them without
pretending they are the same:

| | Coinbase (CLOB) | Pump.fun (bonding curve) |
|---|---|---|
| Liquidity | Resting limit orders, price-time priority | Curve invariant `k = R_sol·R_tok`; no resting orders |
| Latency floor | Feed ~5–30 ms; order entry ~20–80 ms REST | Slot-quantized: state advances every ~400 ms |
| Adversary | Queue competition, spoofing | **Mempool is adversarial**: sandwich/front-run bots audition every visible tx |
| "Cancel" | First-class, safety-critical | Does not exist (nothing rests) |
| Order book | Native L2 delta stream | **Synthesized**: curve projected onto discrete levels + decayed taker-flow skew |
| Night regime | Passive deep bids at VWAP − 3–4σ (be the liquidity) | Exit-only: night wicks are dev dumps, not anomalies |

On Coinbase, microsecond-scale local processing buys queue-relevant reaction
time on the *feed* side. On Solana, raw speed buys nothing within a slot —
**ordering is auctioned**, so execution quality comes from Jito bundles
(private submission, atomic execution, tip-priced position) rather than from
shaving microseconds. The C++ core therefore optimizes the universal part
(book state + signal math), while venue adapters own the venue-specific
notion of "getting a good fill."

## 2. Layer map

```
                 ┌─────────────────────────────────────────────┐
   C++ (pybind11)│  microcore.so                               │  ~ns–µs
                 │  OrderBook   : flat-array L2, int64 ticks   │
                 │  VwapTracker : Σv, Σpv, Σp²v → VWAP, σ, z   │
                 │  SignalGen   : OBI+z rules, REGIME GATE     │
                 │  Checkpointer: mmap A/B seqlock snapshots   │
                 └──────┬───────────────▲──────────────────────┘
                 batch deltas       regime enum, position
                 (GIL released)         │
                 ┌──────▼───────────────┴──────────────────────┐
   Python asyncio│  StrategyEngine  (engine.py)                │  ~ms
                 │  NightMarketRegime │ RiskManager │ Toxicity │
                 │  ExchangeRouter ABC (router.py)             │
                 │   ├── CoinbaseCEX   ws L2 + REST/JWT        │
                 │   └── PumpFunDEX    RPC/accountSubscribe    │
                 │                     + Jito sendBundle       │
                 └─────────────────────────────────────────────┘
```

**Division of labor.** C++ owns everything decided per-tick (book
maintenance, OBI, VWAP σ-bands, regime-gated signal evaluation) — pure,
allocation-free, deterministic. Python owns everything whose latency floor
is the network RTT anyway (transport, signing, retries, venue quirks) —
asyncio is a perfectly good scheduler at that scale. Websocket frames cross
the boundary **once per frame** via `apply_deltas_batch`, which releases the
GIL, so a liquidation-cascade burst is processed by C++ while the event loop
keeps servicing sockets.

**The regime gate lives in C++**, not Python: `NightMarketRegime` (Python,
1 Hz) only decides *which* regime is active; enforcement happens inside
`SignalGenerator::evaluate` on the same code path that produces signals, so
a stale Python-side flag can never leak an order during a halt. Any regime
transition emits a one-shot `CANCEL_ALL_BIDS` posture flush.

## 3. Signal model (C++, `signal_generator.hpp`)

- Dislocation: `z = (mid − VWAP) / σ_vwap` from volume-weighted running sums
  (periodically re-anchored to kill catastrophic cancellation on long
  sessions — see `VwapTracker::fold_anchor`).
- Confirmation: `OBI = (V_bid − V_ask)/(V_bid + V_ask)` over the top-20
  levels. **Entry requires both** `z ≤ −2.5` *and* `OBI ≥ +0.15` — the
  imbalance term is the anti-falling-knife condition: dislocation without
  passive absorption is a knife, not an anomaly.
- Exit: reversion to VWAP (`z ≥ 0`) or support collapse (`OBI ≤ −0.30`).
- Regimes: `NORMAL`, `CEX_NIGHT_PASSIVE` (deep post-only bid at
  `VWAP − 3.5σ`), `DEX_NIGHT_EXIT_ONLY` (trail/exit only), `HALTED`.

## 4. Memory management

- **Zero heap on the hot path.** Both book sides are fixed-capacity
  in-object arrays (`Level levels_[4096]`, 16-byte POD); deltas are
  binary-search + `memmove`. No allocator, no fragmentation, no leaks by
  construction; worst-case memory is a compile-time constant (~128 KiB per
  instrument).
- Prices are `int64` ticks (`llround(price / tick_size)`) — float keys in a
  book create phantom levels.
- Book-full policy: evict the level farthest from the touch, preserving
  top-of-book fidelity where signals are computed.

## 5. State recovery

Double-buffered ("A/B slot") memory-mapped checkpoints with a seqlock-on-disk
commit protocol: payload → trailer-seq → header-seq+CRC32, so a crash mid-
write leaves at worst one torn slot and restore picks the newest *valid*
slot. C++ side (`checkpoint.hpp`) snapshots book/VWAP/position/regime
(~single-digit µs memcpy); Python side (`state.py`) persists portfolio
equity, high-water mark and the **latched breaker flag** — recovering the
book but forgetting a tripped circuit breaker would be the worst possible
restart. Feed adapters treat any sequence gap as book death: clear and
resync from snapshot rather than trade a desynchronized book.

## 6. Risk layer (`risk.py`)

1. **Daily drawdown breaker** — trips at X% (default 3%) below the intraday
   high-water mark and **latches until the UTC day roll** (an un-latching
   breaker is a Martingale with extra steps).
2. **Cross-correlation filter** — halts all buying when BTC/SOL 5-minute
   returns show a *volatile, coherent, downward* window:
   `trendiness = |Σr|/Σ|r| > 0.75` and window vol above floor and `Σr < 0`.
   Chop never halts — chop is the product.
3. **Inverse-vol sizing** — `notional = base · min(1, σ_target/σ_realized)`;
   doubling vol halves size, keeping P&L variance roughly regime-invariant.

## 7. Pump.fun toxicity screen (`toxicity.py`)

Runs on the order path (cached, 5-min TTL) before any DEX buy:
- **Bundled-launch detection** — distinct buyers in the creation slots /
  shared Jito bundle fingerprints; sniped floats have no organic holders to
  revert to.
- **Dev-wallet tracing** — 2-hop BFS of the creator's funding graph ∩ token
  holder list; dev-cluster supply > 15% ⇒ untradeable.

All DEX entries are wrapped in **Jito bundles** with the slippage rail
encoded on-chain (`min_out`), so a breached guard reverts atomically — no
fill instead of a bad fill — and the order is never exposed to the public
mempool where it could be sandwiched.

## 8. Terminal dashboard & running

`python/trade_example.py` is the runnable entry point: multi-symbol Coinbase
trading with a zero-dependency ANSI dashboard (`meanrev/ui.py`) that redraws
in place — risk strip (equity, P&L, drawdown vs breaker, dead-zone
countdown), one row per instrument (mid, VWAP, σ, z, OBI, spread, feed
msg/s, position, regime), an **entry-proximity meter** showing how far price
has travelled toward the −entry_z trigger and whether OBI confirms, and the
recent event log. Logging goes to `meanrev.log`; stdout belongs to the
renderer.

**"It hasn't traded" is usually correct behavior.** An entry needs a −2.5σ
dislocation *and* OBI ≥ +0.15 simultaneously — an anomaly hunter fires a few
times a week per major pair, not per hour. The proximity meter exists so you
can see it stalking setups instead of wondering if it's dead.

### Performance notes

- The signal loop is **event-driven**: feed adapters fire a per-symbol
  `asyncio.Event` after each applied frame, so evaluation happens one event-
  loop hop after the book updates (a 250 ms heartbeat remains only as a
  regime-flush fallback).
- Optional accelerators (`pip install uvloop orjson`): libuv event loop and
  ~5-10× faster websocket frame decode. Auto-detected, never required.
- GC is tuned at engine start (`gc.freeze()` + raised thresholds) to keep
  multi-ms collector pauses off the tick-to-decision path.
- **WSL users**: run from the Linux filesystem (`~/...`), not `/mnt/c/...`.
  The 9p bridge to the Windows drive adds milliseconds to every file
  operation — checkpoint writes and module loads included.

## 9. Building & testing

```bash
# C++ core tests (no Python needed)
g++ -std=c++17 -O2 -Wall -Wextra -Icpp/include cpp/tests/test_core.cpp -o test_core && ./test_core

# Python extension (or use cmake -S cpp -B build)
pip install pybind11
g++ -std=c++17 -O3 -march=native -shared -fPIC \
    $(python3 -m pybind11 --includes) -Icpp/include cpp/src/bindings.cpp \
    -o python/microcore$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Full system tests
cd python && python3 -m pytest tests/ -q
```

## 10. Layout

```
cpp/include/order_book.hpp       L2 book, OBI, VWAP σ-bands (header-only core)
cpp/include/signal_generator.hpp regime-gated mean-reversion rules
cpp/include/checkpoint.hpp       mmap A/B seqlock checkpointing
cpp/src/bindings.cpp             pybind11 bridge (GIL strategy documented)
cpp/tests/test_core.cpp          dependency-free C++ unit tests
python/meanrev/router.py         ExchangeRouter ABC + order types
python/meanrev/exchanges/        CoinbaseCEX, PumpFunDEX adapters
python/meanrev/regime.py         NightMarketRegime (05:00–11:00 UTC dead zone)
python/meanrev/risk.py           breaker, correlation halt, inverse-vol sizing
python/meanrev/toxicity.py       bundled-launch + dev-cluster screens
python/meanrev/state.py          portfolio mmap checkpoint
python/meanrev/engine.py         asyncio orchestration
python/tests/test_system.py      bindings + risk + regime + e2e engine tests
```

Venue ABI seams (pump.fun instruction encoding, curve account layout, PDA
derivation, forensic RPC sweeps) are deliberately isolated behind
`NotImplementedError` stubs with specified contracts — they are exchange
plumbing, not strategy, and they change when the venue redeploys.

> **Not financial advice. Trading crypto — especially meme coins — can lose
> your entire stake. Test against sandboxes/devnet before funding keys.**
