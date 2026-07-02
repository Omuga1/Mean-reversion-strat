"""StrategyEngine — per-instrument asyncio loop tying the layers together.

Data path (hot, C++):
    venue ws frame ──batch──▶ OrderBook.apply_deltas_batch (GIL released)
                              └▶ OBI / VWAP σ / z updated in place

Decision path (event-driven, C++ pure function):
    feed adapter fires a per-symbol asyncio.Event after every applied frame;
    the signal loop wakes on it and calls SignalGenerator.evaluate(book).
    Decision latency is therefore one event-loop hop after the frame lands
    in the book (~tens of µs) instead of a polling interval.

Control path (slow, Python, ~1 Hz):
    RiskManager.halted ┐
    NightMarketRegime  ┴──▶ SignalGenerator.set_regime(...)   (C++ gate)
    plus mark-to-market equity into the drawdown breaker and 5-minute
    BTC/SOL leader samples into the correlation filter.

Execution path (asyncio, ms-scale):
    Signal.action ──▶ ExchangeRouter.submit_order / cancel_all
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass

from .regime import NightMarketRegime
from .risk import RiskManager
from .router import ExchangeRouter, OrderRequest, OrderSide, OrderType

log = logging.getLogger(__name__)

CHECKPOINT_EVERY_S = 5.0     # mmap snapshot cadence (≈128 KiB memcpy)
CONTROL_LOOP_S = 1.0         # regime/risk refresh cadence
SIGNAL_FALLBACK_S = 0.25     # signal-loop wake even if the feed goes quiet
LEADER_SAMPLE_S = 300.0      # 5-minute closes for the correlation filter
DEEP_BID_REFRESH_S = 30.0    # min age before re-quoting the night deep bid
DEEP_BID_REPRICE_SIGMA = 0.1  # ...unless fair value moved this many σ


@dataclass
class Instrument:
    symbol: str
    tick_size: float | None   # None → resolved from the venue at startup
    router: ExchangeRouter


class StrategyEngine:
    def __init__(self, instruments: list[Instrument], risk: RiskManager,
                 ckpt_dir: str = "./state",
                 enable_night_quoting: bool = False,
                 signal_overrides: dict | None = None):
        import microcore

        self._mc = microcore
        self._risk = risk
        self._night = NightMarketRegime()
        self._instruments = instruments
        self._ckpt_dir = ckpt_dir
        # SignalConfig field overrides (e.g. {"entry_z": 2.0}) applied to
        # every instrument AT BUILD TIME. This is the supported way to
        # retune the strategy: instruments with tick_size=None are built
        # inside run(), after user code has already executed, so mutating
        # engine.configs from the outside would silently miss them.
        self._signal_overrides = dict(signal_overrides or {})
        # Night deep-bid quoting is OFF unless explicitly enabled: a filled
        # passive bid is only visible through the authenticated fills
        # stream, which isn't wired yet — an untracked overnight fill is an
        # unacceptable failure mode with real capital. Everything else about
        # the night regime (cancel aggressive orders, exit-only DEX) is
        # unaffected.
        self._enable_night_quoting = enable_night_quoting
        self._books: dict[str, "microcore.OrderBook"] = {}
        self._signals: dict[str, "microcore.SignalGenerator"] = {}
        self._configs: dict[str, "microcore.SignalConfig"] = {}
        self._ckpts: dict[str, "microcore.Checkpointer"] = {}
        self._md_events: dict[str, asyncio.Event] = {}
        self._trail_stops: dict[str, float] = {}
        self._deep_bids: dict[str, tuple[float, float]] = {}  # sym -> (px, ts)
        # P&L accounting (approximate until the fills stream is wired: entry
        # marked at the submitted limit, exits at the touch we crossed).
        self._avg_entry: dict[str, float] = {}
        self._realized_pnl = 0.0
        self.start_ts = time.time()

        os.makedirs(ckpt_dir, exist_ok=True)
        for ins in instruments:
            if ins.tick_size is not None:
                self._init_instrument(ins, ins.tick_size)
            # tick_size None: built in run() once the venue tells us the
            # real quote increment — guessing (e.g. 0.01 for DOGE at $0.18)
            # quantizes the whole book onto wrong prices.

    def _init_instrument(self, ins: Instrument, tick_size: float) -> None:
        microcore = self._mc
        book = microcore.OrderBook(tick_size, 20)
        sig = microcore.SignalGenerator(microcore.SignalConfig())
        for field, value in self._signal_overrides.items():
            if not hasattr(sig.config, field):
                raise AttributeError(f"unknown SignalConfig field: {field}")
            setattr(sig.config, field, value)
        ckpt = microcore.Checkpointer(
            os.path.join(self._ckpt_dir,
                         f"{ins.symbol.replace('/', '_')}.ckpt"))
        # Crash recovery: restore book/VWAP/position/regime if a valid
        # snapshot exists; otherwise start cold and let the feed's
        # snapshot rebuild the book.
        if ckpt.load(book, sig):
            log.info("%s: recovered checkpoint (seq=%d, pos=%.6f)",
                     ins.symbol, book.seq, sig.position)
        self._books[ins.symbol] = book
        self._signals[ins.symbol] = sig
        # Store the generator's LIVE config (a reference to its internal
        # cfg_), not a disconnected copy — retuning engine.configs[...]
        # writes straight through to what evaluate() reads.
        self._configs[ins.symbol] = sig.config
        self._ckpts[ins.symbol] = ckpt
        self._md_events[ins.symbol] = asyncio.Event()

    # ---- read-only surface for the dashboard -------------------------------
    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

    @property
    def books(self) -> dict:
        return self._books

    @property
    def signals(self) -> dict:
        return self._signals

    @property
    def configs(self) -> dict:
        return self._configs

    @property
    def risk(self) -> RiskManager:
        return self._risk

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def unrealized_pnl(self) -> float:
        pnl = 0.0
        for sym, entry in self._avg_entry.items():
            pos = self._signals[sym].position
            mid = self._books[sym].mid
            if pos > 0 and mid > 0:
                pnl += pos * (mid - entry)
        return pnl

    @property
    def total_pnl(self) -> float:
        return self._realized_pnl + self.unrealized_pnl()

    def gross_exposure(self) -> float:
        """Total open notional across all instruments (marked at mid,
        falling back to entry price when the book is momentarily empty)."""
        gross = 0.0
        for sym, sig in self._signals.items():
            pos = sig.position
            if pos > 0:
                mid = self._books[sym].mid
                px = mid if mid > 0 else self._avg_entry.get(sym, 0.0)
                gross += pos * px
        return gross

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        # GC tuning: after startup, everything long-lived (books, routers,
        # loops) is frozen out of collection scans, and thresholds are
        # raised so the young-gen sweep runs rarely — the steady-state loop
        # allocates only small short-lived objects, so this trims periodic
        # multi-ms GC pauses out of the tick-to-decision path.
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 50, 50)

        routers = {ins.router for ins in self._instruments}
        for r in routers:
            await r.connect()
        # Resolve venue-authoritative tick sizes for instruments that were
        # constructed with tick_size=None (needs a live session, hence here
        # and not in __init__).
        for ins in self._instruments:
            if ins.symbol not in self._books:
                tick = await ins.router.get_tick_size(ins.symbol)
                ins.tick_size = tick
                self._init_instrument(ins, tick)
                log.info("%s: tick size %s (venue-resolved)",
                         ins.symbol, tick)
        tasks = []
        for ins in self._instruments:
            event = self._md_events[ins.symbol]
            tasks.append(asyncio.create_task(
                ins.router.stream_market_data(
                    ins.symbol, self._books[ins.symbol],
                    on_update=event.set),
                name=f"md:{ins.symbol}"))
            tasks.append(asyncio.create_task(
                self._signal_loop(ins), name=f"sig:{ins.symbol}"))
            tasks.append(asyncio.create_task(
                self._checkpoint_loop(ins), name=f"ckpt:{ins.symbol}"))
        tasks.append(asyncio.create_task(self._control_loop(), name="control"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            for r in routers:
                await r.close()

    # -------------------------------------------------------------- control
    async def _control_loop(self) -> None:
        """Slow loop: risk state + wall clock → C++ regime gate, plus the
        two feeds the risk layer needs to actually do its job:

          * mark-to-market equity → daily drawdown breaker
          * 5-minute BTC/SOL mids → cross-correlation breakdown filter

        Enforcement of the regime lives inside C++ on the signal path, so
        between control ticks there is zero window where a stale Python
        flag can leak an order."""
        last_leader = 0.0
        while True:
            now = time.time()
            if now - last_leader >= LEADER_SAMPLE_S:
                last_leader = now
                for ins in self._instruments:
                    base = ins.symbol.split("-")[0].upper()
                    if base in ("BTC", "SOL"):
                        mid = self._books[ins.symbol].mid
                        if mid > 0:
                            self._risk.on_leader_price(base, mid)

            self._risk.on_equity(self._risk.starting_equity + self.total_pnl)

            halted = self._risk.halted
            for ins in self._instruments:
                regime = self._night.resolve(ins.router.kind, halted)
                self._signals[ins.symbol].set_regime(regime)
            await asyncio.sleep(CONTROL_LOOP_S)

    # -------------------------------------------------------------- signals
    async def _signal_loop(self, ins: Instrument) -> None:
        book = self._books[ins.symbol]
        sig = self._signals[ins.symbol]
        event = self._md_events[ins.symbol]
        Action = self._mc.Action

        while True:
            # Event-driven: the feed adapter sets the event the moment a
            # frame has been applied to the C++ book, so we evaluate on
            # fresh state immediately instead of polling. The timeout is a
            # heartbeat so regime transitions still flush posture when the
            # market goes quiet. No seq-dedup is needed: every action below
            # is individually idempotent (position guards, deep-bid
            # throttle, stop ratchet).
            try:
                await asyncio.wait_for(event.wait(), timeout=SIGNAL_FALLBACK_S)
            except asyncio.TimeoutError:
                pass
            event.clear()
            s = sig.evaluate(book)
            if s.action == Action.NONE:
                continue

            if s.action == Action.CANCEL_ALL_BIDS:
                self._deep_bids.pop(ins.symbol, None)
                await ins.router.cancel_all(ins.symbol)

            elif s.action == Action.ENTER_LONG:
                # A z-score can be valid (it's built from the trade tape)
                # while the L2 book is momentarily one-sided or empty — right
                # after a resync/clear, or before the first snapshot repopu-
                # lates it. Without a real ask to cross and a real mid to
                # size against, there is nothing to buy: skip rather than
                # divide by zero or post a limit at price 0.
                px = book.best_ask
                if px <= 0 or s.mid <= 0:
                    continue
                # Fee-awareness gate: the trade's expected gross capture is
                # the reversion distance mid → VWAP (the exit target). If
                # that can't clear the round-trip fees plus a minimum profit,
                # the entry is a statistically guaranteed bleed no matter how
                # stretched the z-score is — refuse it. This is what makes a
                # small account survive Coinbase's retail fee tier.
                edge_bps = (book.vwap - s.mid) / s.mid * 1e4
                if edge_bps < self._risk.cfg.min_edge_bps:
                    log.debug("%s entry vetoed: edge %.0fbps < %.0fbps fees",
                              ins.symbol, edge_bps, self._risk.cfg.min_edge_bps)
                    continue
                # σ_vwap / vwap as the instrument's dimensionless vol proxy
                # for inverse-vol sizing; sized inside portfolio-level
                # gross-exposure headroom.
                vol_proxy = book.sigma / book.vwap if book.vwap > 0 else 0.0
                notional = self._risk.position_notional(
                    vol_proxy, gross_exposure=self.gross_exposure())
                if notional <= 0:
                    continue
                qty = notional / s.mid
                res = await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    qty=qty,
                    # Marketable limit at the ask, never a naked market
                    # order: caps slippage to the touch even on a CLOB.
                    limit_price=px,
                    use_mev_protection=True))
                if res.ok:
                    sig.position = sig.position + qty
                    self._avg_entry[ins.symbol] = px
                    log.info("%s ENTER_LONG qty=%.6f @ %.2f z=%.2f obi=%.2f",
                             ins.symbol, qty, px, s.z, s.obi)
                else:
                    log.warning("%s entry rejected: %s", ins.symbol, res.error)

            elif s.action == Action.EXIT_LONG:
                qty = sig.position
                if qty <= 0:
                    continue
                px = book.best_bid
                if px <= 0:
                    # No bid to hit right now; leave the position and retry
                    # on the next tick rather than sell into a price-0 limit.
                    continue
                res = await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.SELL,
                    order_type=OrderType.LIMIT, qty=qty,
                    limit_price=px,
                    use_mev_protection=True))
                if res.ok:
                    sig.position = 0.0
                    self._trail_stops.pop(ins.symbol, None)
                    entry = self._avg_entry.pop(ins.symbol, px)
                    self._realized_pnl += qty * (px - entry)
                    log.info("%s EXIT_LONG qty=%.6f @ %.2f pnl=%+.2f z=%.2f",
                             ins.symbol, qty, px, qty * (px - entry), s.z)
                else:
                    log.warning("%s exit rejected: %s", ins.symbol, res.error)

            elif s.action == Action.PLACE_DEEP_BID:
                # Night CEX posture: keep exactly one deep resting bid.
                # Gated: without the authenticated fills stream, a passive
                # bid that fills overnight becomes an untracked position.
                if not self._enable_night_quoting:
                    continue
                # Throttled cancel-replace — re-quote only when the quote is
                # stale or fair value has moved materially; without this the
                # loop would churn cancel/replace on every book tick, which
                # is both venue-abusive and gives up queue priority for
                # nothing.
                if s.limit_price <= 0:
                    continue
                prev = self._deep_bids.get(ins.symbol)
                if prev is not None:
                    px_old, ts_old = prev
                    moved = abs(s.limit_price - px_old)
                    if (time.time() - ts_old < DEEP_BID_REFRESH_S
                            and moved < DEEP_BID_REPRICE_SIGMA * book.sigma):
                        continue
                vol_proxy = book.sigma / book.vwap if book.vwap > 0 else 0.0
                notional = self._risk.position_notional(
                    vol_proxy, gross_exposure=self.gross_exposure())
                if notional <= 0:
                    continue
                await ins.router.cancel_all(ins.symbol)
                res = await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.BUY,
                    order_type=OrderType.LIMIT_POST_ONLY,
                    qty=notional / s.limit_price,
                    limit_price=s.limit_price))
                if res.ok:
                    self._deep_bids[ins.symbol] = (s.limit_price, time.time())
                    log.info("%s deep bid %.2f (VWAP-%.1fσ)", ins.symbol,
                             s.limit_price, self._configs[ins.symbol].night_k)

            elif s.action == Action.TRAIL_STOP:
                # Ratchet: the stop only ever moves UP. C++ proposes
                # mid − σ; we keep the max of all proposals and exit when
                # mid crosses below it.
                prev = self._trail_stops.get(ins.symbol, 0.0)
                stop = max(prev, s.limit_price)
                self._trail_stops[ins.symbol] = stop
                if s.mid < stop and sig.position > 0:
                    qty = sig.position
                    res = await ins.router.submit_order(OrderRequest(
                        symbol=ins.symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, qty=qty,
                        use_mev_protection=True))
                    if res.ok:
                        sig.position = 0.0
                        self._trail_stops.pop(ins.symbol, None)
                        entry = self._avg_entry.pop(ins.symbol, s.mid)
                        self._realized_pnl += qty * (s.mid - entry)
                        log.info("%s TRAIL_STOP hit at %.8f", ins.symbol, s.mid)

    # ----------------------------------------------------------- checkpoint
    async def _checkpoint_loop(self, ins: Instrument) -> None:
        book = self._books[ins.symbol]
        sig = self._signals[ins.symbol]
        ckpt = self._ckpts[ins.symbol]
        while True:
            await asyncio.sleep(CHECKPOINT_EVERY_S)
            t0 = time.perf_counter_ns()
            ckpt.save(book, sig)
            dt_us = (time.perf_counter_ns() - t0) / 1e3
            if dt_us > 1000:
                log.debug("%s checkpoint took %.0f µs", ins.symbol, dt_us)
