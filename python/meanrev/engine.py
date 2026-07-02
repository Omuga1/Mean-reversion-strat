"""StrategyEngine — per-instrument asyncio loop tying the layers together.

Data path (hot, C++):
    venue ws frame ──batch──▶ OrderBook.apply_deltas_batch (GIL released)
                              └▶ OBI / VWAP σ / z updated in place

Decision path (per book update or 5m close, C++ pure function):
    SignalGenerator.evaluate(book) ──▶ Signal{action, limit_price, z, obi}

Control path (slow, Python, ~1 Hz):
    RiskManager.halted ┐
    NightMarketRegime  ┴──▶ SignalGenerator.set_regime(...)   (C++ gate)

Execution path (asyncio, ms-scale):
    Signal.action ──▶ ExchangeRouter.submit_order / cancel_all
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .regime import NightMarketRegime
from .risk import RiskManager
from .router import ExchangeRouter, OrderRequest, OrderSide, OrderType

log = logging.getLogger(__name__)

CHECKPOINT_EVERY_S = 5.0     # mmap snapshot cadence (≈128 KiB memcpy)
CONTROL_LOOP_S = 1.0         # regime/risk refresh cadence
SIGNAL_LOOP_S = 0.25         # decision cadence (book updates continuously)


@dataclass
class Instrument:
    symbol: str
    tick_size: float
    router: ExchangeRouter


class StrategyEngine:
    def __init__(self, instruments: list[Instrument], risk: RiskManager,
                 ckpt_dir: str = "./state"):
        import microcore

        self._mc = microcore
        self._risk = risk
        self._night = NightMarketRegime()
        self._instruments = instruments
        self._books: dict[str, "microcore.OrderBook"] = {}
        self._signals: dict[str, "microcore.SignalGenerator"] = {}
        self._ckpts: dict[str, "microcore.Checkpointer"] = {}
        self._trail_stops: dict[str, float] = {}

        import os
        os.makedirs(ckpt_dir, exist_ok=True)
        for ins in instruments:
            book = microcore.OrderBook(ins.tick_size, 20)
            sig = microcore.SignalGenerator(microcore.SignalConfig())
            ckpt = microcore.Checkpointer(
                os.path.join(ckpt_dir, f"{ins.symbol.replace('/', '_')}.ckpt"))
            # Crash recovery: restore book/VWAP/position/regime if a valid
            # snapshot exists; otherwise start cold and let the feed's
            # snapshot rebuild the book.
            if ckpt.load(book, sig):
                log.info("%s: recovered checkpoint (seq=%d, pos=%.6f)",
                         ins.symbol, book.seq, sig.position)
            self._books[ins.symbol] = book
            self._signals[ins.symbol] = sig
            self._ckpts[ins.symbol] = ckpt

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        routers = {ins.router for ins in self._instruments}
        for r in routers:
            await r.connect()
        tasks = []
        for ins in self._instruments:
            tasks.append(asyncio.create_task(
                ins.router.stream_market_data(ins.symbol, self._books[ins.symbol]),
                name=f"md:{ins.symbol}"))
            tasks.append(asyncio.create_task(
                self._signal_loop(ins), name=f"sig:{ins.symbol}"))
            tasks.append(asyncio.create_task(
                self._checkpoint_loop(ins), name=f"ckpt:{ins.symbol}"))
        tasks.append(asyncio.create_task(self._control_loop(), name="control"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for r in routers:
                await r.close()

    # -------------------------------------------------------------- control
    async def _control_loop(self) -> None:
        """Slow loop: fold risk state + wall clock into the C++ regime gate.

        Runs at 1 Hz — regimes change on the scale of hours and the halt
        flag on the scale of minutes; what matters is that the *enforcement*
        of the regime is inside C++ on the signal path, so between control
        ticks there is zero window where a stale Python flag can leak an
        order."""
        while True:
            halted = self._risk.halted
            for ins in self._instruments:
                regime = self._night.resolve(ins.router.kind, halted)
                self._signals[ins.symbol].set_regime(regime)
            await asyncio.sleep(CONTROL_LOOP_S)

    # -------------------------------------------------------------- signals
    async def _signal_loop(self, ins: Instrument) -> None:
        book = self._books[ins.symbol]
        sig = self._signals[ins.symbol]
        Action = self._mc.Action
        last_seq = -1

        while True:
            await asyncio.sleep(SIGNAL_LOOP_S)
            if book.seq == last_seq:
                continue  # no new market data; don't re-decide on stale state
            last_seq = book.seq
            s = sig.evaluate(book)

            if s.action == Action.NONE:
                continue

            if s.action == Action.CANCEL_ALL_BIDS:
                await ins.router.cancel_all(ins.symbol)

            elif s.action == Action.ENTER_LONG:
                # σ_vwap / vwap as the instrument's dimensionless vol proxy
                # for inverse-vol sizing.
                vol_proxy = book.sigma / book.vwap if book.vwap > 0 else 0.0
                notional = self._risk.position_notional(vol_proxy)
                if notional <= 0:
                    continue
                qty = notional / s.mid
                res = await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    qty=qty,
                    # Marketable limit at the ask, never a naked market
                    # order: caps slippage to the touch even on a CLOB.
                    limit_price=book.best_ask,
                    use_mev_protection=True))
                if res.ok:
                    sig.position = sig.position + qty
                    log.info("%s ENTER_LONG qty=%.6f z=%.2f obi=%.2f",
                             ins.symbol, qty, s.z, s.obi)

            elif s.action == Action.EXIT_LONG:
                qty = sig.position
                if qty <= 0:
                    continue
                res = await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.SELL,
                    order_type=OrderType.LIMIT, qty=qty,
                    limit_price=book.best_bid,
                    use_mev_protection=True))
                if res.ok:
                    sig.position = 0.0
                    self._trail_stops.pop(ins.symbol, None)
                    log.info("%s EXIT_LONG qty=%.6f z=%.2f", ins.symbol, qty, s.z)

            elif s.action == Action.PLACE_DEEP_BID:
                # Night CEX posture: refresh the deep resting bid. Cancel-
                # replace keeps exactly one working order per instrument.
                await ins.router.cancel_all(ins.symbol)
                vol_proxy = book.sigma / book.vwap if book.vwap > 0 else 0.0
                notional = self._risk.position_notional(vol_proxy)
                if notional <= 0 or s.limit_price <= 0:
                    continue
                await ins.router.submit_order(OrderRequest(
                    symbol=ins.symbol, side=OrderSide.BUY,
                    order_type=OrderType.LIMIT_POST_ONLY,
                    qty=notional / s.limit_price,
                    limit_price=s.limit_price))

            elif s.action == Action.TRAIL_STOP:
                # Ratchet: the stop only ever moves UP. C++ proposes
                # mid − σ; we keep the max of all proposals and exit when
                # mid crosses below it.
                prev = self._trail_stops.get(ins.symbol, 0.0)
                stop = max(prev, s.limit_price)
                self._trail_stops[ins.symbol] = stop
                if s.mid < stop and sig.position > 0:
                    res = await ins.router.submit_order(OrderRequest(
                        symbol=ins.symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, qty=sig.position,
                        use_mev_protection=True))
                    if res.ok:
                        sig.position = 0.0
                        self._trail_stops.pop(ins.symbol, None)
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
