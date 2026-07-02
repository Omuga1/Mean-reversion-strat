"""System tests: C++ bindings, regime mapping, risk layer, checkpoints,
and an end-to-end engine loop against a mock venue.

Requires the compiled `microcore` extension on sys.path (see README build
steps). Run from python/:  python3 -m pytest tests/ -q
"""
from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import microcore as mc
import pytest

from meanrev.regime import NightMarketRegime, in_dead_zone
from meanrev.risk import RiskConfig, RiskManager
from meanrev.router import (ExchangeRouter, OrderRequest, OrderResult,
                            VenueKind)
from meanrev.state import PortfolioCheckpoint


# ---------------------------------------------------------------- C++ core
def test_obi_and_vwap_via_bindings():
    ob = mc.OrderBook(0.01, 2)
    ob.apply_deltas_batch([0, 0, 1, 1], [100.0, 99.9, 100.1, 100.2],
                          [6.0, 2.0, 1.0, 1.0])
    assert ob.obi == pytest.approx((8 - 2) / (8 + 2))
    ob.on_trades_batch([100.0, 102.0, 98.0], [1.0, 1.0, 2.0])
    assert ob.vwap == pytest.approx(99.5)
    assert ob.sigma == pytest.approx(2.75 ** 0.5)
    assert ob.vwap_band(-3.0) == pytest.approx(99.5 - 3 * 2.75 ** 0.5)


def test_signal_regime_gate():
    ob = mc.OrderBook(0.01, 20)
    for _ in range(100):
        ob.on_trades_batch([99.0, 101.0], [1.0, 1.0])   # VWAP=100, σ=1
    ob.apply_deltas_batch([0, 1], [96.99, 97.01], [10.0, 1.0])  # −3σ, OBI>0

    sg = mc.SignalGenerator(mc.SignalConfig())
    assert sg.evaluate(ob).action == mc.Action.ENTER_LONG

    # Dead-zone DEX regime suppresses the identical setup.
    sg.set_regime(mc.Regime.DEX_NIGHT_EXIT_ONLY)
    assert sg.evaluate(ob).action == mc.Action.CANCEL_ALL_BIDS  # posture flush
    assert sg.evaluate(ob).action == mc.Action.NONE


def test_cpp_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "book.ckpt")
    ob = mc.OrderBook(0.01, 20)
    ob.on_trades_batch([100.0, 101.0], [1.0, 3.0])
    ob.apply_delta(mc.Side.BID, 100.0, 2.5)
    sg = mc.SignalGenerator(mc.SignalConfig())
    sg.position = 1.5
    ck = mc.Checkpointer(path)
    assert ck.ok and ck.save(ob, sg)

    ob2, sg2 = mc.OrderBook(0.01, 20), mc.SignalGenerator(mc.SignalConfig())
    assert mc.Checkpointer(path).load(ob2, sg2)
    assert ob2.vwap == pytest.approx(ob.vwap)
    assert ob2.best_bid == pytest.approx(100.0)
    assert sg2.position == pytest.approx(1.5)


# ------------------------------------------------------------------ regime
def test_dead_zone_window():
    utc = dt.timezone.utc
    assert in_dead_zone(dt.datetime(2026, 7, 2, 5, 0, tzinfo=utc))      # 12am EST
    assert in_dead_zone(dt.datetime(2026, 7, 2, 10, 59, tzinfo=utc))
    assert not in_dead_zone(dt.datetime(2026, 7, 2, 11, 0, tzinfo=utc))  # 6am EST
    assert not in_dead_zone(dt.datetime(2026, 7, 2, 4, 59, tzinfo=utc))


def test_regime_mapping():
    night = NightMarketRegime()
    at = dt.datetime(2026, 7, 2, 7, 0, tzinfo=dt.timezone.utc)  # dead zone
    day = dt.datetime(2026, 7, 2, 15, 0, tzinfo=dt.timezone.utc)
    assert night.resolve(VenueKind.CLOB, False, at) == mc.Regime.CEX_NIGHT_PASSIVE
    assert night.resolve(VenueKind.BONDING_CURVE, False, at) == mc.Regime.DEX_NIGHT_EXIT_ONLY
    assert night.resolve(VenueKind.CLOB, False, day) == mc.Regime.NORMAL
    # Risk halt dominates the clock.
    assert night.resolve(VenueKind.CLOB, True, day) == mc.Regime.HALTED


# -------------------------------------------------------------------- risk
def test_drawdown_breaker_latches():
    rm = RiskManager(RiskConfig(max_daily_dd_pct=3.0), starting_equity=100_000)
    rm.on_equity(100_000)
    assert not rm.halted
    rm.on_equity(96_900)          # −3.1% from HWM → trip
    assert rm.halted
    rm.on_equity(99_500)          # bounce does NOT un-latch intraday
    assert rm.halted


def test_correlation_halt_directional_only():
    cfg = RiskConfig(corr_window=10, corr_trendiness_max=0.75,
                     corr_vol_floor=0.01)
    rm = RiskManager(cfg, 100_000)
    # Choppy: alternating ±1% — trendiness ≈ 0 → tradeable.
    p = 100.0
    for i in range(12):
        p *= 1.01 if i % 2 == 0 else 1 / 1.01
        rm.on_leader_price("BTC", p)
    assert not rm.halted
    # Coherent −1% per bar breakdown → halt.
    for _ in range(12):
        p *= 0.99
        rm.on_leader_price("BTC", p)
    assert rm.halted


def test_inverse_vol_sizing():
    cfg = RiskConfig(sigma_target_daily=0.02, base_notional=1000.0,
                     min_notional=50.0)
    rm = RiskManager(cfg, 100_000)
    assert rm.position_notional(0.02) == pytest.approx(1000.0)
    assert rm.position_notional(0.04) == pytest.approx(500.0)   # 2× vol → ½ size
    assert rm.position_notional(0.01) == pytest.approx(1000.0)  # capped at base
    assert rm.position_notional(0.0) == 0.0                     # fail-closed


# ------------------------------------------------------------- persistence
def test_portfolio_checkpoint(tmp_path):
    path = str(tmp_path / "portfolio.ckpt")
    ck = PortfolioCheckpoint(path)
    assert ck.load() is None
    ck.save({"equity": 100000.0, "breaker": False})
    ck.save({"equity": 99000.0, "breaker": True})   # exercises A/B slots
    ck.close()
    ck2 = PortfolioCheckpoint(path)
    assert ck2.load() == {"equity": 99000.0, "breaker": True}
    ck2.close()


# ------------------------------------------------------ engine end-to-end
class MockRouter(ExchangeRouter):
    kind = VenueKind.CLOB

    def __init__(self):
        super().__init__("mock")
        self.orders: list[OrderRequest] = []
        self.cancels = 0

    async def connect(self):  pass
    async def close(self):  pass

    async def stream_market_data(self, symbol, book):
        # Session history: VWAP 100, σ 1. Then a −3σ flush with heavy
        # passive bids underneath — the canonical mean-reversion entry.
        for _ in range(100):
            book.on_trades_batch([99.0, 101.0], [1.0, 1.0])
        book.apply_deltas_batch([0, 1], [96.99, 97.01], [10.0, 1.0])
        await asyncio.sleep(3600)

    async def submit_order(self, req: OrderRequest) -> OrderResult:
        self.orders.append(req)
        return OrderResult(ok=True, order_id=f"mock-{len(self.orders)}",
                           filled_qty=req.qty, avg_price=req.limit_price or 0)

    async def cancel_all(self, symbol):
        self.cancels += 1

    async def fills(self):
        while True:
            await asyncio.sleep(3600)
            yield None


def test_engine_enters_on_dislocation(tmp_path):
    from meanrev.engine import Instrument, StrategyEngine

    router = MockRouter()
    risk = RiskManager(RiskConfig(), starting_equity=100_000)
    engine = StrategyEngine([Instrument("TEST-USD", 0.01, router)],
                            ckpt_dir=str(tmp_path), risk=risk)
    # Pin the regime: the test must not depend on the wall clock.
    engine._night.resolve = lambda *a, **k: mc.Regime.NORMAL

    async def run_briefly():
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(1.0)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    asyncio.run(run_briefly())
    assert any(o.side.value == "buy" for o in router.orders), \
        "engine should have entered long on a −3σ dislocation with OBI support"
    # Position accounting flowed back into the C++ signal state.
    assert engine._signals["TEST-USD"].position > 0
