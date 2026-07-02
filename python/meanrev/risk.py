"""Institutional risk layer: circuit breaker, correlation halt, vol sizing.

Three independent controls, combined conservatively (any veto is a veto):

  1. Hard daily drawdown breaker — equity high-water mark is reset at the
     UTC day roll; if equity falls more than `max_daily_dd_pct` below it,
     the GLOBAL halt latches until the next day roll. Latching matters:
     a breaker that un-trips intraday when price bounces is a Martingale
     with extra steps.

  2. Cross-correlation filter — mean reversion on any single crypto asset
     is implicitly short the market's common factor. When BTC/SOL are in a
     unidirectional volatile breakdown, every "dislocation" in every alt is
     the SAME falling knife. We measure, over a rolling window of 5-minute
     returns on the leaders:

        trendiness  t = |Σ r_i| / Σ |r_i|   ∈ [0, 1]   (directional purity)
        realized vol σ_w = √(Σ r_i²)                     (window vol)

     t → 1 means returns are one-directional (a trend, not chop). Halt
     buying when t > t_max AND σ_w > vol_floor AND Σ r_i < 0 — i.e. a
     *volatile, coherent, downward* move. Chop (t small) is exactly what we
     want to trade, so t alone never halts.

  3. Dynamic inventory sizing — position notional scales inversely with
     realized vol (the crypto-VIX equivalent):

        size = base_notional · min(1, σ_target / σ_realized)

     Constant-vol sizing: doubling vol halves size, so the portfolio's
     daily P&L variance stays approximately flat across regimes rather
     than exploding exactly when tails fatten.
"""
from __future__ import annotations

import collections
import datetime as dt
import logging
import math
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_daily_dd_pct: float = 3.0       # hard global circuit breaker (X%)
    corr_window: int = 36               # 36 × 5m = 3h of leader returns
    corr_trendiness_max: float = 0.75   # directional-purity halt threshold
    corr_vol_floor: float = 0.015       # window vol below this → never halt
    sigma_target_daily: float = 0.02    # 2% target daily vol per position
    base_notional: float = 1_000.0      # position size at σ == σ_target
    min_notional: float = 50.0          # below this, don't bother trading
    # Gross exposure cap: total open notional may never exceed this share
    # of equity. Prevents N independent per-symbol entries from quietly
    # deploying N × base_notional on a correlated market-wide dip.
    max_gross_exposure_pct: float = 50.0
    # Execution economics. A mean-reversion round trip pays fees twice; an
    # entry whose expected capture (mid → VWAP) can't clear
    # 2·fee_bps_per_side + min_profit_bps is a guaranteed bleed and is
    # refused regardless of how pretty the z-score looks. Coinbase retail
    # tiers pay roughly 25-120 bps/side depending on 30-day volume — set
    # this to YOUR tier; the default is deliberately conservative.
    fee_bps_per_side: float = 60.0
    min_profit_bps: float = 10.0

    @property
    def min_edge_bps(self) -> float:
        return 2.0 * self.fee_bps_per_side + self.min_profit_bps


class RiskManager:
    def __init__(self, cfg: RiskConfig, starting_equity: float):
        self.cfg = cfg
        self.starting_equity = starting_equity
        self._equity = starting_equity
        self._hwm = starting_equity
        self._hwm_day = self._utc_day()
        self._breaker_latched = False
        # Rolling 5-minute returns for the market leaders (BTC, SOL).
        self._leader_returns: dict[str, collections.deque[float]] = {
            "BTC": collections.deque(maxlen=cfg.corr_window),
            "SOL": collections.deque(maxlen=cfg.corr_window),
        }
        self._last_price: dict[str, float] = {}

    # ------------------------------------------------------------- equity
    def on_equity(self, equity: float) -> None:
        day = self._utc_day()
        if day != self._hwm_day:
            # Day roll: new high-water mark, breaker resets.
            self._hwm_day = day
            self._hwm = equity
            if self._breaker_latched:
                log.warning("daily breaker RESET at UTC day roll")
            self._breaker_latched = False
        self._equity = equity
        self._hwm = max(self._hwm, equity)
        dd_pct = 100.0 * (self._hwm - equity) / self._hwm if self._hwm > 0 else 0.0
        if dd_pct >= self.cfg.max_daily_dd_pct and not self._breaker_latched:
            self._breaker_latched = True
            log.error("DAILY DRAWDOWN BREAKER: %.2f%% >= %.2f%% — "
                      "halting all entries until UTC day roll",
                      dd_pct, self.cfg.max_daily_dd_pct)

    # -------------------------------------------------------- leader feed
    def on_leader_price(self, symbol: str, price: float) -> None:
        """Call on each 5-minute close of BTC-USD / SOL-USD."""
        last = self._last_price.get(symbol)
        self._last_price[symbol] = price
        if last and last > 0:
            self._leader_returns[symbol].append(math.log(price / last))

    def _breakdown(self) -> bool:
        """Volatile unidirectional downward move in ANY leader → True."""
        for sym, rets in self._leader_returns.items():
            if len(rets) < self.cfg.corr_window // 2:
                continue  # insufficient data: fail-open on this leader only
            gross = sum(abs(r) for r in rets)
            net = sum(rets)
            if gross <= 0:
                continue
            trendiness = abs(net) / gross
            window_vol = math.sqrt(sum(r * r for r in rets))
            if (net < 0 and trendiness > self.cfg.corr_trendiness_max
                    and window_vol > self.cfg.corr_vol_floor):
                log.warning("correlation halt: %s trendiness=%.2f vol=%.3f",
                            sym, trendiness, window_vol)
                return True
        return False

    # ---------------------------------------------------------- decisions
    @property
    def halted(self) -> bool:
        """Consumed by NightMarketRegime.resolve() every engine loop."""
        return self._breaker_latched or self._breakdown()

    # ---- read-only surface for the dashboard ----------------------------
    @property
    def equity(self) -> float:
        return self._equity

    @property
    def breaker_latched(self) -> bool:
        return self._breaker_latched

    @property
    def drawdown_pct(self) -> float:
        if self._hwm <= 0:
            return 0.0
        return 100.0 * (self._hwm - self._equity) / self._hwm

    def position_notional(self, realized_vol_daily: float,
                          gross_exposure: float = 0.0) -> float:
        """Inverse-volatility sizing (see module docstring). `realized_vol
        _daily` should be the instrument's own realized daily vol, e.g.
        σ_vwap annualization-free proxy from the C++ tracker.
        `gross_exposure` is the portfolio's current total open notional;
        the returned size is clipped so gross never exceeds
        max_gross_exposure_pct of equity."""
        if self.halted:
            return 0.0
        if realized_vol_daily <= 0:
            return 0.0  # no vol estimate → no trade (fail-closed)
        scale = min(1.0, self.cfg.sigma_target_daily / realized_vol_daily)
        notional = self.cfg.base_notional * scale
        # Portfolio-level cap: headroom left under the gross exposure limit.
        headroom = (self.cfg.max_gross_exposure_pct / 100.0) * self._equity \
            - gross_exposure
        notional = min(notional, max(headroom, 0.0))
        return notional if notional >= self.cfg.min_notional else 0.0

    @staticmethod
    def _utc_day() -> dt.date:
        return dt.datetime.now(dt.timezone.utc).date()
