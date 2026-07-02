"""Runnable entry point: Coinbase multi-symbol trading with the terminal
dashboard.

    python3 trade_example.py

Expects cdp_api_key.json (the file Coinbase's CDP portal gives you) in the
working directory. Logs go to meanrev.log AND the dashboard's events pane —
stdout itself belongs to the dashboard renderer.

WARNING: this places real orders when a signal fires. All sizing below is
calibrated to ACCOUNT_EQUITY — set it to what is actually in your Coinbase
account, not what you wish were there.
"""
import asyncio
import json
import logging

try:
    # uvloop: drop-in libuv event loop, ~2-4x lower scheduling overhead
    # than the stdlib loop. Optional: pip install uvloop
    import uvloop
    uvloop.install()
except ImportError:
    pass

from meanrev.engine import Instrument, StrategyEngine
from meanrev.exchanges import CoinbaseCEX
from meanrev.risk import RiskConfig, RiskManager
from meanrev.ui import TerminalDashboard

# tick_size is resolved from the venue automatically (Instrument(..., None)),
# so any Coinbase Advanced Trade product id can be listed here.
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]

# ---- account sizing --------------------------------------------------------
# Everything scales off this number. With 3,000:
#   base_notional 250   → ~8% of equity per position at normal volatility
#   max_gross_exposure 50% → at most ~1,500 deployed across all symbols
#   max_daily_dd 3%     → a 90 down day trips the breaker until UTC midnight
ACCOUNT_EQUITY = 3_000.0

# ---- entry tuning ----------------------------------------------------------
# ENTRY_Z: how many σ below VWAP price must fall to trigger a long. Lower =
#   more trades, but each is weaker evidence of a real (mean-reverting)
#   anomaly rather than ordinary noise. Default 2.5; 2.0 is a moderate
#   loosening (~2-3x more entries); below ~1.8 you're mostly buying dips.
# OBI_MIN: required bid-side order-book imbalance to confirm the entry (the
#   anti-falling-knife check). Lower accepts weaker passive support.
ENTRY_Z = 2.0
OBI_MIN = 0.10

# ---- fee economics ---------------------------------------------------------
# Round-trip fees are the silent killer of small-account mean reversion: the
# engine refuses entries whose expected capture (mid → VWAP) is below
# 2*FEE_BPS_PER_SIDE + 10bps. Check your actual tier under Coinbase →
# Advanced Trade fees and set it here; overstating fees means fewer, better
# trades — understating them means bleeding capital on winners.
FEE_BPS_PER_SIDE = 60.0   # 0.60%/side ≈ Coinbase retail taker, lowest tier

logging.basicConfig(
    filename="meanrev.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def main():
    with open("cdp_api_key.json") as f:
        creds = json.load(f)

    exchange = CoinbaseCEX(creds["name"], creds["privateKey"])
    engine = StrategyEngine(
        [Instrument(sym, None, exchange) for sym in SYMBOLS],
        risk=RiskManager(
            RiskConfig(
                max_daily_dd_pct=3.0,
                base_notional=ACCOUNT_EQUITY / 12,   # ≈250 at 3k
                min_notional=25.0,
                max_gross_exposure_pct=50.0,
                fee_bps_per_side=FEE_BPS_PER_SIDE,
            ),
            starting_equity=ACCOUNT_EQUITY,
        ),
        # Night deep-bid quoting stays off until the authenticated fills
        # stream is wired: a passive bid that fills overnight would be an
        # untracked position.
        enable_night_quoting=False,
        # Applied at instrument build time — the supported way to retune
        # (instruments resolve their tick size inside run(), so mutating
        # engine.configs here would run too early and miss them).
        signal_overrides={"entry_z": ENTRY_Z, "obi_min": OBI_MIN},
    )
    dashboard = TerminalDashboard(engine)
    await asyncio.gather(engine.run(), dashboard.run())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
