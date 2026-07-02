"""Runnable entry point: Coinbase multi-symbol trading with the terminal
dashboard.

    python3 trade_example.py

Expects cdp_api_key.json (the file Coinbase's CDP portal gives you) in the
working directory. Logs go to meanrev.log AND the dashboard's events pane —
stdout itself belongs to the dashboard renderer.

WARNING: this places real orders when a signal fires (~base_notional USD
per entry, see RiskConfig). Use a view-only API key to observe first.
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

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ---- entry tuning ----------------------------------------------------------
# ENTRY_Z: how many σ below VWAP price must fall to trigger a long. Lower =
#   more trades, but each is weaker evidence of a real (mean-reverting)
#   anomaly rather than ordinary noise. Default 2.5; 2.0 is a moderate
#   loosening (~2-3x more entries); below ~1.8 you're mostly buying dips.
# OBI_MIN: required bid-side order-book imbalance to confirm the entry (the
#   anti-falling-knife check). Lower accepts weaker passive support.
ENTRY_Z = 2.0
OBI_MIN = 0.10

logging.basicConfig(
    filename="meanrev.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def main():
    with open("cdp_api_key.json") as f:
        creds = json.load(f)

    exchange = CoinbaseCEX(creds["name"], creds["privateKey"])
    engine = StrategyEngine(
        [Instrument(sym, 0.01, exchange) for sym in SYMBOLS],
        risk=RiskManager(RiskConfig(max_daily_dd_pct=3.0),
                         starting_equity=100_000.0),
    )
    # Apply the loosened trigger to every instrument. These mutate the same
    # SignalConfig objects the C++ SignalGenerator reads, so the change
    # takes effect on the next evaluation with no rebuild.
    for cfg in engine.configs.values():
        cfg.entry_z = ENTRY_Z
        cfg.obi_min = OBI_MIN

    dashboard = TerminalDashboard(engine)
    await asyncio.gather(engine.run(), dashboard.run())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
