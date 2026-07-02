"""NightMarketRegime — exchange-aware handling of the Crypto Dead Zone.

12 AM–6 AM EST ≡ 05:00–11:00 UTC (EST = UTC−5; we deliberately pin the
window in UTC rather than tracking US daylight-saving shifts: the liquidity
phenomenon follows the *global* session structure — US asleep, Europe not
yet at desks — which is a UTC phenomenon, not a wall-clock one).

Venue-specific posture, applied by pushing a Regime enum into the C++
SignalGenerator (the gate lives in C++ so it is enforced on the same code
path that generates signals — Python only decides WHICH regime is active):

  CoinbaseCEX  → CEX_NIGHT_PASSIVE
      Books thin out and spreads widen; marketable orders pay the widened
      spread and move the market. Posture: cancel working aggressive
      orders, quote deep POST_ONLY bids at VWAP − 3–4σ, and let anomalous
      liquidation wicks come to us. We are compensated as the liquidity
      provider precisely when liquidity is scarce.

  PumpFunDEX   → DEX_NIGHT_EXIT_ONLY
      A deep night wick on a meme coin is, in the base-rate sense, a
      developer dump or rug — supply-driven, permanent, non-reverting.
      Buying it is adverse selection with extra steps. Posture: dip-buying
      disabled entirely; the engine may only trail stops or exit inventory.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from .router import VenueKind

if TYPE_CHECKING:
    pass

DEAD_ZONE_START_UTC = 5    # 05:00 UTC == 12 AM EST
DEAD_ZONE_END_UTC = 11     # 11:00 UTC == 6 AM EST (exclusive)


def in_dead_zone(now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("in_dead_zone requires a tz-aware datetime")
    hour = now.astimezone(dt.timezone.utc).hour
    return DEAD_ZONE_START_UTC <= hour < DEAD_ZONE_END_UTC


class NightMarketRegime:
    """Maps (venue kind, clock, risk state) → microcore.Regime.

    Risk halts dominate the clock: HALTED always wins. The mapping is pure
    and stateless so the engine can call it every loop iteration; regime
    *transitions* are detected inside the C++ SignalGenerator, which emits
    a one-shot CANCEL_ALL_BIDS on any change.
    """

    def resolve(self, venue_kind: VenueKind, halted: bool,
                now: dt.datetime | None = None):
        import microcore  # C++ extension; imported here so pure-Python
                          # tooling can import this module without the .so

        if halted:
            return microcore.Regime.HALTED
        if in_dead_zone(now):
            if venue_kind == VenueKind.CLOB:
                return microcore.Regime.CEX_NIGHT_PASSIVE
            return microcore.Regime.DEX_NIGHT_EXIT_ONLY
        return microcore.Regime.NORMAL
