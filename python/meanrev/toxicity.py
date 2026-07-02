"""Token toxicity screen for Pump.fun instruments.

Mean reversion presumes the left tail is noise. For meme coins the left tail
is frequently STRUCTURAL — a developer dump or coordinated rug — and no
z-score can distinguish the two from price alone. This module screens the
*instrument*, not the price action, using on-chain forensics:

  1. Bundled-launch detection — did a coordinated set of wallets buy in the
     creation slot / first few slots? Bundled launches concentrate supply in
     wallets that exit together; the resulting "dip" never reverts.
  2. Developer wallet tracing — walk the funding graph out from the creator
     wallet (who funded it, whom it funded) and attribute supply held by
     that cluster. Dev-cluster supply above a threshold means the reversion
     trade is short gamma to one person's exit button.

Results are cached with a TTL: toxicity is a slowly-varying property of the
token, and the screen sits on the ORDER path (see PumpFunDEX.submit_order),
so it must answer from cache in the common case.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# --- thresholds (tuned on historical rug post-mortems; conservative) --------
MAX_CREATION_SLOT_BUYERS = 4       # > N distinct buyers in slot 0–2 → bundled
MAX_DEV_CLUSTER_SUPPLY_PCT = 15.0  # dev cluster holding > 15% supply → toxic
FUNDING_GRAPH_MAX_HOPS = 2         # funding-trace radius around the creator
SCREEN_TTL_S = 300.0               # re-screen every 5 minutes


@dataclass
class ToxicityReport:
    mint: str
    tradeable: bool
    bundled_launch: bool = False
    dev_cluster_supply_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class TokenToxicityScreen:
    def __init__(self, rpc_url: str):
        self._rpc_url = rpc_url
        self._cache: dict[str, ToxicityReport] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def screen(self, mint: str) -> ToxicityReport:
        cached = self._cache.get(mint)
        if cached and time.time() - cached.ts < SCREEN_TTL_S:
            return cached
        # Per-mint lock: N concurrent buy attempts on the same token must
        # not fan out into N identical RPC forensic sweeps.
        lock = self._locks.setdefault(mint, asyncio.Lock())
        async with lock:
            cached = self._cache.get(mint)
            if cached and time.time() - cached.ts < SCREEN_TTL_S:
                return cached
            report = await self._run_screen(mint)
            self._cache[mint] = report
            return report

    async def _run_screen(self, mint: str) -> ToxicityReport:
        reasons: list[str] = []

        bundled, n_slot0 = await self._detect_bundled_launch(mint)
        if bundled:
            reasons.append(f"bundled_launch:{n_slot0}_buyers_in_creation_slots")

        dev_pct = await self._trace_dev_cluster_supply(mint)
        if dev_pct > MAX_DEV_CLUSTER_SUPPLY_PCT:
            reasons.append(f"dev_cluster_supply:{dev_pct:.1f}%")

        report = ToxicityReport(
            mint=mint,
            tradeable=not reasons,
            bundled_launch=bundled,
            dev_cluster_supply_pct=dev_pct,
            reasons=reasons,
        )
        if not report.tradeable:
            log.warning("toxicity BLOCK %s: %s", mint, reasons)
        return report

    async def _detect_bundled_launch(self, mint: str) -> tuple[bool, int]:
        """Fetch the mint's creation transaction and the first ~3 slots of
        activity (getSignaturesForAddress from the earliest signature, then
        getTransaction on each). Count DISTINCT fee payers that bought in
        those slots; also flag if multiple buys share one Jito bundle
        (identical slot + adjacent transaction indices + common tip payer).
        A launch where the creator's snipers already own the float has no
        organic holder base for price to revert to."""
        # RPC forensic sweep — venue ABI seam, see docs/ARCHITECTURE.md.
        raise NotImplementedError("wire getSignaturesForAddress sweep")

    async def _trace_dev_cluster_supply(self, mint: str) -> float:
        """BFS the funding graph around the creator wallet up to
        FUNDING_GRAPH_MAX_HOPS (SOL transfers in either direction within the
        token's lifetime), then sum the cluster's share of token supply via
        getTokenLargestAccounts ∩ cluster. Returns percent of supply."""
        raise NotImplementedError("wire funding-graph BFS")
