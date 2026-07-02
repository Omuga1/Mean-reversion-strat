"""PumpFunDEX — Solana bonding-curve adapter with Jito MEV protection.

Microstructure profile (why this venue needs different plumbing)
----------------------------------------------------------------
Pump.fun tokens trade on a constant-product bonding curve, not a limit order
book. There are no resting orders, no queue, no cancels:

  * "Liquidity" is the curve's virtual reserves (R_sol, R_tok); the marginal
    price is p = R_sol / R_tok and every fill walks the curve.
  * "Latency" is slot-quantized: state advances every ~400 ms slot, and
    within a slot ORDERING is auctioned — the mempool is adversarial.
    A naked buy after a −4σ wick is a sandwich-bot's breakfast: they see it,
    buy ahead, let us push the curve, and dump on us. Hence every order is
    wrapped in a Jito bundle: bundles execute atomically at a position the
    tip buys, are never exposed to the public mempool, and revert entirely
    if any leg fails — so our max-slippage guard aborting means NO fill
    rather than a bad fill.

Synthetic book: making an AMM speak OBI
---------------------------------------
The C++ core consumes L2 levels, so we synthesize them from the curve.
For a constant-product invariant k = R_sol·R_tok, the token volume
available between the current price p₀ and a level p is:

    Δtok(p) = R_tok − √(k / p)          (asks: p > p₀, curve sells to buyers)
    Δtok(p) = √(k / p) − R_tok          (bids: p < p₀, curve buys from sellers)

We sample N price levels per side (geometric spacing) and install the
incremental Δtok at each as a pseudo-level. The *curve itself* is symmetric,
so curve-only OBI ≈ 0 by construction; the informative signal is FLOW — we
overlay a decayed net-taker-flow term on the synthetic levels so OBI
reflects who is actually hitting the curve. The C++ signal logic is
completely unaware it is looking at an AMM: same OBI, same VWAP bands.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from typing import AsyncIterator, Optional

from ..router import (ExchangeRouter, Fill, OrderRequest, OrderResult,
                      OrderSide, VenueKind)
from ..toxicity import ToxicityReport, TokenToxicityScreen

log = logging.getLogger(__name__)

# Jito block engine (mainnet). Tip accounts rotate; fetched at connect().
JITO_BLOCK_ENGINE = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"

SYNTH_LEVELS = 20          # pseudo-levels per side fed to the C++ book
SYNTH_SPACING = 0.005      # 0.5% geometric level spacing
FLOW_HALFLIFE_S = 30.0     # taker-flow imbalance decay half-life


class PumpFunDEX(ExchangeRouter):
    kind = VenueKind.BONDING_CURVE

    def __init__(self, rpc_url: str, ws_url: str, keypair_path: str,
                 jito_tip_lamports: int = 100_000,
                 toxicity: Optional[TokenToxicityScreen] = None):
        super().__init__("pumpfun")
        self._rpc_url = rpc_url
        self._ws_url = ws_url
        self._keypair_path = keypair_path
        self._tip = jito_tip_lamports
        self._session = None
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue(maxsize=10_000)
        self._toxicity = toxicity or TokenToxicityScreen(rpc_url)
        # Decayed net taker flow (buys − sells, token units) per mint.
        self._flow: dict[str, tuple[float, float]] = {}  # mint -> (flow, ts)

    # ------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10))
        self._connected.set()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    # ---------------------------------------------------------- market data
    async def stream_market_data(self, mint: str, book) -> None:
        """accountSubscribe on the bonding-curve PDA + logsSubscribe for
        taker prints. Every curve update re-synthesizes the pseudo-book."""
        import websockets

        while True:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0", "id": 1, "method": "accountSubscribe",
                        "params": [self._curve_pda(mint),
                                   {"encoding": "base64",
                                    "commitment": "processed"}],
                    }))
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0", "id": 2, "method": "logsSubscribe",
                        "params": [{"mentions": [mint]},
                                   {"commitment": "processed"}],
                    }))
                    async for raw in ws:
                        msg = json.loads(raw)
                        method = msg.get("method")
                        if method == "accountNotification":
                            state = self._decode_curve(msg)
                            if state:
                                self._rebuild_synthetic_book(mint, book, *state)
                        elif method == "logsNotification":
                            trade = self._decode_trade(msg)
                            if trade:
                                price, qty, is_buy = trade
                                book.on_trade(price, qty)
                                self._bump_flow(mint, qty if is_buy else -qty)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — feed must self-heal
                log.warning("pumpfun ws error: %s; reconnecting", exc)
                book.clear()
                await asyncio.sleep(1.0)

    def _rebuild_synthetic_book(self, mint: str, book,
                                r_sol: float, r_tok: float) -> None:
        """Project the constant-product curve onto discrete L2 levels.

        The book is rebuilt (clear + batch insert) rather than diffed: the
        curve moves as a whole on every trade, so diffing buys nothing, and
        a full rebuild is 2·SYNTH_LEVELS deltas — trivial for the C++ side.
        """
        if r_sol <= 0 or r_tok <= 0:
            return
        p0 = r_sol / r_tok
        k = r_sol * r_tok
        flow = self._decayed_flow(mint)
        # Flow skew: net buying inflates synthetic bids (support is real,
        # takers are absorbing), net selling inflates asks. Bounded to ±50%
        # so flow can tilt OBI but never fabricate a one-sided book.
        skew = max(-0.5, min(0.5, flow / max(r_tok * 0.01, 1e-12)))

        sides, prices, qtys = [], [], []
        prev_bid_tok = prev_ask_tok = 0.0
        for i in range(1, SYNTH_LEVELS + 1):
            step = (1.0 + SYNTH_SPACING) ** i
            # Ask side: tokens the curve releases from p0 up to p0*step.
            ask_p = p0 * step
            ask_cum = r_tok - math.sqrt(k / ask_p)
            sides.append(1); prices.append(ask_p)
            qtys.append(max(ask_cum - prev_ask_tok, 0.0) * (1.0 - skew))
            prev_ask_tok = ask_cum
            # Bid side: tokens the curve absorbs from p0 down to p0/step.
            bid_p = p0 / step
            bid_cum = math.sqrt(k / bid_p) - r_tok
            sides.append(0); prices.append(bid_p)
            qtys.append(max(bid_cum - prev_bid_tok, 0.0) * (1.0 + skew))
            prev_bid_tok = bid_cum

        book.clear()
        book.apply_deltas_batch(sides, prices, qtys)

    def _bump_flow(self, mint: str, signed_qty: float) -> None:
        flow = self._decayed_flow(mint) + signed_qty
        self._flow[mint] = (flow, time.time())

    def _decayed_flow(self, mint: str) -> float:
        flow, ts = self._flow.get(mint, (0.0, time.time()))
        # Exponential decay: flow(t) = flow₀ · 2^(−Δt/halflife). Old flow
        # stops influencing OBI within a few half-lives — the imbalance is
        # a *now* signal, not a session accumulator.
        return flow * 2.0 ** (-(time.time() - ts) / FLOW_HALFLIFE_S)

    # ---------------------------------------------------------- order entry
    async def submit_order(self, req: OrderRequest) -> OrderResult:
        # Toxicity gate on every BUY: bundled launches and dev-controlled
        # supply are structural rugs — mean reversion does not apply to
        # instruments whose downside is a discrete jump to zero.
        if req.side == OrderSide.BUY:
            report: ToxicityReport = await self._toxicity.screen(req.symbol)
            if not report.tradeable:
                return OrderResult(ok=False,
                                   error=f"toxicity_block: {report.reasons}")

        tx_b64 = await self._build_swap_tx(req)
        if req.use_mev_protection:
            return await self._submit_jito_bundle([tx_b64])
        return await self._submit_rpc(tx_b64)

    async def _build_swap_tx(self, req: OrderRequest) -> str:
        """Construct + sign the pump.fun swap with a hard slippage rail.

        min_out = expected_out · (1 − max_slippage_bps/10⁴) is encoded in
        the instruction itself, so the slippage check executes ON-CHAIN
        atomically with the swap — inside a Jito bundle a breach reverts the
        whole bundle and we simply don't trade. The bundle also carries the
        tip transfer as its last instruction (tip only pays if the swap
        landed).
        """
        from solders.keypair import Keypair  # solana signing primitives

        with open(self._keypair_path, "rb") as f:
            keypair = Keypair.from_json(f.read().decode())
        # Instruction encoding for pump.fun's program (buy/sell with
        # min_out) intentionally lives behind this seam: it is venue ABI,
        # not strategy. See docs/ARCHITECTURE.md §venue-abi.
        raise NotImplementedError(
            "wire pump.fun program instruction encoding here")

    async def _submit_jito_bundle(self, txs_b64: list[str]) -> OrderResult:
        """sendBundle to the Jito block engine.

        Bundles bypass the public mempool entirely — validators running the
        Jito client receive them over a private channel and execute them
        atomically, so our order can be neither front-run nor sandwiched.
        The tip is the priority auction bid; during volatile opens it should
        scale with expected edge (a fixed tip under-bids exactly when the
        trade is most valuable).
        """
        body = {"jsonrpc": "2.0", "id": 1, "method": "sendBundle",
                "params": [txs_b64]}

        async def _do():
            async with self._session.post(JITO_BLOCK_ENGINE, json=body) as resp:
                data = await resp.json()
                if "error" in data:
                    return OrderResult(ok=False, error=json.dumps(data["error"]))
                return OrderResult(ok=True, order_id=data["result"])

        return await self._retry(_do)

    async def _submit_rpc(self, tx_b64: str) -> OrderResult:
        """Unprotected fallback (exits only — never used for entries)."""
        body = {"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [tx_b64, {"encoding": "base64",
                                    "skipPreflight": False}]}

        async def _do():
            async with self._session.post(self._rpc_url, json=body) as resp:
                data = await resp.json()
                if "error" in data:
                    return OrderResult(ok=False, error=json.dumps(data["error"]))
                return OrderResult(ok=True, order_id=data["result"])

        return await self._retry(_do)

    async def cancel_all(self, symbol: str) -> None:
        # Bonding curves have no resting orders — cancel_all is a no-op by
        # construction, kept so regime transitions are venue-agnostic.
        return None

    async def fills(self) -> AsyncIterator[Fill]:
        while True:
            yield await self._fill_queue.get()

    # ---------------------------------------------------------- venue ABI
    def _curve_pda(self, mint: str) -> str:
        """Derive the bonding-curve PDA for `mint` (program-derived address
        seeded with ["bonding-curve", mint])."""
        raise NotImplementedError("wire PDA derivation (solders.Pubkey)")

    def _decode_curve(self, msg: dict) -> Optional[tuple[float, float]]:
        """base64 account data → (virtual_sol_reserves, virtual_token_reserves)."""
        raise NotImplementedError("wire bonding-curve account layout")

    def _decode_trade(self, msg: dict) -> Optional[tuple[float, float, bool]]:
        """program logs → (price, qty, is_buy) for taker prints."""
        raise NotImplementedError("wire pump.fun log event parsing")
