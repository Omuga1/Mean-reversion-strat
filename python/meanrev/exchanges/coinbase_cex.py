"""CoinbaseCEX — Advanced Trade adapter (CLOB venue).

Microstructure profile
----------------------
Coinbase is a price-time-priority central limit order book. Market data is a
sequenced L2 delta stream over websocket; the local book is authoritative
between snapshots as long as no sequence gap occurs. Latency budget:
~5–30 ms feed propagation, ~20–80 ms REST order entry. Queue position
matters: the night-regime deep bids are POST_ONLY so we always earn maker
fees and never accidentally cross a widened spread.

Feed → C++ handoff
------------------
Each websocket frame (which may carry hundreds of level updates during a
cascade) is flattened into three parallel arrays and applied with ONE call
to `book.apply_deltas_batch(...)`. The pybind11 layer releases the GIL for
the duration, so the event loop keeps servicing the socket while C++ chews
through the burst — this is the single most important throughput decision
in the Python layer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from typing import AsyncIterator, Optional

try:
    # orjson decodes Coinbase's L2 frames ~5-10x faster than stdlib json —
    # message decode is the largest Python-side cost on the feed path.
    from orjson import loads as json_loads
except ImportError:                              # pragma: no cover
    from json import loads as json_loads

from ..router import (ExchangeRouter, Fill, OrderRequest, OrderResult,
                      OrderSide, OrderType, VenueKind)

log = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
REST_BASE = "https://api.coinbase.com"


class CoinbaseCEX(ExchangeRouter):
    kind = VenueKind.CLOB

    def __init__(self, api_key_name: str, api_private_key_pem: str):
        super().__init__("coinbase")
        self._key_name = api_key_name
        self._key_pem = api_private_key_pem
        self._session = None          # aiohttp.ClientSession, created lazily
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue(maxsize=10_000)
        self._open_orders: dict[str, set[str]] = {}   # symbol -> order_ids

    # ------------------------------------------------------------------ auth
    def _signing_key(self):
        """CDP issues API keys in two formats depending on when/how they
        were created: legacy EC keys as a PEM string (signed with ES256),
        or the current default — an Ed25519 key as a raw base64 string
        (signed with EdDSA). Detect by shape rather than requiring the
        caller to know which vintage of key they have.
        """
        raw = self._key_pem.strip()
        if raw.startswith("-----BEGIN"):
            return raw, "ES256"

        import base64
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey)

        key_bytes = base64.b64decode(raw)
        # CDP's Ed25519 secret is the 32-byte seed, optionally followed by
        # the 32-byte public key (64 bytes total); only the seed is needed.
        seed = key_bytes[:32]
        return Ed25519PrivateKey.from_private_bytes(seed), "EdDSA"

    def _jwt(self, method: str, path: str) -> str:
        """Coinbase CDP auth: short-lived JWT bound to method+path.

        The `uri` claim pins the token to a single endpoint and the 2-minute
        expiry bounds replay. Built per-request; signing is tens of
        microseconds, irrelevant next to the network RTT.
        """
        import jwt  # PyJWT with cryptography backend

        now = int(time.time())
        payload = {
            "sub": self._key_name,
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,
            "uri": f"{method} {REST_BASE.removeprefix('https://')}{path}",
        }
        key, alg = self._signing_key()
        return jwt.encode(
            payload, key, algorithm=alg,
            headers={"kid": self._key_name, "nonce": secrets.token_hex(16)},
        )

    # ------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10))
        self._connected.set()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------ market data
    async def get_tick_size(self, symbol: str) -> float:
        """quote_increment from the public product endpoint (no auth)."""
        path = f"/api/v3/brokerage/market/products/{symbol}"

        async def _do():
            async with self._session.get(REST_BASE + path) as resp:
                data = await resp.json()
                if resp.status != 200 or "quote_increment" not in data:
                    raise ConnectionError(f"product lookup failed: {data}")
                return float(data["quote_increment"])

        return await self._retry(_do)

    async def stream_market_data(self, symbol: str, book,
                                 on_update=None) -> None:
        """level2 + market_trades channels → C++ book. Reconnect forever."""
        import websockets

        while True:
            try:
                # max_size: the library default (1 MiB) is smaller than a
                # full BTC-USD level2 snapshot, which makes the client abort
                # with 1009 "message too big" on every connect. 32 MiB is
                # ample for any snapshot while still bounding memory.
                async with websockets.connect(
                        WS_URL, max_size=32 * 1024 * 1024,
                        max_queue=None) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe", "product_ids": [symbol],
                        "channel": "level2",
                    }))
                    await ws.send(json.dumps({
                        "type": "subscribe", "product_ids": [symbol],
                        "channel": "market_trades",
                    }))
                    log.info("coinbase %s: subscribed (level2 + trades)",
                             symbol)
                    last_seq: Optional[int] = None
                    async for raw in ws:
                        msg = json_loads(raw)
                        seq = msg.get("sequence_num")
                        if last_seq is not None and seq not in (None, last_seq + 1):
                            # Sequence gap: the local book is now fiction.
                            # Drop it and resubscribe for a fresh snapshot
                            # rather than trade on desynchronized state.
                            log.warning("coinbase seq gap %s→%s; resync",
                                        last_seq, seq)
                            book.clear()
                            break
                        last_seq = seq
                        self._dispatch(msg, book)
                        if on_update is not None:
                            on_update()  # wake the signal loop: fresh state
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — feed must self-heal
                log.warning("coinbase ws error: %s; reconnecting", exc)
                book.clear()
                await asyncio.sleep(1.0)

    def _dispatch(self, msg: dict, book) -> None:
        channel = msg.get("channel")
        if channel == "l2_data":
            for event in msg.get("events", []):
                if event.get("type") == "snapshot":
                    book.clear()
                updates = event.get("updates", [])
                # Flatten frame → parallel arrays → ONE GIL-releasing call.
                sides = [0 if u["side"] == "bid" else 1 for u in updates]
                prices = [float(u["price_level"]) for u in updates]
                qtys = [float(u["new_quantity"]) for u in updates]
                if sides:
                    book.apply_deltas_batch(sides, prices, qtys)
        elif channel == "market_trades":
            for event in msg.get("events", []):
                trades = event.get("trades", [])
                if trades:
                    book.on_trades_batch(
                        [float(t["price"]) for t in trades],
                        [float(t["size"]) for t in trades])

    # ------------------------------------------------------------ order entry
    async def submit_order(self, req: OrderRequest) -> OrderResult:
        client_id = req.client_id or secrets.token_hex(16)  # idempotency key
        cfg: dict
        if req.order_type == OrderType.MARKET:
            cfg = {"market_market_ioc": {"base_size": str(req.qty)}}
        else:
            cfg = {"limit_limit_gtc": {
                "base_size": str(req.qty),
                "limit_price": str(req.limit_price),
                # Night regime posts 3–4σ under VWAP; post_only guarantees
                # we are the resting liquidity, never the taker.
                "post_only": req.order_type == OrderType.LIMIT_POST_ONLY,
            }}
        body = {
            "client_order_id": client_id,
            "product_id": req.symbol,
            "side": req.side.value.upper(),
            "order_configuration": cfg,
        }
        path = "/api/v3/brokerage/orders"

        async def _do():
            async with self._session.post(
                REST_BASE + path, json=body,
                headers={"Authorization": f"Bearer {self._jwt('POST', path)}"},
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or not data.get("success", False):
                    return OrderResult(ok=False, error=json.dumps(data))
                oid = data["success_response"]["order_id"]
                self._open_orders.setdefault(req.symbol, set()).add(oid)
                return OrderResult(ok=True, order_id=oid)

        return await self._retry(_do)

    async def cancel_all(self, symbol: str) -> None:
        oids = list(self._open_orders.get(symbol, ()))
        if not oids:
            return
        path = "/api/v3/brokerage/orders/batch_cancel"

        async def _do():
            async with self._session.post(
                REST_BASE + path, json={"order_ids": oids},
                headers={"Authorization": f"Bearer {self._jwt('POST', path)}"},
            ) as resp:
                await resp.json()

        await self._retry(_do)
        self._open_orders[symbol] = set()

    async def fills(self) -> AsyncIterator[Fill]:
        while True:
            yield await self._fill_queue.get()
