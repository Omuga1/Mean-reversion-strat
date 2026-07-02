"""ExchangeRouter — the asyncio abstraction over heterogeneous venues.

Division of labor with the C++ core
-----------------------------------
The C++ `microcore` owns everything that must be decided in microseconds on
every tick: book maintenance, OBI, VWAP bands, regime-gated signal
evaluation. Python owns everything that is I/O-bound and inherently
milliseconds-scale anyway: websocket/RPC transport, order signing, retry
logic, venue quirks. An asyncio event loop is a perfectly good scheduler for
work whose latency floor is the network RTT — Coinbase order entry is
~20–80 ms over the public internet, and a Solana slot is 400 ms. Shaving
Python overhead there buys nothing; shaving it in the per-tick book update
(where C++ runs at ~10⁷ deltas/sec) buys everything.

Normalization contract
----------------------
Every venue adapter translates its native feed into calls against the shared
C++ `OrderBook` and translates `Signal` actions into venue-native orders.
The strategy engine never sees venue-specific types.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

log = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_POST_ONLY = "limit_post_only"   # maker-or-cancel: night regime uses this


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class VenueKind(Enum):
    CLOB = "clob"            # centralized limit order book (Coinbase)
    BONDING_CURVE = "curve"  # Pump.fun constant-product-ish bonding curve


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float                       # base asset quantity
    limit_price: Optional[float] = None
    client_id: Optional[str] = None
    # DEX-only knobs:
    max_slippage_bps: float = 100.0  # curve venues: reject worse fills
    use_mev_protection: bool = True  # Solana: route through Jito bundle


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_price: float = 0.0
    error: Optional[str] = None


@dataclass
class Fill:
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    ts: float = field(default_factory=time.time)


class ExchangeRouter(abc.ABC):
    """Abstract venue adapter. One instance per venue; instruments multiplex.

    Lifecycle: `connect()` → `stream_market_data()` task feeds the C++ book
    → engine calls order methods → `close()`.

    Backpressure: adapters must apply market-data frames to the C++ book via
    the *batch* entry points (`apply_deltas_batch`) so a burst costs one
    GIL round-trip per frame, not per level. The C++ side is allocation-free
    and strictly bounded in memory, so it can never be the party that falls
    behind — if the asyncio loop lags, the websocket's TCP window is the
    backpressure mechanism, and a sequence-gap triggers a clean resync.
    """

    kind: VenueKind

    def __init__(self, name: str):
        self.name = name
        self._connected = asyncio.Event()

    # ---- lifecycle -------------------------------------------------------
    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    # ---- market data -----------------------------------------------------
    @abc.abstractmethod
    async def stream_market_data(self, symbol: str, book,
                                 on_update=None) -> None:
        """Run forever: pump venue L2/trade events into a microcore.OrderBook.

        `on_update` (a zero-arg callable, typically asyncio.Event.set) MUST
        be invoked after each applied frame — it is what makes the signal
        loop event-driven instead of polled, so forgetting it silently adds
        up to SIGNAL_FALLBACK_S of decision latency.

        Must detect sequence gaps and resync (clear + snapshot) rather than
        apply deltas onto a desynchronized book — a silently wrong book is
        strictly worse than a briefly empty one.
        """

    # ---- order entry -----------------------------------------------------
    @abc.abstractmethod
    async def submit_order(self, req: OrderRequest) -> OrderResult: ...

    @abc.abstractmethod
    async def cancel_all(self, symbol: str) -> None:
        """Pull every resting order for `symbol`. Must be idempotent and is
        invoked on every regime transition (CANCEL_ALL_BIDS) and on any
        drawdown/correlation halt — it is the safety-critical primitive."""

    @abc.abstractmethod
    def fills(self) -> AsyncIterator[Fill]:
        """Async stream of own-order executions for position accounting."""

    # ---- shared helpers ----------------------------------------------------
    async def _retry(self, coro_factory, attempts: int = 4, base_delay: float = 0.25):
        """Exponential backoff for transient transport failures. Order entry
        retries must be paired with idempotent client_ids upstream so a
        timeout-then-retry can never double-fill."""
        delay = base_delay
        for attempt in range(attempts):
            try:
                return await coro_factory()
            except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                if attempt == attempts - 1:
                    raise
                log.warning("%s transient failure (%s); retry in %.2fs",
                            self.name, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
