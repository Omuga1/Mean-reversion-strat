"""Terminal dashboard — zero-dependency ANSI TUI for the strategy engine.

Renders in-place (home-cursor + clear-below, no scrolling) a few times per
second: a risk strip, one row per instrument with live microstructure state,
an entry-proximity meter, and the recent event log. Pure read-only observer:
it touches the same C++ book objects the engine trades from (properties are
lock-free reads of POD state), so rendering costs the hot path nothing.
"""
from __future__ import annotations

import asyncio
import collections
import datetime as dt
import logging
import shutil
import sys
import time

from .regime import DEAD_ZONE_END_UTC, DEAD_ZONE_START_UTC, in_dead_zone

# ---- ANSI ------------------------------------------------------------------
RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
RED, GREEN, YELLOW, CYAN, MAGENTA = ("\x1b[31m", "\x1b[32m", "\x1b[33m",
                                     "\x1b[36m", "\x1b[35m")
HIDE_CUR, SHOW_CUR = "\x1b[?25l", "\x1b[?25h"
ALT_ON, ALT_OFF = "\x1b[?1049h", "\x1b[?1049l"
HOME, CLEAR_BELOW = "\x1b[H", "\x1b[0J"

_REGIME_COLOR = {
    "NORMAL": GREEN,
    "CEX_NIGHT_PASSIVE": CYAN,
    "DEX_NIGHT_EXIT_ONLY": YELLOW,
    "HALTED": RED,
}


class _DequeHandler(logging.Handler):
    """Captures meanrev log records into a ring buffer for the events pane."""

    def __init__(self, dq: collections.deque):
        super().__init__(level=logging.INFO)
        self._dq = dq
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._dq.append(self.format(record))
        except Exception:  # noqa: BLE001 — a UI pane must never kill logging
            pass


def _fmt_px(p: float) -> str:
    if p <= 0:
        return "—"
    return f"{p:,.2f}" if p >= 1.0 else f"{p:.8f}"


class TerminalDashboard:
    def __init__(self, engine, refresh: float = 0.5, event_lines: int = 8):
        self._e = engine
        self._refresh = refresh
        self._events: collections.deque[str] = collections.deque(maxlen=200)
        self._event_lines = event_lines
        self._prev_seq: dict[str, tuple[int, float]] = {}
        self._t0 = time.time()
        # Route all meanrev.* logging into the events pane instead of stdout
        # (stdout belongs to the renderer now).
        logging.getLogger("meanrev").addHandler(_DequeHandler(self._events))

    async def run(self) -> None:
        if not sys.stdout.isatty():
            return  # piped/CI: stay silent, the log file has everything
        out = sys.stdout
        out.write(ALT_ON + HIDE_CUR + HOME + "\x1b[2J")
        try:
            while True:
                out.write(HOME + self._render() + CLEAR_BELOW)
                out.flush()
                await asyncio.sleep(self._refresh)
        finally:
            out.write(SHOW_CUR + ALT_OFF)
            out.flush()

    # ------------------------------------------------------------- rendering
    def _render(self) -> str:
        cols = max(shutil.get_terminal_size((120, 30)).columns, 80)
        now = dt.datetime.now(dt.timezone.utc)
        up = int(time.time() - self._t0)
        lines: list[str] = []

        def rule(label: str = "") -> str:
            body = f"─ {label} " if label else ""
            return DIM + "┌" + body + "─" * max(cols - len(body) - 2, 0) + RESET

        lines.append(
            f"{BOLD}{CYAN} MEANREV{RESET}  {now:%Y-%m-%d %H:%M:%S} UTC"
            f"  {DIM}uptime {up // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}"
            f"  refresh {self._refresh:.2g}s{RESET}")
        lines.append(self._risk_strip(now, cols))
        lines.append("")
        lines.append(
            f"{BOLD}{'SYMBOL':<10} {'MID':>13} {'VWAP':>13} {'σ':>9} "
            f"{'Z':>7} {'OBI':>7} {'SPRbps':>7} {'MSG/s':>6} "
            f"{'POS':>12} {'REGIME':<20} ENTRY{RESET}")
        for ins in self._e.instruments:
            lines.append(self._instrument_row(ins.symbol))
        lines.append("")
        lines.append(f"{BOLD}EVENTS{RESET}")
        events = list(self._events)[-self._event_lines:]
        if not events:
            entry_z = next(iter(self._e.configs.values())).entry_z \
                if self._e.configs else 2.5
            lines.append(f"{DIM}  (none yet — engine waits for −{entry_z:.1f}σ "
                         f"dislocations with OBI confirmation; days without "
                         f"trades are normal){RESET}")
        for ev in events:
            lines.append("  " + ev[: cols - 2])
        return "\x1b[K" + "\n\x1b[K".join(lines) + "\n"

    def _risk_strip(self, now: dt.datetime, cols: int) -> str:
        r = self._e.risk
        pnl = self._e.total_pnl
        pnl_c = GREEN if pnl >= 0 else RED
        dd = r.drawdown_pct
        breaker = (f"{RED}{BOLD}BREAKER TRIPPED{RESET}" if r.breaker_latched
                   else f"{GREEN}breaker ok{RESET}")
        halted = f"  {RED}{BOLD}** ALL BUYING HALTED **{RESET}" if r.halted else ""

        if in_dead_zone(now):
            boundary = now.replace(hour=DEAD_ZONE_END_UTC, minute=0, second=0,
                                   microsecond=0)
            if boundary <= now:
                boundary += dt.timedelta(days=1)
            dz = f"{YELLOW}DEAD ZONE (ends {self._eta(boundary - now)}){RESET}"
        else:
            boundary = now.replace(hour=DEAD_ZONE_START_UTC, minute=0,
                                   second=0, microsecond=0)
            if boundary <= now:
                boundary += dt.timedelta(days=1)
            dz = f"{DIM}dead zone in {self._eta(boundary - now)}{RESET}"

        return (f" equity {BOLD}{r.equity:,.2f}{RESET}"
                f"  pnl {pnl_c}{pnl:+,.2f}{RESET}"
                f"  dd {dd:.2f}%/{r.cfg.max_daily_dd_pct:.1f}%"
                f"  {breaker}  {dz}{halted}")

    def _instrument_row(self, symbol: str) -> str:
        book = self._e.books[symbol]
        sig = self._e.signals[symbol]
        cfg = self._e.configs[symbol]

        # Message rate from the book's monotonic seq counter.
        seq, t = book.seq, time.time()
        pseq, pt = self._prev_seq.get(symbol, (seq, t))
        self._prev_seq[symbol] = (seq, t)
        rate = (seq - pseq) / (t - pt) if t > pt else 0.0

        z, obi = book.vwap_zscore, book.obi
        # z color: green when at/beyond entry depth (opportunity), yellow
        # when within 1σ of it, dim otherwise.
        z_c = (GREEN if z <= -cfg.entry_z
               else YELLOW if z <= -(cfg.entry_z - 1.0) else DIM)
        obi_c = GREEN if obi >= cfg.obi_min else (RED if obi < 0 else DIM)

        regime = str(sig.regime).rsplit(".", 1)[-1]
        reg_c = _REGIME_COLOR.get(regime, DIM)

        # Entry proximity: how far mid has travelled toward the -entry_z
        # trigger, plus whether the OBI confirmation is present.
        frac = min(max(-z / cfg.entry_z, 0.0), 1.0) if cfg.entry_z > 0 else 0.0
        filled = round(frac * 10)
        bar_c = GREEN if frac >= 1.0 else YELLOW if frac >= 0.6 else DIM
        bar = f"{bar_c}{'█' * filled}{DIM}{'░' * (10 - filled)}{RESET}"
        obi_ok = f"{GREEN}obi✓{RESET}" if obi >= cfg.obi_min else f"{DIM}obi✗{RESET}"

        pos = sig.position
        pos_s = (f"{GREEN}{pos:>12.6f}{RESET}" if pos > 0
                 else f"{DIM}{'—':>12}{RESET}")

        return (f"{BOLD}{symbol:<10}{RESET} {_fmt_px(book.mid):>13} "
                f"{_fmt_px(book.vwap):>13} {book.sigma:>9.2f} "
                f"{z_c}{z:>+7.2f}{RESET} {obi_c}{obi:>+7.3f}{RESET} "
                f"{book.spread_bps:>7.2f} {rate:>6.0f} {pos_s} "
                f"{reg_c}{regime:<20}{RESET} {bar} {frac * 100:3.0f}% {obi_ok}")

    @staticmethod
    def _eta(td: dt.timedelta) -> str:
        s = int(td.total_seconds())
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
