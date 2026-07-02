"""Portfolio-level state checkpointing (Python side).

The C++ Checkpointer owns the per-instrument hot state (book, VWAP sums,
position, regime) in a double-buffered mmap. This module persists the small,
slow-moving PORTFOLIO state — equity, high-water mark, breaker latch, per-
symbol positions — with the same torn-write discipline, so a crash+restart
resumes with the risk layer in exactly the state it halted in. (Recovering
the book but forgetting a latched circuit breaker would be the worst
possible failure mode: the restart would happily re-enter the drawdown.)

Format: two fixed-size slots (A/B), each  [u64 seq][u32 len][payload JSON]
[u64 seq]. JSON is fine here — this is written once per fill / few seconds,
not per tick; readability of the recovery file during an incident is worth
more than microseconds.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Any, Optional

_SLOT_PAYLOAD = 64 * 1024
_HDR = struct.Struct("<QI")     # seq, payload_len
_TRL = struct.Struct("<Q")      # seq (torn-write guard)
_SLOT_SIZE = _HDR.size + _SLOT_PAYLOAD + _TRL.size
_FILE_SIZE = 2 * _SLOT_SIZE


class PortfolioCheckpoint:
    def __init__(self, path: str):
        exists = os.path.exists(path)
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        if not exists or os.fstat(self._fd).st_size < _FILE_SIZE:
            os.ftruncate(self._fd, _FILE_SIZE)  # pre-size before mmap
        self._map = mmap.mmap(self._fd, _FILE_SIZE)
        self._seq = self._latest()[0]

    def close(self) -> None:
        self._map.flush()
        self._map.close()
        os.close(self._fd)

    def save(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state, separators=(",", ":")).encode()
        if len(payload) > _SLOT_PAYLOAD:
            raise ValueError("portfolio state exceeds checkpoint slot")
        self._seq += 1
        off = (self._seq % 2) * _SLOT_SIZE
        # Payload → trailer → header: header seq is the commit point, same
        # seqlock-on-disk pattern as the C++ side.
        self._map[off + _HDR.size: off + _HDR.size + len(payload)] = payload
        self._map[off + _HDR.size + _SLOT_PAYLOAD:
                  off + _SLOT_SIZE] = _TRL.pack(self._seq)
        self._map[off: off + _HDR.size] = _HDR.pack(self._seq, len(payload))
        # Flush the whole mapping: msync offsets must be page-aligned on
        # Linux and slots are not; the file is ~128 KiB written every few
        # seconds, so a full flush is cheaper than aligning the layout.
        self._map.flush()

    def load(self) -> Optional[dict[str, Any]]:
        seq, payload = self._latest()
        return json.loads(payload) if seq else None

    def _latest(self) -> tuple[int, Optional[bytes]]:
        best_seq, best_payload = 0, None
        for i in (0, 1):
            off = i * _SLOT_SIZE
            seq, length = _HDR.unpack_from(self._map, off)
            (tseq,) = _TRL.unpack_from(self._map, off + _HDR.size + _SLOT_PAYLOAD)
            if seq == 0 or seq != tseq or length > _SLOT_PAYLOAD:
                continue  # empty or torn slot
            raw = self._map[off + _HDR.size: off + _HDR.size + length]
            try:
                json.loads(raw)
            except ValueError:
                continue  # corrupt payload: fall back to other slot
            if seq > best_seq:
                best_seq, best_payload = seq, raw
        return best_seq, best_payload
