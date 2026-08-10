import logging
from collections import deque
from typing import List, Dict, Tuple, Optional
from regime.models import MarketBar

logger = logging.getLogger("OrderFlow.BarStore")

class BarStore:
    """
    Stores and manages historical closed candles for a single symbol.
    Provides boundaries alignment, gap detection, and uniqueness checks.
    """
    def __init__(self, symbol: str, max_bars: int = 2000):
        self.symbol = symbol.lower()
        self.max_bars = max_bars
        self.bars: Dict[str, List[MarketBar]] = {
            "1m": [],
            "5m": [],
            "15m": [],
            "1h": []
        }
        
        # Telemetry counters
        self.late_trade_count = 0
        self.duplicate_trade_count = 0
        self.out_of_order_trade_count = 0
        self.reconciliation_mismatch_count = 0
        self.history_gap_count = 0
        self.queue_overflow_count = 0
        
        # Track pending gap recovery requests: set of (open_time_ms)
        self.pending_gap_recovery: Dict[str, List[int]] = {
            "1m": [],
            "5m": [],
            "15m": [],
            "1h": []
        }

    def get_bars(self, timeframe: str) -> List[MarketBar]:
        return self.bars.get(timeframe, [])

    def append_bar(self, timeframe: str, bar: MarketBar) -> bool:
        """
        Appends a closed canonical bar to history. Enforces uniqueness and detects gaps.
        Returns True if appended, False if ignored/duplicate.
        """
        history = self.bars[timeframe]
        interval_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000
        }[timeframe]

        if not history:
            history.append(bar)
            return True

        last_bar = history[-1]
        
        # 1. Uniqueness check
        if bar.open_time_ms == last_bar.open_time_ms:
            # Check for float discrepancies within tolerance (0.01% / 1 basis point)
            def close_enough(a, b):
                if b == 0:
                    return a == 0
                return abs(a - b) / b <= 0.0001

            is_mismatch = (
                not close_enough(bar.open, last_bar.open) or
                not close_enough(bar.high, last_bar.high) or
                not close_enough(bar.low, last_bar.low) or
                not close_enough(bar.close, last_bar.close)
            )
            if is_mismatch:
                self.reconciliation_mismatch_count += 1
                logger.warning(
                    f"[{self.symbol.upper()}] {timeframe} bar discrepancy detected at {bar.open_time_ms}. "
                    f"New: {bar.open}/{bar.high}/{bar.low}/{bar.close}, Existing: {last_bar.open}/{last_bar.high}/{last_bar.low}/{last_bar.close}"
                )
            return False

        # 2. Late bar arrival check (historical insert)
        if bar.open_time_ms < last_bar.open_time_ms:
            # Find if this bar open time already exists in history
            for existing in history:
                if existing.open_time_ms == bar.open_time_ms:
                    return False
            # Otherwise insert in order
            history.append(bar)
            history.sort(key=lambda x: x.open_time_ms)
            if len(history) > self.max_bars:
                history.pop(0)
            return True

        # 3. Gap detection
        expected_next = last_bar.open_time_ms + interval_ms
        if bar.open_time_ms > expected_next:
            self.history_gap_count += 1
            missing_count = (bar.open_time_ms - expected_next) // interval_ms
            logger.warning(
                f"[{self.symbol.upper()}] {timeframe} gap detected. "
                f"Last: {last_bar.open_time_ms}, New: {bar.open_time_ms}. Missing: {missing_count} bars."
            )
            # Schedule recovery
            curr = expected_next
            while curr < bar.open_time_ms:
                if curr not in self.pending_gap_recovery[timeframe]:
                    self.pending_gap_recovery[timeframe].append(curr)
                curr += interval_ms

        # Append and maintain bounds
        history.append(bar)
        if len(history) > self.max_bars:
            history.pop(0)
        return True
