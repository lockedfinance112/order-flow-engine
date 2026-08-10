import logging
from typing import List, Dict, Set, Optional, Tuple
from regime.models import MarketBar

logger = logging.getLogger("OrderFlow.BarStore")

class BarStore:
    """
    Stores and manages historical closed candles for a single symbol.
    Maintains relative float tolerances, cumulative and current unresolved gap counts.
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
        
        # Gap State management
        self.unresolved_gaps: Set[Tuple[str, int]] = set() # (timeframe, open_time_ms)
        self.recovery_requests: List[Dict[str, Any]] = []

    @property
    def unresolved_gap_count(self) -> int:
        return len(self.unresolved_gaps)

    def get_bars(self, timeframe: str) -> List[MarketBar]:
        return self.bars.get(timeframe, [])

    def append_bar(self, timeframe: str, bar: MarketBar) -> bool:
        """
        Appends a closed canonical bar to history. Enforces uniqueness and detects gaps.
        Returns True if appended, False if ignored/duplicate.
        """
        # Remove from unresolved gaps if it was there
        self.unresolved_gaps.discard((timeframe, bar.open_time_ms))
        
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
                not close_enough(bar.close, last_bar.close) or
                not close_enough(bar.base_volume, last_bar.base_volume)
            )
            if is_mismatch:
                self.reconciliation_mismatch_count += 1
                logger.warning(
                    f"[{self.symbol.upper()}] {timeframe} bar mismatch at {bar.open_time_ms}."
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
            curr = expected_next
            while curr < bar.open_time_ms:
                self.unresolved_gaps.add((timeframe, curr))
                # Add to recovery requests queue
                self.recovery_requests.append({
                    "symbol": self.symbol,
                    "timeframe": timeframe,
                    "missing_open_time_ms": curr,
                    "reason": "STREAM_GAP"
                })
                curr += interval_ms

        # Append and maintain bounds
        history.append(bar)
        if len(history) > self.max_bars:
            history.pop(0)
        return True
