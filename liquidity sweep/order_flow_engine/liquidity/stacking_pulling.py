from collections import deque
from typing import Dict, Tuple, List

class StackingPullingDetector:
    """
    Tracks order book depth changes over 250ms, 1s, 5s, and 15s lookback windows
    to detect stacking (bids/asks added) and pulling (bids/asks cancelled).
    """
    def __init__(self, history_seconds: float = 20.0):
        self.history = deque()  # stores (timestamp, bids_dict, asks_dict)
        self.max_history = history_seconds

    def record_snapshot(self, timestamp: float, bids: Dict[float, float], asks: Dict[float, float]):
        """Saves a snapshot of the current book state."""
        self.history.append((timestamp, bids.copy(), asks.copy()))
        
        # Prune old snapshots
        cutoff = timestamp - self.max_history
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def get_changes(self, lookback_seconds: float, mid_price: float, range_pct: float = 0.01) -> Dict[str, float]:
        """
        Computes changes in bids/asks over lookback_seconds within range_pct of mid_price.
        Returns: {bids_added, bids_pulled, asks_added, asks_pulled, net_shift}
        """
        if len(self.history) < 2 or mid_price <= 0.0:
            return {"bids_added": 0.0, "bids_pulled": 0.0, "asks_added": 0.0, "asks_pulled": 0.0, "net_shift": 0.0}

        now_ts, cur_bids, cur_asks = self.history[-1]
        target_ts = now_ts - lookback_seconds

        # Find the snapshot closest to target_ts
        hist_idx = 0
        min_diff = abs(self.history[0][0] - target_ts)
        for idx, (ts, _, _) in enumerate(self.history):
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                hist_idx = idx

        _, hist_bids, hist_asks = self.history[hist_idx]

        bids_added = 0.0
        bids_pulled = 0.0
        asks_added = 0.0
        asks_pulled = 0.0

        # Define bounds (e.g. within 1% of mid price)
        lower_bound = mid_price * (1.0 - range_pct)
        upper_bound = mid_price * (1.0 + range_pct)

        # 1. Bids changes
        all_bid_prices = set(cur_bids.keys()).union(hist_bids.keys())
        for p in all_bid_prices:
            if p < lower_bound or p > mid_price:
                continue
            cur_q = cur_bids.get(p, 0.0)
            hist_q = hist_bids.get(p, 0.0)
            diff = cur_q - hist_q
            if diff > 0:
                bids_added += diff * p
            else:
                bids_pulled += abs(diff) * p

        # 2. Asks changes
        all_ask_prices = set(cur_asks.keys()).union(hist_asks.keys())
        for p in all_ask_prices:
            if p > upper_bound or p < mid_price:
                continue
            cur_q = cur_asks.get(p, 0.0)
            hist_q = hist_asks.get(p, 0.0)
            diff = cur_q - hist_q
            if diff > 0:
                asks_added += diff * p
            else:
                asks_pulled += abs(diff) * p

        # Net shift = (Bids Added + Asks Pulled) - (Bids Pulled + Asks Added)
        net_shift = (bids_added + asks_pulled) - (bids_pulled + asks_added)

        return {
            "bids_added": bids_added,
            "bids_pulled": bids_pulled,
            "asks_added": asks_added,
            "asks_pulled": asks_pulled,
            "net_shift": net_shift
        }
