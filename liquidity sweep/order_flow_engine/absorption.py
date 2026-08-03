import logging
from typing import Dict, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger("OrderFlow.Absorption")

class AbsorptionDetector:
    """
    Analyzes a symbol's trade history to detect absorption and delta divergence in USDT Notional:
    - Bullish Absorption: Heavy selling delta (USDT) but price fails to make a new low / closes flat or green.
    - Bearish Absorption: Heavy buying delta (USDT) but price fails to make a new high / closes flat or red.
    - CVD Divergence: Price moves in one direction while Notional CVD moves in the opposite direction.
    """
    def __init__(self, 
                 delta_threshold_usdt_1m: float = 200000.0,  # $200,000 Notional Delta
                 delta_threshold_usdt_5m: float = 500000.0,  # $500,000 Notional Delta
                 price_pct_threshold: float = 0.02): # 0.02% price change
        self.delta_threshold_1m = delta_threshold_usdt_1m
        self.delta_threshold_5m = delta_threshold_usdt_5m
        self.price_pct_threshold = price_pct_threshold

    def check_absorption(self, symbol: str, trades: deque, window_seconds: int) -> Tuple[Optional[str], str]:
        """
        Scans the trades of a specific symbol in the given window in USDT notional.
        Returns (event_type, notes) or (None, "")
        """
        if len(trades) < 10:
            return None, ""

        current_time = trades[-1]["timestamp"]
        cutoff = current_time - window_seconds

        # Extract trades in window
        window_trades = [t for t in trades if t["timestamp"] >= cutoff]
        if len(window_trades) < 10:
            return None, ""

        # Calculate price metrics
        open_price = window_trades[0]["price"]
        close_price = window_trades[-1]["price"]
        high_price = max(t["price"] for t in window_trades)
        low_price = min(t["price"] for t in window_trades)

        # Calculate notional volumes (USDT)
        buy_vol_usdt = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in window_trades if t["side"] == "BUY")
        sell_vol_usdt = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in window_trades if t["side"] == "SELL")
        delta_usdt = buy_vol_usdt - sell_vol_usdt
        total_vol_usdt = buy_vol_usdt + sell_vol_usdt
        
        if total_vol_usdt == 0:
            return None, ""

        buy_ratio = buy_vol_usdt / total_vol_usdt
        sell_ratio = sell_vol_usdt / total_vol_usdt

        # Choose threshold based on window size
        if window_seconds <= 60:
            delta_thresh = self.delta_threshold_1m
        else:
            delta_thresh = self.delta_threshold_5m

        price_pct_change = ((close_price - open_price) / open_price) * 100

        # Bullish Absorption: Strong seller aggression but price is stabilizing/rising
        if delta_usdt < -delta_thresh and sell_ratio >= 0.60:
            is_flat_or_green = price_pct_change >= -self.price_pct_threshold
            bounced_from_low = ((close_price - low_price) / low_price) * 100 >= self.price_pct_threshold * 2
            
            if is_flat_or_green or bounced_from_low:
                notes = (f"[{symbol.upper()}] Sell Delta: ${abs(delta_usdt):,.0f} ({sell_ratio*100:.1f}%), "
                         f"Price: {open_price:.1f} -> {close_price:.1f} ({price_pct_change:+.3f}%)")
                return "BULLISH_ABSORPTION", notes

        # Bearish Absorption: Strong buyer aggression but price is stalling/dropping
        if delta_usdt > delta_thresh and buy_ratio >= 0.60:
            is_flat_or_red = price_pct_change <= self.price_pct_threshold
            rejected_from_high = ((high_price - close_price) / high_price) * 100 >= self.price_pct_threshold * 2
            
            if is_flat_or_red or rejected_from_high:
                notes = (f"[{symbol.upper()}] Buy Delta: ${delta_usdt:,.0f} ({buy_ratio*100:.1f}%), "
                         f"Price: {open_price:.1f} -> {close_price:.1f} ({price_pct_change:+.3f}%)")
                return "BEARISH_ABSORPTION", notes

        return None, ""

    def check_cvd_divergence(self, symbol: str, trades: deque, window_seconds: int) -> Tuple[Optional[str], str]:
        """
        Compares price slope to USDT Notional CVD slope over a window to find divergence.
        """
        if len(trades) < 20:
            return None, ""

        current_time = trades[-1]["timestamp"]
        cutoff = current_time - window_seconds

        window_trades = [t for t in trades if t["timestamp"] >= cutoff]
        if len(window_trades) < 20:
            return None, ""

        # Break the window into 2 halves to check trends
        midpoint = len(window_trades) // 2
        first_half = window_trades[:midpoint]
        second_half = window_trades[midpoint:]

        # Price values
        avg_price_1 = sum(t["price"] for t in first_half) / len(first_half)
        avg_price_2 = sum(t["price"] for t in second_half) / len(second_half)
        price_trend = "UP" if avg_price_2 > avg_price_1 else "DOWN"

        # Calculate delta trend in USDT notional
        first_half_buy = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in first_half if t["side"] == "BUY")
        first_half_sell = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in first_half if t["side"] == "SELL")
        first_half_delta_usdt = first_half_buy - first_half_sell

        second_half_buy = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in second_half if t["side"] == "BUY")
        second_half_sell = sum(t.get("notional_usdt", t["price"] * t["quantity"]) for t in second_half if t["side"] == "SELL")
        second_half_delta_usdt = second_half_buy - second_half_sell

        # CVD divergence:
        # 1. Price is going UP, but delta in second half is significantly more negative than first half
        if price_trend == "UP" and second_half_delta_usdt < first_half_delta_usdt - self.delta_threshold_5m:
            notes = (f"[{symbol.upper()}] Price rising ({avg_price_1:.1f}->{avg_price_2:.1f}), "
                     f"but Delta declining (${first_half_delta_usdt:,.0f}->${second_half_delta_usdt:,.0f})")
            return "BEARISH_DIVERGENCE", notes

        # 2. Price is going DOWN, but delta in second half is significantly more positive than first half
        if price_trend == "DOWN" and second_half_delta_usdt > first_half_delta_usdt + self.delta_threshold_5m:
            notes = (f"[{symbol.upper()}] Price falling ({avg_price_1:.1f}->{avg_price_2:.1f}), "
                     f"but Delta rising (${first_half_delta_usdt:,.0f}->${second_half_delta_usdt:,.0f})")
            return "BULLISH_DIVERGENCE", notes

        return None, ""
