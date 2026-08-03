import time
import logging
from typing import Tuple, Dict, Any, List
from config import IMBALANCE_THRESHOLD

logger = logging.getLogger("OrderFlow.Scoring")

# Tuning constants
DELTA_BIAS_THRESHOLD_USDT = 100000.0
BOOK_IMBALANCE_THRESHOLD = 0.15
SWEEP_CONFLUENCE_THRESHOLD = 7
SWEEP_ACTIVE_SECONDS = 60.0

CONFLICT_ACTIVE_SECONDS = 300.0
BEARISH_CONFLICT_EVENTS = {
    "POSSIBLE_BEARISH_ABSORPTION",
    "CONFIRMED_BEARISH_ABSORPTION",
    "BEARISH_DIVERGENCE",
    "SELL_AGGRESSION"
}
BULLISH_CONFLICT_EVENTS = {
    "POSSIBLE_BULLISH_ABSORPTION",
    "CONFIRMED_BULLISH_ABSORPTION",
    "BULLISH_DIVERGENCE",
    "BUY_AGGRESSION"
}


class OrderFlowScorer:
    """
    Evaluates order flow confluences around a liquidity sweep for a specific symbol in USDT notional.
    Returns: (score, breakdown_details, status, imbalance_at_score, depth_snapshot_age_ms)
    """
    def __init__(self, metrics_engine):
        self.metrics = metrics_engine
        # Track the most recent sweep confluence evaluation per symbol:
        # {symbol_lower: {"direction": str, "score": int, "timestamp": float}}
        self.recent_confluence: Dict[str, dict] = {}

    def evaluate_sweep(self, symbol: str, sweep_type: str, sweep_level: float, 
                       recent_alerts: List[Dict[str, Any]], 
                       last_depth_timestamp: float) -> Tuple[int, Dict[str, Any], str, float, float]:
        """
        Scores a sweep (0 to 10) for the specified symbol.
        Uses USDT notional metrics and disables order book imbalance if depth data is stale (> 3000ms).
        """
        score = 0
        details = {
            "exhaustion": False,
            "absorption": False,
            "divergence": False,
            "imbalance": False,
            "details": []
        }

        # Normalize symbol
        symbol_lower = symbol.lower()
        state = self.metrics.get_state(symbol_lower)

        # Calculate depth snapshot age specifically for this symbol
        now = time.time()
        depth_age_ms = (now - state.last_depth_timestamp) * 1000.0 if state.last_depth_timestamp > 0 else 9999.0
        is_depth_stale = depth_age_ms > 3000.0

        # Retrieve window metrics for this symbol (USDT Notional values)
        m1m = self.metrics.get_metrics_for_window(symbol_lower, "1m")
        m5m = self.metrics.get_metrics_for_window(symbol_lower, "5m")
        m15m = self.metrics.get_metrics_for_window(symbol_lower, "15m")
        
        delta_1m_usdt = m1m.get("delta_usdt", 0.0)
        delta_5m_usdt = m5m.get("delta_usdt", 0.0)
        delta_15m_usdt = m15m.get("delta_usdt", 0.0)
        
        # Current book imbalance for this symbol
        imbalance = state.bid_ask_imbalance

        # Filter recent alerts to match this symbol only
        symbol_alerts = [a for a in recent_alerts if a.get("symbol", "").lower() == symbol_lower]

        event_lookback = 900  # 15 minutes

        # ----------------------------------------------------
        # 1. Evaluate Bullish Sweep (Swept Low -> Expect Buy Reversal)
        # ----------------------------------------------------
        if sweep_type.upper() == "BULLISH":
            # A. Refined Exhaustion check (USDT Notional thresholds):
            # 1. Did recent sell pressure exist? (15m delta <= -$250,000, or SELL_AGGRESSION is in history)
            had_sell_pressure = (delta_15m_usdt <= -250000.0) or any(
                a["type"] == "SELL_AGGRESSION" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            # 2. Is delta recovery visible? (5m delta >= -$50,000, or 1m delta is turning positive)
            delta_recovering = (delta_5m_usdt >= -50000.0) or (delta_1m_usdt > 0)
            
            if had_sell_pressure and delta_recovering:
                score += 2
                details["exhaustion"] = True
                details["details"].append("Seller Exhaustion (Attack -> Stabilization)")
            elif delta_recovering:
                # Award 1 point for partial stabilization without a clear prior attack
                score += 1
                details["details"].append("Partial Seller Exhaustion (Stabilizing delta)")

            # B. Absorption check
            has_absorption = any(
                a["type"] in ["BULLISH_ABSORPTION", "POSSIBLE_BULLISH_ABSORPTION", "CONFIRMED_BULLISH_ABSORPTION"] 
                and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_absorption:
                score += 3
                details["absorption"] = True
                details["details"].append("Recent Bullish Absorption")

            # C. Divergence check
            has_div = any(
                a["type"] == "BULLISH_DIVERGENCE" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_div:
                score += 2
                details["divergence"] = True
                details["details"].append("Recent Bullish Divergence")

            # D. Order Book Imbalance check (disabled if depth is stale)
            if is_depth_stale:
                details["details"].append("Order Book Imbalance (STALE DEPTH - Excluded)")
            elif imbalance >= IMBALANCE_THRESHOLD:
                score += 3
                details["imbalance"] = True
                details["details"].append(f"Bid Pressure Imbalance ({imbalance:+.2f} >= {IMBALANCE_THRESHOLD})")

        # ----------------------------------------------------
        # 2. Evaluate Bearish Sweep (Swept High -> Expect Sell Reversal)
        # ----------------------------------------------------
        elif sweep_type.upper() == "BEARISH":
            # A. Refined Exhaustion check (USDT Notional thresholds):
            # 1. Did recent buy pressure exist? (15m delta >= $250,000, or BUY_AGGRESSION is in history)
            had_buy_pressure = (delta_15m_usdt >= 250000.0) or any(
                a["type"] == "BUY_AGGRESSION" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            # 2. Is delta recovery visible? (5m delta <= $50,000, or 1m delta is turning negative)
            delta_recovering = (delta_5m_usdt <= 50000.0) or (delta_1m_usdt < 0)

            if had_buy_pressure and delta_recovering:
                score += 2
                details["exhaustion"] = True
                details["details"].append("Buyer Exhaustion (Attack -> Stabilization)")
            elif delta_recovering:
                score += 1
                details["details"].append("Partial Buyer Exhaustion (Stabilizing delta)")

            # B. Absorption check
            has_absorption = any(
                a["type"] in ["BEARISH_ABSORPTION", "POSSIBLE_BEARISH_ABSORPTION", "CONFIRMED_BEARISH_ABSORPTION"]
                and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_absorption:
                score += 3
                details["absorption"] = True
                details["details"].append("Recent Bearish Absorption")

            # C. Divergence check
            has_div = any(
                a["type"] == "BEARISH_DIVERGENCE" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_div:
                score += 2
                details["divergence"] = True
                details["details"].append("Recent Bearish Divergence")

            # D. Order Book Imbalance check (disabled if depth is stale)
            if is_depth_stale:
                details["details"].append("Order Book Imbalance (STALE DEPTH - Excluded)")
            elif imbalance <= -IMBALANCE_THRESHOLD:
                score += 3
                details["imbalance"] = True
                details["details"].append(f"Ask Pressure Imbalance ({imbalance:+.2f} <= -{IMBALANCE_THRESHOLD})")

        # Save this confluence result for next action / bias tracking
        self.recent_confluence[symbol_lower] = {
            "direction": sweep_type.upper(),
            "score": score,
            "timestamp": now
        }

        status = "SWEEP_CONFLUENCE" if score >= SWEEP_CONFLUENCE_THRESHOLD else "LOW_CONFLUENCE"
        return score, details, status, imbalance, depth_age_ms

    def get_bias_action(
        self,
        symbol: str,
        metrics_5m: dict,
        imbalance: float,
        recent_events: List[str]
    ) -> Tuple[str, str]:
        """
        Calculates the directional bias suggestion for a symbol based on
        active sweeps and live order flow volume notional dynamics.
        Applies conflict suppression to flow biases.
        """
        symbol_lower = symbol.lower()
        now = time.time()

        # 1. Confirmed active sweep has highest priority (expires after 60s)
        recent = self.recent_confluence.get(symbol_lower)
        if recent:
            age = now - recent.get("timestamp", 0.0)
            if age <= SWEEP_ACTIVE_SECONDS:
                score = recent.get("score", 0)
                if score >= SWEEP_CONFLUENCE_THRESHOLD:
                    direction = recent.get("direction", "")
                    if direction == "BULLISH":
                        return "LONG (SWEEP)", "NONE"
                    elif direction == "BEARISH":
                        return "SHORT (SWEEP)", "NONE"

        # 2. No active sweep -> Evaluate live flow bias
        delta_5m_usdt = metrics_5m.get("delta_usdt", 0.0)
        buy_ratio_5m = metrics_5m.get("buy_ratio", 0.5)

        # Buyers and Sellers aggression definitions based on 5m buy ratio
        is_buyers = buy_ratio_5m >= 0.58
        is_sellers = buy_ratio_5m <= 0.42

        candidate = "WAITING"
        if (
            delta_5m_usdt >= DELTA_BIAS_THRESHOLD_USDT
            and imbalance >= BOOK_IMBALANCE_THRESHOLD
            and is_buyers
        ):
            candidate = "LONG_BIAS"
        elif (
            delta_5m_usdt <= -DELTA_BIAS_THRESHOLD_USDT
            and imbalance <= -BOOK_IMBALANCE_THRESHOLD
            and is_sellers
        ):
            candidate = "SHORT_BIAS"

        # Apply suppression to biases using any matching recent events (last 300s)
        if candidate == "LONG_BIAS":
            bear_conflicts = [e for e in recent_events if e in BEARISH_CONFLICT_EVENTS]
            if bear_conflicts:
                return "WAITING", f"BEARISH_CONFLICT:{bear_conflicts[-1]}"
            return "LONG_BIAS", "NONE"
            
        if candidate == "SHORT_BIAS":
            bull_conflicts = [e for e in recent_events if e in BULLISH_CONFLICT_EVENTS]
            if bull_conflicts:
                return "WAITING", f"BULLISH_CONFLICT:{bull_conflicts[-1]}"
            return "SHORT_BIAS", "NONE"

        return "WAITING", "NONE"
