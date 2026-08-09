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
COOLDOWN_DURATION_SECONDS = 180.0

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
    State machine and scoring engine for Order Flow signals.
    Transitions: WARMING_UP, DATA_INVALID, DATA_STALE, COOLDOWN, WATCH_LONG,
    WATCH_SHORT, CONFIRMED_LONG, CONFIRMED_SHORT, INVALIDATED, WAITING.
    """
    def __init__(self, metrics_engine):
        self.metrics = metrics_engine
        # Track the most recent sweep confluence evaluation per symbol:
        # {symbol_lower: {"direction": str, "score": int, "timestamp": float}}
        self.recent_confluence: Dict[str, dict] = {}
        # Stores last evaluated check gates per symbol for auditable API access
        self.symbol_gates: Dict[str, dict] = {}
        # Cooldown end timestamps
        self.cooldowns: Dict[str, float] = {}

    def evaluate_sweep(self, symbol: str, sweep_type: str, sweep_level: float, 
                       recent_alerts: List[Dict[str, Any]], 
                       last_depth_timestamp: float) -> Tuple[int, Dict[str, Any], str, float, float]:
        """
        Scores a sweep (0 to 10) for the specified symbol.
        """
        score = 0
        details = {
            "exhaustion": False,
            "absorption": False,
            "divergence": False,
            "imbalance": False,
            "details": []
        }

        symbol_lower = symbol.lower()
        state = self.metrics.get_state(symbol_lower)

        now = time.time()
        depth_age_ms = (now - state.last_depth_timestamp) * 1000.0 if state.last_depth_timestamp > 0 else 9999.0
        is_depth_stale = depth_age_ms > 3000.0

        m1m = self.metrics.get_metrics_for_window(symbol_lower, "1m")
        m5m = self.metrics.get_metrics_for_window(symbol_lower, "5m")
        m15m = self.metrics.get_metrics_for_window(symbol_lower, "15m")
        
        delta_1m_usdt = m1m.get("delta_usdt", 0.0)
        delta_5m_usdt = m5m.get("delta_usdt", 0.0)
        delta_15m_usdt = m15m.get("delta_usdt", 0.0)
        
        imbalance = state.bid_ask_imbalance
        symbol_alerts = [a for a in recent_alerts if a.get("symbol", "").lower() == symbol_lower]
        event_lookback = 900  # 15 minutes

        if sweep_type.upper() == "BULLISH":
            had_sell_pressure = (delta_15m_usdt <= -250000.0) or any(
                a["type"] == "SELL_AGGRESSION" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            delta_recovering = (delta_5m_usdt >= -50000.0) or (delta_1m_usdt > 0)
            
            if had_sell_pressure and delta_recovering:
                score += 2
                details["exhaustion"] = True
                details["details"].append("Seller Exhaustion (Attack -> Stabilization)")
            elif delta_recovering:
                score += 1
                details["details"].append("Partial Seller Exhaustion (Stabilizing delta)")

            has_absorption = any(
                a["type"] in ["BULLISH_ABSORPTION", "POSSIBLE_BULLISH_ABSORPTION", "CONFIRMED_BULLISH_ABSORPTION"] 
                and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_absorption:
                score += 3
                details["absorption"] = True
                details["details"].append("Recent Bullish Absorption")

            has_div = any(
                a["type"] == "BULLISH_DIVERGENCE" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_div:
                score += 2
                details["divergence"] = True
                details["details"].append("Recent Bullish Divergence")

            if is_depth_stale:
                details["details"].append("Order Book Imbalance (STALE DEPTH - Excluded)")
            elif imbalance >= IMBALANCE_THRESHOLD:
                score += 3
                details["imbalance"] = True
                details["details"].append(f"Bid Pressure Imbalance ({imbalance:+.2f} >= {IMBALANCE_THRESHOLD})")

        elif sweep_type.upper() == "BEARISH":
            had_buy_pressure = (delta_15m_usdt >= 250000.0) or any(
                a["type"] == "BUY_AGGRESSION" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            delta_recovering = (delta_5m_usdt <= 50000.0) or (delta_1m_usdt < 0)

            if had_buy_pressure and delta_recovering:
                score += 2
                details["exhaustion"] = True
                details["details"].append("Buyer Exhaustion (Attack -> Stabilization)")
            elif delta_recovering:
                score += 1
                details["details"].append("Partial Buyer Exhaustion (Stabilizing delta)")

            has_absorption = any(
                a["type"] in ["BEARISH_ABSORPTION", "POSSIBLE_BEARISH_ABSORPTION", "CONFIRMED_BEARISH_ABSORPTION"]
                and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_absorption:
                score += 3
                details["absorption"] = True
                details["details"].append("Recent Bearish Absorption")

            has_div = any(
                a["type"] == "BEARISH_DIVERGENCE" and (now - a["timestamp"]) <= event_lookback
                for a in symbol_alerts
            )
            if has_div:
                score += 2
                details["divergence"] = True
                details["details"].append("Recent Bearish Divergence")

            if is_depth_stale:
                details["details"].append("Order Book Imbalance (STALE DEPTH - Excluded)")
            elif imbalance <= -IMBALANCE_THRESHOLD:
                score += 3
                details["imbalance"] = True
                details["details"].append(f"Ask Pressure Imbalance ({imbalance:+.2f} <= -{IMBALANCE_THRESHOLD})")

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
        Runs the auditable state machine for the symbol.
        Returns: (state_name, reason_or_failed_gates)
        """
        symbol_lower = symbol.lower()
        state = self.metrics.get_state(symbol_lower)
        now = time.time()

        # Gate 1: Check data validity
        if not state.local_book.is_valid:
            self.symbol_gates[symbol_lower] = {"book_synced": "FAIL"}
            return "DATA_INVALID", "Book sync invalid or sequence gap active"

        # Gate 2: Check freshness (3000ms stale limit)
        depth_age_ms = (now - state.last_depth_timestamp) * 1000.0 if state.last_depth_timestamp > 0 else 9999.0
        if depth_age_ms > 3000.0:
            self.symbol_gates[symbol_lower] = {"freshness": "FAIL"}
            return "DATA_STALE", f"Depth stale (age: {depth_age_ms:.0f}ms)"

        # Gate 3: Check window warming
        m1m = self.metrics.get_metrics_for_window(symbol_lower, "1m")
        m15m = self.metrics.get_metrics_for_window(symbol_lower, "15m")
        if (
            metrics_5m.get("status") == "WARMING_UP"
            or m1m.get("status") == "WARMING_UP"
            or m15m.get("status") == "WARMING_UP"
        ):
            self.symbol_gates[symbol_lower] = {"warmup_complete": "FAIL"}
            return "WARMING_UP", "Rolling window is warming up"

        # Gate 4: Check active cooldown
        cooldown_end = self.cooldowns.get(symbol_lower, 0.0)
        if now < cooldown_end:
            remaining = cooldown_end - now
            self.symbol_gates[symbol_lower] = {"cooldown_inactive": "FAIL"}
            return "COOLDOWN", f"Signal cooldown active ({remaining:.1f}s remaining)"

        # Prepare parameters for evaluation
        delta_1m = m1m.get("delta_usdt", 0.0)
        delta_5m = metrics_5m.get("delta_usdt", 0.0)
        buy_ratio_5m = metrics_5m.get("buy_ratio", 0.5)

        # Get stacking/pulling shift over 5s
        mid_price = (state.best_bid + state.best_ask) / 2.0
        sp_changes = state.stacking_pulling.get_changes(5.0, mid_price)
        net_shift = sp_changes["net_shift"]

        # Evaluate active sweep (active for 60 seconds)
        recent_sweep = self.recent_confluence.get(symbol_lower)
        has_active_sweep = False
        sweep_dir = ""
        if recent_sweep and (now - recent_sweep["timestamp"]) <= SWEEP_ACTIVE_SECONDS:
            if recent_sweep["score"] >= SWEEP_CONFLUENCE_THRESHOLD:
                has_active_sweep = True
                sweep_dir = recent_sweep["direction"]

        # Define check gates for LONG
        long_gates = {
            "1m_delta_positive": "PASS" if delta_1m > 0 else "FAIL",
            "5m_delta_bias": "PASS" if (delta_5m >= DELTA_BIAS_THRESHOLD_USDT or (has_active_sweep and sweep_dir == "BULLISH")) else "FAIL",
            "buy_aggression": "PASS" if buy_ratio_5m >= 0.58 else "FAIL",
            "imbalance_bullish": "PASS" if imbalance >= BOOK_IMBALANCE_THRESHOLD else "FAIL",
            "stacking_bullish": "PASS" if net_shift > 0 else "FAIL",
            "no_bearish_conflict": "PASS" if not any(e in BEARISH_CONFLICT_EVENTS for e in recent_events) else "FAIL"
        }

        # Define check gates for SHORT
        short_gates = {
            "1m_delta_negative": "PASS" if delta_1m < 0 else "FAIL",
            "5m_delta_bias": "PASS" if (delta_5m <= -DELTA_BIAS_THRESHOLD_USDT or (has_active_sweep and sweep_dir == "BEARISH")) else "FAIL",
            "sell_aggression": "PASS" if buy_ratio_5m <= 0.42 else "FAIL",
            "imbalance_bearish": "PASS" if imbalance <= -BOOK_IMBALANCE_THRESHOLD else "FAIL",
            "stacking_bearish": "PASS" if net_shift < 0 else "FAIL",
            "no_bullish_conflict": "PASS" if not any(e in BULLISH_CONFLICT_EVENTS for e in recent_events) else "FAIL"
        }

        # 1. Evaluate CONFIRMED LONG
        if has_active_sweep and sweep_dir == "BULLISH":
            # Gated checks with sweep priority
            if long_gates["no_bearish_conflict"] == "PASS" and long_gates["1m_delta_positive"] == "PASS":
                self.symbol_gates[symbol_lower] = long_gates
                self.cooldowns[symbol_lower] = now + COOLDOWN_DURATION_SECONDS
                return "CONFIRMED_LONG", "Bullish sweep confluence confirmed"
            else:
                self.symbol_gates[symbol_lower] = long_gates
                return "WATCH_LONG", "Bullish sweep waiting for confirmation gates"

        if (
            long_gates["5m_delta_bias"] == "PASS"
            and long_gates["buy_aggression"] == "PASS"
            and long_gates["imbalance_bullish"] == "PASS"
        ):
            if long_gates["no_bearish_conflict"] == "FAIL":
                self.symbol_gates[symbol_lower] = long_gates
                return "WAITING", f"LONG_SUPPRESSED:Bearish conflict active"
            
            # If stack/pull is also positive, CONFIRM. Else, WATCH.
            if long_gates["stacking_bullish"] == "PASS" and long_gates["1m_delta_positive"] == "PASS":
                self.symbol_gates[symbol_lower] = long_gates
                self.cooldowns[symbol_lower] = now + COOLDOWN_DURATION_SECONDS
                return "CONFIRMED_LONG", "Bullish order flow bias confirmed"
            else:
                self.symbol_gates[symbol_lower] = long_gates
                return "WATCH_LONG", "Flow bias watching stacking/pulling details"

        # 2. Evaluate CONFIRMED SHORT
        if has_active_sweep and sweep_dir == "BEARISH":
            if short_gates["no_bullish_conflict"] == "PASS" and short_gates["1m_delta_negative"] == "PASS":
                self.symbol_gates[symbol_lower] = short_gates
                self.cooldowns[symbol_lower] = now + COOLDOWN_DURATION_SECONDS
                return "CONFIRMED_SHORT", "Bearish sweep confluence confirmed"
            else:
                self.symbol_gates[symbol_lower] = short_gates
                return "WATCH_SHORT", "Bearish sweep waiting for confirmation gates"

        if (
            short_gates["5m_delta_bias"] == "PASS"
            and short_gates["sell_aggression"] == "PASS"
            and short_gates["imbalance_bearish"] == "PASS"
        ):
            if short_gates["no_bullish_conflict"] == "FAIL":
                self.symbol_gates[symbol_lower] = short_gates
                return "WAITING", f"SHORT_SUPPRESSED:Bullish conflict active"

            if short_gates["stacking_bearish"] == "PASS" and short_gates["1m_delta_negative"] == "PASS":
                self.symbol_gates[symbol_lower] = short_gates
                self.cooldowns[symbol_lower] = now + COOLDOWN_DURATION_SECONDS
                return "CONFIRMED_SHORT", "Bearish order flow bias confirmed"
            else:
                self.symbol_gates[symbol_lower] = short_gates
                return "WATCH_SHORT", "Flow bias watching stacking/pulling details"

        # 3. Handle WATCH triggers
        # If delta and aggression match but book imbalance or stacking is missing
        if long_gates["5m_delta_bias"] == "PASS" or (delta_5m > 50000.0 and buy_ratio_5m >= 0.55):
            self.symbol_gates[symbol_lower] = long_gates
            return "WATCH_LONG", "Delta/Aggression turning bullish"

        if short_gates["5m_delta_bias"] == "PASS" or (delta_5m < -50000.0 and buy_ratio_5m <= 0.45):
            self.symbol_gates[symbol_lower] = short_gates
            return "WATCH_SHORT", "Delta/Aggression turning bearish"

        # Save default gates
        self.symbol_gates[symbol_lower] = {**long_gates, **short_gates}
        return "WAITING", "No active signal triggers"
