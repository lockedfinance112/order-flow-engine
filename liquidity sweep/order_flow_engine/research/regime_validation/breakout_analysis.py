import math
from typing import List, Dict, Any
from regime.models import MarketBar

class BreakoutAnalyzer:
    """Evaluates breakout success, failure, extension, and return-to-range rates."""
    @staticmethod
    def analyze_breakouts(
        timeline: List[Dict[str, Any]],
        bars_1m: List[MarketBar]
    ) -> List[Dict[str, Any]]:
        results = []
        
        # Precompute timestamp mapping
        bar_map = {b.close_time_ms: (i, b) for i, b in enumerate(bars_1m)}
        
        for idx, t in enumerate(timeline):
            reg = t.get("primary_regime", "UNKNOWN")
            if reg not in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
                continue
                
            close_ms = t.get("latest_1m_close_time", 0)
            if close_ms not in bar_map:
                continue
                
            b_idx, bar = bar_map[close_ms]
            
            # Find ATR at breakout time
            from research.regime_validation.outcome_labels import calculate_atr14_at_t
            atr = calculate_atr14_at_t(bars_1m, b_idx)
            if atr <= 0.0:
                atr = 1.0
                
            # Track forward performance up to 15m
            if b_idx + 15 >= len(bars_1m):
                continue
                
            forward_bars = bars_1m[b_idx + 1 : b_idx + 16]
            closes = [fb.close for fb in forward_bars]
            max_p = max(fb.high for fb in forward_bars)
            min_p = min(fb.low for fb in forward_bars)
            
            # For breakout range estimation, let's assume entry_close
            entry_close = bar.close
            
            outcome = "AMBIGUOUS_BREAKOUT"
            if reg == "BREAKOUT_UP":
                final_move_atr = (closes[-1] - entry_close) / atr
                returned_to_range = any(fb.close <= entry_close - 0.50 * atr for fb in forward_bars[:5])
                
                if final_move_atr >= 0.50 and not returned_to_range:
                    outcome = "SUCCESSFUL_FOLLOW_THROUGH"
                elif returned_to_range and (closes[-1] - entry_close) <= 0:
                    outcome = "FAILED_BREAKOUT"
            else:
                final_move_atr = (entry_close - closes[-1]) / atr
                returned_to_range = any(fb.close >= entry_close + 0.50 * atr for fb in forward_bars[:5])
                
                if final_move_atr >= 0.50 and not returned_to_range:
                    outcome = "SUCCESSFUL_FOLLOW_THROUGH"
                elif returned_to_range and (entry_close - closes[-1]) <= 0:
                    outcome = "FAILED_BREAKOUT"
                    
            results.append({
                "symbol": t.get("symbol", ""),
                "timestamp_ms": close_ms,
                "regime": reg,
                "confidence": t.get("confidence", 0.0),
                "atr": atr,
                "outcome": outcome,
                "max_extension_atr": (max_p - entry_close) / atr if reg == "BREAKOUT_UP" else (entry_close - min_p) / atr,
                "max_retracement_atr": (entry_close - min_p) / atr if reg == "BREAKOUT_UP" else (max_p - entry_close) / atr,
            })
            
        return results
