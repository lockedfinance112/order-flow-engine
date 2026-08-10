import math
from typing import List, Dict, Any
from regime.models import MarketBar

class BreakoutAnalyzer:
    """Evaluates breakout success, failure, extension, and return-to-range rates using Donchian 20 prior range."""
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
            
            # Require at least 20 bars of history before the breakout
            if b_idx < 20 or b_idx + 15 >= len(bars_1m):
                continue
                
            # Point-in-time prior range (Donchian 20)
            prior_bars = bars_1m[b_idx - 20 : b_idx]
            prior_range_high = max(b.high for b in prior_bars)
            prior_range_low = min(b.low for b in prior_bars)
            
            from research.regime_validation.outcome_labels import calculate_atr14_at_t
            atr = calculate_atr14_at_t(bars_1m, b_idx)
            if atr is None or atr <= 0.0:
                continue
                
            forward_bars = bars_1m[b_idx + 1 : b_idx + 61] # Fetch up to 60 bars for 30m and transitions
            closes = [fb.close for fb in forward_bars]
            max_p = max(fb.high for fb in forward_bars[:15])
            min_p = min(fb.low for fb in forward_bars[:15])
            
            entry_close = bar.close
            
            # Determine return inside prior range within 5 bars
            returned_to_range = False
            time_to_return = -1
            
            for offset, fb in enumerate(forward_bars[:5]):
                if reg == "BREAKOUT_UP" and fb.close <= prior_range_high:
                    returned_to_range = True
                    time_to_return = offset + 1
                    break
                elif reg == "BREAKOUT_DOWN" and fb.close >= prior_range_low:
                    returned_to_range = True
                    time_to_return = offset + 1
                    break
            
            outcome = "AMBIGUOUS_BREAKOUT"
            if reg == "BREAKOUT_UP":
                final_move_atr = (closes[14] - entry_close) / atr if len(closes) >= 15 else 0.0
                if final_move_atr >= 0.50 and not returned_to_range:
                    outcome = "SUCCESSFUL_FOLLOW_THROUGH"
                elif returned_to_range and (closes[14] - entry_close if len(closes) >= 15 else 0.0) <= 0:
                    outcome = "FAILED_BREAKOUT"
            else:
                final_move_atr = (entry_close - closes[14]) / atr if len(closes) >= 15 else 0.0
                if final_move_atr >= 0.50 and not returned_to_range:
                    outcome = "SUCCESSFUL_FOLLOW_THROUGH"
                elif returned_to_range and (entry_close - closes[14] if len(closes) >= 15 else 0.0) <= 0:
                    outcome = "FAILED_BREAKOUT"
                    
            # 30m return atr (Requirement 13)
            ret_30m_atr = 0.0
            if len(closes) >= 30:
                ret_30m = closes[29] - entry_close if reg == "BREAKOUT_UP" else entry_close - closes[29]
                ret_30m_atr = ret_30m / atr
            
            # Locate next different canonical regime (Requirement 13)
            next_reg = "UNKNOWN"
            time_to_next = -1
            for f_t in timeline[idx + 1 :]:
                if f_t.get("primary_regime") != reg:
                    next_reg = f_t.get("primary_regime", "UNKNOWN")
                    time_to_next = (f_t.get("latest_1m_close_time", 0) - close_ms) // 60000
                    break
                    
            results.append({
                "symbol": t.get("symbol", "").upper(),
                "timestamp_ms": close_ms,
                "regime": reg,
                "confidence": t.get("confidence", 0.0),
                "atr": atr,
                "outcome": outcome,
                "prior_range_high": prior_range_high,
                "prior_range_low": prior_range_low,
                "return_inside_range": returned_to_range,
                "time_to_return_inside_range": time_to_return,
                "max_extension_atr": (max_p - entry_close) / atr if reg == "BREAKOUT_UP" else (entry_close - min_p) / atr,
                "max_retracement_atr": (entry_close - min_p) / atr if reg == "BREAKOUT_UP" else (max_p - entry_close) / atr,
                "5m_return_atr": (closes[4] - entry_close) / atr if reg == "BREAKOUT_UP" and len(closes) >= 5 else (entry_close - closes[4]) / atr if len(closes) >= 5 else 0.0,
                "15m_return_atr": (closes[14] - entry_close) / atr if reg == "BREAKOUT_UP" and len(closes) >= 15 else (entry_close - closes[14]) / atr if len(closes) >= 15 else 0.0,
                "30m_return_atr": ret_30m_atr,
                "next_canonical_regime": next_reg,
                "time_to_next_regime": time_to_next
            })
            
        return results
