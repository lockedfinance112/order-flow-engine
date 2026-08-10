import math
from typing import List, Dict, Any, Optional, Tuple
from regime.models import MarketBar

def calculate_atr14_at_t(bars_1m: List[MarketBar], t_idx: int) -> float:
    """Calculates point-in-time ATR14 using 1m bars up to t_idx (inclusive)."""
    if t_idx < 14:
        return 1.0 # fallback default if insufficient history
        
    true_ranges = []
    for i in range(t_idx - 13, t_idx + 1):
        curr = bars_1m[i]
        prev = bars_1m[i - 1]
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close)
        )
        true_ranges.append(tr)
        
    return sum(true_ranges) / 14.0

class OutcomeLabeler:
    """Computes ex-post reference labels and forward metrics for validation runs."""
    @staticmethod
    def compute_ex_post_label(
        bars_1m: List[MarketBar],
        t_idx: int,
        horizon_min: int,
        thresholds: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        
        if t_idx + horizon_min >= len(bars_1m):
            return "UNLABELLED", {}
            
        anchor = bars_1m[t_idx]
        atr = calculate_atr14_at_t(bars_1m, t_idx)
        if atr <= 0.0:
            atr = 1.0

        exit_bar = bars_1m[t_idx + horizon_min]
        net_move = exit_bar.close - anchor.close
        net_move_atr = net_move / atr
        
        # Path length calculation
        path_len = 0.0
        prev_close = anchor.close
        for i in range(t_idx + 1, t_idx + horizon_min + 1):
            curr = bars_1m[i]
            path_len += abs(curr.close - prev_close)
            prev_close = curr.close
            
        efficiency = abs(net_move) / path_len if path_len > 0 else 0.0
        
        # Label classification
        up_th = thresholds.get("directional_atr", 1.0)
        eff_th = thresholds.get("efficiency", 0.35)
        r_th = thresholds.get("range_atr", 0.50)
        r_eff_th = thresholds.get("range_efficiency", 0.25)
        
        label = "AMBIGUOUS"
        if net_move_atr >= up_th and efficiency >= eff_th:
            label = "UP_DIRECTIONAL"
        elif net_move_atr <= -up_th and efficiency >= eff_th:
            label = "DOWN_DIRECTIONAL"
        elif abs(net_move_atr) <= r_th and efficiency <= r_eff_th:
            label = "RANGE"
            
        metrics = {
            "net_move_atr": net_move_atr,
            "path_length": path_len,
            "future_efficiency": efficiency,
            "anchor_price": anchor.close,
            "anchor_atr": atr
        }
        return label, metrics

    @staticmethod
    def compute_signal_outcomes(
        bars_1m: List[MarketBar],
        signal_time_ms: int,
        direction: str,
        horizon_min: int,
        entry_price: float
    ) -> Dict[str, Any]:
        """Calculates final return, MFE, MAE for a signal."""
        # Find entry index in 1m bars
        entry_idx = -1
        for i, b in enumerate(bars_1m):
            if b.open_time_ms <= signal_time_ms <= b.close_time_ms:
                entry_idx = i
                break
                
        if entry_idx == -1 or entry_idx + horizon_min >= len(bars_1m):
            return {
                "status": "CENSORED",
                "return": None,
                "mfe": None,
                "mae": None
            }
            
        future_segment = bars_1m[entry_idx + 1 : entry_idx + horizon_min + 1]
        exit_price = future_segment[-1].close
        
        # Calculate returns based on direction
        if direction.upper() == "LONG":
            ret = (exit_price - entry_price) / entry_price
            high_prices = [b.high for b in future_segment]
            low_prices = [b.low for b in future_segment]
            mfe = (max(high_prices) - entry_price) / entry_price
            mae = (entry_price - min(low_prices)) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
            high_prices = [b.high for b in future_segment]
            low_prices = [b.low for b in future_segment]
            mfe = (entry_price - min(low_prices)) / entry_price
            mae = (max(high_prices) - entry_price) / entry_price
            
        return {
            "status": "COMPLETED",
            "return": ret,
            "mfe": max(mfe, 0.0),
            "mae": max(mae, 0.0)
        }
