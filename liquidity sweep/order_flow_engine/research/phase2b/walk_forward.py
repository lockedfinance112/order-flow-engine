import numpy as np
import logging
from research.phase2b.statistics import float_val, calculate_stats

logger = logging.getLogger("OrderFlow.Phase2b.WalkForward")

class WalkForwardValidator:
    def __init__(self, signals: list):
        self.signals = sorted(signals, key=lambda x: float(x.get("entry_time", 0.0)))

    def run_walk_forward(self, hypothesis_filter, min_train_size=100, step_size=25) -> list:
        total = len(self.signals)
        results = []
        
        if total < min_train_size + step_size:
            logger.info("Insufficient signals to execute walk-forward validation.")
            return results
            
        current_idx = min_train_size
        while current_idx + step_size <= total:
            train_sigs = self.signals[:current_idx]
            val_sigs = self.signals[current_idx:current_idx + step_size]
            
            # Filter and parse validation signals
            eligible_val = []
            selected_val = []
            
            for s in val_sigs:
                res = hypothesis_filter(s)
                if res is not None:
                    eligible_val.append(s)
                    if res is True:
                        selected_val.append(s)
            
            # Extract valid 15m returns
            val_returns = []
            for s in selected_val:
                if s.get("horizon_15m_status") == "CAPTURED":
                    val = float_val(s.get("return_15m_pct"))
                    if val is not None:
                        val_returns.append(val)
            
            stats = calculate_stats(val_returns)
            
            results.append({
                "train_end_index": current_idx,
                "validation_start": current_idx,
                "validation_end": current_idx + step_size,
                "eligible_n": len(eligible_val),
                "candidate_n": len(selected_val),
                "validation_n": len(val_sigs),
                "win_rate": stats["win_rate"],
                "mean_return": stats["avg_return"],
                "median_return": stats["median_return"],
                "expectancy": stats["expectancy"]
            })
            
            current_idx += step_size
            
        return results
