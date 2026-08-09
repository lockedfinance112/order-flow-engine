import logging

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
            
            # Evaluate on validation window
            selected_val = [s for s in val_sigs if hypothesis_filter(s)]
            # Simple outcomes
            val_returns = []
            for s in selected_val:
                if s.get("horizon_15m_status") == "CAPTURED":
                    val_returns.append(float(s.get("return_15m_pct", 0.0)))
                    
            win_rate = sum(1 for r in val_returns if r > 0) / len(val_returns) if val_returns else 0.0
            avg_ret = sum(val_returns) / len(val_returns) if val_returns else 0.0
            
            results.append({
                "train_end_index": current_idx,
                "validation_start": current_idx,
                "validation_end": current_idx + step_size,
                "candidate_n": len(selected_val),
                "win_rate": win_rate,
                "mean_return": avg_ret
            })
            
            current_idx += step_size
            
        return results
