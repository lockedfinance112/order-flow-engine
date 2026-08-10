import numpy as np
from typing import List, Dict, Any, Optional

class ExpectancyCalculator:
    """Calculates signal expectancy, win rates, profit factor proxies, and cost adjustments."""
    @staticmethod
    def calculate_net_returns(
        gross_return: float,
        cost_bps: int
    ) -> float:
        # Cost is subtracted symmetrically from already direction-normalized return
        return gross_return - (cost_bps / 10000.0)

    @classmethod
    def calculate_expectancy(
        cls,
        joined_signals: List[Dict[str, Any]],
        horizon_min: int,
        cost_bps: int = 5
    ) -> Dict[str, Any]:
        
        valid_signals = [
            s for s in joined_signals
            if s.get("joined") and s.get("safe") and
            s.get("outcomes", {}).get("status") == "COMPLETED"
        ]
        
        if not valid_signals:
            return {
                "sample_count": 0,
                "win_rate": 0.0,
                "mean_return": 0.0,
                "median_return": 0.0,
                "return_std": 0.0,
                "p25_return": 0.0,
                "p75_return": 0.0,
                "average_MFE": 0.0,
                "median_MFE": 0.0,
                "average_MAE": 0.0,
                "median_MAE": 0.0,
                "forward_return_profit_factor_proxy": 0.0
            }
            
        returns = []
        mfes = []
        maes = []
        
        pos_sum = 0.0
        neg_sum = 0.0
        wins = 0
        
        for s in valid_signals:
            outcomes = s["outcomes"]
            gross_ret = outcomes["return"]
            net_ret = cls.calculate_net_returns(gross_ret, cost_bps)
            returns.append(net_ret)
            
            mfes.append(outcomes["mfe"])
            maes.append(outcomes["mae"])
            
            if net_ret > 0:
                wins += 1
                pos_sum += net_ret
            else:
                neg_sum += abs(net_ret)
                
        returns = np.array(returns)
        mfes = np.array(mfes)
        maes = np.array(maes)
        
        pf_proxy = pos_sum / neg_sum if neg_sum > 0 else 999.0
        
        return {
            "sample_count": len(valid_signals),
            "win_rate": wins / len(valid_signals),
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "return_std": float(np.std(returns)),
            "p25_return": float(np.percentile(returns, 25)),
            "p75_return": float(np.percentile(returns, 75)),
            "average_MFE": float(np.mean(mfes)),
            "median_MFE": float(np.median(mfes)),
            "average_MAE": float(np.mean(maes)),
            "median_MAE": float(np.median(maes)),
            "forward_return_profit_factor_proxy": pf_proxy
        }
