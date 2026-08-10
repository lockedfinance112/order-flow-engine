import numpy as np
from typing import List, Dict, Any, Tuple

class BlockBootstrap:
    """Performs deterministic daily block bootstrapping to estimate confidence intervals."""
    @staticmethod
    def bootstrap_metrics(
        joined_signals: List[Dict[str, Any]],
        horizon: str = "15m",
        reps: int = 1000,
        seed: int = 1729,
        cost_bps: int = 5
    ) -> Dict[str, Any]:
        
        valid_signals = [
            s for s in joined_signals
            if s.get("joined") and s.get("safe") and
            s.get("outcomes", {}).get(horizon, {}).get("status") == "COMPLETED"
        ]
        
        if not valid_signals:
            return {"status": "NO_DATA"}
            
        days = {}
        for s in valid_signals:
            day_ts = (s["timestamp_ms"] // 86400000) * 86400000
            days.setdefault(day_ts, []).append(s)
            
        day_keys = list(days.keys())
        num_days = len(day_keys)
        
        if num_days < 5:
            return {"status": "INSUFFICIENT_BLOCKS"}
            
        rng = np.random.default_rng(seed)
        
        bootstrap_means = []
        bootstrap_wins = []
        
        for _ in range(reps):
            resampled_keys = rng.choice(day_keys, size=num_days, replace=True)
            resampled_signals = []
            for k in resampled_keys:
                resampled_signals.extend(days[k])
                
            if not resampled_signals:
                continue
                
            rets = []
            wins = 0
            for s in resampled_signals:
                net_ret = s["outcomes"][horizon]["return"] - (cost_bps / 10000.0)
                rets.append(net_ret)
                if net_ret > 0:
                    wins += 1
                    
            bootstrap_means.append(np.mean(rets))
            bootstrap_wins.append(wins / len(resampled_signals))
            
        mean_low = float(np.percentile(bootstrap_means, 2.5))
        mean_high = float(np.percentile(bootstrap_means, 97.5))
        win_low = float(np.percentile(bootstrap_wins, 2.5))
        win_high = float(np.percentile(bootstrap_wins, 97.5))
        
        return {
            "status": "SUCCESS",
            "days_sampled": num_days,
            "mean_net_return_ci": (mean_low, mean_high),
            "win_rate_ci": (win_low, win_high)
        }

    @staticmethod
    def bootstrap_allow_minus_block(
        allow_signals: List[Dict[str, Any]],
        block_signals: List[Dict[str, Any]],
        horizon: str = "15m",
        reps: int = 1000,
        seed: int = 1729,
        cost_bps: int = 5
    ) -> Dict[str, Any]:
        """Calculates CI for difference: ALLOW mean net return - BLOCK mean net return."""
        valid_allow = [
            s for s in allow_signals
            if s.get("joined") and s.get("safe") and
            s.get("outcomes", {}).get(horizon, {}).get("status") == "COMPLETED"
        ]
        valid_block = [
            s for s in block_signals
            if s.get("joined") and s.get("safe") and
            s.get("outcomes", {}).get(horizon, {}).get("status") == "COMPLETED"
        ]
        
        if not valid_allow or not valid_block:
            return {"status": "NO_DATA"}
            
        days_allow = {}
        for s in valid_allow:
            day_ts = (s["timestamp_ms"] // 86400000) * 86400000
            days_allow.setdefault(day_ts, []).append(s)
            
        days_block = {}
        for s in valid_block:
            day_ts = (s["timestamp_ms"] // 86400000) * 86400000
            days_block.setdefault(day_ts, []).append(s)
            
        all_day_keys = list(set(days_allow.keys()) | set(days_block.keys()))
        num_days = len(all_day_keys)
        
        if num_days < 5:
            return {"status": "INSUFFICIENT_BLOCKS"}
            
        rng = np.random.default_rng(seed)
        diff_means = []
        
        for _ in range(reps):
            resampled_keys = rng.choice(all_day_keys, size=num_days, replace=True)
            res_allow_signals = []
            res_block_signals = []
            for k in resampled_keys:
                res_allow_signals.extend(days_allow.get(k, []))
                res_block_signals.extend(days_block.get(k, []))
                
            if not res_allow_signals or not res_block_signals:
                continue
                
            allow_rets = [s["outcomes"][horizon]["return"] - (cost_bps / 10000.0) for s in res_allow_signals]
            block_rets = [s["outcomes"][horizon]["return"] - (cost_bps / 10000.0) for s in res_block_signals]
            
            diff_means.append(np.mean(allow_rets) - np.mean(block_rets))
            
        diff_low = float(np.percentile(diff_means, 2.5))
        diff_high = float(np.percentile(diff_means, 97.5))
        
        return {
            "status": "SUCCESS",
            "allow_minus_block_ci": (diff_low, diff_high)
        }
