import copy
import numpy as np
from research.phase2b.statistics import calculate_stats, float_val

class CounterfactualEngine:
    def __init__(self, baseline_signals: list):
        self.baseline_signals = copy.deepcopy(baseline_signals)

    def evaluate_hypothesis(self, filter_func) -> dict:
        total_baseline = len(self.baseline_signals)
        
        # Partition signals into eligible, selected, rejected, missing
        eligible_signals = []
        missing_count = 0
        selected = []
        rejected = []
        
        for s in self.baseline_signals:
            res = filter_func(s)
            if res is None:
                missing_count += 1
            else:
                eligible_signals.append(s)
                if res is True:
                    selected.append(s)
                else:
                    rejected.append(s)
                    
        eligible_count = len(eligible_signals)
        selected_count = len(selected)
        rejected_count = len(rejected)
        
        retention_total = (selected_count / total_baseline) if total_baseline > 0 else 0.0
        retention_eligible = (selected_count / eligible_count) if eligible_count > 0 else 0.0
        
        # Compute outcomes helper
        def get_cohort_stats(sigs, baseline_sigs):
            horizons = ["1m", "3m", "5m", "15m"]
            out = {}
            for h in horizons:
                b_ret, b_mfe, b_mae = [], [], []
                c_ret, c_mfe, c_mae = [], [], []
                
                # Baseline outcomes for horizon
                for s in baseline_sigs:
                    if s.get(f"horizon_{h}_status") == "CAPTURED":
                        val = float_val(s.get(f"return_{h}_pct"))
                        if val is not None: b_ret.append(val)
                    if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE":
                        f = float_val(s.get("max_favorable_pct"))
                        a = float_val(s.get("max_adverse_pct"))
                        if f is not None: b_mfe.append(f)
                        if a is not None: b_mae.append(a)
                        
                # Candidate outcomes for horizon
                for s in sigs:
                    if s.get(f"horizon_{h}_status") == "CAPTURED":
                        val = float_val(s.get(f"return_{h}_pct"))
                        if val is not None: c_ret.append(val)
                    if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE":
                        f = float_val(s.get("max_favorable_pct"))
                        a = float_val(s.get("max_adverse_pct"))
                        if f is not None: c_mfe.append(f)
                        if a is not None: c_mae.append(a)
                        
                out[h] = {
                    "baseline": calculate_stats(b_ret, b_mfe, b_mae),
                    "candidate": calculate_stats(c_ret, c_mfe, c_mae)
                }
            return out

        # Compute Directional cohorts
        overall_metrics = get_cohort_stats(selected, self.baseline_signals)
        
        b_long = [s for s in self.baseline_signals if s.get("direction") == "LONG"]
        c_long = [s for s in selected if s.get("direction") == "LONG"]
        long_metrics = get_cohort_stats(c_long, b_long)
        
        b_short = [s for s in self.baseline_signals if s.get("direction") == "SHORT"]
        c_short = [s for s in selected if s.get("direction") == "SHORT"]
        short_metrics = get_cohort_stats(c_short, b_short)

        # Removed Signal Quality metrics
        removed_winners = 0
        removed_losers = 0
        sum_return_removed_winners = 0.0
        sum_return_removed_losers = 0.0
        removed_returns = []
        
        for s in rejected:
            if s.get("horizon_15m_status") == "CAPTURED":
                val = float_val(s.get("return_15m_pct"))
                if val is not None:
                    removed_returns.append(val)
                    if val > 0:
                        removed_winners += 1
                        sum_return_removed_winners += val
                    elif val < 0:
                        removed_losers += 1
                        sum_return_removed_losers += val
                        
        mean_return_removed = float(np.mean(removed_returns)) if removed_returns else 0.0
        median_return_removed = float(np.median(removed_returns)) if removed_returns else 0.0

        # Symbol breakdown stats
        symbols_present = set(s.get("symbol", "").upper() for s in self.baseline_signals if s.get("symbol"))
        by_symbol = {}
        for sym in symbols_present:
            b_sym = [s for s in self.baseline_signals if s.get("symbol", "").upper() == sym]
            c_sym = [s for s in selected if s.get("symbol", "").upper() == sym]
            
            sym_total = len(b_sym)
            sym_candidate = len(c_sym)
            sym_retention = (sym_candidate / sym_total) if sym_total > 0 else 0.0
            
            sym_b_ret = [float_val(s.get("return_15m_pct")) for s in b_sym if s.get("horizon_15m_status") == "CAPTURED" and float_val(s.get("return_15m_pct")) is not None]
            sym_c_ret = [float_val(s.get("return_15m_pct")) for s in c_sym if s.get("horizon_15m_status") == "CAPTURED" and float_val(s.get("return_15m_pct")) is not None]
            
            b_stat = calculate_stats(sym_b_ret)
            c_stat = calculate_stats(sym_c_ret)
            
            by_symbol[sym] = {
                "count": sym_total,
                "candidate_count": sym_candidate,
                "retention_pct": sym_retention,
                "baseline_expectancy": b_stat["expectancy"],
                "candidate_expectancy": c_stat["expectancy"],
                "expectancy_change": c_stat["expectancy"] - b_stat["expectancy"],
                "median_return_change": c_stat["median_return"] - b_stat["median_return"]
            }

        return {
            "total_baseline": total_baseline,
            "eligible_count": eligible_count,
            "missing_feature_count": missing_count,
            "selected_count": selected_count,
            "rejected_count": rejected_count,
            "retention_pct": retention_total,
            "retention_eligible_pct": retention_eligible,
            
            # Horizon + Direction splits
            "overall": overall_metrics,
            "long": long_metrics,
            "short": short_metrics,
            
            # Removed Quality metrics
            "winners_sacrificed": removed_winners,
            "losers_avoided": removed_losers,
            "sum_return_removed_winners": sum_return_removed_winners,
            "sum_return_removed_losers": sum_return_removed_losers,
            "mean_return_removed": mean_return_removed,
            "median_return_removed": median_return_removed,
            
            # Symbol breakdown
            "by_symbol": by_symbol
        }
