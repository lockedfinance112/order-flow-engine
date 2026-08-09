import copy
from research.phase2b.statistics import calculate_stats, float_val

class CounterfactualEngine:
    def __init__(self, baseline_signals: list):
        self.baseline_signals = copy.deepcopy(baseline_signals)

    def evaluate_hypothesis(self, filter_func) -> dict:
        # Filter signals
        selected = [s for s in self.baseline_signals if filter_func(s)]
        rejected = [s for s in self.baseline_signals if not filter_func(s)]
        
        baseline_cnt = len(self.baseline_signals)
        candidate_cnt = len(selected)
        removed_cnt = len(rejected)
        retention_pct = (candidate_cnt / baseline_cnt) if baseline_cnt > 0 else 0.0
        
        # Extract returns/excursions
        def get_outcomes(sigs):
            returns_15m = []
            mfes, maes = [], []
            for s in sigs:
                if s.get("horizon_15m_status") == "CAPTURED":
                    val = float_val(s.get("return_15m_pct"))
                    if val is not None:
                        returns_15m.append(val)
                if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE":
                    f = float_val(s.get("max_favorable_pct"))
                    a = float_val(s.get("max_adverse_pct"))
                    if f is not None: mfes.append(f)
                    if a is not None: maes.append(a)
            return returns_15m, mfes, maes

        b_returns, b_mfes, b_maes = get_outcomes(self.baseline_signals)
        c_returns, c_mfes, c_maes = get_outcomes(selected)
        
        baseline_stats = calculate_stats(b_returns, b_mfes, b_maes)
        candidate_stats = calculate_stats(c_returns, c_mfes, c_maes)
        
        # Calculate winners sacrificed vs losers avoided
        winners_sacrificed = 0
        losers_avoided = 0
        for s in rejected:
            if s.get("horizon_15m_status") == "CAPTURED":
                val = float_val(s.get("return_15m_pct"))
                if val is not None:
                    if val > 0:
                        winners_sacrificed += 1
                    elif val < 0:
                        losers_avoided += 1
                        
        return {
            "baseline_count": baseline_cnt,
            "candidate_count": candidate_cnt,
            "removed_count": removed_cnt,
            "retention_pct": retention_pct,
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "winners_sacrificed": winners_sacrificed,
            "losers_avoided": losers_avoided
        }
