import math
from typing import List, Dict, Any, Tuple

class StabilityAnalyzer:
    """Analyzes regime stability, transition patterns, and flip-flop rates."""
    @staticmethod
    def analyze_occupancy(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
        if not timeline:
            return {}
        counts = {}
        for t in timeline:
            r = t.get("primary_regime", "UNKNOWN")
            counts[r] = counts.get(r, 0) + 1
        total = len(timeline)
        return {r: c / total for r, c in counts.items()}

    @staticmethod
    def analyze_persistence(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates persistence statistics (median, mean, P90, max etc.) per regime."""
        if not timeline:
            return {}
            
        durations = {} # regime -> list of block sizes
        current_regime = None
        duration = 0
        
        for t in timeline:
            r = t.get("primary_regime", "UNKNOWN")
            if current_regime is None:
                current_regime = r
                duration = 1
            elif r == current_regime:
                duration += 1
            else:
                durations.setdefault(current_regime, []).append(duration)
                current_regime = r
                duration = 1
        if current_regime is not None:
            durations.setdefault(current_regime, []).append(duration)
            
        summary = {}
        for r, vals in durations.items():
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            if n == 0:
                continue
            mean_val = sum(sorted_vals) / n
            median_val = sorted_vals[n // 2]
            p25 = sorted_vals[int(n * 0.25)]
            p75 = sorted_vals[int(n * 0.75)]
            p90 = sorted_vals[int(n * 0.90)] if n > 1 else sorted_vals[0]
            max_val = max(sorted_vals)
            
            summary[r] = {
                "blocks": n,
                "mean": mean_val,
                "median": median_val,
                "p25": p25,
                "p75": p75,
                "p90": p90,
                "max": max_val
            }
        return summary

    @staticmethod
    def analyze_transitions(timeline: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Builds counts and transition matrix between regimes (excluding repeated same-regime rows)."""
        counts = {}
        matrix = {}
        prev = None
        
        for t in timeline:
            r = t.get("primary_regime", "UNKNOWN")
            if prev is not None and r != prev:
                counts[prev] = counts.get(prev, 0) + 1
                matrix.setdefault(prev, {})
                matrix[prev][r] = matrix[prev].get(r, 0) + 1
            prev = r
        return counts, matrix

    @staticmethod
    def analyze_flip_flops(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates 3, 5, and 10 bar flip-flop rates (A -> B -> A)."""
        if len(timeline) < 3:
            return {"flip_3": 0.0, "flip_5": 0.0, "flip_10": 0.0}
            
        flips_3 = 0
        flips_5 = 0
        flips_10 = 0
        total_eval = len(timeline)
        
        for i in range(total_eval):
            r_curr = timeline[i].get("primary_regime", "UNKNOWN")
            
            # 3-bar flip: check if current matches state i+2 and i+1 is different
            if i + 2 < total_eval:
                r_2 = timeline[i+2].get("primary_regime", "UNKNOWN")
                r_1 = timeline[i+1].get("primary_regime", "UNKNOWN")
                if r_curr == r_2 and r_curr != r_1:
                    flips_3 += 1
                    
            # 5-bar flip: check if state i matches i+4 and any intermediate is different
            if i + 4 < total_eval:
                r_4 = timeline[i+4].get("primary_regime", "UNKNOWN")
                if r_curr == r_4 and any(timeline[i+j].get("primary_regime", "UNKNOWN") != r_curr for j in range(1, 4)):
                    flips_5 += 1
                    
            # 10-bar flip: check if state i matches i+9 and any intermediate is different
            if i + 9 < total_eval:
                r_9 = timeline[i+9].get("primary_regime", "UNKNOWN")
                if r_curr == r_9 and any(timeline[i+j].get("primary_regime", "UNKNOWN") != r_curr for j in range(1, 9)):
                    flips_10 += 1
                    
        return {
            "flip_3": flips_3 / max(total_eval - 2, 1),
            "flip_5": flips_5 / max(total_eval - 4, 1),
            "flip_10": flips_10 / max(total_eval - 9, 1)
        }
