import math
import csv
from typing import List, Dict, Any, Tuple


class StabilityAnalyzer:
    """Analyzes regime stability, transition patterns, and flip-flop rates.

    IMPORTANT: All temporal analysis (persistence, transitions, flip-flops,
    candidate episodes) MUST be run per-symbol and then explicitly aggregated.
    Never concatenate multi-symbol timelines before running temporal analysis —
    symbol boundaries create artificial regime transitions that contaminate all
    derived metrics.
    """

    # ------------------------------------------------------------------ #
    #  Per-row / pooled metrics (symbol-boundary-safe)                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def analyze_occupancy(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
        """Row-count occupancy. Safe to pool across symbols (labelled MICRO)."""
        if not timeline:
            return {}
        counts: Dict[str, int] = {}
        for t in timeline:
            r = t.get("primary_regime", "UNKNOWN")
            counts[r] = counts.get(r, 0) + 1
        total = len(timeline)
        return {r: c / total for r, c in counts.items()}

    # ------------------------------------------------------------------ #
    #  Single-symbol temporal helpers                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _persistence_episodes(timeline: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Return completed episode lengths per regime for ONE symbol timeline."""
        durations: Dict[str, List[int]] = {}
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
        return durations

    @staticmethod
    def _flip_flop_numerators(timeline: List[Dict[str, Any]]) -> Tuple[int, int, int, int, int, int]:
        """Return (flip3_num, flip3_den, flip5_num, flip5_den, flip10_num, flip10_den)
        for ONE symbol timeline."""
        n = len(timeline)
        flips_3 = flips_5 = flips_10 = 0
        for i in range(n):
            r = timeline[i].get("primary_regime", "UNKNOWN")
            if i + 2 < n:
                r2 = timeline[i + 2].get("primary_regime", "UNKNOWN")
                r1 = timeline[i + 1].get("primary_regime", "UNKNOWN")
                if r == r2 and r != r1:
                    flips_3 += 1
            if i + 4 < n:
                r4 = timeline[i + 4].get("primary_regime", "UNKNOWN")
                if r == r4 and any(timeline[i + j].get("primary_regime", "UNKNOWN") != r for j in range(1, 4)):
                    flips_5 += 1
            if i + 9 < n:
                r9 = timeline[i + 9].get("primary_regime", "UNKNOWN")
                if r == r9 and any(timeline[i + j].get("primary_regime", "UNKNOWN") != r for j in range(1, 9)):
                    flips_10 += 1
        return (flips_3, max(n - 2, 1),
                flips_5, max(n - 4, 1),
                flips_10, max(n - 9, 1))

    @staticmethod
    def _transition_counts(timeline: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Return (counts, matrix) for ONE symbol timeline."""
        counts: Dict[str, int] = {}
        matrix: Dict[str, Dict[str, int]] = {}
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
    def _candidate_episodes(timeline: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Track candidate episodes sequentially for ONE symbol timeline."""
        results: Dict[str, Dict[str, Any]] = {}
        current_candidate = None
        for t in timeline:
            cand = t.get("candidate_regime")
            if cand in (None, "UNKNOWN", ""):
                cand = None
            prim = t.get("primary_regime")

            if current_candidate is not None:
                if prim == current_candidate:
                    results.setdefault(current_candidate, {"episodes": 0, "confirmed": 0, "reset": 0})
                    results[current_candidate]["confirmed"] += 1
                    current_candidate = None
                elif cand != current_candidate:
                    results.setdefault(current_candidate, {"episodes": 0, "confirmed": 0, "reset": 0})
                    results[current_candidate]["reset"] += 1
                    current_candidate = None

            if current_candidate is None and cand is not None:
                current_candidate = cand
                results.setdefault(cand, {"episodes": 0, "confirmed": 0, "reset": 0})
                results[cand]["episodes"] += 1

        # Any open episode at end of timeline counts as reset
        if current_candidate is not None:
            results.setdefault(current_candidate, {"episodes": 0, "confirmed": 0, "reset": 0})
            results[current_candidate]["reset"] += 1

        return results

    # ------------------------------------------------------------------ #
    #  Aggregating wrappers — accept per_symbol_timelines dict            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def analyze_persistence(
        per_symbol_timelines: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Aggregate persistence statistics across symbols without cross-symbol contamination.

        Args:
            per_symbol_timelines: {symbol_lower: timeline_list}
        """
        all_episodes: Dict[str, List[int]] = {}
        for tl in per_symbol_timelines.values():
            eps = StabilityAnalyzer._persistence_episodes(tl)
            for regime, lengths in eps.items():
                all_episodes.setdefault(regime, []).extend(lengths)

        summary: Dict[str, Any] = {}
        for r, vals in all_episodes.items():
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            if n == 0:
                continue
            summary[r] = {
                "blocks": n,
                "mean": sum(sorted_vals) / n,
                "median": sorted_vals[n // 2],
                "p25": sorted_vals[int(n * 0.25)],
                "p75": sorted_vals[int(n * 0.75)],
                "p90": sorted_vals[int(n * 0.90)] if n > 1 else sorted_vals[0],
                "max": max(sorted_vals),
            }
        return summary

    @staticmethod
    def analyze_flip_flops(
        per_symbol_timelines: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Weighted-numerator/denominator flip-flop aggregation. No cross-symbol contamination."""
        f3_num = f3_den = f5_num = f5_den = f10_num = f10_den = 0
        for tl in per_symbol_timelines.values():
            n3, d3, n5, d5, n10, d10 = StabilityAnalyzer._flip_flop_numerators(tl)
            f3_num += n3; f3_den += d3
            f5_num += n5; f5_den += d5
            f10_num += n10; f10_den += d10
        return {
            "flip_3":  f3_num / f3_den if f3_den > 0 else 0.0,
            "flip_5":  f5_num / f5_den if f5_den > 0 else 0.0,
            "flip_10": f10_num / f10_den if f10_den > 0 else 0.0,
        }

    @staticmethod
    def analyze_transitions(
        per_symbol_timelines: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Sum per-symbol transition counts into a single aggregated matrix."""
        agg_counts: Dict[str, int] = {}
        agg_matrix: Dict[str, Dict[str, int]] = {}
        for tl in per_symbol_timelines.values():
            c, m = StabilityAnalyzer._transition_counts(tl)
            for k, v in c.items():
                agg_counts[k] = agg_counts.get(k, 0) + v
            for from_r, targets in m.items():
                agg_matrix.setdefault(from_r, {})
                for to_r, cnt in targets.items():
                    agg_matrix[from_r][to_r] = agg_matrix[from_r].get(to_r, 0) + cnt
        return agg_counts, agg_matrix

    @staticmethod
    def analyze_candidates(
        per_symbol_timelines: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Per-symbol candidate episode analysis, summed by candidate regime.
        No episode crosses from one symbol into another.
        """
        agg: Dict[str, Dict[str, Any]] = {}
        for tl in per_symbol_timelines.values():
            sym_res = StabilityAnalyzer._candidate_episodes(tl)
            for cand, metrics in sym_res.items():
                agg.setdefault(cand, {"episodes": 0, "confirmed": 0, "reset": 0})
                agg[cand]["episodes"]  += metrics["episodes"]
                agg[cand]["confirmed"] += metrics["confirmed"]
                agg[cand]["reset"]     += metrics["reset"]

        # Calculate rates
        for cand, metrics in agg.items():
            eps = metrics["episodes"]
            metrics["confirmation_rate"] = metrics["confirmed"] / eps if eps > 0 else 0.0
            metrics["reset_rate"]        = metrics["reset"]     / eps if eps > 0 else 0.0

        return agg

    # ------------------------------------------------------------------ #
    #  Legacy single-list overloads (kept for backward compatibility)     #
    # These wrap per_symbol helpers by treating the list as one symbol.   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def analyze_persistence_flat(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single flat timeline persistence. Only valid within one symbol."""
        return StabilityAnalyzer.analyze_persistence({"_flat": timeline})

    @staticmethod
    def analyze_flip_flops_flat(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
        """Single flat timeline flip-flop. Only valid within one symbol."""
        return StabilityAnalyzer.analyze_flip_flops({"_flat": timeline})

    @staticmethod
    def analyze_transitions_flat(timeline: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Single flat timeline transitions. Only valid within one symbol."""
        return StabilityAnalyzer.analyze_transitions({"_flat": timeline})

    @staticmethod
    def analyze_candidates_flat(timeline: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Single flat timeline candidate analysis. Only valid within one symbol."""
        return StabilityAnalyzer.analyze_candidates({"_flat": timeline})
