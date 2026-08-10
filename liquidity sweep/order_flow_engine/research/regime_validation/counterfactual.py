from typing import List, Dict, Any
from research.regime_validation.expectancy import ExpectancyCalculator
from research.regime_validation.bootstrap import BlockBootstrap

class CounterfactualAnalyzer:
    """Compares the pre-registered ALLOW-only filter against the baseline."""
    @staticmethod
    def compare_allow_only(
        joined_signals: List[Dict[str, Any]],
        cost_bps: int = 5,
        bootstrap_reps: int = 1000,
        bootstrap_seed: int = 1729
    ) -> Dict[str, Any]:
        
        # Baseline = all completed safe signals
        baseline_signals = [
            s for s in joined_signals
            if s.get("joined") and s.get("safe") and
            s.get("outcomes", {}).get("status") == "COMPLETED"
        ]
        
        # ALLOW-only counterfactual
        allow_signals = [
            s for s in baseline_signals
            if s.get("advisory_permission") == "ALLOW"
        ]
        
        block_signals = [
            s for s in baseline_signals
            if s.get("advisory_permission") == "BLOCK"
        ]

        baseline_exp = ExpectancyCalculator.calculate_expectancy(baseline_signals, 15, cost_bps)
        allow_exp = ExpectancyCalculator.calculate_expectancy(allow_signals, 15, cost_bps)
        
        retained_pct = len(allow_signals) / len(baseline_signals) if baseline_signals else 0.0
        
        # Bootstrap difference ALLOW vs BLOCK
        allow_minus_block_ci = BlockBootstrap.bootstrap_allow_minus_block(
            allow_signals,
            block_signals,
            reps=bootstrap_reps,
            seed=bootstrap_seed,
            cost_bps=cost_bps
        )
        
        # Bootstrap ALLOW vs BASELINE (optional/informational)
        allow_ci = BlockBootstrap.bootstrap_metrics(
            allow_signals,
            reps=bootstrap_reps,
            seed=bootstrap_seed,
            cost_bps=cost_bps
        )
        
        return {
            "baseline_count": len(baseline_signals),
            "allow_count": len(allow_signals),
            "retention_pct": retained_pct,
            "baseline_metrics": baseline_exp,
            "allow_metrics": allow_exp,
            "allow_minus_block_ci": allow_minus_block_ci.get("allow_minus_block_ci"),
            "allow_ci": allow_ci.get("mean_net_return_ci")
        }
