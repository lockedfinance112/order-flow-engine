import os
import unittest
from research.phase2b.data_loader import partition_signals
from research.phase2b.statistics import calculate_stats, bootstrap_ci
from research.phase2b.counterfactual_engine import CounterfactualEngine
from research.phase2b.hypothesis_registry import HYPOTHESES, combine_and

class TestPhase2BResearch(unittest.TestCase):
    # Eligible Baseline & Total Baseline
    def test_total_vs_eligible_baseline(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "MISSING", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "100.0", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.02"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "-50.0", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.01"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(HYPOTHESES["H01_CVD_ALIGNMENT"])
        
        self.assertEqual(res["total_baseline"], 3)
        self.assertEqual(res["eligible_count"], 2)
        self.assertEqual(res["missing_feature_count"], 1)
        self.assertEqual(res["selected_count"], 1)
        
        # Test cohort level stats
        overall_15m = res["overall"]["15m"]
        self.assertEqual(overall_15m["total_baseline"]["count"], 3)
        self.assertEqual(overall_15m["eligible_baseline"]["count"], 2)
        self.assertEqual(overall_15m["candidate"]["count"], 1)

    # Tri-state Pairwise logic
    def test_tri_state_pairwise(self):
        # Filter A returns True, Filter B returns None -> combined returns None
        fa = lambda s: True
        fb = lambda s: None
        combined = combine_and(fa, fb)
        self.assertIsNone(combined({}))

        # Filter A returns False, Filter B returns True -> combined returns False
        fa = lambda s: False
        fb = lambda s: True
        combined = combine_and(fa, fb)
        self.assertFalse(combined({}))

        # Filter A returns True, Filter B returns True -> combined returns True
        fa = lambda s: True
        fb = lambda s: True
        combined = combine_and(fa, fb)
        self.assertTrue(combined({}))

    # H06 no sweep check
    def test_h06_no_sweep_eligible(self):
        # sweep_active=False + empty sweep_direction -> eligible False (returns False, not None!)
        sig = {
            "direction": "LONG",
            "sweep_active": "False",
            "sweep_direction": ""
        }
        res = HYPOTHESES["H06_ACTIVE_DIRECTIONAL_SWEEP"](sig)
        self.assertFalse(res)
        self.assertIsNotNone(res)

    # DEV vs VAL Independence
    def test_dev_vs_val_independence(self):
        signals = [{"entry_time": str(i)} for i in range(100)]
        parts = partition_signals(signals)
        self.assertEqual(parts.status, "HOLDOUT SPLITS ACTIVATED")
        self.assertEqual(len(parts.development), 60)
        self.assertEqual(len(parts.validation), 20)
        self.assertEqual(len(parts._holdout), 20)

    # Holdout Lock strength
    def test_holdout_lock_strength(self):
        signals = [{"entry_time": str(i)} for i in range(100)]
        parts = partition_signals(signals)
        with self.assertRaises(PermissionError):
            parts.unlock_holdout_for_final_evaluation("ILLEGAL_ACCESS")
            
        unlocked = parts.unlock_holdout_for_final_evaluation("PHASE2C_FINAL_EVALUATION")
        self.assertEqual(len(unlocked), 20)

    # Bootstrap CI
    def test_bootstrap_insufficient_sample(self):
        returns = [0.01] * 10
        res = bootstrap_ci(returns)
        self.assertEqual(res["status"], "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP")
        self.assertIsNone(res["win_rate"])

if __name__ == "__main__":
    unittest.main()
