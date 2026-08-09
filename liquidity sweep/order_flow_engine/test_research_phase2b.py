import os
import unittest
import numpy as np
from research.phase2b.data_loader import partition_signals
from research.phase2b.statistics import calculate_stats, bootstrap_ci
from research.phase2b.counterfactual_engine import CounterfactualEngine
from research.phase2b.hypothesis_registry import HYPOTHESES, combine_and
from research.phase2b.walk_forward import WalkForwardValidator
from research.phase2b.run_phase2b import cohort_matches, compute_buckets

class TestPhase2BResearchHardened(unittest.TestCase):
    # Cohort Matching Tests
    def test_cohort_matches(self):
        long_sig = {"direction": "LONG"}
        short_sig = {"direction": "SHORT"}
        self.assertTrue(cohort_matches(long_sig, "overall"))
        self.assertTrue(cohort_matches(long_sig, "long"))
        self.assertFalse(cohort_matches(long_sig, "short"))
        self.assertTrue(cohort_matches(short_sig, "short"))
        self.assertFalse(cohort_matches(short_sig, "long"))

    # Bootstrap Sizing Tests
    def test_bootstrap_insufficient_sample(self):
        returns = [0.01] * 10
        res = bootstrap_ci(returns)
        self.assertEqual(res["status"], "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP")
        self.assertIsNone(res["win_rate"])

    # Buckets bounds tests
    def test_compute_buckets_dev_only(self):
        dev_pool = [
            {"direction": "LONG", "imbalance": str(i * 0.01), "delta_5m_usdt": "100", "delta_15m_usdt": "200",
             "session_cvd_usdt": "500", "open_interest_change_pct": "0.01", "sweep_score": "8", "book_drift": "0.1",
             "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"}
            for i in range(25)
        ]
        buckets = compute_buckets(dev_pool)
        self.assertIn("imbalance", buckets)
        self.assertIn("delta_5m_usdt", buckets)
        self.assertIn("delta_15m_usdt", buckets)
        self.assertIn("session_cvd_usdt", buckets)
        self.assertIn("open_interest_change_pct", buckets)
        self.assertIn("sweep_score", buckets)
        self.assertIn("book_drift", buckets)
        
        # Test boundaries returned
        self.assertEqual(len(buckets["imbalance"]["boundaries"]), 4)
        self.assertEqual(len(buckets["imbalance"]["buckets"]), 5)

    # Walk forward tests
    def test_walk_forward_window(self):
        signals = [
            {"entry_time": str(i), "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"}
            for i in range(130)
        ]
        wfv = WalkForwardValidator(signals)
        res = wfv.run_walk_forward(lambda s: True, min_train_size=100, step_size=25)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["validation_n"], 25)

    # Tri-state Combine AND tests
    def test_tri_state_pairwise(self):
        fa = lambda s: True
        fb = lambda s: None
        combined = combine_and(fa, fb)
        self.assertIsNone(combined({}))

    # Immutability
    def test_immutability(self):
        signals = [{"dataset_class": "CANONICAL_PHASE2", "direction": "LONG"}]
        engine = CounterfactualEngine(signals)
        engine.evaluate_hypothesis(lambda s: True)
        self.assertEqual(signals[0]["direction"], "LONG")

if __name__ == "__main__":
    unittest.main()
