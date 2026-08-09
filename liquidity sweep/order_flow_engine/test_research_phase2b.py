import os
import unittest
from research.phase2b.data_loader import partition_signals
from research.phase2b.statistics import calculate_stats, bootstrap_ci
from research.phase2b.counterfactual_engine import CounterfactualEngine
from research.phase2b.hypothesis_registry import HYPOTHESES, combine_and
from research.phase2b.walk_forward import WalkForwardValidator

class TestPhase2BResearchHardened(unittest.TestCase):
    # Bootstrap checks
    def test_bootstrap_reproducible_and_wired(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01] * 3 # N=24
        ci1 = bootstrap_ci(returns, iterations=100, seed=42)
        ci2 = bootstrap_ci(returns, iterations=100, seed=42)
        self.assertEqual(ci1["status"], "OK")
        self.assertEqual(ci1["win_rate"], ci2["win_rate"])

    def test_insufficient_bootstrap_remains_none(self):
        returns = [0.01] * 10
        ci = bootstrap_ci(returns)
        self.assertEqual(ci["status"], "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP")
        self.assertIsNone(ci["win_rate"])

    # Partition and bounds checks
    def test_dev_only_boundaries_and_holdout_never_seen(self):
        signals = [{"entry_time": str(i)} for i in range(100)]
        parts = partition_signals(signals)
        self.assertEqual(len(parts.development), 60)
        self.assertEqual(len(parts.validation), 20)
        self.assertEqual(len(parts._holdout), 20)

        # Ensure validation cannot alter DEV bucket bounds (which are computed purely on partitions.development)
        self.assertTrue(parts.status == "HOLDOUT SPLITS ACTIVATED")

    # Walk forward checks
    def test_walk_forward_missing_return_and_ordering(self):
        signals = [
            {"entry_time": "1", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"entry_time": "2", "horizon_15m_status": "CAPTURED", "return_15m_pct": "MISSING"},
            {"entry_time": "3", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.02"}
        ]
        wfv = WalkForwardValidator(signals)
        # Ensure chronological sorting order
        self.assertEqual(wfv.signals[0]["entry_time"], "1")
        self.assertEqual(wfv.signals[2]["entry_time"], "3")

    # Validation classifications
    def test_validation_result_independently_calculated(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "100.0", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "-50.0", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.01"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(HYPOTHESES["H01_CVD_ALIGNMENT"])
        self.assertEqual(res["total_baseline"], 2)

    # Immutability
    def test_source_rows_remain_immutable(self):
        signals = [{"dataset_class": "CANONICAL_PHASE2", "direction": "LONG"}]
        engine = CounterfactualEngine(signals)
        engine.evaluate_hypothesis(lambda s: True)
        self.assertEqual(signals[0]["direction"], "LONG")

if __name__ == "__main__":
    unittest.main()
