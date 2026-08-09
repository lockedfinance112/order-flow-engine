import os
import unittest
import copy
import time
import json
import shutil
from research.phase2.phase2_recorder import Phase2SignalRecorder
from research.phase2.data_quality_report import run_audit
from research.phase2.baseline_signal_quality import run_analysis
from research.phase2b.data_loader import partition_signals
from research.phase2b.statistics import calculate_stats, bootstrap_ci
from research.phase2b.counterfactual_engine import CounterfactualEngine
from research.phase2b.hypothesis_registry import HYPOTHESES

class TestPhase2Research(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_research_output"
        self.csv_name = "test_outcomes.csv"
        self.json_name = "test_active.json"
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
        self.recorder = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def mock_decision(self, action, cycle=1, version=1, ts=None):
        return {
            "action": action,
            "reason": "Test reason",
            "gates": {"1m_delta_positive": "PASS", "1m_delta_negative": "PASS", "5m_delta_bias": "PASS", "buy_aggression": "PASS"},
            "active_sweep": {"active": True, "direction": "bullish", "score": 8, "age_seconds": 12.5},
            "scanner_cycle_id": cycle,
            "action_version": version,
            "timestamp": ts or time.time()
        }

    # Partition Tests
    def test_1_holdout_never_enters_normal_engine(self):
        signals = [{"entry_time": str(i)} for i in range(150)]
        parts = partition_signals(signals)
        self.assertEqual(parts.status, "HOLDOUT SPLITS ACTIVATED")
        self.assertEqual(len(parts.development), 90)
        self.assertEqual(len(parts.validation), 30)
        self.assertEqual(len(parts._holdout), 30)

    def test_2_dev_val_chronological_ordering(self):
        signals = [{"entry_time": str(150 - i)} for i in range(150)]
        parts = partition_signals(signals)
        self.assertTrue(float(parts.development[0]["entry_time"]) < float(parts.development[-1]["entry_time"]))
        self.assertTrue(float(parts.validation[0]["entry_time"]) < float(parts.validation[-1]["entry_time"]))

    # Missing Data Semantics Tests
    def test_3_missing_book_drift_does_not_pass_h13(self):
        # Missing drift -> None
        sig = {"direction": "LONG", "book_drift": "MISSING"}
        res = HYPOTHESES["H13_LOW_BOOK_DRIFT"](sig)
        self.assertIsNone(res)

    def test_4_missing_oi_does_not_pass_h08(self):
        sig = {"open_interest_change_pct": "None"}
        res = HYPOTHESES["H08_OI_RISING"](sig)
        self.assertIsNone(res)

    def test_5_missing_cvd_does_not_pass_h01(self):
        sig = {"direction": "LONG", "session_cvd_usdt": "N/A"}
        res = HYPOTHESES["H01_CVD_ALIGNMENT"](sig)
        self.assertIsNone(res)

    def test_6_missing_event_history_does_not_pass_event_filters(self):
        sig = {"direction": "LONG", "recent_events": "MISSING"}
        res = HYPOTHESES["H14_NO_OPPOSING_DIVERGENCE"](sig)
        self.assertIsNone(res)

    # CVD alignment tests
    def test_7_long_cvd_alignment(self):
        sig1 = {"direction": "LONG", "session_cvd_usdt": "100.0"}
        sig2 = {"direction": "LONG", "session_cvd_usdt": "-50.0"}
        self.assertTrue(HYPOTHESES["H01_CVD_ALIGNMENT"](sig1))
        self.assertFalse(HYPOTHESES["H01_CVD_ALIGNMENT"](sig2))

    def test_8_short_cvd_alignment(self):
        sig1 = {"direction": "SHORT", "session_cvd_usdt": "-100.0"}
        sig2 = {"direction": "SHORT", "session_cvd_usdt": "50.0"}
        self.assertTrue(HYPOTHESES["H01_CVD_ALIGNMENT"](sig1))
        self.assertFalse(HYPOTHESES["H01_CVD_ALIGNMENT"](sig2))

    # Horizon eligibility tests
    def test_9_horizon_1m_eligibility(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_1m_status": "CAPTURED", "return_1m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_1m_status": "INTERRUPTED", "return_1m_pct": "0.02"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: True)
        self.assertEqual(res["overall"]["1m"]["candidate"]["count"], 1)

    # Direction and Symbol break down tests
    def test_10_direction_breakdown(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "SHORT", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.02"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: True)
        self.assertEqual(res["long"]["15m"]["candidate"]["count"], 1)
        self.assertEqual(res["short"]["15m"]["candidate"]["count"], 1)

    def test_11_symbol_breakdown(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "symbol": "BTCUSDT", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "symbol": "ETHUSDT", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.02"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: True)
        self.assertIn("BTCUSDT", res["by_symbol"])
        self.assertIn("ETHUSDT", res["by_symbol"])

    # Statistics and Denominator checks
    def test_12_retention_based_on_correct_denominator(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "MISSING"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "100.0"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "session_cvd_usdt": "-50.0"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(HYPOTHESES["H01_CVD_ALIGNMENT"])
        self.assertEqual(res["total_baseline"], 3)
        self.assertEqual(res["eligible_count"], 2)
        self.assertEqual(res["missing_feature_count"], 1)
        self.assertEqual(res["selected_count"], 1)
        self.assertEqual(res["rejected_count"], 1)
        self.assertAlmostEqual(res["retention_pct"], 1.0/3.0)
        self.assertAlmostEqual(res["retention_eligible_pct"], 0.5)

    def test_13_removed_winners_losers(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.02"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: float(s["return_15m_pct"]) < 0) # select loser, reject winner
        self.assertEqual(res["winners_sacrificed"], 1)
        self.assertEqual(res["losers_avoided"], 0)

    def test_14_bootstrap_ci_insufficient_sample_handling(self):
        returns = [0.01] * 5
        ci = bootstrap_ci(returns, iterations=10)
        self.assertEqual(ci["status"], "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP")
        self.assertIsNone(ci["win_rate"])

    def test_15_baseline_rows_remain_immutable(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.01"}
        ]
        engine = CounterfactualEngine(signals)
        engine.evaluate_hypothesis(lambda s: True)
        self.assertEqual(signals[0]["direction"], "LONG")

if __name__ == "__main__":
    unittest.main()
