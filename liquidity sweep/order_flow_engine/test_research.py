import os
import unittest
import copy
import time
import json
import shutil
from research.phase2.phase2_recorder import Phase2SignalRecorder
from research.phase2.data_quality_report import run_audit
from research.phase2.baseline_signal_quality import run_analysis
from research.phase2b.statistics import calculate_stats, bootstrap_ci
from research.phase2b.counterfactual_engine import CounterfactualEngine

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

    def test_1_confirmed_long_tracked(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.42, 1250000.0, 0.1, "LARGE_TRADE", ["BULLISH_ABSORPTION"], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)
        self.assertEqual(self.recorder.active_tracks[0]["direction"], "LONG")

    def test_2_confirmed_short_tracked(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 65000.0, dec, {}, {}, {}, -0.35, -500000.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)
        self.assertEqual(self.recorder.active_tracks[0]["direction"], "SHORT")

    def test_3_watch_long_not_tracked(self):
        dec = self.mock_decision("WATCH_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "WATCH_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 0)

    def test_4_watch_short_not_tracked(self):
        dec = self.mock_decision("WATCH_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "WATCH_SHORT", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 0)

    def test_5_cvd_and_depth_imbalance_mapping(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.42, 1250000.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["imbalance"], 0.42)
        self.assertEqual(track["session_cvd_usdt"], 1250000.0)

    def test_6_recent_events_snapshot(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", ["BULLISH_ABSORPTION", "LARGE_TRADE"], {})
        track = self.recorder.active_tracks[0]
        recent = json.loads(track["recent_events"])
        self.assertEqual(recent, ["BULLISH_ABSORPTION", "LARGE_TRADE"])

    def test_7_short_gate_mapping(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec, {}, {}, {}, -0.3, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["gate_1m_delta"], "PASS")

    def test_8_all_four_horizon_outcomes(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        track = self.recorder.active_tracks[0]
        track["entry_time"] = time.time() - 901.0
        
        self.recorder.update_price("BTCUSDT", 110.0)
        self.assertEqual(track["price_after_1m"], 110.0)
        self.assertEqual(track["price_after_3m"], 110.0)
        self.assertEqual(track["price_after_5m"], 110.0)
        self.assertEqual(track["price_after_15m"], 110.0)

    def test_9_long_and_short_mfe_mae(self):
        dec1 = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.update_price("BTCUSDT", 105.0)
        self.recorder.update_price("BTCUSDT", 98.0)
        track1 = self.recorder.active_tracks[0]
        self.assertAlmostEqual(track1["max_favorable_pct"], 0.05)
        self.assertAlmostEqual(track1["max_adverse_pct"], 0.02)
        
        dec2 = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("ETHUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec2, {}, {}, {}, -0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.update_price("ETHUSDT", 95.0)
        self.recorder.update_price("ETHUSDT", 103.0)
        track2 = self.recorder.active_tracks[1]
        self.assertAlmostEqual(track2["max_favorable_pct"], 0.05)
        self.assertAlmostEqual(track2["max_adverse_pct"], 0.03)

    def test_10_downtime_missed_horizons(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 240.0
        self.recorder.flush_checkpoint(force=True)
        
        rec2 = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )
        track = rec2.active_tracks[0]
        self.assertEqual(track["horizon_1m_status"], "MISSED_DURING_DOWNTIME")
        self.assertEqual(track["horizon_3m_status"], "MISSED_DURING_DOWNTIME")
        self.assertEqual(track["horizon_5m_status"], "PENDING")

    def test_11_recovered_excursion_exclusion(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.flush_checkpoint(force=True)
        
        rec2 = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )
        track = rec2.active_tracks[0]
        self.assertEqual(track["excursion_coverage_status"], "INTERRUPTED")

    def test_12_canonical_timestamp(self):
        committed_ts = time.time() - 500.0
        dec = self.mock_decision("CONFIRMED_LONG", ts=committed_ts)
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["entry_time"], committed_ts)

    def test_13_duplicate_event_suppression(self):
        dec = self.mock_decision("CONFIRMED_LONG", cycle=10, version=2, ts=1000.0)
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)

    def test_14_append_success_checkpoint_failure_replay(self):
        dec = self.mock_decision("CONFIRMED_LONG", cycle=999, version=1, ts=5555.0)
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.recorder._append_to_csv(track)
        
        rec2 = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )
        res = rec2._append_to_csv(track)
        self.assertTrue(res)

    def test_15_recorder_filesystem_failure_isolated(self):
        self.recorder.signals_csv = "/invalid/dir/path/test.csv"
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 905.0
        try:
            self.recorder.finalize_expired_signals()
        except Exception as e:
            self.fail(f"Filesystem error leaked: {e}")

    # Counterfactual engine tests
    def test_16_baseline_unchanged_by_experiments(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.02"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.01"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: float(s["return_15m_pct"]) > 0)
        self.assertEqual(res["baseline_count"], 2)
        self.assertEqual(res["candidate_count"], 1)
        self.assertEqual(len(engine.baseline_signals), 2) # baseline remains unchanged

    def test_17_retention_calculation(self):
        signals = [
            {"dataset_class": "CANONICAL_PHASE2", "direction": "LONG", "horizon_15m_status": "CAPTURED", "return_15m_pct": "0.02"},
            {"dataset_class": "CANONICAL_PHASE2", "direction": "SHORT", "horizon_15m_status": "CAPTURED", "return_15m_pct": "-0.01"}
        ]
        engine = CounterfactualEngine(signals)
        res = engine.evaluate_hypothesis(lambda s: s["direction"] == "LONG")
        self.assertEqual(res["retention_pct"], 0.5)

    def test_18_bootstrap_ci_reproducibility(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01]
        ci1 = bootstrap_ci(returns, iterations=100, seed=42)
        ci2 = bootstrap_ci(returns, iterations=100, seed=42)
        self.assertEqual(ci1["win_rate"], ci2["win_rate"])

if __name__ == "__main__":
    unittest.main()
