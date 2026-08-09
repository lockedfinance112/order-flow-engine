import os
import unittest
import copy
import time
import json
import shutil
from research.phase2.phase2_recorder import Phase2SignalRecorder
from research.phase2.data_quality_report import run_audit
from research.phase2.baseline_signal_quality import run_analysis

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

    def mock_decision(self, action, cycle=1, version=1):
        return {
            "action": action,
            "reason": "Test reason",
            "gates": {"1m_delta_positive": "PASS", "1m_delta_negative": "PASS", "5m_delta_bias": "PASS", "buy_aggression": "PASS"},
            "active_sweep": {"active": True, "direction": "bullish", "score": 8, "age_seconds": 12.5},
            "scanner_cycle_id": cycle,
            "action_version": version,
            "timestamp": time.time()
        }

    def test_1_confirmed_long_starts_track(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.42, 1250000.0, 0.1, "LARGE_TRADE", ["BULLISH_ABSORPTION"], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["direction"], "LONG")
        self.assertEqual(track["entry_price"], 65000.0)

    def test_2_confirmed_short_starts_track(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 65000.0, dec, {}, {}, {}, -0.35, -500000.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["direction"], "SHORT")

    def test_3_watch_long_starts_zero_tracks(self):
        dec = self.mock_decision("WATCH_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "WATCH_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 0)

    def test_4_watch_short_starts_zero_tracks(self):
        dec = self.mock_decision("WATCH_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "WATCH_SHORT", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 0)

    def test_5_repeated_confirmed_starts_zero_tracks(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.register_signal_change("BTCUSDT", "CONFIRMED_LONG", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 1)

    def test_6_genuine_transition_starts_new_track(self):
        dec1 = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        dec2 = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "CONFIRMED_LONG", "CONFIRMED_SHORT", 64000.0, dec2, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(len(self.recorder.active_tracks), 2)

    def test_7_metadata_captured_deeply(self):
        dec = self.mock_decision("CONFIRMED_LONG", cycle=42, version=3)
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["scanner_cycle_id"], 42)
        self.assertEqual(track["action_version"], 3)
        
        # Test deep copy mutation protection
        dec["gates"]["1m_delta_positive"] = "MUTATED"
        self.assertEqual(track["gate_1m_delta"], "PASS")

    def test_8_horizon_outcomes_and_directions(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        track = self.recorder.active_tracks[0]
        track["entry_time"] = time.time() - 901.0
        
        self.recorder.update_price("BTCUSDT", 110.0)
        self.assertEqual(track["price_after_1m"], 110.0)
        self.assertEqual(track["price_after_3m"], 110.0)
        self.assertEqual(track["price_after_5m"], 110.0)
        self.assertEqual(track["price_after_15m"], 110.0)
        
        self.assertAlmostEqual(track["return_1m_pct"], 0.1)
        self.assertAlmostEqual(track["return_15m_pct"], 0.1)

    def test_9_short_direction_adjust(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        track = self.recorder.active_tracks[0]
        track["entry_time"] = time.time() - 61.0
        self.recorder.update_price("BTCUSDT", 90.0)
        self.assertAlmostEqual(track["return_1m_pct"], 0.1)

    def test_10_mfe_mae_excursions_both_directions(self):
        # LONG check
        dec1 = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.update_price("BTCUSDT", 105.0)
        self.recorder.update_price("BTCUSDT", 98.0)
        track1 = self.recorder.active_tracks[0]
        self.assertAlmostEqual(track1["max_favorable_pct"], 0.05)
        self.assertAlmostEqual(track1["max_adverse_pct"], 0.02)
        
        # SHORT check
        dec2 = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("ETHUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec2, {}, {}, {}, -0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.update_price("ETHUSDT", 95.0)
        self.recorder.update_price("ETHUSDT", 103.0)
        track2 = self.recorder.active_tracks[1]
        self.assertAlmostEqual(track2["max_favorable_pct"], 0.05)
        self.assertAlmostEqual(track2["max_adverse_pct"], 0.03)

    def test_11_binance_context_handling(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        bc = {"funding_rate_pct": 0.01, "current_open_interest": 1000.0, "context_quality": "OK"}
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], bc)
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["context_quality"], "OK")
        self.assertEqual(track["open_interest"], 1000.0)

    def test_12_pending_survives_restart_recovery_and_missed_downtime(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        # Artificially shift entry time back by 4 minutes to simulate downtime
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 240.0
        self.recorder.flush_checkpoint(force=True)
        
        # Reloading in new instance
        rec2 = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )
        self.assertEqual(len(rec2.active_tracks), 1)
        track = rec2.active_tracks[0]
        
        # 1m and 3m targets should be missed during downtime
        self.assertEqual(track["horizon_1m_status"], "MISSED_DURING_DOWNTIME")
        self.assertEqual(track["horizon_3m_status"], "MISSED_DURING_DOWNTIME")
        self.assertEqual(track["horizon_5m_status"], "PENDING")
        
        # Make sure they cannot subsequently be captured
        rec2.update_price("BTCUSDT", 105.0)
        self.assertIsNone(track["price_after_1m"])

    def test_13_no_mutation_of_inputs(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertEqual(dec["action"], "CONFIRMED_LONG")

    def test_14_signal_ids_uniqueness(self):
        dec1 = self.mock_decision("CONFIRMED_LONG")
        dec2 = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec2, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.assertNotEqual(self.recorder.active_tracks[0]["signal_id"], self.recorder.active_tracks[1]["signal_id"])

    def test_15_short_1m_gate_mapping(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec, {}, {}, {}, -0.3, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        # Should check 1m_delta_negative
        self.assertEqual(track["gate_1m_delta"], "PASS")

    def test_16_fs_failure_isolated(self):
        # Force a path to be an invalid directory to trigger file system error
        self.recorder.signals_csv = "/invalid/dir/path/test.csv"
        # Calling finalize_expired_signals should not crash or propagate
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 905.0
        try:
            self.recorder.finalize_expired_signals()
        except Exception as e:
            self.fail(f"Filesystem error leaked: {e}")

if __name__ == "__main__":
    unittest.main()
