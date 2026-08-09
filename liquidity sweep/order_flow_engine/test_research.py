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

    def test_1_entry_time_equals_committed_timestamp(self):
        committed_ts = time.time() - 1000.0
        dec = self.mock_decision("CONFIRMED_LONG", ts=committed_ts)
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 65000.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["entry_time"], committed_ts)

    def test_2_excursion_dirty_flagging(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.dirty = False # Reset
        
        # MFE increases
        self.recorder.update_price("BTCUSDT", 105.0)
        self.assertTrue(self.recorder.dirty)
        self.recorder.dirty = False
        
        # MAE increases
        self.recorder.update_price("BTCUSDT", 98.0)
        self.assertTrue(self.recorder.dirty)

    def test_3_failed_csv_append_keeps_track_active(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        # Force invalid path to break CSV write
        self.recorder.signals_csv = "/invalid/dir/path/test.csv"
        
        # Shift time to force finalization
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 905.0
        self.recorder.finalize_expired_signals()
        
        # Should NOT be removed from active tracks on failed CSV append
        self.assertEqual(len(self.recorder.active_tracks), 1)

    def test_4_recovered_active_track_has_interrupted_excursion(self):
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.flush_checkpoint(force=True)
        
        rec2 = Phase2SignalRecorder(
            output_dir=self.test_dir,
            signals_csv=self.csv_name,
            active_json=self.json_name
        )
        self.assertEqual(len(rec2.active_tracks), 1)
        track = rec2.active_tracks[0]
        self.assertTrue(track["recovered_after_restart"])
        self.assertEqual(track["excursion_coverage_status"], "INTERRUPTED")

    def test_5_duplicate_canonical_event_key_idempotency(self):
        dec1 = self.mock_decision("CONFIRMED_LONG", cycle=123, version=1)
        dec2 = self.mock_decision("CONFIRMED_LONG", cycle=123, version=1) # same cycle & version
        
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec2, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        # Should register exactly ONE track because of idempotency
        self.assertEqual(len(self.recorder.active_tracks), 1)

    def test_6_different_action_version_creates_new_track(self):
        dec1 = self.mock_decision("CONFIRMED_LONG", cycle=123, version=1)
        dec2 = self.mock_decision("CONFIRMED_LONG", cycle=123, version=2) # different version
        
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec1, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec2, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        
        self.assertEqual(len(self.recorder.active_tracks), 2)

    def test_7_short_1m_gate_mapping(self):
        dec = self.mock_decision("CONFIRMED_SHORT")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_SHORT", 100.0, dec, {}, {}, {}, -0.3, 0.0, 0.1, "LARGE_TRADE", [], {})
        track = self.recorder.active_tracks[0]
        self.assertEqual(track["gate_1m_delta"], "PASS")

    def test_8_fs_failure_isolated(self):
        self.recorder.signals_csv = "/invalid/dir/path/test.csv"
        dec = self.mock_decision("CONFIRMED_LONG")
        self.recorder.register_signal_change("BTCUSDT", "WAITING", "CONFIRMED_LONG", 100.0, dec, {}, {}, {}, 0.1, 0.0, 0.1, "LARGE_TRADE", [], {})
        self.recorder.active_tracks[0]["entry_time"] = time.time() - 905.0
        try:
            self.recorder.finalize_expired_signals()
        except Exception as e:
            self.fail(f"Filesystem error leaked: {e}")

if __name__ == "__main__":
    unittest.main()
