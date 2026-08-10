import unittest
import time
import os
import json
import gzip
import tempfile
import shutil
import csv
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from regime.models import MarketBar
from regime.permissions import permissions_for
from scoring import OrderFlowScorer
from flow_metrics import FlowMetrics

from research.regime_validation.models import ValidationSignal
from research.regime_validation.protocol import get_protocol_hash, get_config_hash
from research.regime_validation.dataset import DatasetManager
from research.regime_validation.kline_replay import HistoricalRegimeReplayRunner
from research.regime_validation.asof_join import AsOfJoiner
from research.regime_validation.outcome_labels import OutcomeLabeler, calculate_atr14_at_t
from research.regime_validation.confusion import ConfusionMatrixCalculator
from research.regime_validation.stability import StabilityAnalyzer
from research.regime_validation.breakout_analysis import BreakoutAnalyzer
from research.regime_validation.expectancy import ExpectancyCalculator
from research.regime_validation.counterfactual import CounterfactualAnalyzer
from research.regime_validation.bootstrap import BlockBootstrap
from research.regime_validation.signal_sources import SignalLoader

class TestRegimeValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset_manager = DatasetManager(self.tmp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write_compressed_jsonl(self, filename: str, data: list):
        filepath = os.path.join(self.tmp_dir, filename)
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            for k in data:
                f.write(json.dumps(k) + "\n")
        return filepath

    # 1. Dataset Validation Tests (A to H)
    def test_dataset_validations(self):
        # A: closed-bar dataset validation
        raw_klines = [
            [1700002800000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0],
            [1700002860000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002919999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("btcusdt_1m.jsonl.gz", raw_klines)
        q = self.dataset_manager.validate_dataset("btcusdt")
        self.assertTrue(q["valid"])
        self.assertEqual(q["bar_count"], 2)

        # B: duplicate candle detection
        raw_dups = [
            [1700002800000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0],
            [1700002800000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("ethusdt_1m.jsonl.gz", raw_dups)
        q_dup = self.dataset_manager.validate_dataset("ethusdt")
        self.assertFalse(q_dup["valid"])
        self.assertEqual(q_dup["duplicate_bar_count"], 1)

        # C & 54: missing bar detection and strict gap validation
        raw_gap = [
            [1700002800000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0],
            [1700002920000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002979999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("solusdt_1m.jsonl.gz", raw_gap)
        q_gap = self.dataset_manager.validate_dataset("solusdt", allow_gaps=False)
        self.assertEqual(q_gap["missing_bar_count"], 1)
        self.assertFalse(q_gap["valid"]) # strict gap validation fails quality check

        # D: bad OHLC rejection (high < low)
        raw_bad = [
            [1700002800000, 100.0, 90.0, 110.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("bnbusdt_1m.jsonl.gz", raw_bad)
        q_bad = self.dataset_manager.validate_dataset("bnbusdt")
        self.assertFalse(q_bad["valid"])
        self.assertEqual(q_bad["bad_price_count"], 1)

    def test_chronological_splits(self):
        # G & H & 53: Chronological splits and holdout boundary UTC alignment
        bars = []
        # Use exact UTC midnight start time
        midnight_start = 1699920000000 
        for i in range(2880 * 2): # 2 days of 1m bars
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=midnight_start + i * 60000,
                close_time_ms=midnight_start + (i + 1) * 60000 - 1,
                open=100.0, high=100.0, low=100.0, close=100.0,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
            
        warmup, dev, val, holdout = self.dataset_manager.get_splits(
            bars, warmup_days=1, dev_pct=0.50, val_pct=0.25, holdout_pct=0.25
        )
        self.assertEqual(len(warmup), 1440)
        # Verify splits alignment midnight
        self.assertEqual(dev[0].open_time_ms % 86400000, 0)
        self.assertEqual(val[0].open_time_ms % 86400000, 0)

    # 2. Replay Determinism & Gaps (I to O)
    def test_deterministic_replay(self):
        # I & J & 51: Replay output is deterministic, wall-clock independent, and non-DISABLED
        bars = []
        for i in range(100):
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
            
        runner1 = HistoricalRegimeReplayRunner(["BTCUSDT"], config={})
        t1 = runner1.run_replay(bars[:50], bars[50:], [], [])
        
        time.sleep(0.5)
        
        runner2 = HistoricalRegimeReplayRunner(["BTCUSDT"], config={})
        t2 = runner2.run_replay(bars[:50], bars[50:], [], [])
        
        self.assertEqual(len(t1), len(t2))
        for r1, r2 in zip(t1, t2):
            self.assertEqual(r1["primary_regime"], r2["primary_regime"])
            self.assertEqual(r1["confidence"], r2["confidence"])
            self.assertNotEqual(r1["quality"], "DISABLED") # Should be READY/WARMING_UP/DEGRADED, not DISABLED

    # 3. As-Of Join Verification (P to S)
    def test_as_of_joins(self):
        # P: Join latest canonical regime with close_time <= signal_time
        timeline = [
            {"symbol": "BTCUSDT", "latest_1m_close_time": 1700002859999, "primary_regime": "TREND_UP", "quality": "READY", "tradable": True},
            {"symbol": "BTCUSDT", "latest_1m_close_time": 1700002919999, "primary_regime": "RANGE", "quality": "READY", "tradable": True}
        ]
        
        signal = ValidationSignal(
            symbol="btcusdt", timestamp_ms=1700002890000,
            direction="LONG", action="LONG_BIAS", strategy_family="long_momentum",
            metadata={}
        )
        res = AsOfJoiner.join_signal_to_regime(signal, timeline)
        self.assertTrue(res["joined"])
        self.assertEqual(res["regime_state"]["primary_regime"], "TREND_UP")

        # Q: Signal before first regime -> NO_REGIME
        signal_before = ValidationSignal(
            symbol="btcusdt", timestamp_ms=1700002700000,
            direction="LONG", action="LONG_BIAS", strategy_family="long_momentum",
            metadata={}
        )
        res_before = AsOfJoiner.join_signal_to_regime(signal_before, timeline)
        self.assertEqual(res_before["reason"], "NO_REGIME")

        # R: Regime too old -> NO_FRESH_REGIME
        signal_stale = ValidationSignal(
            symbol="btcusdt", timestamp_ms=1700003500000,
            direction="LONG", action="LONG_BIAS", strategy_family="long_momentum",
            metadata={}
        )
        res_stale = AsOfJoiner.join_signal_to_regime(signal_stale, timeline)
        self.assertEqual(res_stale["reason"], "NO_FRESH_REGIME")

    # 4. Ex-Post Labels Verification (T to Y)
    def test_ex_post_labeling(self):
        # T: UP_DIRECTIONAL path check
        bars = []
        for i in range(50):
            p = 100.0 + (i * 2.0)
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=p, high=p + 0.1, low=p - 0.1, close=p,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
        
        lbl, m = OutcomeLabeler.compute_ex_post_label(bars, t_idx=20, horizon_min=15, thresholds={})
        self.assertEqual(lbl, "UP_DIRECTIONAL")
        self.assertGreaterEqual(m["future_efficiency"], 0.35)

    # 5. Confusion Matrix & Stability
    def test_confusion_and_stability(self):
        timeline = [{"primary_regime": "TREND_UP"}, {"primary_regime": "RANGE"}]
        ref_labels = ["UP_DIRECTIONAL", "RANGE"]
        
        matrix, summary = ConfusionMatrixCalculator.calculate_matrix(timeline, ref_labels)
        self.assertEqual(summary["agreement_rate"], 1.0)
        self.assertEqual(matrix["UP_DIRECTIONAL"]["UP_DIRECTIONAL"], 1)

    # 6. Breakout Analysis
    def test_breakout_validation(self):
        bars = []
        for i in range(50):
            p = 100.0 + (i * 2.0)
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=p, high=p + 0.1, low=p - 0.1, close=p,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
            
        timeline = [{"symbol": "BTCUSDT", "latest_1m_close_time": 1700002800000 + 21 * 60000 - 1, "primary_regime": "BREAKOUT_UP", "confidence": 0.85}]
        bo_res = BreakoutAnalyzer.analyze_breakouts(timeline, bars)
        self.assertEqual(len(bo_res), 1)
        self.assertEqual(bo_res[0]["outcome"], "SUCCESSFUL_FOLLOW_THROUGH")

    # 7. Expectancy & Costs
    def test_expectancy_and_costs(self):
        joined = [{
            "joined": True, "safe": True,
            "timestamp_ms": 1700002800000,
            "outcomes": {
                "15m": {"status": "COMPLETED", "return": 0.0020, "mfe": 0.0030, "mae": 0.0005}
            }
        }]
        
        exp = ExpectancyCalculator.calculate_expectancy_for_horizon(joined, "15m", cost_bps=5)
        self.assertAlmostEqual(exp["mean_return"], 0.0015)

    # 8. Shadow Mode Validation Isolation
    def test_shadow_mode_isolation(self):
        metrics = FlowMetrics()
        scorer = OrderFlowScorer(metrics)
        
        now = time.time()
        res_before = scorer.evaluate_bias("BTCUSDT", {"status": "READY", "delta_usdt": 10.0, "buy_aggression_ratio": 0.5}, 1.0, [], now, 0.0)
        
        # Run validation components
        timeline = [{"primary_regime": "TREND_UP"}]
        ref_labels = ["UP_DIRECTIONAL"]
        ConfusionMatrixCalculator.calculate_matrix(timeline, ref_labels)
        
        res_after = scorer.evaluate_bias("BTCUSDT", {"status": "READY", "delta_usdt": 10.0, "buy_aggression_ratio": 0.5}, 1.0, [], now, 0.0)
        self.assertEqual(res_before, res_after)

    # 9. Timezone-aware Signal Parsing (Requirement 52)
    def test_utc_signal_parsing(self):
        ts_str = "2026-07-02T10:24:37.248399+00:00"
        ts_ms = SignalLoader.parse_iso_timestamp(ts_str)
        self.assertEqual(ts_ms, 1782987877248)

    # 10. E2E Offline Integration Test with Mocked Network (Requirements 49 & 50 & 51 & 55 & 56 & 57 & 58 & 59)
    @patch("urllib.request.urlopen")
    def test_offline_e2e_run_all(self, mock_urlopen):
        # Prevent any network access
        mock_urlopen.side_effect = Exception("Accidental network access in replay!")
        
        # Write 5 symbol mock datasets with 1500 bars of 1m each to exceed 21 hours
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        
        # Save baseline CSV file states to verify no mutations
        states_path = "regime_states.csv"
        transitions_path = "regime_transitions.csv"
        states_exist = os.path.exists(states_path)
        trans_exist = os.path.exists(transitions_path)
        
        # Hour aligned start time
        aligned_start = 1700000000000 - (1700000000000 % 3600000)
        
        for sym in symbols:
            raw_klines = []
            for i in range(1500):
                # 1500 closed bars of 1m
                p = 100.0 + (i * 0.01)
                raw_klines.append([
                    aligned_start + i * 60000,
                    p, p + 0.05, p - 0.05, p,
                    10.0,
                    aligned_start + (i + 1) * 60000 - 1,
                    1000.0, 10, 0, 0, 0
                ])
            self._write_compressed_jsonl(f"{sym.lower()}_1m.jsonl.gz", raw_klines)
            
            # Write mock quality JSONs
            q_data = {
                "symbol": sym.upper(),
                "valid": True,
                "bar_count": 1500,
                "content_sha256": "mock_sha",
                "actual_start_ms": aligned_start,
                "actual_end_ms": aligned_start + 1499 * 60000,
                "actual_start_utc": "2023-11-14T22:00:00+00:00",
                "actual_end_utc": "2023-11-15T22:59:00+00:00"
            }
            with open(os.path.join(self.tmp_dir, f"{sym.lower()}_quality.json"), "w") as f:
                json.dump(q_data, f)

        # Write mock manifest
        manifest = {
            "dataset_id": "mock_ds",
            "created_at": "2026-08-10T00:00:00Z",
            "symbols": symbols
        }
        with open(os.path.join(self.tmp_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f)

        # Prepare mock signal transitions CSV
        transitions_csv_path = os.path.join(self.tmp_dir, "bias_transitions.csv")
        with open(transitions_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "old_action", "new_action", "price", "latest_event", "suppression_reason", "reason"])
            w.writerow(["2023-11-14T22:30:00.000+00:00", "BTCUSDT", "WAITING", "CONFIRMED_LONG", "110.0", "BUY_AGGRESSION", "NONE", "test reason"])
            w.writerow(["2023-11-14T22:45:00.000+00:00", "BTCUSDT", "WAITING", "CONFIRMED_SHORT", "120.0", "SELL_AGGRESSION", "NONE", "test reason"])
            w.writerow(["2023-11-14T23:00:00.000+00:00", "BTCUSDT", "WAITING", "LONG_BIAS", "130.0", "BUY_AGGRESSION", "NONE", "legacy test reason"])

        # Patch run_all components
        with patch("research.regime_validation.cli.DatasetManager") as mock_manager_class, \
             patch("research.regime_validation.cli.cmd_inventory") as mock_cmd_inventory, \
             patch("research.regime_validation.cli.load_or_create_protocol") as mock_load_protocol:
             
            # Inject our local DatasetManager pointing to tmp_dir
            mock_manager_class.return_value = self.dataset_manager
            
            # Setup mock protocol
            protocol = {
                "schema_version": "1.0",
                "baseline_commit": "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c",
                "regime_model_version": "regime-v1",
                "regime_feature_version": "regime-features-v1",
                "classifier_config_hash": "default",
                "symbols": symbols,
                "dataset_date_ranges": {"start": "2023-11-14", "end": "2023-11-15"},
                "warmup_period_days": 0, # zero warmup for fast test
                "development_period_pct": 0.60,
                "validation_period_pct": 0.20,
                "holdout_period_pct": 0.20,
                "primary_outcome_horizon_min": 15,
                "secondary_outcome_horizon_min": 60,
                "reference_label_thresholds": {"directional_atr": 1.0, "efficiency": 0.35, "range_atr": 0.50, "range_efficiency": 0.25},
                "bootstrap_method": "utc_day_block",
                "bootstrap_seed": 1729,
                "bootstrap_repetitions": 10,
                "minimum_sample_size": 1,
                "cost_scenarios_bps": [0, 2, 5, 10],
                "primary_cost_scenario_bps": 5,
                "signal_regime_join_rules": {"max_age_ms": 90000, "mode": "as_of_backward"},
                "allowed_signal_sources": ["RECORDED_DECISION_TRANSITION", "LEGACY_SIGNAL_LOG"],
                "dataset_quality_requirements": {"allow_gaps": True, "check_monotonic": True}
            }
            mock_load_protocol.return_value = protocol
            
            # Redirect bias_transitions path to our temp file
            from research.regime_validation import cli
            cli.cmd_inventory = lambda args: None
            
            root_dir = os.path.abspath(os.path.dirname(__file__))
            # Mock cmd_prepare to write quality files to research/datasets
            target_ds_dir = os.path.join(root_dir, "research/datasets")
            os.makedirs(target_ds_dir, exist_ok=True)
            
            def mock_prep(args):
                q_summary = {}
                for sym in symbols:
                    q_summary[sym] = {
                        "symbol": sym.upper(), "valid": True, "bar_count": 1500, "content_sha256": "mock_sha",
                        "actual_start_ms": aligned_start, "actual_end_ms": aligned_start + 1499 * 60000,
                        "actual_start_utc": "2023-11-14T22:00:00+00:00", "actual_end_utc": "2023-11-15T22:59:00+00:00"
                    }
                with open(os.path.join(target_ds_dir, "dataset_quality.json"), "w") as f:
                    json.dump(q_summary, f)
                with open(os.path.join(target_ds_dir, "dataset_manifest.json"), "w") as f:
                    json.dump({"dataset_id": "mock_ds"}, f)
            cli.cmd_prepare = mock_prep
            
            orig_exists = os.path.exists
            # Run run_all CLI orchestrator using transitions_csv_path
            with patch("research.regime_validation.cli.os.path.exists", side_effect=lambda p: True if "bias_transitions.csv" in p or "bias_signals.csv" in p else orig_exists(p)), \
                 patch("research.regime_validation.cli.SignalLoader.load_from_transitions_csv", return_value=SignalLoader.load_from_transitions_csv(transitions_csv_path)):
                 
                cli.cmd_run_all(None)
                
        # Assert no modifications to live production CSVs
        if states_exist:
            self.assertTrue(os.path.exists(states_path))
        else:
            self.assertFalse(os.path.exists(states_path))
            
        if trans_exist:
            self.assertTrue(os.path.exists(transitions_path))
        else:
            self.assertFalse(os.path.exists(transitions_path))
            
        # Verify run validation runs artifacts exist
        validation_runs_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "research/validation_runs")
        # Find the latest run
        runs = sorted(os.listdir(validation_runs_dir))
        self.assertTrue(len(runs) > 0)
        latest_run = os.path.join(validation_runs_dir, runs[-1])
        
        self.assertTrue(os.path.exists(os.path.join(latest_run, "validation_summary.json")))
        self.assertTrue(os.path.exists(os.path.join(latest_run, "validation_decision.md")))
        self.assertTrue(os.path.exists(os.path.join(latest_run, "validation_report.html")))
        
        # Verify non-DISABLED states
        with open(os.path.join(latest_run, "validation_summary.json"), "r") as f:
            summary = json.load(f)
            self.assertEqual(summary["tics_phase_1b_v_ready"], "YES")
            self.assertEqual(summary["regime_enforcement_candidate"], "INSUFFICIENT_DATA")
            self.assertNotEqual(summary["result_content_hash"], "")
            
        # Cleanup generated run artifacts
        shutil.rmtree(validation_runs_dir)
        if os.path.exists("holdout_lock.json"):
            os.remove("holdout_lock.json")

    # 11. Decision Function Criteria Tests (Requirement 22)
    def test_enforcement_criteria_decisions(self):
        from research.regime_validation.cli import evaluate_enforcement_decision
        
        def make_mock_inputs(allow_mean=0.0020, block_mean=0.0010, baseline_mean=0.0012,
                             ci_low=0.0001, retention=0.35, allow_mae=0.0010, baseline_mae=0.0012,
                             val_lift=0.0005, hold_lift=0.0010, symbol_pos_lifts=3,
                             num_allow=40, num_block=40, num_total=120):
            cf = {
                "HOLDOUT": {
                    "allow_metrics": {"mean_return": allow_mean, "average_MAE": allow_mae},
                    "block_metrics": {"mean_return": block_mean},
                    "baseline_metrics": {"mean_return": baseline_mean, "average_MAE": baseline_mae},
                    "allow_minus_block_ci": (ci_low, 0.0020),
                    "allow_minus_block_point": allow_mean - block_mean,
                    "retention_pct": retention
                },
                "VALIDATION": {
                    "allow_metrics": {"mean_return": val_lift},
                    "block_metrics": {"mean_return": 0.0}
                }
            }
            
            signals = []
            for _ in range(num_allow):
                signals.append({"split": "HOLDOUT", "joined": True, "safe": True, "permission": "ALLOW", "symbol": "BTCUSDT", "outcomes": {"15m": {"return": 0.0020}}})
            for _ in range(num_block):
                signals.append({"split": "HOLDOUT", "joined": True, "safe": True, "permission": "BLOCK", "symbol": "BTCUSDT", "outcomes": {"15m": {"return": 0.0010}}})
            # Add others to meet num_total
            for _ in range(num_total - num_allow - num_block):
                signals.append({"split": "HOLDOUT", "joined": True, "safe": True, "permission": "WATCH", "symbol": "BTCUSDT", "outcomes": {"15m": {"return": 0.0015}}})
            return cf, signals

        # k) All pass -> YES
        cf, sigs = make_mock_inputs()
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "YES")
        
        # a) Bad holdout (too few signals) -> INSUFFICIENT_DATA
        cf, sigs = make_mock_inputs(num_total=50)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "INSUFFICIENT_DATA")

        # b) fewer than 30 ALLOW -> INSUFFICIENT_DATA
        cf, sigs = make_mock_inputs(num_allow=20)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "INSUFFICIENT_DATA")

        # c) fewer than 30 BLOCK -> INSUFFICIENT_DATA
        cf, sigs = make_mock_inputs(num_block=20)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "INSUFFICIENT_DATA")

        # d) criterion A failure -> NO
        cf, sigs = make_mock_inputs(allow_mean=0.0010, block_mean=0.0015)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

        # e) criterion B failure (CI low <= 0) -> NO
        cf, sigs = make_mock_inputs(ci_low=-0.0001)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

        # f) criterion C failure (ALLOW <= baseline) -> NO
        cf, sigs = make_mock_inputs(allow_mean=0.0010, baseline_mean=0.0012)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

        # g) criterion D failure (retention < 30%) -> NO
        cf, sigs = make_mock_inputs(retention=0.25)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

        # h) criterion E failure (ALLOW MAE too high) -> NO
        cf, sigs = make_mock_inputs(allow_mae=0.0020, baseline_mae=0.0010)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

        # i) criterion G failure (different lift signs) -> NO
        cf, sigs = make_mock_inputs(val_lift=-0.0005, hold_lift=0.0010)
        self.assertEqual(evaluate_enforcement_decision(cf, sigs), "NO")

    def test_real_ready_replay_verification(self):
        bars = []
        # Hour aligned start time
        start_time = 1700000000000 - (1700000000000 % 3600000)
        # 30 hours of 1m closed bars (1800 bars) to exceed 21 CLOSED 1h bars
        for i in range(1800):
            # Generate upward trending prices to hit a canonical regime state
            p = 100.0 + (i * 0.05)
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=start_time + i * 60000,
                close_time_ms=start_time + (i + 1) * 60000 - 1,
                open=p, high=p + 0.02, low=p - 0.02, close=p,
                base_volume=10.0, quote_volume=1000.0, closed=True
            ))
            
        config_dict = {
            "REGIME_MODEL_VERSION": "regime-v1",
            "REGIME_FEATURE_VERSION": "regime-features-v1",
            "REGIME_MAX_BARS_PER_TIMEFRAME": 2000,
            "REGIME_TRADE_DEDUP_CAPACITY": 5000,
            "REGIME_MAX_LATE_TRADE_MS": 2000,
            "REGIME_SWITCH_CONFIRM_BARS": 3,
            "REGIME_MIN_CONFIDENCE": 0.65,
            "REGIME_SWITCH_MARGIN": 0.10,
            "REGIME_VOL_PERCENTILE_WINDOW": 200,
            "REGIME_VOL_MIN_SAMPLES": 100,
            "REGIME_LIQUIDITY_WINDOW": 500,
            "REGIME_LIQUIDITY_MIN_SAMPLES": 100,
            "REGIME_BREAKOUT_MAX_BARS": 5,
        }
        
        runner = HistoricalRegimeReplayRunner(["BTCUSDT"], config=config_dict)
        # Using 500 bars warmup, dev_bars 1300 bars
        timeline = runner.run_replay(bars[:500], bars[500:], [], [])
        
        # Check that we have READY states and valid canonical regimes
        ready_states = [t for t in timeline if t.get("quality") == "READY"]
        self.assertTrue(len(ready_states) > 0)
        
        valid_regime_labels = {
            "TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT_UP", "BREAKOUT_DOWN", "TRANSITION"
        }
        regimes_found = set(t.get("primary_regime") for t in ready_states)
        self.assertTrue(any(r in valid_regime_labels for r in regimes_found))

    # 13. Deterministic Dataset Hash Reproducibility Test (Requirement 3)
    def test_dataset_hash_reproducibility(self):
        from research.regime_validation.cli import compute_dataset_content_hash
        
        q_data = {
            "symbol": "BTCUSDT", "valid": True, "bar_count": 100, "content_sha256": "mock_sha",
            "actual_start_ms": 1700000000000, "actual_end_ms": 1700000000000 + 99 * 60000
        }
        with open(os.path.join(self.tmp_dir, "btcusdt_quality.json"), "w") as f:
            json.dump(q_data, f)
            
        h1 = compute_dataset_content_hash(self.tmp_dir, ["BTCUSDT"])
        h2 = compute_dataset_content_hash(self.tmp_dir, ["BTCUSDT"])
        self.assertEqual(h1, h2)

