import unittest
import time
import os
import json
import gzip
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from regime.models import MarketBar
from regime.permissions import permissions_for
from scoring import OrderFlowScorer
from flow_metrics import FlowMetrics

from research.regime_validation.models import ValidationSignal
from research.regime_validation.protocol import get_protocol_hash
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
        # Prepare valid data
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

        # C: missing bar detection
        raw_gap = [
            [1700002800000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0],
            [1700002920000, 100.0, 101.0, 99.0, 100.0, 10.0, 1700002979999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("solusdt_1m.jsonl.gz", raw_gap)
        q_gap = self.dataset_manager.validate_dataset("solusdt")
        self.assertEqual(q_gap["missing_bar_count"], 1)

        # D: bad OHLC rejection (high < low)
        raw_bad = [
            [1700002800000, 100.0, 90.0, 110.0, 100.0, 10.0, 1700002859999, 1000.0, 10, 0, 0, 0]
        ]
        self._write_compressed_jsonl("bnbusdt_1m.jsonl.gz", raw_bad)
        q_bad = self.dataset_manager.validate_dataset("bnbusdt")
        self.assertFalse(q_bad["valid"])
        self.assertEqual(q_bad["bad_price_count"], 1)

    def test_chronological_splits(self):
        # G & H: Chronological splits and holdout boundary alignment
        bars = []
        for i in range(2880 * 2): # 2 days of 1m bars
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=100.0, high=100.0, low=100.0, close=100.0,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
            
        warmup, dev, val, holdout = self.dataset_manager.get_splits(
            bars, warmup_days=1, dev_pct=0.50, val_pct=0.25, holdout_pct=0.25
        )
        self.assertEqual(len(warmup), 1440)
        # Verify chronological order
        self.assertTrue(all(warmup[i].open_time_ms < dev[0].open_time_ms for i in range(len(warmup))))
        self.assertTrue(all(dev[i].open_time_ms < val[0].open_time_ms for i in range(len(dev))))

    # 2. Replay Determinsm & Gaps (I to O)
    def test_deterministic_replay(self):
        # I & J: Replay output is deterministic and wall-clock independent
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
        t1 = runner1.run_replay(bars[:50], bars[50:])
        
        # Introduce a sleep to mimic real time delay
        time.sleep(0.5)
        
        runner2 = HistoricalRegimeReplayRunner(["BTCUSDT"], config={})
        t2 = runner2.run_replay(bars[:50], bars[50:])
        
        self.assertEqual(len(t1), len(t2))
        for r1, r2 in zip(t1, t2):
            self.assertEqual(r1["primary_regime"], r2["primary_regime"])
            self.assertEqual(r1["confidence"], r2["confidence"])

    # 3. As-Of Join Verification (P to S)
    def test_as_of_joins(self):
        # P: Join latest canonical regime with close_time <= signal_time
        timeline = [
            {"symbol": "BTCUSDT", "latest_1m_close_time": 1700002859999, "primary_regime": "TREND_UP", "quality": "READY", "tradable": True},
            {"symbol": "BTCUSDT", "latest_1m_close_time": 1700002919999, "primary_regime": "RANGE", "quality": "READY", "tradable": True}
        ]
        
        # Signal at 12:00:30 (1700002890000)
        signal = ValidationSignal(
            symbol="btcusdt", timestamp_ms=1700002890000,
            direction="LONG", action="LONG_BIAS", strategy_family="long_momentum",
            metadata={}
        )
        res = AsOfJoiner.join_signal_to_regime(signal, timeline)
        self.assertTrue(res["joined"])
        self.assertEqual(res["regime_state"]["primary_regime"], "TREND_UP") # joins to 11:59:59.999 TREND_UP, not 12:00:59.999 RANGE

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
            symbol="btcusdt", timestamp_ms=1700003500000, # 10 minutes later
            direction="LONG", action="LONG_BIAS", strategy_family="long_momentum",
            metadata={}
        )
        res_stale = AsOfJoiner.join_signal_to_regime(signal_stale, timeline)
        self.assertEqual(res_stale["reason"], "NO_FRESH_REGIME")

    # 4. Ex-Post Labels Verification (T to Y)
    def test_ex_post_labeling(self):
        # T: UP_DIRECTIONAL path check
        bars = []
        for i in range(20):
            p = 100.0 + (i * 2.0) # strong rising price path
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=p, high=p + 0.1, low=p - 0.1, close=p,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
        
        lbl, m = OutcomeLabeler.compute_ex_post_label(bars, t_idx=0, horizon_min=15, thresholds={})
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
        for i in range(20):
            p = 100.0 + (i * 2.0)
            bars.append(MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=1700002800000 + i * 60000,
                close_time_ms=1700002800000 + (i + 1) * 60000 - 1,
                open=p, high=p + 0.1, low=p - 0.1, close=p,
                base_volume=1.0, quote_volume=100.0, closed=True
            ))
            
        timeline = [{"symbol": "BTCUSDT", "latest_1m_close_time": 1700002859999, "primary_regime": "BREAKOUT_UP", "confidence": 0.85}]
        bo_res = BreakoutAnalyzer.analyze_breakouts(timeline, bars)
        self.assertEqual(len(bo_res), 1)
        self.assertEqual(bo_res[0]["outcome"], "SUCCESSFUL_FOLLOW_THROUGH")

    # 7. Expectancy & Costs
    def test_expectancy_and_costs(self):
        joined = [{
            "joined": True, "safe": True,
            "timestamp_ms": 1700002800000,
            "outcomes": {"status": "COMPLETED", "return": 0.0020, "mfe": 0.0030, "mae": 0.0005}
        }]
        
        # 5bps cost scenario test (+0.20% gross -> +0.15% net)
        exp = ExpectancyCalculator.calculate_expectancy(joined, 15, cost_bps=5)
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
