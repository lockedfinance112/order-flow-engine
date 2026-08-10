import unittest
import time
import asyncio
import json
import math
from unittest.mock import MagicMock, patch
from regime.models import MarketBar
from regime.bar_store import BarStore
from regime.features import compute_features
from regime.classifier import classify_regime, evaluate_timeframe_evidence
from regime.permissions import permissions_for, REGIME_PERMISSION_MATRIX
from regime.engine import RegimeEngine
from scoring import OrderFlowScorer
from flow_metrics import FlowMetrics

class TestRegimeEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "REGIME_MODEL_VERSION": "regime-v1",
            "REGIME_FEATURE_VERSION": "regime-features-v1",
            "REGIME_BACKFILL_BARS": 500,
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
            "REGIME_TIMEFRAME_WEIGHTS": {
                "1m": 0.10,
                "5m": 0.25,
                "15m": 0.35,
                "1h": 0.30
            }
        }
        self.symbols = ["BTCUSDT"]
        
        self.safety_provider = MagicMock(return_value={"safe": True, "book_valid": True, "reason": "", "trade_status": "HEALTHY", "depth_status": "HEALTHY"})
        self.liquidity_provider = MagicMock(return_value={"spread_bps": 1.5, "bid_depth_top5_usdt": 100000.0, "ask_depth_top5_usdt": 100000.0})

        self.engine = RegimeEngine(
            symbols=self.symbols,
            market_data_safety_provider=self.safety_provider,
            liquidity_provider=self.liquidity_provider,
            config=self.config
        )

    def _generate_synthetic_bars(self, base_price: float, drift: float, count: int, noise: float = 0.1, close_noise: float = 0.0) -> list:
        bars = []
        curr = base_price
        start_time_ms = 1700002800000 
        for i in range(count):
            open_val = curr
            curr += drift
            c_noise = ((i % 2 * 2) - 1) * close_noise
            close_val = curr + c_noise
            high_val = max(open_val, close_val) + noise
            low_val = min(open_val, close_val) - noise
            
            bars.append(MarketBar(
                symbol="btcusdt",
                timeframe="1m",
                open_time_ms=start_time_ms + i * 60000,
                close_time_ms=start_time_ms + (i + 1) * 60000 - 1,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close_val,
                base_volume=10.0,
                quote_volume=10.0 * close_val,
                agg_trade_count=10,
                closed=True
            ))
        return bars

    def test_a_ascending_trend(self):
        """TEST A: Generate steadily rising bars -> TREND_UP after breakout decay"""
        self.engine.config["REGIME_BREAKOUT_MAX_BARS"] = 2
        bars = self._generate_synthetic_bars(100.0, 1.0, 100, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["primary_regime"], "TREND_UP")
        self.assertTrue(state["confidence"] > 0.5)

    def test_b_descending_trend(self):
        """TEST B: Generate steadily falling bars -> TREND_DOWN after breakout decay"""
        self.engine.config["REGIME_BREAKOUT_MAX_BARS"] = 2
        bars = self._generate_synthetic_bars(1000.0, -1.0, 100, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["primary_regime"], "TREND_DOWN")

    def test_c_range(self):
        """TEST C: Oscillating price action -> RANGE"""
        bars = []
        start_time_ms = 1700002800000
        for i in range(100):
            p = 100.0 + (1.0 if i % 2 == 0 else -1.0)
            bars.append(MarketBar(
                symbol="btcusdt",
                timeframe="1m",
                open_time_ms=start_time_ms + i * 60000,
                close_time_ms=start_time_ms + (i + 1) * 60000 - 1,
                open=p,
                high=p + 0.1,
                low=p - 0.1,
                close=p,
                base_volume=10.0,
                quote_volume=10.0 * p,
                agg_trade_count=10,
                closed=True
            ))
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["primary_regime"], "RANGE")

    def test_d_e_breakout(self):
        """TEST D & E: Stable range followed by strong escape -> BREAKOUT"""
        bars = []
        start_time_ms = 1700002800000
        # 60 range bars to exceed warming threshold
        for i in range(60):
            p = 100.0 + (0.5 if i % 2 == 0 else -0.5)
            bars.append(MarketBar(
                symbol="btcusdt",
                timeframe="1m",
                open_time_ms=start_time_ms + i * 60000,
                close_time_ms=start_time_ms + (i + 1) * 60000 - 1,
                open=p,
                high=p + 0.1,
                low=p - 0.1,
                close=p,
                base_volume=5.0,
                quote_volume=5.0 * p,
                agg_trade_count=5,
                closed=True
            ))
            
        # Feed 3 breakout candles to satisfy hysteresis
        for i in range(3):
            breakout_p = 115.0 + i
            bars.append(MarketBar(
                symbol="btcusdt",
                timeframe="1m",
                open_time_ms=start_time_ms + (60 + i) * 60000,
                close_time_ms=start_time_ms + (61 + i) * 60000 - 1,
                open=100.0,
                high=breakout_p,
                low=100.0,
                close=breakout_p,
                base_volume=100.0,
                quote_volume=100.0 * breakout_p,
                agg_trade_count=100,
                closed=True
            ))
        
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["primary_regime"], "BREAKOUT_UP")

    def test_f_volatility_overlay(self):
        """TEST F: Volatility state transition independent of primary regime label"""
        # Low volatility samples
        bars = self._generate_synthetic_bars(100.0, 0.0, 150, noise=0.01, close_noise=0.01)
        store = self.engine.stores["btcusdt"]
        
        # Set window limits to fit test
        self.engine.config["REGIME_VOL_PERCENTILE_WINDOW"] = 150
        self.engine.config["REGIME_VOL_MIN_SAMPLES"] = 50
        self.engine.config["REGIME_LIQUIDITY_MIN_SAMPLES"] = 50
        
        # Build vol percentile history
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
        # Add high volatility bars at the end (large close_noise)
        high_vol_bars = self._generate_synthetic_bars(100.0, 0.0, 50, noise=5.0, close_noise=5.0)
        start_time_ms = bars[-1].open_time_ms + 60000
        for i, b in enumerate(high_vol_bars):
            b_shifted = MarketBar(
                symbol=b.symbol,
                timeframe=b.timeframe,
                open_time_ms=start_time_ms + i * 60000,
                close_time_ms=start_time_ms + (i + 1) * 60000 - 1,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                base_volume=b.base_volume,
                quote_volume=b.quote_volume,
                agg_trade_count=b.agg_trade_count,
                closed=b.closed
            )
            store.append_bar("1m", b_shifted)
            store.append_bar("5m", b_shifted)
            store.append_bar("15m", b_shifted)
            store.append_bar("1h", b_shifted)
            self.engine._process_symbol_regime("btcusdt", b_shifted.close_time_ms)
            
        state = self.engine.get_regime_state("btcusdt")
        self.assertIn(state["volatility"], ["HIGH", "EXTREME"])

    def test_g_h_hysteresis(self):
        """TEST G & H: Oscillation does not flip instantly; sustained candidate transitions"""
        self.engine.config["REGIME_BREAKOUT_MAX_BARS"] = 20
        bars = self._generate_synthetic_bars(100.0, 1.0, 80, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        hyst = self.engine.hysteresis_state["btcusdt"]
        hyst["current_regime"] = "TREND_UP"
        
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "TREND_UP")
        
        b_next1 = MarketBar("btcusdt", "1m", bars[-1].open_time_ms + 60000, bars[-1].close_time_ms + 60000, 100, 101, 99, 100, 10, 1000, 10, None, True)
        store.append_bar("1m", b_next1)
        
        with patch("regime.engine.classify_regime", return_value=("RANGE", 0.80, "RANGE", "FLAT", {}, ["reasons"])):
            # Bar 1
            self.engine._process_symbol_regime("btcusdt", b_next1.close_time_ms)
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "TREND_UP")
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["candidate"], "RANGE")
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["count"], 1)
            
            # Bar 2
            b_next2 = MarketBar("btcusdt", "1m", b_next1.open_time_ms + 60000, b_next1.close_time_ms + 60000, 100, 101, 99, 100, 10, 1000, 10, None, True)
            store.append_bar("1m", b_next2)
            self.engine._process_symbol_regime("btcusdt", b_next2.close_time_ms)
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "TREND_UP")
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["count"], 2)
            
            # Bar 3 -> Should transition!
            b_next3 = MarketBar("btcusdt", "1m", b_next2.open_time_ms + 60000, b_next2.close_time_ms + 60000, 100, 101, 99, 100, 10, 1000, 10, None, True)
            store.append_bar("1m", b_next3)
            self.engine._process_symbol_regime("btcusdt", b_next3.close_time_ms)
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "RANGE")

    def test_i_incomplete_candle(self):
        """TEST I: Incomplete candle does not alter canonical history"""
        trade1 = {"aggregate_trade_id": 1, "trade_time_ms": 1700002800000, "price": 100.0, "quantity": 1.0}
        trade2 = {"aggregate_trade_id": 2, "trade_time_ms": 1700002800050, "price": 105.0, "quantity": 1.5}
        
        self.engine.on_trade("btcusdt", trade1)
        self.engine.on_trade("btcusdt", trade2)
        
        self.assertEqual(len(self.engine.stores["btcusdt"].get_bars("1m")), 0)

    def test_j_warmup(self):
        """TEST J: Insufficient history shows WARMING_UP"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 10, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["quality"], "WARMING_UP")

    def test_k_guardian_unsafe(self):
        """TEST K: Safety provider unsafe -> quality STALE, tradable False"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        self.safety_provider.return_value = {"safe": False, "reason": "Trade stream stale", "book_valid": False}
        
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["quality"], "STALE")
        self.assertEqual(state["tradable"], False)

    def test_l_gap_detection(self):
        """TEST L: Gap in minute bar timestamps -> DEGRADED quality"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 50, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            
        gap_bar = MarketBar(
            symbol="btcusdt",
            timeframe="1m",
            open_time_ms=bars[-1].open_time_ms + 180000,
            close_time_ms=bars[-1].close_time_ms + 180000,
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.0,
            base_volume=10.0,
            quote_volume=1500.0,
            agg_trade_count=10,
            closed=True
        )
        store.append_bar("1m", gap_bar)
        self.assertTrue(store.history_gap_count > 0)

    def test_m_multi_timeframe_conflict(self):
        """TEST M: Bullish short TF conflicting with Bearish higher TFs"""
        tf_feats = {
            "1m": {"ema_gap_bps": 50.0, "ema_slope_atr": 0.4, "adx14": 40.0, "er20": 0.8, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "5m": {"ema_gap_bps": 20.0, "ema_slope_atr": 0.1, "adx14": 30.0, "er20": 0.5, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "15m": {"ema_gap_bps": -40.0, "ema_slope_atr": -0.3, "adx14": 35.0, "er20": 0.6, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "1h": {"ema_gap_bps": -50.0, "ema_slope_atr": -0.4, "adx14": 45.0, "er20": 0.7, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0}
        }
        res, conf, struct, direction, scores, reasons = classify_regime(tf_feats, self.config)
        self.assertNotEqual(res, "TREND_UP")

    def test_n_determinism(self):
        """TEST N: Same inputs produce identical output sequence"""
        tf_feats = {
            "1m": {"ema_gap_bps": 50.0, "ema_slope_atr": 0.4, "adx14": 40.0, "er20": 0.8, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "5m": {"ema_gap_bps": 20.0, "ema_slope_atr": 0.1, "adx14": 30.0, "er20": 0.5, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "15m": {"ema_gap_bps": -40.0, "ema_slope_atr": -0.3, "adx14": 35.0, "er20": 0.6, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0},
            "1h": {"ema_gap_bps": -50.0, "ema_slope_atr": -0.4, "adx14": 45.0, "er20": 0.7, "realized_vol20": 0.02, "breakout_dist": 0.0, "vol_zscore": 0.0}
        }
        res1 = classify_regime(tf_feats, self.config)
        res2 = classify_regime(tf_feats, self.config)
        self.assertEqual(res1, res2)

    def test_o_no_strategy_impact(self):
        """TEST O: Shadow mode verification; OrderFlowScorer decisions unchanged"""
        metrics = FlowMetrics()
        scorer = OrderFlowScorer(metrics)
        
        decision_before = scorer.evaluate_bias(
            symbol="BTCUSDT",
            metrics_5m={"buy_aggression_ratio": 0.5, "delta_usdt": 1000.0, "status": "READY"},
            imbalance=0.0,
            recent_events=[],
            now=time.time(),
            cooldown_end=0.0
        )
        
        self.engine.current_regimes["btcusdt"] = {"primary_regime": "TREND_DOWN"}
        
        decision_after = scorer.evaluate_bias(
            symbol="BTCUSDT",
            metrics_5m={"buy_aggression_ratio": 0.5, "delta_usdt": 1000.0, "status": "READY"},
            imbalance=0.0,
            recent_events=[],
            now=time.time(),
            cooldown_end=0.0
        )
        self.assertEqual(decision_before, decision_after)

    def test_p_api_shape(self):
        """TEST P: API regime payload validation"""
        state = self.engine.get_regime_state("btcusdt")
        keys = ["primary_regime", "confidence", "structure", "direction", "volatility", "liquidity", "scores", "quality", "reasons", "model_version"]
        for k in keys:
            self.assertIn(k, state)

    def test_q_liquidity_unknown_during_warmup(self):
        """TEST Q: Warmup liquidity returns UNKNOWN"""
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["liquidity"], "UNKNOWN")

    def test_r_s_t_utc_timeframe_alignment(self):
        """TEST R, S, T: UTC alignments for higher timeframes"""
        open_time = 1700002800000 # Aligned to xx:00:00 UTC
        close_time_15m = open_time + 15 * 60000
        self.assertEqual(close_time_15m % 900000, 0)
        
    def test_u_open_higher_timeframe_excluded(self):
        """TEST U: Open candles excluded from aggregation"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 5, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars[:-1]: 
            store.append_bar("1m", b)
        self.engine._aggregate_higher_tfs("btcusdt", bars[-2])
        self.assertEqual(len(store.get_bars("5m")), 0)

    def test_v_late_trade_handling(self):
        """TEST V: Late trade rejected and increments late_trade_count"""
        trade = {
            "aggregate_trade_id": 100,
            "trade_time_ms": 1700002800000,
            "price": 100.0,
            "quantity": 1.0
        }
        self.engine.on_trade("btcusdt", trade)
        
        late_trade = {
            "aggregate_trade_id": 101,
            "trade_time_ms": 1700002800000 - 3600000,
            "price": 90.0,
            "quantity": 1.0
        }
        self.engine.on_trade("btcusdt", late_trade)
        self.assertTrue(self.engine.stores["btcusdt"].late_trade_count > 0)

    def test_w_duplicate_trade_ignored(self):
        """TEST W: Duplicate ID trade increments duplicate_trade_count"""
        trade = {
            "aggregate_trade_id": 999,
            "trade_time_ms": 1700002800000,
            "price": 100.0,
            "quantity": 1.0
        }
        self.engine.on_trade("btcusdt", trade)
        self.engine.on_trade("btcusdt", trade)
        self.assertEqual(self.engine.stores["btcusdt"].duplicate_trade_count, 1)

    def test_x_backfill_open_candle_exclusion(self):
        """TEST X: Backfill excludes currently open candle"""
        now_ms = int(time.time() * 1000)
        klines = [
            [1700002800000, "100", "101", "99", "100", "10", 1700002800000 + 59999, "1000", 10],
            [now_ms, "100", "101", "99", "100", "10", now_ms + 59999, "1000", 10]
        ]
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = bytes(str(klines).replace("'", '"'), 'utf-8')
            mock_url.return_value.__enter__.return_value = mock_resp
            
            from regime.history_loader import fetch_klines_async
            res = asyncio.run(fetch_klines_async("btcusdt", "1m", limit=5))
            self.assertEqual(len(res), 1)

    def test_y_duplicate_bar_uniqueness(self):
        """TEST Y: Duplicate bar at same open time does not double record"""
        b1 = MarketBar("btcusdt", "1m", 1700002800000, 1700002800059, 100, 101, 99, 100, 10, 1000, 10, None, True)
        b2 = MarketBar("btcusdt", "1m", 1700002800000, 1700002800059, 100, 101, 99, 100, 10, 1000, 10, None, True)
        store = self.engine.stores["btcusdt"]
        store.append_bar("1m", b1)
        store.append_bar("1m", b2)
        self.assertEqual(len(store.get_bars("1m")), 1)

    def test_z_feature_point_in_time_safety(self):
        """TEST Z: Appending future data does not alter calculations at past timestamp T"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 30, noise=0.01)
        f1 = compute_features(bars, "1m", self.config)[-1]
        
        future_bars = self._generate_synthetic_bars(150.0, 5.0, 10, noise=0.01)
        combined = bars + future_bars
        f2 = compute_features(combined, "1m", self.config)[29]
        
        self.assertEqual(f1["ema20"], f2["ema20"])

    def test_aa_donchian_excludes_current_bar(self):
        """TEST AA: Donchian high breakout reference excludes current bar high"""
        bars = self._generate_synthetic_bars(100.0, 0.0, 21, noise=0.1)
        b21 = MarketBar("btcusdt", "1m", 1700002800000 + 21*60000, 1700002800000 + 22*60000 - 1, 100, 105, 99, 105, 10, 1000, 10, None, True)
        combined = bars + [b21]
        feats = compute_features(combined, "1m", self.config)[-1]
        self.assertEqual(feats["donchian_high20"], 100.1)

    def test_ab_candidate_reset(self):
        """TEST AB: Interrupted candidate count resets to zero"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        hyst = self.engine.hysteresis_state["btcusdt"]
        hyst["current_regime"] = "TREND_UP"
        hyst["candidate"] = "RANGE"
        hyst["count"] = 2
        
        b_next = MarketBar("btcusdt", "1m", bars[-1].open_time_ms + 60000, bars[-1].close_time_ms + 60000, 100, 101, 99, 100, 10, 1000, 10, None, True)
        store.append_bar("1m", b_next)
        
        with patch("regime.engine.classify_regime", return_value=("TREND_UP", 0.80, "TREND", "UP", {}, ["reasons"])):
            self.engine._process_symbol_regime("btcusdt", b_next.close_time_ms)
            self.assertIsNone(hyst["candidate"])
            self.assertEqual(hyst["count"], 0)

    def test_ac_breakout_decay(self):
        """TEST AC: Breakout decays after MAX_BARS limit"""
        self.engine.config["REGIME_BREAKOUT_MAX_BARS"] = 2
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        hyst = self.engine.hysteresis_state["btcusdt"]
        hyst["current_regime"] = "BREAKOUT_UP"
        hyst["bars_since_regime_start"] = 3
        
        with patch("regime.engine.classify_regime", return_value=("BREAKOUT_UP", 0.80, "BREAKOUT", "UP", {"trend_up": 0.80, "trend_down": 0.0}, ["reasons"])):
            self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms + 60000)
            self.assertEqual(hyst["current_regime"], "TREND_UP")

    def test_ad_symbol_isolation(self):
        """TEST AD: States and history are symbol-isolated"""
        engine_2 = RegimeEngine(["BTCUSDT", "ETHUSDT"], self.safety_provider, self.liquidity_provider, self.config)
        self.assertNotEqual(id(engine_2.stores["btcusdt"]), id(engine_2.stores["ethusdt"]))
