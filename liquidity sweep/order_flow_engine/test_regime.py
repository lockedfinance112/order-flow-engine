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
from trade_stream import TradeStream

class TestRegimeEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "REGIME_ENGINE_ENABLED": True,
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

    def tearDown(self):
        asyncio.run(self.engine.stop())

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
                closed=True,
                agg_trade_count=10
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
                closed=True,
                agg_trade_count=10
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
                closed=True,
                agg_trade_count=5
            ))
            
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
                closed=True,
                agg_trade_count=100
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
        bars = self._generate_synthetic_bars(100.0, 0.0, 150, noise=0.01, close_noise=0.01)
        store = self.engine.stores["btcusdt"]
        
        self.engine.config["REGIME_VOL_PERCENTILE_WINDOW"] = 150
        self.engine.config["REGIME_VOL_MIN_SAMPLES"] = 50
        self.engine.config["REGIME_LIQUIDITY_MIN_SAMPLES"] = 50
        
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            self.engine._process_symbol_regime("btcusdt", b.close_time_ms)
            
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
                closed=b.closed,
                agg_trade_count=b.agg_trade_count
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
        
        b_next1 = MarketBar(
            symbol="btcusdt", timeframe="1m",
            open_time_ms=bars[-1].open_time_ms + 60000, close_time_ms=bars[-1].close_time_ms + 60000,
            open=100.0, high=101.0, low=99.0, close=100.0, base_volume=10.0, quote_volume=1000.0,
            closed=True, agg_trade_count=10
        )
        store.append_bar("1m", b_next1)
        
        with patch("regime.engine.classify_regime", return_value=("RANGE", 0.80, "RANGE", "FLAT", {}, ["reasons"])):
            self.engine._process_symbol_regime("btcusdt", b_next1.close_time_ms)
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "TREND_UP")
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["candidate"], "RANGE")
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["count"], 1)
            
            b_next2 = MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=b_next1.open_time_ms + 60000, close_time_ms=b_next1.close_time_ms + 60000,
                open=100.0, high=101.0, low=99.0, close=100.0, base_volume=10.0, quote_volume=1000.0,
                closed=True, agg_trade_count=10
            )
            store.append_bar("1m", b_next2)
            self.engine._process_symbol_regime("btcusdt", b_next2.close_time_ms)
            
            b_next3 = MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=b_next2.open_time_ms + 60000, close_time_ms=b_next2.close_time_ms + 60000,
                open=100.0, high=101.0, low=99.0, close=100.0, base_volume=10.0, quote_volume=1000.0,
                closed=True, agg_trade_count=10
            )
            store.append_bar("1m", b_next3)
            self.engine._process_symbol_regime("btcusdt", b_next3.close_time_ms)
            self.assertEqual(self.engine.hysteresis_state["btcusdt"]["current_regime"], "RANGE")

    def test_audit_a_within_grace_updates_pending(self):
        """TEST A: within-grace previous-minute late trade updates previous pending bar only"""
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 1, "trade_time_ms": 1700002859900, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 2, "trade_time_ms": 1700002860100, "price": 101.0, "quantity": 1.0})
        
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 3, "trade_time_ms": 1700002859950, "price": 95.0, "quantity": 1.0})
        
        pending = self.engine.pending_previous_bars["btcusdt"]
        self.assertIsNotNone(pending)
        self.assertEqual(pending["low"], 95.0)
        
        current = self.engine.current_bars["btcusdt"]
        self.assertEqual(current["open"], 101.0)

    def test_audit_b_beyond_grace_rejected(self):
        """TEST B: late trade beyond grace is rejected"""
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 1, "trade_time_ms": 1700002800000, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 2, "trade_time_ms": 1700002870000, "price": 101.0, "quantity": 1.0})
        
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 3, "trade_time_ms": 1700002850000, "price": 90.0, "quantity": 1.0})
        self.assertEqual(self.engine.stores["btcusdt"].late_trade_count, 1)

    def test_audit_c_out_of_order_ohlc(self):
        """TEST C: same-minute out-of-order OHLC produces correct aggregation"""
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 1, "trade_time_ms": 1700002830000, "price": 105.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 2, "trade_time_ms": 1700002810000, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 3, "trade_time_ms": 1700002850000, "price": 102.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 4, "trade_time_ms": 1700002820000, "price": 110.0, "quantity": 1.0})
        
        current = self.engine.current_bars["btcusdt"]
        self.assertEqual(current["open"], 100.0)
        self.assertEqual(current["high"], 110.0)
        self.assertEqual(current["low"], 100.0)
        self.assertEqual(current["close"], 102.0)

    def test_audit_d_dedup_capacity_eviction_consistency(self):
        """TEST D: deduplication deque/set capacity eviction consistency"""
        self.engine.dedup_capacity = 3
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 1, "trade_time_ms": 1700002800000, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 2, "trade_time_ms": 1700002800001, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 3, "trade_time_ms": 1700002800002, "price": 100.0, "quantity": 1.0})
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 4, "trade_time_ms": 1700002800003, "price": 100.0, "quantity": 1.0})
        
        dq, s_set = self.engine.dedup_ids["btcusdt"]
        self.assertNotIn(1, s_set)
        
        self.engine.on_trade("btcusdt", {"aggregate_trade_id": 1, "trade_time_ms": 1700002800004, "price": 100.0, "quantity": 1.0})
        self.assertEqual(self.engine.stores["btcusdt"].duplicate_trade_count, 0)

    def test_audit_e_queue_overflow_creates_exact_recovery_request(self):
        """TEST E: queue overflow creates exact recovery request"""
        for _ in range(1000):
            self.engine.queue.put_nowait(("btcusdt", None))
            
        bar_dict = {
            "open_time_ms": 1700002800000,
            "close_time_ms": 1700002859999,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "base_volume": 10.0,
            "quote_volume": 1000.0,
            "agg_trade_count": 10
        }
        self.engine._enqueue_bar("btcusdt", bar_dict)
        store = self.engine.stores["btcusdt"]
        self.assertEqual(store.queue_overflow_count, 1)
        self.assertEqual(len(store.recovery_requests), 1)
        self.assertEqual(store.recovery_requests[0]["missing_open_time_ms"], 1700002800000)

    def test_audit_f_g_actual_gap_recovery_lifecycle(self):
        """TEST F & G: actual gap recovery inserts missing candle and returns quality to READY"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        store.unresolved_gaps.clear()
        store.unresolved_gaps.add(("1m", 1600000000000))
            
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        self.assertEqual(self.engine.get_regime_state("btcusdt")["quality"], "DEGRADED")
        
        recovered_bar = MarketBar(
            symbol="btcusdt", timeframe="1m",
            open_time_ms=1600000000000, close_time_ms=1600000059999,
            open=100.0, high=101.0, low=99.0, close=100.0,
            base_volume=10.0, quote_volume=1000.0, closed=True, agg_trade_count=10
        )
        store.append_bar("1m", recovered_bar)
        
        self.assertEqual(store.unresolved_gap_count, 0)
        self.engine.hysteresis_state["btcusdt"]["last_evaluation_close_ms"] = 0
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        self.assertEqual(self.engine.get_regime_state("btcusdt")["quality"], "READY")

    def test_audit_h_missing_1m_component_blocks_5m(self):
        """TEST H: missing 1m component blocks 5m, then recovery generates 5m exactly once"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        for i, b in enumerate(bars):
            if i != 3:
                store.append_bar("1m", b)
                
        self.engine._aggregate_higher_tfs("btcusdt", bars[4])
        self.assertEqual(len(store.get_bars("5m")), 0)
        self.assertIn(("1m", bars[3].open_time_ms), store.unresolved_gaps)
        
        store.append_bar("1m", bars[3])
        self.engine._aggregate_higher_tfs("btcusdt", bars[4])
        self.assertEqual(len(store.get_bars("5m")), 1)

    def test_audit_i_stressed_liquidity_reachable(self):
        """TEST I: STRESSED liquidity state is reachable"""
        self.engine.config["REGIME_LIQUIDITY_MIN_SAMPLES"] = 5
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        baselines = self.engine.liquidity_baselines["btcusdt"]
        baselines["spread"] = [1.5, 1.5, 1.5, 1.5, 1.5]
        baselines["depth"] = [200000.0, 200000.0, 200000.0, 200000.0, 200000.0]
            
        self.liquidity_provider.return_value = {"spread_bps": 10.0, "bid_depth_top5_usdt": 1000.0, "ask_depth_top5_usdt": 1000.0}
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["liquidity"], "STRESSED")

    def test_audit_j_rest_trade_count_semantics(self):
        """TEST J: REST official trade count does not pretend to be aggTrade count"""
        bar = MarketBar(
            symbol="btcusdt", timeframe="1m",
            open_time_ms=1700002800000, close_time_ms=1700002859999,
            open=100.0, high=101.0, low=99.0, close=100.0, base_volume=10.0, quote_volume=1000.0,
            closed=True, agg_trade_count=None, exchange_trade_count=120
        )
        self.assertIsNone(bar.agg_trade_count)
        self.assertEqual(bar.exchange_trade_count, 120)

    def test_audit_k_tradestream_callback_fields(self):
        """TEST K: true TradeStream callback contains aggregate_trade_id, trade_time_ms, event_time_ms"""
        raw = {
            "e": "aggTrade",
            "E": 1700002800100,
            "s": "BTCUSDT",
            "a": 999111,
            "p": "100.5",
            "q": "0.1",
            "f": 100,
            "l": 100,
            "T": 1700002800000,
            "m": True
        }
        
        called_args = []
        async def mock_callback(symbol, parsed):
            called_args.append(parsed)
            
        stream = TradeStream(mock_callback)
        asyncio.run(stream._handle_message("btcusdt", raw))
        
        self.assertEqual(len(called_args), 1)
        parsed = called_args[0]
        self.assertEqual(parsed["aggregate_trade_id"], 999111)
        self.assertEqual(parsed["trade_time_ms"], 1700002800000)
        self.assertEqual(parsed["event_time_ms"], 1700002800100)

    def test_audit_l_true_utc_5m_aggregation(self):
        """TEST L: true UTC 5m aggregation using 5 distinct 1m bars"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        for b in bars:
            store.append_bar("1m", b)
            
        self.engine._aggregate_higher_tfs("btcusdt", bars[4])
        m5 = store.get_bars("5m")
        self.assertEqual(len(m5), 1)
        self.assertEqual(m5[0].open_time_ms, bars[0].open_time_ms)
        self.assertEqual(m5[0].close_time_ms, bars[4].close_time_ms)

    def test_audit_m_true_utc_15m_aggregation(self):
        """TEST M: true UTC 15m aggregation"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 15)
        for b in bars:
            store.append_bar("1m", b)
            
        self.engine._aggregate_higher_tfs("btcusdt", bars[14])
        m15 = store.get_bars("15m")
        self.assertEqual(len(m15), 1)
        self.assertEqual(m15[0].open_time_ms, bars[0].open_time_ms)
        self.assertEqual(m15[0].close_time_ms, bars[14].close_time_ms)

    def test_audit_n_true_utc_1h_aggregation(self):
        """TEST N: true UTC 1h aggregation"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 60)
        for b in bars:
            store.append_bar("1m", b)
            
        self.engine._aggregate_higher_tfs("btcusdt", bars[59])
        m1h = store.get_bars("1h")
        self.assertEqual(len(m1h), 1)
        self.assertEqual(m1h[0].open_time_ms, bars[0].open_time_ms)
        self.assertEqual(m1h[0].close_time_ms, bars[59].close_time_ms)

    def test_audit_o_canonical_close_time_provenance(self):
        """TEST O: canonical close-time provenance populated"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        state = self.engine.get_regime_state("btcusdt")
        self.assertEqual(state["latest_1m_close_time"], bars[-1].close_time_ms)

    def test_audit_p_evaluation_recorder_idempotency(self):
        """TEST P: evaluation/recorder idempotency using close timestamp"""
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        store = self.engine.stores["btcusdt"]
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        last_good = self.engine.last_good_evaluation_ms["btcusdt"]
        
        self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms)
        self.assertEqual(self.engine.last_good_evaluation_ms["btcusdt"], last_good)

    def test_audit_q_engine_disabled_flag(self):
        """TEST Q: engine disabled flag stops activity"""
        cfg_disabled = self.config.copy()
        cfg_disabled["REGIME_ENGINE_ENABLED"] = False
        eng = RegimeEngine(["BTCUSDT"], self.safety_provider, self.liquidity_provider, cfg_disabled)
        
        eng.start()
        self.assertIsNone(eng.worker_task)
        
        state = eng.get_regime_state("btcusdt")
        self.assertEqual(state["quality"], "DISABLED")

    def test_audit_r_clean_async_shutdown(self):
        """TEST R: clean async shutdown leaves no pending loop tasks"""
        async def _run_shutdown():
            self.engine.start()
            await self.engine.stop()
            self.assertIsNone(self.engine.worker_task)
            self.assertIsNone(self.engine.recovery_task)
        asyncio.run(_run_shutdown())

    def test_audit_s_valid_signal_shadow_isolation(self):
        """TEST S: valid-data shadow-mode evaluation check"""
        metrics = FlowMetrics()
        scorer = OrderFlowScorer(metrics)
        
        metrics_5m = {"buy_aggression_ratio": 0.85, "delta_usdt": 50000.0, "status": "READY"}
        imbalance = 1.5
        
        decision_before = scorer.evaluate_bias(
            symbol="BTCUSDT",
            metrics_5m=metrics_5m,
            imbalance=imbalance,
            recent_events=[],
            now=time.time(),
            cooldown_end=0.0
        )
        
        self.engine.current_regimes["btcusdt"] = {
            "primary_regime": "TREND_DOWN",
            "volatility": "EXTREME",
            "liquidity": "STRESSED",
            "quality": "STALE"
        }
        
        decision_after = scorer.evaluate_bias(
            symbol="BTCUSDT",
            metrics_5m=metrics_5m,
            imbalance=imbalance,
            recent_events=[],
            now=time.time(),
            cooldown_end=0.0
        )
        self.assertEqual(decision_before, decision_after)

    # Recovery Finalization Patch targeted tests (1 to 5)

    def test_audit_async_recovery_loop_step(self):
        """TEST 7: test process one recovery request step asynchronously"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        # Skip 12:03 (index 3)
        for i, b in enumerate(bars):
            if i != 3:
                store.append_bar("1m", b)
                
        # Register missing request
        store.unresolved_gaps.add(("1m", bars[3].open_time_ms))
        req = {
            "symbol": "btcusdt",
            "timeframe": "1m",
            "missing_open_time_ms": bars[3].open_time_ms,
            "reason": "STREAM_GAP",
            "attempts": 0,
            "last_attempt_time": 0.0
        }
        store.recovery_requests.append(req)
        
        # Mock fetch response returning the exact missing bar
        with patch("regime.history_loader.fetch_klines_async", return_value=[bars[3]]) as mock_fetch:
            success = asyncio.run(self.engine._process_one_recovery_request("btcusdt", req))
            self.assertTrue(success)
            self.assertEqual(len(store.get_bars("1m")), 5)
            self.assertNotIn(("1m", bars[3].open_time_ms), store.unresolved_gaps)

    def test_audit_automatic_parent_rebuild(self):
        """TEST 8: interior 1m recovery automatically rebuilds parent 5m, 15m, 1h bar idempotently"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        for i, b in enumerate(bars):
            if i != 3:
                store.append_bar("1m", b)
                
        # Before recovery: canonical 5m 12:00 is absent
        self.assertEqual(len(store.get_bars("5m")), 0)
        
        # Run recovery for 12:03
        req = {
            "symbol": "btcusdt",
            "timeframe": "1m",
            "missing_open_time_ms": bars[3].open_time_ms,
            "reason": "STREAM_GAP"
        }
        with patch("regime.history_loader.fetch_klines_async", return_value=[bars[3]]):
            success = asyncio.run(self.engine._process_one_recovery_request("btcusdt", req))
            self.assertTrue(success)
            
        # After recovering 12:03: parent 5m 12:00 is present
        self.assertEqual(len(store.get_bars("5m")), 1)
        
        # Run reconstruction again -> remains exactly one (idempotent)
        self.engine._rebuild_parent_timeframes_for_1m("btcusdt", bars[3].open_time_ms)
        self.assertEqual(len(store.get_bars("5m")), 1)

    def test_audit_empty_response_retry(self):
        """TEST 9: empty recovery response requeues and increments attempt count"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        store.unresolved_gaps.add(("1m", bars[3].open_time_ms))
        
        req = {
            "symbol": "btcusdt",
            "timeframe": "1m",
            "missing_open_time_ms": bars[3].open_time_ms,
            "reason": "STREAM_GAP",
            "attempts": 0,
            "last_attempt_time": 0.0
        }
        
        # First attempt: returns empty response
        with patch("regime.history_loader.fetch_klines_async", return_value=[]):
            success = asyncio.run(self.engine._process_one_recovery_request("btcusdt", req))
            self.assertFalse(success)
            self.assertEqual(req["attempts"], 1)
            self.assertIn(("1m", bars[3].open_time_ms), store.unresolved_gaps)
            self.assertEqual(len(store.get_bars("5m")), 0)
            
        # Second attempt: returns correct candle
        for i, b in enumerate(bars):
            if i != 3:
                store.append_bar("1m", b)
        with patch("regime.history_loader.fetch_klines_async", return_value=[bars[3]]):
            success = asyncio.run(self.engine._process_one_recovery_request("btcusdt", req))
            self.assertTrue(success)
            self.assertEqual(req["attempts"], 2)
            self.assertNotIn(("1m", bars[3].open_time_ms), store.unresolved_gaps)
            self.assertEqual(len(store.get_bars("5m")), 1)

    def test_audit_wrong_candle_recovery(self):
        """TEST 10: wrong candle cannot resolve gap"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 5)
        store.unresolved_gaps.add(("1m", bars[3].open_time_ms))
        
        req = {
            "symbol": "btcusdt",
            "timeframe": "1m",
            "missing_open_time_ms": bars[3].open_time_ms,
            "reason": "STREAM_GAP",
            "attempts": 0,
            "last_attempt_time": 0.0
        }
        
        # Returns wrong candle (e.g. index 2)
        with patch("regime.history_loader.fetch_klines_async", return_value=[bars[2]]):
            success = asyncio.run(self.engine._process_one_recovery_request("btcusdt", req))
            self.assertFalse(success)
            self.assertEqual(req["attempts"], 1)
            self.assertIn(("1m", bars[3].open_time_ms), store.unresolved_gaps)

    def test_audit_canonical_freeze(self):
        """TEST 11: unresolved gap freezes canonical regime transitions, persistence, and last_good updates"""
        store = self.engine.stores["btcusdt"]
        bars = self._generate_synthetic_bars(100.0, 1.0, 60, noise=0.01)
        for b in bars:
            store.append_bar("1m", b)
            store.append_bar("5m", b)
            store.append_bar("15m", b)
            store.append_bar("1h", b)
            
        hyst = self.engine.hysteresis_state["btcusdt"]
        hyst["current_regime"] = "TREND_UP"
        hyst["bars_since_regime_start"] = 10
        hyst["last_evaluation_close_ms"] = bars[-1].close_time_ms
        self.engine.last_good_evaluation_ms["btcusdt"] = 555
        
        # Add unresolved gap to freeze mutation
        store.unresolved_gaps.add(("1m", 1600000000000))
        
        # Evaluate regime with new timestamp
        with patch("regime.engine.classify_regime", return_value=("TREND_DOWN", 0.90, "TREND", "DOWN", {"trend_down": 0.90}, ["reasons"])):
            self.engine._process_symbol_regime("btcusdt", bars[-1].close_time_ms + 60000)
            
            # Assert frozen state
            state = self.engine.get_regime_state("btcusdt")
            self.assertEqual(state["primary_regime"], "TREND_UP")
            self.assertEqual(state["persistence_bars"], 10)
            self.assertEqual(self.engine.last_good_evaluation_ms["btcusdt"], 555)
            self.assertEqual(state["quality"], "DEGRADED")
            self.assertEqual(state["tradable"], False)
            
        # Repair the gap
        store.unresolved_gaps.clear()
        
        # Re-evaluate -> classification should resume
        with patch("regime.engine.classify_regime", return_value=("TREND_DOWN", 0.90, "TREND", "DOWN", {"trend_down": 0.90}, ["reasons"])):
            # Append new consecutive bar to trigger classification
            b_new = MarketBar(
                symbol="btcusdt", timeframe="1m",
                open_time_ms=bars[-1].open_time_ms + 60000, close_time_ms=bars[-1].close_time_ms + 60000,
                open=100.0, high=101.0, low=99.0, close=100.0, base_volume=10.0, quote_volume=1000.0,
                closed=True, agg_trade_count=10
            )
            store.append_bar("1m", b_new)
            self.engine._process_symbol_regime("btcusdt", b_new.close_time_ms)
            
            state = self.engine.get_regime_state("btcusdt")
            # Hysteresis confirm_bars is 3, so first transition TREND_DOWN is candidate
            self.assertEqual(state["primary_regime"], "TREND_UP")
            self.assertEqual(state["candidate_regime"], "TREND_DOWN")
            self.assertEqual(state["candidate_count"], 1)
            self.assertEqual(state["quality"], "READY")
            self.assertEqual(state["tradable"], True)
