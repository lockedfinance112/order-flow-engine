import unittest
import time
import asyncio
from collections import deque
from unittest.mock import MagicMock

from data.sequence_validator import SequenceValidator
from data.stream_health import StreamHealthTracker
from data.local_order_book import LocalOrderBook, SyncOutcome
from flow_metrics import FlowMetrics, SymbolFlowState, TradeEvent
from scoring import OrderFlowScorer

class TestOrderBookDataIntegrity(unittest.TestCase):
    def setUp(self):
        from flow_metrics import FlowMetrics
        self.metrics = FlowMetrics()
        self.state = self.metrics.get_state("btcusdt")
        self.book = self.state.local_book
        self.health = self.state.depth_health_tracker

    def test_sequence_validation(self):
        val = SequenceValidator()
        self.assertEqual(val.validate_and_update(1, 10, 5), "OK")
        self.assertEqual(val.last_u, 10)
        self.assertEqual(val.validate_and_update(11, 20, 10), "OK")
        self.assertEqual(val.last_u, 20)
        self.assertEqual(val.validate_and_update(9, 15, 8), "DUPLICATE")
        self.assertEqual(val.validate_and_update(18, 25, 9), "OUT_OF_ORDER")
        self.assertEqual(val.validate_and_update(25, 30, 22), "GAP")

    def test_apply_diff_update_and_level_deletion(self):
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["60000.0", "1.5"], ["59900.0", "2.0"]],
            "asks": [["60100.0", "0.8"], ["60200.0", "1.2"]]
        }
        connecting_update = {
            "U": 99, "u": 101, "pu": 98,
            "b": [], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.buffer.append(connecting_update)
        self.book._apply_snapshot(snapshot)
        self.assertTrue(self.book.is_valid)
        self.assertEqual(self.book.bids[60000.0], 1.5)
        self.assertEqual(self.book.asks[60100.0], 0.8)

        update_replace = {
            "U": 102, "u": 103, "pu": 101,
            "b": [["60000.0", "3.0"]], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.handle_ws_update(update_replace, time.time())
        self.assertEqual(self.book.bids[60000.0], 3.0)

        update_delete = {
            "U": 104, "u": 105, "pu": 103,
            "b": [["59900.0", "0.0"]], "a": [["60100.0", "0.0"]],
            "E": 123456790, "T": 123456789
        }
        self.book.handle_ws_update(update_delete, time.time())
        self.assertNotIn(59900.0, self.book.bids)
        self.assertNotIn(60100.0, self.book.asks)

    def test_sequence_gap_forces_resync(self):
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["60000.0", "1.5"]],
            "asks": [["60100.0", "0.8"]]
        }
        connecting_update = {
            "U": 99, "u": 101, "pu": 98,
            "b": [], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.buffer.append(connecting_update)
        self.book._apply_snapshot(snapshot)
        self.assertTrue(self.book.is_valid)

        gap_update = {
            "U": 105, "u": 106, "pu": 103,
            "b": [], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.handle_ws_update(gap_update, time.time())
        self.assertEqual(self.book.state, "RESYNCING")
        self.assertFalse(self.book.is_valid)
        self.assertEqual(len(self.book.bids), 0)
        self.assertEqual(len(self.book.asks), 0)

    # Isolated Trackers Integration Test
    def test_trade_depth_isolation(self):
        # Inject depth update
        depth_update = {
            "U": 1, "u": 10, "pu": 0,
            "b": [], "a": [],
            "E": 1000, "T": 999
        }
        self.book.handle_ws_update(depth_update, 1.0)
        self.assertEqual(self.state.depth_health_tracker.current_update_id, 10)
        
        # Inject trade with different update ID
        trade_data = {
            "timestamp": 2.0,
            "price": 60000.0,
            "quantity": 1.0,
            "side": "BUY",
            "aggregate_trade_id": 999
        }
        self.metrics.add_trade("btcusdt", trade_data)
        
        # Invariant checks
        self.assertEqual(self.state.depth_health_tracker.current_update_id, 10)
        self.assertEqual(self.state.trade_health_tracker.current_update_id, 999)

    # Trade Cannot Mask Stale Depth Test
    def test_trade_cannot_mask_stale_depth(self):
        # Setup valid book
        snapshot = {"lastUpdateId": 100, "bids": [["60000.0", "1.0"]], "asks": [["60100.0", "1.0"]]}
        self.book.buffer.append({"U": 99, "u": 101, "pu": 98, "b": [], "a": [], "E": 1000, "T": 999})
        self.book._apply_snapshot(snapshot)
        self.assertTrue(self.book.is_valid)

        # Inject fresh trade
        now = time.time()
        self.metrics.add_trade("btcusdt", {"timestamp": now, "price": 60000.0, "quantity": 1.0, "side": "BUY", "aggregate_trade_id": 100})
        
        # Mock depth stale
        self.state.depth_health_tracker.last_received_wall_time = now - 5.0
        self.state.trade_health_tracker.last_received_wall_time = now
        
        safety = self.metrics.get_market_data_safety("btcusdt", now)
        self.assertFalse(safety["safe"])
        self.assertEqual(safety["depth_status"], "STALE")
        self.assertEqual(safety["trade_status"], "HEALTHY")
        
        # Verify scorer blocks
        scorer = OrderFlowScorer(self.metrics)
        decision = scorer.evaluate_bias(
            symbol="btcusdt",
            metrics_5m={"status": "VALID", "delta_usdt": 500.0, "buy_ratio": 0.6},
            imbalance=0.2,
            recent_events=[],
            now=now,
            cooldown_end=0.0
        )
        self.assertEqual(decision["action"], "DATA_STALE")

    # Buffer Overflow test
    def test_buffer_overflow(self):
        # Fill buffer past limit (5000)
        for i in range(5005):
            self.book.buffer.append({"U": i, "u": i+1, "pu": i, "b": [], "a": []})
        
        # Try handle ws update to trigger size limit check
        self.book.handle_ws_update({
            "U": 6000, "u": 6001, "pu": 6000,
            "b": [], "a": [],
            "E": 2000, "T": 1999
        }, time.time())
        
        self.assertFalse(self.book.is_valid)
        self.assertEqual(self.book.state, "RESYNCING")
        self.assertEqual(len(self.book.buffer), 0) # cleared completely on overflow

    # Fail closed decision gate tests
    def test_fail_closed_signals(self):
        scorer = OrderFlowScorer(self.metrics)
        now = time.time()
        
        # Mark book valid first so we test staleness
        self.book.is_valid = True
        
        # Setup stale depth state
        self.state.depth_health_tracker.last_received_wall_time = now - 10.0
        decision = scorer.evaluate_bias(
            symbol="btcusdt",
            metrics_5m={"status": "VALID", "delta_usdt": 1000.0, "buy_ratio": 0.8},
            imbalance=0.5,
            recent_events=["BUY_AGGRESSION"],
            now=now,
            cooldown_end=0.0
        )
        self.assertEqual(decision["action"], "DATA_STALE")
        self.assertFalse(decision["confirmation_candidate"])

class TestRollingWindowsAndCVD(unittest.TestCase):
    def setUp(self):
        self.metrics = FlowMetrics()
        self.symbol = "btcusdt"
        self.state = self.metrics.get_state(self.symbol)
        
    def test_independent_rolling_windows(self):
        now = time.time()
        self.metrics.add_trade(self.symbol, {"timestamp": now - 600, "price": 60000.0, "quantity": 1.0, "side": "BUY"})
        self.metrics.add_trade(self.symbol, {"timestamp": now - 30, "price": 60000.0, "quantity": 2.0, "side": "SELL"})

        m1m = self.metrics.get_metrics_for_window(self.symbol, "1m")
        m5m = self.metrics.get_metrics_for_window(self.symbol, "5m")
        m15m = self.metrics.get_metrics_for_window(self.symbol, "15m")

        self.assertEqual(m1m["trade_count"], 1)
        self.assertEqual(m1m["delta_usdt"], -120000.0)
        self.assertEqual(m5m["trade_count"], 1)
        self.assertEqual(m5m["delta_usdt"], -120000.0)
        self.assertEqual(m15m["trade_count"], 2)
        self.assertEqual(m15m["delta_usdt"], -60000.0)

    def test_warmup_state(self):
        m15m = self.metrics.get_metrics_for_window(self.symbol, "15m")
        self.assertEqual(m15m["status"], "WARMING_UP")

class TestStreamHealth(unittest.TestCase):
    def test_latency_status(self):
        tracker = StreamHealthTracker("btcusdt")
        
        # Test HEALTHY state (latency <= 1000ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=1.1,
            processed_time_ms=1.2,
            update_id=1
        )
        tracker._clock = 1.2
        self.assertEqual(tracker.get_status(is_book_valid=True), "HEALTHY")

        # Test DEGRADED state (latency 1000-2500ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=2.1,
            processed_time_ms=2.2,
            update_id=2
        )
        tracker._clock = 2.2
        self.assertEqual(tracker.get_status(is_book_valid=True), "DEGRADED")

        # Test STALE state (latency > 2500ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=3.9,
            processed_time_ms=4.0,
            update_id=3
        )
        tracker._clock = 4.0
        self.assertEqual(tracker.get_status(is_book_valid=True), "STALE")

        # Test INVALID state
        self.assertEqual(tracker.get_status(is_book_valid=False), "INVALID")

    def test_no_first_data_health(self):
        tracker = StreamHealthTracker("btcusdt")
        self.assertEqual(tracker.get_status(), "INITIALISING")

    def test_wall_clock_silence(self):
        tracker = StreamHealthTracker("btcusdt")
        tracker.record_event(1.0, 1.0, 1.1, 1.2, 1)
        
        # Advance clock to 5.0 seconds
        tracker._clock = 5.0
        self.assertEqual(tracker.get_status(), "STALE")

if __name__ == '__main__':
    unittest.main()
