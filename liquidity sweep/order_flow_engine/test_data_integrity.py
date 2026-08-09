import unittest
import time
from collections import deque
from data.sequence_validator import SequenceValidator
from data.stream_health import StreamHealthTracker
from data.local_order_book import LocalOrderBook
from flow_metrics import FlowMetrics, SymbolFlowState, TradeEvent

class TestOrderBookDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.health = StreamHealthTracker("btcusdt")
        self.book = LocalOrderBook("btcusdt", self.health)

    def test_sequence_validation(self):
        val = SequenceValidator()
        # Initial bootstrap
        self.assertEqual(val.validate_and_update(1, 10, 5), "OK")
        self.assertEqual(val.last_u, 10)

        # Correct subsequent message
        self.assertEqual(val.validate_and_update(11, 20, 10), "OK")
        self.assertEqual(val.last_u, 20)

        # Duplicate message (u <= last_u)
        self.assertEqual(val.validate_and_update(9, 15, 8), "DUPLICATE")

        # Out of order update (pu < last_u but u > last_u)
        self.assertEqual(val.validate_and_update(18, 25, 9), "OUT_OF_ORDER")

        # Sequence gap (pu != last_u)
        self.assertEqual(val.validate_and_update(25, 30, 22), "GAP")

    def test_apply_diff_update_and_level_deletion(self):
        # Seed snapshot content and WS buffer
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["60000.0", "1.5"], ["59900.0", "2.0"]],
            "asks": [["60100.0", "0.8"], ["60200.0", "1.2"]]
        }
        # Buffer the connecting update
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

        # Apply update: replace level quantity
        update_replace = {
            "U": 102, "u": 103, "pu": 101,
            "b": [["60000.0", "3.0"]], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.handle_ws_update(update_replace, time.time())
        self.assertEqual(self.book.bids[60000.0], 3.0)

        # Apply update: delete level (qty = 0.0)
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

        # Gap message
        gap_update = {
            "U": 105, "u": 106, "pu": 103,  # pu should be 101
            "b": [], "a": [],
            "E": 123456789, "T": 123456788
        }
        self.book.handle_ws_update(gap_update, time.time())
        
        # State should transition to RESYNCING / invalid book, book maps cleared
        self.assertEqual(self.book.state, "RESYNCING")
        self.assertFalse(self.book.is_valid)
        self.assertEqual(len(self.book.bids), 0)
        self.assertEqual(len(self.book.asks), 0)


class TestRollingWindowsAndCVD(unittest.TestCase):
    def setUp(self):
        self.metrics = FlowMetrics()
        self.symbol = "btcusdt"
        self.state = self.metrics.get_state(self.symbol)
        
    def test_independent_rolling_windows(self):
        # Insert trades spanning 16 minutes back
        now = time.time()
        
        # Trade 1: 10 minutes ago
        self.metrics.add_trade(self.symbol, {
            "timestamp": now - 600,
            "price": 60000.0,
            "quantity": 1.0,
            "side": "BUY"
        })
        # Trade 2: 30 seconds ago
        self.metrics.add_trade(self.symbol, {
            "timestamp": now - 30,
            "price": 60000.0,
            "quantity": 2.0,
            "side": "SELL"
        })

        m1m = self.metrics.get_metrics_for_window(self.symbol, "1m")
        m5m = self.metrics.get_metrics_for_window(self.symbol, "5m")
        m15m = self.metrics.get_metrics_for_window(self.symbol, "15m")

        # 1m window should only contain Trade 2 ($120,000 SELL)
        self.assertEqual(m1m["trade_count"], 1)
        self.assertEqual(m1m["delta_usdt"], -120000.0)

        # 5m window should only contain Trade 2 ($120,000 SELL)
        self.assertEqual(m5m["trade_count"], 1)
        self.assertEqual(m5m["delta_usdt"], -120000.0)

        # 15m window should contain both Trade 1 ($60,000 BUY) and Trade 2 ($120,000 SELL)
        self.assertEqual(m15m["trade_count"], 2)
        self.assertEqual(m15m["delta_usdt"], -60000.0)  # +60K - 120K

    def test_warmup_state(self):
        # SymbolState starts with start_time = current time
        m15m = self.metrics.get_metrics_for_window(self.symbol, "15m")
        self.assertEqual(m15m["status"], "WARMING_UP")
        self.assertTrue(m15m["warmup_text"].startswith("WARMING"))

    def test_session_cvd_vs_rolling_delta(self):
        now = time.time()
        # Trade 1: 20 minutes ago (outside rolling windows)
        self.metrics.add_trade(self.symbol, {
            "timestamp": now - 1200,
            "price": 50000.0,
            "quantity": 1.0,
            "side": "BUY"
        })

        # Trade 2: 10 seconds ago
        self.metrics.add_trade(self.symbol, {
            "timestamp": now - 10,
            "price": 50000.0,
            "quantity": 1.0,
            "side": "SELL"
        })

        m15m = self.metrics.get_metrics_for_window(self.symbol, "15m")
        # 15m window should only contain Trade 2 (-$50,000)
        self.assertEqual(m15m["delta_usdt"], -50000.0)

        # Session CVD contains both: +$50,000 (Trade 1) - $50,000 (Trade 2) = $0.0
        self.assertEqual(self.state.session_cvd_usdt, 0.0)


class TestStreamHealth(unittest.TestCase):
    def test_latency_status(self):
        tracker = StreamHealthTracker("btcusdt")
        
        # Test HEALTHY state (latency <= 1000ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=1.1,
            processed_time_ms=1.2, # 200ms age
            update_id=1
        )
        self.assertEqual(tracker.get_status(is_book_valid=True), "HEALTHY")

        # Test DEGRADED state (latency 1000-2500ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=2.1,
            processed_time_ms=2.2, # 1200ms age
            update_id=2
        )
        self.assertEqual(tracker.get_status(is_book_valid=True), "DEGRADED")

        # Test STALE state (latency > 2500ms)
        tracker.record_event(
            event_time_ms=1.0,
            tx_time_ms=1.0,
            received_time_ms=3.9,
            processed_time_ms=4.0, # 3000ms age
            update_id=3
        )
        self.assertEqual(tracker.get_status(is_book_valid=True), "STALE")

        # Test INVALID state
        self.assertEqual(tracker.get_status(is_book_valid=False), "INVALID")


if __name__ == '__main__':
    unittest.main()
