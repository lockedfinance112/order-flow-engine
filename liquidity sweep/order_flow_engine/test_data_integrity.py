import unittest
import time
import asyncio
from collections import deque
from unittest.mock import MagicMock, patch

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
        self.state.trade_health_tracker.last_received_wall_time = now
        
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

    # Blocker 3 Test: Stale trade feed fails closed
    def test_stale_trade_feed_fails_closed(self):
        scorer = OrderFlowScorer(self.metrics)
        now = time.time()

        # Construct: book_valid=True, book_state=HEALTHY, depth_status=HEALTHY, trade_status=STALE
        self.book.is_valid = True
        self.book.state = "HEALTHY"
        self.state.depth_health_tracker.last_received_wall_time = now
        self.state.trade_health_tracker.last_received_wall_time = now - 10.0 # stale

        safety = self.metrics.get_market_data_safety("btcusdt", now)
        self.assertFalse(safety["safe"])
        self.assertEqual(safety["status"], "DATA_STALE")

        # Verify otherwise strongly bullish conditions block
        decision = scorer.evaluate_bias(
            symbol="btcusdt",
            metrics_5m={"status": "VALID", "delta_usdt": 1000000.0, "buy_ratio": 0.9},
            imbalance=0.8,
            recent_events=["BUY_AGGRESSION"],
            now=now,
            cooldown_end=0.0
        )
        self.assertEqual(decision["action"], "DATA_STALE")
        self.assertFalse(decision["confirmation_candidate"])

    # Blocker 4 Test: API aggregate status logic
    def test_api_aggregate_status(self):
        now = time.time()
        
        # Setup BTCUSDT: trade STALE, depth HEALTHY, book HEALTHY
        btc_state = self.metrics.get_state("btcusdt")
        btc_state.local_book.is_valid = True
        btc_state.local_book.state = "HEALTHY"
        btc_state.depth_health_tracker.last_received_wall_time = now
        btc_state.trade_health_tracker.last_received_wall_time = now - 10.0
        
        # Setup ETHUSDT: all HEALTHY
        eth_state = self.metrics.get_state("ethusdt")
        eth_state.local_book.is_valid = True
        eth_state.local_book.state = "HEALTHY"
        eth_state.depth_health_tracker.last_received_wall_time = now
        eth_state.trade_health_tracker.last_received_wall_time = now
        
        # Evaluate aggregate status logic
        guardian_status = "HEALTHY"
        unsafe_symbols = []
        for sym in ["btcusdt", "ethusdt"]:
            safety = self.metrics.get_market_data_safety(sym, now)
            if not safety["safe"]:
                unsafe_symbols.append(sym.upper())
        if unsafe_symbols:
            guardian_status = "UNSAFE"
            
        self.assertEqual(guardian_status, "UNSAFE")
        self.assertIn("BTCUSDT", unsafe_symbols)
        self.assertNotIn("ETHUSDT", unsafe_symbols)

    # Blocker 7 Test: Paper auto-entry safety
    def test_paper_auto_entry_blocked_while_unsafe(self):
        # Setup mock dependencies
        scanner = MagicMock()
        scanner.metrics = self.metrics
        scanner.paper_trader = MagicMock()
        scanner.paper_trader.auto_trade_enabled = True
        scanner.paper_trader.positions = {}
        
        # Setup safety stale trade feed
        self.book.is_valid = True
        self.state.depth_health_tracker.last_received_wall_time = time.time()
        self.state.trade_health_tracker.last_received_wall_time = time.time() - 10.0 # stale
        
        # Call the auto paper trade method
        import main
        main.OrderFlowEngine._run_auto_paper_trade(scanner, "btcusdt", 60000.0, "CONFIRMED_LONG")
        
        # Assert no entry orders were executed
        scanner.paper_trader.execute_order.assert_not_called()

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

class TestOrderBookAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.metrics = FlowMetrics()
        self.state = self.metrics.get_state("btcusdt")
        self.book = self.state.local_book

    # Blocker 5 Test: Second snapshot fetch on RETRY_SNAPSHOT
    async def test_resync_retry_refetches(self):
        bad_snapshot = {"lastUpdateId": 100, "bids": [], "asks": []}
        good_snapshot = {"lastUpdateId": 200, "bids": [["60000.0", "1.0"]], "asks": [["60100.0", "1.0"]]}
        
        fetch_count = 0
        def fake_fetch_snapshot():
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count == 1:
                return bad_snapshot
            if fetch_count == 2:
                # Prepare connecting update in buffer before returning the second snapshot
                self.book.buffer = [{"U": 199, "u": 201, "pu": 198, "b": [], "a": [], "E": 1000, "T": 999}]
                return good_snapshot
            raise AssertionError(f"Unexpected snapshot fetch #{fetch_count}")

        # Overwrite on this specific book instance
        self.book._fetch_snapshot_sync = fake_fetch_snapshot
        
        # Mock _sleep to be async no-op
        async def fake_sleep(seconds):
            return
        self.book._sleep = fake_sleep
        
        # Set state and buffer to trigger retry on first snapshot
        self.book.buffer = [{"U": 105, "u": 106, "pu": 103, "b": [], "a": [], "E": 1000, "T": 999}]
        self.book.state = "RESYNCING"
        
        # Run with a strict timeout
        await asyncio.wait_for(
            self.book._fetch_and_apply_snapshot(),
            timeout=2.0
        )
        
        self.assertEqual(fetch_count, 2)
        self.assertTrue(self.book.is_valid)
        self.assertEqual(self.book.state, "HEALTHY")

    # Blocker 6 Test: Single sync task ownership
    async def test_single_sync_task_ownership(self):
        self.book.state = "RESYNCING"
        async def slow_sync():
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                pass
            
        with patch.object(self.book, '_fetch_and_apply_snapshot', side_effect=slow_sync):
            self.book.trigger_sync()
            task1 = self.book.sync_task
            self.assertIsNotNone(task1)
            
            # Repeat trigger: should reuse existing active task
            self.book.trigger_sync()
            task2 = self.book.sync_task
            self.assertEqual(task1, task2)
            
            # Force trigger: cancels first and spawns replacement
            self.book.trigger_sync(force=True)
            task3 = self.book.sync_task
            await asyncio.sleep(0.01)
            self.assertNotEqual(task1, task3)
            self.assertTrue(task1.cancelled() or task1.done())
            
            # Clean up tasks
            task3.cancel()
            await asyncio.gather(task1, task3, return_exceptions=True)

if __name__ == '__main__':
    unittest.main()
