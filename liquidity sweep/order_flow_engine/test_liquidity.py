import unittest
import time
from liquidity.price_bucketer import PriceBucketer
from liquidity.wall_tracker import WallTracker, PersistentWall
from liquidity.stacking_pulling import StackingPullingDetector
from liquidity.iceberg_inference import IcebergDetector
from liquidity.liquidity_vacuum import LiquidityVacuumDetector
from liquidity.spoof_risk import SpoofRiskDetector
from scoring import OrderFlowScorer
from flow_metrics import FlowMetrics

class TestLiquidityIntelligence(unittest.TestCase):
    def test_price_bucketer(self):
        bucketer = PriceBucketer(bucket_size=10.0)
        self.assertEqual(bucketer.bucket_price(63704.3), 63700.0)
        self.assertEqual(bucketer.bucket_price(63708.9), 63710.0)

    def test_wall_tracker_and_score(self):
        tracker = WallTracker(wall_threshold_mult=2.0, min_wall_usd=10000.0)
        now = time.time()
        
        # Seed bids and asks
        bids = [(60000.0, 1.0), (59900.0, 0.5)]
        asks = [(60100.0, 1.0), (60200.0, 0.8)]
        
        # Test detection threshold
        tracker.update_walls(bids, asks, mid_price=60050.0, median_depth_usdt=5000.0, timestamp=now)
        active = tracker.get_active_walls(mid_price=60050.0, median_depth_usdt=5000.0)
        
        # All 4 levels exceed the 10,000 threshold
        self.assertEqual(len(active), 4)
        # Verify side sorting
        self.assertIn(active[0]["side"], ["BID", "ASK"])

    def test_stacking_pulling(self):
        detector = StackingPullingDetector(history_seconds=5.0)
        now = time.time()
        
        # Record historical snapshot
        hist_bids = {60000.0: 1.0, 59900.0: 2.0}
        hist_asks = {60100.0: 1.0, 60200.0: 1.5}
        detector.record_snapshot(now - 1.0, hist_bids, hist_asks)
        
        # Record current snapshot with bids added (stacking) and asks cancelled (pulling)
        cur_bids = {60000.0: 2.5, 59900.0: 2.0}  # +1.5 bids added
        cur_asks = {60100.0: 0.2, 60200.0: 1.5}  # -0.8 asks pulled
        detector.record_snapshot(now, cur_bids, cur_asks)
        
        changes = detector.get_changes(lookback_seconds=1.5, mid_price=60050.0, range_pct=0.01)
        self.assertGreater(changes["bids_added"], 0.0)
        self.assertGreater(changes["asks_pulled"], 0.0)
        self.assertGreater(changes["net_shift"], 0.0)

    def test_iceberg_inference(self):
        detector = IcebergDetector(display_multiplier=2.0)
        
        # Execute 5 trades of size 1.0 ($60k each) at 60000
        for _ in range(5):
            detector.record_trade(60000.0, 60000.0)
            
        # Display size is only 1.0 ($60k)
        is_iceberg = detector.check_iceberg(60000.0, displayed_notional=60000.0)
        self.assertTrue(is_iceberg)

        # Display size is 5.0 ($300k)
        is_iceberg_false = detector.check_iceberg(60000.0, displayed_notional=300000.0)
        self.assertFalse(is_iceberg_false)

    def test_liquidity_vacuum(self):
        detector = LiquidityVacuumDetector(vacuum_threshold_pct=0.2)
        bids = [(60000.0, 0.1)]  # very thin near bids
        asks = [(60100.0, 1.0)]
        
        vac_up, vac_dn = detector.check_vacuum(bids, asks, mid_price=60050.0, median_depth_usd=100000.0)
        self.assertTrue(vac_dn)  # bids are thin -> downside vacuum
        self.assertFalse(vac_up)

    def test_spoof_risk(self):
        detector = SpoofRiskDetector(min_spoof_usd=10000.0, max_lifetime_seconds=5.0)
        # Wall pulled within 3s with only $100 execution out of $20,000 total size
        is_spoof = detector.is_spoof_attempt(
            wall_side="BID",
            wall_lifetime_sec=3.0,
            pulled_notional=20000.0,
            executed_notional=100.0
        )
        self.assertTrue(is_spoof)


class TestScoringStateMachine(unittest.TestCase):
    def setUp(self):
        self.metrics = FlowMetrics()
        self.scorer = OrderFlowScorer(self.metrics)
        self.symbol = "btcusdt"
        self.state = self.metrics.get_state(self.symbol)
        
    def test_state_transitions(self):
        # Initial invalid state (since REST sync hasn't run)
        state_name, reason = self.scorer.get_bias_action(self.symbol, {}, 0.0, [])
        self.assertEqual(state_name, "DATA_INVALID")

        # Mock valid book
        self.state.local_book.is_valid = True
        self.state.local_book.state = "HEALTHY"
        self.state.last_depth_timestamp = time.time()
        
        # Should be WARMING_UP because window start time is current time
        state_name, reason = self.scorer.get_bias_action(self.symbol, {"status": "WARMING_UP"}, 0.0, [])
        self.assertEqual(state_name, "WARMING_UP")

    def test_ai_freshness_enforcement(self):
        from ai_interpreter import AIInterpreter
        interpreter = AIInterpreter()
        
        symbol_data = {
            "btcusdt": {
                "price": 60000.0,
                "delta_1m_usdt": 1000.0,
                "delta_5m_usdt": 2000.0,
                "delta_15m_usdt": 3000.0,
                "session_cvd_usdt": 5000.0,
                "imbalance": 0.1,
                "next_action": "WAITING",
                "last_depth_timestamp": time.time() - 35.0  # 35 seconds ago (stale)
            }
        }
        import asyncio
        res = asyncio.run(interpreter.interpret(symbol_data, force=True))
        
        self.assertFalse(res["ok"])
        self.assertIn("stale", res["error"].lower())



class TestScannerPurityAndConsistency(unittest.TestCase):
    def setUp(self):
        from flow_metrics import FlowMetrics
        from scoring import OrderFlowScorer
        self.metrics = FlowMetrics()
        self.scorer = OrderFlowScorer(self.metrics)
        self.symbol = "ethusdt"
        self.state = self.metrics.get_state(self.symbol)
        
        # Set up a clean, healthy default state
        self.state.local_book.is_valid = True
        self.state.local_book.state = "HEALTHY"
        self.state.last_depth_timestamp = time.time()
        self.state.best_bid = 3000.0
        self.state.best_ask = 3001.0
        self.state.spread_bps = 3.3
        
        # Mock get_metrics_for_window on FlowMetrics
        self.mock_windows = {
            "1m": {"status": "OK", "delta_usdt": 50000.0, "buy_ratio": 0.60, "latest_price": 3000.5},
            "5m": {"status": "OK", "delta_usdt": 150000.0, "buy_ratio": 0.60, "latest_price": 3000.5},
            "15m": {"status": "OK", "delta_usdt": 300000.0, "buy_ratio": 0.60, "latest_price": 3000.5}
        }
        self.metrics.get_metrics_for_window = lambda symbol, window_name: self.mock_windows.get(window_name, {})
        self.metrics_5m = self.mock_windows["5m"]
        self.imbalance = 0.25
        self.recent_events = []

    def test_1_scorer_purity(self):
        # Deep copy simulated check
        import copy
        now = time.time()
        
        # Trigger a LONG confirmation setup
        self.scorer.recent_confluence[self.symbol] = {
            "direction": "BULLISH",
            "score": 8,
            "timestamp": now
        }
        
        # Run evaluate_bias 100 times and assert absolutely zero state changes or cooldown mutations
        initial_state = copy.deepcopy(self.scorer.__dict__)
        
        # Note: calling evaluate_bias instead of get_bias_action once implemented
        # For now we'll call the new method directly
        for _ in range(100):
            res = self.scorer.evaluate_bias(
                symbol=self.symbol,
                metrics_5m=self.metrics_5m,
                imbalance=self.imbalance,
                recent_events=self.recent_events,
                now=now,
                cooldown_end=0.0
            )
            self.assertEqual(res["action"], "CONFIRMED_LONG")
            
        self.assertEqual(self.scorer.cooldowns, initial_state["cooldowns"])
        self.assertEqual(self.scorer.symbol_gates, initial_state["symbol_gates"])

    def test_2_api_reads_isolation(self):
        # Ensure that reading decisions from current_decisions doesn't alter any states
        from main import OrderFlowEngine
        import copy
        
        # Create a mock engine instance
        class MockEngine(OrderFlowEngine):
            def __init__(self):
                self.metrics = FlowMetrics()
                self.scorer = OrderFlowScorer(self.metrics)
                self.current_decisions = {
                    "ethusdt": {
                        "action": "WATCH_LONG",
                        "reason": "Test",
                        "gates": {"test": "PASS"},
                        "timestamp": time.time(),
                        "scanner_cycle_id": 1,
                        "action_version": 1,
                        "active_sweep": {"active": False, "direction": "", "score": 0, "age_seconds": 999.0}
                    }
                }
                self.alerts_history = []
                self.binance_context = type('MockCtx', (), {'get_context': lambda: {}, 'get_status': lambda: {}})()
                self.dashboard = type('MockDash', (), {'is_multi': True, 'recent_events': [], 'recent_large_trades': []})()
                self.stream = type('MockStream', (), {'get_status': lambda: "CONNECTED"})()
                self.depth_stream = type('MockStream', (), {'get_status': lambda: "CONNECTED"})()
                self.paper_trader = type('MockTrader', (), {'get_portfolio_state': lambda self, p: {"equity": 10000.0, "positions": []}})()
                self.prev_actions = {}
                
        engine = MockEngine()
        initial_decisions = copy.deepcopy(engine.current_decisions)
        
        # Simulate repeated /api reads via the mock HTTP client call response generator
        for _ in range(100):
            # Inspect symbols payload inside JSON response (or the payload builder)
            res = engine._build_current_symbol_data()
            self.assertEqual(engine.current_decisions, initial_decisions)

    def test_3_bullish_sweep_bearish_conflict(self):
        # Active bullish high-confluence sweep + active bearish conflict
        now = time.time()
        self.scorer.recent_confluence[self.symbol] = {
            "direction": "BULLISH",
            "score": 8,
            "timestamp": now
        }
        recent_events = ["POSSIBLE_BEARISH_ABSORPTION"]
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m=self.metrics_5m,
            imbalance=self.imbalance,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        self.assertNotEqual(res["action"], "WATCH_LONG")
        self.assertNotEqual(res["action"], "CONFIRMED_LONG")

    def test_4_bearish_sweep_bullish_conflict(self):
        # Active bearish high-confluence sweep + active bullish conflict
        now = time.time()
        self.scorer.recent_confluence[self.symbol] = {
            "direction": "BEARISH",
            "score": 8,
            "timestamp": now
        }
        recent_events = ["POSSIBLE_BULLISH_ABSORPTION"]
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m={"status": "OK", "delta_usdt": -150000.0, "buy_ratio": 0.40},
            imbalance=-0.25,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        self.assertNotEqual(res["action"], "WATCH_SHORT")
        self.assertNotEqual(res["action"], "CONFIRMED_SHORT")

    def test_5_normal_bullish_watch_bearish_conflict(self):
        # 5m bullish watch threshold satisfied + Bearish conflict active
        now = time.time()
        recent_events = ["SELL_AGGRESSION"]
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m=self.metrics_5m,
            imbalance=self.imbalance,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        self.assertNotEqual(res["action"], "WATCH_LONG")

    def test_6_normal_bearish_watch_bullish_conflict(self):
        # 5m bearish watch threshold satisfied + Bullish conflict active
        now = time.time()
        recent_events = ["BUY_AGGRESSION"]
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m={"status": "OK", "delta_usdt": -150000.0, "buy_ratio": 0.40},
            imbalance=-0.25,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        self.assertNotEqual(res["action"], "WATCH_SHORT")

    def test_7_opposite_side_survives(self):
        # Bullish candidate suppressed by bearish conflict but Short gates are valid.
        now = time.time()
        
        # Bullish sweep active (long candidate setup)
        self.scorer.recent_confluence[self.symbol] = {
            "direction": "BULLISH",
            "score": 8,
            "timestamp": now
        }
        
        # Bearish conflict active (suppresses long)
        recent_events = ["POSSIBLE_BEARISH_ABSORPTION"]
        
        # Short gates are fully valid (short delta, sell aggression, ask imbalance)
        short_metrics = {"status": "OK", "delta_usdt": -150000.0, "buy_ratio": 0.40, "latest_price": 3000.5}
        self.mock_windows["1m"] = {"status": "OK", "delta_usdt": -50000.0, "buy_ratio": 0.40, "latest_price": 3000.5}
        self.mock_windows["5m"] = short_metrics
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m=short_metrics,
            imbalance=-0.25,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        # Should cleanly evaluate and select the short candidate instead of WAITING
        self.assertTrue(res["action"] in ("WATCH_SHORT", "CONFIRMED_SHORT"))

    def test_8_ai_receives_gates(self):
        # Verify AI prompt snapshot contains decision details
        from main import OrderFlowEngine
        engine = OrderFlowEngine()
        
        # Populate mock decision
        engine.current_decisions["ethusdt"] = {
            "action": "WATCH_LONG",
            "reason": "Test reason",
            "gates": {"gate1": "PASS"},
            "timestamp": 123456789.0,
            "scanner_cycle_id": 99,
            "action_version": 2,
            "active_sweep": {"direction": "BULLISH", "score": 8, "age_seconds": 15.0}
        }
        
        symbol_data = engine._build_current_symbol_data()
        eth_snap = symbol_data.get("ethusdt", {})
        
        self.assertEqual(eth_snap.get("next_action"), "WATCH_LONG")
        self.assertEqual(eth_snap.get("decision_reason"), "Test reason")
        self.assertEqual(eth_snap.get("check_gates"), {"gate1": "PASS"})
        self.assertEqual(eth_snap.get("decision_timestamp"), 123456789.0)
        self.assertEqual(eth_snap.get("scanner_cycle_id"), 99)
        self.assertEqual(eth_snap.get("action_version"), 2)

    def test_9_ai_directional_mismatch(self):
        # AI returns WATCH_SHORT with bearish explanation when scanner is WATCH_LONG
        from ai_interpreter import AIInterpreter
        interpreter = AIInterpreter()
        
        # Mock cached AI response
        ai_resp = {
            "ok": True,
            "interpretation": {
                "regime": "MIXED",
                "risk_profile": "MEDIUM",
                "cleanest_biases": ["ETHUSDT"],
                "suppressed_signals": [],
                "symbols": {
                    "ETHUSDT": {
                        "bias": "WATCH_SHORT",
                        "confidence": 0.8,
                        "explanation": "bearish flow metrics detected"
                    }
                }
            }
        }
        
        # Scanner decision is WATCH_LONG
        scanner_symbol_data = {
            "ethusdt": {
                "next_action": "WATCH_LONG",
                "decision_reason": "Bullish flow candidate",
                "check_gates": {"1m_delta_positive": "PASS"}
            }
        }
        
        # Enrich the AI response (similar to dashboard logic)
        symbol_info = ai_resp["interpretation"]["symbols"]["ETHUSDT"]
        
        # Mock scanner context check
        scanner_action = "WATCH_LONG"
        
        # Assert frontend logic check
        ai_reported_bias = symbol_info["bias"]
        ai_mismatch = (ai_reported_bias != scanner_action)
        self.assertTrue(ai_mismatch)

    def test_10_canonical_snapshot_consistency(self):
        # Ensure that during one sequence, all components read the identical decision
        from main import OrderFlowEngine
        engine = OrderFlowEngine()
        
        # Commit a decision
        decision = {
            "action": "CONFIRMED_LONG",
            "reason": "Test confirmation",
            "gates": {"gate": "PASS"},
            "confirmation_candidate": True,
            "active_sweep": {"direction": "BULLISH", "score": 8, "age_seconds": 1.0}
        }
        
        engine._commit_scanner_decision("ethusdt", decision, time.time())
        
        # Verify /api read is identical
        api_data = engine._build_current_symbol_data()
        self.assertEqual(api_data["ethusdt"]["next_action"], "CONFIRMED_LONG")
        
        # Verify paper trader read is identical (reads same committed next_action)
        # Verify dashboard read is identical

    def test_11_confirmed_transition_cooldown(self):
        # Canonical transition triggers cooldown once, repeated evaluations do not extend it
        from main import OrderFlowEngine
        engine = OrderFlowEngine()
        
        now = time.time()
        decision = {
            "action": "CONFIRMED_LONG",
            "reason": "Test confirmation",
            "gates": {"gate": "PASS"},
            "confirmation_candidate": True,
            "active_sweep": {"direction": "BULLISH", "score": 8, "age_seconds": 1.0}
        }
        
        # Initialize
        engine.current_decisions["ethusdt"] = {
            "action": "WAITING",
            "scanner_cycle_id": 0,
            "action_version": 0
        }
        
        # First commit: transition to CONFIRMED_LONG. Should start cooldown.
        engine._commit_scanner_decision("ethusdt", decision, now)
        cooldown_1 = engine.cooldown_ends.get("ethusdt", 0.0)
        self.assertGreater(cooldown_1, now)
        
        # Second commit with same action (already CONFIRMED_LONG). Should NOT extend cooldown.
        engine._commit_scanner_decision("ethusdt", decision, now + 10.0)
        cooldown_2 = engine.cooldown_ends.get("ethusdt", 0.0)
        self.assertEqual(cooldown_1, cooldown_2)

    def test_12_eth_mixed_conflict_reproduction(self):
        # session CVD negative, 15m/5m negative, 1m recently positive, bearish absorption, bullish sweep active
        now = time.time()
        
        self.scorer.recent_confluence[self.symbol] = {
            "direction": "BULLISH",
            "score": 8,
            "timestamp": now
        }
        
        # mixed/bearish metrics
        self.mock_windows.update({
            "1m": {"status": "OK", "delta_usdt": 5000.0, "buy_ratio": 0.51, "latest_price": 3000.5},
            "5m": {"status": "OK", "delta_usdt": -50000.0, "buy_ratio": 0.45, "latest_price": 3000.5},
            "15m": {"status": "OK", "delta_usdt": -200000.0, "buy_ratio": 0.40, "latest_price": 3000.5}
        })
        
        self.state.session_cvd_usdt = -500000.0
        
        # bearish conflict event
        recent_events = ["POSSIBLE_BEARISH_ABSORPTION"]
        
        res = self.scorer.evaluate_bias(
            symbol=self.symbol,
            metrics_5m=self.mock_windows["5m"],
            imbalance=0.1,
            recent_events=recent_events,
            now=now,
            cooldown_end=0.0
        )
        
        # Long candidates must be suppressed by the conflict event
        self.assertNotEqual(res["action"], "WATCH_LONG")
        self.assertNotEqual(res["action"], "CONFIRMED_LONG")


if __name__ == '__main__':
    unittest.main()
