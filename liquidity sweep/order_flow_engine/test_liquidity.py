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


if __name__ == '__main__':
    unittest.main()
