import unittest
import os
import gzip
import json
import time
from data.recorder import StreamRecorder
from research.replay_runner import ReplayRunner
from research.performance_analyzer import PerformanceAnalyzer

class TestReplayAndRecorder(unittest.TestCase):
    def setUp(self):
        self.symbol = "btcusdt"
        self.output_dir = "test_recordings"
        self.recorder = StreamRecorder(self.symbol, output_dir=self.output_dir, flush_interval_secs=0.1)

    def tearDown(self):
        # Clean up files created during test
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)
            
        research_dir = "test_research"
        if os.path.exists(research_dir):
            for f in os.listdir(research_dir):
                os.remove(os.path.join(research_dir, f))
            os.rmdir(research_dir)

    def test_gzip_recording_and_replay(self):
        async def run_test():
            self.recorder.start()
            now = time.time()
            trade_msg = {"e": "aggTrade", "E": int(now*1000), "s": "BTCUSDT", "a": 12345, "p": "60000.0", "q": "2.0", "T": int(now*1000), "m": True}
            depth_msg = {"e": "depthUpdate", "E": int(now*1000), "T": int(now*1000), "U": 100, "u": 105, "pu": 99, "b": [["59990.0", "1.0"]], "a": [["60010.0", "1.0"]]}
            
            await self.recorder.record("btcusdt@aggTrade", trade_msg, now)
            await self.recorder.record("btcusdt@depth", depth_msg, now + 0.1)
            await self.recorder.stop()
            
        import asyncio
        asyncio.run(run_test())

        # Verify file creation
        files = os.listdir(self.output_dir)
        self.assertEqual(len(files), 1)
        filepath = os.path.join(self.output_dir, files[0])
        self.assertTrue(filepath.endswith(".jsonl.gz"))

        # Verify deterministic replay runner
        runner = ReplayRunner(filepath)
        transitions, summary = runner.run_replay()
        
        self.assertEqual(summary["symbol"], "btcusdt")
        self.assertEqual(len(runner.trades_log), 1)
        self.assertEqual(runner.trades_log[0]["price"], 60000.0)

    def test_performance_analyzer(self):
        analyzer = PerformanceAnalyzer(self.symbol, output_dir="test_research")
        
        # Fake transition logs (CONFIRMED_LONG signal)
        transitions = [
            {
                "timestamp": 1000.0,
                "old_action": "WAITING",
                "new_action": "CONFIRMED_LONG",
                "price": 60000.0,
                "reason": "Confluence match"
            }
        ]
        # Fake trade sequences
        trades = [
            {"timestamp": 1001.0, "price": 60100.0}, # +100
            {"timestamp": 1005.0, "price": 59950.0}, # -50 (drawdown/MAE)
            {"timestamp": 1100.0, "price": 60200.0}, # +200 (final/MFE)
        ]
        
        evaluated = analyzer.analyze_signals(transitions, trades)
        self.assertEqual(len(evaluated), 1)
        sig_data = evaluated[0]
        
        # Check MFE/MAE/PnL calculations
        self.assertEqual(sig_data["5m"]["mfe_pct"], 0.3333) # (60200-60000)/60000 = 0.3333%
        self.assertEqual(sig_data["5m"]["mae_pct"], -0.0833) # (59950-60000)/60000 = -0.0833%
        self.assertEqual(sig_data["5m"]["pnl_pct"], 0.3333)
        self.assertTrue(sig_data["5m"]["win"])

        # Check aggregate report win rate compilation
        agg = analyzer.compile_aggregate_report(evaluated)
        self.assertEqual(agg["5m_precision"], 100.0)
        self.assertEqual(agg["5m_avg_mfe_pct"], 0.3333)


if __name__ == '__main__':
    unittest.main()
