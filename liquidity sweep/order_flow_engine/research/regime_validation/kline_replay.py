import csv
import os
import time
from typing import List, Dict, Any, Tuple
from regime.engine import RegimeEngine
from regime.models import MarketBar
from research.regime_validation.replay_clock import ReplayClock

class HistoricalRegimeReplayRunner:
    """Replays historical closed bars chronologically through a localized mock RegimeEngine."""
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        self.symbols = [s.lower() for s in symbols]
        self.config = config
        self.clock = ReplayClock()
        
        # Safety Mode explicitly labeled
        self.safety_mode = "HISTORICAL_CLOSED_BAR_VALIDATION"
        
        # Mocks
        self.safety_provider = lambda sym: {"safe": True, "book_valid": True, "reason": "", "trade_status": "HEALTHY", "depth_status": "HEALTHY"}
        self.liquidity_provider = lambda sym: {"spread_bps": 1.5, "bid_depth_top5_usdt": 100000.0, "ask_depth_top5_usdt": 100000.0}

        self.engine = RegimeEngine(
            symbols=self.symbols,
            market_data_safety_provider=self.safety_provider,
            liquidity_provider=self.liquidity_provider,
            config=self.config
        )
        # Force disable active loops
        self.engine.enabled = False

    def run_replay(self, warmup_bars: List[MarketBar], eval_bars: List[MarketBar]) -> List[Dict[str, Any]]:
        """Preloads warmup bars, then replays evaluation bars step-by-step."""
        timeline = []
        
        # Preload warmup
        for bar in warmup_bars:
            store = self.engine.stores[bar.symbol.lower()]
            store.append_bar("1m", bar)
            self.engine._aggregate_higher_tfs(bar.symbol, bar)
            
        # Replay evaluation bars chronologically
        for bar in eval_bars:
            symbol = bar.symbol.lower()
            store = self.engine.stores[symbol]
            
            # Set clock to bar close time
            self.clock.set_time(bar.close_time_ms)
            
            # Append bar and run evaluation step
            if store.append_bar("1m", bar):
                self.engine._aggregate_higher_tfs(symbol, bar)
                self.engine._process_symbol_regime(symbol, bar.close_time_ms)
                
            state = self.engine.get_regime_state(symbol)
            timeline.append(state.copy())
            
        return timeline

    def save_timeline_csv(self, timeline: List[Dict[str, Any]], filepath: str, protocol_hash: str):
        """Saves evaluation timeline to a CSV artifact."""
        if not timeline:
            return
            
        headers = [
            "evaluation_close_ms", "evaluation_time_utc", "symbol",
            "primary_regime", "structure", "direction", "confidence", "transition_risk",
            "volatility", "liquidity", "quality", "tradable",
            "trend_up_score", "trend_down_score", "range_score", "breakout_up_score", "breakout_down_score",
            "persistence_bars", "candidate_regime", "candidate_count",
            "model_version", "feature_version", "config_hash", "protocol_hash", "source_mode"
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for t in timeline:
                row = {
                    "evaluation_close_ms": t.get("latest_1m_close_time", 0),
                    "evaluation_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t.get("latest_1m_close_time", 0)/1000.0)),
                    "symbol": t.get("symbol", "").upper(),
                    "primary_regime": t.get("primary_regime", "UNKNOWN"),
                    "structure": t.get("structure", "UNKNOWN"),
                    "direction": t.get("direction", "UNKNOWN"),
                    "confidence": t.get("confidence", 0.0),
                    "transition_risk": t.get("transition_risk", 0.0),
                    "volatility": t.get("volatility", "UNKNOWN"),
                    "liquidity": t.get("liquidity", "UNKNOWN"),
                    "quality": t.get("quality", "UNKNOWN"),
                    "tradable": str(t.get("tradable", False)),
                    "trend_up_score": t.get("scores", {}).get("trend_up", 0.0),
                    "trend_down_score": t.get("scores", {}).get("trend_down", 0.0),
                    "range_score": t.get("scores", {}).get("range", 0.0),
                    "breakout_up_score": t.get("scores", {}).get("breakout_up", 0.0),
                    "breakout_down_score": t.get("scores", {}).get("breakout_down", 0.0),
                    "persistence_bars": t.get("persistence_bars", 0),
                    "candidate_regime": t.get("candidate_regime") or "",
                    "candidate_count": t.get("candidate_count", 0),
                    "model_version": t.get("model_version", ""),
                    "feature_version": t.get("feature_version", ""),
                    "config_hash": "default",
                    "protocol_hash": protocol_hash,
                    "source_mode": "HISTORICAL_KLINE_REGIME"
                }
                writer.writerow(row)
