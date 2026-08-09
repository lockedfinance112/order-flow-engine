import gzip
import json
import os
from typing import List, Dict, Any, Tuple
from flow_metrics import FlowMetrics
from scoring import OrderFlowScorer

class ReplayRunner:
    """
    Deterministic replay runner that plays back gzipped JSONL websocket recordings
    into FlowMetrics and OrderFlowScorer to recreate states and test indicators.
    """
    def __init__(self, recording_filepath: str):
        self.recording_filepath = recording_filepath
        self.metrics = FlowMetrics()
        self.scorer = OrderFlowScorer(self.metrics)
        self.symbol = self._infer_symbol(recording_filepath)
        
        # Track state transitions during replay
        self.transition_log: List[dict] = []
        self.trades_log: List[dict] = []
        self.last_action = "WAITING"

    def _infer_symbol(self, filepath: str) -> str:
        basename = os.path.basename(filepath)
        # recording_btcusdt_2026-08-03.jsonl.gz
        parts = basename.split("_")
        if len(parts) >= 2:
            return parts[1].lower()
        return "btcusdt"

    def run_replay(self) -> Tuple[List[dict], Dict[str, Any]]:
        """
        Plays back all events chronologically.
        Returns transition logs and final metrics.
        """
        if not os.path.exists(self.recording_filepath):
            raise FileNotFoundError(f"Recording file not found: {self.recording_filepath}")

        # Ensure local book doesn't try to trigger live REST snapshot fetches
        state = self.metrics.get_state(self.symbol)
        state.local_book.is_valid = True
        state.local_book.state = "HEALTHY"

        with gzip.open(self.recording_filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev = json.loads(line)
                
                stream = ev.get("stream", "")
                data = ev.get("data", {})
                receive_time = ev.get("receive_time", 0.0)
                
                # Update metrics engine based on stream type
                if "depth" in stream:
                    # Update local depth
                    self.metrics.update_depth(self.symbol, data, receive_time)
                elif "trade" in stream or "aggTrade" in stream:
                    # Map WebSocket trade event payload to add_trade expected format
                    # e.g., {'e': 'aggTrade', 'E': ..., 's': 'BTCUSDT', 'a': ..., 'p': '63700.0', 'q': '1.5', 'f': ..., 'l': ..., 'T': ..., 'm': True}
                    trade_payload = {
                        "price": float(data.get("p", 0.0)),
                        "quantity": float(data.get("q", 0.0)),
                        "side": "SELL" if data.get("m", True) else "BUY",
                        "timestamp": float(data.get("T", 0.0)) / 1000.0,
                        "aggregate_trade_id": int(data.get("a", 0)),
                        "received_time": receive_time
                    }
                    self.metrics.add_trade(self.symbol, trade_payload)
                    self.trades_log.append(trade_payload)

                # Update the state machine bias evaluations
                m5m = self.metrics.get_metrics_for_window(self.symbol, "5m")
                bias_action, reason = self.scorer.get_bias_action(
                    symbol=self.symbol,
                    metrics_5m=m5m,
                    imbalance=state.bid_ask_imbalance,
                    recent_events=[]
                )
                
                # Check for state transitions
                if bias_action != self.last_action:
                    self.transition_log.append({
                        "timestamp": receive_time,
                        "old_action": self.last_action,
                        "new_action": bias_action,
                        "price": state.best_bid if bias_action.startswith("CONFIRMED") else (state.best_bid + state.best_ask)/2.0,
                        "reason": reason
                    })
                    self.last_action = bias_action

        # Collect final summary state
        summary_state = {
            "symbol": self.symbol,
            "final_action": self.last_action,
            "total_transitions": len(self.transition_log),
            "bids_count": len(state.local_book.bids),
            "asks_count": len(state.local_book.asks),
            "running_cvd": state.running_cvd_usdt,
            "session_cvd": state.session_cvd_usdt
        }
        return self.transition_log, summary_state
