import os
import csv
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("OrderFlow.RegimeRecorder")

class RegimeRecorder:
    """
    Handles CSV recording of TICS Regime Engine states and transitions.
    Ensures idempotency per symbol + timeframe close timestamp.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.states_csv = os.path.join(output_dir, "regime_states.csv")
        self.transitions_csv = os.path.join(output_dir, "regime_transitions.csv")
        
        # Idempotency caches: symbol -> last_recorded_close_ms
        self.last_state_close_ms: Dict[str, int] = {}
        self.last_transition_close_ms: Dict[str, int] = {}
        
        self._init_files()

    def _init_files(self):
        # Initialize regime_states.csv
        if not os.path.exists(self.states_csv):
            with open(self.states_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "evaluation_time", "latest_1m_close_time", "symbol", "primary_regime",
                    "confidence", "structure", "direction", "volatility", "liquidity",
                    "quality", "tradable", "trend_up_score", "trend_down_score", "range_score",
                    "breakout_up_score", "breakout_down_score", "transition_risk", "persistence_bars",
                    "candidate_regime", "candidate_count", "model_version", "feature_version",
                    "reasons_json", "features_json"
                ])

        # Initialize regime_transitions.csv
        if not os.path.exists(self.transitions_csv):
            with open(self.transitions_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "old_regime", "new_regime", "confidence",
                    "persistence_of_old", "transition_reason", "features_json"
                ])

    def record_state(self, state_dict: Dict[str, Any]):
        symbol = state_dict["symbol"].lower()
        close_ms = state_dict.get("latest_1m_close_time", 0)
        
        if self.last_state_close_ms.get(symbol) == close_ms:
            return # Idempotent skip
            
        self.last_state_close_ms[symbol] = close_ms
        
        try:
            with open(self.states_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    state_dict.get("evaluation_time", ""),
                    close_ms,
                    symbol.upper(),
                    state_dict.get("primary_regime", ""),
                    f"{state_dict.get('confidence', 0.0):.4f}",
                    state_dict.get("structure", ""),
                    state_dict.get("direction", ""),
                    state_dict.get("volatility", ""),
                    state_dict.get("liquidity", ""),
                    state_dict.get("quality", ""),
                    "True" if state_dict.get("tradable", False) else "False",
                    f"{state_dict.get('scores', {}).get('trend_up', 0.0):.4f}",
                    f"{state_dict.get('scores', {}).get('trend_down', 0.0):.4f}",
                    f"{state_dict.get('scores', {}).get('range', 0.0):.4f}",
                    f"{state_dict.get('scores', {}).get('breakout_up', 0.0):.4f}",
                    f"{state_dict.get('scores', {}).get('breakout_down', 0.0):.4f}",
                    f"{state_dict.get('transition_risk', 0.0):.4f}",
                    state_dict.get("persistence_bars", 0),
                    state_dict.get("candidate_regime") or "",
                    state_dict.get("candidate_count", 0),
                    state_dict.get("model_version", ""),
                    state_dict.get("feature_version", ""),
                    json.dumps(state_dict.get("reasons", [])),
                    json.dumps(state_dict.get("features", {}))
                ])
        except Exception as e:
            logger.error(f"Failed to write regime state record: {e}")

    def record_transition(self, trans_dict: Dict[str, Any], close_ms: int):
        symbol = trans_dict["symbol"].lower()
        
        if self.last_transition_close_ms.get(symbol) == close_ms:
            return # Idempotent skip
            
        self.last_transition_close_ms[symbol] = close_ms
        
        try:
            with open(self.transitions_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trans_dict.get("timestamp", ""),
                    symbol.upper(),
                    trans_dict.get("old_regime", ""),
                    trans_dict.get("new_regime", ""),
                    f"{trans_dict.get('confidence', 0.0):.4f}",
                    trans_dict.get("persistence_of_old", 0),
                    trans_dict.get("transition_reason", ""),
                    json.dumps(trans_dict.get("features", {}))
                ])
        except Exception as e:
            logger.error(f"Failed to write regime transition record: {e}")
