import csv
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from research.regime_validation.models import ValidationSignal

class SignalLoader:
    """Loads and normalizes trade scanner signals and transitions with timezone awareness."""
    
    @staticmethod
    def parse_iso_timestamp(ts_str: str) -> int:
        """Parses ISO timestamp with strict timezone verification, returning epoch ms."""
        # datetime.fromisoformat supports +00:00 or Z
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError(f"Naive timestamp rejected: {ts_str}")
        # Normalize to UTC
        dt_utc = dt.astimezone(timezone.utc)
        return int(dt_utc.timestamp() * 1000)

    @classmethod
    def load_from_transitions_csv(cls, filepath: str) -> List[ValidationSignal]:
        """Loads signals from bias_transitions.csv transitions."""
        signals = []
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                action = row["new_action"]
                if action not in ("CONFIRMED_LONG", "CONFIRMED_SHORT", "LONG_BIAS", "SHORT_BIAS"):
                    continue
                    
                symbol = row["symbol"].lower()
                ts_ms = cls.parse_iso_timestamp(row["timestamp"])
                price = float(row["price"])
                
                direction = "LONG" if "LONG" in action else "SHORT"
                
                # Check for sweep setup
                event = row.get("latest_event", "")
                if "SWEEP" in event:
                    strategy_family = "sweep_reversal"
                else:
                    strategy_family = "long_momentum" if direction == "LONG" else "short_momentum"
                    
                prov = "RECORDED_DECISION_TRANSITION"
                if "BIAS" in action:
                    prov = "LEGACY_SIGNAL_LOG"

                signals.append(ValidationSignal(
                    symbol=symbol,
                    timestamp_ms=ts_ms,
                    direction=direction,
                    action=action,
                    strategy_family=strategy_family,
                    metadata={
                        "entry_price": price,
                        "entry_cvd": 0.0,
                        "completed_time": "",
                        "signal_id": f"sig_{symbol}_{ts_ms}",
                        "provenance_mode": prov
                    }
                ))
        return signals

    @classmethod
    def load_legacy_signals_csv(cls, filepath: str) -> List[ValidationSignal]:
        """Loads legacy signals from bias_signals.csv."""
        signals = []
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row["symbol"].lower()
                action = row["action"]
                direction = row["direction"]
                ts_ms = cls.parse_iso_timestamp(row["entry_time"])
                entry_price = float(row["entry_price"])

                event = row.get("entry_latest_event", "")
                if "SWEEP" in event:
                    strategy_family = "sweep_reversal"
                else:
                    strategy_family = "long_momentum" if direction == "LONG" else "short_momentum"

                signals.append(ValidationSignal(
                    symbol=symbol,
                    timestamp_ms=ts_ms,
                    direction=direction,
                    action=action,
                    strategy_family=strategy_family,
                    metadata={
                        "entry_price": entry_price,
                        "entry_cvd": float(row.get("entry_cvd", 0.0)),
                        "completed_time": row.get("completed_time", ""),
                        "signal_id": row.get("signal_id", ""),
                        "provenance_mode": "LEGACY_SIGNAL_LOG"
                    }
                ))
        return signals
