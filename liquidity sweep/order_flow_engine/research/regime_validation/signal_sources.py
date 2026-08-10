import csv
from datetime import datetime
from typing import List, Dict, Any
from research.regime_validation.models import ValidationSignal

class SignalLoader:
    """Loads and normalizes trade scanner signals and legacy signal logs."""
    @staticmethod
    def load_from_csv(filepath: str) -> List[ValidationSignal]:
        signals = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row["symbol"].lower()
                action = row["action"]
                direction = row["direction"]
                entry_time_str = row["entry_time"]
                entry_price = float(row["entry_price"])
                
                # Parse entry timestamp
                # format example: 2026-07-02T10:24:37.248399+00:00
                if "+" in entry_time_str:
                    clean_time = entry_time_str.split("+")[0]
                else:
                    clean_time = entry_time_str
                    
                # Support fractional seconds
                try:
                    dt = datetime.strptime(clean_time, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(clean_time, "%Y-%m-%dT%H:%M:%S")
                    
                timestamp_ms = int(dt.timestamp() * 1000)

                # Legacy map compatibility rule
                strategy_family = "long_momentum" if direction == "LONG" else "short_momentum"
                
                # Check for sweep setup in metadata or events
                event = row.get("entry_latest_event", "")
                if "SWEEP" in event:
                    strategy_family = "sweep_reversal"

                signals.append(ValidationSignal(
                    symbol=symbol,
                    timestamp_ms=timestamp_ms,
                    direction=direction,
                    action=action, # legacy or confirmed
                    strategy_family=strategy_family,
                    metadata={
                        "entry_price": entry_price,
                        "entry_cvd": float(row.get("entry_cvd", 0.0)),
                        "completed_time": row.get("completed_time", ""),
                        "signal_id": row.get("signal_id", "")
                    }
                ))
        return signals
