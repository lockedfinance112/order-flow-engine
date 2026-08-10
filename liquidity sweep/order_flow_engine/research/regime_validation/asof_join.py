from typing import List, Dict, Any, Optional
from research.regime_validation.models import ValidationSignal

MAX_SIGNAL_REGIME_AGE_MS = 90000

class AsOfJoiner:
    """Performs strict backward as-of joins from signals to the regime timeline."""
    @staticmethod
    def join_signal_to_regime(
        signal: ValidationSignal,
        timeline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # Filter for evaluations that completed on or before the signal
        candidates = [
            t for t in timeline 
            if t.get("symbol", "").lower() == signal.symbol.lower() and 
            t.get("latest_1m_close_time", 0) <= signal.timestamp_ms
        ]
        
        if not candidates:
            return {
                "joined": False,
                "reason": "NO_REGIME",
                "regime_state": None
            }
            
        # Find the latest one (chronologically closest but before/equal)
        latest_regime = max(candidates, key=lambda x: x.get("latest_1m_close_time", 0))
        regime_close_ms = latest_regime.get("latest_1m_close_time", 0)
        
        # Check maximum age
        age_ms = signal.timestamp_ms - regime_close_ms
        if age_ms > MAX_SIGNAL_REGIME_AGE_MS:
            return {
                "joined": False,
                "reason": "NO_FRESH_REGIME",
                "regime_state": latest_regime
            }
            
        # Check safety and quality
        quality = latest_regime.get("quality", "UNKNOWN")
        tradable = latest_regime.get("tradable", False)
        
        if quality != "READY" or not tradable:
            return {
                "joined": True,
                "safe": False,
                "reason": f"EXCLUDED_{quality}",
                "regime_state": latest_regime
            }
            
        return {
            "joined": True,
            "safe": True,
            "reason": "SUCCESS",
            "regime_state": latest_regime
        }
