from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ValidationSignal:
    symbol: str
    timestamp_ms: int
    direction: str # LONG or SHORT
    action: str # ALLOW, BLOCK, REDUCE, WATCH
    strategy_family: str
    metadata: Dict[str, Any]

@dataclass
class SignalOutcome:
    symbol: str
    timestamp_ms: int
    direction: str
    entry_price: float
    horizon_returns: Dict[int, Optional[float]] # horizon (min) -> net return
    horizon_mfe: Dict[int, Optional[float]] # horizon (min) -> MFE
    horizon_mae: Dict[int, Optional[float]] # horizon (min) -> MAE
    status: str # COMPLETED or CENSORED
    provenance: str # BAR_APPROX or TRADE_EXACT
