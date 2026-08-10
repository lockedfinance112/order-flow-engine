from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MarketBar:
    symbol: str
    timeframe: str

    open_time_ms: int
    close_time_ms: int

    open: float
    high: float
    low: float
    close: float

    base_volume: float
    quote_volume: float
    agg_trade_count: int
    closed: bool
    exchange_trade_count: Optional[int] = None
