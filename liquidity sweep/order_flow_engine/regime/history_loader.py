import asyncio
import json
import logging
import urllib.request
import time
from typing import List, Optional
from regime.models import MarketBar

logger = logging.getLogger("OrderFlow.HistoryLoader")

async def fetch_klines_async(
    symbol: str, 
    timeframe: str, 
    limit: int = 500, 
    retries: int = 3, 
    timeout_sec: float = 10.0
) -> List[MarketBar]:
    """
    Asynchronously fetches historical klines from Binance USD-M Futures REST API.
    Excludes the currently open forming candle.
    """
    symbol_upper = symbol.upper()
    # Map timeframe to Binance interval string
    interval = timeframe
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol_upper}&interval={interval}&limit={limit + 2}"
    
    backoff = 1.0
    for attempt in range(retries):
        try:
            def _fetch():
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                    return json.loads(response.read().decode())
                    
            klines = await asyncio.to_thread(_fetch)
            
            # Map klines to MarketBar objects
            now_ms = int(time.time() * 1000)
            bars = []
            
            for k in klines:
                open_time_ms = int(k[0])
                close_time_ms = int(k[6])
                
                # Exclude the currently forming open candle
                if close_time_ms >= now_ms:
                    continue
                    
                bar = MarketBar(
                    symbol=symbol.lower(),
                    timeframe=timeframe,
                    open_time_ms=open_time_ms,
                    close_time_ms=close_time_ms,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    base_volume=float(k[5]),
                    quote_volume=float(k[7]),
                    agg_trade_count=int(k[8]),
                    exchange_trade_count=int(k[8]),
                    closed=True
                )
                bars.append(bar)
                
            # Bounded return
            return bars[-limit:]
            
        except Exception as e:
            logger.warning(
                f"[{symbol_upper}] Failed to fetch klines (timeframe={timeframe}, attempt={attempt + 1}/{retries}): {e}"
            )
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2.0
            else:
                raise e
                
    return []
