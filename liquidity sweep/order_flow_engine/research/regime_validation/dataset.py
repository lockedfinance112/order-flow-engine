import os
import gzip
import json
import urllib.request
import time
import hashlib
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

from regime.models import MarketBar

def fetch_binance_klines_with_retry(symbol: str, timeframe: str, start_ms: int, end_ms: int, max_retries: int = 5) -> List[dict]:
    """Downloads 1m closed klines from Binance API with bounded retry and backoff."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.upper()}&interval={timeframe}&startTime={start_ms}&endTime={end_ms}&limit=1500"
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            backoff = 2 ** attempt
            time.sleep(backoff)
    return []

class DatasetManager:
    """Manages downloading, caching, splitting, and quality validation of 1m OHLCV datasets."""
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)

    def prepare_dataset(self, symbols: List[str], start_str: str, end_str: str, warmup_days: int) -> Dict[str, Any]:
        """Downloads historical 1m klines, caches them, and builds a manifest."""
        start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        warmup_start_date = start_date - datetime.timedelta(days=warmup_days) if hasattr(datetime, "timedelta") else start_date - datetime.timedelta(days=warmup_days)
        # Using timedelta correctly
        from datetime import timedelta
        warmup_start_date = start_date - timedelta(days=warmup_days)
        warmup_start_ms = int(warmup_start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000) - 1

        manifest = {
            "dataset_id": f"ds_{start_str}_{end_str}",
            "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "Binance USD-M Futures REST",
            "symbols": sorted(symbols),
            "requested_start": start_str,
            "requested_end": end_str,
            "warmup_start": warmup_start_date.strftime("%Y-%m-%d"),
            "files": {}
        }

        for symbol in symbols:
            filepath = os.path.join(self.dataset_dir, f"{symbol.lower()}_1m.jsonl.gz")
            print(f"Downloading {symbol} data from {warmup_start_date} to {end_date}...")
            
            curr_ms = warmup_start_ms
            all_klines = []
            while curr_ms < end_ms:
                klines = fetch_binance_klines_with_retry(symbol, "1m", curr_ms, end_ms)
                if not klines:
                    break
                all_klines.extend(klines)
                curr_ms = klines[-1][6] + 1
                time.sleep(0.2)
                    
            # Deduplicate by open time
            dedup = {}
            for k in all_klines:
                dedup[k[0]] = k
            sorted_klines = [dedup[t] for t in sorted(dedup.keys())]

            # Save as jsonl.gz
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                for k in sorted_klines:
                    f.write(json.dumps(k) + "\n")

        # Save manifest
        manifest_path = os.path.join(self.dataset_dir, "dataset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        return manifest

    def validate_dataset(self, symbol: str, allow_gaps: bool = False, warmup_start_str: str = "", requested_end_str: str = "") -> Dict[str, Any]:
        """Runs strict quality checks on the 1m dataset including requested completeness and non-negative volumes."""
        filepath = os.path.join(self.dataset_dir, f"{symbol.lower()}_1m.jsonl.gz")
        
        quality = {
            "symbol": symbol.upper(),
            "valid": True,
            "bar_count": 0,
            "duplicate_bar_count": 0,
            "missing_bar_count": 0,
            "bad_price_count": 0,
            "open_candle_count": 0,
            "non_monotonic_count": 0,
            "gaps": [],
            "actual_start_ms": 0,
            "actual_end_ms": 0,
            "actual_start_utc": "",
            "actual_end_utc": "",
            "invalid_ohlc_count": 0,
            "content_sha256": ""
        }

        if not os.path.exists(filepath):
            quality["valid"] = False
            return quality

        # Compute SHA256 of decompressed data
        sha = hashlib.sha256()
        last_open_time = -1
        
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                sha.update(line.encode("utf-8"))
                k = json.loads(line)
                open_time = int(k[0])
                close_time = int(k[6])
                open_val = float(k[1])
                high_val = float(k[2])
                low_val = float(k[3])
                close_val = float(k[4])
                volume_val = float(k[5])
                quote_volume_val = float(k[7])
                
                quality["bar_count"] += 1
                if quality["bar_count"] == 1:
                    quality["actual_start_ms"] = open_time
                    quality["actual_start_utc"] = datetime.fromtimestamp(open_time/1000.0, tz=timezone.utc).isoformat()
                quality["actual_end_ms"] = open_time
                quality["actual_end_utc"] = datetime.fromtimestamp(open_time/1000.0, tz=timezone.utc).isoformat()

                if open_time % 60000 != 0:
                    quality["valid"] = False
                    
                if close_time != open_time + 59999:
                    quality["open_candle_count"] += 1
                    quality["valid"] = False

                if last_open_time != -1:
                    if open_time < last_open_time:
                        quality["non_monotonic_count"] += 1
                        quality["valid"] = False
                    elif open_time == last_open_time:
                        quality["duplicate_bar_count"] += 1
                        quality["valid"] = False
                    elif open_time > last_open_time + 60000:
                        gap_start = last_open_time + 60000
                        gap_end = open_time - 60000
                        quality["gaps"].append((gap_start, gap_end))
                        quality["missing_bar_count"] += int((open_time - last_open_time) / 60000) - 1
                        if not allow_gaps:
                            quality["valid"] = False

                # Validate non-negative volumes and OHLC
                if open_val <= 0 or high_val <= 0 or low_val <= 0 or close_val <= 0 or volume_val < 0 or quote_volume_val < 0:
                    quality["bad_price_count"] += 1
                    quality["invalid_ohlc_count"] += 1
                    quality["valid"] = False
                if high_val < max(open_val, close_val) or low_val > min(open_val, close_val) or low_val > high_val:
                    quality["bad_price_count"] += 1
                    quality["invalid_ohlc_count"] += 1
                    quality["valid"] = False

                last_open_time = open_time

        quality["content_sha256"] = sha.hexdigest()
        
        # Enforce requested dataset completeness (Requirement 4)
        if warmup_start_str and requested_end_str:
            warmup_start_dt = datetime.strptime(warmup_start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            requested_end_dt = datetime.strptime(requested_end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            expected_start_ms = int(warmup_start_dt.timestamp() * 1000)
            expected_end_ms = int(requested_end_dt.timestamp() * 1000) - 1
            
            # Allow 1m tolerance
            if quality["actual_start_ms"] > expected_start_ms + 60000 or quality["actual_end_ms"] < expected_end_ms - 60000:
                quality["valid"] = False

        quality_path = os.path.join(self.dataset_dir, f"{symbol.lower()}_quality.json")
        with open(quality_path, "w") as f:
            json.dump(quality, f, indent=4)

        return quality

    def load_bars(self, symbol: str) -> List[MarketBar]:
        """Loads closed klines from file as MarketBar models."""
        filepath = os.path.join(self.dataset_dir, f"{symbol.lower()}_1m.jsonl.gz")
        bars = []
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                k = json.loads(line)
                bars.append(MarketBar(
                    symbol=symbol.lower(),
                    timeframe="1m",
                    open_time_ms=int(k[0]),
                    close_time_ms=int(k[6]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    base_volume=float(k[5]),
                    quote_volume=float(k[7]),
                    closed=True,
                    agg_trade_count=0,
                    exchange_trade_count=int(k[8])
                ))
        return bars

    def get_splits(self, bars: List[MarketBar], warmup_days: int, dev_pct: float, val_pct: float, holdout_pct: float) -> Tuple[List[MarketBar], List[MarketBar], List[MarketBar], List[MarketBar]]:
        """Splits bars chronologically into warmup, development, validation, and holdout segments."""
        if not bars:
            return [], [], [], []
            
        start_time = bars[0].open_time_ms
        warmup_end_ms = start_time + (warmup_days * 24 * 3600 * 1000)
        
        warmup_bars = [b for b in bars if b.open_time_ms < warmup_end_ms]
        eval_bars = [b for b in bars if b.open_time_ms >= warmup_end_ms]
        
        if not eval_bars:
            return warmup_bars, [], [], []
            
        total_eval_duration = eval_bars[-1].open_time_ms - eval_bars[0].open_time_ms
        dev_end_ms = eval_bars[0].open_time_ms + int(total_eval_duration * dev_pct)
        val_end_ms = dev_end_ms + int(total_eval_duration * val_pct)
        
        def round_to_day(ts_ms):
            return (ts_ms // 86400000) * 86400000

        dev_end_ms = round_to_day(dev_end_ms)
        val_end_ms = round_to_day(val_end_ms)

        dev_bars = [b for b in eval_bars if b.open_time_ms < dev_end_ms]
        val_bars = [b for b in eval_bars if dev_end_ms <= b.open_time_ms < val_end_ms]
        holdout_bars = [b for b in eval_bars if b.open_time_ms >= val_end_ms]
        
        return warmup_bars, dev_bars, val_bars, holdout_bars
