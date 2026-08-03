import csv
import json
import logging
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


logger = logging.getLogger("OrderFlow.BinanceContext")


class BinanceContextManager:
    """
    Read-only Binance USD-M Futures context layer.
    This enriches dashboard/AI output only; it never changes scanner scoring.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, symbols: List[str], base_dir: Optional[str] = None):
        self.symbols = [s.upper() for s in symbols]
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.cache_file = os.path.join(self.base_dir, "binance_cache.json")
        self.log_csv = os.path.join(self.base_dir, "binance_context.csv")
        self.context: Dict[str, Any] = self._load_cache()
        self.last_fast_refresh = 0.0
        self.last_slow_refresh = 0.0
        self.last_error: Optional[str] = None
        self._ensure_log_headers()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to load Binance context cache: {e}")
        return {
            "ok": False,
            "timestamp": "",
            "symbols": {},
            "status": {
                "quality": "STALE",
                "last_refresh": "",
                "last_error": "No Binance context loaded",
            },
        }

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.context, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Binance context cache: {e}")

    def _ensure_log_headers(self):
        headers = [
            "timestamp",
            "symbol",
            "mark_price",
            "index_price",
            "mark_premium_pct",
            "funding_rate_pct",
            "open_interest",
            "trend_1m",
            "trend_5m",
            "trend_15m",
            "orderbook_snapshot_imbalance",
            "context_quality",
            "context_confirm",
            "warnings_json",
        ]
        if not os.path.exists(self.log_csv):
            try:
                with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(headers)
            except Exception as e:
                logger.warning(f"Failed to initialize Binance context CSV: {e}")

    def _request_json(self, path: str, params: Dict[str, Any]) -> Any:
        import config

        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.BASE_URL}{path}?{query}" if query else f"{self.BASE_URL}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "order-flow-engine-v2.6a"})
        context = None if config.BINANCE_SSL_VERIFY else ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=10, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _trend_from_klines(rows: Any) -> str:
        if not rows:
            return "flat"
        row = rows[-1]
        open_price = BinanceContextManager._to_float(row[1])
        close_price = BinanceContextManager._to_float(row[4])
        if open_price <= 0:
            return "flat"
        change_pct = (close_price - open_price) / open_price * 100.0
        if change_pct >= 0.02:
            return "up"
        if change_pct <= -0.02:
            return "down"
        return "flat"

    @staticmethod
    def _book_imbalance(depth: Any) -> Optional[float]:
        try:
            bid_notional = sum(float(price) * float(qty) for price, qty in depth.get("bids", []))
            ask_notional = sum(float(price) * float(qty) for price, qty in depth.get("asks", []))
            total = bid_notional + ask_notional
            if total <= 0:
                return None
            return (bid_notional - ask_notional) / total
        except Exception:
            return None

    @staticmethod
    def _context_label(action: str, ctx: Dict[str, Any]) -> Optional[str]:
        action = (action or "WAITING").upper()
        if ctx.get("context_quality") != "OK":
            return None
        if action == "WAITING":
            return "INSUFFICIENT"

        trend_5m = ctx.get("trend_5m")
        trend_15m = ctx.get("trend_15m")
        premium = BinanceContextManager._to_float(ctx.get("mark_premium_pct"))
        book_imb = BinanceContextManager._to_float(ctx.get("orderbook_snapshot_imbalance"))
        oi_change = BinanceContextManager._to_float(ctx.get("open_interest_change_pct"))

        long_score = int(trend_5m == "up") + int(trend_15m == "up") + int(book_imb > 0) + int(oi_change > 0)
        short_score = int(trend_5m == "down") + int(trend_15m == "down") + int(book_imb < 0) + int(oi_change > 0)
        if "LONG" in action:
            if long_score >= 3 and premium > -0.05:
                return "CONFIRMED"
            if short_score >= 2 or premium > 0.12:
                return "CONFLICTING"
            return "MIXED"
        if "SHORT" in action:
            if short_score >= 3 and premium < 0.08:
                return "CONFIRMED"
            if long_score >= 2:
                return "CONFLICTING"
            return "MIXED"
        return "INSUFFICIENT"

    def _warnings(self, ctx: Dict[str, Any], ws_imbalance: Optional[float]) -> List[str]:
        warnings = []
        quality = ctx.get("context_quality")
        if quality != "OK":
            warnings.append(f"context quality {quality}")
        funding_pct = self._to_float(ctx.get("funding_rate_pct"))
        premium_pct = self._to_float(ctx.get("mark_premium_pct"))
        drift = ctx.get("book_drift_vs_ws_imbalance")
        if abs(funding_pct) >= 0.02:
            warnings.append("elevated funding")
        if abs(premium_pct) >= 0.10:
            warnings.append("mark/index premium stretched")
        if drift is not None and drift >= 0.30:
            warnings.append("REST/WS book imbalance drift high")
        if ws_imbalance is None:
            warnings.append("local WS imbalance unavailable")
        return warnings

    def _build_symbol_context(
        self,
        symbol: str,
        symbol_data: Dict[str, Any],
        fast_due: bool,
        slow_due: bool,
    ) -> Dict[str, Any]:
        previous = dict(self.context.get("symbols", {}).get(symbol, {}))
        ctx = previous
        now_iso = datetime.now(timezone.utc).isoformat()
        ctx.update({
            "symbol": symbol,
            "timestamp": now_iso,
            "context_quality": "OK",
            "error": None,
        })

        if fast_due:
            mark = self._request_json("/fapi/v1/premiumIndex", {"symbol": symbol})
            depth = self._request_json("/fapi/v1/depth", {"symbol": symbol, "limit": 20})
            ticker = self._request_json("/fapi/v1/ticker/bookTicker", {"symbol": symbol})

            mark_price = self._to_float(mark.get("markPrice"))
            index_price = self._to_float(mark.get("indexPrice"))
            premium = ((mark_price - index_price) / index_price * 100.0) if index_price > 0 else None
            snapshot_imbalance = self._book_imbalance(depth)
            ws_imbalance = symbol_data.get("imbalance")
            drift = abs(snapshot_imbalance - ws_imbalance) if snapshot_imbalance is not None and ws_imbalance is not None else None

            ctx.update({
                "mark_price": mark_price,
                "index_price": index_price,
                "mark_premium_pct": premium,
                "funding_rate": self._to_float(mark.get("lastFundingRate")),
                "funding_rate_pct": self._to_float(mark.get("lastFundingRate")) * 100.0,
                "next_funding_time": mark.get("nextFundingTime"),
                "orderbook_snapshot_imbalance": snapshot_imbalance,
                "book_drift_vs_ws_imbalance": drift,
                "book_ticker_bid": self._to_float(ticker.get("bidPrice")),
                "book_ticker_ask": self._to_float(ticker.get("askPrice")),
                "last_depth_snapshot_time": now_iso,
            })

        if slow_due:
            prev_oi = self._to_float(previous.get("current_open_interest"))
            oi = self._request_json("/fapi/v1/openInterest", {"symbol": symbol})
            klines_1m = self._request_json("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 2})
            klines_5m = self._request_json("/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": 2})
            klines_15m = self._request_json("/fapi/v1/klines", {"symbol": symbol, "interval": "15m", "limit": 2})

            current_oi = self._to_float(oi.get("openInterest"))
            oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100.0) if prev_oi > 0 else None
            ctx.update({
                "current_open_interest": current_oi,
                "open_interest_change_pct": oi_change_pct,
                "trend_1m": self._trend_from_klines(klines_1m),
                "trend_5m": self._trend_from_klines(klines_5m),
                "trend_15m": self._trend_from_klines(klines_15m),
                "last_slow_context_time": now_iso,
            })

        required = ["mark_price", "index_price", "current_open_interest", "trend_5m", "orderbook_snapshot_imbalance"]
        if any(ctx.get(key) is None for key in required):
            ctx["context_quality"] = "STALE"

        ws_imbalance = symbol_data.get("imbalance")
        ctx["warnings"] = self._warnings(ctx, ws_imbalance)
        ctx["context_confirm"] = self._context_label(symbol_data.get("next_action", "WAITING"), ctx)
        return ctx

    def _log_symbol_context(self, ctx: Dict[str, Any]):
        row = [
            datetime.now(timezone.utc).isoformat(),
            ctx.get("symbol", ""),
            ctx.get("mark_price"),
            ctx.get("index_price"),
            ctx.get("mark_premium_pct"),
            ctx.get("funding_rate_pct"),
            ctx.get("current_open_interest"),
            ctx.get("trend_1m", ""),
            ctx.get("trend_5m", ""),
            ctx.get("trend_15m", ""),
            ctx.get("orderbook_snapshot_imbalance"),
            ctx.get("context_quality", ""),
            ctx.get("context_confirm"),
            json.dumps(ctx.get("warnings", [])),
        ]
        try:
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.warning(f"Failed to write Binance context CSV: {e}")

    async def refresh(self, symbol_data: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.refresh_sync, symbol_data, force)

    def refresh_sync(self, symbol_data: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        now = time.time()
        fast_due = force or (now - self.last_fast_refresh >= 30.0)
        slow_due = force or (now - self.last_slow_refresh >= 60.0)
        if not fast_due and not slow_due:
            return self.context

        updated = dict(self.context.get("symbols", {}))
        errors = []

        # V2.6B TODO: add taker buy/sell volume and OI statistics once stable in the adapter.
        for symbol in self.symbols:
            try:
                data = symbol_data.get(symbol.lower()) or symbol_data.get(symbol) or {}
                ctx = self._build_symbol_context(symbol, data, fast_due, slow_due)
                updated[symbol] = ctx
                self._log_symbol_context(ctx)
            except Exception as e:
                msg = f"{symbol}: {e}"
                errors.append(msg)
                previous = dict(updated.get(symbol, {}))
                previous.update({
                    "symbol": symbol,
                    "context_quality": "ERROR",
                    "context_confirm": None,
                    "error": str(e),
                    "warnings": [str(e)],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                updated[symbol] = previous

        if fast_due:
            self.last_fast_refresh = now
        if slow_due:
            self.last_slow_refresh = now
        self.last_error = "; ".join(errors) if errors else None
        ok_count = sum(1 for ctx in updated.values() if ctx.get("context_quality") == "OK")
        quality = "OK" if ok_count else ("ERROR" if errors else "STALE")
        self.context = {
            "ok": ok_count > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": updated,
            "status": {
                "quality": quality,
                "symbols_total": len(self.symbols),
                "symbols_ok": ok_count,
                "last_refresh": datetime.now(timezone.utc).isoformat(),
                "last_error": self.last_error,
                "fast_refresh_seconds": 30,
                "slow_refresh_seconds": 60,
            },
        }
        self._save_cache()
        return self.context

    def get_context(self) -> Dict[str, Any]:
        return self.context

    def get_status(self) -> Dict[str, Any]:
        return self.context.get("status", {})
