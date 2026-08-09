import asyncio
import logging
import os
import sys
import time
import json
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timezone
from rich.live import Live

import config
from config import (
    WINDOWS, EVENT_COOLDOWN_SECONDS, SYMBOLS, 
    DASHBOARD_REFRESH_SECONDS, MAX_RECENT_EVENTS, MAX_DISPLAY_SYMBOLS
)
from trade_stream import TradeStream
from orderbook_stream import OrderBookStream
from flow_metrics import FlowMetrics
from absorption import AbsorptionDetector
from storage import EventStorage
from dashboard import OrderFlowDashboard
from sweeps_monitor import SweepsMonitor
from scoring import OrderFlowScorer
from signal_tracker import SignalTracker
from ai_interpreter import AIInterpreter
from ai_runtime_config import RuntimeAIConfig
import ai_providers
from binance_context import BinanceContextManager
from data.paper_trader import PaperTrader

# Configure file logging to avoid messing up the rich console output
log_file = os.path.join(os.path.dirname(__file__), "order_flow.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("OrderFlow.Main")

class OrderFlowEngine:
    def __init__(self):
        self.metrics = FlowMetrics()
        self.detector = AbsorptionDetector()
        self.storage = EventStorage()
        self.dashboard = OrderFlowDashboard(SYMBOLS)
        self.signal_tracker = SignalTracker()
        self.ai_interpretation_lock = asyncio.Lock()
        self.ai_runtime_config = RuntimeAIConfig()
        self.ai_interpreter = AIInterpreter(
            lock=self.ai_interpretation_lock,
            runtime_config=self.ai_runtime_config
        )
        self.binance_context = BinanceContextManager(SYMBOLS)
        
        # Event Recorders per symbol (v2.9)
        from data.recorder import StreamRecorder
        self.recorders: Dict[str, StreamRecorder] = {}
        if config.RECORDING_ENABLED:
            for sym in SYMBOLS:
                self.recorders[sym.lower()] = StreamRecorder(sym)
        
        # Segregated trade queue and streams
        self.trade_queue = asyncio.Queue()
        self.stream = TradeStream(self._queue_trade)
        self.depth_stream = OrderBookStream(self._handle_depth)
        
        # Sweeps monitoring and scoring
        self.scorer = OrderFlowScorer(self.metrics)
        self.sweeps_monitor = SweepsMonitor(self._handle_sweep)

        # Track event cooldowns: {cooldown_key: last_trigger_timestamp}
        self.last_triggered: Dict[str, float] = {}
        self.last_triggered_price: Dict[str, float] = {}

        # Alert history lookback for scoring: [{"symbol": str, "type": str, "timestamp": float}]
        self.alerts_history: List[Dict[str, Any]] = []
        self.processed_sweep_ids = set()

        # Track previous signal action state per symbol to detect transitions
        self.prev_actions: Dict[str, str] = {}
        self.last_ai_manual_time = 0.0
        self.paper_trader = PaperTrader(10000.0)

        # Canonical decision store & engine states
        self.current_decisions: Dict[str, dict] = {}
        self.cooldown_ends: Dict[str, float] = {}
        self.scanner_cycle_id = 0
        
        # Initialize startup default decision states for all configured symbols
        for sym in SYMBOLS:
            sym_lower = sym.lower()
            self.current_decisions[sym_lower] = {
                "action": "WARMING_UP",
                "reason": "Awaiting first canonical scanner evaluation",
                "gates": {},
                "timestamp": None,
                "scanner_cycle_id": 0,
                "action_version": 0,
                "active_sweep": {"active": False, "direction": "", "score": 0, "age_seconds": 999.0}
            }
            self.cooldown_ends[sym_lower] = 0.0

    def _json_response(self, payload: dict, status: str = "200 OK") -> str:
        body = json.dumps(payload)
        return (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )

    def _plain_response(self, body: str, status: str = "200 OK") -> str:
        return (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )

    def _html_response(self, body: str) -> str:
        return (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )

    def _parse_json_body(self, request: str) -> dict:
        headers, _, body = request.partition("\r\n\r\n")
        content_length = 0
        for line in headers.split("\r\n")[1:]:
            name, _, value = line.partition(":")
            if name.lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:
                    content_length = 0
                break
        if content_length <= 0:
            return {}
        raw_body = body[:content_length]
        if not raw_body.strip():
            return {}
        parsed = json.loads(raw_body)
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object.")
        return parsed

    def _safe_ai_settings_payload(self) -> dict:
        payload = self.ai_runtime_config.to_public_dict()
        payload.update({
            "cache_ttl_seconds": config.AI_CACHE_TTL_SECONDS,
            "interval_seconds": config.AI_INTERVAL_SECONDS,
        })
        return payload

    def _test_ai_provider_config(self) -> dict:
        provider = ai_providers.get_provider(
            self.ai_runtime_config.provider,
            config.AI_TIMEOUT_SECONDS,
            self.ai_runtime_config
        )
        valid, err_msg = provider.validate_config()
        return {
            "ok": bool(valid),
            "provider": self.ai_runtime_config.provider,
            "model": provider.model,
            "configured": self.ai_runtime_config.is_configured(),
            "enabled": self.ai_runtime_config.enabled,
            "error": ai_providers.redact_sensitive(err_msg) if err_msg else None,
        }

    async def _queue_trade(self, symbol: str, trade: dict):
        """Websocket callback to put parsed symbol trade into the queue."""
        now = time.time()
        trade["received_time"] = now
        await self.trade_queue.put((symbol.lower(), trade))
        if config.RECORDING_ENABLED:
            rec = self.recorders.get(symbol.lower())
            if rec:
                await rec.record(f"{symbol.lower()}@aggTrade", trade, now)

    async def _handle_depth(self, symbol: str, depth_data: dict):
        """Websocket callback to update order book pressure metrics directly."""
        now = time.time()
        self.metrics.update_depth(symbol.lower(), depth_data, now)
        if config.RECORDING_ENABLED:
            rec = self.recorders.get(symbol.lower())
            if rec:
                await rec.record(f"{symbol.lower()}@depth", depth_data, now)

    async def _handle_sweep(self, sweep: dict):
        """Triggered when SweepsMonitor detects a new liquidity sweep event."""
        sweep_id = sweep["sweep_id"]
        
        # Ensure we only score this sweep once (prevent duplicates if file updates)
        if sweep_id in self.processed_sweep_ids:
            return
        self.processed_sweep_ids.add(sweep_id)
        
        symbol = sweep["symbol"].lower()
        sweep_type = sweep["type"]
        level = sweep["sweep_level"]
        source_time = sweep["timestamp"]
        
        # Retrieve depth timestamp specifically for this symbol
        state = self.metrics.get_state(symbol)
        last_depth_ts = state.last_depth_timestamp

        # Prune old alerts history (> 15 minutes old)
        now = time.time()
        self.alerts_history = [a for a in self.alerts_history if now - a["timestamp"] <= 900]
        
        # Calculate confluence score
        score, details, status, ob_imbalance, depth_age = self.scorer.evaluate_sweep(
            symbol, sweep_type, level, self.alerts_history, last_depth_ts
        )
        
        # Format descriptive log message
        details_str = ", ".join(details["details"]) if details["details"] else "No confluence factors detected"
        notes = f"{sweep_type.upper()} Sweep @ {level:.2f} | Score: {score}/10 | [{details_str}]"
        
        # Log to file, CSV, and show on terminal dashboard
        metrics_1m = self.metrics.get_metrics_for_window(symbol, "1m")
        self._trigger_event(
            symbol=symbol,
            event_type=status,
            window="confluence",
            metrics=metrics_1m,
            notes=notes,
            sweep_id=sweep_id,
            source_sweep_time=source_time,
            confluence_score=score,
            matched_conditions=details_str,
            order_book_imbalance_at_score=ob_imbalance,
            depth_snapshot_age_ms=depth_age
        )

    def _build_current_symbol_data(self) -> dict:
        """Helper to build a unified dictionary snapshot of current metrics for all symbols."""
        symbol_data = {}
        for symbol in SYMBOLS:
            sym_lower = symbol.lower()
            state = self.metrics.get_state(symbol)
            m1m = self.metrics.get_metrics_for_window(symbol, "1m")
            m5m = self.metrics.get_metrics_for_window(symbol, "5m")
            m15m = self.metrics.get_metrics_for_window(symbol, "15m")
            
            # Read-only components read directly from canonical stored decisions
            # DO NOT recompute scanner decisions from read-only consumers.
            decision = self.current_decisions.get(sym_lower)
            if not decision:
                decision = {
                    "action": "WARMING_UP",
                    "reason": "Awaiting first canonical scanner evaluation",
                    "gates": {},
                    "timestamp": None,
                    "scanner_cycle_id": 0,
                    "action_version": 0,
                    "active_sweep": {"active": False, "direction": "", "score": 0, "age_seconds": 999.0}
                }

            now = time.time()
            recent_events = [
                a["type"] for a in self.alerts_history
                if a["symbol"].lower() == sym_lower and (now - a["timestamp"]) <= 300.0
            ]
            
            symbol_data[symbol] = {
                "price": m1m.get("latest_price", 0.0),
                "session_cvd_usdt": state.session_cvd_usdt,
                "delta_1m_usdt": m1m.get("delta_usdt", 0.0),
                "delta_5m_usdt": m5m.get("delta_usdt", 0.0),
                "delta_15m_usdt": m15m.get("delta_usdt", 0.0),
                "buy_ratio_1m": m1m.get("buy_ratio", 0.5),
                "sell_ratio_1m": m1m.get("sell_ratio", 0.5),
                "buy_ratio_5m": m5m.get("buy_ratio", 0.5),
                "imbalance": state.bid_ask_imbalance,
                "spread": state.spread,
                
                # Canonical scanner decisions
                "next_action": decision["action"],
                "decision_reason": decision["reason"],
                "check_gates": decision["gates"],
                "decision_timestamp": decision["timestamp"],
                "scanner_cycle_id": decision["scanner_cycle_id"],
                "action_version": decision["action_version"],
                "active_sweep": decision["active_sweep"],
                "recent_events": recent_events,
                
                "last_large_trade_time": state.last_large_trade_time,
                "last_event_time": state.last_event_time,
                "latest_event": state.latest_event,
                "last_depth_timestamp": state.last_depth_timestamp
            }
        return symbol_data

    def _commit_scanner_decision(self, symbol: str, decision: dict, now: float):
        """
        Commits a pure scanner decision to the engine state.
        This is the ONLY mutator of decisions, cooldowns, and action versions.
        """
        sym_lower = symbol.lower()
        prev_decision = self.current_decisions.get(sym_lower)
        previous_action = prev_decision["action"] if prev_decision else "WARMING_UP"
        action_version = prev_decision["action_version"] if prev_decision else 0
        
        # Increment action version only if the action changes
        if decision["action"] != previous_action:
            action_version += 1
            
        # Check transition cooldown condition:
        # A cooldown starts ONLY when a new confirmed signal transition is committed
        is_new_confirmation = (
            decision["confirmation_candidate"]
            and decision["action"] in ("CONFIRMED_LONG", "CONFIRMED_SHORT")
            and previous_action != decision["action"]
        )
        
        if is_new_confirmation:
            self.cooldown_ends[sym_lower] = now + 180.0 # COOLDOWN_DURATION_SECONDS
            
        # Update canonical current_decisions dictionary
        # DO NOT recompute scanner decisions from read-only consumers.
        self.current_decisions[sym_lower] = {
            "action": decision["action"],
            "reason": decision["reason"],
            "gates": decision["gates"],
            "timestamp": now,
            "scanner_cycle_id": self.scanner_cycle_id,
            "action_version": action_version,
            "active_sweep": decision["active_sweep"]
        }
        
        # Mirror gates in scorer for backward compatibility
        self.scorer.symbol_gates[sym_lower] = decision["gates"]

    async def _handle_http_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Async web server connection handler to serve Web Dashboard and JSON metrics API."""
        try:
            data = await reader.read(16384)
            request = data.decode("utf-8", errors="ignore")
            if not request:
                return
                
            lines = request.split("\r\n")
            if not lines:
                return
            request_line = lines[0]
            parts = request_line.split(" ")
            if len(parts) < 2:
                return
            method, path = parts[0], parts[1]

            if method == "POST":
                if path == "/api/ai/interpret":
                    now_time = time.time()
                    if now_time - self.last_ai_manual_time < 30.0:
                        retry_in = int(30.0 - (now_time - self.last_ai_manual_time))
                        payload = {
                            "ok": False,
                            "provider": self.ai_runtime_config.provider,
                            "model": self.ai_runtime_config.model,
                            "interpretation": {},
                            "error": f"Cooldown active. Please wait {retry_in}s before requesting a new AI analysis."
                        }
                        response = self._json_response(payload, "429 Too Many Requests")
                    else:
                        self.last_ai_manual_time = now_time
                        # Perform cost-controlled paid API call (uses cache if TTL and hash match)
                        symbol_data = self._build_current_symbol_data()
                        payload = await self.ai_interpreter.interpret(
                            symbol_data,
                            force=True,
                            binance_context=self.binance_context.get_context()
                        )
                        response = self._json_response(payload)
                elif path == "/api/ai/settings":
                    try:
                        settings = self._parse_json_body(request)
                        self.ai_runtime_config.update(
                            provider=settings.get("provider"),
                            model=settings.get("model"),
                            api_key=settings.get("api_key") if "api_key" in settings else None,
                            enabled=settings.get("enabled") if "enabled" in settings else None
                        )
                        response = self._json_response(self._safe_ai_settings_payload())
                    except Exception as e:
                        payload = {"ok": False, "error": ai_providers.redact_sensitive(str(e))}
                        response = self._json_response(payload, "400 Bad Request")
                elif path == "/api/ai/test":
                    response = self._json_response(self._test_ai_provider_config())
                elif path == "/api/paper/order":
                    try:
                        body = self._parse_json_body(request)
                        symbol = body.get("symbol", "").lower()
                        side = body.get("side", "").upper()
                        qty = float(body.get("quantity", 0.0))
                        
                        price = self.metrics.get_metrics_for_window(symbol, "1m").get("latest_price", 0.0)
                        if price <= 0:
                            state = self.metrics.get_state(symbol)
                            price = (state.best_bid + state.best_ask) / 2.0 if (state.best_bid + state.best_ask) > 0 else 0.0
                            
                        if price <= 0:
                            response = self._json_response({"ok": False, "error": f"No mark price available for {symbol.upper()}"}, "400 Bad Request")
                        else:
                            success = self.paper_trader.execute_order(symbol, side, qty, price)
                            if success:
                                current_prices = {s: self.metrics.get_metrics_for_window(s, "1m").get("latest_price", 0.0) for s in SYMBOLS}
                                response = self._json_response({"ok": True, "portfolio": self.paper_trader.get_portfolio_state(current_prices)})
                            else:
                                response = self._json_response({"ok": False, "error": "Order execution failed (insufficient margin/funds or invalid parameters)"}, "400 Bad Request")
                    except Exception as e:
                        response = self._json_response({"ok": False, "error": str(e)}, "400 Bad Request")
                elif path == "/api/paper/reset":
                    self.paper_trader.reset()
                    current_prices = {s: self.metrics.get_metrics_for_window(s, "1m").get("latest_price", 0.0) for s in SYMBOLS}
                    response = self._json_response({"ok": True, "portfolio": self.paper_trader.get_portfolio_state(current_prices)})
                elif path == "/api/paper/toggle_auto":
                    try:
                        body = self._parse_json_body(request)
                        self.paper_trader.auto_trade_enabled = bool(body.get("enabled", False))
                        current_prices = {s: self.metrics.get_metrics_for_window(s, "1m").get("latest_price", 0.0) for s in SYMBOLS}
                        response = self._json_response({"ok": True, "portfolio": self.paper_trader.get_portfolio_state(current_prices)})
                    except Exception as e:
                        response = self._json_response({"ok": False, "error": str(e)}, "400 Bad Request")
                else:
                    response = self._plain_response("404 Not Found", "404 Not Found")

            elif method == "GET":
                if path == "/api/ai/latest":
                    payload = self.ai_interpreter.get_latest(self._build_current_symbol_data())
                    response = self._json_response(payload)
                elif path == "/api/ai/settings":
                    response = self._json_response(self._safe_ai_settings_payload())
                elif path == "/api/ai/providers":
                    settings = self._safe_ai_settings_payload()
                    payload = {
                        "enabled": settings["enabled"],
                        "active_provider": settings["provider"],
                        "active_model": settings["model"],
                        "configured": settings["configured"],
                        "cache_ttl_seconds": config.AI_CACHE_TTL_SECONDS,
                        "interval_seconds": config.AI_INTERVAL_SECONDS
                    }
                    response = self._json_response(payload)
                elif path.startswith("/api/symbol"):
                    sym_name = "btcusdt"
                    if "?" in path:
                        parts = path.split("?")
                        if len(parts) >= 2:
                            for q in parts[1].split("&"):
                                if q.startswith("sym="):
                                    sym_name = q.split("=")[1].lower()
                    
                    state = self.metrics.get_state(sym_name)
                    hist_snapshots = []
                    for ts, bids, asks in list(state.stacking_pulling.history)[-300:]:
                        hist_snapshots.append({
                            "timestamp": ts,
                            "bids": bids,
                            "asks": asks
                        })
                        
                    recent_trades = []
                    for t in list(state.trades)[-200:]:
                        recent_trades.append({
                            "timestamp": t.exchange_time_ms / 1000.0,
                            "price": t.price,
                            "quantity": t.quantity,
                            "side": t.aggressor_side,
                            "notional": t.notional
                        })
                        
                    payload = {
                        "symbol": sym_name,
                        "price": state.best_bid if state.best_bid > 0 else 0.0,
                        "best_bid": state.best_bid,
                        "best_ask": state.best_ask,
                        "spread": state.spread,
                        "spread_bps": state.spread_bps,
                        "microprice": state.microprice,
                        "microprice_dev": state.microprice_dev,
                        "imbalance_0_5_bps": state.imbalance_0_5_bps,
                        "imbalance_5_15_bps": state.imbalance_5_15_bps,
                        "imbalance_15_30_bps": state.imbalance_15_30_bps,
                        "imbalance_total": state.imbalance_total,
                        "depth_weighted_imbalance": state.depth_weighted_imbalance,
                        "book_history": hist_snapshots,
                        "trades": recent_trades,
                        "active_walls": state.wall_tracker.get_active_walls(
                            (state.best_bid + state.best_ask)/2.0 if (state.best_bid + state.best_ask) > 0 else 1.0, 
                            state.bid_depth_top5_usdt if state.bid_depth_top5_usdt > 0 else 1.0
                        )
                    }
                    response = self._json_response(payload)
                elif path == "/api/binance/context":
                    response = self._json_response(self.binance_context.get_context())
                elif path == "/api/binance/status":
                    response = self._json_response(self.binance_context.get_status())
                elif path == "/api":
                    trade_status = self.stream.get_status()
                    depth_status = self.depth_stream.get_status()
                    
                    if self.dashboard.is_multi:
                        symbols_payload = {}
                        for sym in SYMBOLS:
                            sym_lower = sym.lower()
                            state = self.metrics.get_state(sym)
                            m1m = self.metrics.get_metrics_for_window(sym, "1m")
                            m5m = self.metrics.get_metrics_for_window(sym, "5m")
                            m15m = self.metrics.get_metrics_for_window(sym, "15m")
                            
                            # Read canonical decision
                            decision = self.current_decisions.get(sym_lower)
                            bias_action = decision["action"]
                            
                            self.metrics.check_duplication(sym, m5m, m15m)
                            mid_price = (state.best_bid + state.best_ask) / 2.0
                            
                            symbols_payload[sym] = {
                                "price": m1m.get("latest_price", 0.0),
                                "session_cvd_usdt": state.session_cvd_usdt,
                                "running_cvd_usdt": state.running_cvd_usdt,
                                "running_cvd_usdt": state.running_cvd_usdt,
                                "metrics_1m": m1m,
                                "metrics_5m": m5m,
                                "metrics_15m": m15m,
                                "imbalance": state.bid_ask_imbalance,
                                "imbalance_0_5_bps": state.imbalance_0_5_bps,
                                "imbalance_5_15_bps": state.imbalance_5_15_bps,
                                "imbalance_15_30_bps": state.imbalance_15_30_bps,
                                "imbalance_total": state.imbalance_total,
                                "depth_weighted_imbalance": state.depth_weighted_imbalance,
                                "spread": state.spread,
                                "spread_bps": state.spread_bps,
                                "microprice_dev": state.microprice_dev,
                                "next_action": bias_action,
                                "last_large_trade_time": state.last_large_trade_time,
                                "last_event_time": state.last_event_time,
                                "latest_event": state.latest_event,
                                "last_depth_timestamp": state.last_depth_timestamp,
                                "binance_context": self.binance_context.get_context().get("symbols", {}).get(sym.upper(), {}),
                                "duplication_suspected": state.duplication_suspected,
                                "book_coverage": "STANDARD L2",
                                "rpi_included": "NO",
                                "book_state": state.local_book.state,
                                "stream_health_status": state.health_tracker.get_status(state.local_book.is_valid),
                                "stream_health_metrics": state.health_tracker.get_metrics(),
                                "check_gates": decision["gates"],
                                "active_walls": state.wall_tracker.get_active_walls(mid_price, state.bid_depth_top5_usdt)
                            }
                        
                        current_prices = {s: self.metrics.get_metrics_for_window(s, "1m").get("latest_price", 0.0) for s in SYMBOLS}
                        payload = {
                            "is_multi": True,
                            "trade_ws_status": trade_status,
                            "depth_ws_status": depth_status,
                            "symbols": symbols_payload,
                            "recent_events": list(self.dashboard.recent_events),
                            "recent_large_trades": list(self.dashboard.recent_large_trades),
                            "binance_context_status": self.binance_context.get_status(),
                            "paper_portfolio": self.paper_trader.get_portfolio_state(current_prices)
                        }
                    else:
                        sym = SYMBOLS[0]
                        sym_lower = sym.lower()
                        m1m = self.metrics.get_metrics_for_window(sym, "1m")
                        m5m = self.metrics.get_metrics_for_window(sym, "5m")
                        m15m = self.metrics.get_metrics_for_window(sym, "15m")
                        state = self.metrics.get_state(sym)
                        
                        decision = self.current_decisions.get(sym_lower)
                        bias_action = decision["action"]
                        
                        payload = {
                            "is_multi": False,
                            "trade_ws_status": trade_status,
                            "depth_ws_status": depth_status,
                            "price": m1m.get("latest_price", 0.0),
                            "running_cvd": state.running_cvd_usdt,
                            "session_cvd": state.session_cvd_usdt,
                            "metrics_1m": m1m,
                            "metrics_5m": m5m,
                            "metrics_15m": m15m,
                            "best_bid": state.best_bid,
                            "best_ask": state.best_ask,
                            "spread": state.spread,
                            "bid_depth": state.bid_depth_top5_usdt,
                            "ask_depth": state.ask_depth_top5_usdt,
                            "imbalance": state.bid_ask_imbalance,
                            "next_action": bias_action,
                            "microprice": state.microprice,
                            "recent_events": list(self.dashboard.recent_events),
                            "recent_large_trades": list(self.dashboard.recent_large_trades),
                            "binance_context": self.binance_context.get_context().get("symbols", {}).get(sym.upper(), {}),
                            "binance_context_status": self.binance_context.get_status()
                        }
                        
                    response = self._json_response(payload)
                elif path == "/":
                    body = self.dashboard.render_html()
                    response = self._html_response(body)
                else:
                    response = self._plain_response("404 Not Found", "404 Not Found")
            else:
                response = self._plain_response("Method Not Allowed", "405 Method Not Allowed")
            writer.write(response.encode("utf-8"))
            await writer.drain()
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def run(self):
        logger.info(f"Starting Order Flow Engine V2.5 (Radar Universe: {len(SYMBOLS)} symbols)...")

        
        # Start HTTP server on the configured stable dashboard port.
        port = config.WEB_DASHBOARD_PORT
        http_server = await asyncio.start_server(self._handle_http_client, '127.0.0.1', port)
        
        # Log web address
        logger.info(f"Web Dashboard started at http://localhost:{port}")
        print(f"\n=======================================================")
        print(f"WEB DASHBOARD ACTIVE: http://localhost:{port}")
        print(f"=======================================================\n")

        if config.RECORDING_ENABLED:
            for rec in self.recorders.values():
                rec.start()

        await self.stream.start()
        await self.depth_stream.start()
        await self.sweeps_monitor.start()

        # Run loops
        processing_task = asyncio.create_task(self._process_trades_loop())
        dashboard_task = asyncio.create_task(self._dashboard_loop())
        ai_task = asyncio.create_task(self._ai_loop())
        binance_context_task = asyncio.create_task(self._binance_context_loop())

        try:
            await asyncio.gather(processing_task, dashboard_task, ai_task, binance_context_task)

        except asyncio.CancelledError:
            logger.info("Order Flow Engine stopped.")
        finally:
            if config.RECORDING_ENABLED:
                for rec in self.recorders.values():
                    await rec.stop()

            await self.stream.stop()
            await self.depth_stream.stop()
            await self.sweeps_monitor.stop()
            http_server.close()
            await http_server.wait_closed()

    async def _process_trades_loop(self):
        """Consumes trades from the queue, feeds them to metrics, and handles trade-level alerts."""
        while True:
            try:
                symbol, trade = await self.trade_queue.get()
                
                # Add to sliding window metrics specifically for this symbol
                alerts = self.metrics.add_trade(symbol, trade)
                
                # Feed price to outcomes tracker
                trade_price = float(trade.get("p", 0.0) or trade.get("price", 0.0) or 0.0)
                if trade_price > 0:
                    self.signal_tracker.update_price(symbol, trade_price)

                # Handle large trade alerts
                for alert in alerts:
                    if alert["type"] == "LARGE_TRADE":
                        self.dashboard.add_large_trade(
                            symbol=symbol,
                            side=alert["side"],
                            qty=alert["quantity"],
                            price=alert["price"],
                            notional_usdt=alert["notional_usdt"]
                        )
                        # Record alert history for scoring
                        self.alerts_history.append({
                            "symbol": symbol,
                            "type": "LARGE_TRADE",
                            "timestamp": time.time()
                        })
                        # Log event
                        self._trigger_event(
                            symbol=symbol,
                            event_type="LARGE_TRADE",
                            window="instant",
                            metrics=self.metrics.get_metrics_for_window(symbol, "1m"),
                            notes=f"Taker {alert['side']} size: ${alert['notional_usdt']:,.0f} @ {alert['price']:.2f}"
                        )

                self.trade_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trade processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _trigger_event(self, symbol: str, event_type: str, window: str, metrics: dict, notes: str, 
                       sweep_id: str = "", source_sweep_time: str = "",
                       confluence_score: Optional[int] = None, matched_conditions: str = "",
                       order_book_imbalance_at_score: Optional[float] = None,
                       depth_snapshot_age_ms: Optional[float] = None):
        """Logs event to CSV, updates the dashboard events, and respects cooldown per symbol:event."""
        now = time.time()
        
        # Track event cooldown key: symbol:event:window
        c_key = f"{symbol.lower()}:{event_type}:{window}"
        
        # Exclude LARGE_TRADE and confluences from standard cooldown checks
        if event_type not in ["LARGE_TRADE", "SWEEP_CONFLUENCE", "LOW_CONFLUENCE"]:
            last_time = self.last_triggered.get(c_key, 0.0)
            last_price = self.last_triggered_price.get(c_key, 0.0)
            current_price = metrics.get("latest_price", 0.0)
            
            # Bypass cooldown if price has changed by >= 0.05%
            price_moved_significantly = False
            if last_price > 0 and current_price > 0:
                price_moved_significantly = (abs(current_price - last_price) / last_price) >= 0.0005

            if now - last_time < EVENT_COOLDOWN_SECONDS and not price_moved_significantly:
                return  # Cooldown active, discard duplicate alert
                
            self.last_triggered[c_key] = now
            self.last_triggered_price[c_key] = current_price

            # Add to historical alerts list specifically for this symbol
            self.alerts_history.append({
                "symbol": symbol.lower(),
                "type": event_type,
                "timestamp": now
            })

        # Update symbol latest event timestamps to score activity ranking
        state = self.metrics.get_state(symbol)
        state.last_event_time = now
        state.latest_event = event_type

        # Add symbol back to metrics dict for storage (overwrite with USDT notionals for CSV schema compatibility)
        metrics_copy = metrics.copy()
        metrics_copy["symbol"] = symbol.lower()
        metrics_copy["buy_volume"] = metrics.get("buy_volume_usdt", 0.0)
        metrics_copy["sell_volume"] = metrics.get("sell_volume_usdt", 0.0)
        metrics_copy["delta"] = metrics.get("delta_usdt", 0.0)
        metrics_copy["cvd"] = state.session_cvd_usdt # Log notional CVD in cvd column

        # Persist and show on dashboard
        self.storage.log_event(
            event_type=event_type,
            window=window,
            metrics=metrics_copy,
            notes=notes,
            sweep_id=sweep_id,
            source_sweep_time=source_sweep_time,
            confluence_score=confluence_score,
            matched_conditions=matched_conditions,
            order_book_imbalance_at_score=order_book_imbalance_at_score,
            depth_snapshot_age_ms=depth_snapshot_age_ms
        )
        self.dashboard.add_event(symbol, event_type, window, metrics_copy.get("latest_price", 0.0), notes)

    def _run_auto_paper_trade(self, symbol: str, price: float, action: str):
        if not self.paper_trader.auto_trade_enabled:
            return
            
        symbol = symbol.lower()
        pos = self.paper_trader.positions.get(symbol)
        
        # Determine if we should exit current position
        if pos:
            should_close = False
            # Check 1% Take Profit target (Futures contracts PnL)
            if pos["side"] == "BUY":
                profit_pct = (price - pos["entry_price"]) / pos["entry_price"]
            else:
                profit_pct = (pos["entry_price"] - price) / pos["entry_price"]
                
            if profit_pct >= 0.01:
                should_close = True
                logger.info(f"[AUTO PAPER TRADE] 1% Profit Target hit for {symbol.upper()}! Closing position at {price} (Entry: {pos['entry_price']})")
            elif profit_pct <= -0.10:
                should_close = True
                logger.info(f"[AUTO PAPER TRADE] 10% Stop Loss hit for {symbol.upper()}! Closing position at {price} (Entry: {pos['entry_price']})")
            elif pos["side"] == "BUY" and action in ("CONFIRMED_SHORT", "SHORT (SWEEP)", "SHORT_BIAS", "WATCH_SHORT"):
                should_close = True
            elif pos["side"] == "SELL" and action in ("CONFIRMED_LONG", "LONG (SWEEP)", "LONG_BIAS", "WATCH_LONG"):
                should_close = True
                
            if should_close:
                close_side = "SELL" if pos["side"] == "BUY" else "BUY"
                self.paper_trader.execute_order(symbol, close_side, pos["qty"], price)
                logger.info(f"[AUTO PAPER TRADE] Closed position for {symbol.upper()} at {price} due to exit condition.")
                
        # Determine if we should open a new position
        pos = self.paper_trader.positions.get(symbol)
        if not pos:
            symbol_state = self.metrics.get_state(symbol)
            if action in ("CONFIRMED_LONG", "LONG (SWEEP)", "LONG_BIAS", "CONFIRMED_SHORT", "SHORT (SWEEP)", "SHORT_BIAS"):
                logger.info(f"[AUTO DEBUG] {symbol.upper()}: action={action}, book_valid={symbol_state.local_book.is_valid}, book_state={symbol_state.local_book.state}, spread_bps={symbol_state.spread_bps:.2f}")
            # 1. Order book synchronization check
            if not symbol_state.local_book.is_valid or symbol_state.local_book.state != "HEALTHY":
                return
            # 2. Tight bid-ask spread check (must be < 15.0 basis points)
            if symbol_state.spread_bps >= 15.0:
                return
            # 3. Data integrity validation
            if symbol_state.duplication_suspected:
                return
                
            # Highest Probability Entry Filters (Confluence Alignment):
            metrics_5m = self.metrics.get_metrics_for_window(symbol, "5m")
            buy_ratio_5m = metrics_5m.get("buy_ratio", 0.5)
            imbalance = symbol_state.bid_ask_imbalance
            cvd = symbol_state.session_cvd_usdt
            
            is_high_prob_long = (
                action in ("CONFIRMED_LONG", "LONG (SWEEP)", "LONG_BIAS")
                and buy_ratio_5m >= 0.58
                and imbalance >= 0.20
                and cvd > 0
            )
            
            is_high_prob_short = (
                action in ("CONFIRMED_SHORT", "SHORT (SWEEP)", "SHORT_BIAS")
                and buy_ratio_5m <= 0.42
                and imbalance <= -0.20
                and cvd < 0
            )
            
            current_prices = {s: self.metrics.get_metrics_for_window(s, "1m").get("latest_price", 0.0) for s in SYMBOLS}
            current_prices[symbol] = price
            
            state = self.paper_trader.get_portfolio_state(current_prices)
            equity = state["equity"]
            
            # Simple trade sizing: 10% of equity, with 5x leverage
            trade_size_usdt = equity * 0.10 * 5.0
            qty = trade_size_usdt / price
            
            if is_high_prob_long:
                self.paper_trader.execute_order(symbol, "BUY", qty, price)
                logger.info(f"[AUTO PAPER TRADE] Opened HIGH PROBABILITY LONG for {symbol.upper()} at {price} (Qty: {qty:.4f})")
            elif is_high_prob_short:
                self.paper_trader.execute_order(symbol, "SELL", qty, price)
                logger.info(f"[AUTO PAPER TRADE] Opened HIGH PROBABILITY SHORT for {symbol.upper()} at {price} (Qty: {qty:.4f})")

    async def _dashboard_loop(self):
        """Periodically refreshes the dashboard UI and runs slower event checks per symbol."""
        # Wait a moment for trades to start flowing
        await asyncio.sleep(2)
        
        with Live(self.dashboard.layout, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    self.scanner_cycle_id += 1
                    trade_ws_status = self.stream.get_status()
                    depth_ws_status = self.depth_stream.get_status()
                    
                    symbol_data = {}
                    
                    # 1. Evaluate metrics and signals per symbol
                    for symbol in SYMBOLS:
                        state = self.metrics.get_state(symbol)
                        metrics_1m = self.metrics.get_metrics_for_window(symbol, "1m")
                        metrics_5m = self.metrics.get_metrics_for_window(symbol, "5m")
                        metrics_15m = self.metrics.get_metrics_for_window(symbol, "15m")
                        
                        price = metrics_1m.get("latest_price", 0.0)
                        
                        if price > 0:
                            # Ticker updates on tracker
                            self.signal_tracker.update_price(symbol, price)

                            # 1. Check Trade Bursts
                            burst = self.metrics.detect_trade_burst(symbol)
                            if burst:
                                notes = f"Count: {burst['current_count']} trades/10s (Avg: {burst['average_count']})"
                                self._trigger_event(symbol, "TRADE_BURST", "10s", metrics_1m, notes)

                            # 2. Check Aggression alerts (ratio >= 65% in 1m)
                            if metrics_1m["buy_ratio"] >= 0.65:
                                notes = f"Buy ratio: {metrics_1m['buy_ratio']*100:.1f}%, Delta: ${metrics_1m['delta_usdt']:,.0f}"
                                self._trigger_event(symbol, "BUY_AGGRESSION", "1m", metrics_1m, notes)
                            elif metrics_1m["sell_ratio"] >= 0.65:
                                notes = f"Sell ratio: {metrics_1m['sell_ratio']*100:.1f}%, Delta: -${abs(metrics_1m['delta_usdt']):,.0f}"
                                self._trigger_event(symbol, "SELL_AGGRESSION", "1m", metrics_1m, notes)

                            # 3. Check Absorption (1m and 5m)
                            imbalance = state.bid_ask_imbalance
                            
                            abs_1m = self.detector.check_absorption(symbol, state.trades, 60)
                            if abs_1m[0]:
                                raw_type = abs_1m[0]
                                if raw_type == "BULLISH_ABSORPTION":
                                    event_type = "CONFIRMED_BULLISH_ABSORPTION" if imbalance >= 0.15 else "POSSIBLE_BULLISH_ABSORPTION"
                                else:
                                    event_type = "CONFIRMED_BEARISH_ABSORPTION" if imbalance <= -0.15 else "POSSIBLE_BEARISH_ABSORPTION"
                                self._trigger_event(symbol, event_type, "1m", metrics_1m, abs_1m[1])

                            abs_5m = self.detector.check_absorption(symbol, state.trades, 300)
                            if abs_5m[0]:
                                raw_type = abs_5m[0]
                                if raw_type == "BULLISH_ABSORPTION":
                                    event_type = "CONFIRMED_BULLISH_ABSORPTION" if imbalance >= 0.15 else "POSSIBLE_BULLISH_ABSORPTION"
                                else:
                                    event_type = "CONFIRMED_BEARISH_ABSORPTION" if imbalance <= -0.15 else "POSSIBLE_BEARISH_ABSORPTION"
                                self._trigger_event(symbol, event_type, "5m", metrics_5m, abs_5m[1])

                            # 4. Check CVD Divergence (15m)
                            div_15m = self.detector.check_cvd_divergence(symbol, state.trades, 900)
                            if div_15m[0]:
                                self._trigger_event(symbol, div_15m[0], "15m", metrics_15m, div_15m[1])

                        # Compile list of active events in the last 300 seconds for this symbol
                        now = time.time()
                        recent_events = [
                            a["type"] for a in self.alerts_history
                            if a["symbol"].lower() == symbol.lower() and (now - a["timestamp"]) <= 300.0
                        ]

                        # Pure state machine evaluation of Order Flow bias
                        decision = self.scorer.evaluate_bias(
                            symbol=symbol,
                            metrics_5m=metrics_5m,
                            imbalance=state.bid_ask_imbalance,
                            recent_events=recent_events,
                            now=now,
                            cooldown_end=self.cooldown_ends.get(symbol.lower(), 0.0)
                        )

                        # Commit canonical decision
                        self._commit_scanner_decision(symbol, decision, now)
                        bias_action = decision["action"]
                        suppression_reason = decision["reason"]

                        # Run auto paper trade checks
                        self._run_auto_paper_trade(symbol, price, bias_action)

                        # Detect signal state changes and register them with SignalTracker
                        prev_action = self.prev_actions.get(symbol, "WAITING")
                        if bias_action != prev_action:
                            self.signal_tracker.register_signal_change(
                                symbol=symbol,
                                old_action=prev_action,
                                new_action=bias_action,
                                price=price,
                                metrics_5m=metrics_5m,
                                imbalance=state.bid_ask_imbalance,
                                latest_event=state.latest_event,
                                cvd=state.session_cvd_usdt,
                                suppression_reason=suppression_reason
                            )
                            self.prev_actions[symbol] = bias_action

                        # Compile render details per symbol
                        symbol_data[symbol] = {
                            "price": price,
                            "session_cvd_usdt": state.session_cvd_usdt,
                            "running_cvd_usdt": state.running_cvd_usdt,
                            "delta_1m_usdt": metrics_1m.get("delta_usdt", 0.0),
                            "delta_5m_usdt": metrics_5m.get("delta_usdt", 0.0),
                            "delta_15m_usdt": metrics_15m.get("delta_usdt", 0.0),
                            "buy_ratio_1m": metrics_1m.get("buy_ratio", 0.5),
                            "sell_ratio_1m": metrics_1m.get("sell_ratio", 0.5),
                            "buy_ratio_5m": metrics_5m.get("buy_ratio", 0.5),
                            "imbalance": state.bid_ask_imbalance,
                            "spread": state.spread,
                            "next_action": bias_action,
                            "last_large_trade_time": state.last_large_trade_time,
                            "last_event_time": state.last_event_time,
                            "latest_event": state.latest_event,
                            "last_depth_timestamp": state.last_depth_timestamp,
                            "metrics_1m": metrics_1m,
                            "metrics_5m": metrics_5m,
                            "metrics_15m": metrics_15m,
                            "duplication_suspected": state.duplication_suspected,
                            "book_state": state.local_book.state
                        }

                    # Periodic finalization checks for outcome tracking
                    self.signal_tracker.finalize_expired_signals()

                    # Render UI based on configuration
                    if self.dashboard.is_multi:
                        live.update(self.dashboard.render_multi(symbol_data, trade_ws_status, depth_ws_status))
                    else:
                        sym = SYMBOLS[0]
                        sym_data = symbol_data[sym]
                        state = self.metrics.get_state(sym)
                        m1m = self.metrics.get_metrics_for_window(sym, "1m")
                        m5m = self.metrics.get_metrics_for_window(sym, "5m")
                        m15m = self.metrics.get_metrics_for_window(sym, "15m")
                        
                        live.update(self.dashboard.render(
                            price=sym_data["price"],
                            running_cvd=state.running_cvd_usdt,
                            session_cvd=sym_data["session_cvd_usdt"],
                            metrics_1m=m1m,
                            metrics_5m=m5m,
                            metrics_15m=m15m,
                            best_bid=state.best_bid,
                            best_ask=state.best_ask,
                            spread=sym_data["spread"],
                            bid_depth=state.bid_depth_top5_usdt,
                            ask_depth=state.ask_depth_top5_usdt,
                            imbalance=sym_data["imbalance"],
                            microprice=state.microprice,
                            burst_status=None,
                            absorption_status_1m=(None, ""),
                            absorption_status_5m=(None, ""),
                            divergence_status=(None, ""),
                            trade_ws_status=trade_ws_status,
                            depth_ws_status=depth_ws_status
                        ))

                    await asyncio.sleep(DASHBOARD_REFRESH_SECONDS)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in dashboard refresh loop: {e}", exc_info=True)
                    await asyncio.sleep(1)

    async def _ai_loop(self):
        """Periodically calls AI Interpreter to generate automated updates if enabled."""
        if self.ai_runtime_config.enabled:
            logger.info(f"AI Auto-Interpretation loop active. Run cadence: {config.AI_INTERVAL_SECONDS} seconds.")
        else:
            logger.info("AI Auto-Interpretation loop disabled (AI_ENABLED=false).")
        
        while True:
            try:
                if not self.ai_runtime_config.enabled:
                    await asyncio.sleep(config.AI_INTERVAL_SECONDS)
                    continue

                symbol_data = self._build_current_symbol_data()
                # Run the interpret operation (handles cost-controlled input hashing and TTL caching internally)
                await self.ai_interpreter.interpret(
                    symbol_data,
                    force=False,
                    binance_context=self.binance_context.get_context()
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AI interpretation loop: {e}", exc_info=True)
            
            await asyncio.sleep(config.AI_INTERVAL_SECONDS)

    async def _binance_context_loop(self):
        """Refreshes read-only Binance public context on a slow cadence."""
        logger.info("V2.6A Binance Context loop active (read-only public market data).")
        await asyncio.sleep(3)
        while True:
            try:
                symbol_data = self._build_current_symbol_data()
                await self.binance_context.refresh(symbol_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Binance context loop: {e}", exc_info=True)
            await asyncio.sleep(5)


def main():
    engine = OrderFlowEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\nShutdown requested by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
