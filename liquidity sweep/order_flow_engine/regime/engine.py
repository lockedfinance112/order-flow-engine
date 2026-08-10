import asyncio
import logging
import time
import math
from typing import List, Dict, Any, Callable, Optional, Tuple
from collections import deque

from regime.models import MarketBar
from regime.bar_store import BarStore
from regime.features import compute_features
from regime.classifier import classify_regime
from regime.recorder import RegimeRecorder
from regime.permissions import permissions_for

logger = logging.getLogger("OrderFlow.RegimeEngine")

class RegimeEngine:
    """
    TICS Market Regime Intelligence Engine. Decoupled and runs in Shadow Mode.
    Evaluates market regime using event-time watermark candle finalization,
    async gap recovery, and pure feature calculations.
    """
    def __init__(
        self,
        symbols: List[str],
        market_data_safety_provider: Callable[[str], Dict[str, Any]],
        liquidity_provider: Callable[[str], Dict[str, float]],
        config: Optional[Dict[str, Any]] = None
    ):
        self.symbols = [s.lower() for s in symbols]
        self.safety_provider = market_data_safety_provider
        self.liquidity_provider = liquidity_provider
        self.config = config or {}
        
        # Load Configuration
        self.enabled = self.config.get("REGIME_ENGINE_ENABLED", True)
        self.model_version = self.config.get("REGIME_MODEL_VERSION", "regime-v1")
        self.feature_version = self.config.get("REGIME_FEATURE_VERSION", "regime-features-v1")
        self.max_bars = self.config.get("REGIME_MAX_BARS_PER_TIMEFRAME", 2000)
        self.dedup_capacity = self.config.get("REGIME_TRADE_DEDUP_CAPACITY", 5000)
        self.max_late_ms = self.config.get("REGIME_MAX_LATE_TRADE_MS", 2000)
        
        # Symbol States
        self.stores: Dict[str, BarStore] = {s: BarStore(s, self.max_bars) for s in self.symbols}
        self.backfill_states: Dict[str, str] = {s: "NOT_STARTED" for s in self.symbols}
        self.current_regimes: Dict[str, Dict[str, Any]] = {}
        
        # Watermark and Bar Builders per symbol
        self.current_bars: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in self.symbols}
        self.pending_previous_bars: Dict[str, Optional[Dict[str, Any]]] = {s: None for s in self.symbols}
        self.max_seen_trade_times_ms: Dict[str, int] = {s: 0 for s in self.symbols}
        
        # Deduplication States: symbol -> (deque of IDs, set of IDs)
        # Bounded O(1) set/deque management
        self.dedup_ids: Dict[str, Tuple[deque, set]] = {
            s: (deque(), set()) for s in self.symbols
        }

        # Liquidity Rolling Baselines: symbol -> { "spread_bps": List, "depth_usdt": List }
        self.liquidity_baselines: Dict[str, Dict[str, List[float]]] = {
            s: {"spread": [], "depth": []} for s in self.symbols
        }
        
        # Hysteresis memory: symbol -> Dict[str, Any]
        self.hysteresis_state: Dict[str, Dict[str, Any]] = {
            s: {
                "candidate": None,
                "count": 0,
                "last_evaluation_close_ms": 0,
                "bars_since_regime_start": 0,
                "regime_since_ms": 0,
                "current_regime": "UNKNOWN",
                "previous_regime": "UNKNOWN"
            } for s in self.symbols
        }
        self.last_good_evaluation_ms: Dict[str, int] = {s: 0 for s in self.symbols}

        # Bounded Queue
        self.queue = asyncio.Queue(maxsize=1000)
        self.recorder = RegimeRecorder()
        
        # Task Trackers
        self.worker_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.backfill_tasks: List[asyncio.Task] = []

    def start(self):
        if not self.enabled:
            logger.info("RegimeEngine is disabled via configuration.")
            return
            
        self.worker_task = asyncio.create_task(self._worker_loop())
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        for symbol in self.symbols:
            t = asyncio.create_task(self._backfill_symbol(symbol))
            self.backfill_tasks.append(t)

    async def stop(self):
        tasks = []
        if self.worker_task:
            self.worker_task.cancel()
            tasks.append(self.worker_task)
        if self.recovery_task:
            self.recovery_task.cancel()
            tasks.append(self.recovery_task)
        for t in self.backfill_tasks:
            t.cancel()
            tasks.append(t)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.backfill_tasks.clear()
        self.worker_task = None
        self.recovery_task = None

    def get_regime_state(self, symbol: str) -> Dict[str, Any]:
        symbol_lower = symbol.lower()
        if not self.enabled:
            return {
                "symbol": symbol.upper(),
                "quality": "DISABLED",
                "primary_regime": "UNKNOWN",
                "confidence": 0.0,
                "model_version": self.model_version
            }
            
        if symbol_lower not in self.current_regimes:
            return {
                "symbol": symbol.upper(),
                "quality": "WARMING_UP",
                "primary_regime": "UNKNOWN",
                "confidence": 0.0,
                "structure": "UNKNOWN",
                "direction": "UNKNOWN",
                "volatility": "UNKNOWN",
                "liquidity": "UNKNOWN",
                "scores": {},
                "reasons": ["Warmup in progress"],
                "model_version": self.model_version
            }
        return self.current_regimes[symbol_lower]

    def on_trade(self, symbol: str, trade: dict):
        if not self.enabled:
            return
            
        symbol_lower = symbol.lower()
        if symbol_lower not in self.symbols:
            return

        trade_id = trade.get("aggregate_trade_id")
        trade_time_ms = trade.get("trade_time_ms")
        if trade_id is None or trade_time_ms is None:
            return

        store = self.stores[symbol_lower]

        # 1. Deduplication Invariant: Set == Deque
        dq, s_set = self.dedup_ids[symbol_lower]
        if trade_id in s_set:
            store.duplicate_trade_count += 1
            return
            
        if len(dq) >= self.dedup_capacity:
            oldest = dq.popleft()
            s_set.discard(oldest)
        dq.append(trade_id)
        s_set.add(trade_id)

        # 2. Watermark update
        max_seen = self.max_seen_trade_times_ms[symbol_lower]
        self.max_seen_trade_times_ms[symbol_lower] = max(max_seen, trade_time_ms)
        max_seen = self.max_seen_trade_times_ms[symbol_lower]

        bucket_open_ms = (trade_time_ms // 60000) * 60000
        bucket_close_ms = bucket_open_ms + 60000 - 1

        price = trade["price"]
        qty = trade["quantity"]
        quote_qty = price * qty

        # Access Symbol Builders
        current = self.current_bars[symbol_lower]
        pending = self.pending_previous_bars[symbol_lower]

        # Initialize current bar if empty
        if not current:
            self.current_bars[symbol_lower] = {
                "open_time_ms": bucket_open_ms,
                "close_time_ms": bucket_close_ms,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "base_volume": qty,
                "quote_volume": quote_qty,
                "agg_trade_count": 1,
                "first_trade_time_ms": trade_time_ms,
                "last_trade_time_ms": trade_time_ms
            }
            return

        if bucket_open_ms == current["open_time_ms"]:
            # Same minute aggregation
            if trade_time_ms < current["last_trade_time_ms"]:
                store.out_of_order_trade_count += 1
            if trade_time_ms < current["first_trade_time_ms"]:
                current["open"] = price
                current["first_trade_time_ms"] = trade_time_ms
            if trade_time_ms > current["last_trade_time_ms"]:
                current["close"] = price
                current["last_trade_time_ms"] = trade_time_ms
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["base_volume"] += qty
            current["quote_volume"] += quote_qty
            current["agg_trade_count"] += 1

        elif bucket_open_ms > current["open_time_ms"]:
            # Shift current to pending
            if pending:
                self._enqueue_bar(symbol_lower, pending)
            self.pending_previous_bars[symbol_lower] = current
            self.current_bars[symbol_lower] = {
                "open_time_ms": bucket_open_ms,
                "close_time_ms": bucket_close_ms,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "base_volume": qty,
                "quote_volume": quote_qty,
                "agg_trade_count": 1,
                "first_trade_time_ms": trade_time_ms,
                "last_trade_time_ms": trade_time_ms
            }

        else:
            # Late trade check: belongs to pending or older?
            if pending and bucket_open_ms == pending["open_time_ms"]:
                # Check lateness watermark limits
                if max_seen <= pending["open_time_ms"] + 60000 + self.max_late_ms:
                    store.out_of_order_trade_count += 1
                    if trade_time_ms < pending["first_trade_time_ms"]:
                        pending["open"] = price
                        pending["first_trade_time_ms"] = trade_time_ms
                    if trade_time_ms > pending["last_trade_time_ms"]:
                        pending["close"] = price
                        pending["last_trade_time_ms"] = trade_time_ms
                    pending["high"] = max(pending["high"], price)
                    pending["low"] = min(pending["low"], price)
                    pending["base_volume"] += qty
                    pending["quote_volume"] += quote_qty
                    pending["agg_trade_count"] += 1
                else:
                    store.late_trade_count += 1
            else:
                store.late_trade_count += 1

        # Check watermark to finalize pending bar
        pending = self.pending_previous_bars[symbol_lower]
        if pending and max_seen > pending["open_time_ms"] + 60000 + self.max_late_ms:
            self._enqueue_bar(symbol_lower, pending)
            self.pending_previous_bars[symbol_lower] = None

    def _enqueue_bar(self, symbol: str, bar_dict: dict):
        store = self.stores[symbol]
        finalized_bar = MarketBar(
            symbol=symbol,
            timeframe="1m",
            open_time_ms=bar_dict["open_time_ms"],
            close_time_ms=bar_dict["close_time_ms"],
            open=bar_dict["open"],
            high=bar_dict["high"],
            low=bar_dict["low"],
            close=bar_dict["close"],
            base_volume=bar_dict["base_volume"],
            quote_volume=bar_dict["quote_volume"],
            closed=True,
            agg_trade_count=bar_dict["agg_trade_count"]
        )
        try:
            self.queue.put_nowait((symbol, finalized_bar))
        except asyncio.QueueFull:
            store.queue_overflow_count += 1
            # Exact dropped candle details preserved for recovery request
            store.recovery_requests.append({
                "symbol": symbol,
                "timeframe": "1m",
                "missing_open_time_ms": bar_dict["open_time_ms"],
                "reason": "QUEUE_OVERFLOW"
            })
            store.unresolved_gaps.add(("1m", bar_dict["open_time_ms"]))
            logger.error(f"[{symbol.upper()}] Queue full! Registered recovery request for {bar_dict['open_time_ms']}.")

    async def _backfill_symbol(self, symbol: str):
        self.backfill_states[symbol] = "LOADING"
        from regime.history_loader import fetch_klines_async
        timeframes = ["1m", "5m", "15m", "1h"]
        
        try:
            for tf in timeframes:
                bars = await fetch_klines_async(symbol, tf, limit=500)
                store = self.stores[symbol]
                for bar in bars:
                    store.append_bar(tf, bar)
                    
            self.backfill_states[symbol] = "READY"
            self._process_symbol_regime(symbol, int(time.time() * 1000))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.backfill_states[symbol] = "FAILED"
            logger.error(f"[{symbol.upper()}] Historical backfill failed: {e}")

    async def _worker_loop(self):
        while True:
            try:
                symbol, bar = await self.queue.get()
                store = self.stores[symbol]
                
                if store.append_bar("1m", bar):
                    self._aggregate_higher_tfs(symbol, bar)
                    self._process_symbol_regime(symbol, bar.close_time_ms)
                    
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regime worker: {e}")
                await asyncio.sleep(0.1)

    async def _recovery_loop(self):
        """Asynchronous background loop resolving gaps from Binance Futures API."""
        from regime.history_loader import fetch_klines_async
        interval_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000
        }
        while True:
            try:
                await asyncio.sleep(5.0)
                for symbol in self.symbols:
                    store = self.stores[symbol]
                    if not store.recovery_requests:
                        continue
                        
                    req = store.recovery_requests.pop(0)
                    tf = req["timeframe"]
                    open_ms = req["missing_open_time_ms"]
                    
                    try:
                        bars = await fetch_klines_async(
                            symbol=symbol,
                            timeframe=tf,
                            start_time_ms=open_ms,
                            end_time_ms=open_ms + interval_ms[tf] - 1
                        )
                        for bar in bars:
                            store.append_bar(tf, bar)
                            
                        # If a 1m bar was recovered, trigger re-evaluation of higher TFs
                        if tf == "1m" and bars:
                            self._aggregate_higher_tfs(symbol, bars[0])
                    except Exception as err:
                        logger.warning(f"[{symbol.upper()}] Gap recovery failed for {open_ms}: {err}")
                        # Put request back
                        store.recovery_requests.append(req)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                await asyncio.sleep(5.0)

    def _aggregate_higher_tfs(self, symbol: str, new_1m_bar: MarketBar):
        store = self.stores[symbol]
        m1_bars = store.get_bars("1m")
        
        def aggregate_components(open_time_ms: int, count: int, tf: str) -> Optional[MarketBar]:
            components = [b for b in m1_bars if open_time_ms <= b.open_time_ms < open_time_ms + (count * 60000)]
            if len(components) < count:
                # Add component recovery request
                expected_times = set(open_time_ms + i * 60000 for i in range(count))
                existing_times = set(b.open_time_ms for b in components)
                missing_times = expected_times - existing_times
                
                for t in missing_times:
                    if not any(req["missing_open_time_ms"] == t and req["timeframe"] == "1m" for req in store.recovery_requests):
                        store.recovery_requests.append({
                            "symbol": symbol,
                            "timeframe": "1m",
                            "missing_open_time_ms": t,
                            "reason": "HIGHER_TF_COMPONENT_MISSING"
                        })
                        store.unresolved_gaps.add(("1m", t))
                return None
                
            open_val = components[0].open
            high_val = max(b.high for b in components)
            low_val = min(b.low for b in components)
            close_val = components[-1].close
            base_vol = sum(b.base_volume for b in components)
            quote_vol = sum(b.quote_volume for b in components)
            
            return MarketBar(
                symbol=symbol,
                timeframe=tf,
                open_time_ms=open_time_ms,
                close_time_ms=open_time_ms + (count * 60000) - 1,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close_val,
                base_volume=base_vol,
                quote_volume=quote_vol,
                closed=True
            )

        close_time = new_1m_bar.open_time_ms + 60000
        if close_time % 300000 == 0:
            m5_open = close_time - 300000
            m5_bar = aggregate_components(m5_open, 5, "5m")
            if m5_bar:
                store.append_bar("5m", m5_bar)

        if close_time % 900000 == 0:
            m15_open = close_time - 900000
            m15_bar = aggregate_components(m15_open, 15, "15m")
            if m15_bar:
                store.append_bar("15m", m15_bar)

        if close_time % 3600000 == 0:
            m1h_open = close_time - 3600000
            m1h_bar = aggregate_components(m1h_open, 60, "1h")
            if m1h_bar:
                store.append_bar("1h", m1h_bar)

    def _process_symbol_regime(self, symbol: str, event_time_ms: int):
        store = self.stores[symbol]
        safety = self.safety_provider(symbol)
        
        # 1. Guardian integration
        if not safety["safe"]:
            last_good = self.last_good_evaluation_ms.get(symbol, 0)
            self.current_regimes[symbol] = {
                "symbol": symbol.upper(),
                "quality": "STALE",
                "tradable": False,
                "primary_regime": self.hysteresis_state[symbol]["current_regime"],
                "last_good_age_seconds": (event_time_ms - last_good) / 1000.0 if last_good > 0 else 999.0,
                "reason": safety["reason"] or "Data Guardian unsafe",
                "model_version": self.model_version
            }
            return

        # 2. Quality/Warmup verification
        bars_1m = store.get_bars("1m")
        bars_5m = store.get_bars("5m")
        bars_15m = store.get_bars("15m")
        bars_1h = store.get_bars("1h")
        
        if len(bars_1m) < 50 or len(bars_5m) < 21 or len(bars_15m) < 21 or len(bars_1h) < 21:
            self.current_regimes[symbol] = {
                "symbol": symbol.upper(),
                "quality": "WARMING_UP",
                "tradable": False,
                "primary_regime": "UNKNOWN",
                "confidence": 0.0,
                "reasons": ["Insufficient historical closed bars for feature calculation"],
                "model_version": self.model_version
            }
            return

        # Calculate features (Pure functions) - VOLATILITY prefix recalculation removed!
        features_1m = compute_features(bars_1m, "1m", self.config)
        features_5m = compute_features(bars_5m, "5m", self.config)
        features_15m = compute_features(bars_15m, "15m", self.config)
        features_1h = compute_features(bars_1h, "1h", self.config)
        
        tf_feats = {
            "1m": features_1m[-1],
            "5m": features_5m[-1],
            "15m": features_15m[-1],
            "1h": features_1h[-1]
        }

        # Classify regime
        primary_cand, confidence, structure, direction, scores, reasons = classify_regime(tf_feats, self.config)

        # 3. Volatility Overlay
        vol_window = self.config.get("REGIME_VOL_PERCENTILE_WINDOW", 200)
        min_vol_samples = self.config.get("REGIME_VOL_MIN_SAMPLES", 100)
        
        # Bounded O(N) extraction of vol percentile values
        vol_samples = [f.get("realized_vol20") for f in features_15m if f.get("realized_vol20") is not None][-vol_window:]
        
        vol_regime = "UNKNOWN"
        curr_vol = tf_feats["15m"].get("realized_vol20")
        
        if len(vol_samples) >= min_vol_samples and curr_vol is not None:
            sorted_vols = sorted(vol_samples)
            rank = sum(1 for v in sorted_vols if v < curr_vol) / len(sorted_vols)
            if rank < 0.20:
                vol_regime = "LOW"
            elif rank < 0.80:
                vol_regime = "NORMAL"
            elif rank < 0.95:
                vol_regime = "HIGH"
            else:
                vol_regime = "EXTREME"

        # 4. Liquidity Overlay
        liq_window = self.config.get("REGIME_LIQUIDITY_WINDOW", 500)
        min_liq_samples = self.config.get("REGIME_LIQUIDITY_MIN_SAMPLES", 100)
        
        liq = self.liquidity_provider(symbol)
        spread_bps = liq.get("spread_bps")
        depth_usdt = liq.get("bid_depth_top5_usdt", 0.0) + liq.get("ask_depth_top5_usdt", 0.0)
        
        if safety["book_valid"] and spread_bps is not None and depth_usdt > 0.0:
            self.liquidity_baselines[symbol]["spread"].append(spread_bps)
            self.liquidity_baselines[symbol]["depth"].append(depth_usdt)
            if len(self.liquidity_baselines[symbol]["spread"]) > liq_window:
                self.liquidity_baselines[symbol]["spread"].pop(0)
                self.liquidity_baselines[symbol]["depth"].pop(0)

        liq_regime = "UNKNOWN"
        spread_history = self.liquidity_baselines[symbol]["spread"]
        depth_history = self.liquidity_baselines[symbol]["depth"]
        
        if len(spread_history) >= min_liq_samples and spread_bps is not None:
            sorted_spreads = sorted(spread_history)
            sorted_depths = sorted(depth_history)
            med_spread = sorted_spreads[len(sorted_spreads) // 2]
            med_depth = sorted_depths[len(sorted_depths) // 2]
            
            # Stressed first check order
            if spread_bps > med_spread * 3.0 or depth_usdt < med_depth * 0.2:
                liq_regime = "STRESSED"
            elif spread_bps > med_spread * 1.5 or depth_usdt < med_depth * 0.5:
                liq_regime = "THIN"
            else:
                liq_regime = "NORMAL"

        # 5. Hysteresis & Breakout Decay
        hyst = self.hysteresis_state[symbol]
        last_eval_ms = bars_1m[-1].close_time_ms
        
        if hyst["last_evaluation_close_ms"] == last_eval_ms:
            return
            
        hyst["last_evaluation_close_ms"] = last_eval_ms
        self.last_good_evaluation_ms[symbol] = last_eval_ms
        
        current_reg = hyst["current_regime"]
        max_breakout_bars = self.config.get("REGIME_BREAKOUT_MAX_BARS", 5)
        
        if current_reg in ("BREAKOUT_UP", "BREAKOUT_DOWN") and hyst["bars_since_regime_start"] >= max_breakout_bars:
            if current_reg == "BREAKOUT_UP":
                decayed_regime = "TREND_UP" if scores["trend_up"] >= 0.65 else "TRANSITION"
            else:
                decayed_regime = "TREND_DOWN" if scores["trend_down"] >= 0.65 else "TRANSITION"
                
            hyst["previous_regime"] = current_reg
            hyst["current_regime"] = decayed_regime
            hyst["regime_since_ms"] = last_eval_ms
            hyst["bars_since_regime_start"] = 0
            hyst["candidate"] = None
            hyst["count"] = 0
            
            self.recorder.record_transition({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "symbol": symbol,
                "old_regime": current_reg,
                "new_regime": decayed_regime,
                "confidence": confidence,
                "persistence_of_old": max_breakout_bars,
                "transition_reason": f"Breakout age decay after {max_breakout_bars} bars",
                "features": tf_feats["15m"]
            }, last_eval_ms)
            current_reg = decayed_regime

        confirm_bars = self.config.get("REGIME_SWITCH_CONFIRM_BARS", 3)
        min_conf = self.config.get("REGIME_MIN_CONFIDENCE", 0.65)
        
        if current_reg == "UNKNOWN" and primary_cand != "UNKNOWN" and confidence >= min_conf:
            hyst["previous_regime"] = "UNKNOWN"
            hyst["current_regime"] = primary_cand
            hyst["regime_since_ms"] = last_eval_ms
            hyst["bars_since_regime_start"] = 0
            hyst["candidate"] = None
            hyst["count"] = 0
            
            self.recorder.record_transition({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "symbol": symbol,
                "old_regime": "UNKNOWN",
                "new_regime": primary_cand,
                "confidence": confidence,
                "persistence_of_old": 0,
                "transition_reason": "Initial warmup assignment",
                "features": tf_feats["15m"]
            }, last_eval_ms)
        
        elif primary_cand != current_reg:
            if current_reg == "TREND_UP" and primary_cand == "BREAKOUT_UP":
                hyst["candidate"] = None
                hyst["count"] = 0
                hyst["bars_since_regime_start"] += 1
            elif current_reg == "TREND_DOWN" and primary_cand == "BREAKOUT_DOWN":
                hyst["candidate"] = None
                hyst["count"] = 0
                hyst["bars_since_regime_start"] += 1
            else:
                if hyst["candidate"] == primary_cand:
                    hyst["count"] += 1
                else:
                    hyst["candidate"] = primary_cand
                    hyst["count"] = 1
                    
                if hyst["count"] >= confirm_bars and confidence >= min_conf:
                    old_reg = current_reg
                    persistence_of_old = hyst["bars_since_regime_start"]
                    
                    hyst["previous_regime"] = old_reg
                    hyst["current_regime"] = primary_cand
                    hyst["regime_since_ms"] = last_eval_ms
                    hyst["bars_since_regime_start"] = 0
                    hyst["candidate"] = None
                    hyst["count"] = 0
                    
                    self.recorder.record_transition({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "symbol": symbol,
                        "old_regime": old_reg,
                        "new_regime": primary_cand,
                        "confidence": confidence,
                        "persistence_of_old": persistence_of_old,
                        "transition_reason": reasons[0] if reasons else "Hysteresis threshold reached",
                        "features": tf_feats["15m"]
                    }, last_eval_ms)
        else:
            hyst["candidate"] = None
            hyst["count"] = 0
            hyst["bars_since_regime_start"] += 1

        quality = "READY"
        if store.unresolved_gap_count > 0:
            quality = "DEGRADED"

        # Transition risk computation
        margin = 0.0
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0][1] - sorted_scores[1][1]
        transition_risk = 0.5 * (1.0 - margin) + (0.3 if hyst["candidate"] is not None else 0.0) + 0.2 * (1.0 if vol_regime in ("HIGH", "EXTREME") else 0.0)
        transition_risk = min(max(transition_risk, 0.0), 1.0)

        self.current_regimes[symbol] = {
            "symbol": symbol.upper(),
            "quality": quality,
            "tradable": True,
            "primary_regime": hyst["current_regime"],
            "confidence": confidence,
            "structure": structure,
            "direction": direction,
            "volatility": vol_regime,
            "liquidity": liq_regime,
            "scores": scores,
            "persistence_bars": hyst["bars_since_regime_start"],
            "regime_since_ms": hyst["regime_since_ms"],
            "candidate_regime": hyst["candidate"],
            "candidate_count": hyst["count"],
            "reasons": reasons,
            "model_version": self.model_version,
            "permissions": permissions_for(hyst["current_regime"]),
            "latest_1m_close_time": bars_1m[-1].close_time_ms if bars_1m else 0,
            "latest_5m_close_time": bars_5m[-1].close_time_ms if bars_5m else 0,
            "latest_15m_close_time": bars_15m[-1].close_time_ms if bars_15m else 0,
            "latest_1h_close_time": bars_1h[-1].close_time_ms if bars_1h else 0,
            "history_gap_count": store.history_gap_count,
            "unresolved_gap_count": store.unresolved_gap_count,
            "late_trade_count": store.late_trade_count,
            "duplicate_trade_count": store.duplicate_trade_count,
            "queue_overflow_count": store.queue_overflow_count,
            "reconciliation_mismatch_count": store.reconciliation_mismatch_count,
            "feature_version": self.feature_version,
            "transition_risk": transition_risk,
            "backfill_state": self.backfill_states.get(symbol, "NOT_STARTED"),
            "queue_depth": self.queue.qsize()
        }

        # Write state log
        state_log = self.current_regimes[symbol].copy()
        state_log["evaluation_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state_log["features"] = tf_feats["15m"]
        self.recorder.record_state(state_log)
