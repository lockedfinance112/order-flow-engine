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
    Processes live minute candles from trade flow, manages multi-timeframe aggregation,
    computes pure point-in-time features, and runs the deterministic regime classifier.
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
        self.model_version = self.config.get("REGIME_MODEL_VERSION", "regime-v1")
        self.feature_version = self.config.get("REGIME_FEATURE_VERSION", "regime-features-v1")
        self.max_bars = self.config.get("REGIME_MAX_BARS_PER_TIMEFRAME", 2000)
        self.dedup_capacity = self.config.get("REGIME_TRADE_DEDUP_CAPACITY", 5000)
        self.max_late_ms = self.config.get("REGIME_MAX_LATE_TRADE_MS", 2000)
        
        # Symbol States
        self.stores: Dict[str, BarStore] = {s: BarStore(s, self.max_bars) for s in self.symbols}
        self.backfill_states: Dict[str, str] = {s: "NOT_STARTED" for s in self.symbols}
        self.current_regimes: Dict[str, Dict[str, Any]] = {}
        
        # Live 1m Bar Builders (current open bar): symbol -> dict
        self.live_bars: Dict[str, Dict[str, Any]] = {}
        
        # Deduplication States: symbol -> (deque of IDs, set of IDs)
        self.dedup_ids: Dict[str, Tuple[deque, set]] = {
            s: (deque(maxlen=self.dedup_capacity), set()) for s in self.symbols
        }

        # Liquidity Rolling Baselines: symbol -> { "spread_bps": List, "depth_usdt": List }
        self.liquidity_baselines: Dict[str, Dict[str, List[float]]] = {
            s: {"spread": [], "depth": []} for s in self.symbols
        }
        
        # Hysteresis memory: symbol -> { "candidate": str, "count": int, "last_evaluation_close_ms": int, "bars_since_regime_start": int, "regime_since_ms": int }
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

        # Bounded Closed Bar Queue (for worker thread processing)
        self.queue = asyncio.Queue(maxsize=1000)
        self.recorder = RegimeRecorder()
        
        # Worker Task
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False

    def start(self):
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        # Start background backfill
        for symbol in self.symbols:
            asyncio.create_task(self._backfill_symbol(symbol))

    def stop(self):
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()

    def get_regime_state(self, symbol: str) -> Dict[str, Any]:
        """Reads cached canonical regime state for API/Dashboard usage."""
        symbol_lower = symbol.lower()
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
        """
        Extremely cheap aggTrade callback.
        Routes trade to deduplication, incremental bar construction, and enqueues closed bars.
        """
        symbol_lower = symbol.lower()
        if symbol_lower not in self.symbols:
            return

        trade_id = trade.get("aggregate_trade_id")
        trade_time_ms = trade.get("trade_time_ms")
        if trade_id is None or trade_time_ms is None:
            return

        store = self.stores[symbol_lower]

        # 1. Deduplication (O(1) set membership)
        dq, s_set = self.dedup_ids[symbol_lower]
        if trade_id in s_set:
            store.duplicate_trade_count += 1
            return
            
        dq.append(trade_id)
        s_set.add(trade_id)
        if len(dq) >= self.dedup_capacity:
            oldest = dq.popleft()
            s_set.discard(oldest)

        # 2. Bucket Authority: T / trade_time_ms determines 1m candle alignment
        bucket_open_ms = (trade_time_ms // 60000) * 60000
        bucket_close_ms = bucket_open_ms + 60000 - 1

        price = trade["price"]
        qty = trade["quantity"]
        quote_qty = price * qty

        live = self.live_bars.get(symbol_lower)

        # First trade initialization
        if not live:
            self.live_bars[symbol_lower] = {
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

        # Handle bucket transition based on exchange trade-time progression
        if bucket_open_ms > live["open_time_ms"]:
            # Finalize previous live bar
            finalized_bar = MarketBar(
                symbol=symbol_lower,
                timeframe="1m",
                open_time_ms=live["open_time_ms"],
                close_time_ms=live["close_time_ms"],
                open=live["open"],
                high=live["high"],
                low=live["low"],
                close=live["close"],
                base_volume=live["base_volume"],
                quote_volume=live["quote_volume"],
                agg_trade_count=live["agg_trade_count"],
                closed=True
            )
            
            # Enqueue to background worker queue
            try:
                self.queue.put_nowait((symbol_lower, finalized_bar))
            except asyncio.QueueFull:
                store.queue_overflow_count += 1
                store.history_gap_count += 1 # Queue dropped bar requires gap backfill
                logger.error(f"[{symbol_lower.upper()}] Regime queue full! Dropped bar at {live['open_time_ms']}.")

            # Open next bucket
            self.live_bars[symbol_lower] = {
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

        elif bucket_open_ms < live["open_time_ms"]:
            # Late trade check
            lateness = live["first_trade_time_ms"] - trade_time_ms
            if lateness > self.max_late_ms:
                store.late_trade_count += 1
                return # Reject stale rewrite
            else:
                # Integrate if it's within lateness allowance and belongs to a preceding open bar
                store.out_of_order_trade_count += 1
                # Incrementally aggregate without full sort
                if trade_time_ms < live["first_trade_time_ms"]:
                    live["open"] = price
                    live["first_trade_time_ms"] = trade_time_ms
                if trade_time_ms > live["last_trade_time_ms"]:
                    live["close"] = price
                    live["last_trade_time_ms"] = trade_time_ms
                live["high"] = max(live["high"], price)
                live["low"] = min(live["low"], price)
                live["base_volume"] += qty
                live["quote_volume"] += quote_qty
                live["agg_trade_count"] += 1
        else:
            # Same-minute normal/out-of-order aggregation
            if trade_time_ms < live["last_trade_time_ms"]:
                store.out_of_order_trade_count += 1
                
            if trade_time_ms < live["first_trade_time_ms"]:
                live["open"] = price
                live["first_trade_time_ms"] = trade_time_ms
            if trade_time_ms > live["last_trade_time_ms"]:
                live["close"] = price
                live["last_trade_time_ms"] = trade_time_ms
                
            live["high"] = max(live["high"], price)
            live["low"] = min(live["low"], price)
            live["base_volume"] += qty
            live["quote_volume"] += quote_qty
            live["agg_trade_count"] += 1

    async def _backfill_symbol(self, symbol: str):
        self.backfill_states[symbol] = "LOADING"
        logger.info(f"[{symbol.upper()}] Starting historical kline backfill...")
        
        from regime.history_loader import fetch_klines_async
        timeframes = ["1m", "5m", "15m", "1h"]
        
        try:
            for tf in timeframes:
                bars = await fetch_klines_async(symbol, tf, limit=500)
                store = self.stores[symbol]
                for bar in bars:
                    store.append_bar(tf, bar)
                    
            self.backfill_states[symbol] = "READY"
            logger.info(f"[{symbol.upper()}] Historical backfill completed successfully.")
            # Trigger initial classification immediately
            self._process_symbol_regime(symbol, int(time.time() * 1000))
        except Exception as e:
            self.backfill_states[symbol] = "FAILED"
            logger.error(f"[{symbol.upper()}] Historical backfill failed: {e}")

    async def _worker_loop(self):
        while self.is_running:
            try:
                symbol, bar = await self.queue.get()
                store = self.stores[symbol]
                
                # Append 1m bar to store
                if store.append_bar("1m", bar):
                    # Check and construct higher TF candles using component completeness
                    self._aggregate_higher_tfs(symbol, bar)
                    # Evaluate regime
                    self._process_symbol_regime(symbol, bar.close_time_ms)
                    
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regime worker loop: {e}")
                await asyncio.sleep(0.1)

    def _aggregate_higher_tfs(self, symbol: str, new_1m_bar: MarketBar):
        """Builds UTC-aligned closed 5m, 15m, and 1h bars from closed 1m bars."""
        store = self.stores[symbol]
        m1_bars = store.get_bars("1m")
        
        def aggregate_components(open_time_ms: int, count: int, tf: str) -> Optional[MarketBar]:
            # Find the component bars in 1m history
            components = [b for b in m1_bars if open_time_ms <= b.open_time_ms < open_time_ms + (count * 60000)]
            if len(components) < count:
                # Completeness check failed -> Degraded gap state
                store.history_gap_count += 1
                return None
                
            open_val = components[0].open
            high_val = max(b.high for b in components)
            low_val = min(b.low for b in components)
            close_val = components[-1].close
            base_vol = sum(b.base_volume for b in components)
            quote_vol = sum(b.quote_volume for b in components)
            agg_cnt = sum(b.agg_trade_count for b in components)
            
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
                agg_trade_count=agg_cnt,
                closed=True
            )

        # Check 5m boundary: just closed 1m bar completed the 5m interval
        # If new_1m_bar close time completes a 5m boundary (e.g. xx:04.999 closes 5m bar open at xx:00)
        close_time = new_1m_bar.open_time_ms + 60000
        if close_time % 300000 == 0:
            m5_open = close_time - 300000
            m5_bar = aggregate_components(m5_open, 5, "5m")
            if m5_bar:
                store.append_bar("5m", m5_bar)

        # Check 15m boundary
        if close_time % 900000 == 0:
            m15_open = close_time - 900000
            m15_bar = aggregate_components(m15_open, 15, "15m")
            if m15_bar:
                store.append_bar("15m", m15_bar)

        # Check 1h boundary
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
            self.current_regimes[symbol] = {
                "symbol": symbol.upper(),
                "quality": "STALE",
                "tradable": False,
                "primary_regime": self.hysteresis_state[symbol]["current_regime"],
                "last_good_age_seconds": (event_time_ms - self.hysteresis_state[symbol]["regime_since_ms"]) / 1000.0 if self.hysteresis_state[symbol]["regime_since_ms"] > 0 else 999.0,
                "reason": safety["reason"] or "Data Guardian unsafe",
                "model_version": self.model_version
            }
            return

        # 2. Quality/Warmup verification
        bars_1m = store.get_bars("1m")
        bars_5m = store.get_bars("5m")
        bars_15m = store.get_bars("15m")
        bars_1h = store.get_bars("1h")
        
        # Require minimum historical closed bars to calculate features
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

        # Calculate features (Pure functions)
        tf_feats = {
            "1m": compute_features(bars_1m, "1m", self.config)[-1],
            "5m": compute_features(bars_5m, "5m", self.config)[-1],
            "15m": compute_features(bars_15m, "15m", self.config)[-1],
            "1h": compute_features(bars_1h, "1h", self.config)[-1]
        }

        # Classify regime
        primary_cand, confidence, structure, direction, scores, reasons = classify_regime(tf_feats, self.config)

        # 3. Volatility Overlay (rolling percentile-based)
        # Track 15m realized volatility in rolling window
        vol_window = self.config.get("REGIME_VOL_PERCENTILE_WINDOW", 200)
        min_vol_samples = self.config.get("REGIME_VOL_MIN_SAMPLES", 100)
        
        vol_samples = [compute_features(bars_15m[:j+1], "15m", self.config)[-1].get("realized_vol20") for j in range(len(bars_15m))]
        vol_samples = [v for v in vol_samples if v is not None][-vol_window:]
        
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
        
        # Sample current spread/depth if book is valid
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
            # Median baseline calculations
            sorted_spreads = sorted(spread_history)
            sorted_depths = sorted(depth_history)
            med_spread = sorted_spreads[len(sorted_spreads) // 2]
            med_depth = sorted_depths[len(sorted_depths) // 2]
            
            # Simple thresholding logic relative to median
            if spread_bps > med_spread * 1.5 or depth_usdt < med_depth * 0.5:
                liq_regime = "THIN"
            elif spread_bps > med_spread * 3.0 or depth_usdt < med_depth * 0.2:
                liq_regime = "STRESSED"
            else:
                liq_regime = "NORMAL"

        # 5. Hysteresis Block (Transitions apply confirm bars)
        hyst = self.hysteresis_state[symbol]
        last_eval_ms = bars_1m[-1].open_time_ms
        
        # Prevent double evaluations of the same closed bar
        if hyst["last_evaluation_close_ms"] == last_eval_ms:
            return
            
        hyst["last_evaluation_close_ms"] = last_eval_ms
        
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
        
        # Skip hysteresis if initial assignment from UNKNOWN and confidence >= min_conf
        if current_reg == "UNKNOWN" and primary_cand != "UNKNOWN" and confidence >= min_conf:
            hyst["previous_regime"] = "UNKNOWN"
            hyst["current_regime"] = primary_cand
            hyst["regime_since_ms"] = last_eval_ms
            hyst["bars_since_regime_start"] = 0
            hyst["candidate"] = None
            hyst["count"] = 0
            
            # Record transition
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
                # Candidate tracking
                if hyst["candidate"] == primary_cand:
                    hyst["count"] += 1
                else:
                    hyst["candidate"] = primary_cand
                    hyst["count"] = 1
                    
                if hyst["count"] >= confirm_bars and confidence >= min_conf:
                    # Transition confirmed!
                    old_reg = current_reg
                    persistence_of_old = hyst["bars_since_regime_start"]
                    
                    hyst["previous_regime"] = old_reg
                    hyst["current_regime"] = primary_cand
                    hyst["regime_since_ms"] = last_eval_ms
                    hyst["bars_since_regime_start"] = 0
                    hyst["candidate"] = None
                    hyst["count"] = 0
                    
                    # Record transition
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
            # Candidate interrupted/reset
            hyst["candidate"] = None
            hyst["count"] = 0
            hyst["bars_since_regime_start"] += 1

        # Cache final state
        quality = "READY"
        if store.history_gap_count > 0:
            quality = "DEGRADED"

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
            # Provenance metadata
            "latest_1m_close_time": last_eval_ms,
            "latest_5m_close_time": bars_5m[-1].open_time_ms if bars_5m else 0,
            "latest_15m_close_time": bars_15m[-1].open_time_ms if bars_15m else 0,
            "latest_1h_close_time": bars_1h[-1].open_time_ms if bars_1h else 0,
            "history_gap_count": store.history_gap_count,
            "late_trade_count": store.late_trade_count,
            "duplicate_trade_count": store.duplicate_trade_count,
            "queue_overflow_count": store.queue_overflow_count,
            "reconciliation_mismatch_count": store.reconciliation_mismatch_count
        }

        # Write to state log
        state_log = self.current_regimes[symbol].copy()
        state_log["evaluation_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state_log["features"] = tf_feats["15m"]
        self.recorder.record_state(state_log)
