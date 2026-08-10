import time
from collections import deque
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import WINDOWS, LARGE_TRADE_NOTIONAL_USDT, TRADE_BURST_WINDOW, TRADE_BURST_MULTIPLIER
from data.stream_health import StreamHealthTracker
from data.local_order_book import LocalOrderBook

logger = logging.getLogger("OrderFlow.FlowMetrics")

class TradeEvent:
    """
    Canonical representation of a single trade event.
    """
    def __init__(self, symbol: str, exchange_time_ms: float, received_time_ms: float, price: float, quantity: float, aggregate_trade_id: int, side: str):
        self.symbol = symbol.lower()
        self.exchange_time_ms = exchange_time_ms
        self.received_time_ms = received_time_ms
        self.price = price
        self.quantity = quantity
        self.notional = price * quantity
        self.aggressor_side = side  # "BUY" or "SELL"
        self.aggregate_trade_id = aggregate_trade_id

class SymbolFlowState:
    """
    Encapsulates all order flow history, stream health, local order book, and CVD metrics for a symbol.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.max_history_seconds = max(WINDOWS.values()) + 10
        self.trades: deque = deque()
        self.start_time = time.time()
        
        # Instantiate health trackers
        self.trade_health_tracker = StreamHealthTracker(self.symbol, stream_name="trade")
        self.depth_health_tracker = StreamHealthTracker(self.symbol, stream_name="depth")
        self.local_book = LocalOrderBook(self.symbol, self.depth_health_tracker)

        # Instantiate liquidity intelligence engines
        from liquidity.wall_tracker import WallTracker
        from liquidity.stacking_pulling import StackingPullingDetector
        from liquidity.iceberg_inference import IcebergDetector
        from liquidity.liquidity_vacuum import LiquidityVacuumDetector
        from liquidity.spoof_risk import SpoofRiskDetector
        
        self.wall_tracker = WallTracker()
        self.stacking_pulling = StackingPullingDetector()
        self.iceberg_detector = IcebergDetector()
        self.vacuum_detector = LiquidityVacuumDetector()
        self.spoof_detector = SpoofRiskDetector()

        # CVD Stats (Normalized to USDT Notional)
        self.running_cvd_usdt = 0.0
        self.session_cvd_usdt = 0.0
        self.last_trade_date: Optional[datetime.date] = None

        # Order Book Stats (Normalized to USDT Notional)
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.spread = 0.0
        self.spread_bps = 0.0
        self.bid_depth_top5_usdt = 0.0
        self.ask_depth_top5_usdt = 0.0
        self.bid_ask_imbalance = 0.0
        self.microprice = 0.0
        self.microprice_dev = 0.0
        self.last_depth_timestamp = 0.0
        
        # Multidepth Imbalances
        self.imbalance_0_5_bps = 0.0
        self.imbalance_5_15_bps = 0.0
        self.imbalance_15_30_bps = 0.0
        self.imbalance_total = 0.0
        self.depth_weighted_imbalance = 0.0

        # Activity Scoring Trackers
        self.last_large_trade_time = 0.0
        self.last_event_time = 0.0
        self.latest_event = ""
        self.duplication_suspected = False

    @property
    def health_tracker(self):
        # Deprecated: use depth_health_tracker or trade_health_tracker explicitly
        return self.depth_health_tracker

class FlowMetrics:
    """
    Manages SymbolFlowState collections and exposes symbol-aware methods
    for trades and order book metric updates in USDT notional.
    """
    def __init__(self):
        self.symbol_states: Dict[str, SymbolFlowState] = {}

    def get_state(self, symbol: str) -> SymbolFlowState:
        """Retrieves or initializes the state object for a symbol (case-insensitive)."""
        symbol_lower = symbol.lower()
        if symbol_lower not in self.symbol_states:
            self.symbol_states[symbol_lower] = SymbolFlowState(symbol_lower)
        return self.symbol_states[symbol_lower]

    def add_trade(self, symbol: str, trade: dict) -> List[dict]:
        """
        Adds a new trade to the specified symbol's state, updates USDT CVD,
        prunes history, and returns any large trade alerts.
        """
        state = self.get_state(symbol)
        
        # Ensure we record stream event
        trade_time = trade["timestamp"]
        price = trade["price"]
        quantity = trade["quantity"]
        side = trade["side"]
        agg_id = trade.get("aggregate_trade_id", 0)

        exchange_time_ms = trade_time * 1000.0
        received_time_ms = trade.get("received_time", time.time()) * 1000.0
        processed_time_ms = time.time() * 1000.0

        # Record event in trade stream health tracker
        # (We use agg_id for the update_id metric parameter)
        state.trade_health_tracker.record_event(
            event_time_ms=trade_time,
            tx_time_ms=trade_time,
            received_time_ms=received_time_ms / 1000.0,
            processed_time_ms=processed_time_ms / 1000.0,
            update_id=agg_id
        )

        event = TradeEvent(
            symbol=symbol,
            exchange_time_ms=exchange_time_ms,
            received_time_ms=received_time_ms,
            price=price,
            quantity=quantity,
            aggregate_trade_id=agg_id,
            side=side
        )

        # 1. Update CVD (in USDT Notional)
        delta_contrib_usdt = event.notional if side == "BUY" else -event.notional
        state.running_cvd_usdt += delta_contrib_usdt
        
        state.iceberg_detector.record_trade(price, event.notional)

        # Session CVD resets at UTC day boundary
        trade_dt = datetime.fromtimestamp(trade_time, tz=timezone.utc)
        trade_date = trade_dt.date()
        if state.last_trade_date is None:
            state.last_trade_date = trade_date
            state.session_cvd_usdt = delta_contrib_usdt
        elif trade_date != state.last_trade_date:
            logger.info(f"[{symbol.upper()}] UTC day boundary detected. Resetting Session CVD (was ${state.session_cvd_usdt:,.2f}).")
            state.last_trade_date = trade_date
            state.session_cvd_usdt = delta_contrib_usdt
        else:
            state.session_cvd_usdt += delta_contrib_usdt

        # 2. Append trade and prune history
        state.trades.append(event)
        cutoff_ms = exchange_time_ms - (state.max_history_seconds * 1000.0)
        while state.trades and state.trades[0].exchange_time_ms < cutoff_ms:
            state.trades.popleft()

        # 3. Detect large trade alerts in USDT Notional
        alerts = []
        if event.notional >= LARGE_TRADE_NOTIONAL_USDT:
            state.last_large_trade_time = time.time()
            alerts.append({
                "type": "LARGE_TRADE",
                "side": side,
                "quantity": quantity,
                "price": price,
                "notional_usdt": event.notional,
                "timestamp": trade_time,
                "symbol": symbol
            })

        return alerts

    def get_metrics_for_window(self, symbol: str, window_name: str) -> Dict[str, Any]:
        """
        Computes aggregate metrics (base and USDT notional) for a symbol's window.
        Uses the canonical trade event deque.
        """
        state = self.get_state(symbol)
        window_seconds = WINDOWS.get(window_name, 60)
        
        elapsed = time.time() - state.start_time
        is_warming = elapsed < window_seconds
        
        if not state.trades:
            res = self._empty_metrics()
            res["symbol"] = symbol
            if is_warming:
                res["status"] = "WARMING_UP"
                elapsed_m, elapsed_s = divmod(int(elapsed), 60)
                total_m, total_s = divmod(window_seconds, 60)
                res["warmup_text"] = f"WARMING {elapsed_m:02d}:{elapsed_s:02d} / {total_m:02d}:{total_s:02d}"
            return res

        current_time_ms = state.trades[-1].exchange_time_ms
        cutoff_ms = current_time_ms - (window_seconds * 1000.0)

        buy_vol_base = 0.0
        sell_vol_base = 0.0
        buy_vol_usdt = 0.0
        sell_vol_usdt = 0.0
        trade_count = 0

        # Scan backwards for efficiency
        for event in reversed(state.trades):
            if event.exchange_time_ms < cutoff_ms:
                break
            trade_count += 1
            if event.aggressor_side == "BUY":
                buy_vol_base += event.quantity
                buy_vol_usdt += event.notional
            else:
                sell_vol_base += event.quantity
                sell_vol_usdt += event.notional

        total_vol_base = buy_vol_base + sell_vol_base
        total_vol_usdt = buy_vol_usdt + sell_vol_usdt
        
        delta_base = buy_vol_base - sell_vol_base
        delta_usdt = buy_vol_usdt - sell_vol_usdt
        
        buy_ratio_usdt = buy_vol_usdt / total_vol_usdt if total_vol_usdt > 0 else 0.5
        sell_ratio_usdt = sell_vol_usdt / total_vol_usdt if total_vol_usdt > 0 else 0.5

        # Check for duplication suspicion after warm-up
        # (This is evaluated over the state object when requested)
        res = {
            "status": "WARMING_UP" if is_warming else "VALID",
            "buy_volume": buy_vol_base,
            "sell_volume": sell_vol_base,
            "buy_volume_usdt": buy_vol_usdt,
            "sell_volume_usdt": sell_vol_usdt,
            "delta": delta_base,
            "delta_usdt": delta_usdt,
            "buy_ratio": buy_ratio_usdt,
            "sell_ratio": sell_ratio_usdt,
            "trade_count": trade_count,
            "total_volume": total_vol_base,
            "total_volume_usdt": total_vol_usdt,
            "latest_price": state.trades[-1].price if state.trades else 0.0,
            "symbol": symbol
        }
        
        if is_warming:
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            total_m, total_s = divmod(window_seconds, 60)
            res["warmup_text"] = f"WARMING {elapsed_m:02d}:{elapsed_s:02d} / {total_m:02d}:{total_s:02d}"
            
        return res

    def check_duplication(self, symbol: str, m5: dict, m15: dict):
        """Checks if 5m and 15m deltas are exactly identical after warm-up."""
        state = self.get_state(symbol)
        elapsed = time.time() - state.start_time
        if elapsed >= 900.0:
            if abs(m5["delta_usdt"] - m15["delta_usdt"]) < 1e-6 and m5["trade_count"] > 0:
                state.duplication_suspected = True
                return
        state.duplication_suspected = False

    def detect_trade_burst(self, symbol: str) -> Optional[dict]:
        """
        Checks if the trade count in the last TRADE_BURST_WINDOW seconds is
        significantly higher than the recent average (preceding 5 minutes) for a symbol.
        """
        state = self.get_state(symbol)
        if not state.trades:
            return None

        current_time_ms = state.trades[-1].exchange_time_ms
        
        # Window of interest (last 10 seconds)
        burst_cutoff_ms = current_time_ms - (TRADE_BURST_WINDOW * 1000.0)
        # Baseline window (last 5 minutes, excluding the last 10 seconds)
        baseline_cutoff_ms = current_time_ms - 300000.0

        current_window_trades = 0
        baseline_trades = 0

        for event in reversed(state.trades):
            t_ms = event.exchange_time_ms
            if t_ms < baseline_cutoff_ms:
                break
            if t_ms >= burst_cutoff_ms:
                current_window_trades += 1
            else:
                baseline_trades += 1

        baseline_intervals = 29.0  # 290 seconds / 10 seconds
        avg_trades_per_interval = baseline_trades / baseline_intervals if baseline_intervals > 0 else 0.0
        
        if avg_trades_per_interval < 5:
            avg_trades_per_interval = 5.0

        threshold = avg_trades_per_interval * TRADE_BURST_MULTIPLIER

        if current_window_trades > threshold:
            return {
                "type": "TRADE_BURST",
                "current_count": current_window_trades,
                "average_count": int(avg_trades_per_interval),
                "timestamp": current_time_ms / 1000.0,
                "price": state.trades[-1].price,
                "symbol": symbol
            }
        return None

    def update_depth(self, symbol: str, depth_data: dict, received_time: float):
        """Updates internal order book and stream health trackers using the diff depth stream."""
        state = self.get_state(symbol)
        state.local_book.handle_ws_update(depth_data, received_time)
        
        if state.local_book.is_valid:
            sorted_bids, sorted_asks = state.local_book.get_top_bids_asks(levels=100)
            if sorted_bids and sorted_asks:
                state.best_bid = sorted_bids[0][0]
                state.best_ask = sorted_asks[0][0]
                state.spread = state.best_ask - state.best_bid
                
                mid_price = (state.best_bid + state.best_ask) / 2.0
                state.spread_bps = (state.spread / mid_price) * 10000.0 if mid_price > 0 else 0.0
                
                sorted_bids_top5 = sorted_bids[:5]
                sorted_asks_top5 = sorted_asks[:5]
                state.bid_depth_top5_usdt = sum(b[0] * b[1] for b in sorted_bids_top5)
                state.ask_depth_top5_usdt = sum(a[0] * a[1] for a in sorted_asks_top5)
                
                total_depth_usdt = state.bid_depth_top5_usdt + state.ask_depth_top5_usdt
                state.bid_ask_imbalance = (state.bid_depth_top5_usdt - state.ask_depth_top5_usdt) / total_depth_usdt if total_depth_usdt > 0 else 0.0
                state.last_depth_timestamp = state.local_book.last_event_time
                
                bid_qty_1 = sorted_bids[0][1]
                ask_qty_1 = sorted_asks[0][1]
                total_qty_1 = bid_qty_1 + ask_qty_1
                state.microprice = (state.best_bid * ask_qty_1 + state.best_ask * bid_qty_1) / total_qty_1 if total_qty_1 > 0 else mid_price
                state.microprice_dev = (state.microprice - mid_price) / mid_price * 10000.0 if mid_price > 0 else 0.0

                # Bucketed Imbalances
                bid_not_0_5 = 0.0; ask_not_0_5 = 0.0
                bid_not_5_15 = 0.0; ask_not_5_15 = 0.0
                bid_not_15_30 = 0.0; ask_not_15_30 = 0.0
                bid_not_tot = 0.0; ask_not_tot = 0.0
                
                bid_w_sum = 0.0; ask_w_sum = 0.0

                for p, q in sorted_bids:
                    notional = p * q
                    bid_not_tot += notional
                    dist_bps = (mid_price - p) / mid_price * 10000.0 if mid_price > 0 else 999.0
                    if dist_bps <= 5.0:
                        bid_not_0_5 += notional
                    elif dist_bps <= 15.0:
                        bid_not_5_15 += notional
                    elif dist_bps <= 30.0:
                        bid_not_15_30 += notional
                    
                    if dist_bps <= 30.0:
                        w = 1.0 - (dist_bps / 30.0)
                        bid_w_sum += w * notional

                for p, q in sorted_asks:
                    notional = p * q
                    ask_not_tot += notional
                    dist_bps = (p - mid_price) / mid_price * 10000.0 if mid_price > 0 else 999.0
                    if dist_bps <= 5.0:
                        ask_not_0_5 += notional
                    elif dist_bps <= 15.0:
                        ask_not_5_15 += notional
                    elif dist_bps <= 30.0:
                        ask_not_15_30 += notional

                    if dist_bps <= 30.0:
                        w = 1.0 - (dist_bps / 30.0)
                        ask_w_sum += w * notional

                state.imbalance_0_5_bps = (bid_not_0_5 - ask_not_0_5) / (bid_not_0_5 + ask_not_0_5) if (bid_not_0_5 + ask_not_0_5) > 0 else 0.0
                state.imbalance_5_15_bps = (bid_not_5_15 - ask_not_5_15) / (bid_not_5_15 + ask_not_5_15) if (bid_not_5_15 + ask_not_5_15) > 0 else 0.0
                state.imbalance_15_30_bps = (bid_not_15_30 - ask_not_15_30) / (bid_not_15_30 + ask_not_15_30) if (bid_not_15_30 + ask_not_15_30) > 0 else 0.0
                state.imbalance_total = (bid_not_tot - ask_not_tot) / (bid_not_tot + ask_not_tot) if (bid_not_tot + ask_not_tot) > 0 else 0.0
                state.depth_weighted_imbalance = (bid_w_sum - ask_w_sum) / (bid_w_sum + ask_w_sum) if (bid_w_sum + ask_w_sum) > 0 else 0.0
                
                # Record stacking/pulling snapshot
                bids_dict = {p: q for p, q in sorted_bids}
                asks_dict = {p: q for p, q in sorted_asks}
                state.stacking_pulling.record_snapshot(received_time, bids_dict, asks_dict)

                # Update Wall Tracker
                state.wall_tracker.update_walls(sorted_bids, sorted_asks, mid_price, state.bid_depth_top5_usdt, received_time)
        else:
            state.best_bid = 0.0
            state.best_ask = 0.0
            state.spread = 0.0
            state.spread_bps = 0.0
            state.bid_depth_top5_usdt = 0.0
            state.ask_depth_top5_usdt = 0.0
            state.bid_ask_imbalance = 0.0
            state.microprice = 0.0
            state.microprice_dev = 0.0
            state.imbalance_0_5_bps = 0.0
            state.imbalance_5_15_bps = 0.0
            state.imbalance_15_30_bps = 0.0
            state.imbalance_total = 0.0
            state.depth_weighted_imbalance = 0.0

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "status": "VALID",
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "buy_volume_usdt": 0.0,
            "sell_volume_usdt": 0.0,
            "delta": 0.0,
            "delta_usdt": 0.0,
            "buy_ratio": 0.5,
            "sell_ratio": 0.5,
            "trade_count": 0,
            "total_volume": 0.0,
            "total_volume_usdt": 0.0,
            "latest_price": 0.0,
            "symbol": ""
        }

    def get_market_data_safety(self, symbol: str, now: Optional[float] = None) -> Dict[str, Any]:
        state = self.get_state(symbol)
        now_val = now if now is not None else time.time()
        
        book_valid = state.local_book.is_valid
        book_state = state.local_book.state
        
        trade_status = state.trade_health_tracker.get_status(now=now_val)
        depth_status = state.depth_health_tracker.get_status(is_book_valid=book_valid, now=now_val)
        
        depth_silence = state.depth_health_tracker.get_silence_age_ms(now=now_val)
        trade_silence = state.trade_health_tracker.get_silence_age_ms(now=now_val)
        
        safe = True
        status = "HEALTHY"
        reason = None
        
        if not book_valid:
            safe = False
            status = "DATA_INVALID"
            reason = f"Local order book invalid (state: {book_state})"
        elif depth_status == "INITIALISING":
            safe = False
            status = "DATA_INVALID"
            reason = "Depth stream has not received valid data"
        elif trade_status == "INITIALISING":
            safe = False
            status = "DATA_INVALID"
            reason = "Trade stream has not received valid data"
        elif depth_status == "STALE":
            safe = False
            status = "DATA_STALE"
            reason = f"Depth stream stale: {depth_silence:.0f}ms silence"
        elif trade_status == "STALE":
            safe = False
            status = "DATA_STALE"
            reason = f"Trade stream stale: {trade_silence:.0f}ms silence"
        elif depth_status == "INVALID":
            safe = False
            status = "DATA_INVALID"
            reason = "Depth stream status INVALID"
            
        return {
            "safe": safe,
            "status": status,
            "reason": reason,
            "book_valid": book_valid,
            "book_state": book_state,
            "trade_status": trade_status,
            "depth_status": depth_status,
            "depth_age_ms": depth_silence if depth_silence is not None else 9999.0,
            "trade_silence_ms": trade_silence if trade_silence is not None else 9999.0,
            "depth_silence_ms": depth_silence if depth_silence is not None else 9999.0
        }

