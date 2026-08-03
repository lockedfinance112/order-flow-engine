from collections import deque
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import WINDOWS, LARGE_TRADE_NOTIONAL_USDT, TRADE_BURST_WINDOW, TRADE_BURST_MULTIPLIER

logger = logging.getLogger("OrderFlow.FlowMetrics")

class SymbolFlowState:
    """
    Encapsulates all order flow history and parameters for a single symbol
    to prevent cross-symbol contamination in a multi-symbol scanner.
    Supports both base token and USDT notional valuations.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.max_history_seconds = max(WINDOWS.values()) + 10
        self.trades: deque = deque()

        # CVD Stats (Normalized to USDT Notional)
        self.running_cvd_usdt = 0.0
        self.session_cvd_usdt = 0.0
        self.last_trade_date: Optional[datetime.date] = None

        # Order Book Stats (Normalized to USDT Notional)
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.spread = 0.0
        self.bid_depth_top5_usdt = 0.0
        self.ask_depth_top5_usdt = 0.0
        self.bid_ask_imbalance = 0.0
        self.microprice = 0.0
        self.last_depth_timestamp = 0.0

        # Activity Scoring Trackers
        self.last_large_trade_time = 0.0
        self.last_event_time = 0.0
        self.latest_event = ""


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
        
        trade_time = trade["timestamp"]
        quantity = trade["quantity"]
        side = trade["side"]
        price = trade["price"]

        # Calculate Notional USDT
        notional_usdt = price * quantity
        trade["notional_usdt"] = notional_usdt

        # 1. Update CVD (in USDT Notional)
        delta_contrib_usdt = notional_usdt if side == "BUY" else -notional_usdt
        state.running_cvd_usdt += delta_contrib_usdt

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
        state.trades.append(trade)
        cutoff = trade_time - state.max_history_seconds
        while state.trades and state.trades[0]["timestamp"] < cutoff:
            state.trades.popleft()

        # 3. Detect large trade alerts in USDT Notional
        alerts = []
        if notional_usdt >= LARGE_TRADE_NOTIONAL_USDT:
            state.last_large_trade_time = time.time()
            alerts.append({
                "type": "LARGE_TRADE",
                "side": side,
                "quantity": quantity,
                "price": price,
                "notional_usdt": notional_usdt,
                "timestamp": trade_time,
                "symbol": symbol
            })

        return alerts

    def get_metrics_for_window(self, symbol: str, window_name: str) -> Dict[str, Any]:
        """
        Computes aggregate metrics (base and USDT notional) for a symbol's window.
        """
        state = self.get_state(symbol)
        window_seconds = WINDOWS.get(window_name, 60)
        
        if not state.trades:
            return self._empty_metrics()

        current_time = state.trades[-1]["timestamp"]
        cutoff = current_time - window_seconds

        buy_vol_base = 0.0
        sell_vol_base = 0.0
        buy_vol_usdt = 0.0
        sell_vol_usdt = 0.0
        trade_count = 0

        # Scan backwards for efficiency
        for trade in reversed(state.trades):
            if trade["timestamp"] < cutoff:
                break
            trade_count += 1
            notional = trade.get("notional_usdt", trade["price"] * trade["quantity"])
            
            if trade["side"] == "BUY":
                buy_vol_base += trade["quantity"]
                buy_vol_usdt += notional
            else:
                sell_vol_base += trade["quantity"]
                sell_vol_usdt += notional

        total_vol_base = buy_vol_base + sell_vol_base
        total_vol_usdt = buy_vol_usdt + sell_vol_usdt
        
        delta_base = buy_vol_base - sell_vol_base
        delta_usdt = buy_vol_usdt - sell_vol_usdt
        
        buy_ratio_usdt = buy_vol_usdt / total_vol_usdt if total_vol_usdt > 0 else 0.5
        sell_ratio_usdt = sell_vol_usdt / total_vol_usdt if total_vol_usdt > 0 else 0.5

        return {
            "buy_volume": buy_vol_base,
            "sell_volume": sell_vol_base,
            "buy_volume_usdt": buy_vol_usdt,
            "sell_volume_usdt": sell_vol_usdt,
            "delta": delta_base,
            "delta_usdt": delta_usdt,
            "buy_ratio": buy_ratio_usdt, # Normalized ratio based on notional flows
            "sell_ratio": sell_ratio_usdt,
            "trade_count": trade_count,
            "total_volume": total_vol_base,
            "total_volume_usdt": total_vol_usdt,
            "latest_price": state.trades[-1]["price"] if state.trades else 0.0,
            "symbol": symbol
        }

    def detect_trade_burst(self, symbol: str) -> Optional[dict]:
        """
        Checks if the trade count in the last TRADE_BURST_WINDOW seconds is
        significantly higher than the recent average (preceding 5 minutes) for a symbol.
        """
        state = self.get_state(symbol)
        if not state.trades:
            return None

        current_time = state.trades[-1]["timestamp"]
        
        # Window of interest (last 10 seconds)
        burst_cutoff = current_time - TRADE_BURST_WINDOW
        # Baseline window (last 5 minutes, excluding the last 10 seconds)
        baseline_cutoff = current_time - 300

        current_window_trades = 0
        baseline_trades = 0

        for trade in reversed(state.trades):
            t = trade["timestamp"]
            if t < baseline_cutoff:
                break
            if t >= burst_cutoff:
                current_window_trades += 1
            else:
                baseline_trades += 1

        # We have 290 seconds of baseline. Average trades per 10 seconds:
        baseline_intervals = (burst_cutoff - baseline_cutoff) / TRADE_BURST_WINDOW
        if baseline_intervals <= 0:
            return None

        avg_trades_per_interval = baseline_trades / baseline_intervals
        
        # Avoid triggering in ultra-low activity environments
        if avg_trades_per_interval < 5:
            avg_trades_per_interval = 5.0

        threshold = avg_trades_per_interval * TRADE_BURST_MULTIPLIER

        if current_window_trades > threshold:
            return {
                "type": "TRADE_BURST",
                "current_count": current_window_trades,
                "average_count": int(avg_trades_per_interval),
                "timestamp": current_time,
                "price": state.trades[-1]["price"],
                "symbol": symbol
            }
        return None

    def update_depth(self, symbol: str, depth_data: dict):
        """Updates internal order book pressure parameters using top 5 Bids/Asks in USDT notional."""
        state = self.get_state(symbol)
        bids = depth_data["bids"]
        asks = depth_data["asks"]
        
        if not bids or not asks:
            return
            
        state.best_bid = bids[0][0]
        state.best_ask = asks[0][0]
        state.spread = state.best_ask - state.best_bid
        
        # Calculate depth top 5 in USDT notionals (price * qty)
        state.bid_depth_top5_usdt = sum(b[0] * b[1] for b in bids)
        state.ask_depth_top5_usdt = sum(a[0] * a[1] for a in asks)
        
        total_depth_usdt = state.bid_depth_top5_usdt + state.ask_depth_top5_usdt
        state.bid_ask_imbalance = (state.bid_depth_top5_usdt - state.ask_depth_top5_usdt) / total_depth_usdt if total_depth_usdt > 0 else 0.0
        state.last_depth_timestamp = depth_data.get("timestamp", time.time())
        
        bid_qty_1 = bids[0][1]
        ask_qty_1 = asks[0][1]
        total_qty_1 = bid_qty_1 + ask_qty_1
        state.microprice = (state.best_bid * ask_qty_1 + state.best_ask * bid_qty_1) / total_qty_1 if total_qty_1 > 0 else (state.best_bid + state.best_ask) / 2.0

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
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
