import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("OrderFlow.PaperTrader")

class PaperTrader:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.realized_pnl = 0.0
        self.positions: Dict[str, dict] = {} # symbol -> {qty: float, entry_price: float, side: str}
        self.trades: List[dict] = [] # list of order/trade details
        self.auto_trade_enabled = False
        
    def get_portfolio_state(self, current_prices: dict) -> dict:
        unrealized_pnl = 0.0
        position_value = 0.0
        active_positions = []
        
        for sym, pos in list(self.positions.items()):
            mark_price = current_prices.get(sym.lower(), pos["entry_price"])
            # Futures PnL (Long vs Short)
            if pos["side"] == "BUY":
                pnl = pos["qty"] * (mark_price - pos["entry_price"])
            else: # SELL (Short)
                pnl = pos["qty"] * (pos["entry_price"] - mark_price)
                
            unrealized_pnl += pnl
            pos_val = pos["qty"] * mark_price
            position_value += pos_val
            
            active_positions.append({
                "symbol": sym.upper(),
                "side": pos["side"],
                "qty": pos["qty"],
                "entry_price": pos["entry_price"],
                "mark_price": mark_price,
                "pnl": pnl,
                "pnl_pct": (pnl / (pos["qty"] * pos["entry_price"])) * 100 if pos["entry_price"] > 0 else 0
            })
            
        equity = self.cash + unrealized_pnl
        
        return {
            "cash": self.cash,
            "equity": equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": self.realized_pnl + unrealized_pnl,
            "total_pnl_pct": ((equity - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            "positions": active_positions,
            "trades": self.trades[-20:], # Return last 20 trades
            "auto_trade_enabled": self.auto_trade_enabled
        }
        
    def execute_order(self, symbol: str, side: str, qty: float, price: float) -> bool:
        symbol = symbol.lower()
        if qty <= 0 or price <= 0:
            return False
            
        notional = qty * price
        
        # Check if closing/reducing existing position
        pos = self.positions.get(symbol)
        if pos:
            if pos["side"] != side:
                # Opposite side -> reduction or close
                if qty >= pos["qty"]:
                    # Close entire position
                    realized = pos["qty"] * (price - pos["entry_price"]) if pos["side"] == "BUY" else pos["qty"] * (pos["entry_price"] - price)
                    self.realized_pnl += realized
                    # Return margin (notional / 5.0) + realized PnL to cash balance
                    margin_held = (pos["qty"] * pos["entry_price"]) / 5.0
                    self.cash += margin_held + realized
                    
                    self.trades.append({
                        "timestamp": time.time(),
                        "symbol": symbol.upper(),
                        "side": side,
                        "qty": pos["qty"],
                        "price": price,
                        "notional": pos["qty"] * price,
                        "realized_pnl": realized,
                        "type": "CLOSE"
                    })
                    
                    # If there's leftover quantity, open new opposite position
                    leftover_qty = qty - pos["qty"]
                    if leftover_qty > 0:
                        leftover_margin = (leftover_qty * price) / 5.0
                        if self.cash >= leftover_margin:
                            self.positions[symbol] = {
                                "qty": leftover_qty,
                                "entry_price": price,
                                "side": side
                            }
                            self.cash -= leftover_margin
                            self.trades.append({
                                "timestamp": time.time(),
                                "symbol": symbol.upper(),
                                "side": side,
                                "qty": leftover_qty,
                                "price": price,
                                "notional": leftover_qty * price,
                                "realized_pnl": 0.0,
                                "type": "OPEN"
                            })
                        else:
                            # Not enough cash for leftover order quantity
                            self.positions.pop(symbol)
                    else:
                        self.positions.pop(symbol)
                else:
                    # Partial close
                    realized = qty * (price - pos["entry_price"]) if pos["side"] == "BUY" else qty * (pos["entry_price"] - price)
                    self.realized_pnl += realized
                    margin_released = (qty * pos["entry_price"]) / 5.0
                    self.cash += margin_released + realized
                    pos["qty"] -= qty
                    
                    self.trades.append({
                        "timestamp": time.time(),
                        "symbol": symbol.upper(),
                        "side": side,
                        "qty": qty,
                        "price": price,
                        "notional": qty * price,
                        "realized_pnl": realized,
                        "type": "PARTIAL_CLOSE"
                    })
                return True
                
        # Open new position (with 5x leverage)
        margin_required = notional / 5.0
        if self.cash < margin_required:
            logger.warning(f"Insufficient cash for order: required {margin_required}, cash {self.cash}")
            return False
            
        self.positions[symbol] = {
            "qty": qty,
            "entry_price": price,
            "side": side
        }
        self.cash -= margin_required
        
        self.trades.append({
            "timestamp": time.time(),
            "symbol": symbol.upper(),
            "side": side,
            "qty": qty,
            "price": price,
            "notional": notional,
            "realized_pnl": 0.0,
            "type": "OPEN"
        })
        return True

    def reset(self):
        self.cash = self.initial_balance
        self.realized_pnl = 0.0
        self.positions.clear()
        self.trades.clear()
