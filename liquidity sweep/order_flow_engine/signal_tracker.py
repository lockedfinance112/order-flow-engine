import os
import csv
import time
import uuid
from datetime import datetime, timezone
import logging

logger = logging.getLogger("OrderFlow.SignalTracker")

class SignalTracker:
    """
    Tracks state transitions and monitors forward price outcomes for entry signals.
    Saves results to separate CSV files: bias_signals.csv and bias_transitions.csv.
    """
    def __init__(self, signals_csv="bias_signals.csv", transitions_csv="bias_transitions.csv"):
        self.signals_csv = os.path.join(os.path.dirname(__file__), signals_csv)
        self.transitions_csv = os.path.join(os.path.dirname(__file__), transitions_csv)
        
        # active_tracks: list of active signal tracking dicts
        self.active_tracks = []
        # last_signal_time: {symbol: last_register_timestamp} for de-duplication/cooldown
        self.last_signal_time = {}
        
        self._ensure_headers()

    def _ensure_headers(self):
        """Creates CSV log files with headers if they do not exist."""
        if not os.path.exists(self.signals_csv):
            headers = [
                "signal_id", "symbol", "action", "direction", "entry_time", "entry_price",
                "entry_delta_5m", "entry_imbalance", "entry_aggression", "entry_latest_event",
                "entry_cvd", "price_after_1m", "price_after_5m", "price_after_15m",
                "max_favorable_pct", "max_adverse_pct", "final_return_pct", "completed_time"
            ]
            with open(self.signals_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        if os.path.exists(self.transitions_csv):
            try:
                with open(self.transitions_csv, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if "suppression_reason" not in first_line:
                    os.remove(self.transitions_csv)
            except Exception:
                pass

        if not os.path.exists(self.transitions_csv):
            headers = ["timestamp", "symbol", "old_action", "new_action", "price", "latest_event", "suppression_reason", "reason"]
            with open(self.transitions_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def register_signal_change(self, symbol: str, old_action: str, new_action: str, price: float,
                               metrics_5m: dict, imbalance: float, latest_event: str | None, cvd: float,
                               suppression_reason: str = "NONE"):
        """Registers a transition and starts a 15-minute tracking window if it is a new entry signal."""
        now = time.time()
        timestamp_str = datetime.now(timezone.utc).isoformat()
        
        # 1. Log transition immediately to bias_transitions.csv
        reason = f"Dashboard change from {old_action} to {new_action}"
        try:
            with open(self.transitions_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, symbol.upper(), old_action, new_action, price, latest_event or "", suppression_reason, reason])
        except Exception as e:
            logger.error(f"Failed to log transition: {e}")

        # 2. Check if this is an entry signal to track
        target_signals = {"LONG_BIAS", "SHORT_BIAS", "LONG (SWEEP)", "SHORT (SWEEP)"}
        if new_action in target_signals:
            # Apply de-duplication: check if direction is unchanged or we are in cooldown (300 seconds)
            last_time = self.last_signal_time.get(symbol.lower(), 0.0)
            if now - last_time >= 300.0:
                direction = "LONG" if "LONG" in new_action else "SHORT"
                signal_id = str(uuid.uuid4())
                
                track = {
                    "signal_id": signal_id,
                    "symbol": symbol.upper(),
                    "action": new_action,
                    "direction": direction,
                    "entry_time": now,
                    "entry_price": price,
                    "entry_delta_5m": metrics_5m.get("delta_usdt", 0.0),
                    "entry_imbalance": imbalance,
                    "entry_aggression": metrics_5m.get("buy_ratio", 0.5),
                    "entry_latest_event": latest_event or "",
                    "entry_cvd": cvd,
                    "price_after_1m": None,
                    "price_after_5m": None,
                    "price_after_15m": None,
                    "max_favorable_pct": 0.0,
                    "max_adverse_pct": 0.0,
                    "final_return_pct": 0.0
                }
                
                self.active_tracks.append(track)
                self.last_signal_time[symbol.lower()] = now
                logger.info(f"Registered signal tracking: {new_action} for {symbol.upper()} @ {price}")

    def update_price(self, symbol: str, price: float):
        """Feeds incoming prices to update running stats for active tracks."""
        now = time.time()
        symbol_upper = symbol.upper()
        
        for track in self.active_tracks:
            if track["symbol"] == symbol_upper:
                entry_price = track["entry_price"]
                direction = track["direction"]
                
                # Excursion percentage calculations
                if direction == "LONG":
                    favorable_pct = (price - entry_price) / entry_price
                    adverse_pct = (entry_price - price) / entry_price
                else:  # SHORT
                    favorable_pct = (entry_price - price) / entry_price
                    adverse_pct = (price - entry_price) / entry_price

                # Store excursions as absolute positive percentages (max(0.0, pct))
                track["max_favorable_pct"] = max(track["max_favorable_pct"], max(0.0, favorable_pct))
                track["max_adverse_pct"] = max(track["max_adverse_pct"], max(0.0, adverse_pct))

                # Window updates based on elapsed time since entry
                elapsed = now - track["entry_time"]
                if elapsed >= 60.0 and track["price_after_1m"] is None:
                    track["price_after_1m"] = price
                if elapsed >= 300.0 and track["price_after_5m"] is None:
                    track["price_after_5m"] = price
                if elapsed >= 900.0 and track["price_after_15m"] is None:
                    track["price_after_15m"] = price

    def finalize_expired_signals(self):
        """Finalizes and logs tracking instances older than 15 minutes."""
        now = time.time()
        remaining_tracks = []
        
        for track in self.active_tracks:
            elapsed = now - track["entry_time"]
            if elapsed >= 900.0:  # 15 minutes
                # Use exit price or fallback to entry if no updates happened
                if track["price_after_15m"] is None:
                    track["price_after_15m"] = track["entry_price"]
                
                entry_price = track["entry_price"]
                exit_price = track["price_after_15m"]
                
                if track["direction"] == "LONG":
                    track["final_return_pct"] = (exit_price - entry_price) / entry_price
                else:
                    track["final_return_pct"] = (entry_price - exit_price) / entry_price
                
                # Append completed track to bias_signals.csv
                try:
                    completed_time_str = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
                    row = [
                        track["signal_id"], track["symbol"], track["action"], track["direction"],
                        datetime.fromtimestamp(track["entry_time"], tz=timezone.utc).isoformat(),
                        track["entry_price"], track["entry_delta_5m"], track["entry_imbalance"],
                        track["entry_aggression"], track["entry_latest_event"], track["entry_cvd"],
                        track["price_after_1m"], track["price_after_5m"], track["price_after_15m"],
                        track["max_favorable_pct"], track["max_adverse_pct"], track["final_return_pct"],
                        completed_time_str
                    ]
                    with open(self.signals_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    logger.info(f"Signal outcome logged: {track['symbol']} {track['action']} return: {track['final_return_pct']*100:.2f}%")
                except Exception as e:
                    logger.error(f"Failed to log finalized signal outcome: {e}")
            else:
                remaining_tracks.append(track)
                
        self.active_tracks = remaining_tracks
