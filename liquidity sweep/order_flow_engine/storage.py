import csv
import os
import logging
from datetime import datetime, timezone
from typing import Optional
from config import CSV_PATH

logger = logging.getLogger("OrderFlow.Storage")

class EventStorage:
    """
    Logs scanner/order flow events to a CSV file.
    Creates the file and writes the header if it does not exist.
    """
    def __init__(self):
        self.csv_path = CSV_PATH
        self._ensure_csv_headers()

    def _ensure_csv_headers(self):
        """Creates the CSV file with headers if it doesn't exist."""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            if not os.path.exists(self.csv_path):
                logger.info(f"Creating CSV file for events at {self.csv_path}")
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "symbol", "window", "event_type", 
                        "buy_volume", "sell_volume", "delta", "cvd", 
                        "buy_ratio", "sell_ratio", "price", 
                        "sweep_id", "source_sweep_time", "confluence_score", 
                        "matched_conditions", "order_book_imbalance_at_score", 
                        "depth_snapshot_age_ms", "notes"
                    ])
        except Exception as e:
            logger.error(f"Failed to initialize CSV storage: {e}")

    def log_event(self, event_type: str, window: str, metrics: dict, notes: str = "",
                  sweep_id: str = "", source_sweep_time: str = "",
                  confluence_score: Optional[int] = None, matched_conditions: str = "",
                  order_book_imbalance_at_score: Optional[float] = None,
                  depth_snapshot_age_ms: Optional[float] = None):
        """
        Appends an event log entry to the CSV file.
        """
        try:
            timestamp_str = datetime.now(timezone.utc).isoformat()
            
            buy_vol = metrics.get("buy_volume", 0.0)
            sell_vol = metrics.get("sell_volume", 0.0)
            delta = metrics.get("delta", 0.0)
            cvd = metrics.get("cvd", 0.0)
            buy_ratio = metrics.get("buy_ratio", 0.5)
            sell_ratio = metrics.get("sell_ratio", 0.5)
            price = metrics.get("latest_price", 0.0)
            symbol = metrics.get("symbol", "btcusdt")

            score_str = str(confluence_score) if confluence_score is not None else ""
            imbalance_str = f"{order_book_imbalance_at_score:.4f}" if order_book_imbalance_at_score is not None else ""
            age_str = f"{depth_snapshot_age_ms:.1f}" if depth_snapshot_age_ms is not None else ""

            row = [
                timestamp_str,
                symbol,
                window,
                event_type,
                f"{buy_vol:.4f}",
                f"{sell_vol:.4f}",
                f"{delta:.4f}",
                f"{cvd:.4f}",
                f"{buy_ratio:.4f}",
                f"{sell_ratio:.4f}",
                f"{price:.2f}",
                sweep_id,
                source_sweep_time,
                score_str,
                matched_conditions,
                imbalance_str,
                age_str,
                notes
            ]

            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            logger.info(f"Event logged: {event_type} | Window: {window} | Delta: {delta:.2f} | Price: {price:.2f}")
        except Exception as e:
            logger.error(f"Failed to log event to CSV: {e}")
