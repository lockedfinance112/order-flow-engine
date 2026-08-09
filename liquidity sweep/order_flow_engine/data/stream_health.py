import time
from typing import Dict, Any

class StreamHealthTracker:
    """
    Tracks real-time health measurements for a WebSocket stream per symbol.
    Provides statuses: HEALTHY, DEGRADED, STALE, INVALID.
    """
    def __init__(self, symbol: str, healthy_threshold_ms: float = 1000.0, degraded_threshold_ms: float = 2500.0):
        self.symbol = symbol.lower()
        self.healthy_threshold_ms = healthy_threshold_ms
        self.degraded_threshold_ms = degraded_threshold_ms

        # Metrics
        self.last_event_age_ms = 0.0
        self.exchange_to_receive_latency_ms = 0.0
        self.receive_to_process_latency_ms = 0.0
        self.previous_update_id = None
        self.current_update_id = None
        
        self.sequence_gap_count = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self.reconnect_count = 0
        self.resync_count = 0
        self.queue_depth = 0
        
        # Event throughput
        self.event_timestamps = []
        
    def record_event(self, event_time_ms: float, tx_time_ms: float, received_time_ms: float, processed_time_ms: float, update_id: int):
        self.previous_update_id = self.current_update_id
        self.current_update_id = update_id

        # Calculate ages and latencies
        self.last_event_age_ms = max(0.0, (processed_time_ms - event_time_ms) * 1000.0)
        self.exchange_to_receive_latency_ms = max(0.0, (received_time_ms - tx_time_ms) * 1000.0)
        self.receive_to_process_latency_ms = max(0.0, (processed_time_ms - received_time_ms) * 1000.0)

        # Update throughput
        now = time.time()
        self.event_timestamps.append(now)
        # Keep only last 10 seconds of timestamps for rolling events-per-second
        self.event_timestamps = [t for t in self.event_timestamps if now - t <= 10.0]

    def get_events_per_second(self) -> float:
        if not self.event_timestamps:
            return 0.0
        span = 10.0
        return len(self.event_timestamps) / span

    def record_reconnect(self):
        self.reconnect_count += 1

    def record_resync(self):
        self.resync_count += 1

    def record_sequence_gap(self):
        self.sequence_gap_count += 1

    def record_duplicate(self):
        self.duplicate_count += 1

    def record_out_of_order(self):
        self.out_of_order_count += 1

    def update_queue_depth(self, depth: int):
        self.queue_depth = depth

    def get_status(self, is_book_valid: bool = True) -> str:
        """
        HEALTHY: age <= 1000 ms
        DEGRADED: age 1000 - 2500 ms
        STALE: age > 2500 ms
        INVALID: sequence gap or unsynchronised book
        """
        if not is_book_valid:
            return "INVALID"
        
        # Check staleness if no events received for too long
        if not self.event_timestamps:
            return "STALE"
            
        age = self.last_event_age_ms
        if age <= self.healthy_threshold_ms:
            return "HEALTHY"
        elif age <= self.degraded_threshold_ms:
            return "DEGRADED"
        else:
            return "STALE"

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "last_event_age_ms": self.last_event_age_ms,
            "exchange_to_receive_latency_ms": self.exchange_to_receive_latency_ms,
            "receive_to_process_latency_ms": self.receive_to_process_latency_ms,
            "previous_update_id": self.previous_update_id,
            "current_update_id": self.current_update_id,
            "sequence_gap_count": self.sequence_gap_count,
            "duplicate_count": self.duplicate_count,
            "out_of_order_count": self.out_of_order_count,
            "reconnect_count": self.reconnect_count,
            "resync_count": self.resync_count,
            "queue_depth": self.queue_depth,
            "events_per_second": self.get_events_per_second()
        }
