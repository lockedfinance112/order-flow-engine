import asyncio
import logging
import time
import urllib.request
import json
from typing import Dict, List, Tuple, Optional, Any
from data.sequence_validator import SequenceValidator
from data.stream_health import StreamHealthTracker

logger = logging.getLogger("OrderFlow.LocalOrderBook")

class LocalOrderBook:
    """
    Manages a local order book copy synchronized with Binance via REST snapshot and diff depth updates.
    Validates event sequencing and updates stream health metrics.
    """
    def __init__(self, symbol: str, health_tracker: StreamHealthTracker):
        self.symbol = symbol.lower()
        self.health_tracker = health_tracker
        self.state = "INITIALISING"  # INITIALISING, SYNCING, HEALTHY, SEQUENCE_GAP, STALE, RESYNCING, DISCONNECTED
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        
        self.validator = SequenceValidator()
        self.buffer: List[dict] = []
        self.is_valid = False
        
        # Last updated book metadata
        self.last_update_id = 0
        self.last_event_time = 0.0
        self.last_transaction_time = 0.0
        self.last_received_time = 0.0
        
        self.sync_task: Optional[asyncio.Task] = None

    def handle_ws_update(self, msg: dict, received_time: float):
        """
        Processes an incoming WebSocket diff depth update.
        Buffers during initialization/sync, validates sequences, and updates levels.
        """
        self.last_received_time = received_time
        processed_time = time.time()
        
        U = int(msg["U"])
        u = int(msg["u"])
        pu = int(msg["pu"])
        event_time = float(msg["E"]) / 1000.0
        tx_time = float(msg["T"]) / 1000.0
        
        # Record health metrics
        self.health_tracker.record_event(
            event_time_ms=event_time,
            tx_time_ms=tx_time,
            received_time_ms=received_time,
            processed_time_ms=processed_time,
            update_id=u
        )

        if self.state in ("INITIALISING", "SYNCING", "RESYNCING"):
            if self.state == "INITIALISING":
                self.state = "SYNCING"
                # Trigger background sync
                self.trigger_sync()
            
            self.buffer.append(msg)
            
            # If snapshot is loaded, check if we can process the buffer now
            if self.state == "SYNCING" and self.validator.last_u is not None and self.validator.last_u > 0:
                self._try_sync_from_buffer()
                
            # Limit buffer size to prevent memory leaks
            if len(self.buffer) > 2000:
                self.buffer.pop(0)
            return

        if self.state == "HEALTHY":
            res = self.validator.validate_and_update(U, u, pu)
            if res == "OK":
                self._apply_diff_update(msg)
                self.last_update_id = u
                self.last_event_time = event_time
                self.last_transaction_time = tx_time
            elif res == "DUPLICATE":
                self.health_tracker.record_duplicate()
            elif res == "OUT_OF_ORDER":
                self.health_tracker.record_out_of_order()
            elif res == "GAP":
                logger.warning(f"[{self.symbol.upper()}] Sequence gap detected: pu={pu}, last_u={self.validator.last_u}. Resynchronising.")
                self.health_tracker.record_sequence_gap()
                self.state = "SEQUENCE_GAP"
                self.is_valid = False
                self.bids.clear()
                self.asks.clear()
                self.validator.reset()
                self.buffer.clear()
                self.state = "RESYNCING"
                self.trigger_sync()

    def trigger_sync(self):
        """Triggers the async REST depth snapshot fetcher."""
        if self.sync_task and not self.sync_task.done():
            return
        try:
            self.sync_task = asyncio.create_task(self._fetch_and_apply_snapshot())
        except RuntimeError:
            # We are outside a running loop (like in synchronous unit tests). Sync runs mock-only.
            pass

    async def _fetch_and_apply_snapshot(self):
        self.health_tracker.record_resync()
        backoff = 1.0
        while True:
            try:
                url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=1000"
                def _fetch():
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return json.loads(response.read().decode())
                
                snapshot = await asyncio.to_thread(_fetch)
                self._apply_snapshot(snapshot)
                break
            except Exception as e:
                logger.error(f"[{self.symbol.upper()}] Failed to fetch order book snapshot: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _apply_snapshot(self, snapshot: dict):
        """Applies the REST snapshot and attempts to play forward the buffer."""
        last_update_id = int(snapshot["lastUpdateId"])
        self.bids = {float(price): float(qty) for price, qty in snapshot["bids"]}
        self.asks = {float(price): float(qty) for price, qty in snapshot["asks"]}
        
        self.validator.reset()
        self.validator.last_u = last_update_id
        self.state = "SYNCING"
        self._try_sync_from_buffer()

    def _try_sync_from_buffer(self):
        last_update_id = self.validator.last_u
        valid_updates = [msg for msg in self.buffer if int(msg["u"]) > last_update_id]
        if not valid_updates:
            return

        # Check if the oldest valid update is already past the snapshot's range (gap)
        oldest_msg = valid_updates[0]
        if int(oldest_msg["U"]) > last_update_id + 1:
            logger.warning(f"[{self.symbol.upper()}] Gap detected: oldest buffered U {oldest_msg['U']} > lastUpdateId + 1 ({last_update_id + 1}). Retrying sync.")
            self.buffer.clear()
            self.validator.last_u = 0
            self.trigger_sync()
            return

        # Find the first update that overlaps with the snapshot: U <= lastUpdateId + 1 and u >= lastUpdateId + 1
        first_idx = -1
        for idx, msg in enumerate(valid_updates):
            U = int(msg["U"])
            u = int(msg["u"])
            if U <= last_update_id + 1 <= u:
                first_idx = idx
                break
                
        if first_idx == -1:
            return

        self.validator.reset()
        
        # Play forward updates from the first overlapping update onwards
        is_first = True
        for msg in valid_updates[first_idx:]:
            U = int(msg["U"])
            u = int(msg["u"])
            pu = int(msg["pu"])
            if is_first:
                self.validator.last_u = u
                self._apply_diff_update(msg)
                self.last_update_id = u
                is_first = False
            else:
                res = self.validator.validate_and_update(U, u, pu)
                if res in ("OK", "DUPLICATE"):
                    self._apply_diff_update(msg)
                    self.last_update_id = u
                else:
                    logger.warning(f"[{self.symbol.upper()}] Sync validation failed: {res} for u={u}, pu={pu}. Resetting sync.")
                    self.buffer.clear()
                    self.validator.last_u = 0
                    self.trigger_sync()
                    return

        self.buffer.clear()
        self.is_valid = True
        self.state = "HEALTHY"
        logger.info(f"[{self.symbol.upper()}] Local order book successfully synchronized at update ID {self.last_update_id}.")

    def _apply_diff_update(self, msg: dict):
        """Applies a single diff depth update payload to local bids/asks dicts."""
        for price_str, qty_str in msg["b"]:
            p, q = float(price_str), float(qty_str)
            if q == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        for price_str, qty_str in msg["a"]:
            p, q = float(price_str), float(qty_str)
            if q == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

    def get_top_bids_asks(self, levels: int = 5) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Returns sorted top N bids and asks as [(price, qty), ...]."""
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]
        return sorted_bids, sorted_asks
