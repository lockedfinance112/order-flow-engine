import asyncio
import logging
import time
import urllib.request
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from data.sequence_validator import SequenceValidator
from data.stream_health import StreamHealthTracker

logger = logging.getLogger("OrderFlow.LocalOrderBook")

class SyncOutcome(Enum):
    SYNCED = "SYNCED"
    WAITING_FOR_BUFFER = "WAITING_FOR_BUFFER"
    RETRY_SNAPSHOT = "RETRY_SNAPSHOT"
    INVALID_SEQUENCE = "INVALID_SEQUENCE"

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

    def set_state(self, new_state: str, details: str = ""):
        if self.state != new_state:
            logger.info(f"DATA_GUARDIAN {self.symbol.upper()} BOOK {self.state} -> {new_state} {details}".strip())
            self.state = new_state

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
                self.set_state("SYNCING")
                # Trigger background sync
                self.trigger_sync()
            
            self.buffer.append(msg)
            
            # If snapshot is loaded, check if we can process the buffer now
            if self.state == "SYNCING" and self.validator.last_u is not None and self.validator.last_u > 0:
                self._try_sync_from_buffer()
                
            # Limit buffer size to prevent memory leaks and handle overflow
            import config
            max_buf = getattr(config, "LOCAL_BOOK_MAX_BUFFER", 5000)
            if len(self.buffer) > max_buf:
                logger.error(f"[{self.symbol.upper()}] Depth buffer overflow ({len(self.buffer)} > {max_buf}). Invalidate current synchronization attempt.")
                self.buffer.clear()
                self.is_valid = False
                self.validator.reset()
                self.bids.clear()
                self.asks.clear()
                self.set_state("RESYNCING")
                self.trigger_sync(force=True)
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
                self.set_state("SEQUENCE_GAP")
                self.is_valid = False
                self.bids.clear()
                self.asks.clear()
                self.validator.reset()
                self.buffer.clear()
                self.set_state("RESYNCING")
                self.trigger_sync(force=True)

    def _fetch_snapshot_sync(self) -> dict:
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=1000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            try:
                return json.loads(response.read().decode())
            except StopIteration as exc:
                raise RuntimeError("Snapshot provider unexpectedly exhausted") from exc

    async def _sleep(self, seconds: float):
        await asyncio.sleep(seconds)

    def trigger_sync(self, force: bool = False):
        """Triggers the async REST depth snapshot fetcher."""
        if not force and self.sync_task and not self.sync_task.done():
            return
        if force and self.sync_task and not self.sync_task.done():
            self.sync_task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # We are outside a running loop (like in synchronous unit tests). Sync runs mock-only.
            return
        self.sync_task = loop.create_task(self._fetch_and_apply_snapshot())

    async def _fetch_and_apply_snapshot(self):
        self.health_tracker.record_resync()
        backoff = 1.0
        while self.state in ("RESYNCING", "SYNCING", "INITIALISING"):
            try:
                snapshot = await asyncio.to_thread(self._fetch_snapshot_sync)
                outcome = self._apply_snapshot(snapshot)
                
                if outcome == SyncOutcome.SYNCED:
                    return
                elif outcome == SyncOutcome.WAITING_FOR_BUFFER:
                    while self.state == "SYNCING":
                        await self._sleep(0.1)
                        outcome = self._try_sync_from_buffer()
                        if outcome == SyncOutcome.SYNCED:
                            return
                        elif outcome in (SyncOutcome.RETRY_SNAPSHOT, SyncOutcome.INVALID_SEQUENCE):
                            break
                    if self.state == "HEALTHY":
                        return
                    await self._sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                elif outcome == SyncOutcome.RETRY_SNAPSHOT:
                    await self._sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
            except Exception as e:
                logger.error(f"[{self.symbol.upper()}] Failed to fetch order book snapshot: {e}. Retrying in {backoff}s...")
                await self._sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _apply_snapshot(self, snapshot: dict) -> SyncOutcome:
        """Applies the REST snapshot and attempts to play forward the buffer."""
        last_update_id = int(snapshot["lastUpdateId"])
        self.bids = {float(price): float(qty) for price, qty in snapshot["bids"]}
        self.asks = {float(price): float(qty) for price, qty in snapshot["asks"]}
        
        self.validator.reset()
        self.validator.last_u = last_update_id
        self.set_state("SYNCING")
        return self._try_sync_from_buffer()

    def _try_sync_from_buffer(self) -> SyncOutcome:
        last_update_id = self.validator.last_u
        if last_update_id is None or last_update_id == 0:
            return SyncOutcome.RETRY_SNAPSHOT

        valid_updates = [msg for msg in self.buffer if int(msg["u"]) > last_update_id]
        if not valid_updates:
            return SyncOutcome.WAITING_FOR_BUFFER

        # Check if the oldest valid update is already past the snapshot's range (gap)
        oldest_msg = valid_updates[0]
        if int(oldest_msg["U"]) > last_update_id + 1:
            logger.warning(f"[{self.symbol.upper()}] Gap detected: oldest buffered U {oldest_msg['U']} > lastUpdateId + 1 ({last_update_id + 1}). Retrying sync.")
            self.buffer.clear()
            self.validator.last_u = 0
            return SyncOutcome.RETRY_SNAPSHOT

        # Find the first update that overlaps with the snapshot: U <= lastUpdateId + 1 and u >= lastUpdateId + 1
        first_idx = -1
        for idx, msg in enumerate(valid_updates):
            U = int(msg["U"])
            u = int(msg["u"])
            if U <= last_update_id + 1 <= u:
                first_idx = idx
                break
                
        if first_idx == -1:
            return SyncOutcome.WAITING_FOR_BUFFER

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
                    return SyncOutcome.INVALID_SEQUENCE

        self.buffer.clear()
        self.is_valid = True
        self.set_state("HEALTHY", f"update_id={self.last_update_id}")
        return SyncOutcome.SYNCED

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
