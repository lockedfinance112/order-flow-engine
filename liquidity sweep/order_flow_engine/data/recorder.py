import os
import json
import gzip
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger("OrderFlow.Recorder")

class StreamRecorder:
    """
    Asynchronously records raw WebSocket events to gzip-compressed JSONLines files.
    """
    def __init__(self, symbol: str, output_dir: str = "recordings", flush_interval_secs: float = 5.0):
        self.symbol = symbol.lower()
        self.output_dir = output_dir
        self.flush_interval_secs = flush_interval_secs
        self.queue: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()
        self.is_running = False
        self._flush_task: Optional[asyncio.Task] = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start(self):
        """Starts the periodic background flush task."""
        if self.is_running:
            return
        self.is_running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"[{self.symbol.upper()}] Stream recorder started.")

    async def record(self, stream: str, raw_msg: dict, receive_time: float):
        """Enqueues an event for recording."""
        if not self.is_running:
            return

        # Extract exchange time if available
        exchange_time = None
        if isinstance(raw_msg, dict):
            # 'T' for aggTrade, 'E' for depthUpdate/events
            exchange_time = raw_msg.get("T") or raw_msg.get("E")
            if exchange_time:
                exchange_time = float(exchange_time) / 1000.0

        event = {
            "stream": stream,
            "data": raw_msg,
            "receive_time": receive_time,
            "exchange_time": exchange_time
        }
        
        async with self.lock:
            self.queue.append(event)

    async def stop(self):
        """Stops the recorder and flushes any remaining items."""
        if not self.is_running:
            return
        self.is_running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()
        logger.info(f"[{self.symbol.upper()}] Stream recorder stopped.")

    async def _periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval_secs)
            await self._flush()

    async def _flush(self):
        async with self.lock:
            if not self.queue:
                return
            events_to_write = list(self.queue)
            self.queue.clear()

        # Date stamp for filename
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = os.path.join(self.output_dir, f"recording_{self.symbol}_{date_str}.jsonl.gz")

        # Write inside a thread to avoid blocking event loop
        def _write():
            try:
                # Open with 'ab' append binary mode
                with gzip.open(filepath, "ab", compresslevel=6) as f:
                    for ev in events_to_write:
                        line = json.dumps(ev) + "\n"
                        f.write(line.encode("utf-8"))
            except Exception as e:
                logger.error(f"[{self.symbol.upper()}] Failed to write recording to file: {e}")

        await asyncio.to_thread(_write)
