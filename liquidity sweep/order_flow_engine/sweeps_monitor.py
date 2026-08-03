import os
import asyncio
import csv
import logging
from typing import Callable, Awaitable
from datetime import datetime, timezone
from config import SWEEPS_CSV_PATH

logger = logging.getLogger("OrderFlow.SweepsMonitor")

class SweepsMonitor:
    """
    Monitors sweeps.csv under live_scanner/ in real-time.
    Invokes callback when a new sweep alert is appended.
    Supports startup replay of active recent sweeps.
    """
    def __init__(self, callback: Callable[[dict], Awaitable[None]]):
        self.callback = callback
        self.csv_path = SWEEPS_CSV_PATH
        self.last_position = 0
        self.is_running = False
        self._task = None

    async def start(self):
        self.is_running = True
        self._ensure_directory_exists()
        
        # 1. Perform startup replay of active sweeps
        await self._replay_recent_sweeps()
        
        # 2. Initialize watchdog position at current file size
        if os.path.exists(self.csv_path):
            self.last_position = os.path.getsize(self.csv_path)
            logger.info(f"SweepsMonitor initialized watcher position at: {self.last_position} bytes")
        else:
            self.last_position = 0
            logger.info(f"SweepsMonitor waiting for sweeps file to be created at {self.csv_path}")
            
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _ensure_directory_exists(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    async def _replay_recent_sweeps(self):
        """
        Replays active, unresolved live sweeps from sweeps.csv within the last 12 hours on startup.
        """
        if not os.path.exists(self.csv_path):
            logger.info("No sweeps.csv found for startup replay.")
            return

        replayed_count = 0
        seen_sweep_ids = set()
        active_states = {"RAW_SWEEP", "MSS_PENDING", "VALID_ENTRY_READY"}
        now_time = datetime.now(timezone.utc)

        try:
            with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                # If header doesn't exist, reset read position to start
                if not header or header[0] != "timestamp":
                    f.seek(0)
                    reader = csv.reader(f)

                for row in reader:
                    if not row or len(row) < 4:
                        continue
                        
                    # Handle V2 schema index positioning vs V1 fallback
                    if len(row) >= 15:
                        timestamp_str = row[0]
                        symbol = row[1]
                        direction = row[2]
                        sweep_level_str = row[3]
                        state = row[7]
                        sweep_id = row[13]
                        source = row[14]
                    else:
                        # Fallback parsing
                        timestamp_str = row[0]
                        symbol = row[1]
                        direction = row[2]
                        sweep_level_str = row[3]
                        state = row[7] if len(row) > 7 else "RAW_SWEEP"
                        sweep_id = row[4] if len(row) > 4 else f"swp_{row[0]}_{row[1]}_{row[2]}_{row[3]}"
                        source = "live"

                    # Only replay 'live' source and active price states
                    if source.lower() != "live" or state not in active_states:
                        continue

                    # Skip duplicate sweep IDs
                    if sweep_id in seen_sweep_ids:
                        continue

                    # Filter: Only sweeps in the last 12 hours
                    try:
                        # Parse UTC ISO string timestamp: e.g. 2026-07-01T07:57:39Z
                        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        age_hours = (now_time - dt).total_seconds() / 3600.0
                        if age_hours > 12.0:
                            continue
                    except Exception:
                        continue

                    # Valid sweep matching replay parameters
                    try:
                        sweep = {
                            "timestamp": timestamp_str,
                            "symbol": symbol,
                            "type": direction, # Callback expects 'type' as the direction field
                            "sweep_level": float(sweep_level_str),
                            "sweep_id": sweep_id
                        }
                        
                        logger.info(f"Replaying active sweep on startup: {sweep_id} ({state})")
                        await self.callback(sweep)
                        seen_sweep_ids.add(sweep_id)
                        replayed_count += 1
                    except Exception as re_err:
                        logger.error(f"Failed to process replayed sweep: {row}. Error: {re_err}")

            logger.info(f"Replayed {replayed_count} recent sweeps from sweeps.csv")

        except Exception as e:
            logger.error(f"Error during SweepsMonitor startup replay: {e}", exc_info=True)

    async def _monitor_loop(self):
        while self.is_running:
            try:
                if not os.path.exists(self.csv_path):
                    await asyncio.sleep(1.0)
                    continue

                current_size = os.path.getsize(self.csv_path)
                if current_size < self.last_position:
                    # File was truncated or recreated
                    logger.info("Sweeps file truncated/recreated. Resetting monitor position.")
                    self.last_position = 0

                if current_size > self.last_position:
                    with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
                        f.seek(self.last_position)
                        reader = csv.reader(f)
                        for row in reader:
                            if not row or len(row) < 4:
                                continue
                            if row[0] == "timestamp":
                                continue
                            try:
                                # Determine columns dynamically or use indices (index 13 in V2)
                                if len(row) >= 15:
                                    sweep_id = row[13]
                                    direction = row[2]
                                    sweep_level = float(row[3])
                                else:
                                    sweep_id = row[4] if len(row) > 4 else f"swp_{row[0]}_{row[1]}_{row[2]}_{row[3]}"
                                    direction = row[2]
                                    sweep_level = float(row[3])

                                sweep = {
                                    "timestamp": row[0],
                                    "symbol": row[1],
                                    "type": direction,
                                    "sweep_level": sweep_level,
                                    "sweep_id": sweep_id
                                }
                                logger.info(f"Detected new sweep alert from CSV: {sweep}")
                                await self.callback(sweep)
                            except Exception as pe:
                                logger.error(f"Failed to parse CSV sweep row: {row}. Error: {pe}")
                        self.last_position = f.tell()

            except Exception as e:
                logger.error(f"Error in SweepsMonitor loop: {e}", exc_info=True)
                
            await asyncio.sleep(1.0)
