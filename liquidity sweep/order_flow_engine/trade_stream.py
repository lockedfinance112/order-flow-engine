import asyncio
import json
import logging
import traceback
import ssl
import time
import websockets
from typing import Callable, Awaitable
from config import WS_TRADE_URL

logger = logging.getLogger("OrderFlow.TradeStream")

class TradeStream:
    """
    Connects to the Binance USD-M futures aggTrade combined stream for multiple symbols.
    Extracts symbol, routes trade payloads, and tracks stream connection health.
    """
    def __init__(self, callback: Callable[[str, dict], Awaitable[None]]):
        self.callback = callback
        self.is_running = False
        self.status = "DISCONNECTED"
        self.last_message_time = 0.0
        self._task = None

    def get_status(self) -> str:
        """Returns the current connection health status, accounting for stale data."""
        if self.status == "CONNECTED":
            if self.last_message_time > 0 and (time.time() - self.last_message_time) > 5.0:
                return "STALE"
        return self.status

    async def start(self):
        self.is_running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.is_running = False
        self.status = "DISCONNECTED"
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self):
        backoff = 1.0
        max_backoff = 60.0

        # Create unverified SSL context to bypass cert verification errors
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        from config import trade_streams
        base_url = "wss://fstream.binance.com/market/stream"

        while self.is_running:
            try:
                self.status = "RECONNECTING"
                logger.info(f"Connecting to trade websocket: {base_url}...")
                async with websockets.connect(base_url, ssl=ssl_context) as ws:
                    self.status = "CONNECTED"
                    self.last_message_time = time.time()
                    logger.info("Trade websocket connected. Sending SUBSCRIBE command...")
                    
                    subscribe_payload = {
                        "method": "SUBSCRIBE",
                        "params": trade_streams,
                        "id": 1
                    }
                    await ws.send(json.dumps(subscribe_payload))
                    logger.info("SUBSCRIBE command sent successfully.")
                    backoff = 1.0  # Reset backoff on successful connection

                    while self.is_running:
                        try:
                            message = await ws.recv()
                            self.last_message_time = time.time()
                            payload = json.loads(message)
                            
                            # Combined stream format: {"stream": "<symbol>@aggTrade", "data": {...}}
                            stream = payload.get("stream", "")
                            data = payload.get("data", {})
                            
                            if not stream or not data:
                                continue

                            # Normalize symbol to lowercase
                            symbol = stream.split("@")[0].lower()
                            await self._handle_message(symbol, data)

                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Trade websocket connection closed by host.")
                            break
                        except Exception as e:
                            logger.error(f"Error reading trade message: {e}\n{traceback.format_exc()}")
                            await asyncio.sleep(0.1)

            except Exception as e:
                self.status = "RECONNECTING"
                logger.error(f"Trade stream connection failed: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)

    async def _handle_message(self, symbol: str, data: dict):
        if data.get("e") != "aggTrade":
            return

        try:
            price = float(data["p"])
            quantity = float(data["q"])
            trade_time = float(data["T"]) / 1000.0  # convert to seconds
            is_buyer_maker = data["m"]
            
            # If buyer is maker, aggressive taker sold (SELL).
            # If buyer is taker, aggressive taker bought (BUY).
            side = "SELL" if is_buyer_maker else "BUY"

            parsed_trade = {
                "timestamp": trade_time,
                "price": price,
                "quantity": quantity,
                "side": side,
                "symbol": symbol
            }

            await self.callback(symbol, parsed_trade)
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse trade message: {e}. Data: {data}")
