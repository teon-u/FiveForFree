"""WebSocket handler for real-time prediction updates."""

import asyncio
import json
import math
from typing import Set, List, Optional, Dict, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.predictor.realtime_predictor import RealtimePredictor


def sanitize_for_json(obj):
    """
    Replace NaN/Inf values with None for JSON serialization.

    JSON doesn't support NaN or Infinity values, so we convert them to None.
    This prevents serialization errors when broadcasting prediction data.

    Args:
        obj: Object to sanitize (dict, list, float, or other type)

    Returns:
        Sanitized object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.

    Maintains active connections and handles broadcasting prediction updates
    to all connected clients.
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.subscribed_tickers: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscribed_tickers[websocket] = set()
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        self.active_connections.discard(websocket)
        if websocket in self.subscribed_tickers:
            del self.subscribed_tickers[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """
        Send a message to a specific client.

        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message dictionary to broadcast
        """
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_ticker_subscribers(
        self, ticker: str, message: Dict[str, Any]
    ) -> None:
        """
        Broadcast a message to clients subscribed to a specific ticker.

        Args:
            ticker: Ticker symbol
            message: Message dictionary to send
        """
        disconnected = set()

        for connection, tickers in self.subscribed_tickers.items():
            if ticker in tickers or "*" in tickers:  # "*" means subscribed to all
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to subscriber: {e}")
                    disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe_tickers(self, websocket: WebSocket, tickers: List[str]) -> None:
        """
        Subscribe a WebSocket to specific tickers.

        Args:
            websocket: WebSocket connection
            tickers: List of ticker symbols to subscribe to
        """
        if websocket in self.subscribed_tickers:
            self.subscribed_tickers[websocket].update(tickers)
            logger.info(f"Client subscribed to: {tickers}")

    def unsubscribe_tickers(self, websocket: WebSocket, tickers: List[str]) -> None:
        """
        Unsubscribe a WebSocket from specific tickers.

        Args:
            websocket: WebSocket connection
            tickers: List of ticker symbols to unsubscribe from
        """
        if websocket in self.subscribed_tickers:
            self.subscribed_tickers[websocket].difference_update(tickers)
            logger.info(f"Client unsubscribed from: {tickers}")

    def get_subscriptions(self, websocket: WebSocket) -> Set[str]:
        """
        Get tickers a WebSocket is subscribed to.

        Args:
            websocket: WebSocket connection

        Returns:
            Set of ticker symbols
        """
        return self.subscribed_tickers.get(websocket, set())


# Global connection manager instance
manager = ConnectionManager()


async def handle_websocket_connection(
    websocket: WebSocket,
    predictor: Optional[RealtimePredictor] = None,
) -> None:
    """
    Handle a WebSocket connection lifecycle.

    Accepts the connection, processes incoming messages, and handles disconnection.

    Args:
        websocket: WebSocket connection
        predictor: RealtimePredictor instance (optional)
    """
    await manager.connect(websocket)

    # Send welcome message
    await manager.send_personal_message(
        {
            "type": "connected",
            "message": "Connected to NASDAQ Prediction System",
            "timestamp": datetime.utcnow().isoformat(),
        },
        websocket,
    )

    try:
        while True:
            # Receive and process messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await process_client_message(websocket, message, predictor)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    websocket,
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def process_client_message(
    websocket: WebSocket,
    message: Dict[str, Any],
    predictor: Optional[RealtimePredictor],
) -> None:
    """
    Process incoming message from WebSocket client.

    Supports commands:
    - subscribe: Subscribe to ticker updates
    - unsubscribe: Unsubscribe from ticker updates
    - predict: Request immediate prediction
    - ping: Health check

    Args:
        websocket: WebSocket connection
        message: Message dictionary from client
        predictor: RealtimePredictor instance
    """
    msg_type = message.get("type", "")

    if msg_type == "subscribe":
        # Subscribe to specific tickers
        tickers = message.get("tickers", [])
        if tickers:
            manager.subscribe_tickers(websocket, tickers)
            await manager.send_personal_message(
                {
                    "type": "subscribed",
                    "tickers": tickers,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )

    elif msg_type == "unsubscribe":
        # Unsubscribe from specific tickers
        tickers = message.get("tickers", [])
        if tickers:
            manager.unsubscribe_tickers(websocket, tickers)
            await manager.send_personal_message(
                {
                    "type": "unsubscribed",
                    "tickers": tickers,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )

    elif msg_type == "predict":
        # Request immediate prediction
        if predictor is None:
            await manager.send_personal_message(
                {"type": "error", "message": "Predictor not available"},
                websocket,
            )
            return

        ticker = message.get("ticker", "").upper()
        if ticker:
            try:
                result = predictor.predict(ticker, include_all_models=True)
                prediction_data = {
                    "type": "prediction",
                    "ticker": result.ticker,
                    "timestamp": result.timestamp.isoformat(),
                    "current_price": result.current_price,
                    "up_probability": result.up_probability,
                    "down_probability": result.down_probability,
                    "best_up_model": result.best_up_model,
                    "best_down_model": result.best_down_model,
                    "up_model_accuracy": result.up_model_accuracy,
                    "down_model_accuracy": result.down_model_accuracy,
                    "trading_signal": result.get_trading_signal(),
                    "confidence_level": result.get_confidence_level(),
                }
                # Sanitize before sending to prevent NaN serialization errors
                await manager.send_personal_message(
                    sanitize_for_json(prediction_data),
                    websocket,
                )
            except Exception as e:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Prediction failed for {ticker}: {str(e)}",
                    },
                    websocket,
                )

    elif msg_type == "ping":
        # Health check / keepalive
        await manager.send_personal_message(
            {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

    else:
        # Unknown message type
        await manager.send_personal_message(
            {
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            },
            websocket,
        )


async def broadcast_predictions(
    predictor: RealtimePredictor,
    tickers: List[str],
    interval_seconds: int = 60,
) -> None:
    """
    Periodically broadcast predictions to all connected clients.

    This is a background task that runs continuously, generating predictions
    for specified tickers and broadcasting them to subscribed clients.

    Args:
        predictor: RealtimePredictor instance
        tickers: List of ticker symbols to generate predictions for
        interval_seconds: Interval between prediction broadcasts (default: 60)
    """
    logger.info(f"Starting prediction broadcast task (interval: {interval_seconds}s)")

    while True:
        try:
            if manager.active_connections:
                logger.debug(f"Broadcasting predictions for {len(tickers)} tickers")

                # Generate predictions for all tickers
                # include_all_models=True enables recording for all models for accuracy tracking
                results = predictor.predict_batch(
                    tickers=tickers,
                    include_all_models=True,
                    include_features=False,
                )

                # Broadcast each prediction to subscribed clients
                for ticker, result in results.items():
                    prediction_message = {
                        "type": "prediction_update",
                        "ticker": result.ticker,
                        "timestamp": result.timestamp.isoformat(),
                        "current_price": result.current_price,
                        "up_probability": result.up_probability,
                        "down_probability": result.down_probability,
                        "best_up_model": result.best_up_model,
                        "best_down_model": result.best_down_model,
                        "up_model_accuracy": result.up_model_accuracy,
                        "down_model_accuracy": result.down_model_accuracy,
                        "trading_signal": result.get_trading_signal(),
                        "confidence_level": result.get_confidence_level(),
                    }

                    # Sanitize before broadcasting to prevent NaN serialization errors
                    await manager.broadcast_to_ticker_subscribers(
                        ticker, sanitize_for_json(prediction_message)
                    )

                logger.debug(f"Broadcast complete for {len(results)} predictions")

        except Exception as e:
            logger.error(f"Error in prediction broadcast: {e}")

        # Wait for next interval
        await asyncio.sleep(interval_seconds)


async def broadcast_price_updates(
    tickers: List[str],
    interval_seconds: int = 15,
) -> None:
    """
    Broadcast real-time price updates to all connected clients.

    Fetches current prices from Yahoo Finance and broadcasts to subscribers.
    Runs more frequently than prediction updates for real-time price tracking.

    Args:
        tickers: List of ticker symbols to track
        interval_seconds: Interval between price updates (default: 15 seconds)
    """
    import yfinance as yf
    import math

    logger.info(f"Starting price update broadcast task (interval: {interval_seconds}s)")

    while True:
        try:
            if manager.active_connections and tickers:
                # Batch fetch current prices
                data = yf.download(
                    tickers=tickers[:50],  # Limit for performance
                    period="1d",
                    interval="1m",
                    progress=False,
                    group_by='ticker',
                    threads=True
                )

                if not data.empty:
                    price_updates = []

                    for ticker in tickers[:50]:
                        try:
                            if len(tickers) > 1:
                                ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None
                            else:
                                ticker_data = data

                            if ticker_data is not None and not ticker_data.empty:
                                latest = ticker_data.iloc[-1]
                                raw_close = latest['Close']

                                # Skip NaN/None prices (JSON doesn't support NaN)
                                if raw_close is None or (hasattr(raw_close, '__float__') and math.isnan(float(raw_close))):
                                    continue
                                current_price = float(raw_close)
                                if math.isnan(current_price):
                                    continue

                                # Calculate change from open
                                raw_open = ticker_data.iloc[0]['Open']
                                if raw_open is None or (hasattr(raw_open, '__float__') and math.isnan(float(raw_open))):
                                    open_price = current_price  # fallback to current price
                                else:
                                    open_price = float(raw_open)
                                if open_price > 0 and not math.isnan(open_price):
                                    change_percent = ((current_price - open_price) / open_price) * 100
                                else:
                                    change_percent = 0.0

                                price_updates.append({
                                    "ticker": ticker,
                                    "price": round(current_price, 2),
                                    "change_percent": round(change_percent, 2),
                                })

                        except Exception as e:
                            logger.debug(f"Error getting price for {ticker}: {e}")
                            continue

                    if price_updates:
                        # Broadcast price update to all clients
                        await manager.broadcast({
                            "type": "price_update",
                            "prices": price_updates,
                            "timestamp": datetime.utcnow().isoformat(),
                        })

                        logger.debug(f"Broadcast price updates for {len(price_updates)} tickers")

        except Exception as e:
            logger.error(f"Error in price update broadcast: {e}")

        # Wait for next interval
        await asyncio.sleep(interval_seconds)


async def send_heartbeat(interval_seconds: int = 30) -> None:
    """
    Send periodic heartbeat messages to all connected clients.

    Helps detect dead connections and keeps connections alive.

    Args:
        interval_seconds: Interval between heartbeats (default: 30)
    """
    logger.info(f"Starting heartbeat task (interval: {interval_seconds}s)")

    while True:
        try:
            if manager.active_connections:
                await manager.broadcast(
                    {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                        "connections": len(manager.active_connections),
                    }
                )
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")

        await asyncio.sleep(interval_seconds)
