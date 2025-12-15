"""FastAPI application for NASDAQ prediction system."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import (
    predictions_router,
    tickers_router,
    models_router,
    health_router,
)
from src.api.websocket import (
    handle_websocket_connection,
    broadcast_predictions,
    send_heartbeat,
)
from src.api.dependencies import (
    init_dependencies,
    shutdown_dependencies,
    get_realtime_predictor,
    get_settings,
)
from config.settings import settings


# Background tasks
background_tasks = set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info("Starting NASDAQ Prediction API...")

    try:
        # Initialize dependencies
        init_dependencies()

        # Start background tasks
        predictor = get_realtime_predictor()
        app_settings = get_settings()

        # Get list of active tickers (you may want to load this from database)
        # For now, we'll use a placeholder - in production, query the database
        # for active tickers or configure in settings
        active_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]  # Example tickers

        # Start prediction broadcast task (every minute)
        prediction_task = asyncio.create_task(
            broadcast_predictions(
                predictor=predictor,
                tickers=active_tickers,
                interval_seconds=60,
            )
        )
        background_tasks.add(prediction_task)
        prediction_task.add_done_callback(background_tasks.discard)

        # Start heartbeat task (every 30 seconds)
        heartbeat_task = asyncio.create_task(send_heartbeat(interval_seconds=30))
        background_tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(background_tasks.discard)

        logger.info(f"API started successfully on {app_settings.API_HOST}:{app_settings.API_PORT}")
        logger.info(f"Broadcasting predictions for: {', '.join(active_tickers)}")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down NASDAQ Prediction API...")

    # Cancel background tasks
    for task in background_tasks:
        task.cancel()

    # Wait for tasks to complete
    await asyncio.gather(*background_tasks, return_exceptions=True)

    # Cleanup dependencies
    shutdown_dependencies()

    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="NASDAQ Prediction API",
    description="Real-time prediction API for NASDAQ stock movements using ensemble ML models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",  # Common React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health_router)
app.include_router(predictions_router)
app.include_router(tickers_router)
app.include_router(models_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> JSONResponse:
    """
    Root endpoint with API information.

    Returns:
        JSONResponse with API details
    """
    return JSONResponse(
        content={
            "name": "NASDAQ Prediction API",
            "version": "1.0.0",
            "description": "Real-time prediction API for NASDAQ stock movements",
            "endpoints": {
                "health": "/api/health",
                "predictions": "/api/predictions",
                "tickers": "/api/tickers",
                "models": "/api/models",
                "websocket": "/ws",
                "docs": "/docs",
                "redoc": "/redoc",
            },
        }
    )


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time prediction updates.

    Clients can connect to receive real-time predictions and market updates.

    Supported client messages:
    - {"type": "subscribe", "tickers": ["AAPL", "GOOGL"]}
    - {"type": "unsubscribe", "tickers": ["AAPL"]}
    - {"type": "predict", "ticker": "AAPL"}
    - {"type": "ping"}

    Server messages:
    - {"type": "connected", ...}
    - {"type": "prediction_update", ...}
    - {"type": "heartbeat", ...}
    - {"type": "error", ...}

    Args:
        websocket: WebSocket connection
    """
    try:
        predictor = get_realtime_predictor()
    except Exception as e:
        logger.error(f"Failed to get predictor for WebSocket: {e}")
        predictor = None

    await handle_websocket_connection(websocket, predictor)


# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 Not Found errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 Internal Server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
        },
    )


# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
