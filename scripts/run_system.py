#!/usr/bin/env python3
"""
Run NASDAQ Prediction System

Main system runner that starts:
1. APScheduler for automated jobs (data collection, training, predictions)
2. FastAPI server for REST API and WebSocket connections

The system runs continuously with scheduled jobs:
- Every minute: Data collection + predictions (market hours)
- Every hour: Incremental training + target ticker update
- Daily at 5 PM ET: Full model retraining after market close

Usage:
    # Run with default settings
    python scripts/run_system.py

    # Run without scheduler (API only)
    python scripts/run_system.py --no-scheduler

    # Run without API (scheduler only)
    python scripts/run_system.py --no-api

    # Custom API host/port
    python scripts/run_system.py --host 0.0.0.0 --port 8080

    # Disable market hours check (run 24/7)
    python scripts/run_system.py --no-market-hours-check
"""

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import uvicorn

from config.settings import settings
from src.utils.database import init_db
from src.models.model_manager import ModelManager
from src.scheduler import NASDAQScheduler


# Global instances for graceful shutdown
scheduler_instance: Optional[NASDAQScheduler] = None
api_server_thread: Optional[threading.Thread] = None
shutdown_event = threading.Event()


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging.

    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"

    # Console logging
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    # File logging
    from datetime import datetime

    log_file = (
        Path("./logs") / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        compression="zip",
    )


def run_api_server(host: str, port: int) -> None:
    """
    Run FastAPI server in background thread.

    Args:
        host: API host address
        port: API port number
    """
    import asyncio

    try:
        logger.info(f"Starting FastAPI server on {host}:{port}...")

        # Import FastAPI app
        from src.api.main import app

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            use_colors=True,
        )

        server = uvicorn.Server(config)

        # Run server with proper async handling
        # server.run() returns immediately in threads, use serve() with asyncio.run() instead
        asyncio.run(server.serve())

    except Exception as e:
        logger.error(f"FastAPI server failed: {e}")
        shutdown_event.set()


def signal_handler(signum, frame) -> None:
    """
    Handle shutdown signals (SIGINT, SIGTERM).

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    signal_name = signal.Signals(signum).name
    logger.warning(f"\nReceived {signal_name}, initiating graceful shutdown...")
    shutdown_event.set()


def check_system_requirements() -> bool:
    """
    Check if system requirements are met.

    Returns:
        True if all requirements met, False otherwise
    """
    logger.info("Checking system requirements...")

    # Check database
    try:
        init_db()
        logger.success("✓ Database connection OK")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False

    # Check Finnhub API key
    if not settings.FINNHUB_API_KEY or settings.FINNHUB_API_KEY == "your_finnhub_api_key_here":
        logger.warning("⚠ FINNHUB_API_KEY not configured (optional for secondary data)")
    else:
        logger.success("✓ Finnhub API key configured")

    # Check GPU availability (optional)
    if settings.USE_GPU:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.success(f"✓ GPU available: {gpu_name}")
            else:
                logger.warning("⚠ GPU not available, will use CPU")
        except ImportError:
            logger.warning("⚠ PyTorch not installed, cannot check GPU")

    # Check data directories
    for directory in ["./data", "./data/models", "./logs"]:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    logger.success("✓ All requirements met")
    return True


def initialize_model_manager() -> ModelManager:
    """
    Initialize and load existing models.

    Returns:
        ModelManager instance
    """
    logger.info("Initializing model manager...")

    model_manager = ModelManager()

    # Try to load existing models
    models_path = Path("./data/models")
    if models_path.exists():
        model_files = list(models_path.glob("*.pkl"))
        if model_files:
            logger.info(f"Found {len(model_files)} saved model files")

            # Extract unique tickers from model files
            tickers = set()
            for model_file in model_files:
                # Model files are named like: AAPL_xgboost_up.pkl
                parts = model_file.stem.split("_")
                if len(parts) >= 3:
                    ticker = parts[0]
                    tickers.add(ticker)

            logger.info(f"Loading models for {len(tickers)} tickers...")

            for ticker in tickers:
                try:
                    model_manager.load_models(ticker)
                except Exception as e:
                    logger.warning(f"Failed to load models for {ticker}: {e}")

            # Get summary
            summary = model_manager.get_summary()
            logger.success(
                f"✓ Loaded {summary['trained_models']}/{summary['total_models']} models"
            )
        else:
            logger.warning("No saved models found, will start with fresh models")
    else:
        logger.warning("Models directory does not exist, will start with fresh models")

    return model_manager


def print_startup_banner() -> None:
    """Print startup banner."""
    banner = """
================================================================================

           NASDAQ PREDICTION SYSTEM - AUTOMATED TRADING ENGINE

  Features:
    - Real-time data collection (1-minute bars)
    - Multi-model predictions (XGBoost, LightGBM, LSTM, Transformer)
    - Automated training & model selection
    - REST API & WebSocket streaming
    - GPU-accelerated inference

================================================================================
    """
    print(banner)


def print_system_status(
    scheduler_enabled: bool,
    api_enabled: bool,
    host: str,
    port: int,
) -> None:
    """
    Print system status information.

    Args:
        scheduler_enabled: Whether scheduler is enabled
        api_enabled: Whether API is enabled
        host: API host
        port: API port
    """
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM STATUS")
    logger.info("=" * 80)

    if scheduler_enabled:
        logger.info("✓ Scheduler: RUNNING")
        logger.info(f"  - Minute jobs: Data collection & predictions")
        logger.info(f"  - Hourly jobs: Incremental training + ticker updates")
        logger.info(f"  - Daily jobs: Full retraining at 5 PM ET")
    else:
        logger.info("✗ Scheduler: DISABLED")

    if api_enabled:
        logger.info(f"✓ API Server: RUNNING on http://{host}:{port}")
        logger.info(f"  - REST API: http://{host}:{port}/docs")
        logger.info(f"  - WebSocket: ws://{host}:{port}/ws")
    else:
        logger.info("✗ API Server: DISABLED")

    logger.info("\nSettings:")
    logger.info(f"  - Market hours check: {settings.MARKET_OPEN_HOUR}:30 - {settings.MARKET_CLOSE_HOUR}:00 ET")
    logger.info(f"  - GPU enabled: {settings.USE_GPU}")
    logger.info(f"  - Parallel workers: {settings.N_PARALLEL_WORKERS}")
    logger.info(f"  - Database: {settings.DATABASE_URL}")

    logger.info("\n" + "=" * 80)
    logger.info("System is ready. Press Ctrl+C to stop.")
    logger.info("=" * 80 + "\n")


def main() -> int:
    """
    Main function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    global scheduler_instance, api_server_thread

    parser = argparse.ArgumentParser(
        description="Run NASDAQ prediction system with scheduler and API"
    )
    parser.add_argument(
        "--host",
        default=settings.API_HOST,
        help=f"API host address (default: {settings.API_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.API_PORT,
        help=f"API port number (default: {settings.API_PORT})",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable scheduler (API only mode)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API server (scheduler only mode)",
    )
    parser.add_argument(
        "--no-market-hours-check",
        action="store_true",
        help="Disable market hours check (run jobs 24/7)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Print banner
    print_startup_banner()

    try:
        # Check system requirements
        if not check_system_requirements():
            logger.error("System requirements not met, exiting")
            return 1

        # Initialize database
        logger.info("Initializing database...")
        init_db()
        logger.success("✓ Database initialized")

        # Initialize model manager
        model_manager = initialize_model_manager()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start scheduler if enabled
        scheduler_enabled = not args.no_scheduler
        if scheduler_enabled:
            logger.info("\nStarting scheduler...")
            scheduler_instance = NASDAQScheduler(
                model_manager,
                enable_market_hours_check=not args.no_market_hours_check,
            )
            scheduler_instance.start()
            logger.success("✓ Scheduler started")

        # Start API server if enabled
        api_enabled = not args.no_api
        if api_enabled:
            logger.info("\nStarting API server in background thread...")

            api_server_thread = threading.Thread(
                target=run_api_server,
                args=(args.host, args.port),
                daemon=True,
            )
            api_server_thread.start()

            # Give server time to start
            time.sleep(2)

            if api_server_thread.is_alive():
                logger.success("✓ API server started")
            else:
                logger.error("✗ API server failed to start")
                return 1

        # Print system status
        print_system_status(
            scheduler_enabled,
            api_enabled,
            args.host,
            args.port,
        )

        # Main loop - wait for shutdown signal
        while not shutdown_event.is_set():
            time.sleep(1)

            # Check if API thread died
            if api_enabled and api_server_thread and not api_server_thread.is_alive():
                logger.error("API server thread died unexpectedly")
                shutdown_event.set()

        # Graceful shutdown
        logger.info("\n" + "=" * 80)
        logger.info("Initiating graceful shutdown...")
        logger.info("=" * 80)

        # Stop scheduler
        if scheduler_instance:
            logger.info("Stopping scheduler...")
            scheduler_instance.stop()
            logger.success("✓ Scheduler stopped")

        # API server will stop when main thread exits
        if api_enabled:
            logger.info("✓ API server stopping...")

        logger.success("\n" + "=" * 80)
        logger.success("System shutdown complete")
        logger.success("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        if args.verbose:
            logger.exception("Traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
