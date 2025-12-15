"""Loguru logger configuration with rotation and structured logging."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


# Remove default logger
logger.remove()

# Define log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(
    module_name: Optional[str] = None,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
):
    """
    Get a configured logger instance with rotation and structured logging.

    Args:
        module_name: Name of the module (used for log file naming and filtering)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console output
        enable_file: Enable file output

    Returns:
        Configured logger instance

    Examples:
        >>> logger = get_logger("collector", log_level="DEBUG")
        >>> logger.info("Starting data collection", ticker="AAPL", count=100)
        >>> logger.error("API error", error=str(e), ticker="TSLA")
    """
    # Create a new logger instance with context
    if module_name:
        context_logger = logger.bind(module=module_name)
    else:
        context_logger = logger

    # Console handler with color formatting
    if enable_console:
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[module]: <15}</cyan> | "
                "<level>{message}</level> | "
                "{extra}"
            ),
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["extra"].get("module") == module_name if module_name else True,
        )

    # File handler - Main log with rotation
    if enable_file:
        logger.add(
            LOG_DIR / "nasdaq_predictor.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{extra[module]: <15} | "
                "{message} | "
                "{extra}"
            ),
            level=log_level,
            rotation="100 MB",  # Rotate when file reaches 100MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe
        )

    # Module-specific log file
    if enable_file and module_name:
        module_log_file = LOG_DIR / f"{module_name}.log"
        logger.add(
            module_log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{message} | "
                "{extra}"
            ),
            level=log_level,
            rotation="50 MB",
            retention="14 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            filter=lambda record: record["extra"].get("module") == module_name,
        )

    # Error-only log file
    if enable_file:
        logger.add(
            LOG_DIR / "errors.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{extra[module]: <15} | "
                "{message} | "
                "{extra} | "
                "{exception}"
            ),
            level="ERROR",
            rotation="50 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    return context_logger


# Pre-configured loggers for different modules
def get_collector_logger(log_level: str = "INFO"):
    """Get logger for data collection modules."""
    return get_logger("collector", log_level=log_level)


def get_processor_logger(log_level: str = "INFO"):
    """Get logger for data processing modules."""
    return get_logger("processor", log_level=log_level)


def get_model_logger(log_level: str = "INFO"):
    """Get logger for model training and inference."""
    return get_logger("model", log_level=log_level)


def get_trainer_logger(log_level: str = "INFO"):
    """Get logger for training modules."""
    return get_logger("trainer", log_level=log_level)


def get_predictor_logger(log_level: str = "INFO"):
    """Get logger for prediction modules."""
    return get_logger("predictor", log_level=log_level)


def get_backtester_logger(log_level: str = "INFO"):
    """Get logger for backtesting modules."""
    return get_logger("backtester", log_level=log_level)


def get_api_logger(log_level: str = "INFO"):
    """Get logger for API modules."""
    return get_logger("api", log_level=log_level)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic logging
    test_logger = get_logger("test", log_level="DEBUG")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")

    # Example 2: Structured logging with context
    collector_logger = get_collector_logger()
    collector_logger.info(
        "Fetching market data",
        ticker="AAPL",
        timeframe="1min",
        count=100,
    )

    # Example 3: Error logging with exception
    try:
        1 / 0
    except Exception as e:
        test_logger.exception("An error occurred")

    # Example 4: Model-specific logging
    model_logger = get_model_logger()
    model_logger.info(
        "Training completed",
        model_type="xgboost",
        accuracy=0.87,
        samples=10000,
        duration_sec=45.2,
    )

    # Example 5: Trading signals
    predictor_logger = get_predictor_logger()
    predictor_logger.warning(
        "High probability signal detected",
        ticker="TSLA",
        direction="up",
        probability=0.85,
        entry_price=245.67,
        target_price=258.0,
    )

    print(f"\nLog files created in: {LOG_DIR.absolute()}")
