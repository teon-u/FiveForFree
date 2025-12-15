"""Dependency injection for FastAPI endpoints."""

from typing import Generator, Optional
from contextlib import contextmanager
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger

from src.models.model_manager import ModelManager
from src.predictor.realtime_predictor import RealtimePredictor
from src.collector.minute_bars import MinuteBarCollector
from src.collector.quotes import QuoteCollector
from src.collector.market_context import MarketContextCollector
from src.utils.database import SessionLocal, init_db
from config.settings import settings


# Database session dependency
def get_db() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup.

    Yields:
        SQLAlchemy session

    Raises:
        HTTPException: If database connection fails
    """
    if SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )
    finally:
        db.close()


# Global instances (initialized on startup)
_model_manager: Optional[ModelManager] = None
_realtime_predictor: Optional[RealtimePredictor] = None
_minute_bar_collector: Optional[MinuteBarCollector] = None
_quote_collector: Optional[QuoteCollector] = None
_market_context_collector: Optional[MarketContextCollector] = None


def init_dependencies():
    """
    Initialize global dependencies.

    This should be called during application startup.
    """
    global _model_manager, _realtime_predictor
    global _minute_bar_collector, _quote_collector, _market_context_collector

    try:
        logger.info("Initializing API dependencies...")

        # Initialize database
        init_db()

        # Initialize model manager
        _model_manager = ModelManager()
        logger.info("ModelManager initialized")

        # Initialize data collectors (with API key from settings)
        try:
            _minute_bar_collector = MinuteBarCollector(api_key=settings.POLYGON_API_KEY)
            logger.info("MinuteBarCollector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MinuteBarCollector: {e}")

        try:
            _quote_collector = QuoteCollector(api_key=settings.POLYGON_API_KEY)
            logger.info("QuoteCollector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize QuoteCollector: {e}")

        try:
            _market_context_collector = MarketContextCollector(api_key=settings.POLYGON_API_KEY)
            logger.info("MarketContextCollector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MarketContextCollector: {e}")

        # Initialize realtime predictor
        _realtime_predictor = RealtimePredictor(
            model_manager=_model_manager,
            minute_bar_collector=_minute_bar_collector,
            quote_collector=_quote_collector,
            market_context_collector=_market_context_collector,
        )
        logger.info("RealtimePredictor initialized")

        logger.info("All API dependencies initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise


def shutdown_dependencies():
    """
    Cleanup dependencies on application shutdown.
    """
    global _model_manager, _realtime_predictor
    global _minute_bar_collector, _quote_collector, _market_context_collector

    logger.info("Shutting down API dependencies...")

    # Clear global references
    _model_manager = None
    _realtime_predictor = None
    _minute_bar_collector = None
    _quote_collector = None
    _market_context_collector = None

    logger.info("API dependencies shutdown complete")


def get_model_manager() -> ModelManager:
    """
    Get ModelManager instance.

    Returns:
        ModelManager instance

    Raises:
        HTTPException: If ModelManager is not initialized
    """
    if _model_manager is None:
        logger.error("ModelManager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )
    return _model_manager


def get_realtime_predictor() -> RealtimePredictor:
    """
    Get RealtimePredictor instance.

    Returns:
        RealtimePredictor instance

    Raises:
        HTTPException: If RealtimePredictor is not initialized
    """
    if _realtime_predictor is None:
        logger.error("RealtimePredictor not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not available"
        )
    return _realtime_predictor


@lru_cache()
def get_settings():
    """
    Get application settings (cached).

    Returns:
        Settings instance
    """
    return settings
