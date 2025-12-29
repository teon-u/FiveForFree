"""System status and data management endpoints."""

from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.api.dependencies import get_model_manager
from src.collector.ticker_selector import TickerSelector
from src.utils.database import get_db, MinuteBar, Ticker
from sqlalchemy import select, func


router = APIRouter(
    prefix="/api/status",
    tags=["status"],
)


class SystemStatus(BaseModel):
    """System status response."""
    trained_tickers: int = Field(..., description="Number of tickers with trained models")
    tickers_with_data: int = Field(..., description="Number of tickers with minute bar data")
    total_minute_bars: int = Field(..., description="Total minute bars in database")
    data_date_range: Optional[Dict[str, str]] = Field(None, description="Data date range")
    model_coverage: float = Field(..., description="Percentage of data tickers with models")
    new_gainers: List[Dict[str, Any]] = Field(default_factory=list, description="New gainers not in system")
    timestamp: str = Field(..., description="Status check timestamp")


class NewTickerInfo(BaseModel):
    """Information about a new ticker discovered."""
    ticker: str
    company_name: Optional[str]
    change_percent: float
    price: float
    volume: float
    has_data: bool
    has_model: bool


class DiscoveredTickersResponse(BaseModel):
    """Response with discovered tickers."""
    market_gainers: List[NewTickerInfo]
    new_tickers: List[str]  # Tickers not yet in system
    tickers_needing_training: List[str]  # Tickers with data but no model
    timestamp: str


@router.get("", response_model=SystemStatus)
@router.get("/", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """
    Get current system status including model coverage and data statistics.
    """
    try:
        model_manager = get_model_manager()
        trained_tickers = set(model_manager.get_tickers())

        with get_db() as db:
            # Count minute bars
            total_bars = db.execute(select(func.count(MinuteBar.id))).scalar() or 0

            # Get tickers with data (normalize to uppercase for consistent comparison)
            symbols_stmt = select(MinuteBar.symbol).distinct()
            tickers_with_data = set(s.upper() for s in db.execute(symbols_stmt).scalars().all())

            # Date range
            min_date = db.execute(select(func.min(MinuteBar.timestamp))).scalar()
            max_date = db.execute(select(func.max(MinuteBar.timestamp))).scalar()

        # Calculate coverage
        coverage = len(trained_tickers & tickers_with_data) / len(tickers_with_data) * 100 if tickers_with_data else 0

        # Get market gainers to find new opportunities
        selector = TickerSelector()
        market_gainers = selector.get_market_top_gainers(limit=20)

        new_gainers = []
        for g in market_gainers[:10]:
            # Normalize case for consistent comparison
            if g.ticker.upper() not in trained_tickers:
                new_gainers.append({
                    "ticker": g.ticker,
                    "name": g.company_name,
                    "change_percent": round(g.change_percent, 2),
                    "price": round(g.price, 2),
                })

        return SystemStatus(
            trained_tickers=len(trained_tickers),
            tickers_with_data=len(tickers_with_data),
            total_minute_bars=total_bars,
            data_date_range={
                "start": min_date.isoformat() if min_date else None,
                "end": max_date.isoformat() if max_date else None,
            } if min_date else None,
            model_coverage=round(coverage, 1),
            new_gainers=new_gainers,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover", response_model=DiscoveredTickersResponse)
async def discover_new_tickers() -> DiscoveredTickersResponse:
    """
    Discover new tickers from market gainers that are not yet in the system.
    """
    try:
        model_manager = get_model_manager()
        trained_tickers = set(model_manager.get_tickers())

        with get_db() as db:
            # Normalize to uppercase for consistent comparison with model_manager
            symbols_stmt = select(MinuteBar.symbol).distinct()
            tickers_with_data = set(s.upper() for s in db.execute(symbols_stmt).scalars().all())

        # Get market gainers
        selector = TickerSelector()
        market_gainers = selector.get_market_top_gainers(limit=50)

        gainer_infos = []
        new_tickers = []
        needing_training = []

        for g in market_gainers:
            ticker_upper = g.ticker.upper()
            has_data = ticker_upper in tickers_with_data
            has_model = ticker_upper in trained_tickers

            gainer_infos.append(NewTickerInfo(
                ticker=g.ticker,
                company_name=g.company_name,
                change_percent=round(g.change_percent, 2),
                price=round(g.price, 2),
                volume=g.volume,
                has_data=has_data,
                has_model=has_model,
            ))

            if not has_data:
                new_tickers.append(g.ticker)
            elif not has_model:
                needing_training.append(g.ticker)

        return DiscoveredTickersResponse(
            market_gainers=gainer_infos,
            new_tickers=new_tickers,
            tickers_needing_training=needing_training,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to discover tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for auto-training
_training_in_progress = False


async def _train_ticker_background(ticker: str):
    """Background task to train models for a ticker."""
    global _training_in_progress

    try:
        _training_in_progress = True
        logger.info(f"Starting background training for {ticker}")

        # Import here to avoid circular imports
        from src.collector.minute_bars import MinuteBarCollector
        from src.processor.feature_engineer import FeatureEngineer
        from src.processor.label_generator import LabelGenerator
        from src.trainer.gpu_trainer import GPUParallelTrainer
        from config.settings import settings
        import numpy as np
        import pandas as pd
        from datetime import timedelta

        model_manager = get_model_manager()

        # Step 1: Collect data if needed
        collector = MinuteBarCollector(use_db=True)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        bars = collector.get_bars(ticker, start_date, end_date)
        if not bars or len(bars) < 1000:
            logger.warning(f"{ticker}: Insufficient data for training ({len(bars) if bars else 0} bars)")
            return

        # Convert to DataFrame
        df = pd.DataFrame(bars)

        # Step 2: Compute features
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.compute_features(df)
        feature_names = feature_engineer.get_feature_names()

        # Step 3: Generate labels
        label_generator = LabelGenerator()
        labels_up = []
        labels_down = []

        for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
            entry_time = df.iloc[idx]["timestamp"]
            entry_price = df.iloc[idx]["close"]
            labels = label_generator.generate_labels(df, entry_time, entry_price)
            labels_up.append(labels["label_up"])
            labels_down.append(labels["label_down"])

        # Prepare arrays
        X = features_df[feature_names].values[: len(labels_up)]
        y_up = np.array(labels_up)
        y_down = np.array(labels_down)

        # Remove NaN rows
        valid_indices = ~np.isnan(X).any(axis=1)
        X = X[valid_indices]
        y_up = y_up[valid_indices]
        y_down = y_down[valid_indices]

        if len(X) < 100:
            logger.warning(f"{ticker}: Too few valid samples ({len(X)})")
            return

        # Step 4: Train models
        trainer = GPUParallelTrainer(model_manager)
        results = trainer.train_single_ticker(ticker, X, y_up, y_down)

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"{ticker}: Training complete - {success_count}/{len(results)} models trained")

        # Step 5: Update model_manager's ticker list after successful training
        # This ensures get_tickers() returns the newly trained ticker
        if success_count > 0:
            if ticker not in model_manager._tickers:
                model_manager._tickers.append(ticker)
                logger.info(f"{ticker}: Added to trained tickers list")

    except Exception as e:
        logger.error(f"Background training failed for {ticker}: {e}")
    finally:
        _training_in_progress = False


@router.post("/train/{ticker}")
async def train_ticker(ticker: str, background_tasks: BackgroundTasks):
    """
    Trigger training for a specific ticker in the background.
    """
    global _training_in_progress

    if _training_in_progress:
        raise HTTPException(status_code=409, detail="Training already in progress")

    ticker = ticker.upper()
    background_tasks.add_task(_train_ticker_background, ticker)

    return {
        "status": "training_started",
        "ticker": ticker,
        "message": f"Training for {ticker} started in background",
    }


@router.get("/training")
async def get_training_status():
    """Check if training is in progress."""
    return {
        "training_in_progress": _training_in_progress,
    }
