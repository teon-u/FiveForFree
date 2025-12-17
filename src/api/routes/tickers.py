"""Ticker endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session
from loguru import logger

from src.api.dependencies import get_db, get_model_manager
from src.models.model_manager import ModelManager
from src.utils.database import Ticker, MinuteBar, Prediction


router = APIRouter(
    prefix="/api/tickers",
    tags=["tickers"],
)


class TickerMetrics(BaseModel):
    """Ticker metrics model."""

    symbol: str = Field(..., description="Ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    current_price: Optional[float] = Field(None, description="Latest price")
    volume_24h: Optional[int] = Field(None, description="24-hour volume")
    price_change_24h: Optional[float] = Field(None, description="24-hour price change percent")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    is_active: bool = Field(True, description="Whether ticker is active")
    has_trained_models: bool = Field(False, description="Whether models are trained")


class TickerListResponse(BaseModel):
    """Ticker list response model."""

    tickers: List[TickerMetrics] = Field(..., description="List of tickers")
    total: int = Field(..., description="Total number of tickers")
    timestamp: str = Field(..., description="Response timestamp")


class TickerDetail(BaseModel):
    """Detailed ticker information."""

    symbol: str = Field(..., description="Ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    sector: Optional[str] = Field(None, description="Sector")
    industry: Optional[str] = Field(None, description="Industry")
    current_price: Optional[float] = Field(None, description="Latest price")
    volume_24h: Optional[int] = Field(None, description="24-hour volume")
    price_change_24h: Optional[float] = Field(None, description="24-hour price change percent")
    high_24h: Optional[float] = Field(None, description="24-hour high")
    low_24h: Optional[float] = Field(None, description="24-hour low")
    is_active: bool = Field(True, description="Whether ticker is active")
    total_predictions: int = Field(0, description="Total predictions made")
    last_prediction_time: Optional[str] = Field(None, description="Last prediction timestamp")
    model_status: dict = Field(default_factory=dict, description="Model training status")


@router.get("", response_model=TickerListResponse)
@router.get("/", response_model=TickerListResponse)
async def list_tickers(
    active_only: bool = Query(True, description="Return only active tickers"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager),
) -> TickerListResponse:
    """
    Get list of tracked tickers with latest metrics.

    Args:
        active_only: Filter for active tickers only
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session
        model_manager: Model manager instance

    Returns:
        TickerListResponse with ticker list and metrics
    """
    try:
        # Query tickers
        stmt = select(Ticker)

        if active_only:
            stmt = stmt.where(Ticker.is_active == True)

        stmt = stmt.order_by(desc(Ticker.last_updated)).limit(limit).offset(offset)

        tickers = list(db.execute(stmt).scalars().all())

        # Get current time for lookback
        now = datetime.utcnow()
        lookback_24h = now - timedelta(hours=24)

        ticker_metrics = []

        for ticker in tickers:
            # Get latest price
            latest_bar_stmt = (
                select(MinuteBar)
                .where(MinuteBar.symbol == ticker.symbol)
                .order_by(desc(MinuteBar.timestamp))
                .limit(1)
            )
            latest_bar = db.execute(latest_bar_stmt).scalar_one_or_none()

            # Calculate 24h metrics
            current_price = latest_bar.close if latest_bar else None

            # Get 24h volume
            volume_stmt = (
                select(func.sum(MinuteBar.volume))
                .where(MinuteBar.symbol == ticker.symbol)
                .where(MinuteBar.timestamp >= lookback_24h)
            )
            volume_24h = db.execute(volume_stmt).scalar_one() or 0

            # Get price change 24h
            price_change_24h = None
            if latest_bar:
                old_bar_stmt = (
                    select(MinuteBar)
                    .where(MinuteBar.symbol == ticker.symbol)
                    .where(MinuteBar.timestamp <= lookback_24h)
                    .order_by(desc(MinuteBar.timestamp))
                    .limit(1)
                )
                old_bar = db.execute(old_bar_stmt).scalar_one_or_none()

                if old_bar and old_bar.close > 0:
                    price_change_24h = ((current_price - old_bar.close) / old_bar.close) * 100

            # Check if models are trained
            has_trained_models = False
            try:
                model_manager.get_best_model(ticker.symbol, "up")
                has_trained_models = True
            except ValueError:
                logger.debug(f"No trained models for {ticker.symbol}")

            ticker_metrics.append(
                TickerMetrics(
                    symbol=ticker.symbol,
                    name=ticker.name,
                    current_price=current_price,
                    volume_24h=int(volume_24h) if volume_24h else None,
                    price_change_24h=price_change_24h,
                    last_updated=ticker.last_updated.isoformat() if ticker.last_updated else None,
                    is_active=ticker.is_active,
                    has_trained_models=has_trained_models,
                )
            )

        # Get total count
        count_stmt = select(func.count(Ticker.id))
        if active_only:
            count_stmt = count_stmt.where(Ticker.is_active == True)
        total = db.execute(count_stmt).scalar_one()

        return TickerListResponse(
            tickers=ticker_metrics,
            total=total,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to list tickers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tickers: {str(e)}",
        )


@router.get("/{ticker}", response_model=TickerDetail)
async def get_ticker_detail(
    ticker: str,
    db: Session = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager),
) -> TickerDetail:
    """
    Get detailed information for a specific ticker.

    Args:
        ticker: Ticker symbol
        db: Database session
        model_manager: Model manager instance

    Returns:
        TickerDetail with comprehensive ticker information

    Raises:
        HTTPException: If ticker not found
    """
    try:
        ticker = ticker.upper()

        # Get ticker from database
        stmt = select(Ticker).where(Ticker.symbol == ticker)
        ticker_obj = db.execute(stmt).scalar_one_or_none()

        if not ticker_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker {ticker} not found",
            )

        # Get latest price
        latest_bar_stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .order_by(desc(MinuteBar.timestamp))
            .limit(1)
        )
        latest_bar = db.execute(latest_bar_stmt).scalar_one_or_none()

        current_price = latest_bar.close if latest_bar else None

        # Get 24h metrics
        now = datetime.utcnow()
        lookback_24h = now - timedelta(hours=24)

        # Volume
        volume_stmt = (
            select(func.sum(MinuteBar.volume))
            .where(MinuteBar.symbol == ticker)
            .where(MinuteBar.timestamp >= lookback_24h)
        )
        volume_24h = db.execute(volume_stmt).scalar_one() or 0

        # Price change, high, low
        bars_24h_stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .where(MinuteBar.timestamp >= lookback_24h)
            .order_by(MinuteBar.timestamp)
        )
        bars_24h = list(db.execute(bars_24h_stmt).scalars().all())

        price_change_24h = None
        high_24h = None
        low_24h = None

        if bars_24h:
            old_price = bars_24h[0].close
            if old_price > 0 and current_price:
                price_change_24h = ((current_price - old_price) / old_price) * 100

            high_24h = max(bar.high for bar in bars_24h)
            low_24h = min(bar.low for bar in bars_24h)

        # Get prediction stats
        pred_count_stmt = (
            select(func.count(Prediction.id))
            .where(Prediction.symbol == ticker)
        )
        total_predictions = db.execute(pred_count_stmt).scalar_one() or 0

        last_pred_stmt = (
            select(Prediction)
            .where(Prediction.symbol == ticker)
            .order_by(desc(Prediction.prediction_time))
            .limit(1)
        )
        last_prediction = db.execute(last_pred_stmt).scalar_one_or_none()
        last_prediction_time = (
            last_prediction.prediction_time.isoformat() if last_prediction else None
        )

        # Get model status
        model_status = model_manager.validate_models(ticker)

        return TickerDetail(
            symbol=ticker_obj.symbol,
            name=ticker_obj.name,
            market_cap=ticker_obj.market_cap,
            sector=ticker_obj.sector,
            industry=ticker_obj.industry,
            current_price=current_price,
            volume_24h=int(volume_24h) if volume_24h else None,
            price_change_24h=price_change_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            is_active=ticker_obj.is_active,
            total_predictions=total_predictions,
            last_prediction_time=last_prediction_time,
            model_status=model_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ticker detail for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ticker details: {str(e)}",
        )
