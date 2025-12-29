"""Price data endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from loguru import logger

from src.api.dependencies import get_db
from src.utils.database import MinuteBar


router = APIRouter(
    prefix="/api/prices",
    tags=["prices"],
)


class PriceBar(BaseModel):
    """Single price bar model."""

    timestamp: str = Field(..., description="Bar timestamp")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: int = Field(..., description="Volume")


class PriceHistoryResponse(BaseModel):
    """Price history response model."""

    ticker: str = Field(..., description="Ticker symbol")
    bars: List[PriceBar] = Field(..., description="Price bars")
    total_bars: int = Field(..., description="Total number of bars")
    start_time: Optional[str] = Field(None, description="Start timestamp")
    end_time: Optional[str] = Field(None, description="End timestamp")
    timestamp: str = Field(..., description="Response timestamp")


@router.get("/{ticker}", response_model=PriceHistoryResponse)
async def get_price_history(
    ticker: str,
    minutes: int = Query(60, ge=1, le=1440, description="Number of minutes to look back"),
    db: Session = Depends(get_db),
) -> PriceHistoryResponse:
    """
    Get price history for a specific ticker.

    Returns minute-by-minute price data (OHLCV) for the specified lookback period.

    Args:
        ticker: Ticker symbol
        minutes: Number of minutes to look back (default: 60, max: 1440 = 24 hours)
        db: Database session

    Returns:
        PriceHistoryResponse with price bars

    Raises:
        HTTPException: If ticker not found or no data available
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Fetching price history for {ticker} (last {minutes} minutes)")

        # Calculate lookback time (use local time to match stored data)
        now = datetime.now()
        lookback_time = now - timedelta(minutes=minutes)

        # Query minute bars
        stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .where(MinuteBar.timestamp >= lookback_time)
            .order_by(MinuteBar.timestamp)
        )

        bars = list(db.execute(stmt).scalars().all())

        # If no data in requested window, fetch most recent available bars
        if not bars:
            logger.info(f"No data in {minutes}min window, fetching most recent bars")
            stmt_recent = (
                select(MinuteBar)
                .where(MinuteBar.symbol == ticker)
                .order_by(desc(MinuteBar.timestamp))
                .limit(minutes)  # Return same number of bars as requested
            )
            bars = list(db.execute(stmt_recent).scalars().all())
            bars.reverse()  # Oldest first for chart display

        if not bars:
            logger.warning(f"No price data found for {ticker}")
            return PriceHistoryResponse(
                ticker=ticker,
                bars=[],
                total_bars=0,
                start_time=None,
                end_time=None,
                timestamp=datetime.utcnow().isoformat(),
            )

        # Convert to response format
        price_bars = []
        for bar in bars:
            price_bars.append(
                PriceBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            )

        start_time = bars[0].timestamp.isoformat() if bars else None
        end_time = bars[-1].timestamp.isoformat() if bars else None

        logger.info(f"Returning {len(price_bars)} price bars for {ticker}")

        return PriceHistoryResponse(
            ticker=ticker,
            bars=price_bars,
            total_bars=len(price_bars),
            start_time=start_time,
            end_time=end_time,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get price history for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve price history: {str(e)}",
        )


class HistoryResponse(BaseModel):
    """History response model for frontend."""

    symbol: str = Field(..., description="Ticker symbol")
    interval: str = Field(..., description="Data interval")
    data: List[dict] = Field(..., description="Price data points")


@router.get("/{ticker}/history")
async def get_price_history_v2(
    ticker: str,
    interval: str = Query("1m", description="Data interval: 1m, 5m, 15m, 1h, 1d"),
    period: str = Query("1D", description="Period: 1D, 1W, 1M, 3M"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """
    Get price history with interval and period for frontend charts.

    Args:
        ticker: Ticker symbol
        interval: Data interval (1m, 5m, 15m, 1h, 1d)
        period: Lookback period (1D, 1W, 1M, 3M)
        db: Database session

    Returns:
        HistoryResponse with price data
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Fetching price history for {ticker} (interval={interval}, period={period})")

        # Calculate lookback minutes based on period
        period_minutes = {
            "1D": 60 * 24,      # 1 day = 1440 minutes
            "1W": 60 * 24 * 7,  # 1 week
            "1M": 60 * 24 * 30, # 1 month
            "3M": 60 * 24 * 90, # 3 months
        }
        minutes = period_minutes.get(period, 1440)

        now = datetime.now()
        lookback_time = now - timedelta(minutes=minutes)

        # Query minute bars
        stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .where(MinuteBar.timestamp >= lookback_time)
            .order_by(MinuteBar.timestamp)
        )

        bars = list(db.execute(stmt).scalars().all())

        # If no data in requested window, fetch most recent available bars
        if not bars:
            logger.info(f"No data in {period} window, fetching most recent bars")
            stmt_recent = (
                select(MinuteBar)
                .where(MinuteBar.symbol == ticker)
                .order_by(desc(MinuteBar.timestamp))
                .limit(500)  # Get enough data points
            )
            bars = list(db.execute(stmt_recent).scalars().all())
            bars.reverse()

        if not bars:
            return HistoryResponse(
                symbol=ticker,
                interval=interval,
                data=[],
            )

        # Aggregate based on interval
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "1d": 1440,
        }
        agg_minutes = interval_minutes.get(interval, 1)

        if agg_minutes == 1:
            # No aggregation needed for 1m
            data = [
                {
                    "time": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        else:
            # Aggregate bars
            data = []
            current_group = []

            for bar in bars:
                # Group by interval
                group_key = bar.timestamp.replace(
                    minute=(bar.timestamp.minute // agg_minutes) * agg_minutes,
                    second=0,
                    microsecond=0
                )

                if not current_group or current_group[0]["group_key"] != group_key:
                    if current_group:
                        # Finalize previous group
                        first = current_group[0]
                        data.append({
                            "time": first["group_key"].isoformat(),
                            "open": first["open"],
                            "high": max(b["high"] for b in current_group),
                            "low": min(b["low"] for b in current_group),
                            "close": current_group[-1]["close"],
                            "volume": sum(b["volume"] for b in current_group),
                        })
                    current_group = []

                current_group.append({
                    "group_key": group_key,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })

            # Don't forget last group
            if current_group:
                first = current_group[0]
                data.append({
                    "time": first["group_key"].isoformat(),
                    "open": first["open"],
                    "high": max(b["high"] for b in current_group),
                    "low": min(b["low"] for b in current_group),
                    "close": current_group[-1]["close"],
                    "volume": sum(b["volume"] for b in current_group),
                })

        logger.info(f"Returning {len(data)} aggregated bars for {ticker}")

        return HistoryResponse(
            symbol=ticker,
            interval=interval,
            data=data,
        )

    except Exception as e:
        logger.error(f"Failed to get price history for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve price history: {str(e)}",
        )


@router.get("/{ticker}/latest", response_model=PriceBar)
async def get_latest_price(
    ticker: str,
    db: Session = Depends(get_db),
) -> PriceBar:
    """
    Get the latest price bar for a specific ticker.

    Args:
        ticker: Ticker symbol
        db: Database session

    Returns:
        PriceBar with latest price data

    Raises:
        HTTPException: If ticker not found or no data available
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Fetching latest price for {ticker}")

        # Query latest bar
        stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .order_by(desc(MinuteBar.timestamp))
            .limit(1)
        )

        bar = db.execute(stmt).scalar_one_or_none()

        if not bar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for {ticker}",
            )

        return PriceBar(
            timestamp=bar.timestamp.isoformat(),
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest price for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest price: {str(e)}",
        )
