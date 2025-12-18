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

        # Calculate lookback time
        now = datetime.utcnow()
        lookback_time = now - timedelta(minutes=minutes)

        # Query minute bars
        stmt = (
            select(MinuteBar)
            .where(MinuteBar.symbol == ticker)
            .where(MinuteBar.timestamp >= lookback_time)
            .order_by(MinuteBar.timestamp)
        )

        bars = list(db.execute(stmt).scalars().all())

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
