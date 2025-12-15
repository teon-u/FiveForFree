"""Prediction endpoints."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from loguru import logger

from src.api.dependencies import get_realtime_predictor, get_settings
from src.predictor.realtime_predictor import RealtimePredictor
from config.settings import Settings


router = APIRouter(
    prefix="/api/predictions",
    tags=["predictions"],
)


class PredictionResponse(BaseModel):
    """Single ticker prediction response."""

    ticker: str = Field(..., description="Ticker symbol")
    timestamp: str = Field(..., description="Prediction timestamp")
    current_price: float = Field(..., description="Current stock price")
    up_probability: float = Field(..., description="Probability of 5%+ upward movement")
    down_probability: float = Field(..., description="Probability of 5%+ downward movement")
    best_up_model: str = Field(..., description="Best performing model for up prediction")
    best_down_model: str = Field(..., description="Best performing model for down prediction")
    up_model_accuracy: float = Field(..., description="50-hour accuracy of up model (0-1)")
    down_model_accuracy: float = Field(..., description="50-hour accuracy of down model (0-1)")
    trading_signal: str = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence_level: str = Field(..., description="Confidence level: VERY_HIGH, HIGH, MEDIUM, LOW")
    all_model_predictions: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Predictions from all models (if requested)"
    )
    market_context: Optional[Dict[str, Any]] = Field(
        None, description="Market context data (if requested)"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""

    tickers: List[str] = Field(..., description="List of ticker symbols", min_length=1, max_length=50)
    include_all_models: bool = Field(False, description="Include predictions from all models")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    predictions: Dict[str, PredictionResponse] = Field(..., description="Predictions by ticker")
    total_requested: int = Field(..., description="Total tickers requested")
    total_successful: int = Field(..., description="Total successful predictions")
    failed_tickers: List[str] = Field(default_factory=list, description="Tickers that failed")
    timestamp: str = Field(..., description="Response timestamp")


class TopOpportunitiesResponse(BaseModel):
    """Top trading opportunities response."""

    direction: str = Field(..., description="Direction: up or down")
    opportunities: List[PredictionResponse] = Field(..., description="Top opportunities sorted by probability")
    total_signals: int = Field(..., description="Total number of signals found")
    timestamp: str = Field(..., description="Response timestamp")


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(
    ticker: str,
    include_all_models: bool = Query(False, description="Include predictions from all models"),
    include_context: bool = Query(False, description="Include market context data"),
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
) -> PredictionResponse:
    """
    Get latest prediction for a specific ticker.

    Args:
        ticker: Ticker symbol
        include_all_models: Include predictions from all models
        include_context: Include market context data
        predictor: RealtimePredictor instance

    Returns:
        PredictionResponse with probabilities and model information

    Raises:
        HTTPException: If prediction fails or ticker not found
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Generating prediction for {ticker}")

        # Generate prediction
        result = predictor.predict(
            ticker=ticker,
            include_all_models=include_all_models,
            include_features=include_context,
        )

        return PredictionResponse(
            ticker=result.ticker,
            timestamp=result.timestamp.isoformat(),
            current_price=result.current_price,
            up_probability=result.up_probability,
            down_probability=result.down_probability,
            best_up_model=result.best_up_model,
            best_down_model=result.best_down_model,
            up_model_accuracy=result.up_model_accuracy,
            down_model_accuracy=result.down_model_accuracy,
            trading_signal=result.get_trading_signal(),
            confidence_level=result.get_confidence_level(),
            all_model_predictions=result.all_model_predictions,
            market_context=result.market_context,
        )

    except ValueError as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trained models available for {ticker}",
        )
    except Exception as e:
        logger.error(f"Failed to generate prediction for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("", response_model=BatchPredictionResponse)
@router.post("/", response_model=BatchPredictionResponse)
async def get_batch_predictions(
    request: BatchPredictionRequest,
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
) -> BatchPredictionResponse:
    """
    Get predictions for multiple tickers.

    Args:
        request: BatchPredictionRequest with ticker list
        predictor: RealtimePredictor instance

    Returns:
        BatchPredictionResponse with predictions for each ticker
    """
    try:
        # Normalize ticker symbols
        tickers = [t.upper() for t in request.tickers]
        logger.info(f"Generating batch predictions for {len(tickers)} tickers")

        # Generate predictions
        results = predictor.predict_batch(
            tickers=tickers,
            include_all_models=request.include_all_models,
            include_features=False,
        )

        # Convert to response format
        predictions = {}
        failed_tickers = []

        for ticker in tickers:
            if ticker in results:
                result = results[ticker]
                predictions[ticker] = PredictionResponse(
                    ticker=result.ticker,
                    timestamp=result.timestamp.isoformat(),
                    current_price=result.current_price,
                    up_probability=result.up_probability,
                    down_probability=result.down_probability,
                    best_up_model=result.best_up_model,
                    best_down_model=result.best_down_model,
                    up_model_accuracy=result.up_model_accuracy,
                    down_model_accuracy=result.down_model_accuracy,
                    trading_signal=result.get_trading_signal(),
                    confidence_level=result.get_confidence_level(),
                    all_model_predictions=result.all_model_predictions,
                    market_context=result.market_context,
                )
            else:
                failed_tickers.append(ticker)

        return BatchPredictionResponse(
            predictions=predictions,
            total_requested=len(tickers),
            total_successful=len(predictions),
            failed_tickers=failed_tickers,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to generate batch predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.get("/top/opportunities", response_model=TopOpportunitiesResponse)
async def get_top_opportunities(
    tickers: List[str] = Query(..., description="List of ticker symbols to evaluate"),
    direction: str = Query("up", regex="^(up|down)$", description="Direction: up or down"),
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum probability threshold"),
    top_n: int = Query(10, ge=1, le=50, description="Number of top opportunities to return"),
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
    settings: Settings = Depends(get_settings),
) -> TopOpportunitiesResponse:
    """
    Get top trading opportunities from a list of tickers.

    Evaluates all provided tickers and returns the top N opportunities
    sorted by probability for the specified direction.

    Args:
        tickers: List of ticker symbols to evaluate
        direction: Direction to evaluate (up or down)
        min_probability: Minimum probability threshold (defaults to settings)
        top_n: Number of top opportunities to return
        predictor: RealtimePredictor instance
        settings: Application settings

    Returns:
        TopOpportunitiesResponse with top opportunities
    """
    try:
        # Normalize ticker symbols
        tickers = [t.upper() for t in tickers]
        logger.info(f"Finding top {top_n} {direction} opportunities from {len(tickers)} tickers")

        # Get top opportunities
        opportunities = predictor.get_top_opportunities(
            tickers=tickers,
            direction=direction,
            min_probability=min_probability or settings.PROBABILITY_THRESHOLD,
            top_n=top_n,
        )

        # Convert to response format
        opportunity_responses = []
        for ticker, result in opportunities:
            opportunity_responses.append(
                PredictionResponse(
                    ticker=result.ticker,
                    timestamp=result.timestamp.isoformat(),
                    current_price=result.current_price,
                    up_probability=result.up_probability,
                    down_probability=result.down_probability,
                    best_up_model=result.best_up_model,
                    best_down_model=result.best_down_model,
                    up_model_accuracy=result.up_model_accuracy,
                    down_model_accuracy=result.down_model_accuracy,
                    trading_signal=result.get_trading_signal(),
                    confidence_level=result.get_confidence_level(),
                )
            )

        return TopOpportunitiesResponse(
            direction=direction,
            opportunities=opportunity_responses,
            total_signals=len(opportunities),
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get top opportunities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get opportunities: {str(e)}",
        )
