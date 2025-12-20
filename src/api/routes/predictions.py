"""Prediction endpoints."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from loguru import logger

from src.api.dependencies import get_realtime_predictor, get_settings
from src.predictor.realtime_predictor import RealtimePredictor
from src.collector.ticker_selector import TickerSelector
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


class SimplePrediction(BaseModel):
    """Simplified prediction for frontend display."""

    ticker: str = Field(..., description="Ticker symbol")
    probability: float = Field(..., description="Prediction probability (0-1)")
    direction: str = Field(..., description="Prediction direction: up or down")
    change_percent: float = Field(0.0, description="Current price change percent")
    best_model: str = Field(..., description="Best performing model")
    hit_rate: float = Field(..., description="Model accuracy (0-1)")
    current_price: float = Field(..., description="Current stock price")
    signal_rate: float = Field(0.0, description="Signal rate (0-1)")
    practicality_grade: str = Field("D", description="Practicality grade (A/B/C/D)")


class CategorizedPredictionsResponse(BaseModel):
    """Predictions categorized by volume and gainers."""

    volume_top_100: List[SimplePrediction] = Field(..., description="Top tickers by volume")
    gainers_top_100: List[SimplePrediction] = Field(..., description="Top tickers by price gain")
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


@router.get("", response_model=CategorizedPredictionsResponse)
@router.get("/", response_model=CategorizedPredictionsResponse)
async def get_categorized_predictions(
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum probability threshold"),
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
    settings: Settings = Depends(get_settings),
) -> CategorizedPredictionsResponse:
    """
    Get predictions for trained tickers from database.

    Returns predictions for tickers that have trained models.
    Uses cached database tickers instead of live Yahoo Finance data for speed.

    Args:
        threshold: Minimum probability threshold (optional filter)
        predictor: RealtimePredictor instance
        settings: Application settings

    Returns:
        CategorizedPredictionsResponse with predictions
    """
    try:
        logger.info("Fetching predictions for trained tickers")

        # Use trained tickers from model manager
        from src.api.dependencies import get_model_manager
        model_manager = get_model_manager()
        trained_tickers = model_manager.get_tickers()

        if not trained_tickers:
            logger.warning("No trained tickers available")
            return CategorizedPredictionsResponse(
                volume_top_100=[],
                gainers_top_100=[],
                timestamp=datetime.utcnow().isoformat(),
            )

        # Get real-time metrics using batch fetch for performance
        import yfinance as yf
        ticker_metrics_map = {}

        # Batch fetch data for all trained tickers (much faster than individual calls)
        tickers_to_fetch = trained_tickers[:100]
        try:
            # Use yfinance download for batch fetching - gets today's data
            data = yf.download(
                tickers=tickers_to_fetch,
                period="2d",  # Get 2 days to calculate change
                interval="1d",
                progress=False,
                group_by='ticker',
                threads=True
            )

            if not data.empty:
                for ticker in tickers_to_fetch:
                    try:
                        if len(tickers_to_fetch) > 1:
                            ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None
                        else:
                            ticker_data = data

                        if ticker_data is not None and not ticker_data.empty and len(ticker_data) >= 2:
                            prev_close = ticker_data['Close'].iloc[-2]
                            current_close = ticker_data['Close'].iloc[-1]
                            # Check for NaN values before calculation
                            import math
                            if prev_close > 0 and not math.isnan(prev_close) and not math.isnan(current_close):
                                change_percent = ((current_close - prev_close) / prev_close) * 100
                                # Ensure result is not NaN
                                if not math.isnan(change_percent):
                                    class BatchTickerMetrics:
                                        def __init__(self, t, cp):
                                            self.ticker = t
                                            self.change_percent = float(cp)  # Ensure float conversion

                                    ticker_metrics_map[ticker] = BatchTickerMetrics(ticker, change_percent)
                    except Exception as e:
                        logger.debug(f"Error processing {ticker}: {e}")
                        continue

            logger.info(f"Batch fetched metrics for {len(ticker_metrics_map)} tickers")
        except Exception as e:
            logger.warning(f"Batch fetch failed: {e}")

        # Create wrapper class for tickers without real-time data
        class TickerMetricsWrapper:
            def __init__(self, ticker, change_percent=0.0):
                self.ticker = ticker
                self.change_percent = change_percent

        # Build categories using real metrics when available
        volume_tickers = []
        gainers_tickers = []

        for ticker in trained_tickers[:50]:
            if ticker in ticker_metrics_map:
                volume_tickers.append(ticker_metrics_map[ticker])
            else:
                volume_tickers.append(TickerMetricsWrapper(ticker))

        # Sort gainers by change_percent descending
        tickers_with_metrics = [(t, m) for t, m in ticker_metrics_map.items() if t in trained_tickers]
        tickers_with_metrics.sort(key=lambda x: x[1].change_percent, reverse=True)

        for ticker, metrics in tickers_with_metrics[:50]:
            gainers_tickers.append(metrics)

        # Fill remaining slots if needed
        for ticker in trained_tickers[:50]:
            if len(gainers_tickers) >= 50:
                break
            if ticker not in ticker_metrics_map:
                gainers_tickers.append(TickerMetricsWrapper(ticker))

        categories = {
            'volume': volume_tickers,
            'gainers': gainers_tickers,
        }

        if not categories:
            logger.warning("No tickers available from ticker selector")
            return CategorizedPredictionsResponse(
                volume_top_100=[],
                gainers_top_100=[],
                timestamp=datetime.utcnow().isoformat(),
            )

        # Helper function to create simplified prediction
        def create_simple_prediction(ticker_metrics, result) -> SimplePrediction:
            # Determine primary direction and probability
            if result.up_probability >= result.down_probability:
                direction = "up"
                probability = result.up_probability
                best_model = result.best_up_model
                hit_rate = result.up_model_accuracy
            else:
                direction = "down"
                probability = result.down_probability
                best_model = result.best_down_model
                hit_rate = result.down_model_accuracy

            # Get signal_rate and practicality_grade from model performance
            signal_rate = 0.0
            practicality_grade = "D"
            try:
                performances = model_manager.get_model_performances(result.ticker)
                if direction in performances and best_model in performances[direction]:
                    model_perf = performances[direction][best_model]
                    signal_rate = model_perf.get('signal_rate', 0.0)
                    # Use precision from model_perf for consistency
                    precision = model_perf.get('precision', 0.0)
                    # Recalculate grade based on actual precision and signal_rate
                    # to ensure consistency (A: prec>=50% & sig>=10%, B: prec>=30% & sig>=10%, C: prec>=30%, D: else)
                    if precision >= 0.50 and signal_rate >= 0.10:
                        practicality_grade = 'A'
                    elif precision >= 0.30 and signal_rate >= 0.10:
                        practicality_grade = 'B'
                    elif precision >= 0.30:
                        practicality_grade = 'C'
                    else:
                        practicality_grade = 'D'
                    # Also update hit_rate to match precision from same source
                    hit_rate = precision
            except Exception:
                pass  # Use defaults if unable to get performance data

            return SimplePrediction(
                ticker=result.ticker,
                probability=probability,
                direction=direction,
                change_percent=ticker_metrics.change_percent,
                best_model=best_model,
                hit_rate=hit_rate,
                current_price=result.current_price,
                signal_rate=signal_rate,
                practicality_grade=practicality_grade,
            )

        # Process volume top tickers
        volume_predictions = []
        for ticker_metrics in categories['volume']:
            try:
                result = predictor.predict(
                    ticker=ticker_metrics.ticker,
                    include_all_models=False,
                    include_features=False,
                )

                simple_pred = create_simple_prediction(ticker_metrics, result)

                # Apply threshold filter if specified
                if threshold is None or simple_pred.probability >= threshold:
                    volume_predictions.append(simple_pred)

            except Exception as e:
                logger.warning(f"Failed to predict for {ticker_metrics.ticker}: {e}")
                continue

        # Process gainers top tickers
        gainers_predictions = []
        for ticker_metrics in categories['gainers']:
            try:
                result = predictor.predict(
                    ticker=ticker_metrics.ticker,
                    include_all_models=False,
                    include_features=False,
                )

                simple_pred = create_simple_prediction(ticker_metrics, result)

                # Apply threshold filter if specified
                if threshold is None or simple_pred.probability >= threshold:
                    gainers_predictions.append(simple_pred)

            except Exception as e:
                logger.warning(f"Failed to predict for {ticker_metrics.ticker}: {e}")
                continue

        logger.info(
            f"Generated predictions: {len(volume_predictions)} volume, "
            f"{len(gainers_predictions)} gainers"
        )

        return CategorizedPredictionsResponse(
            volume_top_100=volume_predictions,
            gainers_top_100=gainers_predictions,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get categorized predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}",
        )
