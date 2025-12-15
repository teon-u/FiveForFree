"""Model performance endpoints."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from loguru import logger

from src.api.dependencies import get_model_manager, get_realtime_predictor
from src.models.model_manager import ModelManager
from src.predictor.realtime_predictor import RealtimePredictor
from config.settings import settings


router = APIRouter(
    prefix="/api/models",
    tags=["models"],
)


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""

    model_type: str = Field(..., description="Model type (xgboost, lightgbm, lstm, transformer, ensemble)")
    target: str = Field(..., description="Prediction target (up or down)")
    hit_rate_50h: float = Field(..., description="50-hour hit rate percentage")
    total_predictions: int = Field(..., description="Total predictions made")
    is_trained: bool = Field(..., description="Whether model is trained")
    last_trained: Optional[str] = Field(None, description="Last training timestamp")


class ModelComparisonResponse(BaseModel):
    """Model comparison response for a ticker."""

    ticker: str = Field(..., description="Ticker symbol")
    up_models: List[ModelPerformanceMetrics] = Field(..., description="Models for up prediction")
    down_models: List[ModelPerformanceMetrics] = Field(..., description="Models for down prediction")
    best_up_model: Optional[str] = Field(None, description="Best performing up model")
    best_down_model: Optional[str] = Field(None, description="Best performing down model")
    timestamp: str = Field(..., description="Response timestamp")


class AllModelsPerformanceResponse(BaseModel):
    """Performance metrics for all models across all tickers."""

    ticker_performances: Dict[str, ModelComparisonResponse] = Field(
        ..., description="Performance by ticker"
    )
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    timestamp: str = Field(..., description="Response timestamp")


class ModelPredictionStats(BaseModel):
    """Prediction statistics for a model."""

    ticker: str = Field(..., description="Ticker symbol")
    model_type: str = Field(..., description="Model type")
    target: str = Field(..., description="Prediction target")
    statistics: Dict[str, Any] = Field(..., description="Detailed statistics")


@router.get("/{ticker}", response_model=ModelComparisonResponse)
async def get_model_performance(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelComparisonResponse:
    """
    Get model performance comparison for a specific ticker.

    Returns performance metrics for all models (up and down) trained on this ticker,
    showing 50-hour hit rates and identifying the best performing model.

    Args:
        ticker: Ticker symbol
        model_manager: Model manager instance

    Returns:
        ModelComparisonResponse with performance metrics for all models

    Raises:
        HTTPException: If ticker not found or no models exist
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting model performance for {ticker}")

        # Get performance metrics for all models
        performances = model_manager.get_model_performances(ticker)

        if not performances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No models found for ticker {ticker}",
            )

        # Build response for up models
        up_models = []
        if "up" in performances:
            for model_type, metrics in performances["up"].items():
                up_models.append(
                    ModelPerformanceMetrics(
                        model_type=model_type,
                        target="up",
                        hit_rate_50h=metrics["hit_rate_50h"],
                        total_predictions=metrics["total_predictions"],
                        is_trained=metrics["is_trained"],
                        last_trained=metrics["last_trained"],
                    )
                )

        # Build response for down models
        down_models = []
        if "down" in performances:
            for model_type, metrics in performances["down"].items():
                down_models.append(
                    ModelPerformanceMetrics(
                        model_type=model_type,
                        target="down",
                        hit_rate_50h=metrics["hit_rate_50h"],
                        total_predictions=metrics["total_predictions"],
                        is_trained=metrics["is_trained"],
                        last_trained=metrics["last_trained"],
                    )
                )

        # Get best models
        best_up_model = None
        best_down_model = None

        try:
            best_up_type, _ = model_manager.get_best_model(ticker, "up")
            best_up_model = best_up_type
        except ValueError:
            pass

        try:
            best_down_type, _ = model_manager.get_best_model(ticker, "down")
            best_down_model = best_down_type
        except ValueError:
            pass

        # Sort models by hit rate (descending)
        up_models.sort(key=lambda x: x.hit_rate_50h, reverse=True)
        down_models.sort(key=lambda x: x.hit_rate_50h, reverse=True)

        return ModelComparisonResponse(
            ticker=ticker,
            up_models=up_models,
            down_models=down_models,
            best_up_model=best_up_model,
            best_down_model=best_down_model,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model performance: {str(e)}",
        )


@router.get("", response_model=AllModelsPerformanceResponse)
@router.get("/", response_model=AllModelsPerformanceResponse)
async def get_all_models_performance(
    model_manager: ModelManager = Depends(get_model_manager),
) -> AllModelsPerformanceResponse:
    """
    Get performance metrics for all models across all tickers.

    Returns comprehensive performance data for every ticker and model,
    along with summary statistics.

    Args:
        model_manager: Model manager instance

    Returns:
        AllModelsPerformanceResponse with all performance data
    """
    try:
        logger.info("Getting performance for all models")

        # Get all tickers with models
        summary = model_manager.get_summary()
        tickers = summary["tickers"]

        # Collect performance for each ticker
        ticker_performances = {}

        for ticker in tickers:
            try:
                performances = model_manager.get_model_performances(ticker)

                # Build up models
                up_models = []
                if "up" in performances:
                    for model_type, metrics in performances["up"].items():
                        up_models.append(
                            ModelPerformanceMetrics(
                                model_type=model_type,
                                target="up",
                                hit_rate_50h=metrics["hit_rate_50h"],
                                total_predictions=metrics["total_predictions"],
                                is_trained=metrics["is_trained"],
                                last_trained=metrics["last_trained"],
                            )
                        )

                # Build down models
                down_models = []
                if "down" in performances:
                    for model_type, metrics in performances["down"].items():
                        down_models.append(
                            ModelPerformanceMetrics(
                                model_type=model_type,
                                target="down",
                                hit_rate_50h=metrics["hit_rate_50h"],
                                total_predictions=metrics["total_predictions"],
                                is_trained=metrics["is_trained"],
                                last_trained=metrics["last_trained"],
                            )
                        )

                # Get best models
                best_up_model = None
                best_down_model = None

                try:
                    best_up_type, _ = model_manager.get_best_model(ticker, "up")
                    best_up_model = best_up_type
                except ValueError:
                    pass

                try:
                    best_down_type, _ = model_manager.get_best_model(ticker, "down")
                    best_down_model = best_down_type
                except ValueError:
                    pass

                # Sort models by hit rate
                up_models.sort(key=lambda x: x.hit_rate_50h, reverse=True)
                down_models.sort(key=lambda x: x.hit_rate_50h, reverse=True)

                ticker_performances[ticker] = ModelComparisonResponse(
                    ticker=ticker,
                    up_models=up_models,
                    down_models=down_models,
                    best_up_model=best_up_model,
                    best_down_model=best_down_model,
                    timestamp=datetime.utcnow().isoformat(),
                )

            except Exception as e:
                logger.error(f"Failed to get performance for {ticker}: {e}")
                # Continue with other tickers

        return AllModelsPerformanceResponse(
            ticker_performances=ticker_performances,
            summary={
                "total_tickers": summary["total_tickers"],
                "total_models": summary["total_models"],
                "trained_models": summary["trained_models"],
                "untrained_models": summary["untrained_models"],
            },
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get all models performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models performance: {str(e)}",
        )


@router.get("/{ticker}/stats", response_model=ModelPredictionStats)
async def get_model_prediction_stats(
    ticker: str,
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
) -> ModelPredictionStats:
    """
    Get detailed prediction statistics for a ticker.

    Returns comprehensive statistics about prediction accuracy, confidence,
    and performance over different time periods.

    Args:
        ticker: Ticker symbol
        predictor: RealtimePredictor instance

    Returns:
        ModelPredictionStats with detailed statistics

    Raises:
        HTTPException: If ticker not found
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting prediction stats for {ticker}")

        # Get stats from predictor
        stats = predictor.get_prediction_stats(ticker)

        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No statistics available for {ticker}",
            )

        return ModelPredictionStats(
            ticker=ticker,
            model_type="ensemble",  # Or get from best model
            target="both",
            statistics=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction stats for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction statistics: {str(e)}",
        )


@router.get("/summary/overall", response_model=Dict[str, Any])
async def get_overall_summary(
    model_manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Get overall summary of model manager status.

    Returns high-level statistics about all models, tickers, and performance.

    Args:
        model_manager: Model manager instance

    Returns:
        Dictionary with overall summary
    """
    try:
        summary = model_manager.get_summary()

        # Calculate average performance across all models
        total_tickers = summary["tickers"]
        total_trained = 0
        avg_hit_rate = 0.0
        hit_rate_count = 0

        for ticker in total_tickers:
            try:
                performances = model_manager.get_model_performances(ticker)

                for target in ["up", "down"]:
                    if target in performances:
                        for model_type, metrics in performances[target].items():
                            if metrics["is_trained"]:
                                total_trained += 1
                                avg_hit_rate += metrics["hit_rate_50h"]
                                hit_rate_count += 1
            except Exception as e:
                logger.warning(f"Failed to get performance for {ticker}: {e}")

        if hit_rate_count > 0:
            avg_hit_rate = avg_hit_rate / hit_rate_count
        else:
            avg_hit_rate = 0.0

        return {
            "total_tickers": summary["total_tickers"],
            "total_models": summary["total_models"],
            "trained_models": summary["trained_models"],
            "untrained_models": summary["untrained_models"],
            "average_hit_rate_50h": round(avg_hit_rate, 2),
            "models_path": summary["models_path"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get overall summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary: {str(e)}",
        )
