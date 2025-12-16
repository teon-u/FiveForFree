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


# New endpoints for detailed model analysis


class ModelOverviewResponse(BaseModel):
    """Overview response for model detail page."""

    ticker: str
    prediction: Dict[str, Any]
    ranking: List[Dict[str, Any]]
    quick_stats: Dict[str, Any]
    risk_indicators: Dict[str, Any]
    timestamp: str


class PerformanceMetricsResponse(BaseModel):
    """Detailed performance metrics response."""

    ticker: str
    confusion_matrix: Dict[str, int]
    metrics: Dict[str, float]
    roc_curve: Dict[str, Any]
    pr_curve: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    calibration: Dict[str, Any]
    timestamp: str


@router.get("/{ticker}/overview", response_model=ModelOverviewResponse)
async def get_model_overview(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
    predictor: RealtimePredictor = Depends(get_realtime_predictor),
) -> ModelOverviewResponse:
    """
    Get model overview for investment decision support.

    Provides quick summary of model performance, prediction confidence,
    and risk indicators.

    Args:
        ticker: Ticker symbol
        model_manager: Model manager instance
        predictor: Realtime predictor instance

    Returns:
        ModelOverviewResponse with overview data
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting model overview for {ticker}")

        # Get best models for up and down
        try:
            best_up_type, best_up_model = model_manager.get_best_model(ticker, "up")
            up_stats = best_up_model.get_prediction_stats(hours=50)
        except ValueError:
            best_up_type = None
            up_stats = {'accuracy': 0.0}

        try:
            best_down_type, best_down_model = model_manager.get_best_model(ticker, "down")
            down_stats = best_down_model.get_prediction_stats(hours=50)
        except ValueError:
            best_down_type = None
            down_stats = {'accuracy': 0.0}

        # Determine current prediction
        if up_stats['accuracy'] >= down_stats['accuracy']:
            direction = "up"
            probability = up_stats.get('avg_probability', 0.5)
            best_model = best_up_type or "unknown"
            accuracy = up_stats['accuracy']
        else:
            direction = "down"
            probability = down_stats.get('avg_probability', 0.5)
            best_model = best_down_type or "unknown"
            accuracy = down_stats['accuracy']

        # Determine risk level
        if probability >= 0.80:
            risk_level = "low"
        elif probability >= 0.70:
            risk_level = "moderate"
        else:
            risk_level = "high"

        # Build ranking
        performances = model_manager.get_model_performances(ticker)
        ranking = []

        if "up" in performances:
            for model_type, metrics in performances["up"].items():
                ranking.append({
                    "model": model_type,
                    "hit_rate": metrics["hit_rate_50h"] / 100.0
                })

        ranking.sort(key=lambda x: x["hit_rate"], reverse=True)

        # Calculate model agreement
        model_agreement = 0.0
        if len(ranking) > 1:
            top_hit_rate = ranking[0]["hit_rate"]
            agreeing_models = sum(1 for m in ranking if abs(m["hit_rate"] - top_hit_rate) < 0.05)
            model_agreement = agreeing_models / len(ranking)

        # Calculate false positive rate
        fp_rate = up_stats.get('false_positives', 0) / max(1, up_stats.get('total_predictions', 1))

        # Performance trend (compare recent 10h vs 50h)
        try:
            recent_acc = best_up_model.get_recent_accuracy(hours=10) if best_up_model else 0.0
            full_acc = accuracy
            if recent_acc > full_acc * 1.05:
                trend = "improving"
            elif recent_acc < full_acc * 0.95:
                trend = "declining"
            else:
                trend = "stable"
        except Exception:
            trend = "unknown"

        # Get actual backtest metrics for avg_return and sharpe
        avg_return = 0.0
        sharpe_ratio = 0.0
        try:
            if best_up_model and best_up_model.is_trained:
                backtest_results = best_up_model.get_backtest_results(hours=settings.BACKTEST_HOURS)
                if backtest_results and 'metrics' in backtest_results:
                    metrics = backtest_results['metrics']
                    avg_return = metrics.get('avg_return', 0.0)
                    sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        except Exception as e:
            logger.warning(f"Failed to get backtest metrics for {ticker}: {e}")

        return ModelOverviewResponse(
            ticker=ticker,
            prediction={
                "direction": direction,
                "probability": round(probability, 3),
                "best_model": best_model,
                "expected_change": 5.0,  # Target is always 5%
                "risk_level": risk_level
            },
            ranking=ranking[:5],  # Top 5
            quick_stats={
                "accuracy": round(accuracy, 3),
                "win_rate": round(up_stats.get('recall', 0.0), 3),
                "avg_return": round(avg_return, 4),
                "sharpe": round(sharpe_ratio, 2)
            },
            risk_indicators={
                "false_positive_rate": round(fp_rate, 3),
                "model_agreement": round(model_agreement, 3),
                "performance_trend": trend
            },
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to get model overview for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model overview: {str(e)}",
        )


@router.get("/{ticker}/performance", response_model=PerformanceMetricsResponse)
async def get_model_performance_details(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
) -> PerformanceMetricsResponse:
    """
    Get detailed performance metrics including confusion matrix, ROC curve, etc.

    Args:
        ticker: Ticker symbol
        model_manager: Model manager instance

    Returns:
        PerformanceMetricsResponse with detailed metrics
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting detailed performance for {ticker}")

        # Get best model (up direction for now)
        try:
            best_type, best_model = model_manager.get_best_model(ticker, "up")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained models found for ticker {ticker}",
            )

        # Get confusion matrix and basic stats
        stats = best_model.get_prediction_stats(hours=50)

        # Get ROC curve
        roc_data = best_model.get_roc_curve_data(hours=50)

        # Get PR curve
        pr_data = best_model.get_precision_recall_curve(hours=50)

        # Get performance over time
        time_series = best_model.get_performance_over_time(hours=50, window_hours=5)

        # Get calibration curve
        calibration = best_model.get_calibration_curve(hours=50, n_bins=10)

        return PerformanceMetricsResponse(
            ticker=ticker,
            confusion_matrix={
                "tp": stats['true_positives'],
                "fp": stats['false_positives'],
                "tn": stats['true_negatives'],
                "fn": stats['false_negatives']
            },
            metrics={
                "accuracy": round(stats['accuracy'], 4),
                "precision": round(stats['precision'], 4),
                "recall": round(stats['recall'], 4),
                "f1_score": round(2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0.0, 4),
                "fp_rate": round(stats['false_positives'] / max(1, stats['false_positives'] + stats['true_negatives']), 4),
                "fn_rate": round(stats['false_negatives'] / max(1, stats['true_positives'] + stats['false_negatives']), 4)
            },
            roc_curve=roc_data,
            pr_curve=pr_data,
            time_series=time_series,
            calibration=calibration,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance details for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance details: {str(e)}",
        )


class EnsembleAnalysisResponse(BaseModel):
    """Ensemble analysis response with base models breakdown."""

    ticker: str
    meta_learner: Dict[str, Any]
    base_models: List[Dict[str, Any]]
    current_agreement: Dict[str, Any]
    ensemble_vs_base: Dict[str, Any]
    timestamp: str


@router.get("/{ticker}/ensemble", response_model=EnsembleAnalysisResponse)
async def get_ensemble_analysis(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
) -> EnsembleAnalysisResponse:
    """
    Get ensemble model analysis including base models comparison.

    Args:
        ticker: Ticker symbol
        model_manager: Model manager instance

    Returns:
        EnsembleAnalysisResponse with ensemble breakdown
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting ensemble analysis for {ticker}")

        # Get ensemble model
        try:
            ensemble_type, ensemble_model = model_manager.get_or_create_model(ticker, "ensemble", "up")
        except Exception as e:
            logger.error(f"Failed to get ensemble model: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ensemble model found for ticker {ticker}",
            )

        # Check if it's an ensemble model
        if not hasattr(ensemble_model, 'get_base_model_weights'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model is not an ensemble model",
            )

        # Get meta learner info
        weights = ensemble_model.get_base_model_weights()
        ensemble_stats = ensemble_model.get_ensemble_stats()

        meta_learner = {
            "type": ensemble_stats.get("meta_learner_type", "LogisticRegression"),
            "weights": weights,
            "trained_base_models": ensemble_stats.get("trained_base_models", 0),
            "total_base_models": ensemble_stats.get("total_base_models", 0)
        }

        # Get base models performance
        base_models = []
        model_types = ["xgboost", "lightgbm", "lstm", "transformer"]

        for model_type in model_types:
            try:
                _, base_model = model_manager.get_or_create_model(ticker, model_type, "up")

                if base_model.is_trained:
                    stats = base_model.get_prediction_stats(hours=50)

                    base_models.append({
                        "type": model_type,
                        "accuracy": round(stats['accuracy'], 4),
                        "precision": round(stats['precision'], 4),
                        "recall": round(stats['recall'], 4),
                        "f1_score": round(2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0.0, 4),
                        "total_predictions": stats['total_predictions'],
                        "is_trained": True
                    })
                else:
                    base_models.append({
                        "type": model_type,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "total_predictions": 0,
                        "is_trained": False
                    })
            except Exception as e:
                logger.warning(f"Failed to get {model_type} model: {e}")
                base_models.append({
                    "type": model_type,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "total_predictions": 0,
                    "is_trained": False
                })

        # Calculate current agreement using actual prediction data
        ensemble_accuracy = ensemble_model.get_recent_accuracy(hours=50)

        # Determine ensemble direction based on up vs down model performance
        try:
            _, up_ensemble = model_manager.get_or_create_model(ticker, "ensemble", "up")
            _, down_ensemble = model_manager.get_or_create_model(ticker, "ensemble", "down")
            up_acc = up_ensemble.get_recent_accuracy(hours=50) if up_ensemble.is_trained else 0.0
            down_acc = down_ensemble.get_recent_accuracy(hours=50) if down_ensemble.is_trained else 0.0
            ensemble_direction = "up" if up_acc >= down_acc else "down"
            ensemble_prob = max(up_acc, down_acc)
        except Exception:
            ensemble_direction = "up"
            ensemble_prob = ensemble_accuracy

        base_predictions = []
        for model_info in base_models:
            if model_info['is_trained']:
                # Determine direction based on model accuracy (higher = more confident in up)
                model_direction = "up" if model_info['accuracy'] >= 0.5 else "down"
                base_predictions.append({
                    "model": model_info['type'],
                    "direction": model_direction,
                    "probability": model_info['accuracy'],
                })

        # Calculate agreement rate based on direction consensus
        if len(base_predictions) > 1:
            up_votes = sum(1 for p in base_predictions if p['direction'] == 'up')
            down_votes = len(base_predictions) - up_votes
            # Agreement is the proportion of models agreeing with majority
            majority_votes = max(up_votes, down_votes)
            agreement_rate = majority_votes / len(base_predictions)
            # Also consider accuracy variance
            accuracies = [p['probability'] for p in base_predictions]
            avg_accuracy = sum(accuracies) / len(accuracies)
            variance = sum((acc - avg_accuracy) ** 2 for acc in accuracies) / len(accuracies)
        else:
            agreement_rate = 1.0 if len(base_predictions) == 1 else 0.0
            variance = 0.0

        current_agreement = {
            "ensemble_prediction": {
                "direction": ensemble_direction,
                "probability": ensemble_prob
            },
            "base_predictions": base_predictions,
            "agreement_rate": round(agreement_rate, 3),
            "variance": round(variance if len(base_predictions) > 1 else 0.0, 4)
        }

        # Get actual ensemble metrics (precision, recall, f1)
        ensemble_precision = 0.0
        ensemble_recall = 0.0
        ensemble_f1 = 0.0
        try:
            if ensemble_model.is_trained:
                ensemble_stats = ensemble_model.get_prediction_stats(hours=50)
                ensemble_precision = ensemble_stats.get('precision', 0.0)
                ensemble_recall = ensemble_stats.get('recall', 0.0)
                # Calculate F1 score
                if ensemble_precision + ensemble_recall > 0:
                    ensemble_f1 = 2 * ensemble_precision * ensemble_recall / (ensemble_precision + ensemble_recall)
        except Exception as e:
            logger.warning(f"Failed to get ensemble metrics: {e}")

        # Compare ensemble vs individual base models
        ensemble_vs_base = {
            "ensemble_accuracy": round(ensemble_accuracy, 4),
            "ensemble_precision": round(ensemble_precision, 4),
            "ensemble_recall": round(ensemble_recall, 4),
            "ensemble_f1": round(ensemble_f1, 4),
            "best_base_accuracy": round(max([m['accuracy'] for m in base_models if m['is_trained']], default=0.0), 4),
            "avg_base_accuracy": round(sum([m['accuracy'] for m in base_models if m['is_trained']]) / len([m for m in base_models if m['is_trained']]) if any(m['is_trained'] for m in base_models) else 0.0, 4),
            "improvement": round(ensemble_accuracy - max([m['accuracy'] for m in base_models if m['is_trained']], default=0.0), 4)
        }

        return EnsembleAnalysisResponse(
            ticker=ticker,
            meta_learner=meta_learner,
            base_models=base_models,
            current_agreement=current_agreement,
            ensemble_vs_base=ensemble_vs_base,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ensemble analysis for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ensemble analysis: {str(e)}",
        )


class FinancialAnalysisResponse(BaseModel):
    """Financial performance and backtest results."""

    ticker: str
    equity_curve: List[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    trade_metrics: Dict[str, Any]
    trade_distribution: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]
    timestamp: str


@router.get("/{ticker}/financial", response_model=FinancialAnalysisResponse)
async def get_financial_analysis(
    ticker: str,
    hours: int = Query(50, description="Hours to look back for backtest"),
    model_manager: ModelManager = Depends(get_model_manager),
) -> FinancialAnalysisResponse:
    """
    Get financial performance and backtest results for a ticker.

    Includes equity curve, risk metrics (Sharpe, Sortino, Max Drawdown),
    trade statistics, and recent trade history.

    Args:
        ticker: Ticker symbol
        hours: Hours to look back (default 50)
        model_manager: Model manager instance

    Returns:
        FinancialAnalysisResponse with backtest results and metrics

    Raises:
        HTTPException: If ticker not found or no models exist
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Getting financial analysis for {ticker} (last {hours}h)")

        # Get best model for "up" direction (primary trading direction)
        try:
            best_model_type, _ = model_manager.get_best_model(ticker, "up")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained models found for {ticker}",
            )

        # Get the best model instance
        _, best_model = model_manager.get_or_create_model(ticker, best_model_type, "up")

        if not best_model.is_trained:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model not trained for {ticker}",
            )

        # Get backtest results
        backtest = best_model.get_backtest_results(hours=hours)

        # Extract metrics
        metrics = backtest['metrics']

        # Split metrics into risk and trade categories
        risk_metrics = {
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'calmar_ratio': metrics['calmar_ratio']
        }

        trade_metrics = {
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'total_return': metrics['total_return'],
            'avg_return': metrics['avg_return'],
            'avg_win': metrics['avg_win'],
            'avg_loss': metrics['avg_loss'],
            'best_trade': metrics['best_trade'],
            'worst_trade': metrics['worst_trade'],
            'profit_factor': metrics['profit_factor']
        }

        return FinancialAnalysisResponse(
            ticker=ticker,
            equity_curve=backtest['equity_curve'],
            risk_metrics=risk_metrics,
            trade_metrics=trade_metrics,
            trade_distribution=backtest['trade_distribution'],
            recent_trades=backtest['recent_trades'],
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get financial analysis for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve financial analysis: {str(e)}",
        )
