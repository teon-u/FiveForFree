"""
Real-time Prediction System for NASDAQ Trading

Provides real-time predictions for active trading:
- Collects latest market data via Polygon.io
- Computes 57 features in real-time
- Runs predictions using best-performing models
- Returns probabilities optimized for UI display

Hybrid-Ensemble Approach:
- Structure A: Direct up/down probability prediction
- Structure B: Volatility × Direction probability prediction
- Final: Weighted ensemble of Structure A and B predictions
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd
from loguru import logger

from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.collector.minute_bars import MinuteBarCollector
from src.collector.quotes import QuoteCollector
from src.collector.market_context import MarketContextCollector
from config.settings import settings


@dataclass
class HybridPredictionDetail:
    """
    Detailed breakdown of hybrid-ensemble prediction.

    Attributes:
        direct_up_prob: Structure A direct up probability
        direct_down_prob: Structure A direct down probability
        volatility_prob: Structure B volatility probability
        direction_up_prob: Structure B direction (up) probability
        hybrid_up_prob: Structure B combined up probability (vol × dir)
        hybrid_down_prob: Structure B combined down probability (vol × (1-dir))
        ensemble_alpha: Weight used for ensemble (α for direct, 1-α for hybrid)
    """
    direct_up_prob: float
    direct_down_prob: float
    volatility_prob: float
    direction_up_prob: float
    hybrid_up_prob: float
    hybrid_down_prob: float
    ensemble_alpha: float

    def to_dict(self) -> Dict[str, float]:
        def sanitize_value(val):
            """Replace NaN/Inf with None for JSON compatibility."""
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val

        return {
            'direct_up_prob': sanitize_value(self.direct_up_prob),
            'direct_down_prob': sanitize_value(self.direct_down_prob),
            'volatility_prob': sanitize_value(self.volatility_prob),
            'direction_up_prob': sanitize_value(self.direction_up_prob),
            'hybrid_up_prob': sanitize_value(self.hybrid_up_prob),
            'hybrid_down_prob': sanitize_value(self.hybrid_down_prob),
            'ensemble_alpha': sanitize_value(self.ensemble_alpha)
        }


@dataclass
class PredictionResult:
    """
    Result of a real-time prediction.

    Attributes:
        ticker: Stock ticker symbol
        timestamp: When prediction was made
        current_price: Current stock price
        up_probability: Final ensemble probability of 5%+ upward movement
        down_probability: Final ensemble probability of 5%+ downward movement
        best_up_model: Best performing model for upward prediction
        best_down_model: Best performing model for downward prediction
        up_model_accuracy: 50-hour accuracy of up model
        down_model_accuracy: 50-hour accuracy of down model
        hybrid_detail: Detailed hybrid-ensemble breakdown (optional)
        all_model_predictions: Predictions from all models (optional)
        features: Computed features (optional)
        market_context: Market context data (optional)
    """
    ticker: str
    timestamp: datetime
    current_price: float
    up_probability: float
    down_probability: float
    best_up_model: str
    best_down_model: str
    up_model_accuracy: float
    down_model_accuracy: float
    hybrid_detail: Optional[HybridPredictionDetail] = None
    all_model_predictions: Optional[Dict[str, Dict[str, float]]] = None
    features: Optional[np.ndarray] = None
    market_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def sanitize_value(val):
            """Replace NaN/Inf with None for JSON compatibility."""
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val

        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'current_price': sanitize_value(self.current_price),
            'up_probability': sanitize_value(self.up_probability),
            'down_probability': sanitize_value(self.down_probability),
            'best_up_model': self.best_up_model,
            'best_down_model': self.best_down_model,
            'up_model_accuracy': sanitize_value(self.up_model_accuracy),
            'down_model_accuracy': sanitize_value(self.down_model_accuracy),
            'hybrid_detail': self.hybrid_detail.to_dict() if self.hybrid_detail else None,
            'all_model_predictions': self.all_model_predictions,
            'market_context': self.market_context
        }

    def get_trading_signal(self) -> str:
        """
        Get trading signal based on probabilities and threshold.

        Returns:
            "BUY" if up_probability >= threshold
            "SELL" if down_probability >= threshold
            "HOLD" otherwise
        """
        threshold = settings.PROBABILITY_THRESHOLD

        if self.up_probability >= threshold:
            return "BUY"
        elif self.down_probability >= threshold:
            return "SELL"
        else:
            return "HOLD"

    def get_confidence_level(self) -> str:
        """
        Get confidence level based on probability magnitude.

        Returns:
            "VERY_HIGH" (>= 0.80)
            "HIGH" (>= 0.70)
            "MEDIUM" (>= 0.60)
            "LOW" (< 0.60)
        """
        max_prob = max(self.up_probability, self.down_probability)

        if max_prob >= 0.80:
            return "VERY_HIGH"
        elif max_prob >= 0.70:
            return "HIGH"
        elif max_prob >= 0.60:
            return "MEDIUM"
        else:
            return "LOW"


class RealtimePredictor:
    """
    Real-time prediction engine for live trading.

    Workflow:
    1. Collect latest minute bars (60+ minutes for feature computation)
    2. Collect order book snapshot
    3. Collect market context (SPY, QQQ, VIX, sectors)
    4. Compute 57 features
    5. Run predictions using best models
    6. Return formatted results for UI

    Performance:
    - Predictions complete in < 1 second for single ticker
    - Supports batch predictions for multiple tickers
    - Automatic caching of market context data
    """

    def __init__(
        self,
        model_manager: ModelManager,
        minute_bar_collector: Optional[MinuteBarCollector] = None,
        quote_collector: Optional[QuoteCollector] = None,
        market_context_collector: Optional[MarketContextCollector] = None,
        lookback_minutes: int = 120
    ):
        """
        Initialize real-time predictor.

        Args:
            model_manager: ModelManager instance with trained models
            minute_bar_collector: Collector for minute bar data (optional)
            quote_collector: Collector for order book data (optional)
            market_context_collector: Collector for market context (optional)
            lookback_minutes: Minutes of historical data to collect for features
        """
        self.model_manager = model_manager
        self.minute_bar_collector = minute_bar_collector
        self.quote_collector = quote_collector
        self.market_context_collector = market_context_collector
        self.lookback_minutes = lookback_minutes

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Cache for market context (updated less frequently)
        self._market_context_cache: Optional[Dict] = None
        self._market_context_cache_time: Optional[datetime] = None
        self._market_context_cache_ttl = 60  # seconds

        logger.info("RealtimePredictor initialized")

    def predict(
        self,
        ticker: str,
        include_all_models: bool = False,
        include_features: bool = False,
        use_hybrid: bool = None
    ) -> PredictionResult:
        """
        Generate real-time prediction for a ticker using hybrid-ensemble approach.

        Args:
            ticker: Stock ticker symbol
            include_all_models: Include predictions from all models (not just best)
            include_features: Include computed features in result
            use_hybrid: Use hybrid-ensemble approach (default: from settings)

        Returns:
            PredictionResult with probabilities and metadata

        Raises:
            ValueError: If required data cannot be collected
            RuntimeError: If prediction fails
        """
        logger.debug(f"Generating prediction for {ticker}")
        start_time = datetime.now()

        # Determine if hybrid mode is enabled
        use_hybrid = use_hybrid if use_hybrid is not None else settings.USE_HYBRID_ENSEMBLE

        # Step 1: Collect minute bars
        minute_bars = self._collect_minute_bars(ticker)
        if minute_bars is None or len(minute_bars) < 60:
            raise ValueError(f"Insufficient minute bar data for {ticker}")

        current_price = minute_bars.iloc[-1]['close']

        # Validate price is not NaN or Inf
        if current_price is None or pd.isna(current_price) or math.isinf(float(current_price)):
            logger.warning(f"Invalid current_price for {ticker}: {current_price}")
            raise ValueError(f"Invalid price data (NaN/Inf) for {ticker}")

        # Step 2: Collect order book
        order_book = self._collect_order_book(ticker)

        # Step 3: Get market context
        market_context = self._get_market_context()

        # Step 4: Compute features
        features_df = self.feature_engineer.compute_features(
            minute_bars,
            order_book=order_book,
            market_data=market_context
        )

        # Get feature vector for latest timestamp
        feature_names = self.feature_engineer.get_feature_names()
        X = features_df[feature_names].iloc[-1:].values  # Shape: (1, n_features)

        # For sequence models (LSTM, Transformer), get 60 rows of data
        sequence_length = 60
        if len(features_df) >= sequence_length:
            X_seq = features_df[feature_names].iloc[-sequence_length:].values  # Shape: (60, n_features)
        else:
            # If not enough data, use what we have (models will pad internally)
            X_seq = features_df[feature_names].values

        # Step 5: Run predictions
        try:
            # ===== Structure A: Direct Prediction =====
            best_up_type, best_up_model = self.model_manager.get_best_model(ticker, "up")
            best_down_type, best_down_model = self.model_manager.get_best_model(ticker, "down")

            # Use X_seq for sequence models (LSTM, Transformer), X for tree models
            X_up = X_seq if best_up_type in ('lstm', 'transformer') else X
            X_down = X_seq if best_down_type in ('lstm', 'transformer') else X

            direct_up_prob = best_up_model.predict_proba(X_up)[0]
            direct_down_prob = best_down_model.predict_proba(X_down)[0]

            # Get model precision (more meaningful than overall accuracy)
            # Precision = When model predicts positive, how often is it correct?
            up_accuracy = best_up_model.get_precision_at_threshold(hours=settings.BACKTEST_HOURS, threshold=0.5)
            down_accuracy = best_down_model.get_precision_at_threshold(hours=settings.BACKTEST_HOURS, threshold=0.5)

            # Initialize final probabilities (default to direct predictions)
            final_up_prob = direct_up_prob
            final_down_prob = direct_down_prob
            hybrid_detail = None

            # ===== Structure B: Hybrid Prediction (if enabled) =====
            if use_hybrid:
                hybrid_result = self._predict_hybrid(ticker, X, X_seq)

                if hybrid_result is not None:
                    volatility_prob, direction_up_prob, hybrid_up_prob, hybrid_down_prob = hybrid_result

                    # ===== Ensemble: Combine Structure A and B =====
                    alpha = settings.ENSEMBLE_ALPHA

                    final_up_prob = alpha * direct_up_prob + (1 - alpha) * hybrid_up_prob
                    final_down_prob = alpha * direct_down_prob + (1 - alpha) * hybrid_down_prob

                    # Apply calibration if needed
                    final_up_prob, final_down_prob = self._calibrate_probabilities(
                        final_up_prob, final_down_prob
                    )

                    # Store hybrid detail
                    hybrid_detail = HybridPredictionDetail(
                        direct_up_prob=float(direct_up_prob),
                        direct_down_prob=float(direct_down_prob),
                        volatility_prob=float(volatility_prob),
                        direction_up_prob=float(direction_up_prob),
                        hybrid_up_prob=float(hybrid_up_prob),
                        hybrid_down_prob=float(hybrid_down_prob),
                        ensemble_alpha=alpha
                    )

                    logger.debug(
                        f"Hybrid prediction for {ticker}: "
                        f"Direct(UP={direct_up_prob:.3f}, DOWN={direct_down_prob:.3f}), "
                        f"Hybrid(VOL={volatility_prob:.3f}, DIR={direction_up_prob:.3f}), "
                        f"Final(UP={final_up_prob:.3f}, DOWN={final_down_prob:.3f})"
                    )

            # Create timestamp for prediction recording
            prediction_timestamp = datetime.now()

            # Optionally get predictions from all models (also records predictions)
            all_predictions = None
            if include_all_models:
                all_predictions = self._get_all_model_predictions(
                    ticker, X, X_seq,
                    timestamp=prediction_timestamp,
                    record_predictions=True
                )

            # Create result
            result = PredictionResult(
                ticker=ticker,
                timestamp=prediction_timestamp,
                current_price=float(current_price),
                up_probability=float(final_up_prob),
                down_probability=float(final_down_prob),
                best_up_model=best_up_type,
                best_down_model=best_down_type,
                up_model_accuracy=float(up_accuracy),
                down_model_accuracy=float(down_accuracy),
                hybrid_detail=hybrid_detail,
                all_model_predictions=all_predictions,
                features=X[0] if include_features else None,
                market_context=market_context if include_features else None
            )

            # Log prediction to best models for accuracy tracking
            # (only if not already recorded by _get_all_model_predictions)
            if not include_all_models:
                # Record direct predictions to direct models (not ensemble)
                best_up_model.record_prediction(direct_up_prob, result.timestamp, X)
                best_down_model.record_prediction(direct_down_prob, result.timestamp, X)

                # Record ensemble model predictions (if ensemble model exists)
                try:
                    # Record for "up" target ensemble
                    ensemble_up_models = self.model_manager.get_all_models(ticker, "up")
                    if "ensemble" in ensemble_up_models and ensemble_up_models["ensemble"].is_trained:
                        ensemble_up_model = ensemble_up_models["ensemble"]
                        ensemble_up_model.record_prediction(final_up_prob, result.timestamp, X)

                    # Record for "down" target ensemble
                    ensemble_down_models = self.model_manager.get_all_models(ticker, "down")
                    if "ensemble" in ensemble_down_models and ensemble_down_models["ensemble"].is_trained:
                        ensemble_down_model = ensemble_down_models["ensemble"]
                        ensemble_down_model.record_prediction(final_down_prob, result.timestamp, X)
                except Exception as e:
                    logger.debug(f"Could not record ensemble predictions: {e}")

                # If hybrid mode is enabled, also record hybrid component predictions
                if use_hybrid and hybrid_detail is not None:
                    try:
                        # Record volatility and direction predictions from Structure B
                        vol_type, vol_model = self.model_manager.get_best_model(ticker, "volatility")
                        dir_type, dir_model = self.model_manager.get_best_model(ticker, "direction")

                        vol_model.record_prediction(hybrid_detail.volatility_prob, result.timestamp, X)
                        dir_model.record_prediction(hybrid_detail.direction_up_prob, result.timestamp, X)

                        # Record ensemble for hybrid targets if they exist
                        try:
                            vol_ensemble = self.model_manager.get_all_models(ticker, "volatility")
                            if "ensemble" in vol_ensemble and vol_ensemble["ensemble"].is_trained:
                                vol_ensemble["ensemble"].record_prediction(hybrid_detail.volatility_prob, result.timestamp, X)

                            dir_ensemble = self.model_manager.get_all_models(ticker, "direction")
                            if "ensemble" in dir_ensemble and dir_ensemble["ensemble"].is_trained:
                                dir_ensemble["ensemble"].record_prediction(hybrid_detail.direction_up_prob, result.timestamp, X)
                        except Exception as e:
                            logger.debug(f"Could not record hybrid ensemble predictions: {e}")
                    except Exception as e:
                        logger.debug(f"Could not record hybrid predictions: {e}")

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Prediction for {ticker}: UP={final_up_prob:.3f} ({best_up_type}), "
                f"DOWN={final_down_prob:.3f} ({best_down_type}), "
                f"hybrid={'enabled' if use_hybrid and hybrid_detail else 'disabled'}, "
                f"elapsed={elapsed:.3f}s"
            )

            return result

        except ValueError as e:
            logger.error(f"No trained models available for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _predict_hybrid(
        self,
        ticker: str,
        X: np.ndarray,
        X_seq: np.ndarray = None
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Generate Structure B hybrid prediction (volatility × direction).

        Args:
            ticker: Stock ticker symbol
            X: Feature vector for tree models (1, n_features)
            X_seq: Sequence data for LSTM/Transformer (60, n_features)

        Returns:
            Tuple of (volatility_prob, direction_up_prob, hybrid_up_prob, hybrid_down_prob)
            or None if hybrid models not available
        """
        # If X_seq not provided, fall back to X
        if X_seq is None:
            X_seq = X

        try:
            # Get volatility model
            vol_type, vol_model = self.model_manager.get_best_model(ticker, "volatility")
            X_vol = X_seq if vol_type in ('lstm', 'transformer') else X
            volatility_prob = vol_model.predict_proba(X_vol)[0]

            # Get direction model
            dir_type, dir_model = self.model_manager.get_best_model(ticker, "direction")
            X_dir = X_seq if dir_type in ('lstm', 'transformer') else X
            direction_up_prob = dir_model.predict_proba(X_dir)[0]

            # Calculate hybrid probabilities
            # P(+5% up) = P(volatility) × P(direction=up | volatility)
            # P(-5% down) = P(volatility) × P(direction=down | volatility)
            hybrid_up_prob = volatility_prob * direction_up_prob
            hybrid_down_prob = volatility_prob * (1 - direction_up_prob)

            return volatility_prob, direction_up_prob, hybrid_up_prob, hybrid_down_prob

        except (ValueError, KeyError) as e:
            logger.warning(f"Hybrid models not available for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Hybrid prediction failed for {ticker}: {e}")
            return None

    def _calibrate_probabilities(
        self,
        up_prob: float,
        down_prob: float
    ) -> Tuple[float, float]:
        """
        Calibrate ensemble probabilities to prevent over/under estimation.

        When combining probabilities from multiple structures, the product
        can lead to underestimation. This method applies calibration.

        Args:
            up_prob: Raw ensemble up probability
            down_prob: Raw ensemble down probability

        Returns:
            Tuple of (calibrated_up_prob, calibrated_down_prob)
        """
        # Simple calibration: ensure probabilities are in valid range
        # More sophisticated calibration (Platt/Isotonic) would require
        # historical data and is handled separately in training

        # Clip to valid range
        up_prob = max(0.0, min(1.0, up_prob))
        down_prob = max(0.0, min(1.0, down_prob))

        # Optional: Normalize if sum > 1 (rare case)
        # This can happen with independent models
        total = up_prob + down_prob
        if total > 1.0:
            up_prob = up_prob / total
            down_prob = down_prob / total

        return up_prob, down_prob

    def predict_batch(
        self,
        tickers: List[str],
        include_all_models: bool = False,
        include_features: bool = False
    ) -> Dict[str, PredictionResult]:
        """
        Generate predictions for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            include_all_models: Include predictions from all models
            include_features: Include computed features in results

        Returns:
            Dictionary mapping ticker to PredictionResult
        """
        logger.info(f"Generating predictions for {len(tickers)} tickers")
        results = {}

        for ticker in tickers:
            try:
                result = self.predict(ticker, include_all_models, include_features)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {e}")
                # Continue with other tickers

        logger.info(f"Completed predictions for {len(results)}/{len(tickers)} tickers")
        return results

    def get_top_opportunities(
        self,
        tickers: List[str],
        direction: str = "up",
        min_probability: float = None,
        top_n: int = 10
    ) -> List[Tuple[str, PredictionResult]]:
        """
        Get top trading opportunities from a list of tickers.

        Args:
            tickers: List of stock ticker symbols to evaluate
            direction: "up" or "down" for which probabilities to sort by
            min_probability: Minimum probability threshold (default: settings)
            top_n: Number of top opportunities to return

        Returns:
            List of (ticker, PredictionResult) tuples, sorted by probability
        """
        min_prob = min_probability or settings.PROBABILITY_THRESHOLD

        # Get predictions for all tickers
        predictions = self.predict_batch(tickers, include_all_models=False)

        # Filter and sort
        opportunities = []
        for ticker, result in predictions.items():
            prob = result.up_probability if direction == "up" else result.down_probability

            if prob >= min_prob:
                opportunities.append((ticker, result))

        # Sort by probability (descending)
        opportunities.sort(
            key=lambda x: x[1].up_probability if direction == "up" else x[1].down_probability,
            reverse=True
        )

        return opportunities[:top_n]

    def _collect_minute_bars(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Collect minute bar data for feature computation from database only.

        Uses cached database data to avoid slow yfinance API calls.
        This is faster and works outside market hours.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with minute bars, or None if collection fails
        """
        if self.minute_bar_collector is None:
            logger.warning("MinuteBarCollector not configured, returning None")
            return None

        try:
            # Get bars from database only (no yfinance calls)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Get last 7 days of data

            # Use load_bars_from_db directly to avoid yfinance calls
            bars = self.minute_bar_collector.load_bars_from_db(
                ticker,
                start_time,
                end_time
            )

            if bars and len(bars) > 0:
                # Convert to DataFrame
                df = pd.DataFrame([bar.to_dict() for bar in bars])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.debug(f"Loaded {len(df)} minute bars for {ticker} from database")
                return df
            else:
                logger.warning(f"No minute bars found for {ticker} in database")
                return None

        except Exception as e:
            logger.error(f"Failed to collect minute bars for {ticker}: {e}")
            return None

    def _collect_order_book(self, ticker: str) -> Optional[Dict]:
        """
        Collect current order book snapshot.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with order book data, or None if collection fails
        """
        if self.quote_collector is None:
            logger.debug("QuoteCollector not configured, returning None")
            return None

        try:
            snapshot = self.quote_collector.get_order_book_snapshot(ticker)
            return snapshot
        except Exception as e:
            logger.warning(f"Failed to collect order book for {ticker}: {e}")
            return None

    def _get_market_context(self) -> Optional[Dict]:
        """
        Get market context data with caching.

        Returns:
            Dictionary with market context (SPY, QQQ, VIX, sectors), or None
        """
        # Check cache
        if self._market_context_cache is not None and self._market_context_cache_time:
            elapsed = (datetime.now() - self._market_context_cache_time).total_seconds()
            if elapsed < self._market_context_cache_ttl:
                return self._market_context_cache

        # Collect fresh data
        if self.market_context_collector is None:
            logger.debug("MarketContextCollector not configured, returning None")
            return None

        try:
            context = self.market_context_collector.get_current_context()

            # Convert to format expected by FeatureEngineer
            market_data = {
                'spy_return': context.spy_return if hasattr(context, 'spy_return') else 0.0,
                'qqq_return': context.qqq_return if hasattr(context, 'qqq_return') else 0.0,
                'vix_level': context.vix_level if hasattr(context, 'vix_level') else 20.0,
                'sector_return': context.sector_return if hasattr(context, 'sector_return') else 0.0,
                'correlation': 0.0  # Would need historical data to compute
            }

            # Update cache
            self._market_context_cache = market_data
            self._market_context_cache_time = datetime.now()

            return market_data

        except Exception as e:
            logger.warning(f"Failed to collect market context: {e}")
            return None

    def _get_all_model_predictions(
        self,
        ticker: str,
        X: np.ndarray,
        X_seq: np.ndarray = None,
        timestamp: datetime = None,
        record_predictions: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Get predictions from all available models (Structure A + B).

        Args:
            ticker: Stock ticker symbol
            X: Feature vector for tree models (1, n_features)
            X_seq: Sequence data for LSTM/Transformer (60, n_features)
            timestamp: Timestamp for recording predictions
            record_predictions: Whether to record predictions for accuracy tracking

        Returns:
            Nested dictionary: {target: {model_type: probability}}
        """
        all_predictions = {}

        # If X_seq not provided, fall back to X
        if X_seq is None:
            X_seq = X

        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = datetime.now()

        # Structure A targets
        for target in settings.PREDICTION_TARGETS:
            all_predictions[target] = {}

            try:
                models = self.model_manager.get_all_models(ticker, target)

                for model_type, model in models.items():
                    if model.is_trained:
                        try:
                            # Use X_seq for sequence models, X for tree models
                            X_input = X_seq if model_type in ('lstm', 'transformer') else X
                            prob = model.predict_proba(X_input)
                            # Handle both scalar (LSTM/Transformer) and array (tree models) returns
                            if hasattr(prob, '__len__'):
                                prob = prob[0]
                            precision = model.get_precision_at_threshold(hours=settings.BACKTEST_HOURS, threshold=0.5)

                            all_predictions[target][model_type] = {
                                'probability': float(prob),
                                'accuracy': float(precision)  # Now reports precision
                            }

                            # Record prediction for accuracy tracking
                            if record_predictions:
                                model.record_prediction(float(prob), timestamp, X_input)
                        except Exception as e:
                            logger.error(f"Failed to get prediction from {model_type}: {e}")
            except Exception as e:
                logger.debug(f"Models not available for {target}: {e}")

        # Structure B targets (if hybrid is enabled)
        if settings.USE_HYBRID_ENSEMBLE:
            for target in settings.HYBRID_TARGETS:
                all_predictions[target] = {}

                try:
                    models = self.model_manager.get_all_models(ticker, target)

                    for model_type, model in models.items():
                        if model.is_trained:
                            try:
                                # Use X_seq for sequence models, X for tree models
                                X_input = X_seq if model_type in ('lstm', 'transformer') else X
                                prob = model.predict_proba(X_input)
                                # Handle both scalar (LSTM/Transformer) and array (tree models) returns
                                if hasattr(prob, '__len__'):
                                    prob = prob[0]
                                precision = model.get_precision_at_threshold(hours=settings.BACKTEST_HOURS, threshold=0.5)

                                all_predictions[target][model_type] = {
                                    'probability': float(prob),
                                    'accuracy': float(precision)  # Now reports precision
                                }

                                # Record prediction for accuracy tracking
                                if record_predictions:
                                    model.record_prediction(float(prob), timestamp, X_input)
                            except Exception as e:
                                logger.error(f"Failed to get hybrid prediction from {model_type}: {e}")
                except Exception as e:
                    logger.debug(f"Hybrid models not available for {target}: {e}")

        return all_predictions

    def update_prediction_outcomes(
        self,
        ticker: str,
        prediction_time: datetime,
        actual_up: bool,
        actual_down: bool
    ) -> None:
        """
        Update models with actual outcomes for a previous prediction.

        This enables continuous accuracy tracking for model selection.

        Args:
            ticker: Stock ticker symbol
            prediction_time: When the prediction was made
            actual_up: Whether 5%+ upward movement occurred
            actual_down: Whether 5%+ downward movement occurred
        """
        try:
            # Update all models that made predictions
            for target in settings.PREDICTION_TARGETS:
                actual_outcome = actual_up if target == "up" else actual_down

                models = self.model_manager.get_all_models(ticker, target)

                for model in models.values():
                    model.update_outcome(prediction_time, actual_outcome)

            logger.debug(
                f"Updated outcomes for {ticker} at {prediction_time}: "
                f"up={actual_up}, down={actual_down}"
            )

        except Exception as e:
            logger.error(f"Failed to update outcomes for {ticker}: {e}")

    def get_prediction_stats(self, ticker: str) -> Dict[str, Any]:
        """
        Get prediction statistics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with statistics for all models
        """
        return self.model_manager.get_model_performances(ticker)

    def clear_market_context_cache(self) -> None:
        """Clear market context cache to force fresh data collection."""
        self._market_context_cache = None
        self._market_context_cache_time = None
        logger.debug("Market context cache cleared")

    def validate_models(self, ticker: str) -> Dict[str, bool]:
        """
        Validate that all required models are trained for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary mapping model keys to training status
        """
        status = {}

        for target in settings.PREDICTION_TARGETS:
            for model_type in settings.MODEL_TYPES:
                try:
                    model = self.model_manager.get_or_create_model(ticker, model_type, target)
                    status[f"{model_type}_{target}"] = model.is_trained
                except Exception as e:
                    logger.error(f"Error checking model {model_type}_{target}: {e}")
                    status[f"{model_type}_{target}"] = False

        return status

    def get_prediction_summary(
        self,
        tickers: List[str]
    ) -> Dict[str, Any]:
        """
        Get summary of predictions for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary with aggregated prediction statistics
        """
        predictions = self.predict_batch(tickers, include_all_models=False)

        buy_signals = []
        sell_signals = []
        hold_signals = []

        for ticker, result in predictions.items():
            signal = result.get_trading_signal()

            if signal == "BUY":
                buy_signals.append((ticker, result.up_probability))
            elif signal == "SELL":
                sell_signals.append((ticker, result.down_probability))
            else:
                hold_signals.append((ticker, max(result.up_probability, result.down_probability)))

        # Sort by probability
        buy_signals.sort(key=lambda x: x[1], reverse=True)
        sell_signals.sort(key=lambda x: x[1], reverse=True)

        return {
            'total_tickers': len(tickers),
            'successful_predictions': len(predictions),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'hold_signals': len(hold_signals),
            'top_buy_opportunities': buy_signals[:5],
            'top_sell_opportunities': sell_signals[:5],
            'timestamp': datetime.now().isoformat()
        }
