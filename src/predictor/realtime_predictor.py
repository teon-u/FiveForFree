"""
Real-time Prediction System for NASDAQ Trading

Provides real-time predictions for active trading:
- Collects latest market data via Polygon.io
- Computes 57 features in real-time
- Runs predictions using best-performing models
- Returns probabilities optimized for UI display
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
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
class PredictionResult:
    """
    Result of a real-time prediction.

    Attributes:
        ticker: Stock ticker symbol
        timestamp: When prediction was made
        current_price: Current stock price
        up_probability: Probability of 5%+ upward movement
        down_probability: Probability of 5%+ downward movement
        best_up_model: Best performing model for upward prediction
        best_down_model: Best performing model for downward prediction
        up_model_accuracy: 50-hour accuracy of up model
        down_model_accuracy: 50-hour accuracy of down model
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
    all_model_predictions: Optional[Dict[str, Dict[str, float]]] = None
    features: Optional[np.ndarray] = None
    market_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'up_probability': self.up_probability,
            'down_probability': self.down_probability,
            'best_up_model': self.best_up_model,
            'best_down_model': self.best_down_model,
            'up_model_accuracy': self.up_model_accuracy,
            'down_model_accuracy': self.down_model_accuracy,
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
        include_features: bool = False
    ) -> PredictionResult:
        """
        Generate real-time prediction for a ticker.

        Args:
            ticker: Stock ticker symbol
            include_all_models: Include predictions from all models (not just best)
            include_features: Include computed features in result

        Returns:
            PredictionResult with probabilities and metadata

        Raises:
            ValueError: If required data cannot be collected
            RuntimeError: If prediction fails
        """
        logger.debug(f"Generating prediction for {ticker}")
        start_time = datetime.now()

        # Step 1: Collect minute bars
        minute_bars = self._collect_minute_bars(ticker)
        if minute_bars is None or len(minute_bars) < 60:
            raise ValueError(f"Insufficient minute bar data for {ticker}")

        current_price = minute_bars.iloc[-1]['close']

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

        # Step 5: Run predictions
        try:
            # Get best models and their predictions
            best_up_type, best_up_model = self.model_manager.get_best_model(ticker, "up")
            best_down_type, best_down_model = self.model_manager.get_best_model(ticker, "down")

            up_prob = best_up_model.predict_proba(X)[0]
            down_prob = best_down_model.predict_proba(X)[0]

            # Get model accuracies
            up_accuracy = best_up_model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)
            down_accuracy = best_down_model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)

            # Optionally get predictions from all models
            all_predictions = None
            if include_all_models:
                all_predictions = self._get_all_model_predictions(ticker, X)

            # Create result
            result = PredictionResult(
                ticker=ticker,
                timestamp=datetime.now(),
                current_price=float(current_price),
                up_probability=float(up_prob),
                down_probability=float(down_prob),
                best_up_model=best_up_type,
                best_down_model=best_down_type,
                up_model_accuracy=float(up_accuracy),
                down_model_accuracy=float(down_accuracy),
                all_model_predictions=all_predictions,
                features=X[0] if include_features else None,
                market_context=market_context if include_features else None
            )

            # Log prediction to models for accuracy tracking
            best_up_model.record_prediction(up_prob, result.timestamp, X)
            best_down_model.record_prediction(down_prob, result.timestamp, X)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Prediction for {ticker}: UP={up_prob:.3f} ({best_up_type}), "
                f"DOWN={down_prob:.3f} ({best_down_type}), "
                f"elapsed={elapsed:.3f}s"
            )

            return result

        except ValueError as e:
            logger.error(f"No trained models available for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

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
        Collect minute bar data for feature computation.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with minute bars, or None if collection fails
        """
        if self.minute_bar_collector is None:
            logger.warning("MinuteBarCollector not configured, returning None")
            return None

        try:
            # Get bars from current time back to lookback period
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=self.lookback_minutes)

            bars = self.minute_bar_collector.collect_range(
                ticker,
                start_time,
                end_time
            )

            if bars:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': bar.vwap
                    }
                    for bar in bars
                ])

                logger.debug(f"Collected {len(df)} minute bars for {ticker}")
                return df
            else:
                logger.warning(f"No minute bars collected for {ticker}")
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
        X: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Get predictions from all available models.

        Args:
            ticker: Stock ticker symbol
            X: Feature vector

        Returns:
            Nested dictionary: {target: {model_type: probability}}
        """
        all_predictions = {}

        for target in settings.PREDICTION_TARGETS:
            all_predictions[target] = {}

            models = self.model_manager.get_all_models(ticker, target)

            for model_type, model in models.items():
                if model.is_trained:
                    try:
                        prob = model.predict_proba(X)[0]
                        accuracy = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)

                        all_predictions[target][model_type] = {
                            'probability': float(prob),
                            'accuracy': float(accuracy)
                        }
                    except Exception as e:
                        logger.error(f"Failed to get prediction from {model_type}: {e}")

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
