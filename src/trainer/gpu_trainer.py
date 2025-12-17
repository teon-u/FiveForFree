"""
GPU Parallel Trainer for NASDAQ Prediction System

Optimized for NVIDIA RTX 5080:
- Tree models (XGBoost, LightGBM): Parallel training with ThreadPoolExecutor
- Neural models (LSTM, Transformer): Sequential training for GPU memory management
- Batch training for multiple tickers

Hybrid-Ensemble Support:
- Structure A: Direct up/down prediction models
- Structure B: Volatility + Direction models
- Combined training for hybrid-ensemble approach
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gc

import numpy as np
import torch
from loguru import logger

from src.models.model_manager import ModelManager
from config.settings import settings


class GPUParallelTrainer:
    """
    GPU-accelerated parallel trainer for multiple tickers and models.

    Architecture:
    - Tree models run in parallel using CPU threads (GPU operations are async)
    - Neural models run sequentially to manage GPU memory efficiently
    - Automatic garbage collection and CUDA cache clearing between batches

    RTX 5080 Optimization:
    - XGBoost/LightGBM: Use GPU via tree_method="gpu_hist"
    - LSTM/Transformer: Use CUDA with automatic memory management
    - ThreadPool sized for optimal CPU-GPU coordination
    """

    def __init__(
        self,
        model_manager: ModelManager,
        n_parallel: int = None,
        validation_split: float = 0.2
    ):
        """
        Initialize GPU parallel trainer.

        Args:
            model_manager: ModelManager instance for model coordination
            n_parallel: Number of parallel workers (default from settings)
            validation_split: Fraction of data to use for validation
        """
        self.model_manager = model_manager
        self.n_parallel = n_parallel or settings.N_PARALLEL_WORKERS
        self.validation_split = validation_split

        # GPU setup
        self.use_gpu = settings.USE_GPU
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            logger.warning("GPU not available, using CPU")

        # Track training history
        self.training_history: Dict[str, List[Dict]] = {}

        logger.info(f"GPUParallelTrainer initialized with {self.n_parallel} parallel workers")

    def train_ticker_batch(
        self,
        ticker_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        targets: List[str] = None
    ) -> Dict[str, Dict[str, bool]]:
        """
        Train all models for a batch of tickers.

        Args:
            ticker_data: Dictionary mapping ticker to (X, y) tuples
            targets: List of targets to train (default: ["up", "down"])

        Returns:
            Dictionary mapping ticker to model training results
        """
        targets = targets or settings.PREDICTION_TARGETS
        results = {}

        logger.info(f"Training batch of {len(ticker_data)} tickers")
        start_time = datetime.now()

        for ticker, (X, y) in ticker_data.items():
            logger.info(f"Training all models for {ticker}")
            ticker_results = {}

            for target in targets:
                # Split data for validation
                X_train, X_val, y_train, y_val = self._train_val_split(X, y)

                # Train tree models in parallel
                tree_results = self._train_tree_models_parallel(
                    ticker, target, X_train, y_train, X_val, y_val
                )
                ticker_results.update(tree_results)

                # Train neural models sequentially (GPU memory management)
                neural_results = self._train_neural_models_sequential(
                    ticker, target, X_train, y_train, X_val, y_val
                )
                ticker_results.update(neural_results)

            results[ticker] = ticker_results

            # Log progress
            logger.info(f"Completed training for {ticker}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch training completed in {elapsed:.2f}s")

        return results

    def train_single_ticker(
        self,
        ticker: str,
        X: np.ndarray,
        y_up: np.ndarray,
        y_down: np.ndarray,
        y_volatility: np.ndarray = None,
        y_direction: np.ndarray = None
    ) -> Dict[str, bool]:
        """
        Train all models for a single ticker with separate up/down labels.

        Supports both Structure A (direct) and Structure B (hybrid) training.

        Args:
            ticker: Stock ticker symbol
            X: Feature matrix (n_samples, n_features)
            y_up: Binary labels for upward movement (Structure A)
            y_down: Binary labels for downward movement (Structure A)
            y_volatility: Binary labels for volatility (Structure B, optional)
            y_direction: Binary labels for direction (Structure B, optional)

        Returns:
            Dictionary mapping model keys to training success status
        """
        logger.info(f"Training all models for {ticker}")
        logger.info(f"Samples: {len(X)}, Features: {X.shape[1]}")

        results = {}

        # Split data for Structure A
        X_train, X_val, y_up_train, y_up_val = self._train_val_split(X, y_up)
        _, _, y_down_train, y_down_val = self._train_val_split(X, y_down)

        # ===== Structure A: Direct Prediction Models =====
        # Train "up" models
        logger.info(f"Training 'up' prediction models for {ticker}")
        tree_results = self._train_tree_models_parallel(
            ticker, "up", X_train, y_up_train, X_val, y_up_val
        )
        results.update(tree_results)

        neural_results = self._train_neural_models_sequential(
            ticker, "up", X_train, y_up_train, X_val, y_up_val
        )
        results.update(neural_results)

        # Train "down" models
        logger.info(f"Training 'down' prediction models for {ticker}")
        tree_results = self._train_tree_models_parallel(
            ticker, "down", X_train, y_down_train, X_val, y_down_val
        )
        results.update(tree_results)

        neural_results = self._train_neural_models_sequential(
            ticker, "down", X_train, y_down_train, X_val, y_down_val
        )
        results.update(neural_results)

        # ===== Structure B: Hybrid Models (if labels provided) =====
        if y_volatility is not None and y_direction is not None:
            hybrid_results = self._train_hybrid_models(
                ticker, X, y_volatility, y_direction
            )
            results.update(hybrid_results)

        # ===== Train Ensemble Models =====
        logger.info(f"Training ensemble models for {ticker}")
        ensemble_results = self._train_ensemble_models(
            ticker, X_train, y_up_train, y_down_train, X_val, y_up_val, y_down_val
        )
        results.update(ensemble_results)

        # Save all trained models
        self.model_manager.save_models(ticker)

        logger.info(f"Training completed for {ticker}")
        return results

    def _train_hybrid_models(
        self,
        ticker: str,
        X: np.ndarray,
        y_volatility: np.ndarray,
        y_direction: np.ndarray
    ) -> Dict[str, bool]:
        """
        Train Structure B hybrid models (volatility + direction).

        Args:
            ticker: Stock ticker symbol
            X: Feature matrix
            y_volatility: Binary labels for volatility (±5% movement)
            y_direction: Binary labels for direction (1=up, 0=down)

        Returns:
            Dictionary mapping model keys to training success status
        """
        logger.info(f"Training hybrid models (Structure B) for {ticker}")
        results = {}

        # Split data
        X_train, X_val, y_vol_train, y_vol_val = self._train_val_split(X, y_volatility)
        _, _, y_dir_train, y_dir_val = self._train_val_split(X, y_direction)

        # ===== Train Volatility Models =====
        logger.info(f"Training 'volatility' prediction models for {ticker}")

        # Tree models for volatility
        tree_results = self._train_tree_models_parallel(
            ticker, "volatility", X_train, y_vol_train, X_val, y_vol_val
        )
        results.update(tree_results)

        # Neural models for volatility
        neural_results = self._train_neural_models_sequential(
            ticker, "volatility", X_train, y_vol_train, X_val, y_vol_val
        )
        results.update(neural_results)

        # ===== Train Direction Models =====
        # Direction model is trained on ALL samples, but predictions are weighted
        # by volatility probability during inference
        logger.info(f"Training 'direction' prediction models for {ticker}")

        # Tree models for direction
        tree_results = self._train_tree_models_parallel(
            ticker, "direction", X_train, y_dir_train, X_val, y_dir_val
        )
        results.update(tree_results)

        # Neural models for direction
        neural_results = self._train_neural_models_sequential(
            ticker, "direction", X_train, y_dir_train, X_val, y_dir_val
        )
        results.update(neural_results)

        logger.info(f"Hybrid model training completed for {ticker}")
        return results

    def train_single_ticker_hybrid(
        self,
        ticker: str,
        X: np.ndarray,
        y_up: np.ndarray,
        y_down: np.ndarray,
        y_volatility: np.ndarray,
        y_direction: np.ndarray
    ) -> Dict[str, bool]:
        """
        Train all models for hybrid-ensemble approach (Structure A + B combined).

        This is the main entry point for full hybrid training.

        Args:
            ticker: Stock ticker symbol
            X: Feature matrix (n_samples, n_features)
            y_up: Binary labels for upward movement (Structure A)
            y_down: Binary labels for downward movement (Structure A)
            y_volatility: Binary labels for volatility (Structure B)
            y_direction: Binary labels for direction (Structure B)

        Returns:
            Dictionary mapping model keys to training success status
        """
        return self.train_single_ticker(
            ticker, X, y_up, y_down, y_volatility, y_direction
        )

    def _train_tree_models_parallel(
        self,
        ticker: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """
        Train tree-based models (XGBoost, LightGBM) in parallel.

        Args:
            ticker: Stock ticker
            target: "up" or "down"
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary mapping model type to success status
        """
        tree_models = ["xgboost", "lightgbm"]
        results = {}

        logger.debug(f"Training tree models in parallel for {ticker} ({target})")

        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            # Submit training jobs
            futures = {}
            for model_type in tree_models:
                future = executor.submit(
                    self._train_single_model,
                    ticker, model_type, target,
                    X_train, y_train, X_val, y_val
                )
                futures[future] = model_type

            # Collect results
            for future in as_completed(futures):
                model_type = futures[future]
                try:
                    success = future.result()
                    results[f"{model_type}_{target}"] = success
                    logger.debug(f"Completed {model_type} for {ticker} ({target}): {success}")
                except Exception as e:
                    logger.error(f"Failed to train {model_type} for {ticker} ({target}): {e}")
                    results[f"{model_type}_{target}"] = False

        return results

    def _train_neural_models_sequential(
        self,
        ticker: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """
        Train neural models (LSTM, Transformer) sequentially for GPU memory management.

        Args:
            ticker: Stock ticker
            target: "up" or "down"
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary mapping model type to success status
        """
        neural_models = ["lstm", "transformer"]
        results = {}

        logger.debug(f"Training neural models sequentially for {ticker} ({target})")

        for model_type in neural_models:
            # Clear GPU cache before training each model
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            try:
                success = self._train_single_model(
                    ticker, model_type, target,
                    X_train, y_train, X_val, y_val
                )
                results[f"{model_type}_{target}"] = success
                logger.debug(f"Completed {model_type} for {ticker} ({target}): {success}")
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {ticker} ({target}): {e}")
                results[f"{model_type}_{target}"] = False

            # Clear GPU cache after training
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return results

    def _train_single_model(
        self,
        ticker: str,
        model_type: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> bool:
        """
        Train a single model.

        Args:
            ticker: Stock ticker
            model_type: Type of model to train
            target: "up" or "down"
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            True if training succeeded, False otherwise
        """
        try:
            # Get or create model (returns tuple of model_type, model)
            _, model = self.model_manager.get_or_create_model(ticker, model_type, target)

            # Train model
            model.train(X_train, y_train, X_val, y_val)

            # Record training event
            self._record_training(ticker, model_type, target, len(X_train))

            return True

        except Exception as e:
            logger.error(f"Training failed for {ticker} {model_type} ({target}): {e}")
            return False

    def _train_ensemble_models(
        self,
        ticker: str,
        X_train: np.ndarray,
        y_up_train: np.ndarray,
        y_down_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_up_val: Optional[np.ndarray] = None,
        y_down_val: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """
        Train ensemble models for a single ticker.

        Uses the new multi-strategy ensemble architecture.

        Args:
            ticker: Stock ticker symbol
            X_train: Training features
            y_up_train: Training labels for 'up' target
            y_down_train: Training labels for 'down' target
            X_val: Validation features
            y_up_val: Validation labels for 'up' target
            y_down_val: Validation labels for 'down' target

        Returns:
            Dictionary mapping model keys to training success status
        """
        results = {}

        # Train ensemble for 'up' target
        try:
            _, ensemble_up = self.model_manager.get_or_create_model(ticker, "ensemble", "up")
            ensemble_up.train(X_train, y_up_train, X_val, y_up_val)
            results["ensemble_up"] = True

            # Log ensemble stats
            stats = ensemble_up.get_ensemble_stats()
            logger.info(
                f"Ensemble 'up' for {ticker}: "
                f"strategy={stats['strategy']}, "
                f"base_models={stats['trained_base_models']}/{stats['total_base_models']}, "
                f"best_model={stats.get('best_model', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Failed to train ensemble 'up' for {ticker}: {e}")
            results["ensemble_up"] = False

        # Train ensemble for 'down' target
        try:
            _, ensemble_down = self.model_manager.get_or_create_model(ticker, "ensemble", "down")
            ensemble_down.train(X_train, y_down_train, X_val, y_down_val)
            results["ensemble_down"] = True

            # Log ensemble stats
            stats = ensemble_down.get_ensemble_stats()
            logger.info(
                f"Ensemble 'down' for {ticker}: "
                f"strategy={stats['strategy']}, "
                f"base_models={stats['trained_base_models']}/{stats['total_base_models']}, "
                f"best_model={stats.get('best_model', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Failed to train ensemble 'down' for {ticker}: {e}")
            results["ensemble_down"] = False

        return results

    def train_ensemble_models(
        self,
        ticker_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        targets: List[str] = None
    ) -> Dict[str, bool]:
        """
        Train ensemble models for tickers (requires base models to be trained first).

        Args:
            ticker_data: Dictionary mapping ticker to (X, y) tuples
            targets: List of targets to train (default: ["up", "down"])

        Returns:
            Dictionary mapping model keys to training success status
        """
        targets = targets or settings.PREDICTION_TARGETS
        results = {}

        logger.info("Training ensemble models")

        for ticker, (X, y) in ticker_data.items():
            X_train, X_val, y_train, y_val = self._train_val_split(X, y)

            for target in targets:
                try:
                    _, model = self.model_manager.get_or_create_model(ticker, "ensemble", target)
                    model.train(X_train, y_train, X_val, y_val)
                    results[f"{ticker}_ensemble_{target}"] = True
                    logger.info(f"Trained ensemble for {ticker} ({target})")
                except Exception as e:
                    logger.error(f"Failed to train ensemble for {ticker} ({target}): {e}")
                    results[f"{ticker}_ensemble_{target}"] = False

        return results

    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.

        Args:
            X: Features
            y: Labels

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if self.validation_split <= 0:
            return X, None, y, None

        n_samples = len(X)
        n_train = int(n_samples * (1 - self.validation_split))

        # Use chronological split (don't shuffle for time series)
        X_train = X[:n_train]
        X_val = X[n_train:]
        y_train = y[:n_train]
        y_val = y[n_train:]

        return X_train, X_val, y_train, y_val

    def _record_training(
        self,
        ticker: str,
        model_type: str,
        target: str,
        n_samples: int
    ) -> None:
        """
        Record training event in history.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            target: "up" or "down"
            n_samples: Number of training samples
        """
        key = f"{ticker}_{model_type}_{target}"

        if key not in self.training_history:
            self.training_history[key] = []

        self.training_history[key].append({
            'timestamp': datetime.now(),
            'n_samples': n_samples,
            'model_type': model_type,
            'target': target
        })

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training activities.

        Returns:
            Dictionary with training statistics
        """
        total_trainings = sum(len(events) for events in self.training_history.values())

        unique_tickers = set()
        for key in self.training_history.keys():
            ticker = key.rsplit('_', 2)[0]
            unique_tickers.add(ticker)

        return {
            'total_trainings': total_trainings,
            'unique_tickers': len(unique_tickers),
            'tickers': sorted(list(unique_tickers)),
            'models_trained': len(self.training_history),
            'gpu_available': self.use_gpu and torch.cuda.is_available(),
            'device': str(self.device),
            'parallel_workers': self.n_parallel
        }

    def clear_gpu_memory(self) -> None:
        """
        Clear GPU memory cache.

        Call this between training batches to free up memory.
        """
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cache cleared")

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics (in GB)
        """
        if not self.use_gpu or not torch.cuda.is_available():
            return {'available': False}

        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        return {
            'available': True,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'utilization_pct': (allocated / total) * 100
        }
