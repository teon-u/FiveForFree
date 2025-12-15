#!/usr/bin/env python3
"""
Train All Models for NASDAQ Prediction System

Performs initial training for all tickers with available historical data.
Creates 10 models per ticker (5 model types x 2 targets: up/down).

Model Types:
- XGBoost (GPU-accelerated)
- LightGBM (GPU-accelerated)
- LSTM (GPU-accelerated)
- Transformer (GPU-accelerated)
- Ensemble (meta-model)

Usage:
    # Train models for all tickers with historical data
    python scripts/train_all_models.py

    # Train specific tickers
    python scripts/train_all_models.py --tickers AAPL MSFT GOOGL

    # Use more parallel workers
    python scripts/train_all_models.py --workers 8

    # Train only tree models (faster)
    python scripts/train_all_models.py --model-types xgboost lightgbm
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import warnings

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file before importing settings
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=True)

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar, Ticker
from src.models.model_manager import ModelManager
from src.trainer.gpu_trainer import GPUParallelTrainer
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator

warnings.filterwarnings("ignore")


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging.

    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    # Also log to file
    log_file = (
        Path("./logs")
        / f"train_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="100 MB")


def get_tickers_with_data(min_bars: int = 1000) -> List[str]:
    """
    Get tickers that have sufficient historical data.

    Args:
        min_bars: Minimum number of bars required

    Returns:
        List of ticker symbols
    """
    try:
        from sqlalchemy import text

        with get_db() as db:
            # Use raw SQL for better compatibility
            result = db.execute(
                text(
                    f"SELECT symbol, COUNT(*) as bar_count FROM minute_bars "
                    f"GROUP BY symbol HAVING bar_count >= {min_bars} "
                    f"ORDER BY bar_count DESC"
                )
            )

            tickers = [row[0] for row in result]

            logger.info(f"Found {len(tickers)} tickers with >= {min_bars} bars")
            return tickers

    except Exception as e:
        logger.error(f"Failed to get tickers with data: {e}")
        return []


def load_ticker_data(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Load historical minute bar data for a ticker.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to load

    Returns:
        DataFrame with minute bars or None
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)

        with get_db() as db:
            stmt = (
                select(DBMinuteBar)
                .where(DBMinuteBar.symbol == ticker)
                .where(DBMinuteBar.timestamp >= cutoff_date)
                .order_by(DBMinuteBar.timestamp)
            )

            bars = db.execute(stmt).scalars().all()

            if not bars:
                logger.warning(f"{ticker}: No data found")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "vwap": bar.vwap,
                    }
                    for bar in bars
                ]
            )

            logger.info(f"{ticker}: Loaded {len(df)} bars")
            return df

    except Exception as e:
        logger.error(f"{ticker}: Failed to load data: {e}")
        return None


def prepare_training_data(
    ticker: str, df: pd.DataFrame
) -> Optional[tuple]:
    """
    Prepare features and labels for training.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with minute bars

    Returns:
        Tuple of (X, y_up, y_down) or None
    """
    try:
        logger.info(f"{ticker}: Preparing training data...")

        # Initialize processors
        feature_engineer = FeatureEngineer()
        label_generator = LabelGenerator(
            target_percent=settings.TARGET_PERCENT,
            prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
        )

        # Compute features
        logger.debug(f"{ticker}: Computing features...")
        features_df = feature_engineer.compute_features(df)

        # Generate labels
        logger.debug(f"{ticker}: Generating labels...")
        labels_up = []
        labels_down = []

        for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
            entry_time = df.iloc[idx]["timestamp"]
            entry_price = df.iloc[idx]["close"]

            labels = label_generator.generate_labels(df, entry_time, entry_price)

            labels_up.append(labels["label_up"])
            labels_down.append(labels["label_down"])

        # Align features and labels
        feature_names = feature_engineer.get_feature_names()
        X = features_df[feature_names].values[: len(labels_up)]
        y_up = np.array(labels_up)
        y_down = np.array(labels_down)

        # Remove any NaN rows
        valid_indices = ~np.isnan(X).any(axis=1)
        X = X[valid_indices]
        y_up = y_up[valid_indices]
        y_down = y_down[valid_indices]

        logger.info(
            f"{ticker}: Prepared {len(X)} samples with {X.shape[1]} features"
        )
        logger.info(
            f"{ticker}: Label distribution - Up: {y_up.sum()}/{len(y_up)} "
            f"({y_up.mean()*100:.1f}%), Down: {y_down.sum()}/{len(y_down)} "
            f"({y_down.mean()*100:.1f}%)"
        )

        if len(X) < 100:
            logger.warning(f"{ticker}: Insufficient samples ({len(X)})")
            return None

        return X, y_up, y_down

    except Exception as e:
        logger.error(f"{ticker}: Failed to prepare training data: {e}")
        return None


def train_ticker_models(
    ticker: str,
    trainer: GPUParallelTrainer,
    days: int = 30,
    model_types: Optional[List[str]] = None,
) -> dict:
    """
    Train all models for a single ticker.

    Args:
        ticker: Stock ticker symbol
        trainer: GPUParallelTrainer instance
        days: Number of days of data to use
        model_types: List of model types to train (None = all)

    Returns:
        Dictionary with training statistics
    """
    stats = {
        "ticker": ticker,
        "success": False,
        "models_trained": 0,
        "models_failed": 0,
        "samples": 0,
        "features": 0,
    }

    try:
        # Load data
        df = load_ticker_data(ticker, days)
        if df is None or len(df) < 1000:
            logger.warning(f"{ticker}: Insufficient data")
            return stats

        # Prepare training data
        training_data = prepare_training_data(ticker, df)
        if training_data is None:
            return stats

        X, y_up, y_down = training_data
        stats["samples"] = len(X)
        stats["features"] = X.shape[1]

        # Train models
        logger.info(f"{ticker}: Training models...")
        start_time = datetime.now()

        results = trainer.train_single_ticker(ticker, X, y_up, y_down)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Count results
        for model_key, success in results.items():
            if success:
                stats["models_trained"] += 1
            else:
                stats["models_failed"] += 1

        stats["success"] = stats["models_trained"] > 0
        stats["elapsed_seconds"] = elapsed

        logger.success(
            f"{ticker}: Training complete - {stats['models_trained']}/{len(results)} "
            f"models successful ({elapsed:.1f}s)"
        )

    except Exception as e:
        logger.error(f"{ticker}: Training failed: {e}")

    return stats


def main() -> int:
    """
    Main function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Train all models for NASDAQ prediction system"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific tickers to train (default: all with data)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to use for training (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.N_PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {settings.N_PARALLEL_WORKERS})",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["xgboost", "lightgbm", "lstm", "transformer", "ensemble"],
        help="Specific model types to train (default: all)",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=1000,
        help="Minimum bars required per ticker (default: 1000)",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        help="Maximum number of tickers to train",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("NASDAQ Prediction System - Model Training")
    logger.info("=" * 80)
    logger.info(f"Training data: {args.days} days")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"GPU enabled: {settings.USE_GPU}")
    logger.info("")

    try:
        # Get target tickers
        if args.tickers:
            tickers = args.tickers
            logger.info(f"Using specified tickers: {tickers}")
        else:
            tickers = get_tickers_with_data(min_bars=args.min_bars)
            logger.info(f"Found {len(tickers)} tickers with sufficient data")

        if not tickers:
            logger.error("No tickers to train")
            return 1

        # Apply max tickers limit
        if args.max_tickers:
            tickers = tickers[: args.max_tickers]
            logger.info(f"Limited to {len(tickers)} tickers")

        logger.info(f"\nTickers to process ({len(tickers)}): {', '.join(tickers[:10])}")
        if len(tickers) > 10:
            logger.info(f"... and {len(tickers) - 10} more")
        logger.info("")

        # Initialize model manager and trainer
        model_manager = ModelManager()
        trainer = GPUParallelTrainer(model_manager, n_parallel=args.workers)

        # Display GPU info
        gpu_info = trainer.get_gpu_memory_usage()
        if gpu_info.get("available"):
            logger.info(
                f"GPU Memory: {gpu_info['allocated_gb']:.2f}GB / "
                f"{gpu_info['total_gb']:.2f}GB ({gpu_info['utilization_pct']:.1f}%)"
            )
        logger.info("")

        # Train models for each ticker
        all_stats = []
        successful = 0
        failed = 0

        start_time = datetime.now()

        # Create progress bar
        pbar = tqdm(
            tickers,
            desc="Training models",
            unit="ticker",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for ticker in pbar:
            # Update progress bar description
            pbar.set_description(f"Training {ticker}")
            pbar.set_postfix({"success": successful, "failed": failed})

            stats = train_ticker_models(
                ticker,
                trainer,
                days=args.days,
                model_types=args.model_types,
            )

            all_stats.append(stats)

            if stats["success"]:
                successful += 1
            else:
                failed += 1

            # Clear GPU memory
            trainer.clear_gpu_memory()

            # Update postfix with latest stats
            pbar.set_postfix({
                "success": successful,
                "failed": failed,
                "last": f"{stats.get('elapsed_seconds', 0):.1f}s"
            })

        # Summary
        elapsed_total = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Total tickers: {len(tickers)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(
            f"Total models trained: {sum(s['models_trained'] for s in all_stats)}"
        )
        logger.info(
            f"Total samples processed: {sum(s['samples'] for s in all_stats):,}"
        )
        logger.info(f"Time elapsed: {elapsed_total/60:.1f} minutes")
        logger.info(f"Average time per ticker: {elapsed_total/len(tickers):.1f} seconds")

        # Model manager summary
        manager_summary = model_manager.get_summary()
        logger.info(f"\nModel Manager Statistics:")
        logger.info(f"  Total models: {manager_summary['total_models']}")
        logger.info(f"  Trained models: {manager_summary['trained_models']}")
        logger.info(f"  Untrained models: {manager_summary['untrained_models']}")

        logger.info("=" * 80)

        # List failed tickers
        if failed > 0:
            failed_tickers = [s["ticker"] for s in all_stats if not s["success"]]
            logger.warning(f"\nFailed tickers: {', '.join(failed_tickers)}")

        if successful > 0:
            logger.success("\nModel training complete!")
            logger.info(f"\nModels saved to: {model_manager.models_dir}")
            return 0
        else:
            logger.error("\nAll training failed!")
            return 1

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        if args.verbose:
            logger.exception("Traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
