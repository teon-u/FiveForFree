"""
APScheduler Configuration for NASDAQ Prediction System

Manages all scheduled jobs:
- Every minute: Data collection + predictions (during market hours)
- Every hour: Incremental training + target ticker update
- After market close (5 PM ET): Full model retraining

Uses timezone-aware scheduling for market hours (Eastern Time).
"""

from datetime import datetime, time, timedelta
from typing import Optional, List
import pytz

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from loguru import logger

from config.settings import settings
from src.utils.database import get_db, get_or_create_ticker, MinuteBar as DBMinuteBar
from src.collector.minute_bars import MinuteBarCollector
from src.collector.ticker_selector import TickerSelector
from src.models.model_manager import ModelManager
from src.trainer.gpu_trainer import GPUParallelTrainer
from src.trainer.incremental import IncrementalTrainer
from src.predictor.realtime_predictor import RealtimePredictor
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator

# Eastern timezone for market hours
ET = pytz.timezone("America/New_York")


class NASDAQScheduler:
    """
    Scheduler for automated NASDAQ prediction system operations.

    Job Schedule:
    - Every minute (market hours): Collect data and generate predictions
    - Every hour: Incremental model training + update target tickers
    - 5:00 PM ET: Full model retraining after market close
    """

    def __init__(
        self,
        model_manager: ModelManager,
        enable_market_hours_check: bool = True,
    ):
        """
        Initialize scheduler.

        Args:
            model_manager: ModelManager instance for model operations
            enable_market_hours_check: If True, only run during market hours (9:30-16:00 ET)
        """
        self.model_manager = model_manager
        self.enable_market_hours_check = enable_market_hours_check

        # Initialize components
        self.minute_bar_collector = MinuteBarCollector(use_db=True)  # Enable incremental collection
        self.ticker_selector = TickerSelector()
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator()
        self.gpu_trainer = GPUParallelTrainer(model_manager)
        self.incremental_trainer = IncrementalTrainer(model_manager)
        self.predictor = RealtimePredictor(
            model_manager, self.minute_bar_collector
        )

        # Active ticker lists by category (updated hourly)
        self.active_tickers: dict[str, List[str]] = {'volume': [], 'gainers': []}

        # Scheduler instance
        self.scheduler: Optional[BackgroundScheduler] = None

        # Job execution tracking
        self.job_stats = {
            "minute_collect": {"runs": 0, "errors": 0, "last_run": None},
            "hourly_training": {"runs": 0, "errors": 0, "last_run": None},
            "daily_retrain": {"runs": 0, "errors": 0, "last_run": None},
            "ticker_update": {"runs": 0, "errors": 0, "last_run": None},
            "outcome_update": {"runs": 0, "errors": 0, "last_run": None, "updated": 0},
        }

        logger.info("NASDAQScheduler initialized")

    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours (9:30 AM - 4:00 PM ET).

        Returns:
            True if within market hours, False otherwise
        """
        if not self.enable_market_hours_check:
            return True

        now_et = datetime.now(ET)
        current_time = now_et.time()

        market_open = time(
            settings.MARKET_OPEN_HOUR, settings.MARKET_OPEN_MINUTE
        )
        market_close = time(
            settings.MARKET_CLOSE_HOUR, settings.MARKET_CLOSE_MINUTE
        )

        # Check if weekday (Monday=0, Sunday=6)
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False

        return market_open <= current_time <= market_close

    # ========== JOB DEFINITIONS ==========

    def job_collect_and_predict(self) -> None:
        """
        Minute job: Collect latest data and generate predictions.

        Runs every minute during market hours.
        """
        job_name = "minute_collect"

        try:
            if not self.is_market_hours():
                logger.debug("Outside market hours, skipping collection")
                return

            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now()

            logger.info("=" * 80)
            logger.info("MINUTE JOB: Data Collection & Predictions")
            logger.info("=" * 80)

            # Get all unique tickers from both categories
            all_tickers = set(self.active_tickers['volume'] + self.active_tickers['gainers'])

            if not all_tickers:
                logger.warning("No active tickers, skipping")
                return

            logger.info(
                f"Processing {len(all_tickers)} tickers "
                f"(volume: {len(self.active_tickers['volume'])}, "
                f"gainers: {len(self.active_tickers['gainers'])})"
            )

            # Collect latest minute bars for all tickers using incremental collection
            collected = 0
            predicted = 0

            # Use incremental collection (fetches last 10 minutes, auto-saves to DB)
            from_time = datetime.now() - timedelta(minutes=10)
            to_time = datetime.now()

            for ticker in all_tickers:
                try:
                    # Use incremental collection - automatically handles DB storage
                    bars = self.minute_bar_collector.get_bars(ticker, from_time, to_time)

                    if bars:
                        latest_bar = bars[-1]  # Get most recent bar
                        collected += 1

                        # Generate prediction
                        try:
                            prediction = self.predictor.predict(ticker)
                            logger.info(
                                f"{ticker}: UP={prediction.up_probability:.3f} "
                                f"({prediction.best_up_model}), "
                                f"DOWN={prediction.down_probability:.3f} "
                                f"({prediction.best_down_model})"
                            )
                            predicted += 1

                        except Exception as e:
                            logger.warning(f"{ticker}: Prediction failed: {e}")

                except Exception as e:
                    logger.error(f"{ticker}: Failed to process: {e}")
                    continue

            logger.info(
                f"Minute job complete: {collected} tickers processed, "
                f"{predicted} predictions generated"
            )

        except Exception as e:
            logger.error(f"Minute job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    def job_hourly_training(self) -> None:
        """
        Hourly job: Incremental model training.

        Runs every hour during market hours.
        """
        job_name = "hourly_training"

        try:
            if not self.is_market_hours():
                logger.debug("Outside market hours, skipping hourly training")
                return

            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now()

            logger.info("=" * 80)
            logger.info("HOURLY JOB: Incremental Training")
            logger.info("=" * 80)

            # Update models for all tickers with buffered data
            results = self.incremental_trainer.update_all_tickers()

            if results:
                logger.info(f"Updated models for {len(results)} tickers")

                # Log summary
                total_success = sum(
                    sum(1 for v in ticker_results.values() if v)
                    for ticker_results in results.values()
                )
                total_attempted = sum(
                    len(ticker_results) for ticker_results in results.values()
                )

                logger.info(
                    f"Incremental training: {total_success}/{total_attempted} "
                    f"models updated successfully"
                )
            else:
                logger.info("No models ready for incremental training")

        except Exception as e:
            logger.error(f"Hourly training job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    def job_update_tickers(self) -> None:
        """
        Hourly job: Update target ticker list.

        Runs every hour to refresh the list of target tickers based on
        current volume and price movements.
        """
        job_name = "ticker_update"

        try:
            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now()

            logger.info("=" * 80)
            logger.info("HOURLY JOB: Update Target Tickers")
            logger.info("=" * 80)

            # Get fresh ticker lists by category
            categories = self.ticker_selector.get_both_categories()

            if categories and (categories['volume'] or categories['gainers']):
                old_volume_count = len(self.active_tickers['volume'])
                old_gainers_count = len(self.active_tickers['gainers'])

                # Extract ticker symbols from TickerMetrics
                self.active_tickers = {
                    'volume': [m.ticker for m in categories['volume']],
                    'gainers': [m.ticker for m in categories['gainers']]
                }

                # Calculate total unique tickers
                all_tickers = set(self.active_tickers['volume'] + self.active_tickers['gainers'])

                logger.info(
                    f"Updated ticker lists:\n"
                    f"  Volume:  {old_volume_count} -> {len(self.active_tickers['volume'])} tickers\n"
                    f"  Gainers: {old_gainers_count} -> {len(self.active_tickers['gainers'])} tickers\n"
                    f"  Total unique: {len(all_tickers)} tickers"
                )
                logger.info(
                    f"Top 5 volume: {', '.join(self.active_tickers['volume'][:5])}"
                )
                logger.info(
                    f"Top 5 gainers: {', '.join(self.active_tickers['gainers'][:5])}"
                )
            else:
                logger.warning("Failed to get new ticker lists, keeping old lists")

        except Exception as e:
            logger.error(f"Ticker update job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    def job_daily_retraining(self) -> None:
        """
        Daily job: Full model retraining after market close.

        Runs at 5:00 PM ET (after market close at 4:00 PM).
        Performs full retraining on all models with latest data.
        """
        job_name = "daily_retrain"

        try:
            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now()

            logger.info("=" * 80)
            logger.info("DAILY JOB: Full Model Retraining")
            logger.info("=" * 80)

            # Get all unique tickers from both categories
            all_tickers = list(set(self.active_tickers['volume'] + self.active_tickers['gainers']))

            if not all_tickers:
                logger.warning("No active tickers, skipping retraining")
                return

            logger.info(f"Retraining models for {len(all_tickers)} tickers...")

            # Load and prepare data for each ticker
            import pandas as pd
            import numpy as np
            from datetime import timedelta

            successful = 0
            failed = 0

            for i, ticker in enumerate(all_tickers, 1):
                try:
                    logger.info(f"[{i}/{len(all_tickers)}] Retraining {ticker}...")

                    # Load last 30 days of data
                    cutoff_date = datetime.now() - timedelta(days=30)

                    with get_db() as db:
                        from sqlalchemy import select

                        stmt = (
                            select(DBMinuteBar)
                            .where(DBMinuteBar.symbol == ticker)
                            .where(DBMinuteBar.timestamp >= cutoff_date)
                            .order_by(DBMinuteBar.timestamp)
                        )

                        bars = db.execute(stmt).scalars().all()

                        if len(bars) < 1000:
                            logger.warning(f"{ticker}: Insufficient data ({len(bars)} bars)")
                            failed += 1
                            continue

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

                    # Compute features and labels
                    features_df = self.feature_engineer.compute_features(df)
                    feature_names = self.feature_engineer.get_feature_names()

                    labels_up = []
                    labels_down = []

                    for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
                        entry_time = df.iloc[idx]["timestamp"]
                        entry_price = df.iloc[idx]["close"]

                        labels = self.label_generator.generate_labels(
                            df, entry_time, entry_price
                        )

                        labels_up.append(labels["label_up"])
                        labels_down.append(labels["label_down"])

                    # Prepare arrays
                    X = features_df[feature_names].values[: len(labels_up)]
                    y_up = np.array(labels_up)
                    y_down = np.array(labels_down)

                    # Remove NaN rows
                    valid_indices = ~np.isnan(X).any(axis=1)
                    X = X[valid_indices]
                    y_up = y_up[valid_indices]
                    y_down = y_down[valid_indices]

                    if len(X) < 100:
                        logger.warning(f"{ticker}: Too few samples ({len(X)})")
                        failed += 1
                        continue

                    # Train all models
                    results = self.gpu_trainer.train_single_ticker(
                        ticker, X, y_up, y_down
                    )

                    models_success = sum(1 for v in results.values() if v)
                    logger.info(f"{ticker}: {models_success}/{len(results)} models trained")

                    if models_success > 0:
                        successful += 1
                    else:
                        failed += 1

                    # Clear GPU memory
                    self.gpu_trainer.clear_gpu_memory()

                except Exception as e:
                    logger.error(f"{ticker}: Retraining failed: {e}")
                    failed += 1
                    continue

            logger.info("=" * 80)
            logger.info(f"Daily retraining complete: {successful} success, {failed} failed")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Daily retraining job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    def job_keep_alive(self) -> None:
        """
        Keep-alive job to prevent Windows from terminating idle process.

        Runs every 30 seconds to maintain process activity.
        This prevents Windows from marking the process as unresponsive
        and terminating it during long idle periods between scheduled jobs.
        """
        job_name = "keep_alive"
        self.job_stats[job_name] = self.job_stats.get(job_name, {"runs": 0, "errors": 0, "last_run": None})
        self.job_stats[job_name]["runs"] += 1
        self.job_stats[job_name]["last_run"] = datetime.now(ET)
        # Debug level to avoid log spam
        logger.debug("Keep-alive ping")

    def job_auto_train_new_tickers(self) -> None:
        """
        Hourly job: Auto-train new market gainers.

        Runs every hour to:
        1. Discover new market gainers
        2. Collect data for tickers without data
        3. Train models for tickers with data but no models

        Limited to 3 tickers per cycle to avoid overloading.
        """
        job_name = "auto_train"
        self.job_stats[job_name] = self.job_stats.get(job_name, {"runs": 0, "errors": 0, "last_run": None, "trained": 0})

        try:
            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now(ET)

            logger.info("=" * 80)
            logger.info("HOURLY JOB: Auto-Train New Tickers")
            logger.info("=" * 80)

            import pandas as pd
            import numpy as np

            # Get trained tickers from model_manager
            trained_tickers = set(self.model_manager.get_tickers())

            # Get tickers with data from DB
            with get_db() as db:
                from sqlalchemy import select
                stmt = select(DBMinuteBar.symbol).distinct()
                tickers_with_data = set(s.upper() for s in db.execute(stmt).scalars().all())

            # Get current market gainers
            gainers = self.ticker_selector.get_market_top_gainers(limit=20)

            if not gainers:
                logger.info("No market gainers found")
                return

            # Find tickers that need training (have data but no model)
            tickers_need_training = []
            for g in gainers:
                ticker_upper = g.ticker.upper()
                if ticker_upper in tickers_with_data and ticker_upper not in trained_tickers:
                    tickers_need_training.append(g.ticker)

            if not tickers_need_training:
                logger.info("All gainers already have trained models or lack data")
                return

            # Limit to 3 tickers per cycle
            MAX_PER_CYCLE = 3
            tickers_to_train = tickers_need_training[:MAX_PER_CYCLE]

            logger.info(f"Found {len(tickers_need_training)} tickers needing training, "
                       f"processing {len(tickers_to_train)}: {', '.join(tickers_to_train)}")

            trained_count = 0
            for ticker in tickers_to_train:
                try:
                    logger.info(f"Auto-training {ticker}...")

                    # Load last 30 days of data
                    cutoff_date = datetime.now() - timedelta(days=30)

                    with get_db() as db:
                        from sqlalchemy import select

                        stmt = (
                            select(DBMinuteBar)
                            .where(DBMinuteBar.symbol == ticker.upper())
                            .where(DBMinuteBar.timestamp >= cutoff_date)
                            .order_by(DBMinuteBar.timestamp)
                        )
                        bars = db.execute(stmt).scalars().all()

                        if len(bars) < 1000:
                            logger.warning(f"{ticker}: Insufficient data ({len(bars)} bars)")
                            continue

                        # Convert to DataFrame
                        df = pd.DataFrame([{
                            "timestamp": bar.timestamp,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                            "vwap": bar.vwap,
                        } for bar in bars])

                    # Compute features
                    features_df = self.feature_engineer.compute_features(df)
                    feature_names = self.feature_engineer.get_feature_names()

                    # Generate labels
                    labels_up = []
                    labels_down = []

                    for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
                        entry_time = df.iloc[idx]["timestamp"]
                        entry_price = df.iloc[idx]["close"]
                        labels = self.label_generator.generate_labels(df, entry_time, entry_price)
                        labels_up.append(labels["label_up"])
                        labels_down.append(labels["label_down"])

                    # Prepare arrays
                    X = features_df[feature_names].values[:len(labels_up)]
                    y_up = np.array(labels_up)
                    y_down = np.array(labels_down)

                    # Remove NaN rows
                    valid_indices = ~np.isnan(X).any(axis=1)
                    X = X[valid_indices]
                    y_up = y_up[valid_indices]
                    y_down = y_down[valid_indices]

                    if len(X) < 100:
                        logger.warning(f"{ticker}: Too few valid samples ({len(X)})")
                        continue

                    # Train models
                    results = self.gpu_trainer.train_single_ticker(ticker, X, y_up, y_down)

                    models_success = sum(1 for v in results.values() if v)
                    logger.info(f"{ticker}: {models_success}/{len(results)} models trained")

                    if models_success > 0:
                        trained_count += 1
                        # Update model_manager's ticker list
                        if ticker.upper() not in self.model_manager._tickers:
                            self.model_manager._tickers.append(ticker.upper())

                    # Clear GPU memory
                    self.gpu_trainer.clear_gpu_memory()

                except Exception as e:
                    logger.error(f"{ticker}: Auto-training failed: {e}")
                    continue

            self.job_stats[job_name]["trained"] += trained_count
            logger.info(f"Auto-train complete: {trained_count} tickers trained this cycle")

        except Exception as e:
            logger.error(f"Auto-train job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    def job_update_prediction_outcomes(self) -> None:
        """
        Job: Update prediction outcomes for predictions made 60+ minutes ago.

        Runs every 5 minutes to:
        1. Find predictions in prediction_history with outcome=None
        2. Check if enough time has passed (60 minutes)
        3. Look up actual price movement from DB
        4. Update the outcome (hit or miss)

        This enables accurate hit_rate calculation for model performance.
        """
        job_name = "outcome_update"

        try:
            self.job_stats[job_name]["runs"] += 1
            self.job_stats[job_name]["last_run"] = datetime.now(ET)

            logger.debug("OUTCOME UPDATE JOB: Checking pending predictions...")

            horizon_minutes = settings.PREDICTION_HORIZON_MINUTES
            target_percent = settings.TARGET_PERCENT
            cutoff_time = datetime.now() - timedelta(minutes=horizon_minutes + 5)

            total_updated = 0
            total_checked = 0

            # Get all tickers with models
            all_tickers = self.model_manager.get_tickers()

            for ticker in all_tickers:
                try:
                    # Process each target type (up, down)
                    for target in settings.PREDICTION_TARGETS:
                        models = self.model_manager.get_all_models(ticker, target)

                        for model_type, model in models.items():
                            if not hasattr(model, 'prediction_history'):
                                continue

                            # Find predictions needing outcome update
                            pending_predictions = []
                            for pred in model.prediction_history:
                                if (pred.get('actual_outcome') is None and
                                    pred.get('timestamp') is not None and
                                    pred['timestamp'] < cutoff_time):
                                    pending_predictions.append(pred)

                            if not pending_predictions:
                                continue

                            total_checked += len(pending_predictions)

                            # Get price data for outcome determination
                            for pred in pending_predictions:
                                try:
                                    pred_time = pred['timestamp']

                                    # Query DB for price data after prediction
                                    with get_db() as db:
                                        from sqlalchemy import select

                                        # Get the entry price (at prediction time)
                                        stmt_entry = (
                                            select(DBMinuteBar)
                                            .where(DBMinuteBar.symbol == ticker.upper())
                                            .where(DBMinuteBar.timestamp <= pred_time)
                                            .order_by(DBMinuteBar.timestamp.desc())
                                            .limit(1)
                                        )
                                        entry_bar = db.execute(stmt_entry).scalar()

                                        if not entry_bar:
                                            continue

                                        entry_price = entry_bar.close

                                        # Get price data for the prediction horizon
                                        stmt_future = (
                                            select(DBMinuteBar)
                                            .where(DBMinuteBar.symbol == ticker.upper())
                                            .where(DBMinuteBar.timestamp > pred_time)
                                            .where(DBMinuteBar.timestamp <= pred_time + timedelta(minutes=horizon_minutes))
                                            .order_by(DBMinuteBar.timestamp)
                                        )
                                        future_bars = db.execute(stmt_future).scalars().all()

                                        if len(future_bars) < 10:
                                            # Not enough data yet, skip
                                            continue

                                        # Calculate max gain and max loss
                                        max_high = max(bar.high for bar in future_bars)
                                        min_low = min(bar.low for bar in future_bars)

                                        max_gain_pct = (max_high - entry_price) / entry_price * 100
                                        max_loss_pct = (min_low - entry_price) / entry_price * 100

                                        # Determine outcome based on target
                                        if target == "up":
                                            actual_outcome = max_gain_pct >= target_percent
                                        else:  # down
                                            actual_outcome = max_loss_pct <= -target_percent

                                        # Update the prediction outcome
                                        model.update_outcome(pred_time, actual_outcome)
                                        total_updated += 1

                                        logger.debug(
                                            f"{ticker}/{target}/{model_type}: "
                                            f"prediction at {pred_time.strftime('%H:%M')} -> "
                                            f"{'HIT' if actual_outcome else 'MISS'} "
                                            f"(gain={max_gain_pct:.2f}%, loss={max_loss_pct:.2f}%)"
                                        )

                                except Exception as e:
                                    logger.warning(f"Failed to update outcome for {ticker}/{target}: {e}")
                                    continue

                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    continue

            self.job_stats[job_name]["updated"] += total_updated

            if total_updated > 0:
                logger.info(
                    f"Outcome update complete: {total_updated}/{total_checked} predictions updated"
                )

                # Refresh ensemble weights after outcomes are updated
                try:
                    self.model_manager.refresh_all_ensemble_weights()
                except Exception as e:
                    logger.warning(f"Failed to refresh ensemble weights: {e}")
            else:
                logger.debug(f"Outcome update: no pending predictions to update")

        except Exception as e:
            logger.error(f"Outcome update job failed: {e}")
            self.job_stats[job_name]["errors"] += 1

    # ========== SCHEDULER MANAGEMENT ==========

    def start(self) -> None:
        """Start the scheduler with all configured jobs."""
        if self.scheduler is not None:
            logger.warning("Scheduler already running")
            return

        logger.info("Starting NASDAQ prediction scheduler...")

        # Create scheduler
        self.scheduler = BackgroundScheduler(timezone=ET)

        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR,
        )

        # Job 1: Every minute - Data collection and predictions (market hours only)
        self.scheduler.add_job(
            self.job_collect_and_predict,
            trigger=IntervalTrigger(minutes=1, timezone=ET),
            id="minute_collect",
            name="Data Collection & Predictions",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Minute data collection & predictions")

        # Job 2: Every hour - Incremental training
        self.scheduler.add_job(
            self.job_hourly_training,
            trigger=IntervalTrigger(hours=1, timezone=ET),
            id="hourly_training",
            name="Incremental Training",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Hourly incremental training")

        # Job 3: Every hour - Update target tickers
        self.scheduler.add_job(
            self.job_update_tickers,
            trigger=IntervalTrigger(hours=1, timezone=ET),
            id="ticker_update",
            name="Ticker List Update",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Hourly ticker list update")

        # Job 4: Daily at 5 PM ET - Full retraining
        self.scheduler.add_job(
            self.job_daily_retraining,
            trigger=CronTrigger(hour=17, minute=0, timezone=ET),
            id="daily_retrain",
            name="Daily Full Retraining",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Daily retraining at 5:00 PM ET")

        # Job 5: Keep-alive - Prevent Windows from terminating idle process
        self.scheduler.add_job(
            self.job_keep_alive,
            trigger=IntervalTrigger(seconds=30, timezone=ET),
            id="keep_alive",
            name="Process Keep-Alive",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Keep-alive ping every 30 seconds")

        # Job 6: Auto-train new tickers - Train models for new market gainers
        self.scheduler.add_job(
            self.job_auto_train_new_tickers,
            trigger=IntervalTrigger(hours=1, timezone=ET),
            id="auto_train",
            name="Auto-Train New Tickers",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Auto-train new tickers every hour")

        # Job 7: Update prediction outcomes - Check outcomes for 60+ minute old predictions
        self.scheduler.add_job(
            self.job_update_prediction_outcomes,
            trigger=IntervalTrigger(minutes=5, timezone=ET),
            id="outcome_update",
            name="Update Prediction Outcomes",
            max_instances=1,
            coalesce=True,
        )
        logger.info("✓ Scheduled: Update prediction outcomes every 5 minutes")

        # Initialize ticker list immediately
        logger.info("\nInitializing ticker list...")
        self.job_update_tickers()

        # Start scheduler
        self.scheduler.start()
        logger.success("\nScheduler started successfully!")

        # Display next job times
        self._print_next_run_times()

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler is None:
            logger.warning("Scheduler not running")
            return

        logger.info("Stopping scheduler...")
        self.scheduler.shutdown(wait=True)
        self.scheduler = None
        logger.success("Scheduler stopped")

    def _job_executed_listener(self, event) -> None:
        """
        Listen for job execution events.

        Args:
            event: APScheduler event
        """
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.debug(f"Job {event.job_id} completed successfully")

    def _print_next_run_times(self) -> None:
        """Print next run times for all jobs."""
        if self.scheduler is None:
            return

        logger.info("\nNext scheduled run times:")
        logger.info("-" * 80)

        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            if next_run:
                logger.info(f"  {job.name:30s} -> {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        logger.info("-" * 80)

    def get_status(self) -> dict:
        """
        Get scheduler status and statistics.

        Returns:
            Dictionary with status information
        """
        if self.scheduler is None:
            return {"running": False}

        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        job.next_run_time.isoformat()
                        if job.next_run_time
                        else None
                    ),
                }
            )

        # Get all unique tickers
        all_tickers = list(set(self.active_tickers['volume'] + self.active_tickers['gainers']))

        return {
            "running": True,
            "active_tickers": len(all_tickers),
            "tickers_by_category": {
                "volume": self.active_tickers['volume'][:20],  # First 20
                "gainers": self.active_tickers['gainers'][:20],
            },
            "total_unique_tickers": len(all_tickers),
            "jobs": jobs_info,
            "job_stats": self.job_stats,
            "market_hours": self.is_market_hours(),
        }

    def pause(self) -> None:
        """Pause all jobs."""
        if self.scheduler:
            self.scheduler.pause()
            logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all jobs."""
        if self.scheduler:
            self.scheduler.resume()
            logger.info("Scheduler resumed")
