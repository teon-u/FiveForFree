#!/usr/bin/env python3
"""
Collect Historical Data for NASDAQ Prediction System

Fetches historical minute bar data for target tickers and stores in database.
Supports both top volume tickers and specific ticker lists.

Usage:
    # Collect 30 days of data for top tickers
    python scripts/collect_historical.py

    # Collect specific number of days
    python scripts/collect_historical.py --days 60

    # Collect for specific tickers
    python scripts/collect_historical.py --tickers AAPL MSFT GOOGL

    # Update existing data (collect only missing bars)
    python scripts/collect_historical.py --update
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from sqlalchemy import select

from config.settings import settings
from src.utils.database import (
    get_db,
    get_or_create_ticker,
    MinuteBar as DBMinuteBar,
)
from src.collector.minute_bars import MinuteBarCollector
from src.collector.ticker_selector import TickerSelector


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
    log_file = Path("./logs") / f"collect_historical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="100 MB")


def get_target_tickers(
    custom_tickers: Optional[List[str]] = None,
) -> List[str]:
    """
    Get list of tickers to collect data for.

    Args:
        custom_tickers: Optional list of specific tickers

    Returns:
        List of ticker symbols
    """
    if custom_tickers:
        logger.info(f"Using custom ticker list: {custom_tickers}")
        return custom_tickers

    # Use ticker selector to get top tickers
    logger.info("Selecting target tickers based on volume and gainers...")
    selector = TickerSelector()
    tickers = selector.get_target_tickers()

    logger.info(f"Selected {len(tickers)} target tickers")
    return tickers


def get_existing_data_range(ticker: str) -> Optional[tuple]:
    """
    Get the date range of existing data for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (earliest_timestamp, latest_timestamp) or None
    """
    try:
        with get_db() as db:
            stmt = (
                select(DBMinuteBar.timestamp)
                .where(DBMinuteBar.symbol == ticker)
                .order_by(DBMinuteBar.timestamp)
            )
            result = db.execute(stmt).scalars().all()

            if result:
                return result[0], result[-1]

        return None

    except Exception as e:
        logger.error(f"Failed to get existing data range for {ticker}: {e}")
        return None


def collect_ticker_data(
    ticker: str,
    days: int,
    collector: MinuteBarCollector,
    update_mode: bool = False,
) -> dict:
    """
    Collect historical data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to collect
        collector: MinuteBarCollector instance
        update_mode: If True, only collect missing data

    Returns:
        Dictionary with collection statistics
    """
    stats = {
        "ticker": ticker,
        "success": False,
        "bars_collected": 0,
        "bars_saved": 0,
        "errors": 0,
    }

    try:
        # Determine date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if update_mode:
            # Check existing data
            existing_range = get_existing_data_range(ticker)
            if existing_range:
                latest_timestamp = existing_range[1]
                start_date = latest_timestamp + timedelta(minutes=1)
                logger.info(
                    f"{ticker}: Updating from {start_date.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.info(f"{ticker}: No existing data, collecting full history")
        else:
            logger.info(
                f"{ticker}: Collecting {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            )

        # Collect minute bars
        bars = collector.get_bars(ticker, start_date, end_date)

        if not bars:
            logger.warning(f"{ticker}: No data collected")
            return stats

        stats["bars_collected"] = len(bars)
        logger.info(f"{ticker}: Collected {len(bars)} bars")

        # Save to database
        with get_db() as db:
            # Get or create ticker record
            ticker_record = get_or_create_ticker(db, ticker)

            # Insert minute bars
            saved_count = 0
            for bar in bars:
                try:
                    # Convert to database model
                    db_bar = DBMinuteBar(
                        ticker_id=ticker_record.id,
                        symbol=ticker,
                        timestamp=bar.datetime,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        vwap=bar.vwap,
                        trade_count=bar.transactions,
                    )

                    db.add(db_bar)
                    saved_count += 1

                    # Commit in batches
                    if saved_count % 1000 == 0:
                        db.commit()
                        logger.debug(f"{ticker}: Saved {saved_count} bars...")

                except Exception as e:
                    # Skip duplicate or invalid bars
                    logger.debug(f"{ticker}: Failed to save bar: {e}")
                    stats["errors"] += 1
                    continue

            # Final commit
            db.commit()

        stats["bars_saved"] = saved_count
        stats["success"] = True
        logger.success(f"{ticker}: Saved {saved_count}/{len(bars)} bars to database")

    except Exception as e:
        logger.error(f"{ticker}: Collection failed: {e}")
        stats["errors"] += 1

    return stats


def main() -> int:
    """
    Main function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Collect historical data for NASDAQ prediction system"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=settings.HISTORICAL_DAYS,
        help=f"Number of days to collect (default: {settings.HISTORICAL_DAYS})",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific tickers to collect (default: top volume + gainers)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update mode: only collect missing data since last collection",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        help="Maximum number of tickers to process",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("NASDAQ Prediction System - Historical Data Collection")
    logger.info("=" * 80)
    logger.info(f"Days to collect: {args.days}")
    logger.info(f"Update mode: {args.update}")
    logger.info("")

    try:
        # Get target tickers
        tickers = get_target_tickers(args.tickers)

        if not tickers:
            logger.error("No tickers to process")
            return 1

        # Apply max tickers limit if specified
        if args.max_tickers:
            tickers = tickers[: args.max_tickers]
            logger.info(f"Limited to {len(tickers)} tickers")

        logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers[:10])}")
        if len(tickers) > 10:
            logger.info(f"... and {len(tickers) - 10} more")
        logger.info("")

        # Initialize collector
        collector = MinuteBarCollector()

        # Collect data for each ticker
        all_stats = []
        successful = 0
        failed = 0

        start_time = datetime.now()

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

            stats = collect_ticker_data(
                ticker,
                args.days,
                collector,
                update_mode=args.update,
            )

            all_stats.append(stats)

            if stats["success"]:
                successful += 1
            else:
                failed += 1

            # Progress update
            if i % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / i
                remaining = (len(tickers) - i) * avg_time
                logger.info(
                    f"\nProgress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%) "
                    f"| Success: {successful} | Failed: {failed} "
                    f"| ETA: {remaining/60:.1f} min"
                )

        # Summary
        elapsed_total = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("Collection Summary")
        logger.info("=" * 80)
        logger.info(f"Total tickers processed: {len(tickers)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(
            f"Total bars collected: {sum(s['bars_collected'] for s in all_stats):,}"
        )
        logger.info(
            f"Total bars saved: {sum(s['bars_saved'] for s in all_stats):,}"
        )
        logger.info(f"Total errors: {sum(s['errors'] for s in all_stats)}")
        logger.info(f"Time elapsed: {elapsed_total/60:.1f} minutes")
        logger.info("=" * 80)

        # List failed tickers
        if failed > 0:
            failed_tickers = [s["ticker"] for s in all_stats if not s["success"]]
            logger.warning(f"\nFailed tickers: {', '.join(failed_tickers)}")

        if successful > 0:
            logger.success("\nHistorical data collection complete!")
            return 0
        else:
            logger.error("\nAll collections failed!")
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
