#!/usr/bin/env python3
"""
Initialize Database Schema and Tables

Creates all required database tables for the NASDAQ prediction system:
- tickers: Stock ticker information
- minute_bars: 1-minute OHLCV data
- predictions: Model prediction results
- trades: Executed trades for backtesting
- model_performance: Model performance metrics

Usage:
    python scripts/init_database.py [--reset] [--verbose]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from sqlalchemy import inspect, text

from config.settings import settings
from src.utils.database import (
    init_db,
    Base,
    Ticker,
    MinuteBar,
    Prediction,
    Trade,
    ModelPerformance,
)
import src.utils.database as db_module


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


def check_database_exists() -> bool:
    """
    Check if database tables already exist.

    Returns:
        True if tables exist, False otherwise
    """
    try:
        inspector = inspect(db_module.engine)
        existing_tables = inspector.get_table_names()
        required_tables = {
            "tickers",
            "minute_bars",
            "predictions",
            "trades",
            "model_performance",
        }

        if existing_tables:
            logger.info(f"Found existing tables: {existing_tables}")
            return required_tables.issubset(set(existing_tables))

        return False

    except Exception as e:
        logger.warning(f"Could not check existing tables: {e}")
        return False


def drop_all_tables() -> None:
    """Drop all existing tables (use with caution!)."""
    logger.warning("Dropping all existing tables...")

    try:
        Base.metadata.drop_all(bind=db_module.engine)
        logger.info("All tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


def create_all_tables() -> None:
    """Create all database tables."""
    logger.info("Creating database tables...")

    try:
        Base.metadata.create_all(bind=db_module.engine)
        logger.info("All tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def verify_tables() -> bool:
    """
    Verify that all required tables were created.

    Returns:
        True if all tables exist, False otherwise
    """
    try:
        inspector = inspect(db_module.engine)
        existing_tables = set(inspector.get_table_names())
        required_tables = {
            "tickers",
            "minute_bars",
            "predictions",
            "trades",
            "model_performance",
        }

        missing_tables = required_tables - existing_tables

        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False

        logger.success("All required tables exist")

        # Verify table structure
        for table_name in required_tables:
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            logger.debug(
                f"Table '{table_name}': {len(columns)} columns, {len(indexes)} indexes"
            )

        return True

    except Exception as e:
        logger.error(f"Table verification failed: {e}")
        return False


def get_table_stats() -> dict:
    """
    Get statistics about database tables.

    Returns:
        Dictionary with table row counts
    """
    from src.utils.database import get_db

    stats = {}

    try:
        with get_db() as db:
            # Count rows in each table
            for table_name in [
                "tickers",
                "minute_bars",
                "predictions",
                "trades",
                "model_performance",
            ]:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                stats[table_name] = count

        return stats

    except Exception as e:
        logger.error(f"Failed to get table stats: {e}")
        return {}


def create_data_directories() -> None:
    """Create required data directories."""
    directories = [
        Path("./data"),
        Path("./data/models"),
        Path("./logs"),
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.debug(f"Directory already exists: {directory}")


def main() -> int:
    """
    Main function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Initialize database schema for NASDAQ prediction system"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing tables and recreate (WARNING: destroys all data)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify tables exist, don't create",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("NASDAQ Prediction System - Database Initialization")
    logger.info("=" * 80)
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    logger.info("")

    try:
        # Create data directories
        create_data_directories()

        # Initialize database connection
        logger.info("Initializing database connection...")
        init_db()
        logger.success("Database connection established")

        # Check if database already exists
        tables_exist = check_database_exists()

        if args.verify_only:
            # Just verify and report
            if tables_exist:
                logger.success("Database verification passed")
                stats = get_table_stats()
                logger.info("\nTable Statistics:")
                for table, count in stats.items():
                    logger.info(f"  {table}: {count:,} rows")
                return 0
            else:
                logger.error("Database verification failed")
                return 1

        if tables_exist and not args.reset:
            logger.warning("Database tables already exist")
            response = input("Do you want to continue anyway? (y/N): ")
            if response.lower() != "y":
                logger.info("Aborted by user")
                return 0

        # Reset database if requested
        if args.reset:
            logger.warning("RESET MODE: All existing data will be destroyed!")
            response = input("Are you sure you want to continue? (yes/N): ")
            if response.lower() != "yes":
                logger.info("Aborted by user")
                return 0

            drop_all_tables()

        # Create tables
        create_all_tables()

        # Verify tables were created
        if not verify_tables():
            logger.error("Table verification failed")
            return 1

        # Get and display table stats
        stats = get_table_stats()
        logger.info("\nDatabase initialized successfully!")
        logger.info("\nTable Statistics:")
        for table, count in stats.items():
            logger.info(f"  {table}: {count:,} rows")

        logger.info("\n" + "=" * 80)
        logger.success("Database initialization complete!")
        logger.info("=" * 80)

        return 0

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
