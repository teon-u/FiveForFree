#!/usr/bin/env python3
"""
Data Verification Script for NASDAQ Prediction System

Quick check of data collection and model status.

Usage:
    python scripts/check_data.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_manager import ModelManager
from src.utils.database import get_db, MinuteBar, Ticker
from sqlalchemy import select, func


def main():
    print("=" * 60)
    print("NASDAQ Prediction System - Data Status Check")
    print("=" * 60)
    print()

    # 1. Check trained models
    print("[1] Trained Models")
    print("-" * 40)
    mm = ModelManager()
    loaded = mm.load_all_models()
    trained_tickers = mm.get_tickers()
    print(f"    Model directory: {mm.models_path}")
    print(f"    Loaded models: {loaded}")
    print(f"    Tickers with models: {len(trained_tickers)}")
    if trained_tickers:
        print(f"    Sample tickers: {', '.join(sorted(trained_tickers)[:10])}")
    print()

    # 2. Check database
    print("[2] Database Status")
    print("-" * 40)
    with get_db() as db:
        # Ticker count
        ticker_count = db.execute(select(func.count(Ticker.id))).scalar()
        print(f"    Registered tickers: {ticker_count}")

        # Minute bar stats
        bar_count = db.execute(select(func.count(MinuteBar.id))).scalar()
        print(f"    Total minute bars: {bar_count:,}")

        # Tickers with data
        symbols_stmt = select(MinuteBar.symbol).distinct()
        symbols = db.execute(symbols_stmt).scalars().all()
        print(f"    Tickers with bar data: {len(symbols)}")

        # Date range
        min_date = db.execute(select(func.min(MinuteBar.timestamp))).scalar()
        max_date = db.execute(select(func.max(MinuteBar.timestamp))).scalar()
        if min_date and max_date:
            days = (max_date - min_date).days
            print(f"    Data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({days} days)")

        # Bar count per ticker (top 10)
        print()
        print("    Top 10 tickers by bar count:")
        bar_count_stmt = (
            select(MinuteBar.symbol, func.count(MinuteBar.id).label('count'))
            .group_by(MinuteBar.symbol)
            .order_by(func.count(MinuteBar.id).desc())
            .limit(10)
        )
        for row in db.execute(bar_count_stmt):
            print(f"      {row.symbol}: {row.count:,} bars")
    print()

    # 3. Check coverage
    print("[3] Model Coverage Analysis")
    print("-" * 40)
    tickers_with_data = set(symbols)
    tickers_with_models = set(trained_tickers)

    both = tickers_with_data & tickers_with_models
    data_only = tickers_with_data - tickers_with_models
    model_only = tickers_with_models - tickers_with_data

    print(f"    Both data and model: {len(both)} tickers")
    print(f"    Data only (no model): {len(data_only)} tickers")
    print(f"    Model only (no data): {len(model_only)} tickers")

    if data_only:
        print(f"    Data-only tickers: {', '.join(sorted(data_only)[:5])}...")
    if model_only:
        print(f"    Model-only tickers: {', '.join(sorted(model_only)[:5])}...")
    print()

    # 4. Summary
    print("[4] Summary & Recommendations")
    print("-" * 40)

    issues = []
    if len(trained_tickers) == 0:
        issues.append("No trained models - run: python scripts/train_all_models.py")
    if bar_count < 50000:
        issues.append("Insufficient data - run: python scripts/collect_historical.py --days 60")
    if len(data_only) > 0:
        issues.append(f"{len(data_only)} tickers need model training")

    if issues:
        print("    Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("    System is healthy!")
        print(f"    - {len(both)} tickers ready for predictions")
        print(f"    - {bar_count:,} minute bars available")
        print(f"    - Data spanning {days} days")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
