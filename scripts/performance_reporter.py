"""
Performance Reporter - Automated Trading Performance Analysis

Generates comprehensive performance reports:
- Precision (When signal given, how often correct)
- Win Rate (Profitable trades / Total trades)
- Signal Rate (How often signals are generated)
- Daily/Weekly/Monthly reports
- Model comparison analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from loguru import logger
from config.settings import settings

# Paths
DATA_DIR = Path(settings.DATA_DIR)
REPORTS_DIR = DATA_DIR / "reports"
PAPER_TRADING_DB = DATA_DIR / "paper_trading.db"
SIGNALS_DB = DATA_DIR / "signals.db"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    total_signals: int
    actionable_signals: int  # BUY/SELL only
    total_trades: int
    wins: int
    losses: int
    precision: float  # wins / actionable_signals
    win_rate: float  # wins / total_trades
    signal_rate: float  # actionable_signals / total_signals
    avg_pnl: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float


class PerformanceReporter:
    """Generates performance reports from trading data."""

    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a date range."""
        # Get signals data
        signal_stats = self._get_signal_stats(start_date, end_date)

        # Get trade data
        trade_stats = self._get_trade_stats(start_date, end_date)

        # Calculate derived metrics
        total_signals = signal_stats.get("total", 0)
        actionable_signals = signal_stats.get("actionable", 0)
        total_trades = trade_stats.get("total", 0)
        wins = trade_stats.get("wins", 0)
        losses = trade_stats.get("losses", 0)

        # Precision: When we signal, how often are we right?
        precision = wins / actionable_signals if actionable_signals > 0 else 0.0

        # Win Rate: Of trades taken, how many profitable?
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Signal Rate: How often do we generate actionable signals?
        signal_rate = actionable_signals / total_signals if total_signals > 0 else 0.0

        # PnL metrics
        avg_pnl = trade_stats.get("avg_pnl", 0.0)
        total_pnl = trade_stats.get("total_pnl", 0.0)
        max_drawdown = trade_stats.get("max_drawdown", 0.0)

        # Sharpe ratio (simplified - assuming daily returns)
        sharpe = self._calculate_sharpe(start_date, end_date)

        return PerformanceMetrics(
            total_signals=total_signals,
            actionable_signals=actionable_signals,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            precision=precision,
            win_rate=win_rate,
            signal_rate=signal_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe
        )

    def _get_signal_stats(self, start: datetime, end: datetime) -> Dict:
        """Get signal statistics from signals database."""
        if not SIGNALS_DB.exists():
            return {"total": 0, "actionable": 0}

        with sqlite3.connect(SIGNALS_DB) as conn:
            total = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp BETWEEN ? AND ?
            """, (start.isoformat(), end.isoformat())).fetchone()[0]

            actionable = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp BETWEEN ? AND ?
                AND direction IN ('BUY', 'SELL')
            """, (start.isoformat(), end.isoformat())).fetchone()[0]

            return {"total": total, "actionable": actionable}

    def _get_trade_stats(self, start: datetime, end: datetime) -> Dict:
        """Get trade statistics from paper trading database."""
        if not PAPER_TRADING_DB.exists():
            return {"total": 0, "wins": 0, "losses": 0, "avg_pnl": 0.0, "total_pnl": 0.0}

        with sqlite3.connect(PAPER_TRADING_DB) as conn:
            # Closed trades in period
            total = conn.execute("""
                SELECT COUNT(*) FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed'
            """, (start.isoformat(), end.isoformat())).fetchone()[0]

            wins = conn.execute("""
                SELECT COUNT(*) FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed' AND outcome = 'win'
            """, (start.isoformat(), end.isoformat())).fetchone()[0]

            # PnL stats
            pnl_stats = conn.execute("""
                SELECT AVG(pnl_percent), SUM(pnl_percent)
                FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed'
            """, (start.isoformat(), end.isoformat())).fetchone()

            avg_pnl = pnl_stats[0] or 0.0
            total_pnl = pnl_stats[1] or 0.0

            # Max drawdown (simplified)
            max_drawdown = self._calculate_max_drawdown(conn, start, end)

            return {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
                "max_drawdown": max_drawdown
            }

    def _calculate_max_drawdown(self, conn, start: datetime, end: datetime) -> float:
        """Calculate maximum drawdown from trade history."""
        cursor = conn.execute("""
            SELECT pnl_percent FROM paper_trades
            WHERE entry_time BETWEEN ? AND ?
            AND status = 'closed'
            ORDER BY entry_time
        """, (start.isoformat(), end.isoformat()))

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for (pnl,) in cursor:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe(self, start: datetime, end: datetime) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if not PAPER_TRADING_DB.exists():
            return 0.0

        with sqlite3.connect(PAPER_TRADING_DB) as conn:
            cursor = conn.execute("""
                SELECT DATE(entry_time) as date, SUM(pnl_percent) as daily_pnl
                FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed'
                GROUP BY DATE(entry_time)
            """, (start.isoformat(), end.isoformat()))

            daily_returns = [row[1] for row in cursor.fetchall()]

        if len(daily_returns) < 2:
            return 0.0

        import statistics
        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1.0

        # Annualized Sharpe (assuming 252 trading days)
        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * (252 ** 0.5)
        return sharpe

    def get_ticker_performance(
        self,
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get performance breakdown by ticker."""
        if not PAPER_TRADING_DB.exists():
            return []

        with sqlite3.connect(PAPER_TRADING_DB) as conn:
            cursor = conn.execute("""
                SELECT
                    ticker,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percent) as avg_pnl,
                    SUM(pnl_percent) as total_pnl
                FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed'
                GROUP BY ticker
                ORDER BY total_pnl DESC
            """, (start.isoformat(), end.isoformat()))

            results = []
            for row in cursor.fetchall():
                ticker, total, wins, avg_pnl, total_pnl = row
                results.append({
                    "ticker": ticker,
                    "total_trades": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate": wins / total if total > 0 else 0.0,
                    "avg_pnl": avg_pnl or 0.0,
                    "total_pnl": total_pnl or 0.0
                })

            return results

    def get_model_performance(
        self,
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get performance breakdown by model type."""
        if not PAPER_TRADING_DB.exists():
            return []

        with sqlite3.connect(PAPER_TRADING_DB) as conn:
            cursor = conn.execute("""
                SELECT
                    model_type,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percent) as avg_pnl,
                    SUM(pnl_percent) as total_pnl
                FROM paper_trades
                WHERE entry_time BETWEEN ? AND ?
                AND status = 'closed'
                GROUP BY model_type
                ORDER BY total_pnl DESC
            """, (start.isoformat(), end.isoformat()))

            results = []
            for row in cursor.fetchall():
                model, total, wins, avg_pnl, total_pnl = row
                results.append({
                    "model": model,
                    "total_trades": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate": wins / total if total > 0 else 0.0,
                    "avg_pnl": avg_pnl or 0.0,
                    "total_pnl": total_pnl or 0.0
                })

            return results

    def generate_daily_report(self, date: datetime = None) -> Dict:
        """Generate daily performance report."""
        if date is None:
            date = datetime.now()

        start = datetime(date.year, date.month, date.day, 0, 0, 0)
        end = start + timedelta(days=1)

        metrics = self.calculate_metrics(start, end)
        ticker_perf = self.get_ticker_performance(start, end)
        model_perf = self.get_model_performance(start, end)

        report = {
            "report_type": "daily",
            "date": date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "total_signals": metrics.total_signals,
                "actionable_signals": metrics.actionable_signals,
                "total_trades": metrics.total_trades,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "precision": round(metrics.precision, 4),
                "win_rate": round(metrics.win_rate, 4),
                "signal_rate": round(metrics.signal_rate, 4),
                "avg_pnl": round(metrics.avg_pnl, 4),
                "total_pnl": round(metrics.total_pnl, 4),
            },
            "top_tickers": ticker_perf[:10],
            "model_performance": model_perf,
        }

        # Save report
        report_file = self.reports_dir / f"daily_{date.strftime('%Y-%m-%d')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Daily report saved: {report_file}")
        return report

    def generate_weekly_report(self, end_date: datetime = None) -> Dict:
        """Generate weekly performance report."""
        if end_date is None:
            end_date = datetime.now()

        end = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
        start = end - timedelta(days=7)

        metrics = self.calculate_metrics(start, end)
        ticker_perf = self.get_ticker_performance(start, end)
        model_perf = self.get_model_performance(start, end)

        report = {
            "report_type": "weekly",
            "period": {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d")
            },
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "total_signals": metrics.total_signals,
                "actionable_signals": metrics.actionable_signals,
                "total_trades": metrics.total_trades,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "precision": round(metrics.precision, 4),
                "win_rate": round(metrics.win_rate, 4),
                "signal_rate": round(metrics.signal_rate, 4),
                "avg_pnl": round(metrics.avg_pnl, 4),
                "total_pnl": round(metrics.total_pnl, 4),
                "max_drawdown": round(metrics.max_drawdown, 4),
                "sharpe_ratio": round(metrics.sharpe_ratio, 4),
            },
            "top_tickers": ticker_perf[:20],
            "model_performance": model_perf,
        }

        # Save report
        report_file = self.reports_dir / f"weekly_{end.strftime('%Y-%m-%d')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Weekly report saved: {report_file}")
        return report

    def print_report(self, report: Dict):
        """Print report to console in formatted way."""
        print("\n" + "=" * 60)
        print(f"  PERFORMANCE REPORT - {report['report_type'].upper()}")
        print("=" * 60)

        if report['report_type'] == 'daily':
            print(f"Date: {report['date']}")
        else:
            print(f"Period: {report['period']['start']} to {report['period']['end']}")

        print("-" * 60)
        m = report['metrics']

        print("\n[SIGNALS]")
        print(f"  Total Signals:      {m['total_signals']}")
        print(f"  Actionable:         {m['actionable_signals']}")
        print(f"  Signal Rate:        {m['signal_rate']:.1%}")

        print("\n[TRADES]")
        print(f"  Total Trades:       {m['total_trades']}")
        print(f"  Wins:               {m['wins']}")
        print(f"  Losses:             {m['losses']}")
        print(f"  Win Rate:           {m['win_rate']:.1%}")
        print(f"  Precision:          {m['precision']:.1%}")

        print("\n[PnL]")
        print(f"  Average PnL:        {m['avg_pnl']:+.2f}%")
        print(f"  Total PnL:          {m['total_pnl']:+.2f}%")

        if 'max_drawdown' in m:
            print(f"  Max Drawdown:       {m['max_drawdown']:.2f}%")
        if 'sharpe_ratio' in m:
            print(f"  Sharpe Ratio:       {m['sharpe_ratio']:.2f}")

        if report['top_tickers']:
            print("\n[TOP TICKERS]")
            for t in report['top_tickers'][:5]:
                print(f"  {t['ticker']:6s}  {t['total_trades']} trades  "
                      f"{t['win_rate']:.0%} WR  {t['total_pnl']:+.2f}% PnL")

        if report['model_performance']:
            print("\n[MODEL PERFORMANCE]")
            for m in report['model_performance']:
                print(f"  {m['model']:12s}  {m['total_trades']} trades  "
                      f"{m['win_rate']:.0%} WR  {m['total_pnl']:+.2f}% PnL")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Reporter")
    parser.add_argument("--daily", action="store_true",
                        help="Generate daily report")
    parser.add_argument("--weekly", action="store_true",
                        help="Generate weekly report")
    parser.add_argument("--date", type=str,
                        help="Date for report (YYYY-MM-DD)")
    parser.add_argument("--output", type=str,
                        help="Output file path")

    args = parser.parse_args()

    reporter = PerformanceReporter()

    # Parse date if provided
    report_date = None
    if args.date:
        report_date = datetime.strptime(args.date, "%Y-%m-%d")

    if args.weekly:
        report = reporter.generate_weekly_report(report_date)
    else:
        # Default to daily
        report = reporter.generate_daily_report(report_date)

    reporter.print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
