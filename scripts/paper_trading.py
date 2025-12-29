"""
Paper Trading Script for Real-time Prediction Testing

Monitors predictions and records virtual trades:
- Entry when probability >= 70%
- Records entry price, time, direction
- Checks outcome after 1 hour
- Stores results in SQLite database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

from loguru import logger
from config.settings import settings

# Database path
DB_PATH = Path(settings.DATA_DIR) / "paper_trading.db"


@dataclass
class PaperTrade:
    """Represents a paper trade."""
    id: Optional[int]
    ticker: str
    direction: str  # "up" or "down"
    entry_time: datetime
    entry_price: float
    entry_probability: float
    model_type: str
    target_price: float  # entry_price * (1 + target_percent) for up
    stop_loss_price: float  # entry_price * (1 - stop_loss_percent) for up
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # "win", "loss", "timeout"
    pnl_percent: Optional[float] = None
    status: str = "open"  # "open", "closed"


class PaperTradingDB:
    """Database manager for paper trading records."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_probability REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss_price REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    outcome TEXT,
                    pnl_percent REAL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON paper_trades(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ticker_time
                ON paper_trades(ticker, entry_time)
            """)

            conn.commit()

    def record_entry(self, trade: PaperTrade) -> int:
        """Record a new trade entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO paper_trades
                (ticker, direction, entry_time, entry_price, entry_probability,
                 model_type, target_price, stop_loss_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.ticker,
                trade.direction,
                trade.entry_time.isoformat(),
                trade.entry_price,
                trade.entry_probability,
                trade.model_type,
                trade.target_price,
                trade.stop_loss_price,
                trade.status
            ))
            conn.commit()
            return cursor.lastrowid

    def update_exit(self, trade_id: int, exit_time: datetime, exit_price: float,
                    outcome: str, pnl_percent: float):
        """Update trade with exit information."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE paper_trades
                SET exit_time = ?, exit_price = ?, outcome = ?,
                    pnl_percent = ?, status = 'closed'
                WHERE id = ?
            """, (
                exit_time.isoformat(),
                exit_price,
                outcome,
                pnl_percent,
                trade_id
            ))
            conn.commit()

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM paper_trades WHERE status = 'open'
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_by_date(self, date: datetime) -> List[Dict]:
        """Get all trades for a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM paper_trades
                WHERE entry_time LIKE ?
                ORDER BY entry_time DESC
            """, (f"{date_str}%",))
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Total trades
            total = conn.execute("""
                SELECT COUNT(*) FROM paper_trades
                WHERE entry_time >= ? AND status = 'closed'
            """, (cutoff,)).fetchone()[0]

            # Wins
            wins = conn.execute("""
                SELECT COUNT(*) FROM paper_trades
                WHERE entry_time >= ? AND status = 'closed' AND outcome = 'win'
            """, (cutoff,)).fetchone()[0]

            # Average PnL
            avg_pnl = conn.execute("""
                SELECT AVG(pnl_percent) FROM paper_trades
                WHERE entry_time >= ? AND status = 'closed'
            """, (cutoff,)).fetchone()[0] or 0.0

            # By direction
            up_trades = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE entry_time >= ? AND status = 'closed' AND direction = 'up'
            """, (cutoff,)).fetchone()

            down_trades = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)
                FROM paper_trades
                WHERE entry_time >= ? AND status = 'closed' AND direction = 'down'
            """, (cutoff,)).fetchone()

            return {
                "period_days": days,
                "total_trades": total,
                "wins": wins,
                "losses": total - wins if total > 0 else 0,
                "win_rate": wins / total if total > 0 else 0.0,
                "avg_pnl_percent": avg_pnl,
                "up_trades": up_trades[0] or 0,
                "up_wins": up_trades[1] or 0,
                "down_trades": down_trades[0] or 0,
                "down_wins": down_trades[1] or 0,
            }


class PaperTrader:
    """Paper trading engine."""

    def __init__(
        self,
        probability_threshold: float = 0.70,
        target_percent: float = None,
        stop_loss_percent: float = None,
        horizon_minutes: int = 60
    ):
        self.probability_threshold = probability_threshold
        self.target_percent = target_percent or settings.TARGET_PERCENT
        self.stop_loss_percent = stop_loss_percent or settings.STOP_LOSS_PERCENT
        self.horizon_minutes = horizon_minutes

        self.db = PaperTradingDB()

        # Initialize predictor (lazy load)
        self._predictor = None
        self._model_manager = None

    def _get_predictor(self):
        """Lazy load predictor."""
        if self._predictor is None:
            from src.models.model_manager import ModelManager
            from src.predictor.realtime_predictor import RealtimePredictor
            from src.collector.minute_bars import MinuteBarCollector

            self._model_manager = ModelManager()
            minute_bar_collector = MinuteBarCollector()

            self._predictor = RealtimePredictor(
                model_manager=self._model_manager,
                minute_bar_collector=minute_bar_collector
            )
        return self._predictor

    def check_entry_signal(self, ticker: str) -> Optional[PaperTrade]:
        """
        Check if ticker has entry signal (probability >= threshold).

        Returns PaperTrade if signal found, None otherwise.
        """
        try:
            predictor = self._get_predictor()
            result = predictor.predict(ticker, include_all_models=False)

            # Check UP signal
            if result.up_probability >= self.probability_threshold:
                target_price = result.current_price * (1 + self.target_percent / 100)
                stop_loss_price = result.current_price * (1 - self.stop_loss_percent / 100)

                return PaperTrade(
                    id=None,
                    ticker=ticker,
                    direction="up",
                    entry_time=result.timestamp,
                    entry_price=result.current_price,
                    entry_probability=result.up_probability,
                    model_type=result.best_up_model,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price
                )

            # Check DOWN signal
            if result.down_probability >= self.probability_threshold:
                target_price = result.current_price * (1 - self.target_percent / 100)
                stop_loss_price = result.current_price * (1 + self.stop_loss_percent / 100)

                return PaperTrade(
                    id=None,
                    ticker=ticker,
                    direction="down",
                    entry_time=result.timestamp,
                    entry_price=result.current_price,
                    entry_probability=result.down_probability,
                    model_type=result.best_down_model,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to check signal for {ticker}: {e}")
            return None

    def enter_trade(self, trade: PaperTrade) -> int:
        """Record trade entry."""
        trade_id = self.db.record_entry(trade)
        logger.info(
            f"[ENTRY] {trade.ticker} {trade.direction.upper()} @ ${trade.entry_price:.2f} "
            f"(prob={trade.entry_probability:.1%}, model={trade.model_type})"
        )
        return trade_id

    def check_and_close_trades(self) -> List[Dict]:
        """
        Check open trades and close those that:
        1. Hit target price
        2. Hit stop loss
        3. Exceeded time horizon (1 hour)
        """
        closed_trades = []
        open_trades = self.db.get_open_trades()

        for trade_dict in open_trades:
            trade_id = trade_dict["id"]
            ticker = trade_dict["ticker"]
            direction = trade_dict["direction"]
            entry_time = datetime.fromisoformat(trade_dict["entry_time"])
            entry_price = trade_dict["entry_price"]
            target_price = trade_dict["target_price"]
            stop_loss_price = trade_dict["stop_loss_price"]

            # Get current price
            try:
                predictor = self._get_predictor()
                result = predictor.predict(ticker, include_all_models=False)
                current_price = result.current_price
                current_time = result.timestamp
            except Exception as e:
                logger.warning(f"Failed to get price for {ticker}: {e}")
                continue

            # Check exit conditions
            time_elapsed = (current_time - entry_time).total_seconds() / 60

            outcome = None
            exit_reason = None

            if direction == "up":
                pnl_percent = (current_price - entry_price) / entry_price * 100

                if current_price >= target_price:
                    outcome = "win"
                    exit_reason = "target_hit"
                elif current_price <= stop_loss_price:
                    outcome = "loss"
                    exit_reason = "stop_loss"
                elif time_elapsed >= self.horizon_minutes:
                    outcome = "win" if current_price > entry_price else "loss"
                    exit_reason = "timeout"
            else:  # down
                pnl_percent = (entry_price - current_price) / entry_price * 100

                if current_price <= target_price:
                    outcome = "win"
                    exit_reason = "target_hit"
                elif current_price >= stop_loss_price:
                    outcome = "loss"
                    exit_reason = "stop_loss"
                elif time_elapsed >= self.horizon_minutes:
                    outcome = "win" if current_price < entry_price else "loss"
                    exit_reason = "timeout"

            # Close trade if exit condition met
            if outcome:
                self.db.update_exit(
                    trade_id=trade_id,
                    exit_time=current_time,
                    exit_price=current_price,
                    outcome=outcome,
                    pnl_percent=pnl_percent
                )

                logger.info(
                    f"[EXIT] {ticker} {direction.upper()} {outcome.upper()} "
                    f"@ ${current_price:.2f} (PnL={pnl_percent:+.2f}%, reason={exit_reason})"
                )

                closed_trades.append({
                    "ticker": ticker,
                    "direction": direction,
                    "outcome": outcome,
                    "pnl_percent": pnl_percent,
                    "exit_reason": exit_reason
                })

        return closed_trades

    def run_scan(self, tickers: List[str]) -> List[PaperTrade]:
        """
        Scan tickers for entry signals and record trades.

        Returns list of new trades entered.
        """
        new_trades = []

        for ticker in tickers:
            trade = self.check_entry_signal(ticker)
            if trade:
                trade_id = self.enter_trade(trade)
                trade.id = trade_id
                new_trades.append(trade)

        return new_trades

    def get_summary(self) -> Dict:
        """Get current trading summary."""
        return self.db.get_performance_summary()


def main():
    """Main entry point for paper trading."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading System")
    parser.add_argument("--tickers", nargs="+", help="Tickers to monitor")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Probability threshold (default: 0.70)")
    parser.add_argument("--scan-interval", type=int, default=300,
                        help="Scan interval in seconds (default: 300)")
    parser.add_argument("--summary", action="store_true",
                        help="Show performance summary and exit")
    parser.add_argument("--check-exits", action="store_true",
                        help="Check and close expired trades")

    args = parser.parse_args()

    trader = PaperTrader(probability_threshold=args.threshold)

    if args.summary:
        summary = trader.get_summary()
        print("\n=== Paper Trading Performance Summary ===")
        print(f"Period: Last {summary['period_days']} days")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print(f"Avg PnL: {summary['avg_pnl_percent']:+.2f}%")
        print(f"\nUP Trades: {summary['up_trades']} (Wins: {summary['up_wins']})")
        print(f"DOWN Trades: {summary['down_trades']} (Wins: {summary['down_wins']})")
        return

    if args.check_exits:
        closed = trader.check_and_close_trades()
        print(f"\nClosed {len(closed)} trades")
        for t in closed:
            print(f"  {t['ticker']} {t['direction']} -> {t['outcome']} ({t['pnl_percent']:+.2f}%)")
        return

    # Default: Get tickers from model manager
    if not args.tickers:
        from src.models.model_manager import ModelManager
        mm = ModelManager()
        tickers = mm.get_tickers()[:20]  # Top 20 trained tickers
    else:
        tickers = args.tickers

    print(f"\n=== Paper Trading Started ===")
    print(f"Monitoring {len(tickers)} tickers")
    print(f"Probability threshold: {args.threshold:.0%}")
    print(f"Scan interval: {args.scan_interval}s")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            # Check exits first
            closed = trader.check_and_close_trades()

            # Scan for new entries
            new_trades = trader.run_scan(tickers)

            if new_trades:
                print(f"\n[{datetime.now():%H:%M:%S}] New entries: {len(new_trades)}")
            if closed:
                print(f"[{datetime.now():%H:%M:%S}] Closed trades: {len(closed)}")

            time.sleep(args.scan_interval)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")
        summary = trader.get_summary()
        print(f"\nFinal Win Rate: {summary['win_rate']:.1%}")


if __name__ == "__main__":
    main()
