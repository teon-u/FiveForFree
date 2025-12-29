"""
Signal Logger - Real-time Trading Signal Recording

Records all trading signals to a structured log:
- Timestamp (UTC and local)
- Ticker symbol
- Probability (up/down)
- Direction (BUY/SELL/HOLD)
- Model type
- Current price
- Confidence level

Logs are stored in both SQLite and JSON format for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

from loguru import logger
from config.settings import settings

# Paths
DATA_DIR = Path(settings.DATA_DIR)
SIGNALS_DB = DATA_DIR / "signals.db"
SIGNALS_LOG_DIR = DATA_DIR / "signal_logs"


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    timestamp: datetime
    ticker: str
    direction: str  # "BUY", "SELL", "HOLD"
    up_probability: float
    down_probability: float
    best_model: str
    current_price: float
    confidence_level: str  # "VERY_HIGH", "HIGH", "MEDIUM", "LOW"
    model_accuracy: float


class SignalLogger:
    """Logs trading signals to database and files."""

    def __init__(self):
        self.db_path = SIGNALS_DB
        self.log_dir = SIGNALS_LOG_DIR
        self._init_db()
        self._init_log_dir()

    def _init_db(self):
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    up_probability REAL NOT NULL,
                    down_probability REAL NOT NULL,
                    best_model TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    model_accuracy REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp
                ON signals(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_ticker
                ON signals(ticker)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_direction
                ON signals(direction)
            """)

            conn.commit()

    def _init_log_dir(self):
        """Initialize log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_signal(self, signal: TradingSignal) -> int:
        """
        Log a trading signal to database and daily file.

        Returns the signal ID.
        """
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO signals
                (timestamp, ticker, direction, up_probability, down_probability,
                 best_model, current_price, confidence_level, model_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.timestamp.isoformat(),
                signal.ticker,
                signal.direction,
                signal.up_probability,
                signal.down_probability,
                signal.best_model,
                signal.current_price,
                signal.confidence_level,
                signal.model_accuracy
            ))
            signal_id = cursor.lastrowid
            conn.commit()

        # Append to daily JSON log
        self._append_to_daily_log(signal)

        # Log to console
        prob = signal.up_probability if signal.direction == "BUY" else signal.down_probability
        logger.info(
            f"[SIGNAL] {signal.timestamp:%H:%M:%S} {signal.ticker} "
            f"{signal.direction} ({prob:.1%}) @ ${signal.current_price:.2f} "
            f"[{signal.confidence_level}]"
        )

        return signal_id

    def _append_to_daily_log(self, signal: TradingSignal):
        """Append signal to daily JSON log file."""
        date_str = signal.timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"signals_{date_str}.jsonl"

        signal_dict = {
            "timestamp": signal.timestamp.isoformat(),
            "ticker": signal.ticker,
            "direction": signal.direction,
            "up_probability": signal.up_probability,
            "down_probability": signal.down_probability,
            "best_model": signal.best_model,
            "current_price": signal.current_price,
            "confidence_level": signal.confidence_level,
            "model_accuracy": signal.model_accuracy
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(signal_dict) + "\n")

    def get_signals_by_date(self, date: datetime) -> List[Dict]:
        """Get all signals for a specific date."""
        date_str = date.strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM signals
                WHERE timestamp LIKE ?
                ORDER BY timestamp DESC
            """, (f"{date_str}%",))
            return [dict(row) for row in cursor.fetchall()]

    def get_signals_by_ticker(self, ticker: str, limit: int = 100) -> List[Dict]:
        """Get recent signals for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM signals
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (ticker, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_actionable_signals(
        self,
        min_probability: float = 0.70,
        limit: int = 50
    ) -> List[Dict]:
        """Get recent actionable signals (BUY/SELL only, above threshold)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM signals
                WHERE direction IN ('BUY', 'SELL')
                AND (
                    (direction = 'BUY' AND up_probability >= ?)
                    OR (direction = 'SELL' AND down_probability >= ?)
                )
                ORDER BY timestamp DESC
                LIMIT ?
            """, (min_probability, min_probability, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_signal_stats(self, hours: int = 24) -> Dict:
        """Get signal statistics for last N hours."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Total signals
            total = conn.execute("""
                SELECT COUNT(*) FROM signals WHERE timestamp >= ?
            """, (cutoff,)).fetchone()[0]

            # By direction
            buy_count = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp >= ? AND direction = 'BUY'
            """, (cutoff,)).fetchone()[0]

            sell_count = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp >= ? AND direction = 'SELL'
            """, (cutoff,)).fetchone()[0]

            hold_count = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp >= ? AND direction = 'HOLD'
            """, (cutoff,)).fetchone()[0]

            # High confidence signals
            high_conf = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp >= ? AND confidence_level IN ('VERY_HIGH', 'HIGH')
            """, (cutoff,)).fetchone()[0]

            # Average probabilities
            avg_up = conn.execute("""
                SELECT AVG(up_probability) FROM signals
                WHERE timestamp >= ? AND direction = 'BUY'
            """, (cutoff,)).fetchone()[0] or 0.0

            avg_down = conn.execute("""
                SELECT AVG(down_probability) FROM signals
                WHERE timestamp >= ? AND direction = 'SELL'
            """, (cutoff,)).fetchone()[0] or 0.0

            # Top tickers by signal count
            top_tickers = conn.execute("""
                SELECT ticker, COUNT(*) as count
                FROM signals
                WHERE timestamp >= ? AND direction IN ('BUY', 'SELL')
                GROUP BY ticker
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff,)).fetchall()

            return {
                "period_hours": hours,
                "total_signals": total,
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "hold_signals": hold_count,
                "high_confidence_signals": high_conf,
                "avg_buy_probability": avg_up,
                "avg_sell_probability": avg_down,
                "top_tickers": [{"ticker": t, "count": c} for t, c in top_tickers]
            }

    def export_to_csv(self, output_path: Path, days: int = 7):
        """Export signals to CSV file."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM signals
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff,))
            rows = cursor.fetchall()

        if not rows:
            logger.warning("No signals to export")
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        logger.info(f"Exported {len(rows)} signals to {output_path}")


class SignalMonitor:
    """Monitors predictions and logs signals."""

    def __init__(self, signal_threshold: float = 0.60):
        self.signal_threshold = signal_threshold
        self.logger = SignalLogger()
        self._predictor = None

    def _get_predictor(self):
        """Lazy load predictor."""
        if self._predictor is None:
            from src.models.model_manager import ModelManager
            from src.predictor.realtime_predictor import RealtimePredictor
            from src.collector.minute_bars import MinuteBarCollector

            model_manager = ModelManager()
            minute_bar_collector = MinuteBarCollector()

            self._predictor = RealtimePredictor(
                model_manager=model_manager,
                minute_bar_collector=minute_bar_collector
            )
        return self._predictor

    def check_and_log(self, ticker: str) -> Optional[TradingSignal]:
        """
        Check prediction for ticker and log if actionable.

        Returns the signal if logged, None otherwise.
        """
        try:
            predictor = self._get_predictor()
            result = predictor.predict(ticker, include_all_models=False)

            # Determine direction
            direction = result.get_trading_signal()

            # Only log BUY/SELL signals or if probability is notable
            max_prob = max(result.up_probability, result.down_probability)

            if direction in ("BUY", "SELL") or max_prob >= self.signal_threshold:
                if direction == "BUY":
                    best_model = result.best_up_model
                    model_accuracy = result.up_model_accuracy
                else:
                    best_model = result.best_down_model
                    model_accuracy = result.down_model_accuracy

                signal = TradingSignal(
                    timestamp=result.timestamp,
                    ticker=ticker,
                    direction=direction,
                    up_probability=result.up_probability,
                    down_probability=result.down_probability,
                    best_model=best_model,
                    current_price=result.current_price,
                    confidence_level=result.get_confidence_level(),
                    model_accuracy=model_accuracy
                )

                self.logger.log_signal(signal)
                return signal

            return None

        except Exception as e:
            logger.warning(f"Failed to check signal for {ticker}: {e}")
            return None

    def scan_tickers(self, tickers: List[str]) -> List[TradingSignal]:
        """Scan multiple tickers and log all signals."""
        signals = []

        for ticker in tickers:
            signal = self.check_and_log(ticker)
            if signal:
                signals.append(signal)

        return signals


def main():
    """Main entry point for signal logger."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Trading Signal Logger")
    parser.add_argument("--tickers", nargs="+", help="Tickers to monitor")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Signal threshold (default: 0.60)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Scan interval in seconds (default: 60)")
    parser.add_argument("--stats", action="store_true",
                        help="Show signal statistics and exit")
    parser.add_argument("--export", type=str,
                        help="Export signals to CSV file")

    args = parser.parse_args()

    signal_logger = SignalLogger()

    if args.stats:
        stats = signal_logger.get_signal_stats(hours=24)
        print("\n=== Signal Statistics (Last 24 Hours) ===")
        print(f"Total Signals: {stats['total_signals']}")
        print(f"BUY: {stats['buy_signals']} | SELL: {stats['sell_signals']} | HOLD: {stats['hold_signals']}")
        print(f"High Confidence: {stats['high_confidence_signals']}")
        print(f"Avg BUY Probability: {stats['avg_buy_probability']:.1%}")
        print(f"Avg SELL Probability: {stats['avg_sell_probability']:.1%}")
        print("\nTop Tickers:")
        for t in stats['top_tickers']:
            print(f"  {t['ticker']}: {t['count']} signals")
        return

    if args.export:
        signal_logger.export_to_csv(Path(args.export))
        return

    # Get tickers
    if not args.tickers:
        from src.models.model_manager import ModelManager
        mm = ModelManager()
        tickers = mm.get_tickers()[:30]
    else:
        tickers = args.tickers

    monitor = SignalMonitor(signal_threshold=args.threshold)

    print(f"\n=== Signal Logger Started ===")
    print(f"Monitoring {len(tickers)} tickers")
    print(f"Signal threshold: {args.threshold:.0%}")
    print(f"Scan interval: {args.interval}s")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            signals = monitor.scan_tickers(tickers)

            if signals:
                actionable = [s for s in signals if s.direction in ("BUY", "SELL")]
                print(f"[{datetime.now():%H:%M:%S}] Logged {len(signals)} signals "
                      f"({len(actionable)} actionable)")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopping signal logger...")
        stats = signal_logger.get_signal_stats(hours=1)
        print(f"Last hour: {stats['total_signals']} signals logged")


if __name__ == "__main__":
    main()
