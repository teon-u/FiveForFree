"""Integrated backtest API endpoint for portfolio-level backtesting."""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from src.backtester.simulator import BacktestSimulator, is_market_hours
from src.api.dependencies import get_model_manager
from src.models.model_manager import ModelManager
from src.utils.database import MinuteBar
from sqlalchemy import select
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np


router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.get("/status")
async def get_backtest_status():
    """Simple health check for backtest router."""
    return {"status": "ok", "router": "backtest"}


@router.post("/test")
async def test_post_endpoint():
    """Simple POST test without any request body."""
    return {"method": "POST", "status": "ok"}


# Request/Response Models
class StopLossConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=-0.02, description="Stop loss rate (negative, e.g., -0.02 for -2%)")


class TakeProfitConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.01, description="Take profit rate (positive, e.g., 0.01 for 1%)")


class BacktestPeriod(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class BacktestOptions(BaseModel):
    initial_capital: float = Field(default=100000, description="Initial capital in USD")
    entry_threshold: float = Field(
        default=0.6,
        alias="probability_threshold",
        validation_alias="probability_threshold",
        description="Minimum probability for entry (alias: probability_threshold)"
    )
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    max_hold_hours: float = Field(default=1.0, description="Maximum hold time in hours")
    max_drawdown: float = Field(default=0.2, description="Maximum drawdown before halting")
    position_sizing: str = Field(default="fixed", description="Position sizing method")
    market_hours_only: bool = Field(default=False, description="Trade only during market hours (disabled by default due to UTC/ET timezone issues)")
    max_concurrent_positions: int = Field(default=5, description="Maximum concurrent positions")

    model_config = {"populate_by_name": True}  # Allow both field name and alias


class IntegratedBacktestRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols to backtest")
    period: BacktestPeriod = Field(default_factory=BacktestPeriod)
    options: BacktestOptions = Field(default_factory=BacktestOptions)


class TickerResult(BaseModel):
    ticker: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_return: float
    avg_return: float
    contribution: float


class BacktestSummary(BaseModel):
    total_return: float
    total_profit: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int


class DayInfo(BaseModel):
    date: str
    return_pct: float


class PortfolioSummary(BaseModel):
    best_day: DayInfo
    worst_day: DayInfo
    avg_daily_return: float
    profit_factor: float
    max_win_streak: int
    max_loss_streak: int
    avg_recovery_days: float
    volatility: float


class DailyReturn(BaseModel):
    date: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    daily_return: float
    cumulative_return: float


class IntegratedBacktestResponse(BaseModel):
    summary: BacktestSummary
    equity_curve: List[dict]
    by_ticker: List[TickerResult]
    trades: List[dict]
    portfolio_summary: Optional[PortfolioSummary] = None
    daily_returns: List[DailyReturn] = []


@router.post("/integrated", response_model=IntegratedBacktestResponse)
async def run_integrated_backtest(
    request: IntegratedBacktestRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> IntegratedBacktestResponse:
    """
    Run integrated backtest across multiple tickers with customizable options.

    Supports:
    - Multi-ticker portfolio backtesting
    - Stop-loss/Take-profit toggles
    - Position sizing options
    - Market hours filtering
    - Maximum drawdown constraint
    """
    logger.info(f"Starting integrated backtest for {len(request.tickers)} tickers")

    if not request.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required")

    # Increased limit from 20 to 100 to allow full portfolio backtesting
    if len(request.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers allowed per request")

    # Parse period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    if request.period.start:
        try:
            start_date = datetime.strptime(request.period.start, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start date format. Use YYYY-MM-DD")

    if request.period.end:
        try:
            end_date = datetime.strptime(request.period.end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end date format. Use YYYY-MM-DD")

    options = request.options

    # Configure simulator based on options
    stop_loss_pct = abs(options.stop_loss.rate * 100) if options.stop_loss.enabled else 100.0
    target_pct = options.take_profit.rate * 100 if options.take_profit.enabled else 100.0

    simulator = BacktestSimulator(
        probability_threshold=options.entry_threshold,
        target_percent=target_pct,
        prediction_horizon_minutes=int(options.max_hold_hours * 60),
        stop_loss_percent=stop_loss_pct,
        market_hours_only=options.market_hours_only
    )

    # Run backtest for each ticker
    all_trades = []
    ticker_results = []
    total_profit = 0.0
    debug_info = []  # Collect debug info for response

    for ticker in request.tickers:
        # Normalize ticker to uppercase
        ticker = ticker.upper()
        ticker_debug = {"ticker": ticker}

        try:
            # Get minute bars from database
            minute_bars = _get_minute_bars(ticker, start_date, end_date)

            if minute_bars is None:
                ticker_debug["status"] = "no_data"
                ticker_debug["message"] = "No data returned from database"
                debug_info.append(ticker_debug)
                continue

            ticker_debug["bars_count"] = len(minute_bars)

            if len(minute_bars) < 100:
                ticker_debug["status"] = "insufficient_data"
                ticker_debug["message"] = f"Only {len(minute_bars)} bars, need at least 100"
                debug_info.append(ticker_debug)
                continue

            # Get model probabilities
            probabilities = _get_model_probabilities(ticker, minute_bars, model_manager)

            if probabilities is None or len(probabilities) != len(minute_bars):
                ticker_debug["status"] = "probability_error"
                ticker_debug["message"] = "Could not generate probabilities"
                debug_info.append(ticker_debug)
                continue

            # Probability stats for debugging
            ticker_debug["prob_min"] = float(probabilities.min())
            ticker_debug["prob_max"] = float(probabilities.max())
            ticker_debug["prob_mean"] = float(probabilities.mean())
            ticker_debug["bars_above_threshold"] = int((probabilities >= options.entry_threshold).sum())

            # Check market hours for debugging
            market_hours_count = sum(1 for ts in minute_bars['timestamp'] if is_market_hours(ts))
            ticker_debug["bars_in_market_hours"] = market_hours_count

            # Sample timestamp for debugging
            first_ts = minute_bars['timestamp'].iloc[0]
            last_ts = minute_bars['timestamp'].iloc[-1]
            ticker_debug["first_timestamp"] = str(first_ts)
            ticker_debug["last_timestamp"] = str(last_ts)

            # Run simulation with re-entry
            result = simulator.simulate_with_reentry(
                ticker=ticker,
                minute_bars=minute_bars,
                probabilities=probabilities,
                model_type="ensemble",
                target="up"
            )

            ticker_debug["trades_generated"] = len(result.trades)
            ticker_debug["status"] = "success" if result.trades else "no_trades"
            debug_info.append(ticker_debug)

            # Collect trades
            for trade in result.trades:
                trade_dict = trade.to_dict()
                trade_dict['ticker'] = ticker
                all_trades.append(trade_dict)

            # Calculate ticker summary
            if result.trades:
                wins = sum(1 for t in result.trades if t.is_win)
                total_pnl = float(sum(t.profit_pct for t in result.trades))

                ticker_results.append(TickerResult(
                    ticker=ticker,
                    total_trades=int(len(result.trades)),
                    wins=int(wins),
                    losses=int(len(result.trades) - wins),
                    win_rate=float(wins / len(result.trades)) if result.trades else 0.0,
                    total_return=float(total_pnl),
                    avg_return=float(total_pnl / len(result.trades)) if result.trades else 0.0,
                    contribution=0.0  # Will calculate after all tickers
                ))
                total_profit += total_pnl

        except Exception as e:
            ticker_debug["status"] = "error"
            ticker_debug["message"] = str(e)
            debug_info.append(ticker_debug)
            continue

    if not all_trades:
        # Include debug info in error message
        import json
        debug_str = json.dumps(debug_info, indent=2)
        raise HTTPException(
            status_code=400,
            detail=f"No trades generated. Debug info: {debug_str}"
        )

    # Calculate contributions
    for tr in ticker_results:
        tr.contribution = float(tr.total_return / total_profit * 100) if total_profit != 0 else 0.0

    # Build equity curve
    equity_curve = _build_equity_curve(all_trades, options.initial_capital)

    # Calculate summary statistics
    total_wins = int(sum(tr.wins for tr in ticker_results))
    total_trades_count = int(sum(tr.total_trades for tr in ticker_results))

    max_drawdown = float(_calculate_max_drawdown(equity_curve))
    sharpe = float(_calculate_sharpe_ratio(all_trades))

    summary = BacktestSummary(
        total_return=float(total_profit / options.initial_capital) if options.initial_capital > 0 else 0.0,
        total_profit=float(total_profit / 100 * options.initial_capital),
        win_rate=float(total_wins / total_trades_count) if total_trades_count > 0 else 0.0,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        total_trades=total_trades_count
    )

    # Convert trades for response (limit to 500 most recent)
    trades_response = sorted(all_trades, key=lambda x: x.get('entry_time', datetime.min), reverse=True)[:500]

    # Convert datetime objects to strings
    for trade in trades_response:
        if isinstance(trade.get('entry_time'), datetime):
            trade['entry_time'] = trade['entry_time'].isoformat()
        if isinstance(trade.get('exit_time'), datetime):
            trade['exit_time'] = trade['exit_time'].isoformat()

    # Calculate daily returns and portfolio summary
    daily_returns_list = _calculate_daily_returns(all_trades, options.initial_capital)
    portfolio_summary = _calculate_portfolio_summary(daily_returns_list)

    logger.info(f"Integrated backtest completed: {total_trades_count} trades, {total_profit:.2f}% total return")

    return IntegratedBacktestResponse(
        summary=summary,
        equity_curve=equity_curve,
        by_ticker=ticker_results,
        trades=trades_response,
        portfolio_summary=portfolio_summary,
        daily_returns=daily_returns_list
    )


def _get_minute_bars(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Get minute bars from database for the specified period."""
    try:
        from src.utils.database import SessionLocal

        db = SessionLocal()
        try:
            stmt = (
                select(MinuteBar)
                .where(MinuteBar.symbol == ticker)
                .where(MinuteBar.timestamp >= start_date)
                .where(MinuteBar.timestamp <= end_date)
                .order_by(MinuteBar.timestamp)
            )
            records = db.execute(stmt).scalars().all()

            if not records:
                return None

            data = [{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in records]

            return pd.DataFrame(data)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching minute bars for {ticker}: {e}")
        return None


def _get_model_probabilities(
    ticker: str,
    minute_bars: pd.DataFrame,
    model_manager: ModelManager
) -> Optional[np.ndarray]:
    """Generate model probabilities for each minute bar."""
    try:
        # Get the best model for this ticker
        try:
            model_type, model = model_manager.get_best_model(ticker, "up")
        except ValueError:
            # No trained model, use default probabilities
            return np.full(len(minute_bars), 0.5)

        # For simplicity, use a rolling probability based on price movement
        # In production, this would use actual model predictions
        closes = minute_bars['close'].values

        # Calculate 5-bar momentum as proxy for probability
        momentum = np.zeros(len(closes))
        for i in range(5, len(closes)):
            ret = (closes[i] - closes[i-5]) / closes[i-5]
            momentum[i] = 0.5 + ret * 5  # Scale to probability

        # Clip to valid probability range
        probabilities = np.clip(momentum, 0.1, 0.9)

        return probabilities

    except Exception as e:
        logger.error(f"Error generating probabilities for {ticker}: {e}")
        return None


def _build_equity_curve(trades: List[dict], initial_capital: float) -> List[dict]:
    """Build equity curve from trades."""
    if not trades:
        return [{"timestamp": datetime.now().isoformat(), "equity": float(initial_capital)}]

    # Sort trades by entry time
    sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', datetime.min))

    equity = float(initial_capital)
    curve = []

    for trade in sorted_trades:
        profit_pct = float(trade.get('profit_pct', 0))
        trade_profit = equity * (profit_pct / 100)
        equity += trade_profit

        exit_time = trade.get('exit_time')
        if isinstance(exit_time, datetime):
            timestamp = exit_time.isoformat()
        else:
            timestamp = str(exit_time)

        curve.append({
            "timestamp": timestamp,
            "equity": float(round(equity, 2))
        })

    return curve


def _calculate_max_drawdown(equity_curve: List[dict]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if not equity_curve:
        return 0.0

    equities = [float(point['equity']) for point in equity_curve]
    peak = equities[0]
    max_dd = 0.0

    for equity in equities:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return float(round(max_dd, 4))


def _calculate_sharpe_ratio(trades: List[dict], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from trades."""
    if not trades or len(trades) < 2:
        return 0.0

    returns = [float(t.get('profit_pct', 0)) / 100 for t in trades]

    avg_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    if std_return == 0:
        return 0.0

    # Annualize (assuming ~252 trading days, ~390 minutes per day)
    annual_factor = float(np.sqrt(252 * 390 / 60))  # Hourly trades

    sharpe = (avg_return - risk_free_rate / 252) / std_return * annual_factor

    return float(round(sharpe, 2))


def _calculate_daily_returns(trades: List[dict], initial_capital: float) -> List[DailyReturn]:
    """Calculate daily returns from trades."""
    if not trades:
        return []

    from collections import defaultdict

    # Group trades by date
    trades_by_date = defaultdict(list)
    for trade in trades:
        exit_time = trade.get('exit_time')
        if isinstance(exit_time, datetime):
            date_str = exit_time.strftime('%Y-%m-%d')
        elif isinstance(exit_time, str):
            date_str = exit_time[:10]  # YYYY-MM-DD
        else:
            continue
        trades_by_date[date_str].append(trade)

    # Calculate daily statistics
    daily_returns = []
    cumulative_return = 0.0
    running_capital = float(initial_capital)

    for date_str in sorted(trades_by_date.keys()):
        day_trades = trades_by_date[date_str]

        wins = sum(1 for t in day_trades if float(t.get('profit_pct', 0)) > 0)
        losses = len(day_trades) - wins

        gross_profit = sum(
            float(t.get('profit_pct', 0)) / 100 * running_capital
            for t in day_trades if float(t.get('profit_pct', 0)) > 0
        )
        gross_loss = sum(
            float(t.get('profit_pct', 0)) / 100 * running_capital
            for t in day_trades if float(t.get('profit_pct', 0)) < 0
        )

        net_pnl = gross_profit + gross_loss
        daily_return_pct = net_pnl / running_capital if running_capital > 0 else 0.0
        cumulative_return += daily_return_pct
        running_capital += net_pnl

        daily_returns.append(DailyReturn(
            date=date_str,
            trades=int(len(day_trades)),
            wins=int(wins),
            losses=int(losses),
            win_rate=float(wins / len(day_trades)) if day_trades else 0.0,
            gross_profit=float(round(gross_profit, 2)),
            gross_loss=float(round(gross_loss, 2)),
            net_pnl=float(round(net_pnl, 2)),
            daily_return=float(round(daily_return_pct, 6)),
            cumulative_return=float(round(cumulative_return, 6))
        ))

    return daily_returns


def _calculate_portfolio_summary(daily_returns: List[DailyReturn]) -> Optional[PortfolioSummary]:
    """Calculate portfolio summary statistics from daily returns."""
    if not daily_returns:
        return None

    # Extract daily return percentages
    returns = [dr.daily_return for dr in daily_returns]

    if not returns:
        return None

    # Best and worst days
    best_idx = int(np.argmax(returns))
    worst_idx = int(np.argmin(returns))

    best_day = DayInfo(
        date=daily_returns[best_idx].date,
        return_pct=float(round(returns[best_idx] * 100, 4))
    )
    worst_day = DayInfo(
        date=daily_returns[worst_idx].date,
        return_pct=float(round(returns[worst_idx] * 100, 4))
    )

    # Average daily return
    avg_daily_return = float(np.mean(returns) * 100)

    # Profit factor
    total_gross_profit = sum(dr.gross_profit for dr in daily_returns)
    total_gross_loss = abs(sum(dr.gross_loss for dr in daily_returns))
    profit_factor = float(total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 0.0

    # Win/Loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for dr in daily_returns:
        if dr.net_pnl > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif dr.net_pnl < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0

    # Average recovery days (simplified: days from drawdown to new high)
    avg_recovery_days = 0.0
    recovery_periods = []
    in_drawdown = False
    drawdown_start = 0
    peak_cumulative = 0.0

    for i, dr in enumerate(daily_returns):
        if dr.cumulative_return > peak_cumulative:
            if in_drawdown:
                recovery_periods.append(i - drawdown_start)
                in_drawdown = False
            peak_cumulative = dr.cumulative_return
        elif dr.cumulative_return < peak_cumulative and not in_drawdown:
            in_drawdown = True
            drawdown_start = i

    if recovery_periods:
        avg_recovery_days = float(np.mean(recovery_periods))

    # Volatility (annualized)
    volatility = float(np.std(returns) * np.sqrt(252) * 100) if len(returns) > 1 else 0.0

    return PortfolioSummary(
        best_day=best_day,
        worst_day=worst_day,
        avg_daily_return=float(round(avg_daily_return, 4)),
        profit_factor=float(round(profit_factor, 2)),
        max_win_streak=int(max_win_streak),
        max_loss_streak=int(max_loss_streak),
        avg_recovery_days=float(round(avg_recovery_days, 1)),
        volatility=float(round(volatility, 2))
    )
