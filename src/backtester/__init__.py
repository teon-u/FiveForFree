"""
Backtesting Module for NASDAQ Prediction System

This module provides backtesting functionality to evaluate model performance
on historical data using realistic trading rules:
- Long Only strategy
- Entry: up_prob >= threshold (default 70%)
- Exit: 5% profit OR 60 minutes elapsed
- Commission: 0.2% round-trip (0.1% entry + 0.1% exit)

Components:
- simulator: Backtest execution engine with trade simulation
- metrics: Performance metrics calculation (win rate, Sharpe, profit factor, etc.)
- investment_strategy: Enhanced investment strategy with S+ grade and rebalancing
"""

from .simulator import BacktestSimulator, Trade, BacktestResult
from .metrics import PerformanceMetrics, ModelPerformanceTracker
from .investment_strategy import (
    InvestmentStrategy,
    StockGrade,
    StockPerformance,
    PortfolioAllocation,
    TransactionCosts,
    calculate_enhanced_metrics
)

__all__ = [
    'BacktestSimulator',
    'Trade',
    'BacktestResult',
    'PerformanceMetrics',
    'ModelPerformanceTracker',
    'InvestmentStrategy',
    'StockGrade',
    'StockPerformance',
    'PortfolioAllocation',
    'TransactionCosts',
    'calculate_enhanced_metrics',
]
