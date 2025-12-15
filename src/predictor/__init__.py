"""
Real-time prediction modules for NASDAQ prediction system.

This package provides real-time prediction capabilities:

Real-time Predictor:
    - RealtimePredictor: Main prediction engine for live trading
    - Collects latest market data
    - Computes 57 features
    - Generates predictions using best-performing models
    - Returns probabilities for UI display
"""

from src.predictor.realtime_predictor import RealtimePredictor, PredictionResult

__all__ = [
    'RealtimePredictor',
    'PredictionResult',
]

__version__ = '1.0.0'
