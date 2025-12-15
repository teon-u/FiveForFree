"""Global configuration settings."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Polygon.io API
    POLYGON_API_KEY: str

    # Collection Settings
    TOP_N_VOLUME: int = 100
    TOP_N_GAINERS: int = 100
    HISTORICAL_DAYS: int = 30

    # Prediction Settings
    PREDICTION_HORIZON_MINUTES: int = 60
    TARGET_PERCENT: float = 5.0
    PROBABILITY_THRESHOLD: float = 0.70

    # Backtesting
    BACKTEST_HOURS: int = 50

    # GPU Settings
    USE_GPU: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./data/nasdaq_predictor.db"

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Trading Settings
    COMMISSION_PERCENT: float = 0.1  # One-way commission

    # Model Settings
    MODEL_TYPES: list[str] = ["xgboost", "lightgbm", "lstm", "transformer", "ensemble"]
    PREDICTION_TARGETS: list[str] = ["up", "down"]

    # Hybrid-Ensemble Settings (Structure B)
    HYBRID_TARGETS: list[str] = ["volatility", "direction"]  # Structure B targets
    USE_HYBRID_ENSEMBLE: bool = True  # Enable hybrid-ensemble approach
    ENSEMBLE_ALPHA: float = 0.5  # Weight for direct prediction (1-alpha for hybrid)
    CALIBRATION_METHOD: str = "isotonic"  # "platt" or "isotonic" for probability calibration

    # Feature Engineering
    NUM_FEATURES: int = 57

    # Training Settings
    N_PARALLEL_WORKERS: int = 4
    SEQUENCE_LENGTH: int = 60  # For LSTM/Transformer

    # Market Hours (ET)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16
    MARKET_CLOSE_MINUTE: int = 0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
