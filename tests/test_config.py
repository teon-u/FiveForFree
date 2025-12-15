"""Test configuration settings."""
import pytest
from config.settings import Settings


def test_settings_load():
    """Test that settings can be loaded with mock API key."""
    settings = Settings(FINNHUB_API_KEY="test_key", _env_file=None)
    assert settings is not None
    assert settings.FINNHUB_API_KEY == "test_key"


def test_default_values():
    """Test default configuration values."""
    settings = Settings(FINNHUB_API_KEY="test_key", _env_file=None)
    assert settings.TOP_N_VOLUME == 50
    assert settings.TOP_N_GAINERS == 50
    assert settings.TARGET_PERCENT == 5.0
    assert settings.PROBABILITY_THRESHOLD == 0.70
    assert settings.BACKTEST_HOURS == 50
