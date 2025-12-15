"""Test configuration settings."""
import pytest
from config.settings import Settings


def test_settings_load():
    """Test that settings can be loaded."""
    settings = Settings(_env_file=None)
    assert settings is not None


def test_default_values():
    """Test default configuration values."""
    settings = Settings(_env_file=None)
    assert settings.TOP_N_VOLUME == 100
    assert settings.TOP_N_GAINERS == 100
    assert settings.TARGET_PERCENT == 5.0
    assert settings.PROBABILITY_THRESHOLD == 0.70
    assert settings.BACKTEST_HOURS == 50
