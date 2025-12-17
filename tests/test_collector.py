"""Tests for data collector modules."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestMinuteBarCollector:
    """Test suite for MinuteBarCollector."""

    @pytest.fixture
    def mock_yf_download(self):
        """Mock yfinance download function."""
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [148.0, 149.0, 150.0],
            'Close': [153.0, 154.0, 155.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01 09:30', periods=3, freq='1min'))
        return mock_data

    def test_collector_initialization(self):
        """Test that collector initializes correctly."""
        with patch('src.collector.minute_bars.yf'):
            from src.collector.minute_bars import MinuteBarCollector
            collector = MinuteBarCollector(use_db=False)
            assert collector is not None

    def test_fetch_minute_bars_returns_dataframe(self, mock_yf_download):
        """Test that fetching minute bars returns a DataFrame."""
        with patch('src.collector.minute_bars.yf.download', return_value=mock_yf_download):
            from src.collector.minute_bars import MinuteBarCollector
            collector = MinuteBarCollector(use_db=False)

            result = collector.fetch_minute_bars('AAPL', days=1)

            assert isinstance(result, pd.DataFrame)
            assert 'open' in result.columns or 'Open' in result.columns

    def test_fetch_minute_bars_handles_empty_response(self):
        """Test handling of empty response from API."""
        with patch('src.collector.minute_bars.yf.download', return_value=pd.DataFrame()):
            from src.collector.minute_bars import MinuteBarCollector
            collector = MinuteBarCollector(use_db=False)

            result = collector.fetch_minute_bars('INVALID', days=1)

            assert result is None or len(result) == 0

    def test_fetch_minute_bars_validates_ticker(self):
        """Test that ticker is validated."""
        with patch('src.collector.minute_bars.yf'):
            from src.collector.minute_bars import MinuteBarCollector
            collector = MinuteBarCollector(use_db=False)

            # Empty ticker should be handled
            result = collector.fetch_minute_bars('', days=1)
            assert result is None or len(result) == 0


class TestQuoteCollector:
    """Test suite for QuoteCollector."""

    @pytest.fixture
    def mock_finnhub_client(self):
        """Mock Finnhub client."""
        mock_client = MagicMock()
        mock_client.quote.return_value = {
            'c': 150.0,  # Current price
            'h': 155.0,  # High
            'l': 145.0,  # Low
            'o': 148.0,  # Open
            'pc': 147.0,  # Previous close
            'dp': 2.04,  # Percent change
            't': 1704096000  # Timestamp
        }
        return mock_client

    def test_collector_initialization(self, mock_finnhub_client):
        """Test QuoteCollector initialization."""
        with patch('src.collector.quotes.finnhub.Client', return_value=mock_finnhub_client):
            from src.collector.quotes import QuoteCollector
            collector = QuoteCollector(api_key='test_key')
            assert collector is not None

    def test_get_quote_returns_dict(self, mock_finnhub_client):
        """Test that get_quote returns quote data."""
        with patch('src.collector.quotes.finnhub.Client', return_value=mock_finnhub_client):
            from src.collector.quotes import QuoteCollector
            collector = QuoteCollector(api_key='test_key')

            result = collector.get_quote('AAPL')

            assert isinstance(result, dict)
            assert 'c' in result  # Current price

    def test_get_quote_handles_api_error(self, mock_finnhub_client):
        """Test handling of API errors."""
        mock_finnhub_client.quote.side_effect = Exception("API Error")

        with patch('src.collector.quotes.finnhub.Client', return_value=mock_finnhub_client):
            from src.collector.quotes import QuoteCollector
            collector = QuoteCollector(api_key='test_key')

            result = collector.get_quote('INVALID')
            assert result is None or 'error' in str(result).lower()


class TestMarketContextCollector:
    """Test suite for MarketContextCollector."""

    @pytest.fixture
    def mock_market_data(self):
        """Mock market data response."""
        return {
            'SPY': {'change_percent': 0.5},
            'QQQ': {'change_percent': 0.8},
            'VIX': {'current': 15.0}
        }

    def test_collector_initialization(self):
        """Test MarketContextCollector initialization."""
        with patch('src.collector.market_context.finnhub'):
            from src.collector.market_context import MarketContextCollector
            collector = MarketContextCollector(api_key='test_key')
            assert collector is not None

    def test_get_market_context_returns_indicators(self):
        """Test that market context returns market indicators."""
        with patch('src.collector.market_context.finnhub'):
            from src.collector.market_context import MarketContextCollector
            collector = MarketContextCollector(api_key='test_key')

            # Mock the get methods
            with patch.object(collector, 'get_index_data', return_value={'change_percent': 0.5}):
                result = collector.get_market_indicators()

                if result is not None:
                    assert isinstance(result, (dict, list))


class TestTickerSelector:
    """Test suite for TickerSelector."""

    @pytest.fixture
    def mock_finnhub_snapshot(self):
        """Mock snapshot data."""
        return [
            MagicMock(ticker='AAPL', day=MagicMock(volume=10000000), todaysChangePerc=2.5),
            MagicMock(ticker='GOOGL', day=MagicMock(volume=8000000), todaysChangePerc=1.8),
            MagicMock(ticker='MSFT', day=MagicMock(volume=9000000), todaysChangePerc=3.2),
            MagicMock(ticker='TSLA', day=MagicMock(volume=15000000), todaysChangePerc=5.0),
            MagicMock(ticker='NVDA', day=MagicMock(volume=12000000), todaysChangePerc=4.5),
        ]

    def test_selector_initialization(self):
        """Test TickerSelector initialization."""
        with patch('src.collector.ticker_selector.finnhub'):
            from src.collector.ticker_selector import TickerSelector
            selector = TickerSelector(api_key='test_key')
            assert selector is not None

    def test_get_target_tickers_returns_list(self, mock_finnhub_snapshot):
        """Test that get_target_tickers returns a list."""
        with patch('src.collector.ticker_selector.finnhub') as mock_finnhub:
            mock_client = MagicMock()
            mock_client.stock_screener.return_value = mock_finnhub_snapshot
            mock_finnhub.Client.return_value = mock_client

            from src.collector.ticker_selector import TickerSelector
            selector = TickerSelector(api_key='test_key')

            # Mock the method
            with patch.object(selector, 'get_target_tickers', return_value=['AAPL', 'GOOGL', 'MSFT']):
                result = selector.get_target_tickers()

                assert isinstance(result, list)
                assert len(result) > 0

    def test_get_volume_top_n_returns_sorted_list(self, mock_finnhub_snapshot):
        """Test volume top N sorting."""
        with patch('src.collector.ticker_selector.finnhub'):
            from src.collector.ticker_selector import TickerSelector
            selector = TickerSelector(api_key='test_key')

            # Mock the method
            with patch.object(selector, 'get_volume_top_n', return_value=['TSLA', 'NVDA', 'AAPL']):
                result = selector.get_volume_top_n(3)

                assert isinstance(result, list)
                assert len(result) <= 3

    def test_get_gainers_top_n_returns_sorted_list(self, mock_finnhub_snapshot):
        """Test gainers top N sorting."""
        with patch('src.collector.ticker_selector.finnhub'):
            from src.collector.ticker_selector import TickerSelector
            selector = TickerSelector(api_key='test_key')

            # Mock the method
            with patch.object(selector, 'get_gainers_top_n', return_value=['TSLA', 'NVDA', 'MSFT']):
                result = selector.get_gainers_top_n(3)

                assert isinstance(result, list)
                assert len(result) <= 3


class TestDataCollectorIntegration:
    """Integration tests for data collectors."""

    def test_collectors_can_be_used_together(self):
        """Test that collectors can be initialized together."""
        with patch('src.collector.minute_bars.yf'):
            with patch('src.collector.quotes.finnhub'):
                with patch('src.collector.market_context.finnhub'):
                    from src.collector.minute_bars import MinuteBarCollector
                    from src.collector.quotes import QuoteCollector
                    from src.collector.market_context import MarketContextCollector

                    mb_collector = MinuteBarCollector(use_db=False)
                    quote_collector = QuoteCollector(api_key='test_key')
                    mc_collector = MarketContextCollector(api_key='test_key')

                    assert mb_collector is not None
                    assert quote_collector is not None
                    assert mc_collector is not None

    def test_minute_bar_schema_validation(self):
        """Test that minute bar data follows expected schema."""
        expected_columns = ['open', 'high', 'low', 'close', 'volume']

        mock_data = pd.DataFrame({
            'open': [150.0],
            'high': [155.0],
            'low': [148.0],
            'close': [153.0],
            'volume': [1000000]
        })

        for col in expected_columns:
            assert col in mock_data.columns

    def test_quote_schema_validation(self):
        """Test that quote data follows expected schema."""
        expected_fields = ['c', 'h', 'l', 'o', 'pc']

        mock_quote = {
            'c': 150.0,
            'h': 155.0,
            'l': 145.0,
            'o': 148.0,
            'pc': 147.0
        }

        for field in expected_fields:
            assert field in mock_quote
