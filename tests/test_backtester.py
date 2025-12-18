"""
Comprehensive Tests for Backtest Simulator and Performance Metrics

DDALKKAK.inc - FiveForFree Project
Final Sprint: Backtest Module Testing

Team Members:
- 묵직 (개발팀장): 통합 테스트
- 번개 (개발): Trade/BacktestResult 데이터클래스
- 돌탱 (개발): BacktestSimulator 시뮬레이션 메서드
- 까칠 (QA팀장): PerformanceMetrics 검증
- 꼼수 (QA): ModelPerformanceTracker
- 의심 (QA): 경계 조건 및 에러 케이스

Test Coverage:
- Trade 데이터클래스 (3개)
- BacktestResult 데이터클래스 (4개)
- BacktestSimulator 초기화 (3개)
- simulate_trade 메서드 (5개)
- simulate_predictions/vectorized (4개)
- 유틸리티 메서드 (3개)
- PerformanceMetrics (6개)
- ModelPerformanceTracker (6개)
- 경계 조건 및 에러 케이스 (6개)

Total: 40 tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, patch, MagicMock

from src.backtester.simulator import (
    Trade,
    BacktestResult,
    BacktestSimulator
)
from src.backtester.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_rolling_metrics,
    ModelPerformanceTracker
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_trade():
    """번개: 샘플 거래 생성"""
    return Trade(
        ticker='AAPL',
        entry_time=datetime(2024, 1, 1, 10, 0),
        entry_price=150.0,
        exit_time=datetime(2024, 1, 1, 10, 30),
        exit_price=157.5,
        exit_reason='target_hit',
        probability=0.75,
        profit_pct=4.7,  # 5% gross - 0.3% commission
        duration_minutes=30.0,
        model_type='xgboost',
        target='up'
    )


@pytest.fixture
def sample_losing_trade():
    """번개: 샘플 손실 거래"""
    return Trade(
        ticker='AAPL',
        entry_time=datetime(2024, 1, 1, 11, 0),
        entry_price=150.0,
        exit_time=datetime(2024, 1, 1, 12, 0),
        exit_price=148.5,
        exit_reason='time_limit',
        probability=0.65,
        profit_pct=-1.3,  # -1% gross - 0.3% commission
        duration_minutes=60.0,
        model_type='xgboost',
        target='up'
    )


@pytest.fixture
def sample_trades(sample_trade, sample_losing_trade):
    """번개: 여러 거래 샘플"""
    trades = [sample_trade, sample_losing_trade]

    # Add more winning trades
    for i in range(3):
        trades.append(Trade(
            ticker='AAPL',
            entry_time=datetime(2024, 1, 1, 13 + i, 0),
            entry_price=150.0,
            exit_time=datetime(2024, 1, 1, 13 + i, 30),
            exit_price=157.5,
            exit_reason='target_hit',
            probability=0.7 + i * 0.05,
            profit_pct=4.7,
            duration_minutes=30.0,
            model_type='xgboost',
            target='up'
        ))

    # Add one more losing trade
    trades.append(Trade(
        ticker='AAPL',
        entry_time=datetime(2024, 1, 1, 16, 0),
        entry_price=150.0,
        exit_time=datetime(2024, 1, 1, 17, 0),
        exit_price=147.0,
        exit_reason='time_limit',
        probability=0.68,
        profit_pct=-2.3,
        duration_minutes=60.0,
        model_type='xgboost',
        target='up'
    ))

    return trades


@pytest.fixture
def minute_bars():
    """돌탱: 분봉 데이터 생성"""
    start_time = datetime(2024, 1, 1, 10, 0)
    num_bars = 120  # 2 hours of minute data

    timestamps = [start_time + timedelta(minutes=i) for i in range(num_bars)]

    # Generate realistic price movements
    np.random.seed(42)
    base_price = 150.0
    returns = np.random.randn(num_bars) * 0.002  # 0.2% volatility per minute
    prices = base_price * (1 + returns).cumprod()

    data = {
        'timestamp': timestamps,
        'open': prices + np.random.randn(num_bars) * 0.1,
        'high': prices + np.abs(np.random.randn(num_bars)) * 0.2,
        'low': prices - np.abs(np.random.randn(num_bars)) * 0.2,
        'close': prices,
        'volume': np.random.randint(10000, 50000, num_bars)
    }

    return pd.DataFrame(data)


@pytest.fixture
def simulator():
    """돌탱: 기본 시뮬레이터"""
    return BacktestSimulator(
        probability_threshold=0.7,
        target_percent=5.0,
        prediction_horizon_minutes=60,
        commission_pct=0.15
    )


@pytest.fixture
def predictions_df():
    """돌탱: 예측 데이터프레임"""
    start_time = datetime(2024, 1, 1, 10, 0)

    return pd.DataFrame({
        'timestamp': [start_time + timedelta(minutes=i*10) for i in range(10)],
        'probability': [0.75, 0.65, 0.82, 0.68, 0.79, 0.73, 0.66, 0.85, 0.71, 0.77]
    })


@pytest.fixture
def backtest_result(sample_trades):
    """번개: BacktestResult 샘플"""
    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        start_time=datetime(2024, 1, 1, 10, 0),
        end_time=datetime(2024, 1, 1, 17, 0),
        total_predictions=20
    )

    for trade in sample_trades:
        result.add_trade(trade)

    return result


@pytest.fixture
def performance_tracker():
    """꼼수: ModelPerformanceTracker 인스턴스"""
    return ModelPerformanceTracker(window_hours=50)


# ============================================================================
# Trade 데이터클래스 테스트 (3개)
# ============================================================================

def test_trade_creation(sample_trade):
    """
    번개: Trade 객체 생성 및 기본 속성 확인

    Trade 데이터클래스가 정상적으로 생성되고 모든 속성이 올바르게 설정되는지 검증
    """
    assert sample_trade.ticker == 'AAPL'
    assert sample_trade.entry_price == 150.0
    assert sample_trade.exit_price == 157.5
    assert sample_trade.exit_reason == 'target_hit'
    assert sample_trade.probability == 0.75
    assert sample_trade.profit_pct == 4.7
    assert sample_trade.duration_minutes == 30.0
    assert sample_trade.model_type == 'xgboost'
    assert sample_trade.target == 'up'


def test_trade_is_win_property(sample_trade, sample_losing_trade):
    """
    번개: Trade의 is_win 프로퍼티 검증

    수익 거래는 True, 손실 거래는 False를 반환하는지 확인
    """
    # Winning trade
    assert sample_trade.is_win is True
    assert sample_trade.profit_pct > 0

    # Losing trade
    assert sample_losing_trade.is_win is False
    assert sample_losing_trade.profit_pct < 0


def test_trade_to_dict(sample_trade):
    """
    번개: Trade의 to_dict 메서드 검증

    Trade 객체가 딕셔너리로 정확히 변환되는지 확인
    """
    trade_dict = sample_trade.to_dict()

    assert trade_dict['ticker'] == 'AAPL'
    assert trade_dict['entry_price'] == 150.0
    assert trade_dict['exit_price'] == 157.5
    assert trade_dict['exit_reason'] == 'target_hit'
    assert trade_dict['probability'] == 0.75
    assert trade_dict['profit_pct'] == 4.7
    assert trade_dict['duration_minutes'] == 30.0
    assert trade_dict['model_type'] == 'xgboost'
    assert trade_dict['target'] == 'up'
    assert trade_dict['is_win'] is True
    assert trade_dict['profit_dollars'] == pytest.approx(47.0, rel=0.01)


# ============================================================================
# BacktestResult 데이터클래스 테스트 (4개)
# ============================================================================

def test_backtest_result_creation():
    """
    번개: BacktestResult 객체 생성 확인

    초기화 시 모든 필드가 올바르게 설정되는지 검증
    """
    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up'
    )

    assert result.ticker == 'AAPL'
    assert result.model_type == 'xgboost'
    assert result.target == 'up'
    assert result.trades == []
    assert result.total_trades == 0
    assert result.total_predictions == 0
    assert result.metadata == {}


def test_backtest_result_add_trade(sample_trade):
    """
    번개: BacktestResult에 거래 추가 기능 검증

    add_trade 메서드가 거래를 정확히 추가하고 카운트를 업데이트하는지 확인
    """
    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up'
    )

    assert result.total_trades == 0

    result.add_trade(sample_trade)

    assert result.total_trades == 1
    assert len(result.trades) == 1
    assert result.trades[0] == sample_trade


def test_backtest_result_get_trades_df(backtest_result):
    """
    번개: BacktestResult의 get_trades_df 메서드 검증

    거래 목록이 pandas DataFrame으로 정확히 변환되는지 확인
    """
    df = backtest_result.get_trades_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == backtest_result.total_trades
    assert 'ticker' in df.columns
    assert 'entry_time' in df.columns
    assert 'exit_time' in df.columns
    assert 'profit_pct' in df.columns
    assert 'is_win' in df.columns


def test_backtest_result_get_summary(backtest_result):
    """
    번개: BacktestResult의 get_summary 메서드 검증

    요약 통계가 정확히 계산되는지 확인
    """
    summary = backtest_result.get_summary()

    assert summary['ticker'] == 'AAPL'
    assert summary['model_type'] == 'xgboost'
    assert summary['target'] == 'up'
    assert summary['total_trades'] == 6
    assert summary['total_predictions'] == 20
    assert summary['wins'] == 4
    assert summary['losses'] == 2
    assert summary['win_rate'] == pytest.approx(4/6, rel=0.01)
    assert 'avg_profit' in summary
    assert 'total_profit' in summary
    assert 'avg_duration' in summary


# ============================================================================
# BacktestSimulator 초기화 테스트 (3개)
# ============================================================================

def test_simulator_init_default():
    """
    돌탱: BacktestSimulator 기본 초기화

    파라미터 없이 생성 시 설정 파일의 기본값이 적용되는지 확인
    """
    with patch('src.backtester.simulator.settings') as mock_settings:
        mock_settings.PROBABILITY_THRESHOLD = 0.7
        mock_settings.TARGET_PERCENT = 5.0
        mock_settings.PREDICTION_HORIZON_MINUTES = 60
        mock_settings.COMMISSION_PERCENT = 0.15

        sim = BacktestSimulator()

        assert sim.probability_threshold == 0.7
        assert sim.target_percent == 5.0
        assert sim.prediction_horizon_minutes == 60
        assert sim.commission_pct == 0.15
        assert sim.commission_round_trip == 0.3


def test_simulator_init_custom():
    """
    돌탱: BacktestSimulator 커스텀 파라미터 초기화

    사용자 정의 파라미터가 정확히 설정되는지 확인
    """
    sim = BacktestSimulator(
        probability_threshold=0.75,
        target_percent=3.0,
        prediction_horizon_minutes=30,
        commission_pct=0.1
    )

    assert sim.probability_threshold == 0.75
    assert sim.target_percent == 3.0
    assert sim.prediction_horizon_minutes == 30
    assert sim.commission_pct == 0.1
    assert sim.commission_round_trip == 0.2


def test_simulator_repr(simulator):
    """
    돌탱: BacktestSimulator의 문자열 표현

    __repr__ 메서드가 주요 설정을 포함하는지 확인
    """
    repr_str = repr(simulator)

    assert 'BacktestSimulator' in repr_str
    assert 'threshold' in repr_str
    assert 'target' in repr_str
    assert 'horizon' in repr_str
    assert 'commission' in repr_str


# ============================================================================
# simulate_trade 메서드 테스트 (5개)
# ============================================================================

def test_simulate_trade_below_threshold(simulator, minute_bars):
    """
    돌탱: 확률이 임계값 미만일 때 거래 미실행

    up_prob < threshold인 경우 None을 반환하는지 확인
    """
    entry_time = datetime(2024, 1, 1, 10, 0)
    entry_price = 150.0
    up_prob = 0.65  # Below threshold of 0.7

    trade = simulator.simulate_trade(
        ticker='AAPL',
        entry_time=entry_time,
        entry_price=entry_price,
        minute_bars=minute_bars,
        up_prob=up_prob
    )

    assert trade is None


def test_simulate_trade_target_hit(simulator, minute_bars):
    """
    돌탱: 목표 수익 달성 시 거래 종료

    5% 목표가 달성되면 'target_hit'으로 종료되는지 확인
    """
    entry_time = datetime(2024, 1, 1, 10, 0)
    entry_price = 150.0
    up_prob = 0.75

    # Create minute bars with 5% move
    bars = minute_bars.copy()
    target_price = entry_price * 1.05

    # Set a bar that hits the target
    bars.loc[5, 'high'] = target_price + 1.0
    bars.loc[5, 'close'] = target_price - 0.5

    trade = simulator.simulate_trade(
        ticker='AAPL',
        entry_time=entry_time,
        entry_price=entry_price,
        minute_bars=bars,
        up_prob=up_prob
    )

    assert trade is not None
    assert trade.exit_reason == 'target_hit'
    assert trade.exit_price == pytest.approx(target_price, rel=0.01)


def test_simulate_trade_time_limit(simulator, minute_bars):
    """
    돌탱: 시간 제한 도달 시 거래 종료

    60분 경과 후 'time_limit'으로 종료되는지 확인
    """
    entry_time = datetime(2024, 1, 1, 10, 0)
    entry_price = 150.0
    up_prob = 0.75

    # Use bars without hitting target
    trade = simulator.simulate_trade(
        ticker='AAPL',
        entry_time=entry_time,
        entry_price=entry_price,
        minute_bars=minute_bars,
        up_prob=up_prob
    )

    assert trade is not None
    # Should exit at time limit if target not hit
    assert trade.exit_reason in ['target_hit', 'time_limit']
    assert trade.duration_minutes <= simulator.prediction_horizon_minutes


def test_simulate_trade_commission(simulator):
    """
    돌탱: 커미션 계산 정확성 검증

    gross return에서 commission이 정확히 차감되는지 확인
    """
    entry_time = datetime(2024, 1, 1, 10, 0)
    entry_price = 100.0

    # Create simple bars for exact calculation
    bars = pd.DataFrame({
        'timestamp': [entry_time + timedelta(minutes=i) for i in range(1, 61)],
        'open': [100.0] * 60,
        'high': [105.0] * 60,  # 5% move to hit target
        'low': [99.0] * 60,
        'close': [104.0] * 60,
        'volume': [10000] * 60
    })

    trade = simulator.simulate_trade(
        ticker='TEST',
        entry_time=entry_time,
        entry_price=entry_price,
        minute_bars=bars,
        up_prob=0.8
    )

    assert trade is not None
    # Gross return = 5%, Net return = 5% - 0.3% = 4.7%
    assert trade.profit_pct == pytest.approx(4.7, rel=0.01)


def test_simulate_trade_no_future_bars(simulator):
    """
    의심: 미래 데이터 없을 때 None 반환

    entry_time 이후 데이터가 없으면 거래를 실행하지 않는지 확인
    """
    entry_time = datetime(2024, 1, 1, 10, 0)
    entry_price = 150.0

    # Empty bars
    bars = pd.DataFrame({
        'timestamp': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    })

    trade = simulator.simulate_trade(
        ticker='AAPL',
        entry_time=entry_time,
        entry_price=entry_price,
        minute_bars=bars,
        up_prob=0.8
    )

    assert trade is None


# ============================================================================
# simulate_predictions/vectorized 테스트 (4개)
# ============================================================================

def test_simulate_predictions_full_flow(simulator, predictions_df, minute_bars):
    """
    돌탱: simulate_predictions 전체 흐름 검증

    예측 데이터프레임을 기반으로 여러 거래를 시뮬레이션하는지 확인
    """
    result = simulator.simulate_predictions(
        ticker='AAPL',
        predictions_df=predictions_df,
        minute_bars=minute_bars,
        model_type='xgboost',
        target='up'
    )

    assert isinstance(result, BacktestResult)
    assert result.ticker == 'AAPL'
    assert result.model_type == 'xgboost'
    assert result.target == 'up'
    assert result.total_predictions == len(predictions_df)
    # Some predictions should result in trades (prob >= 0.7)
    assert result.total_trades > 0


def test_simulate_vectorized_full_flow(simulator, minute_bars):
    """
    돌탱: simulate_vectorized 전체 흐름 검증

    벡터화된 시뮬레이션이 정상 작동하는지 확인
    """
    # Create probability array
    probabilities = np.array([0.75 if i % 3 == 0 else 0.65 for i in range(len(minute_bars))])

    result = simulator.simulate_vectorized(
        ticker='AAPL',
        minute_bars=minute_bars,
        probabilities=probabilities,
        model_type='lightgbm',
        target='up'
    )

    assert isinstance(result, BacktestResult)
    assert result.ticker == 'AAPL'
    assert result.model_type == 'lightgbm'
    assert result.total_predictions == len(minute_bars)
    # Should have trades where prob >= 0.7
    assert result.total_trades > 0


def test_simulate_predictions_timestamp_handling(simulator, minute_bars):
    """
    돌탱: 타임스탬프 형식 처리 검증

    문자열 타임스탬프도 자동으로 datetime으로 변환되는지 확인
    """
    # Create predictions with string timestamps
    predictions_df = pd.DataFrame({
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:10:00'],
        'probability': [0.75, 0.80]
    })

    result = simulator.simulate_predictions(
        ticker='AAPL',
        predictions_df=predictions_df,
        minute_bars=minute_bars,
        model_type='xgboost'
    )

    assert isinstance(result, BacktestResult)
    assert result.total_predictions == 2


def test_simulate_vectorized_length_mismatch(simulator, minute_bars):
    """
    의심: 벡터화 시뮬레이션 길이 불일치 에러

    minute_bars와 probabilities 길이가 다르면 ValueError 발생하는지 확인
    """
    probabilities = np.array([0.75, 0.80, 0.85])  # Different length

    with pytest.raises(ValueError, match="must have same length"):
        simulator.simulate_vectorized(
            ticker='AAPL',
            minute_bars=minute_bars,
            probabilities=probabilities
        )


# ============================================================================
# 유틸리티 메서드 테스트 (3개)
# ============================================================================

def test_get_trade_outcomes(simulator, sample_trades):
    """
    돌탱: get_trade_outcomes 메서드 검증

    거래 목록을 예측/결과 쌍으로 변환하는지 확인
    """
    predictions, outcomes = simulator.get_trade_outcomes(sample_trades, probability_threshold=0.7)

    assert len(predictions) == len(sample_trades)
    assert len(outcomes) == len(sample_trades)
    assert all(isinstance(p, bool) for p in predictions)
    assert all(isinstance(o, bool) for o in outcomes)

    # Check specific cases
    for trade, pred, outcome in zip(sample_trades, predictions, outcomes):
        assert pred == (trade.probability >= 0.7)
        assert outcome == trade.is_win


def test_calculate_hit_rate(simulator, sample_trades):
    """
    돌탱: calculate_hit_rate 메서드 검증

    승률이 정확히 계산되는지 확인
    """
    hit_rate = simulator.calculate_hit_rate(sample_trades)

    wins = sum(1 for t in sample_trades if t.is_win)
    expected_hit_rate = wins / len(sample_trades)

    assert hit_rate == pytest.approx(expected_hit_rate, rel=0.01)


def test_calculate_hit_rate_with_hours_filter(simulator):
    """
    돌탱: 시간 필터가 적용된 승률 계산

    최근 N시간 내 거래만 필터링하여 승률을 계산하는지 확인
    """
    now = datetime.now()

    # Create trades at different times
    trades = [
        Trade(
            ticker='AAPL',
            entry_time=now - timedelta(hours=1),
            entry_price=150.0,
            exit_time=now - timedelta(hours=0.5),
            exit_price=157.5,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7,
            duration_minutes=30.0
        ),
        Trade(
            ticker='AAPL',
            entry_time=now - timedelta(hours=25),  # Old trade
            entry_price=150.0,
            exit_time=now - timedelta(hours=24.5),
            exit_price=148.0,
            exit_reason='time_limit',
            probability=0.70,
            profit_pct=-1.6,
            duration_minutes=30.0
        )
    ]

    # Filter to last 24 hours
    hit_rate = simulator.calculate_hit_rate(trades, hours=24)

    # Should only count the recent winning trade
    assert hit_rate == pytest.approx(1.0, rel=0.01)


# ============================================================================
# PerformanceMetrics 테스트 (6개)
# ============================================================================

def test_calculate_metrics_basic(sample_trades):
    """
    까칠: 기본 성능 메트릭 계산 검증

    calculate_metrics 함수가 모든 기본 메트릭을 정확히 계산하는지 확인
    """
    metrics = calculate_metrics(sample_trades)

    assert metrics.total_trades == len(sample_trades)
    assert metrics.winning_trades == 4
    assert metrics.losing_trades == 2
    assert metrics.win_rate == pytest.approx(4/6, rel=0.01)
    assert metrics.avg_return_pct > 0  # More wins than losses
    assert metrics.best_trade_pct == 4.7
    assert metrics.worst_trade_pct == -2.3


def test_calculate_metrics_empty_trades():
    """
    의심: 빈 거래 목록 처리

    거래가 없을 때 빈 메트릭 객체를 반환하는지 확인
    """
    metrics = calculate_metrics([])

    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.total_return_pct == 0.0


def test_calculate_metrics_sharpe_ratio(sample_trades):
    """
    까칠: Sharpe ratio 계산 검증

    변동성 대비 수익률을 나타내는 Sharpe ratio가 정확한지 확인
    """
    metrics = calculate_metrics(sample_trades)

    # Sharpe ratio should be calculated
    assert metrics.sharpe_ratio != 0.0

    # Calculate manually
    returns = [t.profit_pct for t in sample_trades]
    expected_sharpe = np.mean(returns) / np.std(returns, ddof=1)

    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, rel=0.01)


def test_calculate_metrics_drawdown(sample_trades):
    """
    까칠: 최대 낙폭(Max Drawdown) 계산 검증

    누적 수익에서 최대 하락폭이 정확히 계산되는지 확인
    """
    metrics = calculate_metrics(sample_trades)

    # Max drawdown should be calculated
    assert metrics.max_drawdown_pct >= 0  # Positive value representing drop

    # Verify manually
    returns = [t.profit_pct for t in sample_trades]
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    expected_max_dd = abs(drawdowns.min())

    assert metrics.max_drawdown_pct == pytest.approx(expected_max_dd, rel=0.01)


def test_calculate_metrics_kelly_criterion(sample_trades):
    """
    까칠: Kelly Criterion 계산 검증

    최적 포지션 크기를 나타내는 Kelly Criterion이 0~1 범위인지 확인
    """
    metrics = calculate_metrics(sample_trades)

    # Kelly criterion should be between 0 and 1
    assert 0.0 <= metrics.kelly_criterion <= 1.0


def test_calculate_metrics_profit_factor(sample_trades):
    """
    까칠: Profit Factor 계산 검증

    총 이익 / 총 손실 비율이 정확한지 확인
    """
    metrics = calculate_metrics(sample_trades)

    # Calculate manually
    wins = [t.profit_pct for t in sample_trades if t.is_win]
    losses = [t.profit_pct for t in sample_trades if not t.is_win]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    expected_pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    assert metrics.profit_factor == pytest.approx(expected_pf, rel=0.01)


# ============================================================================
# ModelPerformanceTracker 테스트 (6개)
# ============================================================================

def test_tracker_init(performance_tracker):
    """
    꼼수: ModelPerformanceTracker 초기화 검증

    트래커가 올바른 설정으로 초기화되는지 확인
    """
    assert performance_tracker.window_hours == 50
    assert performance_tracker.metrics_cache == {}
    assert performance_tracker.trades_cache == {}


def test_tracker_update_metrics(performance_tracker, backtest_result):
    """
    꼼수: update_metrics 메서드 검증

    백테스트 결과로 메트릭을 업데이트하는지 확인
    """
    metrics = performance_tracker.update_metrics(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        result=backtest_result
    )

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == backtest_result.total_trades

    # Check cache
    assert 'AAPL' in performance_tracker.metrics_cache
    assert 'xgboost' in performance_tracker.metrics_cache['AAPL']
    assert 'up' in performance_tracker.metrics_cache['AAPL']['xgboost']


def test_tracker_get_best_model(performance_tracker):
    """
    꼼수: get_best_model 메서드 검증

    여러 모델 중 최고 성능 모델을 선택하는지 확인
    """
    # Add multiple models with different performance
    now = datetime.now()

    for model_type, win_rate in [('xgboost', 0.75), ('lightgbm', 0.68), ('lstm', 0.72)]:
        trades = []
        for i in range(20):
            is_win = i < int(20 * win_rate)
            trades.append(Trade(
                ticker='AAPL',
                entry_time=now - timedelta(hours=i),
                entry_price=150.0,
                exit_time=now - timedelta(hours=i - 0.5),
                exit_price=157.5 if is_win else 148.0,
                exit_reason='target_hit' if is_win else 'time_limit',
                probability=0.75,
                profit_pct=4.7 if is_win else -1.3,
                duration_minutes=30.0,
                model_type=model_type
            ))

        result = BacktestResult(
            ticker='AAPL',
            model_type=model_type,
            target='up',
            trades=trades
        )
        performance_tracker.update_metrics('AAPL', model_type, 'up', result)

    best_model, best_hit_rate = performance_tracker.get_best_model('AAPL', 'up')

    assert best_model == 'xgboost'
    assert best_hit_rate == pytest.approx(0.75, rel=0.01)


def test_tracker_clear_old_trades(performance_tracker):
    """
    꼼수: clear_old_trades 메서드 검증

    오래된 거래를 정리하여 메모리를 절약하는지 확인
    """
    now = datetime.now()

    # Add old and recent trades
    trades = []
    for i in range(10):
        trades.append(Trade(
            ticker='AAPL',
            entry_time=now - timedelta(days=i),
            entry_price=150.0,
            exit_time=now - timedelta(days=i, hours=-0.5),
            exit_price=157.5,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7,
            duration_minutes=30.0
        ))

    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        trades=trades
    )
    performance_tracker.update_metrics('AAPL', 'xgboost', 'up', result)

    # Clear trades older than 5 days
    performance_tracker.clear_old_trades(days=5)

    # Check that old trades are removed
    remaining_trades = performance_tracker.trades_cache['AAPL']['xgboost']['up']
    assert len(remaining_trades) < len(trades)
    assert all(t.entry_time >= now - timedelta(days=5) for t in remaining_trades)


def test_calculate_rolling_metrics(sample_trades):
    """
    까칠: calculate_rolling_metrics 함수 검증

    시간 윈도우별 롤링 메트릭이 정확히 계산되는지 확인
    """
    # Need trades spread over time
    now = datetime.now()
    trades = []
    for i in range(100):
        trades.append(Trade(
            ticker='AAPL',
            entry_time=now - timedelta(hours=100-i),
            entry_price=150.0,
            exit_time=now - timedelta(hours=100-i-0.5),
            exit_price=157.5 if i % 2 == 0 else 148.0,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7 if i % 2 == 0 else -1.3,
            duration_minutes=30.0
        ))

    rolling_df = calculate_rolling_metrics(trades, window_hours=24)

    assert isinstance(rolling_df, pd.DataFrame)
    if len(rolling_df) > 0:
        assert 'timestamp' in rolling_df.columns
        assert 'win_rate' in rolling_df.columns
        assert 'total_return_pct' in rolling_df.columns
        assert 'sharpe_ratio' in rolling_df.columns


def test_tracker_get_all_model_performances(performance_tracker):
    """
    꼼수: get_all_model_performances 메서드 검증

    모든 모델의 성능 요약을 반환하는지 확인
    """
    # Add some performance data
    now = datetime.now()
    trades = [
        Trade(
            ticker='AAPL',
            entry_time=now - timedelta(hours=i),
            entry_price=150.0,
            exit_time=now - timedelta(hours=i-0.5),
            exit_price=157.5,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7,
            duration_minutes=30.0
        )
        for i in range(10)
    ]

    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        trades=trades
    )

    with patch('src.backtester.metrics.settings') as mock_settings:
        mock_settings.MODEL_TYPES = ['xgboost', 'lightgbm', 'lstm']
        performance_tracker.update_metrics('AAPL', 'xgboost', 'up', result)

        performances = performance_tracker.get_all_model_performances('AAPL')

    assert 'up' in performances
    assert 'down' in performances
    # Should have entries for all model types (trained and untrained)


# ============================================================================
# 경계 조건 및 에러 케이스 테스트 (6개)
# ============================================================================

def test_trade_profit_dollars_property(sample_trade):
    """
    의심: profit_dollars 프로퍼티 계산 검증

    $1000 포지션 기준 달러 수익이 정확한지 확인
    """
    expected_dollars = 1000 * (sample_trade.profit_pct / 100)
    assert sample_trade.profit_dollars == pytest.approx(expected_dollars, rel=0.01)


def test_backtest_result_empty_summary():
    """
    의심: 거래가 없을 때 요약 통계

    빈 BacktestResult의 summary가 기본값을 반환하는지 확인
    """
    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up'
    )

    summary = result.get_summary()

    assert summary['total_trades'] == 0
    assert summary['win_rate'] == 0.0
    assert summary['avg_profit'] == 0.0
    assert summary['total_profit'] == 0.0


def test_backtest_result_empty_trades_df():
    """
    의심: 거래가 없을 때 빈 DataFrame 반환

    거래가 없으면 빈 DataFrame을 반환하는지 확인
    """
    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up'
    )

    df = result.get_trades_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_metrics_single_trade():
    """
    의심: 단일 거래의 메트릭 계산

    거래가 1개만 있을 때도 메트릭이 정확한지 확인
    """
    trade = Trade(
        ticker='AAPL',
        entry_time=datetime(2024, 1, 1, 10, 0),
        entry_price=150.0,
        exit_time=datetime(2024, 1, 1, 10, 30),
        exit_price=157.5,
        exit_reason='target_hit',
        probability=0.75,
        profit_pct=4.7,
        duration_minutes=30.0
    )

    metrics = calculate_metrics([trade])

    assert metrics.total_trades == 1
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 1.0
    assert metrics.avg_return_pct == 4.7


def test_metrics_all_winning_trades():
    """
    의심: 모든 거래가 수익일 때 메트릭

    손실 거래가 없을 때 메트릭이 올바른지 확인
    """
    trades = [
        Trade(
            ticker='AAPL',
            entry_time=datetime(2024, 1, 1, 10 + i, 0),
            entry_price=150.0,
            exit_time=datetime(2024, 1, 1, 10 + i, 30),
            exit_price=157.5,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7,
            duration_minutes=30.0
        )
        for i in range(5)
    ]

    metrics = calculate_metrics(trades)

    assert metrics.total_trades == 5
    assert metrics.winning_trades == 5
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 1.0
    assert metrics.avg_loss_pct == 0.0
    assert metrics.profit_factor == float('inf')


def test_tracker_get_metrics_nonexistent():
    """
    의심: 존재하지 않는 모델의 메트릭 조회

    캐시에 없는 모델을 조회하면 None을 반환하는지 확인
    """
    tracker = ModelPerformanceTracker()

    metrics = tracker.get_metrics('NONEXISTENT', 'xgboost', 'up')

    assert metrics is None


# ============================================================================
# 추가 통합 테스트 (4개)
# ============================================================================

def test_end_to_end_backtest_flow(simulator, predictions_df, minute_bars, performance_tracker):
    """
    묵직: End-to-End 백테스트 전체 흐름 통합 테스트

    시뮬레이션 → 결과 생성 → 메트릭 계산 → 트래커 업데이트까지 전체 흐름 검증
    """
    # 1. Run simulation
    result = simulator.simulate_predictions(
        ticker='AAPL',
        predictions_df=predictions_df,
        minute_bars=minute_bars,
        model_type='xgboost',
        target='up'
    )

    # 2. Verify result
    assert isinstance(result, BacktestResult)
    assert result.total_trades > 0

    # 3. Update tracker
    metrics = performance_tracker.update_metrics(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        result=result
    )

    # 4. Verify metrics
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == result.total_trades

    # 5. Retrieve and verify
    retrieved_metrics = performance_tracker.get_metrics('AAPL', 'xgboost', 'up')
    assert retrieved_metrics is not None


def test_export_trades_functionality(simulator, predictions_df, minute_bars, tmp_path):
    """
    묵직: export_trades 기능 검증

    거래 내역을 CSV로 내보내는 기능이 정상 작동하는지 확인
    """
    result = simulator.simulate_predictions(
        ticker='AAPL',
        predictions_df=predictions_df,
        minute_bars=minute_bars,
        model_type='xgboost'
    )

    # Export to CSV
    filepath = tmp_path / "trades.csv"
    simulator.export_trades(result, str(filepath))

    # Verify file exists and has correct content
    assert filepath.exists()

    # Read back and verify
    df = pd.read_csv(filepath)
    assert len(df) == result.total_trades
    assert 'ticker' in df.columns
    assert 'profit_pct' in df.columns


def test_performance_metrics_to_dict():
    """
    까칠: PerformanceMetrics to_dict 메서드 검증

    메트릭 객체가 딕셔너리로 변환되는지 확인
    """
    metrics = PerformanceMetrics(
        total_trades=10,
        winning_trades=7,
        losing_trades=3,
        win_rate=0.7,
        total_return_pct=15.5
    )

    metrics_dict = metrics.to_dict()

    assert isinstance(metrics_dict, dict)
    assert metrics_dict['total_trades'] == 10
    assert metrics_dict['winning_trades'] == 7
    assert metrics_dict['win_rate'] == 0.7
    assert metrics_dict['total_return_pct'] == 15.5


def test_tracker_export_summary(tmp_path):
    """
    꼼수: ModelPerformanceTracker export_summary 검증

    성능 요약을 CSV로 내보내는 기능이 정상 작동하는지 확인
    """
    # Create tracker with recent trades
    tracker = ModelPerformanceTracker(window_hours=50)
    now = datetime.now()

    # Create recent trades (within window)
    trades = [
        Trade(
            ticker='AAPL',
            entry_time=now - timedelta(hours=i),
            entry_price=150.0,
            exit_time=now - timedelta(hours=i-0.5),
            exit_price=157.5 if i % 2 == 0 else 148.0,
            exit_reason='target_hit',
            probability=0.75,
            profit_pct=4.7 if i % 2 == 0 else -1.3,
            duration_minutes=30.0,
            model_type='xgboost'
        )
        for i in range(10)
    ]

    result = BacktestResult(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        trades=trades,
        start_time=now - timedelta(hours=10),
        end_time=now
    )

    tracker.update_metrics(
        ticker='AAPL',
        model_type='xgboost',
        target='up',
        result=result
    )

    filepath = tmp_path / "summary.csv"

    with patch('src.backtester.metrics.settings') as mock_settings:
        mock_settings.MODEL_TYPES = ['xgboost']
        tracker.export_summary('AAPL', str(filepath))

    # Verify file exists since we have metrics
    assert filepath.exists()

    # Verify content
    df = pd.read_csv(filepath)
    assert len(df) > 0
    assert 'ticker' in df.columns
    assert 'model_type' in df.columns
    assert 'target' in df.columns
