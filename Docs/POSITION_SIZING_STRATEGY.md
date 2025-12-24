# Position Sizing 전략 분석서

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 개발팀장
**목적**: 리스크 관리를 위한 포지션 사이징 전략 수립

---

## 1. 개요

### 1.1 현재 문제
```python
# 현재 코드 (simulator.py)
def profit_dollars(self) -> float:
    return 1000 * (self.profit_pct / 100)  # 고정 $1000
```

**문제점**:
- 계좌 크기와 무관한 고정 금액
- 리스크 대비 수익 최적화 불가
- 연속 손실 시 계좌 전체 위험

### 1.2 Position Sizing의 중요성

| 시나리오 | 고정 $1000 | 계좌의 2% |
|---------|-----------|-----------|
| 계좌 $10,000 | 10% 리스크 | 2% 리스크 |
| 계좌 $50,000 | 2% 리스크 | 2% 리스크 |
| 계좌 $100,000 | 1% 리스크 | 2% 리스크 |

**결론**: Position Sizing은 일관된 리스크 관리의 핵심

---

## 2. Position Sizing 방법론

### 2.1 Kelly Criterion (켈리 공식)

#### 이론
수학적으로 장기 복리 수익을 최대화하는 베팅 비율

```
f* = (p × b - q) / b

f* = 최적 베팅 비율
p  = 승률 (Win Rate)
q  = 패률 (1 - p)
b  = 손익비 (Reward/Risk Ratio)
```

#### 예시
- 승률 (p): 55%
- 평균 수익: 1.0%
- 평균 손실: 0.5%
- 손익비 (b): 1.0 / 0.5 = 2.0

```
f* = (0.55 × 2.0 - 0.45) / 2.0
f* = (1.1 - 0.45) / 2.0
f* = 0.325 = 32.5%
```

#### 실제 적용 (Half-Kelly)
Full Kelly는 변동성이 너무 크므로 **Half-Kelly (50%)** 권장

```python
class KellyCriterion:
    def __init__(self, kelly_fraction: float = 0.5):
        """
        Args:
            kelly_fraction: Kelly 비율 (0.5 = Half-Kelly)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        account_balance: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float
    ) -> float:
        """
        Kelly Criterion 기반 포지션 크기 계산

        Args:
            account_balance: 계좌 잔고
            win_rate: 승률 (0.0 ~ 1.0)
            avg_win_pct: 평균 수익률 (%)
            avg_loss_pct: 평균 손실률 (%, 양수로 입력)

        Returns:
            position_size: 포지션 금액
        """
        if avg_loss_pct == 0:
            return 0

        # 손익비 계산
        reward_risk_ratio = avg_win_pct / avg_loss_pct

        # Kelly 공식
        kelly = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio

        # 음수면 베팅하지 않음
        if kelly <= 0:
            return 0

        # Half-Kelly 적용
        kelly_adjusted = kelly * self.kelly_fraction

        # 최대 25% 제한
        kelly_capped = min(kelly_adjusted, 0.25)

        return account_balance * kelly_capped
```

#### 장단점
| 장점 | 단점 |
|------|------|
| 수학적 최적화 | 정확한 승률/손익비 필요 |
| 장기 복리 극대화 | 높은 변동성 |
| 파산 방지 내장 | 모델 오류에 민감 |

---

### 2.2 Fixed Fractional (고정 비율)

#### 이론
계좌의 일정 비율만 리스크에 노출

```
Position Size = Account × Risk Percent / Stop Loss Percent
```

#### 예시
- 계좌: $50,000
- 거래당 리스크: 1%
- Stop Loss: 0.5%

```
Position Size = $50,000 × 0.01 / 0.005
Position Size = $10,000
```

#### 구현
```python
class FixedFractional:
    def __init__(
        self,
        risk_per_trade: float = 0.01,  # 거래당 1% 리스크
        max_position_pct: float = 0.25  # 최대 25%
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_pct: float
    ) -> float:
        """
        Fixed Fractional 기반 포지션 크기 계산

        Args:
            account_balance: 계좌 잔고
            stop_loss_pct: 손절 비율 (%, 양수로 입력)

        Returns:
            position_size: 포지션 금액
        """
        if stop_loss_pct <= 0:
            return 0

        # 최대 손실 금액
        max_loss_amount = account_balance * self.risk_per_trade

        # 포지션 크기 계산
        position_size = max_loss_amount / (stop_loss_pct / 100)

        # 최대 비율 제한
        max_position = account_balance * self.max_position_pct

        return min(position_size, max_position)
```

#### 장단점
| 장점 | 단점 |
|------|------|
| 간단하고 직관적 | 수익 최적화 아님 |
| 일관된 리스크 관리 | 승률/손익비 무시 |
| 모델 오류에 강건 | 보수적 (낮은 수익) |

---

### 2.3 변동성 기반 (ATR Sizing)

#### 이론
시장 변동성에 따라 포지션 크기 조절
- 변동성 높음 → 작은 포지션
- 변동성 낮음 → 큰 포지션

```
Position Size = Account × Risk / (ATR × Multiplier)
```

#### 구현
```python
class ATRPositionSizer:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        atr_multiplier: float = 2.0,
        atr_period: int = 14
    ):
        """
        Args:
            risk_per_trade: 거래당 리스크 비율
            atr_multiplier: ATR 배수 (손절 거리)
            atr_period: ATR 계산 기간
        """
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        atr: float
    ) -> float:
        """
        ATR 기반 포지션 크기 계산

        Args:
            account_balance: 계좌 잔고
            current_price: 현재 가격
            atr: ATR 값 (Average True Range)

        Returns:
            position_size: 포지션 금액
        """
        if atr <= 0 or current_price <= 0:
            return 0

        # ATR 기반 손절 거리 (가격 기준)
        stop_distance = atr * self.atr_multiplier

        # 손절 비율
        stop_loss_pct = (stop_distance / current_price) * 100

        # 최대 손실 금액
        max_loss_amount = account_balance * self.risk_per_trade

        # 포지션 크기 (주식 수)
        shares = max_loss_amount / stop_distance

        # 포지션 금액
        position_size = shares * current_price

        # 최대 25% 제한
        max_position = account_balance * 0.25

        return min(position_size, max_position)

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """ATR 기반 손절가 계산"""
        return entry_price - (atr * self.atr_multiplier)
```

#### 장단점
| 장점 | 단점 |
|------|------|
| 시장 상황 반영 | ATR 계산 필요 |
| 변동성 적응형 | 추가 데이터 필요 |
| 리스크 정규화 | 급변 시 지연 |

---

## 3. 위험 관리 규칙

### 3.1 종목당 최대 비중

```python
class PositionLimits:
    # 단일 종목 최대 비중
    MAX_SINGLE_POSITION = 0.10  # 10%

    # 섹터당 최대 비중
    MAX_SECTOR_EXPOSURE = 0.30  # 30%

    # 전체 투자 비중
    MAX_TOTAL_EXPOSURE = 0.80  # 80% (20% 현금 유지)
```

**이유**:
- 단일 종목 리스크 제한
- 섹터 집중 방지
- 현금 버퍼 유지

---

### 3.2 동시 보유 종목 수 제한

```python
class PortfolioLimits:
    def __init__(self):
        # 동시 보유 최대 종목 수
        self.max_positions = 5

        # 종목당 최소 금액 (거래 비용 고려)
        self.min_position_size = 1000  # $1,000

    def can_open_position(
        self,
        current_positions: int,
        position_size: float
    ) -> bool:
        """새 포지션 진입 가능 여부"""
        if current_positions >= self.max_positions:
            return False
        if position_size < self.min_position_size:
            return False
        return True
```

**권장 설정**:
| 계좌 크기 | 최대 종목 수 | 종목당 최소 금액 |
|----------|-------------|-----------------|
| $10,000 | 3 | $1,000 |
| $25,000 | 5 | $2,000 |
| $50,000 | 7 | $3,000 |
| $100,000+ | 10 | $5,000 |

---

### 3.3 일일 최대 손실 한도

```python
class DailyRiskManager:
    def __init__(
        self,
        max_daily_loss: float = 0.03,  # 일일 최대 3% 손실
        max_trades_per_day: int = 10,
        cooldown_after_loss: int = 2   # 연속 손실 후 대기 거래 수
    ):
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_after_loss = cooldown_after_loss

        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_remaining = 0

    def record_trade(self, profit_pct: float):
        """거래 결과 기록"""
        self.daily_pnl += profit_pct
        self.trades_today += 1

        if profit_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.cooldown_remaining = self.cooldown_after_loss
        else:
            self.consecutive_losses = 0

    def can_trade(self, account_balance: float) -> tuple[bool, str]:
        """거래 가능 여부 및 사유"""
        # 일일 손실 한도 체크
        if abs(self.daily_pnl) >= self.max_daily_loss * 100:
            return False, "Daily loss limit reached"

        # 일일 거래 횟수 체크
        if self.trades_today >= self.max_trades_per_day:
            return False, "Max trades per day reached"

        # 쿨다운 체크
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False, f"Cooldown active ({self.cooldown_remaining} trades remaining)"

        return True, "OK"

    def reset_daily(self):
        """일일 초기화 (매일 장 시작 전)"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
```

---

## 4. 통합 리스크 관리 시스템

### 4.1 전체 구조

```python
class RiskManager:
    """통합 리스크 관리 시스템"""

    def __init__(
        self,
        account_balance: float,
        risk_method: str = "fixed_fractional",  # kelly, fixed_fractional, atr
        risk_per_trade: float = 0.01,
        max_daily_loss: float = 0.03,
        max_positions: int = 5
    ):
        self.account_balance = account_balance
        self.risk_method = risk_method

        # Position Sizers
        self.kelly = KellyCriterion(kelly_fraction=0.5)
        self.fixed = FixedFractional(risk_per_trade=risk_per_trade)
        self.atr_sizer = ATRPositionSizer(risk_per_trade=risk_per_trade)

        # Risk Limits
        self.position_limits = PositionLimits()
        self.portfolio_limits = PortfolioLimits()
        self.portfolio_limits.max_positions = max_positions
        self.daily_risk = DailyRiskManager(max_daily_loss=max_daily_loss)

        # Portfolio State
        self.current_positions = {}  # {ticker: position_info}

    def calculate_position_size(
        self,
        ticker: str,
        current_price: float,
        stop_loss_pct: float = 0.5,
        win_rate: float = None,
        avg_win_pct: float = None,
        avg_loss_pct: float = None,
        atr: float = None
    ) -> dict:
        """
        포지션 크기 계산 (모든 제한 적용)

        Returns:
            {
                'can_trade': bool,
                'position_size': float,
                'shares': int,
                'reason': str
            }
        """
        result = {
            'can_trade': False,
            'position_size': 0,
            'shares': 0,
            'reason': ''
        }

        # 1. 일일 리스크 체크
        can_trade, reason = self.daily_risk.can_trade(self.account_balance)
        if not can_trade:
            result['reason'] = reason
            return result

        # 2. 포지션 수 체크
        if len(self.current_positions) >= self.portfolio_limits.max_positions:
            result['reason'] = "Max positions reached"
            return result

        # 3. 포지션 크기 계산
        if self.risk_method == "kelly" and all([win_rate, avg_win_pct, avg_loss_pct]):
            raw_size = self.kelly.calculate_position_size(
                self.account_balance, win_rate, avg_win_pct, avg_loss_pct
            )
        elif self.risk_method == "atr" and atr:
            raw_size = self.atr_sizer.calculate_position_size(
                self.account_balance, current_price, atr
            )
        else:
            raw_size = self.fixed.calculate_position_size(
                self.account_balance, stop_loss_pct
            )

        # 4. 최대 비중 제한
        max_single = self.account_balance * self.position_limits.MAX_SINGLE_POSITION
        position_size = min(raw_size, max_single)

        # 5. 최소 금액 체크
        if position_size < self.portfolio_limits.min_position_size:
            result['reason'] = f"Position too small (${position_size:.0f} < ${self.portfolio_limits.min_position_size})"
            return result

        # 6. 주식 수 계산
        shares = int(position_size / current_price)
        if shares <= 0:
            result['reason'] = "Cannot afford even 1 share"
            return result

        # 최종 결과
        result['can_trade'] = True
        result['position_size'] = shares * current_price
        result['shares'] = shares
        result['reason'] = "OK"

        return result

    def open_position(self, ticker: str, shares: int, entry_price: float):
        """포지션 오픈 기록"""
        self.current_positions[ticker] = {
            'shares': shares,
            'entry_price': entry_price,
            'position_value': shares * entry_price
        }

    def close_position(self, ticker: str, exit_price: float, profit_pct: float):
        """포지션 청산 기록"""
        if ticker in self.current_positions:
            del self.current_positions[ticker]
        self.daily_risk.record_trade(profit_pct)

    def update_balance(self, new_balance: float):
        """계좌 잔고 업데이트"""
        self.account_balance = new_balance
```

---

## 5. Simulator.py 적용 가이드

### 5.1 기존 코드 수정

```python
# simulator.py - 수정 버전

from src.risk.risk_manager import RiskManager

class BacktestSimulator:
    def __init__(
        self,
        probability_threshold: float = 0.60,  # 10% → 60%
        target_percent: float = 1.0,
        stop_loss_percent: float = 0.5,       # 신규 추가
        slippage_percent: float = 0.05,       # 신규 추가
        prediction_horizon_minutes: int = 60,
        commission_pct: float = 0.1,
        # 리스크 관리 파라미터
        initial_balance: float = 10000,
        risk_per_trade: float = 0.01,
        max_daily_loss: float = 0.03,
        max_positions: int = 5
    ):
        # 기존 파라미터
        self.probability_threshold = probability_threshold
        self.target_percent = target_percent
        self.stop_loss_percent = stop_loss_percent  # 신규
        self.slippage_percent = slippage_percent    # 신규
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.commission_pct = commission_pct

        # 리스크 관리자 초기화
        self.risk_manager = RiskManager(
            account_balance=initial_balance,
            risk_method="fixed_fractional",
            risk_per_trade=risk_per_trade,
            max_daily_loss=max_daily_loss,
            max_positions=max_positions
        )

    def simulate_trade(
        self,
        ticker: str,
        entry_time: datetime,
        entry_price: float,
        minute_bars: pd.DataFrame,
        up_prob: float,
        model_type: str = None
    ) -> Optional[Trade]:
        """수정된 거래 시뮬레이션"""

        # 1. 확률 임계값 체크
        if up_prob < self.probability_threshold:
            return None

        # 2. 포지션 크기 계산
        sizing_result = self.risk_manager.calculate_position_size(
            ticker=ticker,
            current_price=entry_price,
            stop_loss_pct=self.stop_loss_percent
        )

        if not sizing_result['can_trade']:
            logger.debug(f"Trade rejected: {sizing_result['reason']}")
            return None

        # 3. Slippage 적용
        actual_entry_price = entry_price * (1 + self.slippage_percent / 100)

        # 4. 손절가 계산
        stop_loss_price = actual_entry_price * (1 - self.stop_loss_percent / 100)

        # 5. 거래 시뮬레이션 (Stop Loss 포함)
        exit_price, exit_time, exit_reason = self._simulate_with_stop_loss(
            minute_bars, entry_time, actual_entry_price, stop_loss_price
        )

        # 6. Slippage 적용 (청산)
        if exit_reason == 'stop_loss':
            actual_exit_price = exit_price * (1 - self.slippage_percent / 100)
        else:
            actual_exit_price = exit_price * (1 - self.slippage_percent / 100)

        # 7. 수익률 계산
        gross_return = ((actual_exit_price - actual_entry_price) / actual_entry_price) * 100
        net_return = gross_return - (self.commission_pct * 2)

        # 8. 포지션 기록
        self.risk_manager.close_position(ticker, actual_exit_price, net_return)

        return Trade(
            ticker=ticker,
            entry_time=entry_time,
            entry_price=actual_entry_price,
            exit_time=exit_time,
            exit_price=actual_exit_price,
            exit_reason=exit_reason,
            probability=up_prob,
            profit_pct=net_return,
            duration_minutes=(exit_time - entry_time).total_seconds() / 60,
            model_type=model_type,
            position_size=sizing_result['position_size'],  # 신규
            shares=sizing_result['shares']                 # 신규
        )

    def _simulate_with_stop_loss(
        self,
        minute_bars: pd.DataFrame,
        entry_time: datetime,
        entry_price: float,
        stop_loss_price: float
    ) -> tuple:
        """Stop Loss 포함 시뮬레이션"""
        end_time = entry_time + timedelta(minutes=self.prediction_horizon_minutes)

        future_bars = minute_bars[
            (minute_bars['timestamp'] > entry_time) &
            (minute_bars['timestamp'] <= end_time)
        ].sort_values('timestamp')

        for _, row in future_bars.iterrows():
            # Stop Loss 체크 (먼저!)
            if row['low'] <= stop_loss_price:
                return stop_loss_price, row['timestamp'], 'stop_loss'

            # Target 체크
            target_price = entry_price * (1 + self.target_percent / 100)
            if row['high'] >= target_price:
                return target_price, row['timestamp'], 'target_hit'

        # 시간 만료
        last_bar = future_bars.iloc[-1]
        return last_bar['close'], last_bar['timestamp'], 'time_limit'
```

### 5.2 Trade 클래스 수정

```python
@dataclass
class Trade:
    ticker: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # 'target_hit', 'stop_loss', 'time_limit'
    probability: float
    profit_pct: float
    duration_minutes: float
    model_type: Optional[str] = None
    position_size: float = 1000.0   # 신규: 포지션 금액
    shares: int = 0                  # 신규: 주식 수

    @property
    def profit_dollars(self) -> float:
        """실제 수익 금액"""
        return self.position_size * (self.profit_pct / 100)
```

---

## 6. 권장 설정

### 6.1 보수적 설정 (초보자/소액)

```python
# 계좌: $10,000 이하
simulator = BacktestSimulator(
    probability_threshold=0.70,    # 70% 이상만 진입
    stop_loss_percent=0.5,         # -0.5% 손절
    slippage_percent=0.05,
    initial_balance=10000,
    risk_per_trade=0.01,           # 거래당 1% 리스크
    max_daily_loss=0.02,           # 일일 2% 손실 제한
    max_positions=3
)
```

### 6.2 중간 설정 (경험자/중간 금액)

```python
# 계좌: $10,000 ~ $50,000
simulator = BacktestSimulator(
    probability_threshold=0.60,    # 60% 이상만 진입
    stop_loss_percent=0.5,
    slippage_percent=0.05,
    initial_balance=25000,
    risk_per_trade=0.015,          # 거래당 1.5% 리스크
    max_daily_loss=0.03,           # 일일 3% 손실 제한
    max_positions=5
)
```

### 6.3 공격적 설정 (전문가/대규모)

```python
# 계좌: $50,000+
simulator = BacktestSimulator(
    probability_threshold=0.55,    # 55% 이상 진입
    stop_loss_percent=0.5,
    slippage_percent=0.03,         # 대량 주문 슬리피지 낮음
    initial_balance=100000,
    risk_per_trade=0.02,           # 거래당 2% 리스크
    max_daily_loss=0.05,           # 일일 5% 손실 제한
    max_positions=10
)
```

---

## 7. 검증 체크리스트

구현 후 확인 사항:

- [ ] 포지션 크기가 계좌 대비 10% 이하인지
- [ ] 동시 보유 종목이 제한 내인지
- [ ] 일일 손실이 3% 이하인지
- [ ] Stop Loss가 정상 작동하는지
- [ ] Slippage가 반영되는지
- [ ] 수익 금액이 포지션 크기 기준인지

---

## 8. 요약

| 방법 | 적합 상황 | 복잡도 |
|------|----------|--------|
| Fixed Fractional | 대부분의 경우 (권장) | 낮음 |
| Kelly Criterion | 검증된 전략 + 정확한 통계 | 중간 |
| ATR Sizing | 변동성 높은 시장 | 중간 |

**권장**: Fixed Fractional (1~2% 리스크) + Daily Loss Limit (3%)로 시작

---

*분석팀장 작성*
