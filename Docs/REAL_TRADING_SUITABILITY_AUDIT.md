# 실제 트레이딩 적합성 감사 보고서

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 전체 팀
**심각도**: 높음 (실거래 전 반드시 수정 필요)

---

## 요약

FiveForFree 시스템의 실제 트레이딩 적합성을 검토한 결과, **12가지 주요 문제점**을 발견했습니다. 현재 상태로는 실거래에 적합하지 않습니다.

| 카테고리 | 문제 수 | 심각도 |
|---------|--------|--------|
| 리스크 관리 | 4 | 치명적 |
| 백테스트 현실성 | 4 | 높음 |
| 시장 현실 반영 | 2 | 높음 |
| 모델 품질 | 2 | 높음 |

---

## 1. 리스크 관리 부재 (치명적)

### 1.1 Stop Loss 없음

**현재 상태**: 손절 로직이 전혀 없음
```python
# simulator.py - 현재 로직
# 진입 후 1% 익절 또는 60분 타임아웃만 존재
if high_return >= self.target_percent:  # 익절만 있음
    exit_price = entry_price * (1 + self.target_percent / 100)
```

**문제점**:
- 60분 동안 -10%, -20% 손실도 그대로 유지
- 극단적 손실 발생 가능

**해결 방안**:
```python
# 손절 로직 추가
STOP_LOSS_PERCENT = 0.5  # -0.5%에서 손절

if low_return <= -STOP_LOSS_PERCENT:
    exit_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
    exit_reason = 'stop_loss'
    break
```

---

### 1.2 Position Sizing 없음

**현재 상태**: 고정 $1000 포지션
```python
# Trade.profit_dollars
def profit_dollars(self) -> float:
    return 1000 * (self.profit_pct / 100)  # 고정 금액
```

**문제점**:
- 계좌 크기 무관하게 동일 금액 베팅
- 리스크 관리 불가능

**해결 방안**:
```python
class PositionSizer:
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float = 0.01,  # 1% 리스크
        stop_loss_pct: float = 0.5
    ) -> float:
        """
        Kelly Criterion 또는 Fixed Fractional 기반 포지션 사이징
        """
        max_loss_amount = account_balance * risk_per_trade
        position_size = max_loss_amount / (stop_loss_pct / 100)
        return min(position_size, account_balance * 0.25)  # 최대 25%
```

---

### 1.3 Maximum Drawdown 제어 없음

**현재 상태**: 연속 손실에 대한 제어 없음

**문제점**:
- 나쁜 날 연속 손실로 계좌 전체 위험
- 시스템 리스크 관리 부재

**해결 방안**:
```python
class RiskManager:
    def __init__(self):
        self.daily_loss = 0.0
        self.max_daily_loss = 0.03  # 일일 최대 3% 손실

    def can_trade(self) -> bool:
        return self.daily_loss < self.max_daily_loss

    def record_trade(self, profit_pct: float):
        if profit_pct < 0:
            self.daily_loss += abs(profit_pct)
```

---

### 1.4 낮은 확률 임계값

**현재 상태**:
```python
# settings.py
PROBABILITY_THRESHOLD: float = 0.10  # 10%만 넘으면 진입
```

**문제점**:
- 10% 확률로 진입 = 거의 모든 신호에 베팅
- 낮은 확신도 거래가 많아짐

**해결 방안**:
```python
PROBABILITY_THRESHOLD: float = 0.60  # 60% 이상만 진입
# 또는 동적 임계값
MIN_PRECISION_THRESHOLD: float = 0.50  # 최소 50% Precision 요구
```

---

## 2. 백테스트 현실성 문제 (높음)

### 2.1 Slippage 미반영

**현재 상태**: 원하는 가격에 정확히 체결된다고 가정
```python
# simulator.py
exit_price = entry_price * (1 + self.target_percent / 100)  # 정확한 가격
```

**실제 시장**:
- 지정가 주문: 체결 안 될 수 있음
- 시장가 주문: 호가창 따라 슬리피지 발생
- 변동성 큰 시간대: 0.1~0.5% 슬리피지 일반적

**해결 방안**:
```python
SLIPPAGE_PERCENT = 0.05  # 0.05% 슬리피지 가정

# 진입 시
actual_entry_price = entry_price * (1 + SLIPPAGE_PERCENT / 100)

# 청산 시
actual_exit_price = exit_price * (1 - SLIPPAGE_PERCENT / 100)
```

---

### 2.2 Look-ahead Bias (미래 정보 사용)

**현재 상태**:
```python
# simulator.py - 문제 코드
high_return = ((high - entry_price) / entry_price) * 100
if high_return >= self.target_percent:  # 봉이 완성되기 전에 high를 알 수 없음
    exit_price = entry_price * (1 + self.target_percent / 100)
```

**문제점**:
- 1분봉의 `high` 가격은 봉이 완성된 후에만 알 수 있음
- 실시간에서는 현재가만 알 수 있음

**해결 방안**:
```python
# 실시간 시뮬레이션 (tick-by-tick)
def simulate_realistic(self, ticker, entry_time, entry_price, ticks):
    for tick in ticks:
        current_price = tick['price']
        current_return = ((current_price - entry_price) / entry_price) * 100

        if current_return >= self.target_percent:
            # 실제로 도달한 가격 사용
            exit_price = current_price
            break
```

---

### 2.3 낮은 수수료 가정

**현재 상태**:
```python
COMMISSION_PERCENT: float = 0.1  # 편도 0.1%
commission_round_trip = 0.2%  # 왕복 0.2%
```

**실제 비용**:
- 브로커 수수료: 0.05~0.3% (브로커마다 다름)
- SEC Fee: ~0.003%
- FINRA TAF: ~0.0001%
- 스프레드: 0.02~0.1% (유동성에 따라)

**해결 방안**:
```python
class TradingCosts:
    broker_commission = 0.001  # 0.1%
    sec_fee = 0.00003  # SEC fee
    finra_taf = 0.000001  # FINRA TAF
    estimated_spread = 0.0005  # 평균 스프레드

    def total_round_trip(self) -> float:
        return (self.broker_commission * 2 +
                self.sec_fee +
                self.finra_taf +
                self.estimated_spread * 2)
```

---

### 2.4 Market Impact 미반영

**현재 상태**: 주문이 시장에 영향을 주지 않는다고 가정

**문제점**:
- 대량 주문 시 가격 밀림 현상
- $10,000+ 주문은 소형주에서 significant impact

**해결 방안**:
```python
def calculate_market_impact(position_size: float, avg_volume: float) -> float:
    """
    Kyle's Lambda 기반 시장 충격 추정
    """
    participation_rate = position_size / avg_volume
    if participation_rate < 0.01:
        return 0.0  # 무시 가능
    elif participation_rate < 0.05:
        return 0.001 * participation_rate * 100  # 선형 증가
    else:
        return 0.01  # 최대 1% 충격
```

---

## 3. 시장 현실 미반영 (높음)

### 3.1 유동성 검증 없음

**현재 상태**: 거래량 확인 없이 진입
```python
# 현재 코드에 volume 체크 없음
if up_prob >= self.probability_threshold:
    # 바로 진입
```

**문제점**:
- 저유동성 종목에서 체결 불가
- 스프레드가 넓어 손실 발생

**해결 방안**:
```python
MIN_DOLLAR_VOLUME = 1_000_000  # 최소 $1M 일일 거래대금

def check_liquidity(ticker: str, minute_bars: pd.DataFrame) -> bool:
    recent_volume = minute_bars['volume'].tail(20).mean()
    recent_price = minute_bars['close'].iloc[-1]
    dollar_volume = recent_volume * recent_price * 390  # 하루 분 수

    return dollar_volume >= MIN_DOLLAR_VOLUME
```

---

### 3.2 거래 시간 검증 없음

**현재 상태**: 장 외 시간에도 신호 생성 가능

**문제점**:
- Pre-market, After-hours는 유동성 낮음
- 스프레드 넓고 슬리피지 큼

**해결 방안**:
```python
def is_regular_trading_hours(timestamp: datetime) -> bool:
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = timestamp.time()

    return market_open <= current_time <= market_close
```

---

## 4. 모델 품질 문제 (높음)

### 4.1 Signal Rate 0% 문제 (이미 보고됨)

**현재 상태**: 모델이 0만 예측
- Accuracy: 95%+ (착시)
- Signal Rate: 0~1%
- Precision/Recall: 0%

**해결 방안**: `SIGNAL_RATE_IMPROVEMENT_STRATEGY.md` 참조

---

### 4.2 과적합 의심 피처

**현재 상태**: `feature_engineer.py`에서 일부 피처 제외
```python
EXCLUDED_FEATURES = {
    'returns_1m', 'returns_5m', ...  # Sign flip features
    'vpt_cumsum',  # High importance but unstable
}
```

**추가 검토 필요**:
- 남은 피처들도 Walk-forward 검증 필요
- 특히 `volatility_ratio`, `price_acceleration` 검토

---

## 5. 구현 우선순위

### 즉시 수정 (거래 전 필수)

| 순위 | 항목 | 난이도 | 영향도 |
|------|------|--------|--------|
| 1 | Stop Loss 추가 | 낮음 | 치명적 |
| 2 | 확률 임계값 상향 (60%+) | 낮음 | 높음 |
| 3 | Signal Rate 개선 | 중간 | 치명적 |
| 4 | Slippage 반영 | 낮음 | 높음 |

### 단기 수정 (1주 내)

| 순위 | 항목 | 난이도 | 영향도 |
|------|------|--------|--------|
| 5 | Position Sizing | 중간 | 높음 |
| 6 | Daily Drawdown 제어 | 중간 | 높음 |
| 7 | 유동성 체크 | 낮음 | 중간 |
| 8 | 거래시간 검증 | 낮음 | 중간 |

### 중기 수정 (1개월 내)

| 순위 | 항목 | 난이도 | 영향도 |
|------|------|--------|--------|
| 9 | Market Impact 모델링 | 높음 | 중간 |
| 10 | Look-ahead Bias 제거 | 높음 | 중간 |
| 11 | 실제 비용 구조 반영 | 중간 | 낮음 |
| 12 | Walk-forward 검증 | 높음 | 높음 |

---

## 6. 수정 대상 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `config/settings.py` | PROBABILITY_THRESHOLD, STOP_LOSS_PERCENT 추가 |
| `src/backtester/simulator.py` | Stop loss, Slippage, Liquidity check |
| `src/predictor/realtime_predictor.py` | Trading hours 검증 |
| `src/trainer/gpu_trainer.py` | Class weight (이미 보고됨) |
| 신규: `src/risk/position_sizer.py` | Position sizing 로직 |
| 신규: `src/risk/risk_manager.py` | Daily drawdown 관리 |

---

## 7. 검증 체크리스트

실거래 전 반드시 확인:

- [ ] Stop Loss 구현 및 테스트
- [ ] Signal Rate > 10% 확인
- [ ] Precision > 50% 확인
- [ ] 백테스트에 Slippage 반영
- [ ] Position Sizing 구현
- [ ] Daily Drawdown 제어 구현
- [ ] 유동성 필터 추가
- [ ] 정규 거래시간 필터 추가
- [ ] Paper Trading 최소 2주 진행
- [ ] 실제 체결 데이터와 백테스트 비교

---

## 8. 결론

현재 시스템은 **연구/백테스트 단계**에 적합하며, **실거래에는 부적합**합니다.

**주요 위험**:
1. 리스크 관리 부재로 큰 손실 가능
2. 백테스트가 실제 성과를 과대 추정
3. 모델이 실제 매매 신호를 생성하지 않음

**권장 사항**:
1. 위 12가지 문제 수정
2. Paper Trading으로 2주+ 검증
3. 소액으로 Live Trading 시작 (최대 $1,000)
4. 실제 성과 데이터 수집 후 점진적 확대

---

*분석팀장 작성*
