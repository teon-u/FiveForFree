# Morgorithm Analytics - Project Specification

> "이유는 몰라도 돼요. 데이터는 거짓말하지 않습니다."

---

## 1. Project Overview

### 1.1 Mission
NASDAQ 종목의 단기(1시간) 가격 변동을 예측하여, 데이터 기반 투자 의사결정을 지원하는 시스템

### 1.2 Core Philosophy
- **확률 기반 의사결정**: 이진 분류가 아닌 확률 정보 보존
- **투명한 데이터**: 모든 예측의 이력과 실제 결과 추적
- **모듈화**: 각 컴포넌트 독립 테스트 및 교체 가능

### 1.3 Key Metrics
| 목표 | 기준 |
|------|------|
| 변동성 임계값 | 1시간 내 ±1% |
| 목표 승률 | 66%+ (수수료 고려 시 손익분기) |
| 목표 일 수익률 | 0.8%+ |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Morgorithm Analytics                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
│  │ Collect │──▶│ Process │──▶│  Train  │──▶│ Predict │──▶│Evaluate │ │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘ │
│       │             │             │             │             │       │
│       ▼             ▼             ▼             ▼             ▼       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                      PostgreSQL Database                        │  │
│  │  minute_bars │ features │ labels │ models │ predictions│outcomes│  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│                            ┌─────────────┐                           │
│                            │  Backtest   │                           │
│                            └─────────────┘                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 2-Stage Soft-Gated Pipeline

```
                    Input: 5분봉 데이터 + 57개 피처
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Stage 1: Volatility │
                    │   LightGBM Classifier │
                    └───────────┬───────────┘
                                │
                                ▼
                    p_v = P(|Δprice| ≥ 1% in 1hr)
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Stage 2: Direction  │
                    │   LightGBM Classifier │
                    │   (p_v as feature)    │
                    └───────────┬───────────┘
                                │
                                ▼
                    p_up = P(up | volatile)
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Soft Gating       │
                    │  p_combined = p_v×p_up│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Ranking by EV       │
                    │  EV = p×1% - (1-p)×L  │
                    └───────────────────────┘
```

---

## 3. Module Specifications

### 3.1 Module Interface Contract

모든 모듈은 다음 패턴을 따름:
```python
class BaseModule:
    def run(self, **kwargs) -> ModuleResult:
        """
        Returns:
            ModuleResult:
                - success: bool
                - message: str
                - data: dict (모듈별 상이)
                - errors: List[str]
        """
        pass
```

### 3.2 Collect Module

**책임**: NASDAQ 종목 5분봉 데이터 수집

| 항목 | 내용 |
|------|------|
| 입력 | 없음 (외부 API: yfinance) |
| 출력 | `minute_bars` 테이블 |
| 실행 주기 | 5분마다 (장 중) |
| 데이터 소스 | yfinance (무료) |

**핵심 로직**:
```python
def run(self) -> CollectResult:
    1. 활성 종목 목록 조회 (tickers 테이블)
    2. 각 종목별 최근 5분봉 수집
    3. 중복 체크 후 minute_bars 삽입
    4. 수집 결과 반환
```

**출력 스키마**:
```sql
CREATE TABLE minute_bars (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);
```

---

### 3.3 Process Module

**책임**: 피처 엔지니어링 + 레이블 생성

| 항목 | 내용 |
|------|------|
| 입력 | `minute_bars` 테이블 |
| 출력 | `features`, `labels` 테이블 |
| 실행 주기 | 5분마다 (Collect 직후) |

**피처 카테고리 (57개)**:
| 카테고리 | 개수 | 예시 |
|----------|------|------|
| 가격 기반 | 15 | returns_5m, ma_cross, price_vs_vwap |
| 변동성 | 10 | atr_14, bb_position, volatility_15m |
| 거래량 | 8 | volume_ratio, obv, mfi_14 |
| 모멘텀 | 8 | rsi_14, macd, stoch_k |
| 시장 맥락 | 5 | spy_return, vix_level |
| 시간 기반 | 3 | minutes_since_open, day_of_week |
| 호가창 | 8 | bid_ask_spread, imbalance |

**레이블 정의**:
```python
# label_volatile: 1시간 내 |가격변동| >= 1%
label_volatile = (max_gain >= 0.01) or (max_loss <= -0.01)

# label_direction: 변동 발생 시 상승 여부
label_direction = max_gain >= abs(max_loss)  # if volatile
```

**출력 스키마**:
```sql
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    -- 개별 피처 컬럼 또는 JSONB
    returns_5m DECIMAL(8, 6),
    returns_15m DECIMAL(8, 6),
    rsi_14 DECIMAL(6, 4),
    -- ... 57개 피처
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);

CREATE TABLE labels (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    label_volatile BOOLEAN,
    label_direction BOOLEAN,  -- NULL if not volatile
    actual_return DECIMAL(8, 6),
    max_gain DECIMAL(8, 6),
    max_loss DECIMAL(8, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);
```

---

### 3.4 Train Module

**책임**: 2-Stage 모델 학습 및 버전 관리

| 항목 | 내용 |
|------|------|
| 입력 | `features`, `labels` 테이블 |
| 출력 | `models/` 디렉토리, `model_versions` 테이블 |
| 실행 주기 | 증분: 30분, 전체: 1일 1회 |

**모델 구성**:
```
종목당 2개 모델:
├── volatility_model.pkl  (Stage 1)
└── direction_model.pkl   (Stage 2)

총: NASDAQ 종목 수 × 2
```

**학습 파라미터**:
```python
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
```

**모델 버전 관리**:
```
models/
├── active/           # 현재 사용 중
│   ├── volatility/
│   │   └── {ticker}_v{version}.pkl
│   └── direction/
│       └── {ticker}_v{version}.pkl
├── training/         # 학습 중
└── archive/          # 이전 버전
```

**출력 스키마**:
```sql
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    model_type VARCHAR(20) NOT NULL,  -- 'volatility' or 'direction'
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(20) DEFAULT 'lightgbm',
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    samples_count INT,
    positive_ratio DECIMAL(5, 4),
    val_auc DECIMAL(5, 4),
    val_precision DECIMAL(5, 4),
    val_recall DECIMAL(5, 4),
    file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 3.5 Predict Module

**책임**: 현재 데이터로 예측 생성

| 항목 | 내용 |
|------|------|
| 입력 | `features`, `models/active/` |
| 출력 | `predictions` 테이블 |
| 실행 주기 | 5분마다 |

**예측 흐름**:
```python
def run(self) -> PredictResult:
    1. 최신 features 조회
    2. 각 종목별 active 모델 로드
    3. Stage 1: p_v 예측
    4. Stage 2: p_up 예측 (p_v를 피처로 추가)
    5. Soft Gating: p_combined = p_v × p_up
    6. EV 계산: p_combined × 0.01 - (1 - p_combined) × 0.012
    7. EV 기준 랭킹
    8. predictions 테이블 저장
```

**EV (Expected Value) 계산**:
```python
# 승리 시: +1% - 0.2% 수수료 = +0.8%
# 패배 시: 평균 손실 (시간 만료) ≈ -1% - 0.2% = -1.2%
EV = p_combined * 0.008 - (1 - p_combined) * 0.012
```

**출력 스키마**:
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,  -- timestamp + 1 hour
    model_version_id INT REFERENCES model_versions(id),
    entry_price DECIMAL(12, 4),
    p_volatile DECIMAL(5, 4),
    p_direction DECIMAL(5, 4),
    p_combined DECIMAL(5, 4),
    expected_value DECIMAL(8, 6),
    ev_rank INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);
```

---

### 3.6 Evaluate Module

**책임**: 예측 결과 평가 및 지표 계산

| 항목 | 내용 |
|------|------|
| 입력 | `predictions`, `minute_bars` |
| 출력 | `outcomes`, `model_metrics` 테이블 |
| 실행 주기 | 5분마다 (만료된 예측 평가) |

**평가 로직**:
```python
def update_outcomes(self):
    1. target_time이 지난 predictions 조회 (outcome 없는 것)
    2. 각 예측에 대해:
       - 1시간 후 가격 조회
       - actual_return 계산
       - actual_volatile, actual_direction 판정
       - profit_loss 계산 (수수료 포함)
    3. outcomes 테이블 저장
```

**Data Leak 방지**:
```python
# 중요: 평가는 반드시 target_time 이후에만
assert now >= prediction.target_time
# 사용 데이터: prediction.timestamp ~ prediction.target_time
```

**출력 스키마**:
```sql
CREATE TABLE outcomes (
    id SERIAL PRIMARY KEY,
    prediction_id INT REFERENCES predictions(id) UNIQUE,
    actual_volatile BOOLEAN,
    actual_direction BOOLEAN,
    actual_return DECIMAL(8, 6),
    max_gain DECIMAL(8, 6),
    max_loss DECIMAL(8, 6),
    exit_price DECIMAL(12, 4),
    profit_loss DECIMAL(8, 6),  -- 수수료 포함
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    model_version_id INT REFERENCES model_versions(id),
    window_hours INT,  -- 평가 기간 (예: 50시간)
    total_predictions INT,
    volatile_precision DECIMAL(5, 4),
    volatile_recall DECIMAL(5, 4),
    direction_precision DECIMAL(5, 4),
    direction_accuracy DECIMAL(5, 4),
    combined_accuracy DECIMAL(5, 4),
    total_return DECIMAL(10, 6),
    win_rate DECIMAL(5, 4),
    avg_profit DECIMAL(8, 6),
    avg_loss DECIMAL(8, 6),
    sharpe_ratio DECIMAL(6, 4),
    max_drawdown DECIMAL(6, 4),
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 3.7 Backtest Module

**책임**: 히스토리컬 시뮬레이션

| 항목 | 내용 |
|------|------|
| 입력 | `predictions`, `outcomes` |
| 출력 | `backtest_results` 테이블 |
| 실행 주기 | 요청 시 |

**백테스트 전략**:
```python
@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    min_p_combined: float = 0.5
    top_k: int = 5
    commission: float = 0.002  # 0.2%
    initial_capital: float = 100000.0
```

**시뮬레이션 로직**:
```python
def run(self, config: BacktestConfig) -> BacktestResult:
    1. 기간 내 predictions + outcomes 조회
    2. 각 시점별:
       - EV 상위 K개 선택
       - 가상 매수 (position_size = capital / K)
       - 1시간 후 청산
       - profit_loss 누적
    3. 성과 지표 계산
    4. 결과 저장
```

---

## 4. Database Schema (Complete)

```sql
-- 종목 정보
CREATE TABLE tickers (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5분봉 데이터
CREATE TABLE minute_bars (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);

-- 피처
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    feature_json JSONB,  -- 또는 개별 컬럼
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);

-- 레이블
CREATE TABLE labels (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    label_volatile BOOLEAN,
    label_direction BOOLEAN,
    actual_return DECIMAL(8, 6),
    max_gain DECIMAL(8, 6),
    max_loss DECIMAL(8, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);

-- 모델 버전
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    model_type VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(20) DEFAULT 'lightgbm',
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    samples_count INT,
    metrics JSONB,
    file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 예측
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    model_version_id INT REFERENCES model_versions(id),
    entry_price DECIMAL(12, 4),
    p_volatile DECIMAL(5, 4),
    p_direction DECIMAL(5, 4),
    p_combined DECIMAL(5, 4),
    expected_value DECIMAL(8, 6),
    ev_rank INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, timestamp)
);

-- 실제 결과
CREATE TABLE outcomes (
    id SERIAL PRIMARY KEY,
    prediction_id INT REFERENCES predictions(id) UNIQUE,
    actual_volatile BOOLEAN,
    actual_direction BOOLEAN,
    actual_return DECIMAL(8, 6),
    max_gain DECIMAL(8, 6),
    max_loss DECIMAL(8, 6),
    exit_price DECIMAL(12, 4),
    profit_loss DECIMAL(8, 6),
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 모델 성능 지표
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    model_version_id INT REFERENCES model_versions(id),
    window_hours INT,
    metrics JSONB,
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 백테스트 결과
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    config JSONB,
    start_date DATE,
    end_date DATE,
    total_trades INT,
    win_rate DECIMAL(5, 4),
    total_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(6, 4),
    max_drawdown DECIMAL(6, 4),
    equity_curve JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스
CREATE INDEX idx_minute_bars_ticker_time ON minute_bars(ticker, timestamp DESC);
CREATE INDEX idx_features_ticker_time ON features(ticker, timestamp DESC);
CREATE INDEX idx_predictions_ticker_time ON predictions(ticker, timestamp DESC);
CREATE INDEX idx_predictions_target_time ON predictions(target_time);
CREATE INDEX idx_outcomes_prediction ON outcomes(prediction_id);
```

---

## 5. Execution Schedule

### 5.1 Market Hours (ET)
- 개장: 09:30 ET (한국 23:30)
- 마감: 16:00 ET (한국 07:00)

### 5.2 cron Schedule

```bash
# 장 중 5분마다: 수집 → 처리 → 예측 → 평가
*/5 9-15 * * 1-5 /app/run_pipeline.sh

# 장 중 30분마다: 증분 학습
0,30 10-15 * * 1-5 /app/run_incremental_train.sh

# 장 마감 후: 전체 학습
0 17 * * 1-5 /app/run_full_train.sh
```

### 5.3 Pipeline Execution Order

```
:00 수집 시작
:01 수집 완료, 처리 시작
:02 피처/레이블 생성 완료, 예측 시작
:03 예측 완료, 평가 시작
:04 평가 완료
:05 다음 주기
```

---

## 6. Configuration

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Data Collection
    COLLECT_INTERVAL_MINUTES: int = 5
    YFINANCE_TIMEOUT: int = 30

    # Feature Engineering
    FEATURE_LOOKBACK_BARS: int = 60  # 5시간

    # Model Training
    TRAIN_WINDOW_DAYS: int = 30
    VALIDATION_RATIO: float = 0.2
    MIN_SAMPLES: int = 1000

    # Prediction
    PREDICTION_HORIZON_MINUTES: int = 60
    VOLATILITY_THRESHOLD: float = 0.01  # 1%

    # Trading
    COMMISSION_RATE: float = 0.002  # 0.2%

    # Market Hours (ET)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    TIMEZONE: str = "America/New_York"

    class Config:
        env_file = ".env"
```

---

## 7. Error Handling

### 7.1 Module Error Handling

```python
class ModuleResult:
    success: bool
    message: str
    data: dict
    errors: List[str]
    warnings: List[str]

# 각 모듈은 예외를 삼키고 ModuleResult로 반환
def run(self) -> ModuleResult:
    try:
        # ... 로직
        return ModuleResult(success=True, data={...})
    except Exception as e:
        logger.error(f"Module failed: {e}")
        return ModuleResult(success=False, errors=[str(e)])
```

### 7.2 데이터 수집 실패 처리

```python
# yfinance 실패 시 재시도
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

for attempt in range(MAX_RETRIES):
    try:
        data = yf.download(ticker, ...)
        break
    except Exception:
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
        else:
            log_error(ticker, "collection_failed")
```

---

## 8. Testing Requirements

### 8.1 Unit Tests (필수)

| 모듈 | 테스트 항목 |
|------|------------|
| Collect | API 호출 성공, 데이터 파싱, 중복 방지 |
| Process | 각 피처 계산 정확성, 레이블 정확성 |
| Train | 모델 저장/로드, 버전 관리 |
| Predict | 예측값 범위 [0,1], EV 계산 |
| Evaluate | Outcome 정확성, 지표 계산 |
| Backtest | 수익률 계산, 수수료 반영 |

### 8.2 Data Leak Tests (필수)

```python
def test_feature_no_future_data():
    """피처 계산 시 미래 데이터 사용 금지"""
    pass

def test_label_uses_future_only():
    """레이블은 미래 데이터로만 생성"""
    pass

def test_prediction_no_label_leak():
    """예측 시점에 레이블 접근 불가"""
    pass
```

### 8.3 Integration Tests

```python
def test_full_pipeline():
    """Collect → Process → Train → Predict → Evaluate 전체 흐름"""
    pass

def test_incremental_training():
    """증분 학습 후 모델 교체"""
    pass
```

---

## 9. File Structure

```
morgorithm-analytics/
├── spec.md                    # 본 문서
├── todo.md                    # 개발 로드맵
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── morgorithm/
│   ├── __init__.py
│   ├── collect/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   └── yfinance_client.py
│   ├── process/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   ├── features.py
│   │   └── labels.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   ├── volatility.py
│   │   ├── direction.py
│   │   └── registry.py
│   ├── predict/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   └── soft_gate.py
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   └── metrics.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   └── simulator.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── session.py
│   └── common/
│       ├── __init__.py
│       ├── logger.py
│       └── types.py
│
├── models/                    # 학습된 모델
│   ├── active/
│   ├── training/
│   └── archive/
│
├── tests/
│   ├── conftest.py
│   ├── test_collect.py
│   ├── test_process.py
│   ├── test_train.py
│   ├── test_predict.py
│   ├── test_evaluate.py
│   ├── test_backtest.py
│   └── test_data_leak.py
│
└── scripts/
    ├── init_db.py
    ├── run_collect.py
    ├── run_pipeline.py
    └── seed_tickers.py
```

---

## 10. Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"
# Data
yfinance = "^0.2.0"
pandas = "^2.0"
numpy = "^1.24"
ta = "^0.10"  # Technical Analysis

# ML
lightgbm = "^4.0"
scikit-learn = "^1.3"

# Database
sqlalchemy = "^2.0"
psycopg2-binary = "^2.9"
alembic = "^1.12"

# API (Phase 6+)
# fastapi = "^0.100"
# uvicorn = "^0.23"

# Utils
pydantic-settings = "^2.0"
loguru = "^0.7"
pytz = "^2024"
python-dotenv = "^1.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4"
pytest-cov = "^4.1"
black = "^23.0"
ruff = "^0.1"
```

---

**Version**: 1.0
**Last Updated**: 2026-01-08
