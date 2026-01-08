# Morgorithm Analytics - Migration Plan

> "이유는 몰라도 돼요. 데이터는 거짓말하지 않습니다."

**Version**: 1.0
**Date**: 2026-01-08
**Status**: Planning Phase

---

## 1. Executive Summary

### 1.1 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Morgorithm Analytics (모르고리즘 애널리틱스) |
| **목적** | 2-Stage Soft-Gated Trading Pipeline 기반 NASDAQ 단기 예측 시스템 |
| **마이그레이션 원본** | FiveForFree |
| **핵심 변경** | 모듈화 재설계, 2-Stage 모델, 구독 서비스 구조 |

### 1.2 핵심 설계 원칙

1. **인터페이스 계약 방식** - 모듈 간 DB 테이블로 통신, 독립 테스트 가능
2. **확률 정보 보존** - Hard decision 최소화, Soft Gating
3. **데이터 투명성** - 모든 예측의 이력과 결과를 추적 가능하게
4. **단순한 실행 구조** - cron + Python 스크립트

### 1.3 비즈니스 모델

| 티어 | 가격 | 기능 |
|------|------|------|
| **Free** | $0 | 과거 데이터, 백테스팅 결과, 모델 성능 이력 |
| **Pro** | $800/월, $5000/년 | 실시간 예측, 최신 데이터 접근 |

---

## 2. Architecture Overview

### 2.1 전체 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Morgorithm Analytics                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │   Collect   │───▶│  Process    │───▶│    Train    │───▶│   Predict   │  │
│   │   Module    │    │   Module    │    │   Module    │    │   Module    │  │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│          │                  │                  │                  │          │
│          ▼                  ▼                  ▼                  ▼          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        PostgreSQL Database                           │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐│   │
│   │  │ tickers │ │minute_  │ │features │ │ models  │ │   predictions   ││   │
│   │  │         │ │  bars   │ │         │ │         │ │   + outcomes    ││   │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘│   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │  Evaluate   │───▶│  Backtest   │───▶│     API     │                     │
│   │   Module    │    │   Module    │    │   Module    │                     │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                     │
│                                                 │                            │
└─────────────────────────────────────────────────┼────────────────────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────┐
                                         │  Frontend   │
                                         │  (React)    │
                                         └─────────────┘
```

### 2.2 2-Stage Soft-Gated Pipeline

```
Raw Market Data (5분봉)
        │
        ▼
┌───────────────────┐
│ Feature Engineer  │  57개 피처 계산
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 1:         │  p_v = P(|Δ| ≥ 1%)
│  Volatility Model │  "1시간 내 1% 이상 변동할 확률"
└────────┬──────────┘
         │ p_v
         ▼
┌───────────────────┐
│  Stage 2:         │  p_up = P(up | volatile)
│  Direction Model  │  "변동 시 상승 방향일 확률"
│  (p_v as feature) │
└────────┬──────────┘
         │ p_up
         ▼
┌───────────────────┐
│   Soft Gating     │  p_combined = p_v × p_up
│                   │  "무조건적 상승 확률"
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Ranking by EV    │  EV = p × 1% - (1-p) × loss
│  (Expected Value) │
└────────┬──────────┘
         │
         ▼
    User Decision
```

---

## 3. Module Specifications

### 3.1 모듈 개요

| 모듈 | 입력 (테이블/파일) | 출력 (테이블/파일) | 실행 주기 |
|------|-------------------|-------------------|----------|
| **Collect** | - | `minute_bars` | 5분마다 (장 중) |
| **Process** | `minute_bars` | `features` | 5분마다 |
| **Train** | `features`, `labels` | `models/*.pkl` | 30분 증분, 1일 전체 |
| **Predict** | `features`, `models` | `predictions` | 5분마다 |
| **Evaluate** | `predictions`, `minute_bars` | `outcomes`, `model_metrics` | 5분마다 |
| **Backtest** | `predictions`, `outcomes` | `backtest_results` | 요청 시 |
| **API** | 모든 테이블 | JSON Response | 상시 |

### 3.2 Collect Module (데이터 수집)

```python
"""
모듈: collect
입력: 없음 (외부 API)
출력: minute_bars 테이블

책임:
- NASDAQ 전체 종목 5분봉 수집
- yfinance API 호출
- 데이터 정합성 검증
"""

# 인터페이스 계약
class CollectModule:
    def run(self) -> CollectResult:
        """
        Returns:
            CollectResult:
                - success: bool
                - tickers_collected: int
                - bars_inserted: int
                - errors: List[str]
        """
        pass

# 출력 테이블 스키마
minute_bars:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - timestamp: TIMESTAMP WITH TIME ZONE
    - open: DECIMAL(10, 4)
    - high: DECIMAL(10, 4)
    - low: DECIMAL(10, 4)
    - close: DECIMAL(10, 4)
    - volume: BIGINT
    - vwap: DECIMAL(10, 4)
    - created_at: TIMESTAMP DEFAULT NOW()

    UNIQUE(ticker, timestamp)
```

### 3.3 Process Module (피처 엔지니어링)

```python
"""
모듈: process
입력: minute_bars 테이블
출력: features 테이블, labels 테이블

책임:
- 57개 피처 계산
- 레이블 생성 (label_volatile, label_direction)
- 결측치 처리
"""

# 인터페이스 계약
class ProcessModule:
    def run(self, since: datetime = None) -> ProcessResult:
        """
        Args:
            since: 이 시점 이후 데이터만 처리 (증분 처리용)

        Returns:
            ProcessResult:
                - success: bool
                - rows_processed: int
                - features_generated: int
                - labels_generated: int
        """
        pass

# 출력 테이블 스키마
features:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - timestamp: TIMESTAMP WITH TIME ZONE
    - feature_vector: JSONB  # 또는 개별 컬럼
    - created_at: TIMESTAMP DEFAULT NOW()

labels:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - timestamp: TIMESTAMP WITH TIME ZONE
    - label_volatile: BOOLEAN  # |Δ| ≥ 1% in 1 hour
    - label_direction: BOOLEAN  # up=True if volatile
    - actual_return: DECIMAL(6, 4)  # 실제 수익률
    - max_gain: DECIMAL(6, 4)
    - max_loss: DECIMAL(6, 4)
    - created_at: TIMESTAMP DEFAULT NOW()
```

### 3.4 Train Module (모델 학습)

```python
"""
모듈: train
입력: features 테이블, labels 테이블
출력: models/ 디렉토리 (pkl 파일)

책임:
- Stage 1 (Volatility) 모델 학습
- Stage 2 (Direction) 모델 학습
- 모델 버전 관리 (active/training)
- 증분 학습 및 전체 학습
"""

# 인터페이스 계약
class TrainModule:
    def run_incremental(self, hours: int = 2) -> TrainResult:
        """최근 N시간 데이터로 증분 학습"""
        pass

    def run_full(self, days: int = 30) -> TrainResult:
        """전체 데이터로 완전 재학습"""
        pass

# 모델 파일 구조
models/
├── active/                    # 현재 예측에 사용
│   ├── volatility/
│   │   ├── AAPL_v1.2.pkl
│   │   └── ...
│   └── direction/
│       ├── AAPL_v1.2.pkl
│       └── ...
├── training/                  # 학습 중 (완료 시 active로 교체)
│   └── ...
└── archive/                   # 이전 버전 보관
    └── ...

# 모델 메타데이터 테이블
model_versions:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - model_type: VARCHAR(20)  # 'volatility' or 'direction'
    - version: VARCHAR(20)
    - algorithm: VARCHAR(20)   # 'lightgbm'
    - train_start: TIMESTAMP
    - train_end: TIMESTAMP
    - samples_count: INT
    - metrics: JSONB           # PR-AUC, Precision, Recall 등
    - file_path: VARCHAR(255)
    - is_active: BOOLEAN
    - created_at: TIMESTAMP DEFAULT NOW()
```

### 3.5 Predict Module (예측)

```python
"""
모듈: predict
입력: features 테이블, models/ 디렉토리
출력: predictions 테이블

책임:
- 현재 active 모델로 예측 수행
- p_v, p_up, p_combined 계산
- EV 계산 및 랭킹
"""

# 인터페이스 계약
class PredictModule:
    def run(self) -> PredictResult:
        """
        Returns:
            PredictResult:
                - success: bool
                - predictions_generated: int
                - top_predictions: List[Prediction]
        """
        pass

# 출력 테이블 스키마
predictions:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - timestamp: TIMESTAMP WITH TIME ZONE  # 예측 시점
    - target_timestamp: TIMESTAMP          # 목표 시점 (1시간 후)
    - model_version_id: INT REFERENCES model_versions(id)
    - p_volatile: DECIMAL(5, 4)            # Stage 1 출력
    - p_direction: DECIMAL(5, 4)           # Stage 2 출력
    - p_combined: DECIMAL(5, 4)            # p_v × p_up
    - expected_value: DECIMAL(6, 4)        # EV
    - rank: INT                            # EV 기준 순위
    - entry_price: DECIMAL(10, 4)          # 예측 시점 가격
    - created_at: TIMESTAMP DEFAULT NOW()
```

### 3.6 Evaluate Module (평가)

```python
"""
모듈: evaluate
입력: predictions 테이블, minute_bars 테이블
출력: outcomes 테이블, model_metrics 테이블

책임:
- 1시간 후 실제 결과 업데이트
- 모델별 성능 지표 계산
- Data Leak 없이 평가
"""

# 인터페이스 계약
class EvaluateModule:
    def update_outcomes(self) -> EvaluateResult:
        """만료된 예측의 실제 결과 업데이트"""
        pass

    def calculate_metrics(self, ticker: str, window_hours: int = 50) -> MetricsResult:
        """최근 N시간 성능 지표 계산"""
        pass

# 출력 테이블 스키마
outcomes:
    - prediction_id: INT REFERENCES predictions(id) PRIMARY KEY
    - actual_volatile: BOOLEAN
    - actual_direction: BOOLEAN
    - actual_return: DECIMAL(6, 4)
    - max_gain: DECIMAL(6, 4)
    - max_loss: DECIMAL(6, 4)
    - exit_price: DECIMAL(10, 4)
    - exit_reason: VARCHAR(20)  # 'time_limit'
    - profit_loss: DECIMAL(6, 4)  # 수수료 포함
    - evaluated_at: TIMESTAMP DEFAULT NOW()

model_metrics:
    - id: SERIAL PRIMARY KEY
    - ticker: VARCHAR(10)
    - model_version_id: INT
    - window_start: TIMESTAMP
    - window_end: TIMESTAMP
    - total_predictions: INT
    - volatile_precision: DECIMAL(5, 4)
    - volatile_recall: DECIMAL(5, 4)
    - volatile_pr_auc: DECIMAL(5, 4)
    - direction_precision: DECIMAL(5, 4)
    - direction_recall: DECIMAL(5, 4)
    - combined_accuracy: DECIMAL(5, 4)
    - total_return: DECIMAL(8, 4)        # 누적 수익률
    - win_rate: DECIMAL(5, 4)
    - avg_profit: DECIMAL(6, 4)
    - avg_loss: DECIMAL(6, 4)
    - calculated_at: TIMESTAMP DEFAULT NOW()
```

### 3.7 Backtest Module (백테스팅)

```python
"""
모듈: backtest
입력: predictions 테이블, outcomes 테이블
출력: backtest_results 테이블

책임:
- 히스토리컬 시뮬레이션
- 다양한 전략 비교
- 수익률 곡선 생성
"""

# 인터페이스 계약
class BacktestModule:
    def run(
        self,
        start_date: date,
        end_date: date,
        strategy: BacktestStrategy
    ) -> BacktestResult:
        pass

# 백테스트 전략 설정
@dataclass
class BacktestStrategy:
    min_p_combined: float = 0.5      # 최소 확률 임계값
    top_k: int = 5                   # 상위 K개만 거래
    position_size: float = 0.1       # 종목당 비중
    commission: float = 0.002        # 왕복 수수료 0.2%
```

### 3.8 API Module

```python
"""
모듈: api
입력: 모든 테이블
출력: REST API / WebSocket

책임:
- 예측 데이터 제공
- 모델 성능 지표 제공
- 사용자 인증 (Pro tier)
"""

# 주요 엔드포인트
GET  /api/predictions/latest          # 최신 예측 (EV 순)
GET  /api/predictions/{ticker}        # 종목별 예측
GET  /api/models/{ticker}/metrics     # 모델 성능 지표
GET  /api/models/{ticker}/history     # 과거 예측 + 실제 결과
GET  /api/backtest/results            # 백테스트 결과
GET  /api/health                      # 시스템 상태
WS   /ws/predictions                  # 실시간 예측 스트림
```

---

## 4. Database Schema

### 4.1 ERD (Entity Relationship Diagram)

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   tickers   │       │ minute_bars │       │  features   │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ symbol (PK) │──┐    │ id (PK)     │       │ id (PK)     │
│ name        │  │    │ ticker (FK) │◀──────│ ticker (FK) │
│ sector      │  │    │ timestamp   │       │ timestamp   │
│ market_cap  │  └───▶│ open        │       │ feature_vec │
│ is_active   │       │ high        │       └─────────────┘
└─────────────┘       │ low         │              │
                      │ close       │              │
                      │ volume      │              ▼
                      │ vwap        │       ┌─────────────┐
                      └─────────────┘       │   labels    │
                                            ├─────────────┤
┌─────────────────┐                         │ id (PK)     │
│ model_versions  │                         │ ticker (FK) │
├─────────────────┤                         │ timestamp   │
│ id (PK)         │◀────────────┐           │ label_vol   │
│ ticker (FK)     │             │           │ label_dir   │
│ model_type      │             │           │ actual_ret  │
│ version         │             │           └─────────────┘
│ algorithm       │             │
│ train_start     │             │
│ train_end       │             │
│ metrics (JSONB) │             │
│ is_active       │             │
└─────────────────┘             │
                                │
┌─────────────────┐             │       ┌─────────────────┐
│  predictions    │             │       │    outcomes     │
├─────────────────┤             │       ├─────────────────┤
│ id (PK)         │─────────────┼──────▶│ prediction_id   │
│ ticker (FK)     │             │       │ actual_volatile │
│ timestamp       │             │       │ actual_direction│
│ model_ver_id(FK)│─────────────┘       │ actual_return   │
│ p_volatile      │                     │ profit_loss     │
│ p_direction     │                     │ evaluated_at    │
│ p_combined      │                     └─────────────────┘
│ expected_value  │
│ rank            │
│ entry_price     │
└─────────────────┘
```

### 4.2 인덱스 전략

```sql
-- 시계열 쿼리 최적화
CREATE INDEX idx_minute_bars_ticker_time ON minute_bars(ticker, timestamp DESC);
CREATE INDEX idx_features_ticker_time ON features(ticker, timestamp DESC);
CREATE INDEX idx_predictions_ticker_time ON predictions(ticker, timestamp DESC);

-- 랭킹 쿼리 최적화
CREATE INDEX idx_predictions_rank ON predictions(timestamp, rank);
CREATE INDEX idx_predictions_ev ON predictions(timestamp, expected_value DESC);

-- 평가 쿼리 최적화
CREATE INDEX idx_outcomes_evaluated ON outcomes(evaluated_at);
```

---

## 5. Execution Schedule

### 5.1 cron 스케줄

```bash
# 장 중 (한국시간 23:30 - 06:00, ET 09:30 - 16:00)
# 미국 동부시간 기준 cron

# 5분마다: 데이터 수집 → 피처 계산 → 예측 → 평가
*/5 9-15 * * 1-5  cd /app && python -m morgorithm.collect && python -m morgorithm.process && python -m morgorithm.predict && python -m morgorithm.evaluate

# 30분마다: 증분 학습 (백그라운드)
0,30 10-15 * * 1-5  cd /app && python -m morgorithm.train --incremental &

# 장 마감 후: 전체 학습 (17:00 ET)
0 17 * * 1-5  cd /app && python -m morgorithm.train --full
```

### 5.2 5분 주기 타임라인 (상세)

```
:00 ── 이전 5분봉 데이터 수집 완료 (yfinance)
:01 ── 피처 계산 시작
:02 ── 피처 계산 완료, 예측 시작
:03 ── 예측 완료, DB 저장
:04 ── 1시간 전 예측 outcome 업데이트
:05 ── 다음 주기 시작
```

### 5.3 모델 버전 관리 흐름

```
[증분 학습 - 30분마다]
1. training/ 디렉토리에 새 모델 학습
2. 학습 완료 후 validation 체크
3. 기존 active 모델과 성능 비교
4. 개선되었으면:
   - active → archive 이동
   - training → active 이동 (atomic swap)
5. 개선 안 되었으면:
   - training 삭제

[전체 학습 - 장 마감 후]
1. 30일 전체 데이터로 처음부터 학습
2. 충분한 validation 수행
3. 모든 종목 모델 교체
```

---

## 6. Development Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] 프로젝트 구조 설정
- [ ] PostgreSQL 스키마 생성
- [ ] 기본 모듈 인터페이스 정의
- [ ] 설정 관리 (Pydantic Settings)
- [ ] 로깅 설정

### Phase 2: Data Pipeline (Week 3-4)

- [ ] **Collect Module**
  - [ ] yfinance wrapper 구현
  - [ ] NASDAQ 전체 종목 리스트 관리
  - [ ] 5분봉 수집 및 저장
  - [ ] 에러 핸들링 및 재시도 로직
  - [ ] 단위 테스트

- [ ] **Process Module**
  - [ ] 피처 엔지니어링 (기존 코드 참조)
  - [ ] 레이블 생성 (label_volatile, label_direction)
  - [ ] 결측치 처리
  - [ ] 단위 테스트

### Phase 3: Model Training (Week 5-6)

- [ ] **Train Module**
  - [ ] LightGBM 모델 구현
  - [ ] 2-Stage 학습 파이프라인
  - [ ] 모델 버전 관리
  - [ ] 증분 학습 구현
  - [ ] 전체 학습 구현
  - [ ] 단위 테스트

### Phase 4: Prediction & Evaluation (Week 7-8)

- [ ] **Predict Module**
  - [ ] 모델 로딩 및 예측
  - [ ] Soft Gating (p_combined 계산)
  - [ ] EV 계산 및 랭킹
  - [ ] 단위 테스트

- [ ] **Evaluate Module**
  - [ ] Outcome 업데이트 로직
  - [ ] 모델 성능 지표 계산
  - [ ] Data Leak 방지 검증
  - [ ] 단위 테스트

### Phase 5: Backtesting (Week 9)

- [ ] **Backtest Module**
  - [ ] 히스토리컬 시뮬레이션
  - [ ] 수익률 계산
  - [ ] 다양한 전략 지원
  - [ ] 단위 테스트

### Phase 6: API & Integration (Week 10-11)

- [ ] **API Module**
  - [ ] FastAPI 설정
  - [ ] 엔드포인트 구현
  - [ ] 인증 (Free/Pro tier)
  - [ ] WebSocket (선택)
  - [ ] 통합 테스트

### Phase 7: Deployment & Monitoring (Week 12)

- [ ] 배포 설정 (DigitalOcean/Railway)
- [ ] cron 스케줄 설정
- [ ] 모니터링 대시보드
- [ ] 알림 설정

### Phase 8: Frontend (Week 13+)

- [ ] 모든 백엔드 모듈 100% 검증 후 시작
- [ ] React 프로젝트 설정
- [ ] 예측 대시보드
- [ ] 모델 성능 시각화
- [ ] 백테스트 결과 시각화

---

## 7. Testing Strategy

### 7.1 테스트 원칙

```
각 모듈은 "입력 테이블 → 출력 테이블" 계약만 검증

Given: 입력 테이블에 테스트 데이터 삽입
When: 모듈.run() 실행
Then: 출력 테이블 검증
```

### 7.2 Data Leak 방지 테스트

```python
def test_no_data_leak_in_features():
    """피처 계산 시 미래 데이터 사용 금지"""
    # Given: timestamp T의 minute_bars
    # When: T 시점 피처 계산
    # Then: T 이후 데이터는 사용되지 않음
    pass

def test_no_data_leak_in_labels():
    """레이블 생성 시 예측 시점 이후 데이터만 사용"""
    # 레이블은 "1시간 후" 결과이므로,
    # T+60분까지의 데이터만 사용해야 함
    pass

def test_no_data_leak_in_evaluation():
    """평가 시 예측 시점 이후 데이터만 사용"""
    # 예측 시점 T의 outcome은
    # T+60분의 데이터로만 계산
    pass
```

### 7.3 모듈별 테스트 체크리스트

| 모듈 | 테스트 항목 |
|------|------------|
| Collect | API 호출 성공, 데이터 정합성, 중복 방지 |
| Process | 피처 계산 정확성, 결측치 처리, 레이블 정확성 |
| Train | 모델 저장/로드, 증분학습 동작, 버전 관리 |
| Predict | 예측값 범위 [0,1], EV 계산, 랭킹 정확성 |
| Evaluate | Outcome 정확성, 지표 계산, Data Leak 없음 |
| Backtest | 수익률 계산, 전략 적용, 수수료 반영 |

---

## 8. Project Structure

```
morgorithm-analytics/
├── README.md
├── MIGRATION_PLAN.md          # 본 문서
├── pyproject.toml             # 의존성 관리 (Poetry)
├── .env.example
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Pydantic Settings
│
├── morgorithm/                # 메인 패키지
│   ├── __init__.py
│   │
│   ├── collect/               # 데이터 수집 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # CollectModule
│   │   └── yfinance_client.py # yfinance 래퍼
│   │
│   ├── process/               # 전처리 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # ProcessModule
│   │   ├── feature_engineer.py
│   │   └── label_generator.py
│   │
│   ├── train/                 # 학습 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # TrainModule
│   │   ├── volatility_model.py
│   │   ├── direction_model.py
│   │   └── model_registry.py  # 버전 관리
│   │
│   ├── predict/               # 예측 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # PredictModule
│   │   └── soft_gating.py
│   │
│   ├── evaluate/              # 평가 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # EvaluateModule
│   │   └── metrics.py
│   │
│   ├── backtest/              # 백테스팅 모듈
│   │   ├── __init__.py
│   │   ├── module.py          # BacktestModule
│   │   └── strategies.py
│   │
│   ├── api/                   # API 모듈
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/
│   │   │   ├── predictions.py
│   │   │   ├── models.py
│   │   │   └── backtest.py
│   │   └── auth.py
│   │
│   ├── db/                    # 데이터베이스
│   │   ├── __init__.py
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── session.py
│   │   └── migrations/        # Alembic
│   │
│   └── common/                # 공통 유틸
│       ├── __init__.py
│       ├── logger.py
│       └── exceptions.py
│
├── models/                    # 학습된 모델 파일
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
│   └── test_data_leak.py      # Data Leak 전용 테스트
│
├── scripts/
│   ├── init_db.py
│   ├── run_collect.py
│   └── run_pipeline.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── frontend/                  # React (Phase 8)
    └── ...
```

---

## 9. Configuration

### 9.1 환경 변수

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/morgorithm"

    # Data Collection
    YFINANCE_RATE_LIMIT: int = 2000  # requests per hour
    COLLECT_INTERVAL_MINUTES: int = 5

    # Trading Parameters
    PREDICTION_HORIZON_MINUTES: int = 60
    VOLATILITY_THRESHOLD: float = 0.01  # 1%
    COMMISSION_RATE: float = 0.002  # 0.2% round trip

    # Model Training
    TRAIN_WINDOW_DAYS: int = 30
    INCREMENTAL_WINDOW_HOURS: int = 2
    MODEL_ALGORITHM: str = "lightgbm"

    # Market Hours (ET)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16

    class Config:
        env_file = ".env"
```

---

## 10. Deployment

### 10.1 추천 인프라

| 구성요소 | 서비스 | 비용 (월) |
|---------|--------|----------|
| API + 수집 서버 | DigitalOcean Droplet 4GB | $24 |
| PostgreSQL | DigitalOcean Managed DB | $15 |
| 모델 저장소 | Cloudflare R2 | ~$5 |
| **총합** | | **~$44** |

### 10.2 학습 서버 분리

```
[로컬 PC - RTX 5080]
- 전체 학습 (장 마감 후)
- 모델 파일 → R2 업로드

[클라우드 서버]
- 데이터 수집
- 예측 (모델 다운로드)
- API 서빙
- 증분 학습 (경량)
```

---

## 11. Risk & Mitigation

| 리스크 | 영향 | 대응 |
|--------|------|------|
| yfinance API 불안정 | 데이터 수집 실패 | 재시도 로직, 알림 |
| 모델 성능 저하 | 손실 발생 | 자동 모니터링, 거래 중단 |
| Data Leak | 과적합, 실전 손실 | 전용 테스트, 코드 리뷰 |
| DB 장애 | 서비스 중단 | 자동 백업, 복구 절차 |

---

## 12. Success Criteria

### 12.1 기술적 성공 기준

- [ ] 모든 모듈 독립 테스트 통과
- [ ] Data Leak 테스트 100% 통과
- [ ] 5분 주기 파이프라인 안정적 운영 (99% 가동률)
- [ ] 예측 생성 지연 < 1분

### 12.2 비즈니스 성공 기준

- [ ] 100개+ 종목 중 상위 5% 모델의 승률 > 60%
- [ ] 상위 5개 모델 기준 일 평균 수익률 > 0.5%
- [ ] 1개월 Paper Trading 양성 수익

---

## Appendix A: Feature List (57개)

기존 FiveForFree의 피처 목록 참조:
- `/home/user/FiveForFree/docs/FEATURES_REFERENCE.md`

### Stage 2 추가 피처

| 피처 | 설명 |
|------|------|
| `p_volatile` | Stage 1 출력 (변동성 확률) |

---

## Appendix B: Decision Log

| 날짜 | 결정 사항 | 이유 |
|------|----------|------|
| 2026-01-08 | 프로젝트명: Morgorithm Analytics | 브랜딩, IR 설명 용이 |
| 2026-01-08 | 모듈화: 인터페이스 계약 방식 | 테스트 용이성 |
| 2026-01-08 | DB: PostgreSQL | 단일 DB로 사용자+시계열 관리 |
| 2026-01-08 | 모델: LightGBM only | 빠른 학습, 증분 학습 지원 |
| 2026-01-08 | 데이터: yfinance | 무료, 충분한 기능 |
| 2026-01-08 | 손절: 없음 (시간 만료만) | 백테스트 결과 기반 |
| 2026-01-08 | 종목: NASDAQ 전체 | 확장성 |

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-08 | Initial migration plan |
