# Morgorithm Analytics - Development Roadmap

> **기간**: 2026-01-12 ~ 2026-01-25 (2주)
> **목표**: Phase 1-5 완료 (Core → Backtest)
> **방식**: Full-time AI 협업 개발

---

## Overview

```
Week 1 (1/12-1/18): Infrastructure + Data Pipeline + Training 시작
Week 2 (1/19-1/25): Training 완료 + Prediction + Evaluation + Backtest
```

| Phase | 기간 | 일수 | 내용 |
|-------|------|------|------|
| 1 | 1/12-1/13 | 2 | Core Infrastructure |
| 2 | 1/14-1/17 | 4 | Collect + Process |
| 3 | 1/18-1/20 | 3 | Train (2-Stage) |
| 4 | 1/21-1/23 | 3 | Predict + Evaluate |
| 5 | 1/24-1/25 | 2 | Backtest + 검증 |

---

## Phase 1: Core Infrastructure (Day 1-2)

### Day 1 (1/12 일) - 프로젝트 설정

#### 1.1 프로젝트 초기화
```bash
# 작업 항목
- [ ] Poetry 프로젝트 생성 (pyproject.toml)
- [ ] 디렉토리 구조 생성
- [ ] .gitignore, .env.example 생성
- [ ] Git 저장소 초기화
```

**AI 작업 지시**:
```
morgorithm-analytics/ 디렉토리에 Poetry 프로젝트를 생성하고,
spec.md의 File Structure에 맞게 디렉토리와 __init__.py 파일들을 생성해줘.
```

#### 1.2 설정 관리
```bash
# 작업 항목
- [ ] config/settings.py 구현 (Pydantic Settings)
- [ ] .env.example 작성
- [ ] 로깅 설정 (loguru)
```

**AI 작업 지시**:
```
config/settings.py에 Pydantic Settings 클래스를 구현해줘.
spec.md의 Section 6 Configuration을 참조해서 모든 설정 항목을 포함해야 해.
```

#### 1.3 공통 유틸리티
```bash
# 작업 항목
- [ ] morgorithm/common/logger.py
- [ ] morgorithm/common/types.py (ModuleResult 등)
- [ ] morgorithm/common/exceptions.py
```

**AI 작업 지시**:
```
morgorithm/common/ 디렉토리에 공통 유틸리티를 구현해줘:
1. logger.py - loguru 기반 로거 설정
2. types.py - ModuleResult, 공통 타입 정의
3. exceptions.py - 커스텀 예외 클래스
```

---

### Day 2 (1/13 월) - 데이터베이스 설정

#### 2.1 SQLAlchemy 모델
```bash
# 작업 항목
- [ ] morgorithm/db/models.py (모든 테이블)
- [ ] morgorithm/db/session.py (세션 관리)
```

**AI 작업 지시**:
```
morgorithm/db/models.py에 SQLAlchemy 2.0 스타일로 모든 테이블 모델을 정의해줘.
spec.md Section 4의 스키마를 정확히 반영해야 해.
테이블: tickers, minute_bars, features, labels, model_versions,
predictions, outcomes, model_metrics, backtest_results
```

#### 2.2 데이터베이스 마이그레이션
```bash
# 작업 항목
- [ ] Alembic 설정
- [ ] 초기 마이그레이션 생성
- [ ] scripts/init_db.py 작성
```

**AI 작업 지시**:
```
Alembic을 설정하고 초기 마이그레이션을 생성해줘.
scripts/init_db.py는 DB 초기화 + 마이그레이션 실행 스크립트야.
```

#### 2.3 PostgreSQL 테스트
```bash
# 작업 항목
- [ ] Docker Compose로 PostgreSQL 실행
- [ ] 연결 테스트
- [ ] 테이블 생성 확인
```

**AI 작업 지시**:
```
docker-compose.yml에 PostgreSQL 서비스를 추가하고,
테이블이 정상 생성되는지 테스트하는 스크립트를 작성해줘.
```

#### Day 2 완료 기준
- [ ] `python scripts/init_db.py` 실행 시 모든 테이블 생성
- [ ] DB 연결 테스트 통과

---

## Phase 2: Data Pipeline (Day 3-6)

### Day 3 (1/14 화) - Collect Module 기본

#### 3.1 yfinance 클라이언트
```bash
# 작업 항목
- [ ] morgorithm/collect/yfinance_client.py
    - [ ] 5분봉 데이터 수집 함수
    - [ ] 에러 핸들링 + 재시도 로직
    - [ ] Rate limit 관리
```

**AI 작업 지시**:
```
morgorithm/collect/yfinance_client.py를 구현해줘.
주요 함수:
1. get_minute_bars(ticker, period="1d") -> DataFrame
2. get_multiple_tickers_bars(tickers, period="1d") -> Dict[str, DataFrame]

에러 발생 시 3회 재시도, 5초 간격.
```

#### 3.2 Collect Module
```bash
# 작업 항목
- [ ] morgorithm/collect/module.py
    - [ ] CollectModule 클래스
    - [ ] run() 메서드 구현
    - [ ] DB 저장 로직
```

**AI 작업 지시**:
```
morgorithm/collect/module.py에 CollectModule을 구현해줘.
spec.md Section 3.2의 인터페이스 계약을 따라야 해.

run() 흐름:
1. tickers 테이블에서 is_active=True인 종목 조회
2. 각 종목별 최근 5분봉 수집
3. minute_bars 테이블에 저장 (중복 무시)
4. CollectResult 반환
```

---

### Day 4 (1/15 수) - Collect Module 완성 + 테스트

#### 4.1 종목 시딩
```bash
# 작업 항목
- [ ] scripts/seed_tickers.py
    - [ ] NASDAQ 종목 리스트 가져오기
    - [ ] tickers 테이블 초기화
```

**AI 작업 지시**:
```
scripts/seed_tickers.py를 작성해줘.
yfinance 또는 다른 소스에서 NASDAQ 종목 리스트를 가져와서 tickers 테이블에 삽입.
우선 테스트용으로 Top 100 거래량 종목만.
```

#### 4.2 Collect 테스트
```bash
# 작업 항목
- [ ] tests/test_collect.py
    - [ ] test_yfinance_client_single_ticker
    - [ ] test_yfinance_client_multiple_tickers
    - [ ] test_collect_module_run
    - [ ] test_collect_no_duplicates
```

**AI 작업 지시**:
```
tests/test_collect.py에 Collect 모듈 테스트를 작성해줘.
실제 yfinance API를 호출하는 통합 테스트와,
모킹을 사용하는 단위 테스트 모두 포함.
```

#### 4.3 실제 데이터 수집 테스트
```bash
# 작업 항목
- [ ] 10개 종목으로 실제 수집 테스트
- [ ] DB에 데이터 저장 확인
- [ ] 수집 시간 측정 (100개 종목 예상)
```

---

### Day 5 (1/16 목) - Process Module (Features)

#### 5.1 피처 엔지니어링 기본
```bash
# 작업 항목
- [ ] morgorithm/process/features.py
    - [ ] 가격 기반 피처 (15개)
    - [ ] 변동성 피처 (10개)
    - [ ] 거래량 피처 (8개)
```

**AI 작업 지시**:
```
morgorithm/process/features.py에 피처 계산 함수들을 구현해줘.

calculate_price_features(df) -> DataFrame:
  - returns_1m, returns_5m, returns_15m, returns_30m, returns_60m
  - ma_5, ma_15, ma_60
  - ma_cross (ma_5 > ma_15)
  - price_vs_ma_5, price_vs_ma_15, price_vs_ma_60
  - price_momentum
  - price_acceleration
  - high_low_range

calculate_volatility_features(df) -> DataFrame:
  - atr_14
  - bb_upper, bb_lower, bb_position, bb_width
  - volatility_5m, volatility_15m, volatility_60m
  - true_range_pct

calculate_volume_features(df) -> DataFrame:
  - volume_ma_5, volume_ma_15
  - volume_ratio (current / ma)
  - obv (On Balance Volume)
  - money_flow
  - mfi_14 (Money Flow Index)
  - volume_price_trend
  - accumulation_distribution

FiveForFree의 src/processor/feature_engineer.py를 참조해도 됨.
```

#### 5.2 피처 엔지니어링 고급
```bash
# 작업 항목
- [ ] 모멘텀 피처 (8개)
- [ ] 시장 맥락 피처 (5개)
- [ ] 시간 기반 피처 (3개)
```

**AI 작업 지시**:
```
추가 피처 함수들:

calculate_momentum_features(df) -> DataFrame:
  - rsi_14
  - macd, macd_signal, macd_hist
  - stoch_k, stoch_d
  - williams_r
  - cci_14

calculate_time_features(timestamp) -> dict:
  - minutes_since_open
  - day_of_week
  - is_first_hour, is_last_hour

calculate_market_context(spy_df, qqq_df) -> DataFrame:
  - spy_return_1h
  - qqq_return_1h
  - spy_vs_qqq
  - market_direction
  - (vix는 별도 API 필요, 일단 제외)
```

---

### Day 6 (1/17 금) - Process Module (Labels) + 완성

#### 6.1 레이블 생성
```bash
# 작업 항목
- [ ] morgorithm/process/labels.py
    - [ ] generate_volatility_label()
    - [ ] generate_direction_label()
```

**AI 작업 지시**:
```
morgorithm/process/labels.py에 레이블 생성 로직을 구현해줘.

def generate_labels(
    minute_bars: DataFrame,
    current_timestamp: datetime,
    horizon_minutes: int = 60,
    volatility_threshold: float = 0.01
) -> LabelResult:
    """
    현재 시점에서 1시간 후까지의 레이블 생성

    Returns:
        label_volatile: bool (|max_change| >= threshold)
        label_direction: bool or None (상승=True, 하락=False, not volatile=None)
        actual_return: float
        max_gain: float
        max_loss: float
    """

주의: Data Leak 방지
- current_timestamp 이후의 데이터만 사용
- current_timestamp + horizon_minutes까지만 조회
```

#### 6.2 Process Module 통합
```bash
# 작업 항목
- [ ] morgorithm/process/module.py
    - [ ] ProcessModule 클래스
    - [ ] run() 메서드
```

**AI 작업 지시**:
```
morgorithm/process/module.py에 ProcessModule을 구현해줘.

run(since: datetime = None) 흐름:
1. minute_bars에서 처리 안 된 데이터 조회
2. 각 (ticker, timestamp) 조합에 대해:
   - 피처 계산 (최근 60봉 필요)
   - features 테이블 저장
3. 레이블 생성 가능한 데이터에 대해:
   - 레이블 계산 (미래 12봉 = 1시간 필요)
   - labels 테이블 저장
4. ProcessResult 반환

주의: 레이블은 1시간 후 데이터가 있어야 생성 가능
```

#### 6.3 Process 테스트
```bash
# 작업 항목
- [ ] tests/test_process.py
    - [ ] test_price_features
    - [ ] test_volatility_features
    - [ ] test_label_generation
    - [ ] test_no_data_leak (중요!)
```

**AI 작업 지시**:
```
tests/test_process.py 작성.

특히 test_no_data_leak_in_features와 test_no_data_leak_in_labels가 중요:

def test_no_data_leak_in_features():
    # timestamp T에서 피처 계산 시
    # T 이후 데이터가 사용되지 않음을 검증

def test_no_data_leak_in_labels():
    # timestamp T의 레이블은
    # T+1 ~ T+60분 데이터로만 생성됨을 검증
```

#### Day 6 완료 기준
- [ ] Collect → Process 파이프라인 정상 작동
- [ ] minute_bars → features, labels 변환 성공
- [ ] Data Leak 테스트 통과

---

## Phase 3: Model Training (Day 7-9)

### Day 7 (1/18 토) - Volatility Model (Stage 1)

#### 7.1 기본 모델 클래스
```bash
# 작업 항목
- [ ] morgorithm/train/base.py
    - [ ] BaseModel 추상 클래스
    - [ ] 공통 인터페이스 정의
```

**AI 작업 지시**:
```
morgorithm/train/base.py에 기본 모델 클래스 정의:

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y, X_val=None, y_val=None) -> TrainResult

    @abstractmethod
    def predict(self, X) -> np.ndarray

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray

    @abstractmethod
    def save(self, path: str)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel'
```

#### 7.2 Volatility Model
```bash
# 작업 항목
- [ ] morgorithm/train/volatility.py
    - [ ] VolatilityModel 클래스 (LightGBM)
    - [ ] 학습/예측/저장/로드
```

**AI 작업 지시**:
```
morgorithm/train/volatility.py 구현:

class VolatilityModel(BaseModel):
    """Stage 1: 변동성 예측 모델"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1
        }

    def train(self, X, y, X_val=None, y_val=None):
        # LightGBM 학습
        # 클래스 불균형 처리 (scale_pos_weight)
        # early_stopping 적용
        pass
```

#### 7.3 모델 레지스트리
```bash
# 작업 항목
- [ ] morgorithm/train/registry.py
    - [ ] 모델 버전 관리
    - [ ] active/training 교체 로직
```

**AI 작업 지시**:
```
morgorithm/train/registry.py 구현:

class ModelRegistry:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path

    def save_training_model(self, model, ticker, model_type, metrics):
        """training/ 디렉토리에 저장"""

    def promote_to_active(self, ticker, model_type, version):
        """training → active 교체 (atomic)"""

    def get_active_model(self, ticker, model_type) -> BaseModel:
        """현재 active 모델 로드"""

    def archive_model(self, ticker, model_type, version):
        """active → archive 이동"""
```

---

### Day 8 (1/19 일) - Direction Model (Stage 2)

#### 8.1 Direction Model
```bash
# 작업 항목
- [ ] morgorithm/train/direction.py
    - [ ] DirectionModel 클래스
    - [ ] p_v를 피처로 추가하는 로직
```

**AI 작업 지시**:
```
morgorithm/train/direction.py 구현:

class DirectionModel(BaseModel):
    """Stage 2: 방향 예측 모델"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        # Volatility Model과 유사한 구조
        # 단, p_volatile을 추가 피처로 받음

    def prepare_features(self, X: DataFrame, p_volatile: np.ndarray) -> DataFrame:
        """p_volatile을 피처에 추가"""
        X_extended = X.copy()
        X_extended['p_volatile'] = p_volatile
        return X_extended

    def train(self, X, y, p_volatile, X_val=None, y_val=None, p_volatile_val=None):
        X_ext = self.prepare_features(X, p_volatile)
        # 학습
```

#### 8.2 Train Module 통합
```bash
# 작업 항목
- [ ] morgorithm/train/module.py
    - [ ] TrainModule 클래스
    - [ ] run_full(), run_incremental()
```

**AI 작업 지시**:
```
morgorithm/train/module.py 구현:

class TrainModule:
    def run_full(self, days: int = 30) -> TrainResult:
        """
        전체 재학습:
        1. 최근 N일 features + labels 조회
        2. 각 종목별:
           a. Volatility Model 학습
           b. Direction Model 학습 (Stage 1 예측값 사용)
        3. 모델 저장 (training/)
        4. 성능 비교 후 교체 (training → active)
        """

    def run_incremental(self, hours: int = 2) -> TrainResult:
        """
        증분 학습:
        1. 기존 active 모델 로드
        2. 최근 N시간 데이터로 추가 학습
        3. 성능 개선 시 교체
        """
```

---

### Day 9 (1/20 월) - Train 테스트 + 검증

#### 9.1 Train 테스트
```bash
# 작업 항목
- [ ] tests/test_train.py
    - [ ] test_volatility_model_train
    - [ ] test_direction_model_train
    - [ ] test_model_save_load
    - [ ] test_model_registry
```

**AI 작업 지시**:
```
tests/test_train.py 작성.

테스트 케이스:
1. 작은 데이터셋으로 모델 학습 성공
2. 모델 저장 후 로드 시 동일 예측
3. 레지스트리 버전 관리 정상 작동
4. 증분 학습 후 모델 성능 유지/개선
```

#### 9.2 실제 데이터 학습 테스트
```bash
# 작업 항목
- [ ] 수집된 데이터로 실제 학습 테스트
- [ ] 10개 종목 학습 시간 측정
- [ ] 모델 성능 지표 확인 (AUC 등)
```

**AI 작업 지시**:
```
scripts/test_training.py 작성:
1. 현재 수집된 데이터 상태 확인
2. 10개 종목 선택
3. Volatility + Direction 모델 학습
4. 학습 시간, 샘플 수, AUC 출력
5. 모델 파일 생성 확인
```

#### Day 9 완료 기준
- [ ] 2-Stage 모델 학습 파이프라인 완성
- [ ] 모델 저장/로드/버전관리 정상 작동
- [ ] 최소 10개 종목 모델 학습 성공

---

## Phase 4: Prediction & Evaluation (Day 10-12)

### Day 10 (1/21 화) - Predict Module

#### 10.1 Soft Gating 로직
```bash
# 작업 항목
- [ ] morgorithm/predict/soft_gate.py
    - [ ] combine_probabilities()
    - [ ] calculate_expected_value()
    - [ ] rank_predictions()
```

**AI 작업 지시**:
```
morgorithm/predict/soft_gate.py 구현:

def combine_probabilities(p_volatile: float, p_direction: float) -> float:
    """Soft Gating: p_combined = p_v × p_up"""
    return p_volatile * p_direction

def calculate_expected_value(
    p_combined: float,
    win_return: float = 0.008,   # 1% - 0.2% 수수료
    loss_return: float = -0.012  # 평균 손실 + 수수료
) -> float:
    """EV = p × win - (1-p) × |loss|"""
    return p_combined * win_return + (1 - p_combined) * loss_return

def rank_predictions(predictions: List[Prediction]) -> List[Prediction]:
    """EV 기준 내림차순 정렬, rank 부여"""
    sorted_preds = sorted(predictions, key=lambda x: x.expected_value, reverse=True)
    for i, pred in enumerate(sorted_preds):
        pred.ev_rank = i + 1
    return sorted_preds
```

#### 10.2 Predict Module
```bash
# 작업 항목
- [ ] morgorithm/predict/module.py
    - [ ] PredictModule 클래스
    - [ ] run() 메서드
```

**AI 작업 지시**:
```
morgorithm/predict/module.py 구현:

class PredictModule:
    def run(self) -> PredictResult:
        """
        1. 최신 features 조회 (각 종목별 가장 최근)
        2. 각 종목별:
           a. Volatility Model 로드 → p_v 예측
           b. Direction Model 로드 → p_up 예측 (p_v 피처 추가)
           c. Soft Gating: p_combined = p_v × p_up
           d. EV 계산
        3. EV 기준 랭킹
        4. predictions 테이블 저장
        5. PredictResult 반환
        """
```

---

### Day 11 (1/22 수) - Evaluate Module

#### 11.1 Outcome 업데이트
```bash
# 작업 항목
- [ ] morgorithm/evaluate/module.py
    - [ ] update_outcomes() 메서드
```

**AI 작업 지시**:
```
morgorithm/evaluate/module.py 구현:

class EvaluateModule:
    def update_outcomes(self) -> EvaluateResult:
        """
        1. target_time이 지난 predictions 조회 (outcome 없는 것)
        2. 각 prediction에 대해:
           a. entry_price ~ target_time 구간의 minute_bars 조회
           b. max_gain, max_loss 계산
           c. actual_volatile = (max_gain >= 0.01) or (max_loss <= -0.01)
           d. actual_direction = max_gain >= abs(max_loss)
           e. actual_return = 1시간 후 close / entry_price - 1
           f. profit_loss = actual_return - commission (0.2%)
        3. outcomes 테이블 저장
        """
```

#### 11.2 Metrics 계산
```bash
# 작업 항목
- [ ] morgorithm/evaluate/metrics.py
    - [ ] calculate_model_metrics()
```

**AI 작업 지시**:
```
morgorithm/evaluate/metrics.py 구현:

def calculate_model_metrics(
    predictions: List[Prediction],
    outcomes: List[Outcome],
    window_hours: int = 50
) -> ModelMetrics:
    """
    Returns:
        - total_predictions
        - volatile_precision, volatile_recall
        - direction_precision, direction_accuracy
        - combined_accuracy (p_combined > 0.5 기준)
        - total_return (누적)
        - win_rate
        - avg_profit, avg_loss
        - sharpe_ratio
        - max_drawdown
    """
```

---

### Day 12 (1/23 목) - Predict + Evaluate 테스트

#### 12.1 테스트 작성
```bash
# 작업 항목
- [ ] tests/test_predict.py
- [ ] tests/test_evaluate.py
- [ ] tests/test_data_leak.py (통합)
```

**AI 작업 지시**:
```
테스트 파일들 작성:

tests/test_predict.py:
- test_soft_gating
- test_expected_value_calculation
- test_ranking
- test_predict_module_run

tests/test_evaluate.py:
- test_outcome_update
- test_metrics_calculation
- test_win_rate_calculation

tests/test_data_leak.py:
- test_no_leak_in_features
- test_no_leak_in_labels
- test_no_leak_in_evaluation
- test_prediction_uses_only_past_data
```

#### 12.2 파이프라인 통합 테스트
```bash
# 작업 항목
- [ ] Collect → Process → Train → Predict → Evaluate 전체 흐름
- [ ] 실제 데이터로 검증
```

**AI 작업 지시**:
```
scripts/test_pipeline.py 작성:

전체 파이프라인 테스트:
1. 10개 종목 데이터 수집 (Collect)
2. 피처/레이블 생성 (Process)
3. 모델 학습 (Train)
4. 예측 생성 (Predict)
5. 결과 평가 (Evaluate - 과거 데이터 기준)
6. 각 단계 성공 여부 및 소요 시간 출력
```

#### Day 12 완료 기준
- [ ] 전체 파이프라인 통합 테스트 통과
- [ ] Data Leak 테스트 100% 통과
- [ ] 예측 → 결과 평가 정상 작동

---

## Phase 5: Backtesting (Day 13-14)

### Day 13 (1/24 금) - Backtest Module

#### 13.1 Backtest Simulator
```bash
# 작업 항목
- [ ] morgorithm/backtest/simulator.py
    - [ ] BacktestSimulator 클래스
    - [ ] 거래 시뮬레이션 로직
```

**AI 작업 지시**:
```
morgorithm/backtest/simulator.py 구현:

@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    min_p_combined: float = 0.5
    top_k: int = 5
    initial_capital: float = 100000.0
    commission: float = 0.002

class BacktestSimulator:
    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        1. 기간 내 predictions + outcomes 조회
        2. 시간순으로 정렬
        3. 각 예측 시점별:
           a. EV 상위 K개 필터링
           b. 가상 매수 (자본 / K)
           c. 1시간 후 outcome 확인
           d. profit_loss 계산
           e. 자본 업데이트
        4. 결과 집계:
           - 총 거래 수
           - 승률
           - 총 수익률
           - Sharpe ratio
           - Max drawdown
           - Equity curve
        """
```

#### 13.2 Backtest Module
```bash
# 작업 항목
- [ ] morgorithm/backtest/module.py
    - [ ] BacktestModule 클래스
```

**AI 작업 지시**:
```
morgorithm/backtest/module.py 구현:

class BacktestModule:
    def run(self, config: BacktestConfig) -> BacktestResult:
        simulator = BacktestSimulator()
        result = simulator.run(config)

        # backtest_results 테이블에 저장
        self._save_result(config, result)

        return result

    def compare_strategies(
        self,
        configs: List[BacktestConfig]
    ) -> ComparisonResult:
        """여러 전략 비교"""
```

---

### Day 14 (1/25 토) - 검증 및 마무리

#### 14.1 백테스트 테스트
```bash
# 작업 항목
- [ ] tests/test_backtest.py
- [ ] 다양한 전략 비교 테스트
```

**AI 작업 지시**:
```
tests/test_backtest.py 작성:
- test_backtest_basic
- test_commission_applied
- test_drawdown_calculation
- test_different_strategies
```

#### 14.2 전체 시스템 검증
```bash
# 작업 항목
- [ ] 실제 수집 데이터로 백테스트 실행
- [ ] 수익률 분석
- [ ] 문제점 식별 및 문서화
```

**AI 작업 지시**:
```
scripts/run_validation.py 작성:

1. 현재까지 수집된 전체 데이터 확인
2. 모든 종목 모델 학습
3. 전체 기간 백테스트 실행
4. 결과 리포트 생성:
   - 종목별 성과
   - 상위/하위 모델
   - 전략별 수익률 비교
   - 개선 포인트
```

#### 14.3 문서 업데이트
```bash
# 작업 항목
- [ ] spec.md 업데이트 (변경사항 반영)
- [ ] README.md 작성
- [ ] 다음 단계 계획 정리
```

#### Day 14 완료 기준
- [ ] 백테스트 모듈 완성
- [ ] 전체 파이프라인 검증 완료
- [ ] 수익성 분석 리포트 생성
- [ ] Phase 1-5 모든 테스트 통과

---

## Phase 6+ (수익성 검증 완료 시)

> 아래 항목들은 Phase 5 백테스트 결과가 유의미한 수익성을 보일 때 진행

### API Module
- [ ] FastAPI 설정
- [ ] 예측 조회 API
- [ ] 모델 성능 API
- [ ] 인증 (Free/Pro tier)

### Deployment
- [ ] Docker 이미지 빌드
- [ ] DigitalOcean/Railway 배포
- [ ] cron 스케줄 설정
- [ ] 모니터링 대시보드

### Frontend
- [ ] React 프로젝트 설정
- [ ] 예측 대시보드
- [ ] 모델 성능 시각화
- [ ] 백테스트 결과 UI

### 고도화
- [ ] 실시간 데이터 소스 업그레이드 (Alpaca SIP)
- [ ] 추가 모델 (XGBoost, Neural Network)
- [ ] 알림 시스템
- [ ] 자동 매매 연동

---

## Daily Standup Template

```markdown
## Date: YYYY-MM-DD

### 어제 한 일
-

### 오늘 할 일
-

### 블로커/이슈
-

### 메모
-
```

---

## Quick Reference

### 실행 명령어

```bash
# 환경 설정
poetry install
cp .env.example .env
# .env 편집

# DB 초기화
poetry run python scripts/init_db.py

# 종목 시딩
poetry run python scripts/seed_tickers.py

# 파이프라인 실행
poetry run python -m morgorithm.collect
poetry run python -m morgorithm.process
poetry run python -m morgorithm.train --full
poetry run python -m morgorithm.predict
poetry run python -m morgorithm.evaluate

# 테스트
poetry run pytest tests/ -v

# 백테스트
poetry run python -m morgorithm.backtest
```

### AI 협업 규칙

1. **작업 전 확인**: 현재 진행 상황과 다음 작업 확인
2. **스펙 참조**: spec.md의 인터페이스 계약 준수
3. **테스트 우선**: 구현 후 즉시 테스트 작성
4. **커밋 단위**: 기능 단위로 작은 커밋
5. **문서화**: 변경사항 즉시 문서 반영

---

**Last Updated**: 2026-01-08
