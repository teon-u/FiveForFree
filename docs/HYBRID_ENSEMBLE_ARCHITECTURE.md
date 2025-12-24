# Hybrid-Ensemble Prediction Architecture

이 문서는 FiveForFree 프로젝트의 하이브리드-앙상블 예측 구조를 설명합니다.
다른 Agent가 이 시스템을 이해하고 수정할 수 있도록 상세히 기술합니다.

---

## 1. 개요

### 1.1 목적

기존 직접 예측 방식(Structure A)의 **클래스 불균형 문제**를 해결하고,
**변동성과 방향을 분리 예측**(Structure B)하여 더 정확한 예측을 달성합니다.

### 1.2 핵심 아이디어

```
기존 문제:
- 1% 상승/하락은 전체 데이터의 일부만 해당 (클래스 불균형)
- 모델이 희소 이벤트를 학습하기 어려움

해결책:
- 변동성(±1%)과 방향(상승/하락)을 분리하여 예측
- 방향 예측은 ~50:50으로 균형적인 데이터셋
- 두 구조를 앙상블하여 최종 확률 계산

Note: TARGET_PERCENT는 settings.py에서 1.0%로 설정됨
```

---

## 2. 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID-ENSEMBLE PREDICTION SYSTEM                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   [입력: 57개 피처]                                                  │
│          │                                                          │
│          ├────────────────────────────────────────────────┐         │
│          │                                                │         │
│          ▼                                                ▼         │
│   ┌─────────────────────┐                    ┌─────────────────────┐│
│   │   STRUCTURE A       │                    │   STRUCTURE B       ││
│   │   (Direct Pred.)    │                    │   (Hybrid Pred.)    ││
│   ├─────────────────────┤                    ├─────────────────────┤│
│   │                     │                    │                     ││
│   │  [UP Model]         │                    │  [VOLATILITY Model] ││
│   │       │             │                    │       │             ││
│   │       ▼             │                    │       ▼             ││
│   │  direct_up_prob     │                    │  volatility_prob    ││
│   │                     │                    │       │             ││
│   │  [DOWN Model]       │                    │       ▼             ││
│   │       │             │                    │  [DIRECTION Model]  ││
│   │       ▼             │                    │       │             ││
│   │  direct_down_prob   │                    │       ▼             ││
│   │                     │                    │  direction_up_prob  ││
│   └──────────┬──────────┘                    └──────────┬──────────┘│
│              │                                          │           │
│              │                               hybrid_up = vol × dir  │
│              │                               hybrid_down = vol×(1-d)│
│              │                                          │           │
│              └──────────────┬───────────────────────────┘           │
│                             │                                       │
│                             ▼                                       │
│              ┌──────────────────────────────┐                       │
│              │      ENSEMBLE COMBINER       │                       │
│              ├──────────────────────────────┤                       │
│              │                              │                       │
│              │  final_up = α × direct_up    │                       │
│              │           + (1-α) × hybrid_up│                       │
│              │                              │                       │
│              │  final_down = α × direct_down│                       │
│              │           + (1-α) × hybrid_dn│                       │
│              │                              │                       │
│              │  (α = ENSEMBLE_ALPHA = 0.5)  │                       │
│              └──────────────┬───────────────┘                       │
│                             │                                       │
│                             ▼                                       │
│              ┌──────────────────────────────┐                       │
│              │    PROBABILITY CALIBRATION   │                       │
│              │    - Clip to [0, 1]          │                       │
│              │    - Normalize if sum > 1    │                       │
│              └──────────────┬───────────────┘                       │
│                             │                                       │
│                             ▼                                       │
│                    [최종 예측 결과]                                  │
│                    - up_probability                                 │
│                    - down_probability                               │
│                    - trading_signal (BUY/SELL/HOLD)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 구조별 상세 설명

### 3.1 Structure A: 직접 예측 (Direct Prediction)

기존 방식으로, TARGET_PERCENT% 상승/하락을 직접 예측합니다. (현재 1%)

```python
# 레이블 정의
label_up = 1   # 60분 내 +1% 도달
label_down = 1 # 60분 내 -1% 도달

# 모델
- UP 모델: P(+1% 상승) 예측
- DOWN 모델: P(-1% 하락) 예측

# 각 모델당 5가지 타입
MODEL_TYPES = ["xgboost", "lightgbm", "lstm", "transformer", "ensemble"]
```

**장점**: 단순하고 직관적
**단점**: 클래스 불균형

### 3.2 Structure B: 하이브리드 예측 (Volatility + Direction)

변동성과 방향을 분리하여 예측합니다.

```python
# 레이블 정의
label_volatility = 1  # 60분 내 ±1% 변동 발생 (up OR down)
label_direction = 1   # 변동 시 상승이 먼저 발생 (up first)
label_direction = 0   # 변동 시 하락이 먼저 발생 (down first)

# 확률 계산
hybrid_up_prob = volatility_prob × direction_up_prob
hybrid_down_prob = volatility_prob × (1 - direction_up_prob)
```

**장점**:
- 방향 예측이 ~50:50으로 균형적
- "변동성은 높지만 방향 불확실" 같은 인사이트 제공

**단점**:
- 두 모델의 오차가 곱해질 수 있음
- 확률 캘리브레이션 필요

---

## 4. 설정값 (config/settings.py)

```python
# 하이브리드-앙상블 설정
HYBRID_TARGETS: list[str] = ["volatility", "direction"]
USE_HYBRID_ENSEMBLE: bool = True      # 하이브리드 활성화 여부
ENSEMBLE_ALPHA: float = 0.5           # 직접 예측 가중치 (0~1)
CALIBRATION_METHOD: str = "isotonic"  # 확률 캘리브레이션 방법
```

### 4.1 ENSEMBLE_ALPHA 가이드

| 값 | 의미 | 권장 상황 |
|---|------|----------|
| 1.0 | Structure A만 사용 | 하이브리드 비활성화 |
| 0.7 | 직접 예측 70% + 하이브리드 30% | 직접 예측이 더 정확할 때 |
| 0.5 | 50:50 균형 (기본값) | 초기 테스트 |
| 0.3 | 직접 예측 30% + 하이브리드 70% | 하이브리드가 더 정확할 때 |
| 0.0 | Structure B만 사용 | 클래스 불균형이 심할 때 |

---

## 5. 파일별 역할

### 5.1 config/settings.py
```
역할: 전역 설정 관리
추가된 설정:
- HYBRID_TARGETS
- USE_HYBRID_ENSEMBLE
- ENSEMBLE_ALPHA
- CALIBRATION_METHOD
```

### 5.2 src/processor/label_generator.py
```
역할: 레이블 생성
추가된 기능:
- label_volatility 생성 (±TARGET_PERCENT% 변동 여부, 현재 1%)
- label_direction 생성 (상승/하락 먼저 도달)
- get_label_statistics()에 Structure B 통계 추가
```

**레이블 생성 로직:**
```python
# label_volatility
label_volatility = 1 if (label_up == 1 or label_down == 1) else 0

# label_direction
if label_volatility == 1:
    if minutes_to_up is not None and minutes_to_down is not None:
        # 둘 다 도달한 경우 - 먼저 도달한 방향
        label_direction = 1 if minutes_to_up <= minutes_to_down else 0
    elif minutes_to_up is not None:
        label_direction = 1  # 상승만 도달
    else:
        label_direction = 0  # 하락만 도달
else:
    # 변동성 없음 - 어느 쪽이 더 가까웠는지
    label_direction = 1 if max_gain > abs(max_loss) else 0
```

### 5.3 src/trainer/gpu_trainer.py
```
역할: GPU 기반 모델 학습
추가된 기능:
- _train_hybrid_models(): Structure B 모델 2-Stage 학습
- train_single_ticker_hybrid(): 하이브리드 학습 진입점
- volatility/direction 모델 병렬/순차 학습 지원
```

**2-Stage 학습 흐름:**
```python
def _train_hybrid_models(ticker, X, y_volatility, y_direction):
    # ===== Stage 1: Volatility (전체 샘플) =====
    train_volatility_models(X, y_volatility)

    # ===== Stage 2: Direction (volatile 샘플만) =====
    # 변동성이 있는 샘플만 필터링
    volatile_mask = y_volatility == 1
    X_volatile = X[volatile_mask]
    y_dir_volatile = y_direction[volatile_mask]

    # 변동성 있는 샘플만으로 방향 모델 학습
    train_direction_models(X_volatile, y_dir_volatile)
```

**2-Stage 학습의 장점:**
- 노이즈 제거: 변동성 없는 샘플의 "방향"은 의미 없음
- 정확도 향상: 실제 ±1% 도달한 케이스만 학습
- 클래스 균형: volatile 샘플에서 direction은 ~50:50

### 5.4 src/predictor/realtime_predictor.py
```
역할: 실시간 예측
추가된 기능:
- HybridPredictionDetail: 상세 분해 정보 저장
- _predict_hybrid(): Structure B 예측
- _calibrate_probabilities(): 확률 캘리브레이션
- predict()에 use_hybrid 파라미터 추가
```

**예측 흐름:**
```python
def predict(ticker, use_hybrid=True):
    # Structure A: 직접 예측
    direct_up = up_model.predict(X)
    direct_down = down_model.predict(X)

    # Structure B: 하이브리드 예측
    if use_hybrid:
        vol_prob = volatility_model.predict(X)
        dir_prob = direction_model.predict(X)
        hybrid_up = vol_prob * dir_prob
        hybrid_down = vol_prob * (1 - dir_prob)

    # 앙상블 결합
    final_up = alpha * direct_up + (1-alpha) * hybrid_up
    final_down = alpha * direct_down + (1-alpha) * hybrid_down

    # 캘리브레이션
    final_up, final_down = calibrate(final_up, final_down)

    return PredictionResult(...)
```

---

## 6. 데이터 클래스

### 6.1 HybridPredictionDetail

```python
@dataclass
class HybridPredictionDetail:
    direct_up_prob: float      # Structure A 상승 확률
    direct_down_prob: float    # Structure A 하락 확률
    volatility_prob: float     # Structure B 변동성 확률
    direction_up_prob: float   # Structure B 방향(상승) 확률
    hybrid_up_prob: float      # vol × dir_up
    hybrid_down_prob: float    # vol × (1 - dir_up)
    ensemble_alpha: float      # 사용된 앙상블 가중치
```

### 6.2 PredictionResult (확장)

```python
@dataclass
class PredictionResult:
    # 기존 필드
    ticker: str
    up_probability: float      # 최종 앙상블 확률
    down_probability: float
    ...

    # 새 필드
    hybrid_detail: Optional[HybridPredictionDetail]  # 상세 분해 정보
```

---

## 7. API 응답 예시

### 7.1 하이브리드 활성화 시

```json
{
    "ticker": "AAPL",
    "current_price": 235.50,
    "up_probability": 0.655,
    "down_probability": 0.195,
    "best_up_model": "transformer",
    "best_down_model": "xgboost",
    "trading_signal": "HOLD",
    "hybrid_detail": {
        "direct_up_prob": 0.75,
        "direct_down_prob": 0.12,
        "volatility_prob": 0.82,
        "direction_up_prob": 0.68,
        "hybrid_up_prob": 0.56,
        "hybrid_down_prob": 0.26,
        "ensemble_alpha": 0.5
    }
}
```

### 7.2 하이브리드 비활성화 시

```json
{
    "ticker": "AAPL",
    "up_probability": 0.75,
    "down_probability": 0.12,
    "hybrid_detail": null
}
```

---

## 8. 통계 예시 (get_label_statistics)

```python
{
    "total_samples": 10000,

    # Structure A 통계
    "up_count": 350,
    "down_count": 420,
    "up_rate": 0.035,        # 3.5% - 불균형
    "down_rate": 0.042,      # 4.2% - 불균형

    # Structure B 통계
    "volatility_count": 680,
    "volatility_rate": 0.068,           # 6.8%
    "direction_up_count": 310,
    "direction_down_count": 370,
    "direction_up_rate": 0.456,         # 45.6% - 균형적!
    "direction_down_rate": 0.544        # 54.4% - 균형적!
}
```

**핵심 포인트**:
- Structure A의 up/down rate: ~3-5% (불균형)
- Structure B의 direction rate: ~45-55% (균형)

---

## 9. 사용 가이드

### 9.1 학습 시

```python
from src.trainer.gpu_trainer import GPUParallelTrainer
from src.processor.label_generator import LabelGenerator

# 레이블 생성
label_gen = LabelGenerator()
labels_df = label_gen.generate_labels_vectorized(minute_bars)

# 레이블 추출
y_up = labels_df['label_up'].values
y_down = labels_df['label_down'].values
y_volatility = labels_df['label_volatility'].values
y_direction = labels_df['label_direction'].values

# 하이브리드 학습
trainer = GPUParallelTrainer(model_manager)
trainer.train_single_ticker_hybrid(
    ticker="AAPL",
    X=features,
    y_up=y_up,
    y_down=y_down,
    y_volatility=y_volatility,
    y_direction=y_direction
)
```

### 9.2 예측 시

```python
from src.predictor.realtime_predictor import RealtimePredictor

predictor = RealtimePredictor(model_manager)

# 하이브리드 예측 (기본값)
result = predictor.predict("AAPL")
print(f"UP: {result.up_probability:.3f}")
print(f"DOWN: {result.down_probability:.3f}")
print(f"Hybrid Detail: {result.hybrid_detail}")

# 직접 예측만 사용
result = predictor.predict("AAPL", use_hybrid=False)
```

### 9.3 하이브리드 비활성화

```python
# settings.py에서
USE_HYBRID_ENSEMBLE = False

# 또는 런타임에서
result = predictor.predict("AAPL", use_hybrid=False)
```

---

## 10. 향후 개선 방향

### 10.1 확률 캘리브레이션 고도화

현재는 기본적인 클리핑과 정규화만 적용합니다.
향후 Platt Scaling 또는 Isotonic Regression 적용 권장:

```python
from sklearn.calibration import CalibratedClassifierCV

# 학습 시 캘리브레이션 모델 생성
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic')
calibrated_model.fit(X_calib, y_calib)
```

### 10.2 동적 Alpha 조정

현재 ENSEMBLE_ALPHA는 고정값입니다.
모델 성능에 따라 동적으로 조정하는 방안:

```python
# 각 구조의 최근 정확도 기반 가중치
direct_accuracy = 0.68
hybrid_accuracy = 0.72

# 정확도 비례 가중치
alpha = direct_accuracy / (direct_accuracy + hybrid_accuracy)  # 0.486
```

### 10.3 Structure B 개선 - 조건부 학습 ✅ (구현 완료)

**구현 완료** (2025-12-22): Direction 모델은 이제 volatile 샘플만으로 학습합니다.

```python
# src/trainer/gpu_trainer.py의 _train_hybrid_models()
# Stage 2: 변동성 샘플만 필터링하여 방향 모델 학습
volatile_mask = y_volatility == 1
X_volatile = X[volatile_mask]
y_direction_volatile = y_direction[volatile_mask]

# 방향 모델 학습 (volatile 샘플만)
direction_model.fit(X_volatile, y_direction_volatile)
```

**기대 효과:**
- 노이즈 제거로 방향 예측 정확도 향상
- 의미 있는 데이터만 학습하여 과적합 방지
- ~50:50 클래스 균형으로 안정적인 학습

---

## 11. 트러블슈팅

### 11.1 하이브리드 모델이 없는 경우

```
Warning: Hybrid models not available for AAPL
```

**원인**: volatility/direction 모델이 학습되지 않음
**해결**: `train_single_ticker_hybrid()`로 학습 또는 `use_hybrid=False` 사용

### 11.2 확률 합이 1을 초과

```
final_up: 0.65, final_down: 0.55, total: 1.20
```

**원인**: 독립적인 모델들의 확률을 결합
**해결**: `_calibrate_probabilities()`에서 자동 정규화됨

### 11.3 방향 예측이 불균형

예상과 달리 direction_up_rate가 80% 이상인 경우:

**원인**: 데이터 기간이 상승장에 편중됨
**해결**: 다양한 시장 상황을 포함한 데이터로 학습

---

## 12. 관련 파일 경로

| 파일 | 위치 | 역할 |
|------|------|------|
| settings.py | `config/settings.py` | 전역 설정 |
| label_generator.py | `src/processor/label_generator.py` | 레이블 생성 |
| gpu_trainer.py | `src/trainer/gpu_trainer.py` | 모델 학습 |
| realtime_predictor.py | `src/predictor/realtime_predictor.py` | 예측 실행 |

---

## 13. Multi-Strategy Ensemble Model (NEW)

### 13.1 개요

기존 단일 앙상블 방식을 개선하여 **다중 전략 앙상블**을 구현했습니다.

```
이전 방식:
- Stacking with LogisticRegression meta-learner

새로운 방식:
- Precision-Weighted Voting
- Stacking with XGBoost meta-learner
- Dynamic Model Selection
- Hybrid (3가지 전략 결합)
```

### 13.2 앙상블 전략 (EnsembleStrategy)

```python
from src.models.ensemble_model import EnsembleStrategy

# 사용 가능한 전략
EnsembleStrategy.PRECISION_WEIGHTED  # Precision 기반 가중 투표
EnsembleStrategy.STACKING            # XGBoost 메타 러너 스태킹
EnsembleStrategy.DYNAMIC_SELECTION   # 최고 성능 모델 선택
EnsembleStrategy.HYBRID              # 3가지 전략 결합 (기본값)
```

### 13.3 전략별 설명

#### A. Precision-Weighted Voting

각 모델의 **Precision**(정밀도)에 따라 가중치를 부여합니다.

```python
# Breakeven Precision (30%)를 초과하는 정도로 가중치 계산
weight = max(0, precision - BREAKEVEN_PRECISION)

# 예시: XGBoost=37.5%, LightGBM=32%, LSTM=28%
# weights: XGBoost=0.075, LightGBM=0.02, LSTM=0 (미달)
# 정규화 후: XGBoost=78.9%, LightGBM=21.1%
```

**장점**: 실제 수익성과 직결된 가중치
**적합 상황**: Precision이 뚜렷하게 차이날 때

#### B. Stacking with XGBoost

베이스 모델의 예측을 입력으로 XGBoost 메타 러너가 최종 예측합니다.

```python
# 메타 러너 설정
XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    objective='binary:logistic'
)
```

**장점**: 비선형 결합으로 복잡한 패턴 포착
**적합 상황**: 모델 간 상호작용이 중요할 때

#### C. Dynamic Model Selection

가장 Precision이 높은 **단일 모델**만 사용합니다.

```python
# 50시간 기준 Precision이 가장 높은 모델 선택
best_model = max(models, key=lambda m: m.get_precision_at_threshold())
```

**장점**: 단순하고 빠름, 최고 모델의 성능 유지
**적합 상황**: 특정 모델이 압도적으로 우수할 때

#### D. Hybrid (기본값)

세 가지 전략을 가중 결합합니다.

```python
# 기본 가중치
strategy_weights = {
    'precision_weighted': 0.4,  # 40%
    'stacking': 0.4,            # 40%
    'dynamic_selection': 0.2    # 20%
}

final_prob = (
    0.4 * precision_weighted_prob +
    0.4 * stacking_prob +
    0.2 * dynamic_selection_prob
)
```

### 13.4 사용 예시

```python
from src.models.model_manager import ModelManager
from src.models.ensemble_model import EnsembleModel, EnsembleStrategy

# ModelManager 초기화 및 모델 로드
mm = ModelManager()
mm.load_all_models()

# 앙상블 모델 생성 (기본: HYBRID)
_, ensemble = mm.get_or_create_model('AAPL', 'ensemble', 'up')

# 전략 변경
ensemble.set_strategy(EnsembleStrategy.PRECISION_WEIGHTED)

# Hybrid 전략 가중치 조정
ensemble.set_strategy_weights({
    'precision_weighted': 0.5,
    'stacking': 0.3,
    'dynamic_selection': 0.2
})

# 앙상블 통계 확인
stats = ensemble.get_ensemble_stats()
print(f"Strategy: {stats['strategy']}")
print(f"Best model: {stats['best_model']}")
print(f"Precision weights: {stats['precision_weights']}")
```

### 13.5 학습 흐름

```python
# GPU Trainer가 자동으로 앙상블도 학습
trainer = GPUParallelTrainer(model_manager)
results = trainer.train_single_ticker(
    ticker='AAPL',
    X=features,
    y_up=labels_up,
    y_down=labels_down
)

# 결과에 ensemble_up, ensemble_down 포함
print(results)
# {'xgboost_up': True, 'lightgbm_up': True, ..., 'ensemble_up': True, 'ensemble_down': True}
```

### 13.6 성능 비교

| 전략 | 장점 | 단점 | 권장 상황 |
|------|------|------|----------|
| Precision-Weighted | 수익성 직결 | 히스토리 필요 | Precision 차이 클 때 |
| Stacking | 비선형 결합 | 과적합 위험 | 충분한 데이터 있을 때 |
| Dynamic Selection | 단순/빠름 | 단일 모델 의존 | 최고 모델이 안정적일 때 |
| Hybrid (기본) | 균형/안정 | 약간 복잡 | 대부분의 경우 |

---

## 14. 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-22 | 2.1 | 2-Stage 학습 구현 (Direction 모델 volatile 샘플만 학습) |
| 2025-12-17 | 2.0 | Multi-Strategy Ensemble 추가 |
| 2025-12-17 | 1.1 | Precision 기반 메트릭으로 변경 (Hit Rate -> Precision) |
| 2025-12-15 | 1.0 | 하이브리드-앙상블 구조 초기 구현 |

---

## 15. 참고 자료

- 클래스 불균형 처리: SMOTE, Undersampling, Class Weights
- 확률 캘리브레이션: Platt Scaling, Isotonic Regression
- 앙상블 학습: Stacking, Blending, Weighted Average
