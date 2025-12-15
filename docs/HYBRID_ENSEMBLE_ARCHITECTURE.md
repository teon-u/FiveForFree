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
- 5% 상승/하락은 전체 데이터의 ~3-5%만 해당 (극심한 클래스 불균형)
- 모델이 희소 이벤트를 학습하기 어려움

해결책:
- 변동성(±5%)과 방향(상승/하락)을 분리하여 예측
- 방향 예측은 ~50:50으로 균형적인 데이터셋
- 두 구조를 앙상블하여 최종 확률 계산
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

기존 방식으로, 5% 상승/하락을 직접 예측합니다.

```python
# 레이블 정의
label_up = 1   # 60분 내 +5% 도달
label_down = 1 # 60분 내 -5% 도달

# 모델
- UP 모델: P(+5% 상승) 예측
- DOWN 모델: P(-5% 하락) 예측

# 각 모델당 5가지 타입
MODEL_TYPES = ["xgboost", "lightgbm", "lstm", "transformer", "ensemble"]
```

**장점**: 단순하고 직관적
**단점**: 클래스 불균형 (~5% positive)

### 3.2 Structure B: 하이브리드 예측 (Volatility + Direction)

변동성과 방향을 분리하여 예측합니다.

```python
# 레이블 정의
label_volatility = 1  # 60분 내 ±5% 변동 발생 (up OR down)
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
- label_volatility 생성 (±5% 변동 여부)
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
- _train_hybrid_models(): Structure B 모델 학습
- train_single_ticker_hybrid(): 하이브리드 학습 진입점
- volatility/direction 모델 병렬/순차 학습 지원
```

**학습 흐름:**
```python
def train_single_ticker(ticker, X, y_up, y_down, y_volatility=None, y_direction=None):
    # Structure A 학습
    train_up_models(X, y_up)
    train_down_models(X, y_down)

    # Structure B 학습 (레이블 제공 시)
    if y_volatility is not None and y_direction is not None:
        train_volatility_models(X, y_volatility)
        train_direction_models(X, y_direction)
```

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

### 10.3 Structure B 개선 - 조건부 학습

현재 Direction 모델은 모든 샘플로 학습합니다.
변동성 발생 샘플만으로 학습하면 더 정확할 수 있습니다:

```python
# 변동성 샘플만 필터링
volatile_mask = y_volatility == 1
X_volatile = X[volatile_mask]
y_direction_volatile = y_direction[volatile_mask]

# 방향 모델 학습
direction_model.fit(X_volatile, y_direction_volatile)
```

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

## 13. 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-15 | 1.0 | 하이브리드-앙상블 구조 초기 구현 |

---

## 14. 참고 자료

- 클래스 불균형 처리: SMOTE, Undersampling, Class Weights
- 확률 캘리브레이션: Platt Scaling, Isotonic Regression
- 앙상블 학습: Stacking, Blending, Weighted Average
