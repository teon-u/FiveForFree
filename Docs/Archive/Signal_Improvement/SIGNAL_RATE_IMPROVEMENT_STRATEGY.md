# Signal Rate 개선 전략서

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 개발팀장
**우선순위**: 긴급 (트레이딩 시스템 핵심 문제)

---

## 1. 현재 문제 요약

### 1.1 증상
- 모델 Accuracy: 95%+ (겉보기에 우수)
- 실제 Signal Rate: **0~1%** (치명적 문제)
- Precision/Recall: **0%** (실용성 없음)

### 1.2 원인
모델이 클래스 불균형으로 인해 **항상 0만 예측**합니다.

```
예시 (AAPL):
- 실제 label_up=1 비율: 4.8%
- 모델 예측 1 비율: 0%
- 결과: 모든 예측이 0 → Accuracy 95.2% (착시)
```

### 1.3 왜 문제인가?
- **매매 신호가 발생하지 않음** → 트레이딩 불가능
- 높은 Accuracy는 무의미 (TN만 높음)
- 실제 수익을 낼 수 없는 시스템

---

## 2. 해결 전략 (4가지)

### 전략 A: 클래스 가중치 적용 (권장 - 1순위)

**개념**: label=1에 더 높은 가중치를 부여하여 모델이 1을 예측하도록 유도

**구현 위치**: `src/trainer/gpu_trainer.py`

#### XGBoost 수정
```python
# 기존
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    ...
}

# 수정 (scale_pos_weight 추가)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 예: 19.0
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'scale_pos_weight': pos_weight,  # 핵심!
    ...
}
```

#### LightGBM 수정
```python
# 기존
lgb_params = {
    'objective': 'binary',
    'device': 'gpu',
    ...
}

# 수정 (is_unbalance 또는 scale_pos_weight 추가)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
lgb_params = {
    'objective': 'binary',
    'device': 'gpu',
    'scale_pos_weight': pos_weight,  # 방법 1
    # 또는 'is_unbalance': True,     # 방법 2 (자동 계산)
    ...
}
```

#### PyTorch (LSTM/Transformer) 수정
```python
# 기존
criterion = nn.BCEWithLogitsLoss()

# 수정 (pos_weight 추가)
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### 전략 B: Focal Loss 적용 (권장 - 2순위)

**개념**: 쉬운 샘플(다수 클래스)의 손실을 줄이고, 어려운 샘플(소수 클래스)에 집중

**구현 위치**: `src/trainer/gpu_trainer.py` (새 함수 추가)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# 사용
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**권장 파라미터**:
- `alpha=0.25`: 소수 클래스 가중치
- `gamma=2.0`: focusing parameter (높을수록 어려운 샘플에 집중)

---

### 전략 C: 예측 임계값 조정 (빠른 적용 가능)

**개념**: 기본 0.5 대신 낮은 임계값 사용

**구현 위치**: `src/predictor/signal_generator.py` 또는 예측 로직

```python
# 기존
prediction = 1 if model.predict_proba(X) > 0.5 else 0

# 수정 (임계값 조정)
THRESHOLD = 0.3  # 또는 0.35, 0.4 등 테스트 필요
prediction = 1 if model.predict_proba(X) > THRESHOLD else 0
```

**동적 임계값 계산** (검증 데이터 기반):
```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_pred_proba, min_precision=0.6):
    """
    최소 precision을 유지하면서 recall을 최대화하는 임계값 찾기
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # min_precision 이상인 것 중 recall이 가장 높은 임계값
    valid_idx = precisions[:-1] >= min_precision
    if valid_idx.sum() == 0:
        return 0.5  # 기본값

    best_idx = recalls[:-1][valid_idx].argmax()
    return thresholds[valid_idx][best_idx]
```

---

### 전략 D: 오버샘플링 (SMOTE)

**개념**: 소수 클래스 샘플을 인위적으로 증가

**구현 위치**: `src/trainer/gpu_trainer.py`

```python
from imblearn.over_sampling import SMOTE

def balance_dataset(X_train, y_train):
    """
    SMOTE를 사용하여 클래스 균형 맞추기
    """
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    return X_balanced, y_balanced

# 학습 전 적용
X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
model.fit(X_train_balanced, y_train_balanced)
```

**주의사항**:
- LSTM/Transformer 같은 시퀀스 모델에는 직접 적용 어려움
- Tree 모델(XGBoost, LightGBM)에 효과적
- 패키지 설치 필요: `pip install imbalanced-learn`

---

## 3. 구현 우선순위 및 일정

| 순위 | 전략 | 난이도 | 예상 효과 | 적용 대상 |
|------|------|--------|----------|----------|
| 1 | 클래스 가중치 | 낮음 | 높음 | 모든 모델 |
| 2 | Focal Loss | 중간 | 높음 | LSTM, Transformer |
| 3 | 임계값 조정 | 낮음 | 중간 | 예측 단계 |
| 4 | SMOTE | 중간 | 중간 | XGBoost, LightGBM |

**권장 구현 순서**:
1. **즉시**: 전략 A (클래스 가중치) - 모든 모델에 적용
2. **1단계 후**: 전략 C (임계값 조정) - 검증 데이터로 최적값 탐색
3. **선택적**: 전략 B, D - 추가 성능 개선 필요 시

---

## 4. 수정 대상 파일 목록

### 4.1 필수 수정
| 파일 | 수정 내용 |
|------|----------|
| `src/trainer/gpu_trainer.py` | 클래스 가중치 추가 |
| `src/models/xgboost_model.py` | scale_pos_weight 파라미터 |
| `src/models/lightgbm_model.py` | scale_pos_weight 파라미터 |
| `src/models/lstm_model.py` | BCEWithLogitsLoss pos_weight |
| `src/models/transformer_model.py` | BCEWithLogitsLoss pos_weight |

### 4.2 선택적 수정
| 파일 | 수정 내용 |
|------|----------|
| `src/predictor/signal_generator.py` | 임계값 조정 |
| `config/settings.py` | SIGNAL_THRESHOLD 설정 추가 |

---

## 5. 검증 방법

### 5.1 수정 후 확인할 지표

```python
# 검증 스크립트 예시
def evaluate_model_quality(y_true, y_pred, y_pred_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'signal_rate': y_pred.mean(),  # 핵심! 0이면 안됨
        'actual_positive_rate': y_true.mean(),
    }
    return metrics
```

### 5.2 목표 지표

| 지표 | 현재 | 목표 |
|------|------|------|
| Signal Rate | 0~1% | **10~30%** |
| Precision | 0% | **50%+** |
| Recall | 0% | **30%+** |
| F1 Score | 0% | **35%+** |

### 5.3 테스트 명령어

```bash
# 수정 후 재학습
python scripts/train_models.py --ticker AAPL --force

# 정확도 검증
python scripts/check_stage_accuracy.py --ticker AAPL

# Signal Rate 확인 (새로 추가 권장)
python scripts/check_signal_rate.py --ticker AAPL
```

---

## 6. 주의사항

### 6.1 과적합 방지
- 클래스 가중치를 너무 높이면 과적합 발생 가능
- 권장: `pos_weight = min(계산값, 10.0)` 으로 상한선 설정

### 6.2 백테스트 필수
- Signal Rate 개선 후 반드시 백테스트 수행
- 높은 Signal Rate가 반드시 수익으로 연결되지 않음
- Precision이 50% 이하면 손실 가능성 높음

### 6.3 점진적 적용
- 한 번에 모든 전략 적용하지 말 것
- 전략 A 적용 → 검증 → 전략 C 적용 → 검증 순서로

---

## 7. 요약

**현재 문제**: 모델이 0만 예측 → Signal Rate 0% → 트레이딩 불가능

**핵심 해결책**:
1. XGBoost/LightGBM: `scale_pos_weight` 파라미터 추가
2. LSTM/Transformer: `BCEWithLogitsLoss(pos_weight=...)` 사용
3. 예측 단계: 임계값 0.5 → 0.3~0.4로 조정

**목표**: Signal Rate 10~30%, Precision 50%+ 달성

---

*분석팀장 작성*
