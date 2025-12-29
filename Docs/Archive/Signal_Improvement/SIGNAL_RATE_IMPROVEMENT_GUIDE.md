# Signal Rate 개선을 위한 모델 재훈련 가이드

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 개발팀장
**우선순위**: 긴급 (핵심 문제 해결)

---

## 1. 개요

### 1.1 현재 문제
- **Accuracy**: 95%+ (착시)
- **Signal Rate**: 0~1% (치명적)
- **Precision/Recall**: 0%
- **원인**: 클래스 불균형으로 모델이 항상 0만 예측

### 1.2 목표
| 지표 | 현재 | 목표 |
|------|------|------|
| Signal Rate | 0~1% | **10~30%** |
| Precision | 0% | **60%+** |
| Recall | 0% | **30%+** |
| F1 Score | 0% | **40%+** |

---

## 2. scale_pos_weight 최적값 계산

### 2.1 계산 공식

```python
# 기본 공식
scale_pos_weight = (negative_samples) / (positive_samples)
                 = (y == 0).sum() / (y == 1).sum()

# 예시: label_up=1이 5%인 경우
# scale_pos_weight = 0.95 / 0.05 = 19.0
```

### 2.2 동적 계산 함수

```python
def calculate_class_weight(y_train: np.ndarray, max_weight: float = 15.0) -> float:
    """
    클래스 불균형에 대한 가중치 계산.

    Args:
        y_train: 학습 레이블 배열
        max_weight: 최대 가중치 (과적합 방지)

    Returns:
        scale_pos_weight 값
    """
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()

    if n_positive == 0:
        return max_weight  # 양성 샘플 없으면 최대값

    weight = n_negative / n_positive

    # 과적합 방지를 위한 상한선 적용
    return min(weight, max_weight)
```

### 2.3 권장 가중치 범위

| 양성 비율 | 계산된 가중치 | 권장 가중치 |
|----------|-------------|------------|
| 5% | 19.0 | **10.0~15.0** |
| 10% | 9.0 | **8.0~9.0** |
| 15% | 5.67 | **5.0~6.0** |
| 20% | 4.0 | **4.0** |

**권장**: `max_weight = 15.0` 으로 상한선 설정 (과적합 방지)

---

## 3. 모델별 수정 방법

### 3.1 XGBoost 수정

**파일**: `src/models/xgboost_model.py`

**현재 코드 (64-72행)**:
```python
self._model = xgb.XGBClassifier(
    n_estimators=self.n_estimators,
    max_depth=self.max_depth,
    learning_rate=self.learning_rate,
    eval_metric='logloss',
    tree_method='hist',
    device='cuda' if use_gpu else 'cpu',
    random_state=42
)
```

**수정 코드**:
```python
def train(self, X, y, X_val=None, y_val=None):
    """Train the XGBoost model."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed")

    # GPU 사용 여부 결정
    use_gpu = settings.USE_GPU and HAS_CUDA

    # 클래스 가중치 계산
    y_array = np.asarray(y)
    n_pos = (y_array == 1).sum()
    n_neg = (y_array == 0).sum()
    scale_pos_weight = min(n_neg / max(n_pos, 1), 15.0)  # 상한 15

    logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f} "
                f"(pos={n_pos}, neg={n_neg})")

    self._model = xgb.XGBClassifier(
        n_estimators=self.n_estimators,
        max_depth=self.max_depth,
        learning_rate=self.learning_rate,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
        scale_pos_weight=scale_pos_weight,  # 핵심 추가!
        random_state=42
    )
    # ... 나머지 코드 동일
```

---

### 3.2 LightGBM 수정

**파일**: `src/models/lightgbm_model.py`

**현재 코드 (73-80행)**:
```python
self._model = lgb.LGBMClassifier(
    n_estimators=self.n_estimators,
    max_depth=self.max_depth,
    learning_rate=self.learning_rate,
    random_state=42,
    verbose=-1,
    **gpu_params
)
```

**수정 코드**:
```python
def train(self, X, y, X_val=None, y_val=None):
    """Train the LightGBM model."""
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is not installed")

    # GPU 사용 시도
    use_gpu = settings.USE_GPU and HAS_CUDA

    # 클래스 가중치 계산
    y_array = np.asarray(y)
    n_pos = (y_array == 1).sum()
    n_neg = (y_array == 0).sum()
    scale_pos_weight = min(n_neg / max(n_pos, 1), 15.0)

    logger.info(f"LightGBM scale_pos_weight: {scale_pos_weight:.2f}")

    gpu_params = {}
    if use_gpu:
        gpu_params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }

    try:
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,  # 핵심 추가!
            random_state=42,
            verbose=-1,
            **gpu_params
        )
        # ... 나머지 코드 동일
```

---

### 3.3 PyTorch 모델 수정 (LSTM/Transformer)

**파일**: `src/models/lstm_model.py`, `src/models/transformer_model.py`

**수정 방법**:
```python
import torch
import torch.nn as nn

def train(self, X, y, X_val=None, y_val=None):
    # 클래스 가중치 계산
    y_array = np.asarray(y)
    n_pos = (y_array == 1).sum()
    n_neg = (y_array == 0).sum()
    pos_weight_value = min(n_neg / max(n_pos, 1), 15.0)

    # pos_weight를 텐서로 변환
    pos_weight = torch.tensor([pos_weight_value]).to(self.device)

    # 손실 함수에 pos_weight 적용
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ... 학습 루프에서 criterion 사용
```

---

## 4. 임계값 튜닝 가이드

### 4.1 Precision-Recall 트레이드오프

```
Threshold ↓  → Signal Rate ↑, Precision ↓
Threshold ↑  → Signal Rate ↓, Precision ↑
```

### 4.2 최적 임계값 탐색 함수

**파일**: 새로 추가 `src/utils/threshold_optimizer.py`

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_precision: float = 0.60,
    min_signal_rate: float = 0.10
) -> float:
    """
    최소 Precision과 Signal Rate를 만족하는 최적 임계값 탐색.

    Args:
        y_true: 실제 레이블
        y_pred_proba: 예측 확률
        min_precision: 최소 요구 Precision
        min_signal_rate: 최소 요구 Signal Rate

    Returns:
        최적 임계값 (기본값: 0.5)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    best_threshold = 0.5
    best_f1 = 0.0

    for i, thresh in enumerate(thresholds):
        # Signal Rate 계산
        signal_rate = (y_pred_proba >= thresh).mean()

        # 조건 충족 여부 확인
        if precisions[i] >= min_precision and signal_rate >= min_signal_rate:
            # F1 Score 계산
            if precisions[i] + recalls[i] > 0:
                f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

    return best_threshold


def evaluate_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> dict:
    """
    여러 임계값에 대한 지표 비교.
    """
    results = {}

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        signal_rate = y_pred.mean()

        results[thresh] = {
            'precision': precision,
            'recall': recall,
            'signal_rate': signal_rate,
            'signals_generated': y_pred.sum()
        }

    return results
```

### 4.3 권장 임계값

| 상황 | 권장 임계값 | 예상 결과 |
|------|-----------|----------|
| 보수적 (High Precision) | 0.6~0.7 | Precision ↑, Signal ↓ |
| 균형 (Balanced) | 0.4~0.5 | 균형 잡힌 성능 |
| 공격적 (High Recall) | 0.3~0.4 | Signal ↑, Precision ↓ |

**초기 권장값**: `0.40` (균형점 탐색 후 조정)

---

## 5. settings.py 추가 설정

**파일**: `config/settings.py`

```python
# === Signal Rate 개선 설정 ===

# 클래스 가중치 최대값 (과적합 방지)
MAX_CLASS_WEIGHT: float = 15.0

# 신호 생성 임계값 (기존 PROBABILITY_THRESHOLD와 별도)
SIGNAL_THRESHOLD: float = 0.40

# 최소 요구 지표
MIN_PRECISION: float = 0.60
MIN_SIGNAL_RATE: float = 0.10

# 임계값 자동 조정 활성화
AUTO_THRESHOLD_TUNING: bool = True
```

---

## 6. 재훈련 절차

### 6.1 단계별 가이드

```bash
# Step 1: 기존 모델 백업
cp -r models/ models_backup_$(date +%Y%m%d)/

# Step 2: 코드 수정 적용
# - xgboost_model.py 수정
# - lightgbm_model.py 수정
# - settings.py에 설정 추가

# Step 3: 단일 티커로 테스트
python scripts/train_models.py --ticker AAPL --force

# Step 4: Signal Rate 검증
python scripts/check_stage_accuracy.py --ticker AAPL

# Step 5: 전체 티커 재훈련 (검증 후)
python scripts/train_models.py --all --force
```

### 6.2 검증 스크립트

**신규 생성 권장**: `scripts/check_signal_rate.py`

```python
#!/usr/bin/env python
"""Signal Rate 및 핵심 지표 검증 스크립트."""

import argparse
import numpy as np
from pathlib import Path

from src.models.model_manager import ModelManager
from src.data.data_loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


def check_signal_rate(ticker: str, threshold: float = 0.40):
    """모델의 Signal Rate와 핵심 지표 확인."""

    model_manager = ModelManager()
    data_loader = DataLoader()

    # 데이터 로드
    X, y = data_loader.load_test_data(ticker)

    results = {}

    for model_type in ['xgboost', 'lightgbm']:
        for target in ['up', 'down']:
            try:
                _, model = model_manager.get_model(ticker, model_type, target)
                y_proba = model.predict_proba(X)
                y_pred = (y_proba >= threshold).astype(int)
                y_true = y[f'label_{target}'].values

                signal_rate = y_pred.mean() * 100
                precision = precision_score(y_true, y_pred, zero_division=0) * 100
                recall = recall_score(y_true, y_pred, zero_division=0) * 100
                f1 = f1_score(y_true, y_pred, zero_division=0) * 100

                results[f'{model_type}_{target}'] = {
                    'signal_rate': signal_rate,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'signals': y_pred.sum(),
                    'total': len(y_pred)
                }

                status = "✅" if signal_rate >= 10 and precision >= 60 else "❌"

                print(f"\n{status} {model_type.upper()} ({target})")
                print(f"   Signal Rate: {signal_rate:.1f}%")
                print(f"   Precision:   {precision:.1f}%")
                print(f"   Recall:      {recall:.1f}%")
                print(f"   F1 Score:    {f1:.1f}%")
                print(f"   Signals:     {y_pred.sum()} / {len(y_pred)}")

            except Exception as e:
                print(f"❌ {model_type}_{target}: {e}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.40)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Signal Rate Check: {args.ticker}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*50}")

    check_signal_rate(args.ticker, args.threshold)
```

---

## 7. 검증 체크리스트

### 7.1 코드 수정 확인

- [ ] `xgboost_model.py`에 `scale_pos_weight` 추가
- [ ] `lightgbm_model.py`에 `scale_pos_weight` 추가
- [ ] `settings.py`에 새 설정 추가
- [ ] `threshold_optimizer.py` 유틸리티 추가

### 7.2 재훈련 후 지표 확인

- [ ] Signal Rate > 10%
- [ ] Precision > 60%
- [ ] Recall > 20%
- [ ] F1 Score > 30%

### 7.3 백테스트 검증

- [ ] 재훈련 모델로 백테스트 실행
- [ ] 수익률이 이전보다 개선되었는지 확인
- [ ] Sharpe Ratio가 양수인지 확인

---

## 8. 예상 결과

### 8.1 Before (현재)

```
Model: XGBoost (up)
Signal Rate: 0.1%
Precision:   0.0%
Signals:     12 / 12000
```

### 8.2 After (수정 후 예상)

```
Model: XGBoost (up)
Signal Rate: 15.2%
Precision:   62.5%
Signals:     1824 / 12000
```

---

## 9. 주의사항

### 9.1 과적합 방지
- `scale_pos_weight`를 15.0 이상으로 설정하지 않기
- 검증 데이터에서 성능 반드시 확인

### 9.2 점진적 적용
1. 먼저 AAPL 단일 티커로 테스트
2. 지표 확인 후 전체 티커 적용

### 9.3 백테스트 필수
- Signal Rate가 높아졌다고 수익이 보장되지 않음
- 반드시 백테스트로 실제 성과 확인

---

## 10. 요약

### 핵심 수정 3가지

1. **XGBoost/LightGBM**: `scale_pos_weight` 파라미터 추가
2. **임계값**: 0.5 → 0.40으로 조정
3. **설정 파일**: 새 파라미터 추가

### 목표 달성 기준

| 지표 | 목표 | 확인 방법 |
|------|------|----------|
| Signal Rate | >10% | `check_signal_rate.py` |
| Precision | >60% | `check_signal_rate.py` |
| 백테스트 수익 | 양수 | `run_backtest.py` |

---

*분석팀장 작성*
