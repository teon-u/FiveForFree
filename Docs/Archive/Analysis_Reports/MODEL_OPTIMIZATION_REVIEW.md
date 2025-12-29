# 모델 성능 최적화 검토 보고서

**작성자**: 개발팀장
**작성일**: 2025-12-21
**상태**: 검토 완료

---

## 1. 현재 모델 설정 분석

### 1.1 Tree-based Models (XGBoost, LightGBM)

| 파라미터 | 현재값 | 평가 |
|---------|-------|------|
| n_estimators | 100 | 보수적 - 증가 가능 |
| max_depth | 6 | 적정 |
| learning_rate | 0.1 | 적정 |
| early_stopping | 없음 | **개선 필요** |

### 1.2 Neural Models (LSTM, Transformer)

| 파라미터 | LSTM | Transformer | 평가 |
|---------|------|-------------|------|
| hidden_size/d_model | 64 | 64 | 적정 |
| num_layers | 2 | 2 | 적정 |
| dropout | 0.2 | 0.1 | 적정 |
| epochs | 50 | 50 | early_stopping 필요 |
| batch_size | 32 | 32 | GPU 활용도 향상 가능 |
| sequence_length | 60 | 60 | 1시간 예측에 적합 |
| learning_rate | 0.001 | 0.001 | 적정 |

### 1.3 Ensemble Model

| 전략 | 가중치 | 평가 |
|------|--------|------|
| precision_weighted | 0.4 | 정확도 기반 적정 |
| stacking | 0.4 | XGBoost meta-learner 사용 |
| dynamic_selection | 0.2 | 최고 성능 모델 선택 |

---

## 2. 긍정적 요소 (유지 권장)

### 2.1 수치 안정성
- BCEWithLogitsLoss 사용 (Sigmoid + BCE 결합)
- Gradient clipping (max_norm=1.0)
- StandardScaler로 입력 정규화
- NaN/Inf 처리 (`np.nan_to_num`)

### 2.2 GPU 최적화
- XGBoost: `tree_method='hist'` (GPU 친화적)
- PyTorch: CUDA 자동 감지
- 병렬 학습: ThreadPoolExecutor 활용

### 2.3 시계열 처리
- 시간순 데이터 분할 (셔플 없음)
- 시퀀스 패딩 전략 (첫 값 반복)

---

## 3. 개선 권장 사항

### 3.1 Priority 1: Early Stopping 추가

**문제**: 과적합 방지 메커니즘 부재

**XGBoost 개선 예시**:
```python
self._model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,  # 추가
    verbose=False
)
```

**LightGBM 개선 예시**:
```python
callbacks = [lgb.early_stopping(10)] if eval_set else None
self._model.fit(
    X_train, y_train,
    eval_set=eval_set,
    callbacks=callbacks
)
```

### 3.2 Priority 2: Learning Rate Scheduler

**Neural Models 개선 예시**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Training loop에서 scheduler.step(val_loss) 호출
```

### 3.3 Priority 3: Batch Size 증가 (GPU 활용도)

| 모델 | 현재 | 권장 | 이유 |
|------|------|------|------|
| LSTM | 32 | 64-128 | GPU 메모리 활용 |
| Transformer | 32 | 64-128 | 병렬 처리 효율 |

### 3.4 Priority 4: Validation 모니터링

현재 validation 데이터가 전달되어도 loss 추적이 안됨.

**개선 방향**:
- Epoch별 validation loss 기록
- Best model checkpoint 저장
- TensorBoard 연동 고려

---

## 4. 즉시 적용 불필요 (현재 적정)

1. **모델 아키텍처**: LSTM 2층, Transformer 2층은 시계열 예측에 충분
2. **Dropout 비율**: 0.1~0.2 범위 적정
3. **Sequence Length**: 60분 데이터로 1시간 예측은 적합
4. **Ensemble 가중치**: 정확도 기반 동적 가중치 적용 중

---

## 5. 결론

| 우선순위 | 개선 항목 | 예상 효과 | 구현 난이도 |
|---------|----------|----------|------------|
| 1 | Early Stopping | 과적합 방지, 학습 시간 단축 | 낮음 |
| 2 | LR Scheduler | 수렴 속도 향상 | 낮음 |
| 3 | Batch Size 증가 | GPU 활용도 향상 | 매우 낮음 |
| 4 | Validation 모니터링 | 성능 추적 개선 | 중간 |

**현재 모델 설정은 전반적으로 양호함.**
즉각적인 성능 문제는 없으나, 위 개선사항 적용 시 학습 안정성과 효율성 향상 기대.

---

## 6. 참고: 수정 완료 항목

- [x] Pydantic V2 마이그레이션 (settings.py)
- [x] sklearn feature names 경고 처리 (xgboost_model.py, lightgbm_model.py, ensemble_model.py)
