# 예측 정확도 개선 가이드

본 문서는 FiveForFree 시스템의 예측 정확도 (~50%)를 65%까지 개선하기 위한 분석 및 로드맵입니다.

---

## 1. 현재 상태 요약

| 지표 | 현재 값 | 문제점 |
|------|---------|--------|
| 예측 정확도 | ~50% | 랜덤 수준 |
| 타겟 | 60분 내 1% | 노이즈의 3~4배, 너무 어려움 |
| 학습 샘플 | ~900개/모델 | 심각하게 부족 |
| 피처 수 | 49개 (실효 33개) | 16개 제외, 8개 제로 |

---

## 2. 핵심 문제점 분석

### 2.1 타겟 정의 문제 (치명적)

**현재 설정:**
```python
TARGET_PERCENT = 1.0          # 1% 움직임
PREDICTION_HORIZON_MINUTES = 60  # 60분 내
```

**왜 문제인가:**
- NASDAQ 주식의 60분 내 랜덤워크 기대값: ~0.17~0.25%
- 1% 타겟 = 노이즈의 **3~4배** 움직임 예측 요구
- 결과: 레이블의 60~70%가 "neither" (상승도 하락도 아님)
- 모델이 학습할 신호가 너무 약함

**수학적 근거:**
```
일간 변동성 (typical): 1.5~2%
60분 = 1/6.5 거래일
60분 기대 변동성: 2% × √(1/6.5) ≈ 0.78%
1% 타겟 = 1.28σ 이상의 움직임 (상위 10% 이벤트)
```

### 2.2 학습 데이터 부족 (치명적)

**현재:**
- 60일 히스토리 × 78 bars/일 = ~4,680 bars
- Train 80% = ~3,744 샘플
- 5개 모델 × 2 타겟 = 10개 모델
- **모델당 유효 샘플: ~374개** (너무 적음)

**필요량:**
- Tree 모델: 최소 1,000~5,000 샘플
- Neural 모델: 최소 10,000+ 샘플
- **권장: 180일+ 데이터 (30,000+ 샘플)**

### 2.3 피처 품질 문제 (높음)

**Sign-Flip 피처 (16개 제외됨):**
```python
EXCLUDED_FEATURES = {
    'returns_1m',      # Train: -0.008, Val: +0.022 (방향 반전)
    'returns_5m',      # Train: -0.003, Val: +0.034
    'returns_15m', 'returns_30m', 'returns_60m',
    'price_vs_ma_5', 'price_vs_ma_15', 'price_vs_ma_60',
    'rsi_14', 'stoch_k', 'stoch_d', 'cci_14', 'mfi_14',
    'bb_position', 'obv_normalized', 'is_option_expiry', 'vpt_cumsum'
}
```

**의미:**
- 학습 데이터와 검증 데이터에서 상관관계 방향이 반전
- 과적합 또는 데이터 누수 징후
- 제외해도 근본 원인 미해결

**제로 피처 (8개):**
```python
# Order Book 피처 - Level 2 데이터 없어서 항상 0
'bid_ask_spread', 'spread_pct', 'bid_depth', 'ask_depth',
'depth_imbalance', 'weighted_mid_price', 'order_imbalance', 'book_pressure'
```

### 2.4 모델 설정 문제 (높음)

**Early Stopping 미적용:**
```python
# XGBoost - early_stopping_rounds 없음
xgb.train(params, dtrain)  # 과적합 제어 안됨

# LightGBM - 동일
lgb.train(params, train_set)
```

**Neural Network 스케일링 문제:**
```python
# 피처 범위가 극단적
np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
# -1,000,000 ~ +1,000,000 범위 → 신경망 학습 불안정
```

### 2.5 클래스 불균형 (높음)

**예상 레이블 분포:**
```
label_up = 1    : ~15-20%
label_down = 1  : ~15-20%
neither         : ~60-70%  ← 다수 클래스
```

**현재 처리:**
- 단순 언더샘플링/오버샘플링
- SMOTE, Cost-sensitive learning 미적용
- 임계값 최적화 없음 (현재 10%로 하드코딩)

---

## 3. 개선 로드맵

### Phase 1: 즉시 적용 (1~2일)

#### 3.1.1 타겟 난이도 조정

```python
# config/settings.py
# Before
TARGET_PERCENT = 1.0
PREDICTION_HORIZON_MINUTES = 60

# After (Option A - 권장)
TARGET_PERCENT = 0.5
PREDICTION_HORIZON_MINUTES = 60

# After (Option B - 더 쉬움)
TARGET_PERCENT = 0.5
PREDICTION_HORIZON_MINUTES = 30

# After (Option C - 가장 쉬움)
TARGET_PERCENT = 0.3
PREDICTION_HORIZON_MINUTES = 60
```

**예상 효과:** +5~8% 정확도

#### 3.1.2 Early Stopping 추가

```python
# src/models/xgboost_model.py
def train(self, X, y, X_val=None, y_val=None):
    if X_val is not None:
        evallist = [(dval, 'eval')]
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            evals=evallist,
            early_stopping_rounds=50,  # 추가
            verbose_eval=False
        )

# src/models/lightgbm_model.py
callbacks = [
    lgb.early_stopping(stopping_rounds=50),  # 추가
    lgb.log_evaluation(period=0)
]
```

**예상 효과:** +3~5% 정확도

#### 3.1.3 제로 피처 제거

```python
# src/processor/feature_engineer.py
# Order Book 피처 8개를 DataFrame에서 제거
ZERO_FEATURES_TO_DROP = [
    'bid_ask_spread', 'spread_pct', 'bid_depth', 'ask_depth',
    'depth_imbalance', 'weighted_mid_price', 'order_imbalance', 'book_pressure'
]

def compute_features(self, ...):
    # ... 기존 로직 ...
    df = df.drop(columns=ZERO_FEATURES_TO_DROP, errors='ignore')
```

**예상 효과:** 노이즈 감소, 학습 안정성 향상

### Phase 2: 단기 개선 (1주일)

#### 3.2.1 데이터 축적

```bash
# 매일 실행하여 데이터 축적
python scripts/collect_historical.py --update

# 목표: 180일 (현재 60일 → 3배)
# 예상 샘플: 30,000+
```

#### 3.2.2 피처 안정성 분석

```python
# scripts/analyze_feature_stability.py (신규 작성 필요)
def analyze_stability(df, feature_cols, n_splits=5):
    """
    시간순 5-fold로 피처별 타겟 상관관계 안정성 분석
    Sign-flip 발생 피처 식별
    """
    for fold in time_series_splits:
        train_corr = df_train[feature].corr(df_train['label_up'])
        val_corr = df_val[feature].corr(df_val['label_up'])
        if sign(train_corr) != sign(val_corr):
            print(f"Sign flip detected: {feature}")
```

#### 3.2.3 클래스 균형 개선

```python
# Option A: SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Option B: Cost-sensitive Learning
# XGBoost
scale_pos_weight = len(y[y==0]) / len(y[y==1])
params['scale_pos_weight'] = scale_pos_weight

# Option C: Threshold 최적화
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
# F1 최대화 threshold 선택
```

### Phase 3: 중기 개선 (2~4주)

#### 3.3.1 피처 엔지니어링 강화

**추가 권장 피처:**
```python
# 시장 레짐 피처
'volatility_regime'      # 고변동/저변동 상태
'trend_strength'         # ADX 기반 추세 강도
'market_breadth'         # 상승/하락 종목 비율

# 상대 강도 피처
'sector_relative_strength'  # 섹터 대비 상대 성과
'spy_beta_rolling'          # SPY 대비 베타 (롤링)

# 거래량 이상 피처
'volume_zscore'          # 거래량 z-score
'volume_breakout'        # 평균 대비 급증 여부
```

#### 3.3.2 예측 대상 재정의

```python
# 현재: 방향 예측 (어려움)
label_up = price_change >= +1%
label_down = price_change <= -1%

# 대안: 변동성 예측 (더 쉬움)
label_volatile = abs(price_change) >= 1%  # 먼저 움직임 여부 예측
label_direction = price_change > 0        # 움직임 있을 때만 방향 예측
```

**장점:**
- 변동성 예측은 방향 예측보다 쉬움
- 2단계 예측으로 정확도 향상 가능

#### 3.3.3 앙상블 전략 개선

```python
# 현재: 단순 평균
ensemble_prob = (xgb_prob + lgb_prob + lstm_prob + transformer_prob) / 4

# 개선: 가중 평균 (검증 성능 기반)
weights = softmax([xgb_acc, lgb_acc, lstm_acc, transformer_acc])
ensemble_prob = sum(w * p for w, p in zip(weights, probs))

# 고급: 시장 상황별 모델 선택
if market_volatility > threshold:
    use_model = 'lstm'  # 고변동성에서 강함
else:
    use_model = 'xgboost'  # 저변동성에서 강함
```

---

## 4. 예상 효과 및 우선순위

| 개선 항목 | 예상 효과 | 난이도 | 우선순위 |
|----------|----------|--------|----------|
| 타겟 0.5%로 조정 | +5~8% | ⭐ | **1** |
| Early Stopping | +3~5% | ⭐ | **2** |
| 제로 피처 제거 | +1~2% | ⭐ | **3** |
| 데이터 3배 축적 | +5~10% | ⏳ | **4** |
| SMOTE 적용 | +2~3% | ⭐⭐ | 5 |
| 피처 안정성 분석 | +2~4% | ⭐⭐ | 6 |
| 2단계 예측 | +3~5% | ⭐⭐⭐ | 7 |
| 앙상블 최적화 | +1~3% | ⭐⭐ | 8 |

**누적 예상:**
- Phase 1 완료: 55~58%
- Phase 2 완료: 60~63%
- Phase 3 완료: 63~67%

---

## 5. 설정 변경 빠른 참조

### 5.1 보수적 변경 (안전)

```python
# config/settings.py
TARGET_PERCENT = 0.5                    # 1.0 → 0.5
PREDICTION_HORIZON_MINUTES = 60         # 유지
PROBABILITY_THRESHOLD = 0.55            # 0.10 → 0.55
```

### 5.2 적극적 변경 (실험적)

```python
# config/settings.py
TARGET_PERCENT = 0.3                    # 더 쉬운 타겟
PREDICTION_HORIZON_MINUTES = 30         # 더 짧은 예측
PROBABILITY_THRESHOLD = 0.60            # 더 높은 신뢰도
```

---

## 6. 모니터링 지표

개선 작업 후 추적해야 할 지표:

```python
# 필수 지표
accuracy = (TP + TN) / Total           # 목표: 65%+
precision = TP / (TP + FP)             # 목표: 60%+
recall = TP / (TP + FN)                # 목표: 50%+
f1_score = 2 * (P * R) / (P + R)       # 목표: 0.55+

# 거래 지표
signal_rate = signals / total_bars     # 목표: 10~20%
profit_factor = gross_profit / gross_loss  # 목표: 1.3+
sharpe_ratio                           # 목표: 1.0+
```

---

## 7. 참고 자료

### 7.1 관련 문서

- `docs/FEATURES_REFERENCE.md` - 피처 상세 정의
- `docs/HYBRID_ENSEMBLE_ARCHITECTURE.md` - 앙상블 구조
- `Docs/MODEL_PERFORMANCE_ANALYSIS.md` - 현재 모델 성능
- `Docs/FEATURE_ENGINEERING_IMPROVEMENT.md` - 피처 개선 아이디어

### 7.2 외부 참고

- [Walk Forward Validation](https://medium.com/@ahmedfahad04/understanding-walk-forward-validation-in-time-series-analysis-a-practical-guide-ea3814015abf)
- [Purged K-Fold CV in Finance](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- [RiskLab AI - Financial CV](https://www.risklab.ai/research/financial-modeling/cross_validation)

---

**Last Updated**: 2024-12-22
**Version**: 1.0
