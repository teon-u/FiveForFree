# 피처 엔지니어링 개선안

**작성일**: 2025-12-21
**작성자**: 분석팀장
**문서 버전**: v1.0

---

## 1. 현재 피처 구성 분석

### 1.1 기존 피처 카테고리 (49개)

| 카테고리 | 개수 | 주요 피처 |
|----------|------|-----------|
| 가격 기반 | 15 | returns_1m/5m/15m/30m/60m, MA 크로스 |
| 변동성 | 10 | ATR, BB position/width, volatility |
| 거래량 | 8 | volume_ratio, OBV, money_flow, MFI |
| 호가창 | 0 | (미구현) |
| 모멘텀 | 8 | RSI, MACD, Stochastic, CCI |
| 시장 맥락 | 5 | SPY/QQQ 수익률, VIX |
| 시간 기반 | 3 | minutes_since_open, day_of_week |

### 1.2 코드 품질 분석

**강점**:
- TA-Lib 미설치 환경 대응 (pandas fallback)
- LRU 캐시 활용
- 로깅 적용

**개선점**:
- 호가창 피처 미구현 (0개)
- 피처 중요도 분석 부재
- 피처 스케일링 일관성 부족

---

## 2. 피처 개선 제안

### 2.1 호가창 피처 추가 (8개 신규)

현재 호가창 피처가 0개입니다. 다음 피처들을 추가할 것을 권장합니다:

```python
# 추가 제안 피처
order_book_features = {
    # 기본 스프레드
    "bid_ask_spread": (best_ask - best_bid) / best_bid * 100,
    "spread_percent": (best_ask - best_bid) / mid_price * 100,

    # 호가 불균형
    "bid_volume_total": sum(bid_volumes[:10]),
    "ask_volume_total": sum(ask_volumes[:10]),
    "order_imbalance": (bid_total - ask_total) / (bid_total + ask_total),

    # 가중 중간가
    "vwap_orderbook": (bid_price * ask_vol + ask_price * bid_vol) / (bid_vol + ask_vol),

    # 깊이 지표
    "depth_ratio": bid_total / ask_total,
    "pressure_index": bid_volume_1 / ask_volume_1,
}
```

### 2.2 고급 기술 지표 추가 (5개 신규)

```python
# 추가 제안 피처
advanced_features = {
    # Ichimoku Cloud
    "ichimoku_tenkan": (highest_high_9 + lowest_low_9) / 2,
    "ichimoku_kijun": (highest_high_26 + lowest_low_26) / 2,

    # Keltner Channel
    "keltner_upper": ema_20 + 2 * atr_10,
    "keltner_lower": ema_20 - 2 * atr_10,
    "keltner_position": (close - keltner_lower) / (keltner_upper - keltner_lower),
}
```

### 2.3 시장 마이크로스트럭처 피처 (4개 신규)

```python
# 체결 데이터 기반 피처
microstructure_features = {
    # 체결 불균형
    "trade_imbalance": (buy_volume - sell_volume) / total_volume,

    # 거래 빈도
    "trade_frequency": trades_per_minute,

    # 평균 체결 크기
    "avg_trade_size": total_volume / trade_count,

    # 가격 충격
    "price_impact": price_change / volume_traded,
}
```

---

## 3. 피처 엔지니어링 파이프라인 개선

### 3.1 피처 스케일링 표준화

**현재 문제점**: 피처별 스케일 불일치

**제안**:
```python
class FeatureScaler:
    SCALING_CONFIG = {
        # 수익률 계열: 이미 %로 표준화
        "returns_*": "none",

        # 지표 계열: MinMax (0-1)
        "rsi_*": "minmax",
        "mfi_*": "minmax",

        # 가격/거래량: Z-score
        "volume_*": "zscore",
        "price_*": "zscore",

        # 비율 계열: 로그 변환
        "ratio_*": "log",
    }

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        for pattern, method in self.SCALING_CONFIG.items():
            cols = [c for c in features.columns if fnmatch(c, pattern)]
            if method == "minmax":
                features[cols] = MinMaxScaler().fit_transform(features[cols])
            elif method == "zscore":
                features[cols] = StandardScaler().fit_transform(features[cols])
            elif method == "log":
                features[cols] = np.log1p(features[cols])
        return features
```

### 3.2 피처 중요도 분석 자동화

```python
class FeatureImportanceAnalyzer:
    def analyze(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # 1. 상관관계 분석
        correlations = X.corrwith(y).abs()

        # 2. 트리 기반 중요도
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        tree_importance = pd.Series(rf.feature_importances_, index=X.columns)

        # 3. SHAP 값 (선택적)
        # shap_values = shap.TreeExplainer(rf).shap_values(X)

        return pd.DataFrame({
            "correlation": correlations,
            "tree_importance": tree_importance,
        }).sort_values("tree_importance", ascending=False)
```

### 3.3 피처 선택 자동화

```python
class AutoFeatureSelector:
    def __init__(self, min_importance: float = 0.01):
        self.min_importance = min_importance
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.analyze(X, y)

        # 중요도 기준 필터링
        self.selected_features = importance[
            importance["tree_importance"] >= self.min_importance
        ].index.tolist()

        logger.info(f"Selected {len(self.selected_features)}/{len(X.columns)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features]
```

---

## 4. 피처 검증 체크리스트

### 4.1 데이터 품질 검증

- [ ] 결측값 비율 < 5%
- [ ] 이상치 비율 < 1%
- [ ] 피처 간 상관관계 < 0.95 (다중공선성 방지)

### 4.2 정보 누수 검증

- [ ] 미래 데이터 참조 없음 확인
- [ ] 레이블과 동시점 데이터 제외
- [ ] 시계열 split 사용 검증

### 4.3 성능 검증

- [ ] 피처 계산 시간 < 100ms/종목
- [ ] 메모리 사용량 모니터링
- [ ] GPU 가속 활용 여부

---

## 5. 권장 피처 최종 구성 (62개)

| 카테고리 | 기존 | 추가 | 합계 |
|----------|------|------|------|
| 가격 기반 | 15 | 0 | 15 |
| 변동성 | 10 | 0 | 10 |
| 거래량 | 8 | 0 | 8 |
| **호가창** | **0** | **8** | **8** |
| 모멘텀 | 8 | 5 | 13 |
| 시장 맥락 | 5 | 0 | 5 |
| 시간 기반 | 3 | 0 | 3 |
| **마이크로스트럭처** | **0** | **4** | **4** |
| **합계** | **49** | **17** | **66** |

*참고: 피처 중요도 분석 후 실제 사용 피처는 줄어들 수 있음*

---

## 6. 구현 우선순위

1. **즉시**: 호가창 피처 8개 추가
2. **단기**: 피처 스케일링 표준화
3. **중기**: 피처 중요도 분석 자동화
4. **장기**: 마이크로스트럭처 피처 (체결 데이터 필요)

---

*이 문서는 분석팀장이 작성하였습니다.*
