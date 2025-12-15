# NASDAQ 단기 변동 예측 시스템

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 1시간 내 5% 이상 상승/하락 확률 예측 → 수동 매매 의사결정 지원 |
| **대상** | NASDAQ 고변동성 종목 (거래량/상승률 Top 100) |
| **데이터** | Polygon.io Developer Plan ($79/월) - 1분봉 + Level 2 호가 |
| **예측** | 상승 확률 / 하락 확률 각각 출력 |
| **청산** | 5% 도달 OR 1시간 후 무조건 청산 |
| **매매** | Long Only, 1종목 집중, 수동 매매 |

## 하드웨어 환경

- GPU: NVIDIA RTX 5080
- CPU: AMD Ryzen 9800X3D
- RAM: 64GB
- 병렬 처리 및 GPU 가속 필수

---

## 디렉토리 구조

```
nasdaq-predictor/
├── config/
│   ├── settings.py          # 전역 설정
│   └── .env                  # API 키
├── src/
│   ├── collector/            # 데이터 수집
│   │   ├── polygon_client.py
│   │   ├── ticker_selector.py
│   │   ├── minute_bars.py
│   │   ├── quotes.py
│   │   └── market_context.py
│   ├── processor/            # Feature Engineering
│   │   ├── feature_engineer.py
│   │   └── label_generator.py
│   ├── models/               # 5개 모델
│   │   ├── base_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   ├── ensemble_model.py
│   │   └── model_manager.py
│   ├── trainer/              # GPU 병렬 학습
│   │   ├── gpu_trainer.py
│   │   └── incremental.py
│   ├── predictor/
│   │   └── realtime_predictor.py
│   ├── backtester/
│   │   ├── simulator.py
│   │   └── metrics.py
│   ├── api/                  # FastAPI
│   │   ├── main.py
│   │   ├── routes/
│   │   └── websocket.py
│   └── utils/
│       ├── database.py
│       └── logger.py
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/{ticker}/
├── logs/
├── scripts/
│   ├── run_system.py
│   ├── collect_historical.py
│   └── train_all_models.py
└── requirements.txt
```

---

## 1. 데이터 수집 모듈

### 1.1 Polygon.io 설정

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    POLYGON_API_KEY: str
    
    # 수집 설정
    TOP_N_VOLUME: int = 100
    TOP_N_GAINERS: int = 100
    HISTORICAL_DAYS: int = 30
    
    # 예측 설정
    PREDICTION_HORIZON_MINUTES: int = 60
    TARGET_PERCENT: float = 5.0
    PROBABILITY_THRESHOLD: float = 0.70
    
    # 백테스팅
    BACKTEST_HOURS: int = 50
    
    # GPU
    USE_GPU: bool = True
    
    class Config:
        env_file = ".env"
```

### 1.2 수집 대상 및 주기

```
[매시간] 타겟 종목 선정
├── 거래량 상위 100개
├── 상승률 상위 100개
└── 중복 제거 → 약 150~180개

[1분마다] 각 종목별 수집
├── OHLCV (1분봉)
├── VWAP
├── Level 2 호가 (bid/ask volume, spread, imbalance)
└── 체결 데이터

[1시간마다] 시장 맥락
├── SPY, QQQ 수익률
├── VIX 수준
└── 섹터 ETF 11개
```

### 1.3 핵심 수집 코드

```python
# src/collector/ticker_selector.py
class TickerSelector:
    def get_target_tickers(self) -> list[str]:
        """거래량 Top 100 + 상승률 Top 100 합집합"""
        snapshot = self.client.get_snapshot_all("stocks")
        
        volume_sorted = sorted(snapshot, key=lambda x: x.day.volume, reverse=True)[:100]
        gainer_sorted = sorted(snapshot, key=lambda x: x.todaysChangePerc, reverse=True)[:100]
        
        return list(set([t.ticker for t in volume_sorted + gainer_sorted]))

# src/collector/quotes.py
class QuoteCollector:
    def get_order_book_snapshot(self, ticker: str) -> dict:
        """Level 2 호가창 데이터"""
        book = self.client.get_snapshot_ticker("stocks", ticker)
        
        bids = [(b.price, b.size) for b in book.book.bids[:10]]
        asks = [(a.price, a.size) for a in book.book.asks[:10]]
        
        bid_total = sum(size for _, size in bids)
        ask_total = sum(size for _, size in asks)
        imbalance = (bid_total - ask_total) / (bid_total + ask_total)
        
        return {
            "bids": bids, "asks": asks,
            "bid_total_volume": bid_total,
            "ask_total_volume": ask_total,
            "imbalance": imbalance,
            "spread": asks[0][0] - bids[0][0] if bids and asks else 0
        }
```

---

## 2. Feature Engineering (57개)

### 2.1 Feature 카테고리

| 카테고리 | 개수 | 주요 Feature |
|----------|------|--------------|
| 가격 기반 | 15 | returns_1m/5m/15m/30m/60m, ma_5/15/60, ma_cross, price_vs_vwap |
| 변동성 기반 | 10 | atr_14, bb_position/width, volatility_5m/15m/60m, price_acceleration |
| 거래량 기반 | 8 | volume_ratio, obv, money_flow, mfi_14 |
| 호가창 기반 | 8 | bid_ask_spread, imbalance, depth_weighted_mid_price |
| 모멘텀 기반 | 8 | rsi_14, macd/signal/hist, stoch_k/d, williams_r, cci_14 |
| 시장 맥락 | 5 | spy_return, qqq_return, vix_level, sector_etf_return |
| 시간 기반 | 3 | minutes_since_open, day_of_week, is_option_expiry |

### 2.2 라벨 생성

```python
# src/processor/label_generator.py
class LabelGenerator:
    def generate_labels(self, minute_bars, entry_time, entry_price) -> dict:
        """
        1시간 후까지의 라벨 생성
        - label_up: 5% 이상 상승 여부
        - label_down: 5% 이상 하락 여부
        """
        future_bars = minute_bars[
            (minute_bars["timestamp"] > entry_time) &
            (minute_bars["timestamp"] <= entry_time + timedelta(minutes=60))
        ]
        
        max_gain = ((future_bars["high"] - entry_price) / entry_price * 100).max()
        max_loss = ((future_bars["low"] - entry_price) / entry_price * 100).min()
        
        return {
            "label_up": max_gain >= 5.0,
            "label_down": max_loss <= -5.0,
            "max_gain": max_gain,
            "max_loss": max_loss
        }
```

---

## 3. 모델링 (종목별 5개 × 상승/하락)

### 3.1 모델 구성

```
종목당 총 10개 모델:
├── 상승 예측 (5개)
│   ├── XGBoost (GPU)
│   ├── LightGBM (GPU)
│   ├── LSTM (PyTorch)
│   ├── Transformer (PyTorch)
│   └── Stacking Ensemble
└── 하락 예측 (5개)
    └── 동일 구조
```

### 3.2 XGBoost 모델 (GPU)

```python
# src/models/xgboost_model.py
class XGBoostModel(BaseModel):
    def __init__(self, ticker: str, target: str = "up"):
        self.params = {
            "objective": "binary:logistic",
            "tree_method": "gpu_hist",  # GPU 가속
            "device": "cuda:0",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }
    
    def train(self, X, y, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)
        self.is_trained = True
    
    def predict_proba(self, X) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def incremental_train(self, X_new, y_new):
        """증분 학습 - 기존 모델 기반 추가 학습"""
        dnew = xgb.DMatrix(X_new, label=y_new)
        self.model = xgb.train(
            self.params, dnew,
            num_boost_round=20,
            xgb_model=self.model  # 기존 모델 기반
        )
```

### 3.3 LSTM 모델 (PyTorch)

```python
# src/models/lstm_model.py
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class LSTMModel(BaseModel):
    def __init__(self, ticker, target="up", seq_length=60):
        self.seq_length = seq_length
        self.device = torch.device("cuda:0")
```

### 3.4 Ensemble 모델

```python
# src/models/ensemble_model.py
class EnsembleModel(BaseModel):
    """
    Stacking Ensemble
    - Base: XGBoost, LightGBM, LSTM, Transformer
    - Meta: Logistic Regression
    """
    def __init__(self, ticker, target="up"):
        self.base_models = [
            XGBoostModel(ticker, target),
            LightGBMModel(ticker, target),
            LSTMModel(ticker, target),
            TransformerModel(ticker, target)
        ]
        self.meta_learner = LogisticRegression()
    
    def predict_proba(self, X):
        meta_features = [m.predict_proba(X) for m in self.base_models]
        return self.meta_learner.predict_proba(np.column_stack(meta_features))[:, 1]
```

### 3.5 모델 매니저 (최고 성능 모델 선택)

```python
# src/models/model_manager.py
class ModelManager:
    def get_best_model(self, ticker: str, target: str = "up") -> tuple[str, BaseModel]:
        """
        최근 50시간 기준 가장 정확한 모델 반환
        UI에 표시되는 예측값은 이 모델 기준
        """
        models = self.get_all_models(ticker, target)
        
        best_type, best_model, best_accuracy = None, None, -1
        for model_type, model in models.items():
            accuracy = model.get_recent_accuracy(hours=50)
            if accuracy > best_accuracy:
                best_type, best_model, best_accuracy = model_type, model, accuracy
        
        return best_type, best_model
```

---

## 4. GPU 병렬 학습

```python
# src/trainer/gpu_trainer.py
class GPUParallelTrainer:
    """
    RTX 5080 활용 병렬 학습
    - Tree 모델: ThreadPoolExecutor (CPU 병렬)
    - Neural 모델: 순차 처리 (GPU 메모리 관리)
    """
    def __init__(self, n_parallel=4):
        self.n_parallel = n_parallel
        self.device = torch.device("cuda:0")
    
    def train_ticker_batch(self, tickers, data_dict, model_manager):
        # Tree 계열 병렬 처리
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            for ticker in tickers:
                executor.submit(self._train_tree_models, ticker, data_dict[ticker])
        
        # Neural 모델 순차 처리 (GPU 메모리)
        for ticker in tickers:
            torch.cuda.empty_cache()
            self._train_neural_models(ticker, data_dict[ticker])
```

---

## 5. 백테스팅

### 5.1 시뮬레이션 규칙

```python
# src/backtester/simulator.py
class BacktestSimulator:
    """
    규칙:
    - Long Only
    - 진입: up_prob >= 70%
    - 청산: 5% 도달 OR 1시간 경과
    - 수수료: 0.1% (왕복 0.2%)
    """
    def simulate_trade(self, ticker, entry_time, entry_price, minute_prices, up_prob):
        if up_prob < self.prob_threshold:
            return None
        
        for _, row in minute_prices.iterrows():
            high_return = (row["high"] - entry_price) / entry_price * 100
            
            if high_return >= 5.0:  # 목표가 도달
                exit_price = entry_price * 1.05
                exit_reason = "target_hit"
                break
            
            if (row["timestamp"] - entry_time).seconds >= 3600:  # 1시간 경과
                exit_price = row["close"]
                exit_reason = "time_limit"
                break
        
        profit = (exit_price - entry_price) / entry_price * 100 - 0.2  # 수수료
        return Trade(ticker, entry_price, exit_price, profit, exit_reason)
```

### 5.2 모델별 50시간 Hit율 (UI 표시용)

```python
def get_model_performances(self, ticker) -> dict:
    """카드 클릭 시 표시할 백테스팅 결과"""
    performances = {"up": {}, "down": {}}
    
    for target in ["up", "down"]:
        for model_type in ["xgboost", "lightgbm", "lstm", "transformer", "ensemble"]:
            model = self.get_or_create_model(ticker, model_type, target)
            performances[target][model_type] = {
                "hit_rate_50h": model.get_recent_accuracy(50) * 100,
                "is_trained": model.is_trained
            }
    
    return performances
```

---

## 6. 웹 시각화

### 6.1 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│  [설정] Threshold: [5]% | 기준확률: [70]% | [상승/하락/전체] │
├─────────────────────────────────────────────────────────────┤
│  📊 거래량 Top 100                              ← 스크롤 →   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ...                            │
│  │ NVDA │ │ TSLA │ │ AMD  │                                │
│  │ 🟢72%│ │ ⚪45%│ │ 🔴68%│  ← 최고 성능 모델 기준 확률    │
│  │+2.3% │ │+0.8% │ │-1.2% │                                │
│  │xgbst │ │ensem │ │lgbm  │  ← 최고 성능 모델명            │
│  └──────┘ └──────┘ └──────┘                                │
├─────────────────────────────────────────────────────────────┤
│  📈 상승률 Top 100                              ← 스크롤 →   │
│  (동일 구조)                                                │
└─────────────────────────────────────────────────────────────┘

[카드 클릭 시 상세 패널]
├── 현재가, 변동률
├── 상승/하락 확률 바
├── 최고 성능 모델 및 Hit율
├── 모델별 50시간 백테스팅 테이블
│   ├── XGBoost: 상승 78%, 하락 65%
│   ├── LightGBM: 상승 72%, 하락 71%
│   ├── LSTM: 상승 68%, 하락 62%
│   ├── Transformer: 상승 70%, 하락 64%
│   └── Ensemble: 상승 75%, 하락 69%
└── 60분 가격 차트
```

### 6.2 카드 색상 규칙

```css
.card-strong-up { border-color: #22c55e; }    /* 80%+ 상승 */
.card-up { border-color: #86efac; }            /* 70-80% 상승 */
.card-strong-down { border-color: #ef4444; }  /* 80%+ 하락 */
.card-down { border-color: #fca5a5; }          /* 70-80% 하락 */
.card-neutral { border-color: #e5e7eb; }       /* 70% 미만 */
```

### 6.3 기술 스택

- Backend: FastAPI + WebSocket
- Frontend: HTML + CSS + Vanilla JS
- 차트: Chart.js
- DB: SQLite

---

## 7. 실행 스크립트

### 7.1 초기 설정

```bash
# 1. 환경 설정
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. .env 파일 생성
echo "POLYGON_API_KEY=your_key_here" > .env

# 3. DB 초기화
python scripts/init_database.py

# 4. 과거 데이터 수집 (30일)
python scripts/collect_historical.py 30

# 5. 초기 모델 학습
python scripts/train_all_models.py

# 6. 시스템 실행
python scripts/run_system.py
```

### 7.2 메인 스케줄러

```python
# src/scheduler.py
class MainScheduler:
    def setup_jobs(self):
        # 매시간: 타겟 종목 업데이트
        self.scheduler.add_job(self.update_target_tickers, CronTrigger(hour='9-15', minute='30'))
        
        # 1분마다: 데이터 수집 + 예측
        self.scheduler.add_job(self.collect_minute_data, IntervalTrigger(minutes=1))
        self.scheduler.add_job(self.run_predictions, IntervalTrigger(minutes=1))
        
        # 매시간: 증분 학습 + 백테스팅 업데이트
        self.scheduler.add_job(self.incremental_training, IntervalTrigger(hours=1))
        self.scheduler.add_job(self.update_backtest_results, IntervalTrigger(hours=1))
        
        # 장 마감 후: 전체 모델 재학습
        self.scheduler.add_job(self.full_training, CronTrigger(hour='17', minute='0'))
```

---

## 8. Requirements

```txt
# Data
polygon-api-client==1.13.4
pandas==2.2.0
numpy==1.26.3
ta-lib==0.4.28

# ML
scikit-learn==1.4.0
xgboost==2.0.3
lightgbm==4.3.0
torch==2.2.0

# API
fastapi==0.109.0
uvicorn==0.27.0
websockets==12.0

# Utils
sqlalchemy==2.0.25
apscheduler==3.10.4
pydantic-settings==2.1.0
loguru==0.7.2
pytz==2024.1
```

---

## 9. 핵심 로직 요약

### 예측 흐름
```
1. 1분마다 데이터 수집 (Polygon.io)
2. 57개 Feature 계산
3. 종목별 5개 모델로 상승/하락 확률 예측
4. 최근 50시간 백테스팅 기준 최고 성능 모델 선택
5. 해당 모델의 확률만 UI에 표시
```

### 학습 흐름
```
1. 매시간: 증분 학습 (Tree 모델만, 새 데이터 반영)
2. 매일 장 마감 후: 전체 모델 재학습 (GPU 병렬)
3. 새 종목: 기존 데이터로 빠른 학습 후 예측 시작
```

### UI 표시 로직
```
1. 카드: 최고 성능 모델의 예측 확률 표시
2. 상세 패널: 5개 모델 전체의 50시간 Hit율 테이블
3. 색상: 70% 이상이면 강조 (초록/빨강)
```

---

## 10. 성공 가능성 평가

| 시나리오 | 예측 정확도 | 월 거래 기회 | 예상 수익 |
|----------|------------|-------------|----------|
| 낙관적 | 75% | 200건 | +30% |
| 현실적 | 65% | 100건 | +10% |
| 비관적 | 55% | 50건 | ±0% |

### 주요 리스크
- 과적합: 충분한 백테스팅 + Paper Trading으로 검증
- 시장 변화: 증분 학습으로 지속 적응
- 슬리피지: 거래량 확인 후 진입

---

**Claude Code로 구현 시 이 문서를 참조하여 각 모듈을 순차적으로 개발하면 됩니다.**
