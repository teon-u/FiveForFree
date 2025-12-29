# Dynamic Ticker Discovery & Auto-Training Analysis Report

**작성자**: 분석팀장
**작성일**: 2025-12-22
**분석 대상**: 동적 종목 발견 API 및 자동 학습 로직

---

## 1. 개요

FiveForFree 시스템의 동적 종목 발견 및 자동 학습 기능을 분석하였습니다.

### 1.1 분석 범위
- `/api/status/discover` API 엔드포인트
- 새 종목 자동 학습 트리거 로직
- 관련 핵심 클래스 및 데이터 흐름

---

## 2. 동적 종목 발견 API (`/api/status/discover`)

### 2.1 위치
- **파일**: `src/api/routes/status.py`
- **엔드포인트**: `GET /api/status/discover`

### 2.2 기능
시장에서 새로운 상승 종목을 발견하고 시스템 내 상태를 파악합니다.

### 2.3 응답 구조 (`DiscoveredTickersResponse`)

| 필드 | 타입 | 설명 |
|------|------|------|
| `market_gainers` | `List[NewTickerInfo]` | 시장 상위 상승 종목 목록 |
| `new_tickers` | `List[str]` | 시스템에 데이터가 없는 새 종목 |
| `tickers_needing_training` | `List[str]` | 데이터는 있지만 모델이 없는 종목 |
| `timestamp` | `str` | 조회 시점 |

### 2.4 NewTickerInfo 구조

| 필드 | 타입 | 설명 |
|------|------|------|
| `ticker` | `str` | 종목 심볼 |
| `company_name` | `Optional[str]` | 회사명 |
| `change_percent` | `float` | 변동률 (%) |
| `price` | `float` | 현재가 |
| `volume` | `float` | 거래량 |
| `has_data` | `bool` | 분봉 데이터 존재 여부 |
| `has_model` | `bool` | 학습된 모델 존재 여부 |

### 2.5 데이터 흐름

```
1. ModelManager.get_tickers() → 학습된 종목 목록 조회
2. DB Query → 분봉 데이터가 있는 종목 목록 조회
3. TickerSelector.get_market_top_gainers(50) → Yahoo Finance API로 시장 상승 종목 조회
4. 종목별 상태 분류:
   - has_data = False → new_tickers에 추가
   - has_data = True & has_model = False → tickers_needing_training에 추가
```

---

## 3. TickerSelector 클래스

### 3.1 위치
- **파일**: `src/collector/ticker_selector.py`

### 3.2 주요 메서드

#### `get_market_top_gainers(limit=50)`
- **데이터 소스**: Yahoo Finance Screener API
- **URL**: `https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved`
- **필터 조건**:
  - NASDAQ 거래소만 (NMS, NGM, NCM, NASDAQ)
  - 최소 가격: $5.00
  - 최소 거래량: 1,000,000
- **정렬**: 변동률 기준 내림차순

#### `get_top_by_volume()`
- NASDAQ_UNIVERSE 리스트 기반 거래량 상위 종목 조회

#### `get_top_by_gainers()`
- NASDAQ_UNIVERSE 리스트 기반 상승률 상위 종목 조회

### 3.3 NASDAQ Universe
- **파일**: `config/nasdaq_universe.json`
- 사전 정의된 주요 NASDAQ 종목 리스트

---

## 4. 자동 학습 트리거 API

### 4.1 엔드포인트
- **POST** `/api/status/train/{ticker}`

### 4.2 기능
지정된 종목에 대해 백그라운드에서 모델 학습을 시작합니다.

### 4.3 응답

```json
{
  "status": "training_started",
  "ticker": "AAPL",
  "message": "Training for AAPL started in background"
}
```

### 4.4 동시 실행 제한
- `_training_in_progress` 플래그로 동시 학습 방지
- 학습 중 추가 요청 시 HTTP 409 Conflict 반환

---

## 5. 백그라운드 학습 프로세스 (`_train_ticker_background`)

### 5.1 학습 파이프라인

```
Step 1: 데이터 수집
├── MinuteBarCollector.get_bars(ticker, 30일)
└── 최소 1,000개 분봉 필요

Step 2: 피처 엔지니어링
├── FeatureEngineer.compute_features(df)
└── 기술적 지표 계산

Step 3: 레이블 생성
├── LabelGenerator.generate_labels()
├── label_up: 상승 신호
└── label_down: 하락 신호

Step 4: 데이터 정제
├── NaN 제거
└── 최소 100개 유효 샘플 필요

Step 5: 모델 학습
├── GPUParallelTrainer.train_single_ticker()
└── XGBoost, LightGBM, LSTM, Transformer, Ensemble
```

### 5.2 학습 종료 조건 (실패 케이스)
1. 데이터 부족: `len(bars) < 1000`
2. 유효 샘플 부족: `len(X) < 100`

---

## 6. GPUParallelTrainer 클래스

### 6.1 위치
- **파일**: `src/trainer/gpu_trainer.py`

### 6.2 GPU 최적화 (RTX 5080 기준)
- **Tree 모델** (XGBoost, LightGBM): ThreadPoolExecutor 병렬 학습
- **Neural 모델** (LSTM, Transformer): 순차 학습 (GPU 메모리 관리)

### 6.3 모델 아키텍처

#### Structure A (Direct Prediction)
- `up` 모델: 상승 예측
- `down` 모델: 하락 예측

#### Structure B (Hybrid)
- `volatility` 모델: 변동성 예측 (±1% 움직임)
- `direction` 모델: 방향 예측 (변동성 있는 샘플만)

### 6.4 학습되는 모델 목록

| 모델 타입 | 학습 방식 | 설명 |
|-----------|-----------|------|
| XGBoost | GPU 병렬 | tree_method="gpu_hist" |
| LightGBM | GPU 병렬 | GPU 가속 |
| LSTM | 순차 | PyTorch CUDA |
| Transformer | 순차 | PyTorch CUDA |
| Ensemble | 순차 | 위 4개 모델 앙상블 |

---

## 7. ModelManager 클래스

### 7.1 위치
- **파일**: `src/models/model_manager.py`

### 7.2 주요 기능
- 모델 생성, 저장, 로드 관리
- 종목별/모델별 성능 추적
- 최적 모델 선택 (50시간 히트레이트 기준)

### 7.3 저장 경로
- **디렉토리**: `data/models/{TICKER}/`
- **파일 형식**: `{target}_{model_type}.pkl`
- **예시**: `up_xgboost.pkl`, `down_lstm.pkl`

---

## 8. 시스템 통합 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React)                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  발견 종목 UI   │  │  학습 트리거 UI │  │  학습 상태 UI   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ GET /discover   │  │ POST /train/{t} │  │ GET /training   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            ▼                     ▼                     │
┌───────────────────────────────────────────┐          │
│           TickerSelector                   │          │
│  ┌─────────────────────────────────────┐  │          │
│  │  Yahoo Finance Screener API         │  │          │
│  │  (day_gainers, NASDAQ only)         │  │          │
│  └─────────────────────────────────────┘  │          │
└───────────────────────────────────────────┘          │
                                                       │
┌──────────────────────────────────────────────────────┼──────────┐
│                  Background Training                  │          │
│  ┌─────────────────┐  ┌─────────────────┐            │          │
│  │ MinuteBarCollector│  │FeatureEngineer│            │          │
│  └────────┬────────┘  └────────┬────────┘            │          │
│           │                     │                     │          │
│           ▼                     ▼                     │          │
│  ┌─────────────────┐  ┌─────────────────┐            │          │
│  │ LabelGenerator  │  │ GPUParallelTrainer│◄─────────┘          │
│  └─────────────────┘  └────────┬────────┘                       │
│                                │                                 │
│                                ▼                                 │
│                     ┌─────────────────┐                         │
│                     │  ModelManager   │                         │
│                     │ (data/models/)  │                         │
│                     └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. UI 구현 현황 (2025-12-22 QA팀장 확인)

### 9.1 구현 완료된 UI
QA팀장의 브라우저 테스트 결과, 다음 UI가 이미 구현되어 있음을 확인:

1. **동적 종목 발견 UI**: Settings 패널 > "새로 발견된 상승 종목" 섹션
2. **자동 학습 트리거 UI**: 각 종목별 "학습 시작" 버튼
3. **종목 상태 표시**: 10개 발견 종목 + 학습 버튼

### 9.2 테스트 결과 (PASS)
- 발견 종목 목록 (10개 + 학습 버튼): ✅ PASS
- Settings Panel 열기/닫기: ✅ PASS
- 전체 12개 테스트 항목 PASS

---

## 10. 결론

FiveForFree 시스템의 동적 종목 발견 및 자동 학습 로직은 백엔드에 완전히 구현되어 있습니다.

**핵심 플로우**:
1. Yahoo Finance API로 시장 상승 종목 실시간 발견
2. 시스템 내 데이터/모델 존재 여부 확인
3. API 호출로 백그라운드 학습 트리거
4. GPU 병렬 학습으로 5개 모델 동시 학습
5. 학습 완료 후 자동 저장 및 서비스 투입

**다음 단계**: 프론트엔드 UI 구현 필요

---

*분석팀장 작성*
