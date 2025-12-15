# Project Architecture

FiveForFree NASDAQ 예측 시스템의 전체 아키텍처 문서입니다.

## 목차

- [시스템 개요](#시스템-개요)
- [디렉토리 구조](#디렉토리-구조)
- [핵심 모듈](#핵심-모듈)
- [데이터 흐름](#데이터-흐름)
- [기술 스택](#기술-스택)

---

## 시스템 개요

FiveForFree는 NASDAQ 주식의 단기 가격 움직임을 예측하는 머신러닝 시스템입니다.

### 핵심 목표
- **예측 대상**: 60분 내 5% 이상 상승/하락 확률
- **대상 종목**: NASDAQ 100 + 고거래량 종목 (최대 100개)
- **모델 앙상블**: XGBoost, LightGBM, LSTM, Transformer

### 시스템 구성
```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│                    실시간 예측 대시보드                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket / REST API
┌───────────────────────────▼─────────────────────────────────────┐
│                     FastAPI Backend                              │
│              /predictions, /models, /health                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Collector   │  │   Predictor   │  │   Trainer     │
│  (Data Feed)  │  │  (Inference)  │  │  (Training)   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                   │
        ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLite Database                              │
│        minute_bars, predictions, trades, model_performance       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
FiveForFree/
├── config/                 # 설정 관리
│   ├── __init__.py
│   └── settings.py         # Pydantic 기반 환경 설정
│
├── src/                    # 핵심 소스 코드
│   ├── api/                # FastAPI 백엔드
│   │   ├── routes/         # API 엔드포인트
│   │   │   ├── health.py   # 헬스 체크
│   │   │   ├── models.py   # 모델 관리
│   │   │   ├── predictions.py # 예측 API
│   │   │   └── tickers.py  # 종목 정보
│   │   ├── main.py         # FastAPI 앱
│   │   ├── dependencies.py # 의존성 주입
│   │   └── websocket.py    # WebSocket 핸들러
│   │
│   ├── collector/          # 데이터 수집
│   │   ├── minute_bars.py  # yfinance 분봉 수집
│   │   ├── ticker_selector.py # 종목 선정
│   │   ├── finnhub_client.py # Finnhub API
│   │   ├── polygon_client.py # Polygon.io API
│   │   ├── quotes.py       # 실시간 호가
│   │   └── market_context.py # 시장 컨텍스트
│   │
│   ├── processor/          # 데이터 처리
│   │   ├── feature_engineer.py # 57개 피처 생성
│   │   └── label_generator.py  # 레이블 생성
│   │
│   ├── trainer/            # 모델 학습
│   │   ├── gpu_trainer.py  # GPU 기반 학습
│   │   └── incremental.py  # 증분 학습
│   │
│   ├── predictor/          # 예측 엔진
│   │   └── realtime_predictor.py # 실시간 예측
│   │
│   ├── models/             # 모델 정의
│   │   └── base_model.py   # 기본 모델 클래스
│   │
│   ├── backtester/         # 백테스팅
│   │   ├── simulator.py    # 거래 시뮬레이션
│   │   └── metrics.py      # 성과 지표
│   │
│   ├── utils/              # 유틸리티
│   │   ├── database.py     # SQLAlchemy 모델
│   │   └── logger.py       # 로깅 설정
│   │
│   └── scheduler.py        # 작업 스케줄러
│
├── scripts/                # 실행 스크립트
│   ├── init_database.py    # DB 초기화
│   ├── collect_historical.py # 히스토리 수집
│   ├── train_all_models.py # 전체 모델 학습
│   └── run_system.py       # 시스템 실행
│
├── frontend/               # React 프론트엔드
│   ├── src/
│   └── package.json
│
├── tests/                  # 테스트
│   ├── test_config.py
│   └── test_models.py
│
├── docs/                   # 문서
│   ├── PROJECT_ARCHITECTURE.md  # 본 문서
│   ├── DATA_COLLECTION.md       # 데이터 수집
│   ├── FEATURE_ENGINEERING_IMPLEMENTATION.md
│   ├── FEATURES_REFERENCE.md
│   ├── HYBRID_ENSEMBLE_ARCHITECTURE.md
│   └── TESTING.md
│
├── data/                   # 데이터 저장소
│   └── nasdaq_predictor.db # SQLite DB
│
├── models/                 # 학습된 모델 저장
│
├── logs/                   # 로그 파일
│
├── .env                    # 환경 변수 (API 키)
├── requirements.txt        # Python 의존성
├── claude.md              # Claude 작업 지침
└── README.md
```

---

## 핵심 모듈

### 1. Collector (데이터 수집)

| 모듈 | 역할 | 데이터 소스 |
|------|------|------------|
| `MinuteBarCollector` | 분봉 OHLCV 수집 | Yahoo Finance |
| `TickerSelector` | 대상 종목 선정 | Yahoo Finance |
| `FinnhubClient` | 실시간 호가, 뉴스 | Finnhub API |
| `PolygonClient` | Level 2 데이터 | Polygon.io (미사용) |

**참고**: [DATA_COLLECTION.md](./DATA_COLLECTION.md)

### 2. Processor (데이터 처리)

| 모듈 | 역할 | 출력 |
|------|------|------|
| `FeatureEngineer` | 57개 피처 계산 | 피처 DataFrame |
| `LabelGenerator` | Up/Down 레이블 생성 | 레이블 Series |

**참고**: [FEATURE_ENGINEERING_IMPLEMENTATION.md](./FEATURE_ENGINEERING_IMPLEMENTATION.md)

### 3. Trainer (모델 학습)

| 모델 | 타입 | 특징 |
|------|------|------|
| XGBoost | Gradient Boosting | 빠른 학습, 피처 중요도 |
| LightGBM | Gradient Boosting | 대용량 데이터, 범주형 |
| LSTM | Deep Learning | 시계열 패턴 |
| Transformer | Deep Learning | 어텐션 메커니즘 |
| Ensemble | Hybrid | 가중 앙상블 |

**참고**: [HYBRID_ENSEMBLE_ARCHITECTURE.md](./HYBRID_ENSEMBLE_ARCHITECTURE.md)

### 4. Predictor (예측)

| 모듈 | 역할 |
|------|------|
| `RealtimePredictor` | 실시간 예측 생성 |

### 5. Backtester (백테스팅)

| 모듈 | 역할 |
|------|------|
| `Simulator` | 거래 시뮬레이션 |
| `Metrics` | 성과 지표 계산 |

---

## 데이터 흐름

### 학습 파이프라인
```
Yahoo Finance API
       │
       ▼ (1분봉/5분봉 수집)
MinuteBarCollector ──▶ SQLite (minute_bars)
       │
       ▼ (피처 계산)
FeatureEngineer ──▶ 57개 피처 DataFrame
       │
       ▼ (레이블 생성)
LabelGenerator ──▶ up/down 레이블
       │
       ▼ (모델 학습)
GPUTrainer ──▶ 모델 저장 (models/)
```

### 예측 파이프라인
```
실시간 데이터
       │
       ▼
MinuteBarCollector (최근 60분)
       │
       ▼
FeatureEngineer (피처 계산)
       │
       ▼
RealtimePredictor (앙상블 추론)
       │
       ▼
API Response / WebSocket Broadcast
```

---

## 기술 스택

### Backend
| 기술 | 용도 |
|------|------|
| Python 3.10+ | 메인 언어 |
| FastAPI | REST API / WebSocket |
| SQLAlchemy 2.0 | ORM |
| SQLite | 데이터베이스 |
| Pydantic | 설정 / 검증 |
| Loguru | 로깅 |

### ML/DL
| 기술 | 용도 |
|------|------|
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| PyTorch | LSTM / Transformer |
| TA-Lib | 기술적 지표 |
| pandas / numpy | 데이터 처리 |

### Frontend
| 기술 | 용도 |
|------|------|
| React | UI 프레임워크 |
| Vite | 빌드 도구 |
| WebSocket | 실시간 통신 |

### 데이터 소스
| 소스 | 용도 | 제한 |
|------|------|------|
| Yahoo Finance | 분봉 데이터 | 1분봉 7일, 5분봉 60일 |
| Finnhub | 실시간 호가 | 60 calls/min |
| Polygon.io | Level 2 데이터 | 유료 (미사용) |

---

## 설정 파라미터

`config/settings.py`에서 관리:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `TOP_N_VOLUME` | 50 | 거래량 상위 종목 수 |
| `TOP_N_GAINERS` | 50 | 상승률 상위 종목 수 |
| `HISTORICAL_DAYS` | 30 | 수집 기간 (일) |
| `PREDICTION_HORIZON_MINUTES` | 60 | 예측 시간 범위 |
| `TARGET_PERCENT` | 1.0 | 목표 수익률 (%) |
| `PROBABILITY_THRESHOLD` | 0.70 | 예측 확률 임계값 |

---

## 데이터베이스 스키마

### 주요 테이블

| 테이블 | 용도 |
|--------|------|
| `tickers` | 종목 정보 |
| `minute_bars` | 분봉 OHLCV |
| `predictions` | 예측 결과 |
| `trades` | 거래 내역 |
| `model_performance` | 모델 성과 |

**참고**: `src/utils/database.py`

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [DATA_COLLECTION.md](./DATA_COLLECTION.md) | 데이터 수집 상세 |
| [FEATURE_ENGINEERING_IMPLEMENTATION.md](./FEATURE_ENGINEERING_IMPLEMENTATION.md) | 피처 엔지니어링 |
| [FEATURES_REFERENCE.md](./FEATURES_REFERENCE.md) | 57개 피처 레퍼런스 |
| [HYBRID_ENSEMBLE_ARCHITECTURE.md](./HYBRID_ENSEMBLE_ARCHITECTURE.md) | 앙상블 아키텍처 |
| [TESTING.md](./TESTING.md) | 테스트 가이드 |

---

**Last Updated**: 2024-12-15
**Version**: 1.0
