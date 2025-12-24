# FiveForFree 프로젝트 분석 리포트

**작성일**: 2025-12-21 13:46
**작성자**: 분석팀장

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | FiveForFree (NASDAQ 단기 변동성 예측 시스템) |
| **목적** | NASDAQ 주식의 1시간 내 1%+ 가격 변동 확률 예측 |
| **대상** | 고변동성 NASDAQ 종목 (거래량/상승률 Top 100) |
| **기술스택** | Python(FastAPI) + React(Vite) + ML(PyTorch, XGBoost) |

---

## 2. 아키텍처 분석

### 2.1 디렉토리 구조

```
FiveForFree/
├── config/              # 설정 (settings.py, nasdaq_universe.json)
├── src/                 # 백엔드 소스코드
│   ├── api/             # FastAPI REST API + WebSocket
│   ├── models/          # ML 모델 (5종)
│   ├── collector/       # 데이터 수집 (Finnhub, Polygon, Yahoo)
│   ├── processor/       # 피처 엔지니어링
│   ├── predictor/       # 실시간 예측
│   ├── trainer/         # GPU 학습
│   ├── backtester/      # 백테스팅
│   └── utils/           # 유틸리티
├── frontend/            # React + Vite + Tailwind
├── scripts/             # 자동화 스크립트
├── tests/               # pytest 테스트
└── Docs/                # 문서화
```

### 2.2 핵심 컴포넌트

1. **데이터 수집** (`src/collector/`)
   - `finnhub_client.py`: Finnhub API 클라이언트
   - `polygon_client.py`: Polygon.io API 클라이언트
   - `minute_bars.py`: 1분봉 OHLCV 데이터
   - `market_context.py`: 시장 맥락 (SPY, QQQ, VIX)
   - `ticker_selector.py`: 종목 선정 로직

2. **ML 모델** (`src/models/`)
   - `xgboost_model.py`: XGBoost (GPU 가속)
   - `lightgbm_model.py`: LightGBM (GPU 가속)
   - `lstm_model.py`: LSTM (PyTorch)
   - `transformer_model.py`: Transformer (PyTorch)
   - `ensemble_model.py`: Stacking Ensemble
   - `model_manager.py`: 모델 관리 및 최적 모델 선택

3. **피처 엔지니어링** (`src/processor/`)
   - 49개 피처 (가격, 변동성, 거래량, 모멘텀, 시장맥락, 시간)
   - 자동 라벨 생성 (1% 임계값)

4. **API** (`src/api/`)
   - FastAPI 기반 REST API
   - WebSocket 실시간 가격 업데이트
   - Rate Limiter 미들웨어

5. **프론트엔드** (`frontend/`)
   - React 18 + Vite
   - Tailwind CSS
   - 다국어 지원 (ko/en)
   - Zustand 상태관리

---

## 3. 기술 스택 상세

### 3.1 백엔드
| 분류 | 기술 |
|------|------|
| API | FastAPI + WebSocket |
| ML | XGBoost, LightGBM, PyTorch (LSTM/Transformer) |
| 데이터 | Yahoo Finance, Finnhub, Pandas, NumPy |
| DB | SQLite (SQLAlchemy) |
| 스케줄러 | APScheduler |

### 3.2 프론트엔드
| 분류 | 기술 |
|------|------|
| 프레임워크 | React 18 + Vite |
| 스타일 | Tailwind CSS |
| 차트 | Recharts |
| 상태관리 | React Query + Zustand |

### 3.3 인프라
- Docker 지원 (Dockerfile, docker-compose.yml)
- GitHub Actions CI/CD
- pytest 테스트 (19개 테스트 파일)

---

## 4. 주요 기능

### 4.1 데이터 수집
- 매시간 종목 선정 (거래량 상위 + 상승률 상위)
- 1분봉 OHLCV 데이터 (Yahoo Finance, 최근 7일)
- 실시간 호가 (Finnhub)
- 시장 맥락 (SPY, QQQ, VXX, 섹터 ETF)

### 4.2 예측 시스템
- 종목당 5개 모델 x 2방향 (상승/하락) = 10개 모델
- 50시간 롤링 윈도우 기반 최적 모델 자동 선택
- 매시간 증분 학습 + 매일 전체 재학습

### 4.3 백테스팅
- 50시간 롤링 윈도우 시뮬레이션
- "1% 달성 또는 1시간 경과" 청산 규칙
- 모델별 적중률 추적

---

## 5. 코드 품질 분석

### 5.1 강점
- 체계적인 모듈화 구조
- 상세한 문서화 (README, PROJECT_SPEC, API_EXAMPLES 등)
- 포괄적인 테스트 커버리지 (.coverage 파일 존재)
- GPU 가속 지원
- 다국어 지원 (i18n)

### 5.2 개선 가능 영역
- 데이터 소스 통합 필요 (Finnhub, Polygon, Yahoo 혼재)
- 프론트엔드 node_modules 부재 (npm install 필요)
- .env 파일 부재 (API 키 설정 필요)

---

## 6. 파일 통계

| 분류 | 파일 수 | 주요 확장자 |
|------|---------|-------------|
| Python 소스 | ~50개 | .py |
| React 컴포넌트 | ~15개 | .jsx |
| 테스트 | 19개 | test_*.py |
| 문서 | ~10개 | .md |
| 설정 | ~5개 | .json, .yml |

---

## 7. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt
cd frontend && npm install

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 추가

# 3. 초기화
python scripts/init_database.py
python scripts/collect_historical.py 30
python scripts/train_all_models.py

# 4. 실행
python scripts/run_system.py  # 백엔드
cd frontend && npm run dev    # 프론트엔드
```

---

## 8. 결론

FiveForFree는 NASDAQ 주식 단기 변동성을 예측하는 잘 설계된 ML 기반 시스템입니다.
- **아키텍처**: 모듈화가 잘 되어 있고 확장 가능한 구조
- **ML 파이프라인**: 다양한 모델과 앙상블 전략 적용
- **UI/UX**: 실시간 WebSocket 기반 대시보드
- **품질**: 테스트 커버리지와 문서화가 우수함

---

*이 리포트는 분석팀장이 작성하였습니다.*
