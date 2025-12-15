# Claude Code 작업 지침

이 문서는 FiveForFree 프로젝트에서 Claude가 작업할 때 따라야 할 지침입니다.

---

## 프로젝트 개요

**FiveForFree**는 NASDAQ 주식의 단기 가격 움직임을 예측하는 머신러닝 시스템입니다.

- **예측 대상**: 60분 내 상승/하락 확률
- **데이터 소스**: Yahoo Finance (분봉), Finnhub (실시간 호가)
- **모델**: XGBoost, LightGBM, LSTM, Transformer, Ensemble
- **기술 스택**: Python, FastAPI, SQLite, React

---

## 문서 참조 규칙

### 작업 시작 전 반드시 확인

| 작업 영역 | 참조 문서 |
|----------|----------|
| 전체 구조 파악 | `docs/PROJECT_ARCHITECTURE.md` |
| 데이터 수집 관련 | `docs/DATA_COLLECTION.md` |
| 피처 엔지니어링 | `docs/FEATURE_ENGINEERING_IMPLEMENTATION.md` |
| 피처 상세 | `docs/FEATURES_REFERENCE.md` |
| 모델/앙상블 | `docs/HYBRID_ENSEMBLE_ARCHITECTURE.md` |
| 테스트 | `docs/TESTING.md` |

### 문서 확인 우선순위

1. **코드 수정 전**: 관련 문서에서 현재 구조 확인
2. **새 기능 추가 시**: `PROJECT_ARCHITECTURE.md`에서 위치 확인
3. **버그 수정 시**: 관련 모듈 문서 참조

---

## 문서 업데이트 규칙

### 코드 변경 시 문서 업데이트 필수

작업 완료 후 **반드시** 관련 문서를 업데이트하세요:

#### 1. 데이터 수집 변경 시
- `docs/DATA_COLLECTION.md` 업데이트
  - 새 데이터 소스 추가
  - API 제한사항 변경
  - 수집 로직 변경

#### 2. 피처 변경 시
- `docs/FEATURES_REFERENCE.md` 업데이트
  - 피처 추가/삭제
  - 피처 계산 방식 변경
- `docs/FEATURE_ENGINEERING_IMPLEMENTATION.md` 업데이트
  - 파이프라인 변경

#### 3. 모델 변경 시
- `docs/HYBRID_ENSEMBLE_ARCHITECTURE.md` 업데이트
  - 새 모델 추가
  - 앙상블 방식 변경
  - 하이퍼파라미터 변경

#### 4. 구조 변경 시
- `docs/PROJECT_ARCHITECTURE.md` 업데이트
  - 새 모듈 추가
  - 디렉토리 구조 변경
  - 데이터 흐름 변경

#### 5. 테스트 변경 시
- `docs/TESTING.md` 업데이트
  - 새 테스트 추가
  - 테스트 실행 방법 변경

### 문서 업데이트 형식

```markdown
---

**Last Updated**: YYYY-MM-DD
**Version**: X.Y
```

각 문서 하단의 날짜와 버전을 업데이트하세요.

---

## 코드 작성 규칙

### 스타일 가이드

- **포맷터**: Black
- **린터**: Flake8 (max-line-length=100)
- **타입 힌트**: 모든 함수에 적용
- **독스트링**: Google 스타일

### 디렉토리 구조

```
src/
├── collector/     # 데이터 수집
├── processor/     # 데이터 처리 (피처, 레이블)
├── trainer/       # 모델 학습
├── predictor/     # 예측
├── backtester/    # 백테스팅
├── api/           # FastAPI 백엔드
├── models/        # 모델 정의
└── utils/         # 유틸리티
```

### 새 파일 생성 시

1. 적절한 디렉토리에 배치
2. `__init__.py`에 export 추가
3. 관련 문서 업데이트

---

## 주요 설정 파라미터

`config/settings.py` 참조:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `TOP_N_VOLUME` | 50 | 거래량 상위 종목 수 |
| `TOP_N_GAINERS` | 50 | 상승률 상위 종목 수 |
| `HISTORICAL_DAYS` | 30 | 히스토리 수집 기간 |
| `PREDICTION_HORIZON_MINUTES` | 60 | 예측 시간 범위 |
| `TARGET_PERCENT` | 1.0 | 목표 수익률 (%) |
| `PROBABILITY_THRESHOLD` | 0.70 | 예측 확률 임계값 |

---

## 데이터베이스 스키마

`src/utils/database.py` 참조:

| 테이블 | 용도 |
|--------|------|
| `tickers` | 종목 정보 |
| `minute_bars` | 분봉 OHLCV |
| `predictions` | 예측 결과 |
| `trades` | 거래 내역 |
| `model_performance` | 모델 성과 |

---

## 자주 사용하는 명령어

### 데이터 수집
```bash
python scripts/collect_historical.py
python scripts/collect_historical.py --days 60 --tickers AAPL MSFT
```

### 모델 학습
```bash
python scripts/train_all_models.py
```

### 시스템 실행
```bash
python scripts/run_system.py
```

### API 실행
```bash
python run_api.py
```

### 테스트
```bash
pytest tests/ -v
pytest tests/test_config.py
```

---

## 알려진 제한사항

### 데이터 수집
- Yahoo Finance: 1분봉 최대 7일, 5분봉 최대 60일
- VWAP: 실제 값이 아닌 근사값 `(H+L+C)/3` 사용
- Level 2 데이터: 무료 tier에서 미지원

### Order Book 피처
- `docs/FEATURES_REFERENCE.md`의 Order Book Features (8개)는 현재 비활성
- Level 2 데이터 소스 필요 (Polygon.io 유료 플랜)

---

## 트러블슈팅

### DB 초기화
```bash
python scripts/init_database.py
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 환경 변수
`.env` 파일 필수:
```
FINNHUB_API_KEY=your_api_key_here
```

---

## 체크리스트

### 코드 수정 완료 후

- [ ] 관련 테스트 실행 및 통과
- [ ] Black / Flake8 검사 통과
- [ ] 관련 문서 업데이트
- [ ] 커밋 메시지 작성

### 새 기능 추가 후

- [ ] 유닛 테스트 작성
- [ ] 통합 테스트 확인
- [ ] `docs/PROJECT_ARCHITECTURE.md` 업데이트
- [ ] 해당 영역 문서 업데이트

---

## 문서 목록

| 문서 | 경로 | 내용 |
|------|------|------|
| 프로젝트 아키텍처 | `docs/PROJECT_ARCHITECTURE.md` | 전체 구조, 모듈, 데이터 흐름 |
| 데이터 수집 | `docs/DATA_COLLECTION.md` | 수집 로직, API, 제한사항 |
| 피처 엔지니어링 | `docs/FEATURE_ENGINEERING_IMPLEMENTATION.md` | 피처 구현 상세 |
| 피처 레퍼런스 | `docs/FEATURES_REFERENCE.md` | 57개 피처 목록 |
| 앙상블 아키텍처 | `docs/HYBRID_ENSEMBLE_ARCHITECTURE.md` | 모델 앙상블 구조 |
| 테스트 가이드 | `docs/TESTING.md` | 테스트 실행 방법 |
| Claude 지침 | `claude.md` | 본 문서 |

---

**Last Updated**: 2024-12-15
