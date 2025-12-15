# NASDAQ 단기 변동성 예측 시스템

NASDAQ 주식의 1시간 내 5% 이상 가격 변동 확률을 예측하는 AI 기반 시스템입니다.

## 🎯 개요

- **타겟**: 고변동성 NASDAQ 주식 (인기 종목 중 거래량 상위 50개 + 상승률 상위 50개)
- **예측**: 향후 60분 내 5% 이상 상승/하락 확률
- **모델**: 종목당 5가지 ML 모델 (XGBoost, LightGBM, LSTM, Transformer, Ensemble)
- **데이터 소스**: Finnhub 무료 티어 (무료!) + Yahoo Finance
- **하드웨어**: RTX 5080 GPU, AMD Ryzen 9800X3D, 64GB RAM

## 🚀 빠른 시작

### 1. 사전 요구사항

```bash
# Python 3.10 이상
python --version

# GPU 지원을 위한 CUDA 12.0 이상
nvidia-smi
```

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/teon-u/FiveForFree.git
cd FiveForFree

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Python 의존성 설치
pip install -r requirements.txt

# 프론트엔드 설정
cd frontend
npm install
cd ..
```

### 3. 환경 설정

```bash
# 환경 변수 템플릿 복사
cp .env.example .env

# .env 파일을 열어 Finnhub API 키 추가
# Finnhub 무료 API 키 발급: https://finnhub.io/register
nano .env
```

### 4. 초기 설정

```bash
# 데이터베이스 초기화
python scripts/init_database.py

# 과거 데이터 수집 (30일)
python scripts/collect_historical.py 30

# 초기 모델 학습
python scripts/train_all_models.py
```

### 5. 시스템 실행

```bash
# 터미널 1: 백엔드 시작
python scripts/run_system.py

# 터미널 2: 프론트엔드 개발 서버 시작
cd frontend
npm run dev
```

브라우저에서 http://localhost:5173 을 여세요.

## 📁 프로젝트 구조

```
FiveForFree/
├── config/              # 설정 파일
├── src/
│   ├── collector/       # Polygon.io 데이터 수집
│   ├── processor/       # 피처 엔지니어링 및 레이블링
│   ├── models/          # ML 모델 (XGBoost, LSTM 등)
│   ├── trainer/         # GPU 가속 학습
│   ├── predictor/       # 실시간 예측
│   ├── backtester/      # 성능 시뮬레이션
│   ├── api/             # FastAPI 백엔드
│   └── utils/           # 유틸리티
├── frontend/            # React + Vite + Tailwind UI
├── data/                # 원시 및 처리된 데이터
├── scripts/             # 자동화 스크립트
└── tests/               # 테스트 스위트
```

## 🎨 기술 스택

### 백엔드
- **API**: FastAPI + WebSocket
- **ML**: XGBoost, LightGBM, PyTorch (LSTM/Transformer)
- **데이터**: Finnhub API (무료), Yahoo Finance, Pandas, NumPy
- **데이터베이스**: SQLite (SQLAlchemy)
- **스케줄러**: APScheduler

### 프론트엔드
- **프레임워크**: React 18 + Vite
- **스타일링**: Tailwind CSS
- **차트**: Recharts
- **상태관리**: React Query + Zustand
- **WebSocket**: 네이티브 WebSocket API

## 📊 주요 기능

### 데이터 수집
- ✅ 매시간 종목 선정 (인기 종목 중 거래량 상위 + 상승률 상위)
- ✅ 5분봉 OHLCV 데이터 + VWAP 계산
- ✅ 실시간 호가 (현재가, 고가, 저가)
- ✅ 시장 맥락 (SPY, QQQ, VXX, 주요 섹터 ETF)

### 피처 엔지니어링
- 📈 6개 카테고리에 걸친 49개 피처 생성
  - 가격 기반 (15), 변동성 (10), 거래량 (8)
  - 모멘텀 (8), 시장 맥락 (5), 시간 (3)
- 🎯 자동 레이블 생성 (5% 임계값)
- ⚡ GPU 가속 처리

### 머신러닝
- 🤖 종목당 방향(상승/하락)별 5개 모델
- 🏆 자동 최적 모델 선택 (50시간 정확도 기반)
- 🔄 증분 학습 (매시간)
- 📊 전체 재학습 (장 마감 후 매일)

### 백테스팅
- 📉 50시간 롤링 윈도우 시뮬레이션
- 💰 "5% 달성 또는 1시간 경과" 청산 규칙
- 📈 모델별 적중률 추적

### 실시간 UI
- 🎴 카드 기반 종목 표시 (거래량/상승률)
- 🟢🔴 색상으로 구분된 확률
- 📊 모델 성능 대시보드
- 📈 60분 가격 차트
- 🔄 WebSocket 실시간 업데이트

## ⚙️ 자동화 워크플로우

### GitHub Actions
- ✅ Push/PR 시 자동 테스트
- ✅ 코드 품질 검사 (Black, Flake8)
- ✅ 프론트엔드 빌드 검증
- ✅ Docker 이미지 빌드

### 지속적 학습
- 🔄 매시간: 증분 학습
- 📅 매일: 전체 모델 재학습
- 🎯 자동: 최적 모델 선택

## 📈 성능 지표

| 시나리오 | 정확도 | 월간 거래 횟수 | 예상 수익률 |
|----------|----------|----------------|-----------------|
| 낙관적 | 75% | 200 | +30% |
| 현실적 | 65% | 100 | +10% |
| 비관적 | 55% | 50 | ±0% |

## 🛠️ 개발

```bash
# 테스트 실행
pytest

# 코드 포맷팅
black .

# 코드 린트
flake8 src/

# 프론트엔드 개발
cd frontend
npm run dev        # 개발 서버
npm run build      # 프로덕션 빌드
npm run preview    # 빌드 미리보기
```

## 📝 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🤝 기여하기

기여를 환영합니다! 먼저 [CONTRIBUTING.md](CONTRIBUTING.md)를 읽어주세요.

## ⚠️ 면책 조항

본 시스템은 **교육 및 연구 목적으로만** 제작되었습니다. 투자 조언이 아닙니다. 트레이딩은 상당한 손실 위험을 수반합니다. 항상 스스로 조사하고 감당할 수 있는 범위 내에서만 투자하세요.

## 📞 지원

- 📧 이메일: support@example.com
- 🐛 이슈: [GitHub Issues](https://github.com/teon-u/FiveForFree/issues)
- 💬 토론: [GitHub Discussions](https://github.com/teon-u/FiveForFree/discussions)
