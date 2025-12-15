# Data Collection

FiveForFree 데이터 수집 시스템의 상세 문서입니다.

## 목차

- [개요](#개요)
- [데이터 축적 전략](#데이터-축적-전략)
- [수집 모듈](#수집-모듈)
- [사용법](#사용법)
- [제한사항](#제한사항)
- [트러블슈팅](#트러블슈팅)

---

## 개요

데이터 수집 시스템은 NASDAQ 주식의 5분봉 데이터를 수집하고 SQLite에 저장합니다.

### 핵심 설계
- **데이터 소스**: Yahoo Finance (무료, API 키 불필요)
- **수집 간격**: 5분봉 (60일 제한 극복)
- **증분 수집**: DB의 최신 데이터 이후만 가져옴
- **장기 축적**: 지속적 실행으로 데이터 누적

---

## 데이터 축적 전략

### Yahoo Finance 제한

| 간격 | 최대 기간 | 용도 |
|------|----------|------|
| 1분봉 | 7일 | 실시간 분석 |
| **5분봉** | **60일** | **장기 축적 (권장)** |

### 증분 수집으로 데이터 누적

```
┌─────────────────────────────────────────────────────────────────┐
│                    데이터 축적 타임라인                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  t=0 (첫 실행)                                                  │
│  ────────────────────────────────────────────────────────────   │
│  │◄──────────── 60일 수집 ────────────►│                        │
│  day -60                              day 0                     │
│  DB: 60일치 데이터                                              │
│                                                                 │
│  t=30일 (1개월 후)                                              │
│  ────────────────────────────────────────────────────────────   │
│  │◄──────── DB 기존 ────────►│◄─ 신규 ─►│                       │
│  day -60                    day 0      day 30                   │
│  DB: 90일치 데이터                                              │
│                                                                 │
│  t=60일 (2개월 후)                                              │
│  ────────────────────────────────────────────────────────────   │
│  │◄────────── DB 기존 ──────────►│◄─ 신규 ─►│                   │
│  day -60                        day 30      day 60              │
│  DB: 120일치 데이터 (4개월)                                     │
│                                                                 │
│  t=90일 (3개월 후)                                              │
│  ────────────────────────────────────────────────────────────   │
│  │◄──────────── DB 기존 ────────────►│◄─ 신규 ─►│              │
│  day -60                            day 60      day 90          │
│  DB: 150일치 데이터 (5개월)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 예상 데이터 축적량

| 운영 기간 | DB 데이터 | 5분봉 수 (종목당) |
|----------|----------|------------------|
| 0일 (첫 실행) | 60일 | ~4,680 bars |
| 1개월 | 90일 | ~7,020 bars |
| **2개월** | **120일 (4개월)** | **~9,360 bars** |
| 3개월 | 150일 (5개월) | ~11,700 bars |
| 6개월 | 240일 (8개월) | ~18,720 bars |

> **결론**: 2개월 운영 후 약 4개월치(120일) 데이터가 축적됩니다.

### 권장 운영 방식

```bash
# 1. 첫 실행: 60일치 5분봉 전체 수집
python scripts/collect_historical.py --days 60

# 2. 이후 매일 (cron 등): 증분 수집만
python scripts/collect_historical.py --update
```

---

## 수집 모듈

### MinuteBarCollector

**위치**: `src/collector/minute_bars.py`

```python
from src.collector.minute_bars import MinuteBarCollector
from datetime import datetime, timedelta

collector = MinuteBarCollector(use_db=True)

# 단일 종목 수집
bars = collector.get_bars(
    ticker="AAPL",
    from_date=datetime.now() - timedelta(days=60),
    to_date=datetime.now()
)

# 배치 수집
result = collector.get_bars_batch(
    tickers=["AAPL", "MSFT", "GOOGL"],
    from_date=from_date,
    to_date=to_date
)
```

#### 증분 수집 흐름

```python
def get_bars(ticker, from_date, to_date):
    # 1. DB에서 최신 timestamp 확인
    latest = get_latest_timestamp(ticker)

    # 2. 기존 데이터 로드
    existing = load_bars_from_db(ticker, from_date, to_date)

    # 3. 새 데이터만 API에서 가져옴
    if latest:
        fetch_from = latest + 5분  # 5분봉 기준
    else:
        fetch_from = from_date

    # 4. yfinance에서 수집
    new_bars = _fetch_from_yfinance(ticker, fetch_from, to_date)

    # 5. DB에 저장
    save_bars(new_bars)

    # 6. 기존 + 신규 병합 반환
    return existing + new_bars
```

### TickerSelector

**위치**: `src/collector/ticker_selector.py`

```python
from src.collector.ticker_selector import TickerSelector

selector = TickerSelector(
    top_n_volume=50,     # 거래량 상위
    top_n_gainers=50,    # 상승률 상위
    min_price=5.0,       # 최소 $5
    min_volume=1_000_000 # 최소 거래량
)

# 전체 유니크 종목
tickers = selector.get_target_tickers()

# 카테고리별 조회
categories = selector.get_both_categories()
# {'volume': [...], 'gainers': [...]}
```

---

## 사용법

### 첫 실행 (전체 수집)

```bash
# 60일치 5분봉 수집 (권장)
python scripts/collect_historical.py --days 60

# 특정 종목만
python scripts/collect_historical.py --days 60 --tickers AAPL MSFT GOOGL

# 종목 수 제한
python scripts/collect_historical.py --days 60 --max-tickers 20
```

### 증분 수집 (정기 실행)

```bash
# --update: DB의 마지막 데이터 이후만 수집
python scripts/collect_historical.py --update
```

### Cron 설정 예시

```bash
# 매일 오전 6시에 증분 수집 (마켓 오픈 전)
0 6 * * 1-5 cd /path/to/FiveForFree && python scripts/collect_historical.py --update >> logs/collect.log 2>&1
```

### 프로그래밍 방식

```python
from datetime import datetime, timedelta
from src.collector.minute_bars import MinuteBarCollector
from src.collector.ticker_selector import TickerSelector

# 종목 선정
selector = TickerSelector()
tickers = selector.get_target_tickers()

# 수집기 초기화
collector = MinuteBarCollector(use_db=True)

# 60일치 수집
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

for ticker in tickers:
    bars = collector.get_bars(ticker, start_date, end_date)
    print(f"{ticker}: {len(bars)} bars")
```

---

## 제한사항

### Yahoo Finance 제한

| 항목 | 제한 |
|------|------|
| 1분봉 | 최근 7일 |
| 5분봉 | 최근 60일 |
| Rate Limit | 없음 (합리적 사용 시) |

### VWAP 근사값

Yahoo Finance는 VWAP을 제공하지 않아 근사값 사용:

```python
vwap = (high + low + close) / 3  # 실제 VWAP과 차이 있음
```

### Level 2 데이터 없음

Order Book 피처 (8개)는 현재 비활성:
- 무료 tier에서 Level 2 데이터 미제공
- Polygon.io 유료 플랜 필요

---

## 트러블슈팅

### 데이터가 수집되지 않음

1. **인터넷 연결 확인**
2. **종목 심볼 확인**: NASDAQ 종목만 지원
3. **마켓 시간 확인**: 휴일/장외 시간은 데이터 없음

### DB 연결 오류

```bash
python scripts/init_database.py
```

### 중복 데이터

증분 수집 시 중복은 자동 제거됨 (timestamp 기준 unique)

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `src/collector/minute_bars.py` | 분봉 수집 |
| `src/collector/ticker_selector.py` | 종목 선정 |
| `scripts/collect_historical.py` | 수집 스크립트 |
| `src/utils/database.py` | DB 모델 |
| `config/settings.py` | 설정 |

---

**Last Updated**: 2024-12-15
**Version**: 1.1
