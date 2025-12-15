# Data Collection

FiveForFree 데이터 수집 시스템의 상세 문서입니다.

## 목차

- [개요](#개요)
- [데이터 소스](#데이터-소스)
- [수집 모듈](#수집-모듈)
- [데이터 흐름](#데이터-흐름)
- [사용법](#사용법)
- [제한사항 및 주의점](#제한사항-및-주의점)
- [트러블슈팅](#트러블슈팅)

---

## 개요

데이터 수집 시스템은 NASDAQ 주식의 분봉 데이터를 수집하고 SQLite 데이터베이스에 저장합니다.

### 핵심 특징
- **증분 수집**: DB의 최신 데이터 이후만 가져옴
- **무료 데이터**: Yahoo Finance 기반 (API 키 불필요)
- **자동 간격 선택**: 기간에 따라 1분봉/5분봉 자동 선택

---

## 데이터 소스

### 1. Yahoo Finance (주 데이터 소스)

**사용 모듈**: `src/collector/minute_bars.py`

| 기능 | 제한 |
|------|------|
| 1분봉 데이터 | 최근 7일 |
| 5분봉 데이터 | 최근 60일 |
| API 키 | 불필요 |
| Rate Limit | 없음 (합리적 사용 시) |

**장점**:
- 완전 무료
- API 키 불필요
- 신뢰성 높음

**단점**:
- 1분봉 7일 제한 (장기 데이터 축적 어려움)
- VWAP 미제공 (근사값 계산)
- Level 2 데이터 없음

### 2. Finnhub (보조 데이터)

**사용 모듈**: `src/collector/finnhub_client.py`

| 기능 | 제한 |
|------|------|
| 실시간 호가 | 60 calls/min |
| 캔들 데이터 | 1년 히스토리 |
| 회사 정보 | 60 calls/min |

**현재 용도**:
- TickerSelector에서 종목 메트릭 조회 (보조)
- 실시간 호가 (선택적)

### 3. Polygon.io (미사용)

**사용 모듈**: `src/collector/polygon_client.py`

코드 구현되어 있으나 현재 비활성 상태.
유료 API 키 필요.

---

## 수집 모듈

### MinuteBarCollector

**위치**: `src/collector/minute_bars.py`

분봉 OHLCV 데이터를 Yahoo Finance에서 수집합니다.

```python
from src.collector.minute_bars import MinuteBarCollector
from datetime import datetime, timedelta

# 초기화
collector = MinuteBarCollector(use_db=True)

# 분봉 데이터 수집
bars = collector.get_bars(
    ticker="AAPL",
    from_date=datetime.now() - timedelta(days=5),
    to_date=datetime.now()
)

# DataFrame으로 변환
df = collector.get_bars_as_dataframe("AAPL", from_date, to_date)

# 여러 종목 배치 수집
result = collector.get_bars_batch(
    tickers=["AAPL", "MSFT", "GOOGL"],
    from_date=from_date,
    to_date=to_date
)
```

#### 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `get_bars()` | 단일 종목 분봉 수집 |
| `get_bars_batch()` | 여러 종목 배치 수집 |
| `get_bars_as_dataframe()` | DataFrame으로 반환 |
| `get_recent_bars()` | 최근 N분 데이터 |
| `get_latest_bar()` | 가장 최근 1개 바 |
| `save_bars()` | DB에 저장 |
| `load_bars_from_db()` | DB에서 로드 |

#### MinuteBar 데이터 구조

```python
@dataclass
class MinuteBar:
    ticker: str          # 종목 심볼
    timestamp: int       # Unix timestamp (ms)
    open: float          # 시가
    high: float          # 고가
    low: float           # 저가
    close: float         # 종가
    volume: float        # 거래량
    vwap: float          # VWAP (근사값)
    transactions: int    # 거래 건수 (미제공)
```

### TickerSelector

**위치**: `src/collector/ticker_selector.py`

대상 종목을 선정합니다.

```python
from src.collector.ticker_selector import TickerSelector

# 초기화
selector = TickerSelector(
    top_n_volume=50,     # 거래량 상위
    top_n_gainers=50,    # 상승률 상위
    min_price=5.0,       # 최소 주가
    min_volume=1_000_000 # 최소 거래량
)

# 거래량 상위 종목
top_volume = selector.get_top_by_volume()

# 상승률 상위 종목
top_gainers = selector.get_top_by_gainers()

# 양쪽 카테고리
both = selector.get_both_categories()
# {'volume': [...], 'gainers': [...]}

# 전체 유니크 종목 리스트
tickers = selector.get_target_tickers()
```

#### 종목 유니버스

NASDAQ 100 + 고변동성 종목 총 약 110개:
- 메가캡: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META...
- 대형 기술주: AMD, INTC, CSCO, ADBE, ORCL...
- 고변동성: COIN, RIVN, LCID, HOOD, SOFI, PLTR...

### FinnhubClient

**위치**: `src/collector/finnhub_client.py`

```python
from src.collector.finnhub_client import get_finnhub_client

client = get_finnhub_client()

# 실시간 호가
quote = client.get_quote("AAPL")
# {'c': 150.0, 'h': 152.0, 'l': 149.0, ...}

# 캔들 데이터
candles = client.get_candles("AAPL", "5", from_ts, to_ts)

# 회사 정보
profile = client.get_company_profile("AAPL")
```

---

## 데이터 흐름

### 히스토리 데이터 수집

```
scripts/collect_historical.py
         │
         ▼
TickerSelector.get_target_tickers()
         │ (종목 리스트)
         ▼
MinuteBarCollector.get_bars()
         │
         ├──▶ get_latest_timestamp() - DB 최신 시점 확인
         │
         ├──▶ load_bars_from_db() - 기존 데이터 로드
         │
         ├──▶ _fetch_from_yfinance() - 새 데이터 수집
         │         │
         │         ├── days <= 7: 1분봉
         │         └── days > 7: 5분봉
         │
         └──▶ save_bars() - DB 저장
                  │
                  ▼
          SQLite (minute_bars)
```

### 실시간 데이터 수집

```
실시간 요청
    │
    ▼
MinuteBarCollector.get_recent_bars(minutes=60)
    │
    ├── DB에서 최근 데이터 로드
    │
    └── 필요 시 yfinance에서 추가 수집
         │
         ▼
    FeatureEngineer로 전달
```

---

## 사용법

### 1. 히스토리 데이터 수집

```bash
# 기본 실행 (30일, 상위 종목)
python scripts/collect_historical.py

# 60일 수집
python scripts/collect_historical.py --days 60

# 특정 종목만
python scripts/collect_historical.py --tickers AAPL MSFT GOOGL

# 업데이트 모드 (새 데이터만)
python scripts/collect_historical.py --update

# 최대 종목 수 제한
python scripts/collect_historical.py --max-tickers 10

# 상세 로그
python scripts/collect_historical.py --verbose
```

### 2. 프로그래밍 방식

```python
from datetime import datetime, timedelta
from src.collector.minute_bars import MinuteBarCollector
from src.collector.ticker_selector import TickerSelector

# 종목 선정
selector = TickerSelector()
tickers = selector.get_target_tickers()

# 데이터 수집
collector = MinuteBarCollector(use_db=True)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

for ticker in tickers:
    bars = collector.get_bars(ticker, start_date, end_date)
    print(f"{ticker}: {len(bars)} bars collected")
```

---

## 제한사항 및 주의점

### Yahoo Finance 제한

| 간격 | 최대 기간 | 비고 |
|------|----------|------|
| 1분 | 7일 | 고해상도 |
| 5분 | 60일 | 자동 전환 |

**중요**: 7일 넘는 기간 요청 시 자동으로 5분봉으로 전환됩니다.

### VWAP 계산

Yahoo Finance는 VWAP을 제공하지 않아 근사값을 사용합니다:

```python
vwap = (high + low + close) / 3
```

실제 VWAP과 차이가 있을 수 있습니다.

### 증분 수집 로직

```python
# 증분 수집 흐름
1. DB에서 해당 종목의 최신 timestamp 조회
2. 최신 timestamp 이후 데이터만 yfinance에서 수집
3. 새 데이터를 DB에 저장
4. 기존 + 새 데이터 병합하여 반환
```

### Rate Limiting

| 소스 | 제한 | 대응 |
|------|------|------|
| Yahoo Finance | 없음 | - |
| Finnhub | 60 calls/min | 1초 딜레이 |
| Polygon.io | 플랜별 | 지수 백오프 |

---

## 트러블슈팅

### 데이터가 수집되지 않음

1. **인터넷 연결 확인**
2. **종목 심볼 확인**: NYSE는 지원 안 됨 (NASDAQ만)
3. **시간대 확인**: EST 기준 마켓 시간

### 빈 데이터 반환

```python
# 마켓 시간 외 또는 휴일
bars = collector.get_bars("AAPL", from_date, to_date)
if not bars:
    print("No data - market may be closed")
```

### DB 연결 오류

```bash
# DB 초기화
python scripts/init_database.py
```

### Finnhub Rate Limit

```
Error: 429 Too Many Requests
```

해결: `API_CALL_DELAY` 설정값 증가 (기본 1.0초)

---

## 성능 고려사항

### 배치 수집 권장

```python
# 좋은 예: 배치로 한 번에
collector.get_bars_batch(tickers, from_date, to_date)

# 나쁜 예: 개별 호출 반복
for ticker in tickers:
    collector.get_bars(ticker, from_date, to_date)
```

### 메모리 사용량

| 데이터량 | 메모리 |
|---------|--------|
| 1,000 bars | ~500 KB |
| 10,000 bars | ~5 MB |
| 100,000 bars | ~50 MB |

### 저장 공간

| 종목 수 | 30일 데이터 | 예상 DB 크기 |
|--------|------------|-------------|
| 10 | ~1,950 bars/종목 | ~10 MB |
| 50 | ~1,950 bars/종목 | ~50 MB |
| 100 | ~1,950 bars/종목 | ~100 MB |

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `src/collector/minute_bars.py` | 분봉 수집 |
| `src/collector/ticker_selector.py` | 종목 선정 |
| `src/collector/finnhub_client.py` | Finnhub API |
| `src/collector/polygon_client.py` | Polygon API |
| `scripts/collect_historical.py` | 수집 스크립트 |
| `src/utils/database.py` | DB 모델 |

---

**Last Updated**: 2024-12-15
**Version**: 1.0
