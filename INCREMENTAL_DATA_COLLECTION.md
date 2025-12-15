# 증분 데이터 수집 (Incremental Data Collection)

## 🎯 목표

종목이 매일 변경되는 경우(예: 수요일 포함 → 목요일 제외 → 금요일 포함) 데이터를 **중복 다운로드하지 않고** 효율적으로 수집하는 시스템 구현.

## ✅ 구현된 기능

### 1. 데이터베이스 기반 증분 수집

`MinuteBarCollector`가 SQLite 데이터베이스를 활용하여:
- 기존에 저장된 데이터는 **재다운로드하지 않음**
- 마지막 저장 시점 이후의 **신규 데이터만 추가로 수집**
- 종목이 목록에서 제외되어도 **과거 데이터는 유지**

### 2. 주요 메서드

#### `get_latest_timestamp(ticker: str) -> Optional[datetime]`
- 데이터베이스에서 특정 종목의 **가장 최근 타임스탬프** 조회
- 없으면 `None` 반환

#### `load_bars_from_db(ticker, from_date, to_date) -> List[MinuteBar]`
- 데이터베이스에서 기존 데이터 로드
- 요청 범위 내의 모든 저장된 분봉 데이터 반환

#### `save_bars(bars: List[MinuteBar]) -> int`
- 새로 수집한 데이터를 데이터베이스에 저장
- 중복 체크: 동일한 `(symbol, timestamp)` 조합은 저장하지 않음
- 저장된 개수 반환

#### `get_bars(ticker, from_date, to_date) -> List[MinuteBar]`
**증분 수집 로직 (6단계):**

1. **데이터베이스 확인**: 해당 종목의 최신 타임스탬프 조회
2. **기존 데이터 로드**: DB에서 요청 범위의 기존 데이터 가져오기
3. **필요 구간 계산**:
   - 기존 데이터가 최신이면 → Yahoo Finance 호출 **스킵**
   - 최신 타임스탬프 이후만 → **신규 데이터만 fetch**
4. **Yahoo Finance 호출**: 신규 구간만 다운로드
5. **데이터베이스 저장**: 새로 받은 데이터를 DB에 저장
6. **결합 및 반환**: 기존 + 신규 데이터 합쳐서 반환

## 📊 예시 시나리오

### 시나리오: AAPL 종목의 주간 데이터 수집

**수요일 (2025-12-17)**
```python
collector = MinuteBarCollector()
bars = collector.get_bars("AAPL", from_date="2025-12-10", to_date="2025-12-17")
# ✅ Yahoo Finance에서 7일치 데이터 다운로드 (1,950개 분봉)
# ✅ 데이터베이스에 1,950개 바 저장
```

**목요일 (2025-12-18)**
AAPL이 Top 50 목록에서 **제외됨** → 수집 안함
**BUT**: 데이터베이스의 기존 데이터는 **삭제되지 않음**

**금요일 (2025-12-19)**
AAPL이 다시 Top 50 목록에 **포함됨**
```python
bars = collector.get_bars("AAPL", from_date="2025-12-10", to_date="2025-12-19")
# ✅ DB에서 2025-12-10 ~ 2025-12-17 데이터 로드 (1,950개)
# ✅ 최신 타임스탬프: 2025-12-17 16:00:00
# ✅ Yahoo Finance에서 2025-12-18 ~ 2025-12-19만 다운로드 (390개)
# ✅ 새 데이터 390개만 DB에 저장
# ✅ 총 2,340개 바 반환 (1,950 + 390)
```

**결과**:
- ❌ 중복 다운로드 없음 (1,950개 재다운로드 회피)
- ✅ Yahoo Finance API 호출 최소화 (rate limit 회피)
- ✅ 데이터 무결성 보장 (삭제 없음, 이어붙이기)

## 🔧 사용 방법

### 기본 사용 (증분 수집 활성화)
```python
from src.collector import MinuteBarCollector
from datetime import datetime, timedelta

collector = MinuteBarCollector(use_db=True)  # 기본값

# 데이터 수집 (자동으로 증분 수집)
bars = collector.get_bars(
    ticker="AAPL",
    from_date=datetime.now() - timedelta(days=7),
    to_date=datetime.now()
)

# 로그 출력 예시:
# AAPL: Found existing data up to 2025-12-17 16:00:00, fetching only new data from 2025-12-18
# Fetched 390 1m bars from Yahoo Finance for AAPL
# Saved 390 new bars for AAPL to database
# AAPL: Total bars returned: 2340 (existing: 1950, new: 390)
```

### 증분 수집 비활성화 (테스트용)
```python
collector = MinuteBarCollector(use_db=False)

# 항상 Yahoo Finance에서 전체 다운로드 (DB 사용 안함)
bars = collector.get_bars("AAPL", from_date, to_date)
```

## 🗄️ 데이터베이스 스키마

### `tickers` 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | Primary Key |
| symbol | VARCHAR(10) | 종목 심볼 (UNIQUE) |
| name | VARCHAR(255) | 회사명 |
| is_active | BOOLEAN | 현재 활성 여부 |
| added_at | DATETIME | 최초 추가 시각 |
| last_updated | DATETIME | 마지막 업데이트 |

### `minute_bars` 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | Primary Key |
| ticker_id | INTEGER | Foreign Key → tickers.id |
| symbol | VARCHAR(10) | 종목 심볼 (인덱스) |
| timestamp | DATETIME | 분봉 시각 (인덱스) |
| open | FLOAT | 시가 |
| high | FLOAT | 고가 |
| low | FLOAT | 저가 |
| close | FLOAT | 종가 |
| volume | BIGINT | 거래량 |
| vwap | FLOAT | 가중평균가 |
| trade_count | INT | 거래 횟수 (NULL) |
| created_at | DATETIME | DB 저장 시각 |

**인덱스**:
- `UNIQUE(symbol, timestamp)` ← 중복 방지
- `INDEX(ticker_id, timestamp)`
- `INDEX(timestamp)`

## ⚡ 성능 이점

### Yahoo Finance Rate Limit 회피
- **문제**: Yahoo Finance는 과도한 요청 시 차단 (IP 단위)
- **해결**:
  - 기존 데이터는 DB에서 즉시 로드 (API 호출 0회)
  - 신규 데이터만 최소한으로 fetch
  - 예시: 100개 종목 × 7일 = 700 API 호출 → **신규 종목만** 호출

### 저장 공간 효율성
- SQLite 데이터베이스 크기 예상치:
  - 1개 종목 × 7일 × 390분/일 = 2,730개 바
  - 1개 바 ≈ 80 bytes
  - 100개 종목 ≈ **21 MB** (무시 가능한 크기)
- 결론: **용량 걱정 없이 모든 과거 데이터 보관 가능**

### 데이터 일관성
- 종목이 목록에서 빠졌다가 다시 들어와도 **데이터 연속성 보장**
- 백테스팅 시 **완전한 히스토리** 확보

## 🔍 동작 확인

### 로그 메시지 예시
```
# 최초 수집
✅ AAPL: No existing data in database
✅ Fetching AAPL 1m bars from Yahoo Finance: 2025-12-10 to 2025-12-17
✅ Fetched 1950 1m bars from Yahoo Finance for AAPL
✅ Saved 1950 new bars for AAPL to database
✅ AAPL: Total bars returned: 1950 (existing: 0, new: 1950)

# 증분 수집 (2일 후)
✅ AAPL: Latest stored timestamp: 2025-12-17 16:00:00
✅ AAPL: Found existing data up to 2025-12-17 16:00:00, fetching only new data from 2025-12-18
✅ Loaded 1950 bars for AAPL from database (2025-12-10 to 2025-12-17)
✅ Fetching AAPL 1m bars from Yahoo Finance: 2025-12-18 to 2025-12-19
✅ Fetched 780 1m bars from Yahoo Finance for AAPL
✅ Saved 780 new bars for AAPL to database
✅ AAPL: Total bars returned: 2730 (existing: 1950, new: 780)

# 이미 모든 데이터가 있는 경우
✅ AAPL: All requested data (2730 bars) available in database, no fetch needed
```

## 📝 주의사항

### 1. 데이터베이스 초기화
```python
from src.utils.database import init_db

# 최초 실행 시 테이블 생성
init_db()
```

### 2. 타임존 처리
- Yahoo Finance는 **미국 동부 시간 (ET)** 기준
- 데이터베이스 저장 시 **UTC 변환** 권장
- 현재 구현: `datetime.timestamp()` 사용 (UTC 기준)

### 3. 시장 휴장일
- 주말, 공휴일 데이터는 없음
- `get_bars()` 호출 시 빈 리스트 반환 가능
- 로그: `"No data returned from Yahoo Finance for AAPL"`

### 4. 데이터 갭 처리
- 장 중 데이터 누락 시 빈 구간 발생 가능
- 피처 엔지니어링에서 `fillna()` 처리 필요

## 🚀 다음 단계

### 1. 스케줄러 통합
```python
from apscheduler.schedulers.background import BackgroundScheduler
from src.collector import TickerSelector, MinuteBarCollector

def collect_data():
    selector = TickerSelector()
    collector = MinuteBarCollector(use_db=True)

    # 볼륨 + 상승률 종목 모두 수집
    categories = selector.get_both_categories()
    all_tickers = set()

    for metrics_list in categories.values():
        all_tickers.update(m.ticker for m in metrics_list)

    # 증분 수집
    for ticker in all_tickers:
        bars = collector.get_bars(ticker, from_date, to_date)
        logger.info(f"Collected {len(bars)} bars for {ticker}")

scheduler = BackgroundScheduler()
scheduler.add_job(collect_data, 'interval', hours=1)
scheduler.start()
```

### 2. UI 통합
- 프론트엔드에서 `volume` / `gainers` 토글 버튼 구현
- API 엔드포인트: `/api/tickers?category=volume` 또는 `?category=gainers`

---

**상태**: ✅ 구현 완료
**마지막 업데이트**: 2025-12-15
**테스트**: 로컬 환경에서 실행 필요
