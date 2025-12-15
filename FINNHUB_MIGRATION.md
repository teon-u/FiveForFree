# Yahoo Finance로 데이터 소스 전환

## ⚠️ Finnhub 무료 티어 제약사항 발견

Finnhub 무료 티어 테스트 중 **403 Forbidden** 에러 발생:
```
FinnhubAPIException(status_code: 403):
You don't have access to this resource.
```

### 원인
Finnhub 무료 티어는 **intraday candles (분봉 데이터) 접근이 제한**됨:
- ✅ Real-time quotes: 가능
- ✅ Daily candles: 가능
- ❌ **Intraday candles (1분, 5분, 15분 등)**: **무료 플랜에서 차단**

## ✅ 해결 방법: Yahoo Finance로 전환

### 최종 데이터 소스 구성
| 데이터 타입 | 소스 | 이유 |
|------------|------|------|
| **분봉 OHLCV** | **Yahoo Finance** | 무료, 1분봉 제공 |
| 실시간 Quote | Finnhub | 보조 |
| 시장 맥락 | Finnhub | SPY/QQQ/VXX |
| 종목 선정 | Yahoo Finance | 거래량/변동률 |

### Yahoo Finance 제약사항
- ✅ **완전 무료** (API 키 불필요)
- ✅ **1분봉**: 최근 7일
- ✅ **5분봉**: 최근 60일
- ⚠️ 간헐적 rate limiting (우회 가능)
- ⚠️ 15-20분 데이터 지연 가능

## 🔄 수정된 코드

### 1. `src/collector/minute_bars.py`
```python
# 변경 전: Finnhub
from src.collector.finnhub_client import get_finnhub_client
client = get_finnhub_client()
candles = client.get_candles(symbol, '5', from_ts, to_ts)

# 변경 후: Yahoo Finance
import yfinance as yf
ticker = yf.Ticker(symbol)
df = ticker.history(start=from_date, end=to_date, interval='1m')
```

### 2. 데이터 해상도
- Finnhub: 5분봉 (403 에러로 실패)
- **Yahoo Finance**: **1분봉** (7일) 또는 5분봉 (60일)

### 3. API 호출
- Finnhub: Rate limit 필요 (60 calls/min)
- Yahoo Finance: 제약 적지만 과도한 요청 시 차단 가능

## 📊 최종 아키텍처

```
데이터 수집 계층:
┌──────────────────────┐
│  Yahoo Finance (주)  │ ← 1분봉 OHLCV 데이터
└──────────────────────┘
         ↓
┌──────────────────────┐
│   Finnhub (보조)     │ ← 실시간 quote, 시장 맥락
└──────────────────────┘
         ↓
┌──────────────────────┐
│  Feature Engineer    │ ← 49개 피처 생성
└──────────────────────┘
```

## 🚀 테스트 방법

### 1. Yahoo Finance 테스트
```bash
python examples/yfinance_test.py
```

**예상 결과**:
```
✅ AAPL 실시간 데이터:
   현재가: $178.65
   거래량: 52,345,678

✅ 1분봉 데이터: 780개 포인트
   최신 종가: $178.45
```

### 2. 전체 시스템 테스트
```bash
# 1일치 데이터 수집 (Yahoo Finance)
python scripts/collect_historical.py --days 1

# 실시간 수집 테스트
python scripts/run_system.py
```

## ✅ 변경 요약

| 항목 | 이전 (Finnhub 전용) | 현재 (Yahoo Finance 주) |
|------|-------------------|------------------------|
| 분봉 데이터 | Finnhub 5분봉 (❌ 403) | **Yahoo 1분봉** (✅) |
| 데이터 범위 | 1년 | 7일 (1분봉) |
| API 키 | 필수 | **불필요** |
| 비용 | 무료 | **무료** |
| 제약 | 403 에러 | Rate limiting (적음) |

## 📝 Sources

- [Finnhub Stock Candles API](https://finnhub.io/docs/api/stock-candles)
- [Finnhub Intraday Limitations](https://github.com/finnhubio/Finnhub-API/issues/349)
- [Yahoo Finance Python (yfinance)](https://github.com/ranaroussi/yfinance)

---

**상태**: ✅ Yahoo Finance로 전환 완료
**마지막 업데이트**: 2025-12-15
**테스트**: 로컬에서 실행 필요
