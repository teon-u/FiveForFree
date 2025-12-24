# FiveForFree API Reference

**작성일**: 2025-12-21
**작성자**: 분석팀장
**API 버전**: v1.0
**Base URL**: `http://localhost:8000`

---

## 1. API 개요

### 1.1 인증

현재 버전은 인증을 요구하지 않습니다. 프로덕션 환경에서는 API 키 또는 JWT 인증 추가를 권장합니다.

### 1.2 Rate Limiting

| 엔드포인트 | 제한 |
|------------|------|
| REST API | 100 req/min |
| WebSocket | 10 connections/IP |

### 1.3 응답 형식

모든 응답은 JSON 형식입니다.

```json
{
  "data": { ... },
  "timestamp": "2025-12-21T13:00:00Z",
  "status": "success"
}
```

---

## 2. REST API 엔드포인트

### 2.1 Health Check

#### GET /api/health

시스템 상태 확인

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": "ok",
    "models": "ok",
    "scheduler": "ok"
  }
}
```

#### GET /api/health/ready

서비스 준비 상태 확인

#### GET /api/health/live

서비스 생존 상태 확인

---

### 2.2 Tickers

#### GET /api/tickers

모든 활성 티커 목록 조회

**Parameters**
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| limit | int | N | 반환할 최대 개수 (기본: 100) |
| offset | int | N | 시작 위치 (기본: 0) |
| category | string | N | 카테고리 필터 (volume/gainers) |

**Response**
```json
{
  "tickers": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "category": "volume",
      "last_updated": "2025-12-21T13:00:00Z"
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

#### GET /api/tickers/{symbol}

특정 티커 상세 정보 조회

**Response**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "current_price": 185.50,
  "day_change": 2.35,
  "day_change_pct": 1.28,
  "volume": 45000000,
  "avg_volume": 50000000,
  "market_cap": 2900000000000
}
```

---

### 2.3 Predictions

#### GET /api/predictions/{symbol}

특정 티커의 예측 결과 조회

**Parameters**
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| include_all_models | bool | N | 모든 모델 결과 포함 (기본: false) |
| include_context | bool | N | 시장 맥락 포함 (기본: false) |

**Response**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-12-21T13:00:00Z",
  "current_price": 185.50,
  "prediction": {
    "up_probability": 0.75,
    "down_probability": 0.25,
    "best_up_model": "xgboost",
    "best_down_model": "lightgbm",
    "up_model_accuracy": 0.68,
    "down_model_accuracy": 0.72,
    "trading_signal": "BUY",
    "confidence_level": "HIGH"
  }
}
```

#### POST /api/predictions

배치 예측 요청

**Request Body**
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT"],
  "include_all_models": false
}
```

**Response**
```json
{
  "predictions": [
    { "symbol": "AAPL", "up_probability": 0.75, ... },
    { "symbol": "GOOGL", "up_probability": 0.62, ... },
    { "symbol": "MSFT", "up_probability": 0.58, ... }
  ]
}
```

#### GET /api/predictions/top/opportunities

상위 기회 티커 조회

**Parameters**
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| tickers | array | N | 대상 티커 목록 |
| direction | string | N | up/down (기본: up) |
| top_n | int | N | 상위 N개 (기본: 5) |
| min_probability | float | N | 최소 확률 (기본: 0.70) |

---

### 2.4 Models

#### GET /api/models

모든 모델 성능 조회

#### GET /api/models/{symbol}

특정 티커의 모델 성능 조회

**Response**
```json
{
  "symbol": "AAPL",
  "models": {
    "up": {
      "xgboost": { "accuracy_50h": 0.72, "predictions_count": 48 },
      "lightgbm": { "accuracy_50h": 0.68, "predictions_count": 48 },
      "lstm": { "accuracy_50h": 0.65, "predictions_count": 48 },
      "transformer": { "accuracy_50h": 0.70, "predictions_count": 48 },
      "ensemble": { "accuracy_50h": 0.74, "predictions_count": 48 }
    },
    "down": { ... }
  },
  "best_model": {
    "up": "ensemble",
    "down": "xgboost"
  }
}
```

#### GET /api/models/{symbol}/stats

모델 통계 조회

#### GET /api/models/summary/overall

전체 모델 요약

---

### 2.5 Prices

#### GET /api/prices/{symbol}

가격 히스토리 조회

**Parameters**
| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| interval | string | N | 1m/5m/15m/1h/1d (기본: 1m) |
| limit | int | N | 반환할 개수 (기본: 60) |

**Response**
```json
{
  "symbol": "AAPL",
  "interval": "1m",
  "bars": [
    {
      "timestamp": "2025-12-21T13:00:00Z",
      "open": 185.00,
      "high": 185.75,
      "low": 184.80,
      "close": 185.50,
      "volume": 150000
    }
  ]
}
```

---

## 3. WebSocket API

### 3.1 연결

**URL**: `ws://localhost:8000/ws`

### 3.2 메시지 타입

#### Client → Server

| 타입 | 설명 | 예시 |
|------|------|------|
| subscribe | 티커 구독 | `{"type": "subscribe", "tickers": ["AAPL"]}` |
| unsubscribe | 구독 해제 | `{"type": "unsubscribe", "tickers": ["AAPL"]}` |
| predict | 즉시 예측 요청 | `{"type": "predict", "ticker": "AAPL"}` |
| ping | 연결 유지 | `{"type": "ping"}` |

#### Server → Client

| 타입 | 설명 |
|------|------|
| connected | 연결 성공 |
| prediction_update | 예측 업데이트 |
| price_update | 가격 업데이트 |
| heartbeat | 하트비트 |
| error | 에러 |

---

## 4. 에러 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 404 | 리소스 없음 |
| 422 | 유효성 검증 실패 |
| 429 | Rate Limit 초과 |
| 500 | 서버 에러 |
| 503 | 서비스 불가 |

---

## 5. SDK 예제

### 5.1 Python

```python
import httpx

class FiveForFreeClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = httpx.Client(base_url=base_url)

    def get_prediction(self, ticker: str) -> dict:
        response = self.client.get(f"/api/predictions/{ticker}")
        return response.json()

    def get_top_opportunities(self, direction: str = "up", top_n: int = 5) -> list:
        response = self.client.get(
            "/api/predictions/top/opportunities",
            params={"direction": direction, "top_n": top_n}
        )
        return response.json()
```

### 5.2 TypeScript

```typescript
class FiveForFreeClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async getPrediction(ticker: string): Promise<Prediction> {
    const response = await fetch(`${this.baseUrl}/api/predictions/${ticker}`);
    return response.json();
  }

  async getTopOpportunities(direction: string = "up", topN: number = 5): Promise<Prediction[]> {
    const params = new URLSearchParams({ direction, top_n: topN.toString() });
    const response = await fetch(`${this.baseUrl}/api/predictions/top/opportunities?${params}`);
    return response.json();
  }
}
```

---

*이 문서는 분석팀장이 작성하였습니다.*
