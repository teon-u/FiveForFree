# FiveForFree 코드 구조 개선 제안서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**문서 버전**: v1.0

---

## 1. 현재 구조 분석

### 1.1 잘 설계된 부분

| 영역 | 설명 | 평가 |
|------|------|------|
| 모듈화 | src/ 하위 기능별 분리 | 우수 |
| 관심사 분리 | collector/processor/models/trainer/predictor/api | 우수 |
| 설정 관리 | config/settings.py + pydantic-settings | 양호 |
| 테스트 구조 | tests/ 디렉토리, pytest 설정 | 양호 |

### 1.2 개선 필요 영역

| 영역 | 현재 상태 | 권장 개선 |
|------|-----------|-----------|
| 의존성 주입 | 하드코딩된 의존성 | DI 컨테이너 도입 |
| 데이터 소스 | 3개 분산 (Yahoo, Finnhub, Polygon) | 통합 어댑터 패턴 |
| 에러 처리 | 개별 모듈 처리 | 중앙 집중식 에러 핸들링 |
| 캐싱 | 미구현 | Redis 또는 인메모리 캐시 |

---

## 2. 구조 개선 제안

### 2.1 데이터 소스 통합 어댑터 패턴

**현재 문제점**:
- `finnhub_client.py`, `polygon_client.py`가 각각 독립적
- 호출부에서 데이터 소스 선택 로직이 분산됨

**제안**:
```python
# src/collector/data_adapter.py
from abc import ABC, abstractmethod

class DataSourceAdapter(ABC):
    @abstractmethod
    async def get_quote(self, ticker: str) -> Quote:
        pass

    @abstractmethod
    async def get_bars(self, ticker: str, interval: str) -> list[Bar]:
        pass

class FinnhubAdapter(DataSourceAdapter):
    def __init__(self, client: FinnhubClient):
        self.client = client

    async def get_quote(self, ticker: str) -> Quote:
        return await self.client.get_quote(ticker)

class YahooAdapter(DataSourceAdapter):
    # Yahoo Finance 구현

class DataSourceFactory:
    @staticmethod
    def create(source_type: str) -> DataSourceAdapter:
        adapters = {
            "finnhub": FinnhubAdapter,
            "yahoo": YahooAdapter,
            "polygon": PolygonAdapter,
        }
        return adapters[source_type]()
```

### 2.2 의존성 주입 개선

**현재 문제점**:
- 모듈 간 직접 import로 결합도 높음
- 테스트 시 모킹 어려움

**제안**:
```python
# src/container.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    database = providers.Singleton(
        Database,
        connection_string=config.database.url,
    )

    model_manager = providers.Factory(
        ModelManager,
        database=database,
    )

    predictor = providers.Factory(
        RealtimePredictor,
        model_manager=model_manager,
    )
```

### 2.3 중앙 집중식 에러 처리

**제안**:
```python
# src/utils/exceptions.py
class FiveForFreeException(Exception):
    """베이스 예외 클래스"""
    pass

class DataCollectionError(FiveForFreeException):
    """데이터 수집 실패"""
    pass

class ModelTrainingError(FiveForFreeException):
    """모델 학습 실패"""
    pass

class PredictionError(FiveForFreeException):
    """예측 실패"""
    pass

# src/api/middleware/error_handler.py
@app.exception_handler(FiveForFreeException)
async def handle_app_exception(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": exc.__class__.__name__, "message": str(exc)}
    )
```

### 2.4 캐싱 레이어 도입

**제안**:
```python
# src/utils/cache.py
from functools import lru_cache
from cachetools import TTLCache

class CacheManager:
    def __init__(self, ttl: int = 300):
        self.cache = TTLCache(maxsize=1000, ttl=ttl)

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value

    def invalidate(self, key: str):
        self.cache.pop(key, None)

# 데코레이터 활용
def cached(ttl: int = 300):
    def decorator(func):
        cache = TTLCache(maxsize=100, ttl=ttl)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator
```

---

## 3. 디렉토리 구조 개선안

### 3.1 현재 구조
```
src/
├── api/
├── backtester/
├── collector/
├── models/
├── predictor/
├── processor/
├── trainer/
└── utils/
```

### 3.2 제안 구조
```
src/
├── core/                    # 핵심 도메인 로직
│   ├── entities/            # 도메인 엔티티
│   ├── repositories/        # 저장소 인터페이스
│   └── services/            # 비즈니스 로직
├── infrastructure/          # 외부 인프라
│   ├── database/            # DB 구현체
│   ├── cache/               # 캐시 구현체
│   └── external/            # 외부 API (Finnhub, Yahoo 등)
├── application/             # 애플리케이션 계층
│   ├── commands/            # CQRS Command
│   ├── queries/             # CQRS Query
│   └── handlers/            # 이벤트 핸들러
├── presentation/            # 프레젠테이션 계층
│   ├── api/                 # REST API
│   └── websocket/           # WebSocket
└── ml/                      # ML 전용 모듈
    ├── models/              # 모델 정의
    ├── training/            # 학습 로직
    ├── prediction/          # 예측 로직
    └── evaluation/          # 평가/백테스팅
```

---

## 4. 우선순위별 개선 로드맵

### Phase 1: 긴급 (1-2주)
1. [ ] 중앙 집중식 에러 처리 도입
2. [ ] 데이터 소스 어댑터 패턴 구현
3. [ ] Pydantic V2 마이그레이션 (개발팀장 협업)

### Phase 2: 중요 (3-4주)
1. [ ] 의존성 주입 컨테이너 도입
2. [ ] 캐싱 레이어 구현
3. [ ] 로깅 표준화

### Phase 3: 장기 (1-2개월)
1. [ ] 디렉토리 구조 리팩토링
2. [ ] CQRS 패턴 도입 검토
3. [ ] 마이크로서비스 분리 가능성 검토

---

## 5. 결론

FiveForFree 프로젝트는 기본적인 모듈화가 잘 되어있으나, 확장성과 유지보수성을 위해
위 개선사항들을 단계적으로 적용할 것을 권장합니다.

특히 **데이터 소스 통합**과 **에러 처리 표준화**는 즉시 적용 가능하며
코드 품질을 크게 향상시킬 수 있습니다.

---

*이 문서는 분석팀장이 작성하였습니다.*
