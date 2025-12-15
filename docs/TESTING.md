# Testing Guide

이 문서는 FiveForFree 프로젝트의 테스트 실행 방법과 테스트 작성 가이드를 설명합니다.

## 목차

- [테스트 환경 설정](#테스트-환경-설정)
- [테스트 실행 방법](#테스트-실행-방법)
- [테스트 구조](#테스트-구조)
- [테스트 작성 가이드](#테스트-작성-가이드)
- [CI/CD 파이프라인](#cicd-파이프라인)
- [코드 품질 도구](#코드-품질-도구)

---

## 테스트 환경 설정

### Backend (Python)

1. **가상환경 활성화** (권장)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

### Frontend (JavaScript/React)

```bash
cd frontend
npm install
```

---

## 테스트 실행 방법

### Backend 테스트

#### 기본 테스트 실행
```bash
# 프로젝트 루트에서 실행
pytest
```

#### 상세 출력과 함께 실행
```bash
pytest -v
```

#### 특정 테스트 파일 실행
```bash
pytest tests/test_config.py
pytest tests/test_models.py
```

#### 특정 테스트 함수 실행
```bash
pytest tests/test_config.py::test_settings_loads_correctly
```

#### 마커별 테스트 실행
```bash
# 단위 테스트만 실행
pytest -m unit

# 통합 테스트만 실행
pytest -m integration

# 느린 테스트 제외
pytest -m "not slow"
```

#### 코드 커버리지와 함께 실행
```bash
# 콘솔에 커버리지 출력
pytest --cov=src

# HTML 리포트 생성
pytest --cov=src --cov-report=html

# XML 리포트 생성 (CI/CD용)
pytest --cov=src --cov-report=xml
```

커버리지 HTML 리포트는 `htmlcov/index.html`에서 확인할 수 있습니다.

#### 실패 시 즉시 중단
```bash
pytest -x
```

#### 마지막 실패한 테스트만 재실행
```bash
pytest --lf
```

### Frontend 테스트

#### 린트 검사
```bash
cd frontend
npm run lint
```

#### 빌드 테스트
```bash
cd frontend
npm run build
```

---

## 테스트 구조

### 디렉토리 구조

```
tests/
├── __init__.py          # 테스트 패키지 초기화
├── test_config.py       # 설정 관련 테스트
├── test_models.py       # 모델 관련 테스트
└── (추가 테스트 파일)
```

### pytest.ini 설정

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short
markers:
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### 기존 테스트 파일

| 파일 | 설명 |
|------|------|
| `test_config.py` | 설정 로드 및 기본값 검증 |
| `test_models.py` | 예측 모델 기본 기능 테스트 |

---

## 테스트 작성 가이드

### 기본 테스트 템플릿

```python
import pytest

class TestClassName:
    """클래스에 대한 테스트 그룹"""

    def test_function_name(self):
        """테스트 설명"""
        # Arrange (준비)
        expected = "expected_value"

        # Act (실행)
        result = some_function()

        # Assert (검증)
        assert result == expected
```

### 마커 사용

```python
import pytest

@pytest.mark.unit
def test_quick_unit_test():
    """빠른 단위 테스트"""
    pass

@pytest.mark.integration
def test_integration_with_db():
    """데이터베이스 통합 테스트"""
    pass

@pytest.mark.slow
def test_heavy_computation():
    """시간이 오래 걸리는 테스트"""
    pass
```

### Fixture 사용

```python
import pytest

@pytest.fixture
def sample_data():
    """테스트 데이터 fixture"""
    return {
        "ticker": "AAPL",
        "price": 150.0,
        "volume": 1000000
    }

def test_with_fixture(sample_data):
    """fixture를 사용한 테스트"""
    assert sample_data["ticker"] == "AAPL"
```

### Mock 사용 (외부 API 테스트)

```python
from unittest.mock import Mock, patch

@patch('src.collector.finnhub_client.FinnhubClient')
def test_api_call(mock_client):
    """외부 API 호출 모킹"""
    mock_client.return_value.get_quote.return_value = {"c": 150.0}
    # 테스트 로직
```

### 권장 테스트 영역

현재 테스트 커버리지가 부족한 영역:

1. **API 엔드포인트** (`src/api/routes/`)
   ```python
   # tests/test_api_health.py
   from fastapi.testclient import TestClient
   from src.api.main import app

   client = TestClient(app)

   def test_health_endpoint():
       response = client.get("/health")
       assert response.status_code == 200
   ```

2. **데이터 수집기** (`src/collector/`)
3. **피처 엔지니어링** (`src/processor/`)
4. **백테스터** (`src/backtester/`)
5. **예측기** (`src/predictor/`)

---

## CI/CD 파이프라인

GitHub Actions를 통해 자동으로 테스트가 실행됩니다.

### 파이프라인 단계

1. **Backend Tests**
   - Python 3.10 환경 설정
   - Black 코드 포맷 검사
   - Flake8 린트 검사
   - Pytest 실행 및 커버리지 측정
   - Codecov 업로드

2. **Frontend Tests**
   - Node.js 18 환경 설정
   - ESLint 린트 검사
   - Vite 빌드

3. **Docker Build**
   - 테스트 통과 후 Docker 이미지 빌드

### 로컬에서 CI 검사 실행

```bash
# Backend 전체 검사
black --check .
flake8 src/ --max-line-length=100
pytest tests/ -v --cov=src

# Frontend 전체 검사
cd frontend
npm run lint
npm run build
```

---

## 코드 품질 도구

### Black (코드 포맷터)

```bash
# 포맷 검사만
black --check .

# 자동 포맷팅
black .

# 특정 파일/디렉토리
black src/
```

### Flake8 (린터)

```bash
# 기본 실행
flake8 src/

# 라인 길이 제한 설정
flake8 src/ --max-line-length=100

# 특정 에러 무시
flake8 src/ --ignore=E501,W503
```

### ESLint (Frontend)

```bash
cd frontend

# 린트 검사
npm run lint

# 자동 수정
npm run lint -- --fix
```

---

## 문제 해결

### 일반적인 오류

#### ModuleNotFoundError
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
```

#### ImportError: No module named 'src'
```bash
# 프로젝트 루트에서 실행하세요
cd F:\Git\FiveForFree
pytest
```

#### pytest가 테스트를 찾지 못함
- 파일명이 `test_`로 시작하는지 확인
- 함수명이 `test_`로 시작하는지 확인
- `tests/` 디렉토리에 `__init__.py` 존재 확인

### 디버깅 팁

```bash
# 출력 캡처 비활성화
pytest -s

# 디버거 진입 (실패 시)
pytest --pdb

# 특정 키워드로 테스트 필터링
pytest -k "config"
```

---

## 추가 리소스

- [pytest 공식 문서](https://docs.pytest.org/)
- [pytest-cov 문서](https://pytest-cov.readthedocs.io/)
- [FastAPI 테스팅 가이드](https://fastapi.tiangolo.com/tutorial/testing/)
- [Black 문서](https://black.readthedocs.io/)
- [Flake8 문서](https://flake8.pycqa.org/)
