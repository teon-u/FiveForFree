# 종목 간 상관관계 및 포트폴리오 다각화 구현 방안

## 1. 현황 분석

### 현재 구현된 것
| 기능 | 상태 | 파일 |
|------|------|------|
| 개별 리스크 관리 | 완전 구현 | `risk_manager.py` |
| Kelly/ATR 포지션 사이징 | 완전 구현 | `risk_manager.py` |
| 섹터 분류 및 추적 | 부분 구현 | `enhanced_features.py` |
| 성과 기반 포트폴리오 할당 | 부분 구현 | `investment_strategy.py` |
| SPY/QQQ 시장 체제 | 부분 구현 | `enhanced_features.py` |

### 미구현 항목
1. 종목 간 상관계수 계산
2. 공분산 행렬 (Covariance Matrix)
3. 포트폴리오 최적화 (Markowitz)
4. 조건부 상관관계 (스트레스 시)
5. 동적 리밸런싱
6. 포트폴리오 VaR

---

## 2. 구현 방안

### 2.1 상관관계 행렬 계산기

**새 파일**: `src/portfolio/correlation.py`

```python
import numpy as np
import pandas as pd
from typing import Dict, List

class CorrelationCalculator:
    """종목 간 상관관계 계산"""

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self.price_history: Dict[str, pd.Series] = {}

    def update_price(self, ticker: str, price: float, timestamp):
        """실시간 가격 업데이트"""
        pass

    def get_correlation_matrix(self, tickers: List[str]) -> pd.DataFrame:
        """상관관계 행렬 반환"""
        returns = self._calculate_returns(tickers)
        return returns.corr()

    def get_rolling_correlation(self, ticker1: str, ticker2: str,
                                window: int = 20) -> pd.Series:
        """롤링 상관계수"""
        pass

    def get_covariance_matrix(self, tickers: List[str]) -> pd.DataFrame:
        """공분산 행렬"""
        returns = self._calculate_returns(tickers)
        return returns.cov() * 252  # 연간화

    def _calculate_returns(self, tickers: List[str]) -> pd.DataFrame:
        """일별 수익률 계산"""
        pass
```

### 2.2 포트폴리오 최적화기

**새 파일**: `src/portfolio/optimizer.py`

```python
import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Markowitz 포트폴리오 최적화"""

    def __init__(self, correlation_calc: CorrelationCalculator):
        self.corr_calc = correlation_calc

    def optimize_sharpe(self, tickers: List[str],
                        expected_returns: Dict[str, float],
                        risk_free_rate: float = 0.05) -> Dict[str, float]:
        """샤프 비율 최대화 포트폴리오"""
        pass

    def optimize_min_variance(self, tickers: List[str]) -> Dict[str, float]:
        """최소 분산 포트폴리오"""
        cov_matrix = self.corr_calc.get_covariance_matrix(tickers)
        n = len(tickers)

        # 제약 조건: 가중치 합 = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 0.3) for _ in range(n)]  # 최대 30% 단일 종목

        result = minimize(
            self._portfolio_variance,
            x0=np.ones(n) / n,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return dict(zip(tickers, result.x))

    def _portfolio_variance(self, weights, cov_matrix):
        return weights.T @ cov_matrix @ weights
```

### 2.3 다각화 스코어 계산기

**새 파일**: `src/portfolio/diversification.py`

```python
class DiversificationScore:
    """포트폴리오 다각화 점수 계산"""

    def __init__(self, correlation_calc: CorrelationCalculator):
        self.corr_calc = correlation_calc

    def calculate_diversification_ratio(self,
                                        positions: Dict[str, float]) -> float:
        """다각화 비율 (DR)

        DR = (가중평균 변동성) / (포트폴리오 변동성)
        DR > 1: 다각화 효과 있음
        """
        pass

    def calculate_effective_n(self, positions: Dict[str, float]) -> float:
        """실효 종목 수 (Effective N)

        완전 집중: 1, 완전 분산: n
        """
        weights = np.array(list(positions.values()))
        return 1 / np.sum(weights ** 2)

    def get_correlation_warning(self,
                                positions: Dict[str, float],
                                threshold: float = 0.7) -> List[tuple]:
        """고상관 종목 쌍 경고"""
        tickers = list(positions.keys())
        corr_matrix = self.corr_calc.get_correlation_matrix(tickers)

        warnings = []
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i < j and corr_matrix.loc[t1, t2] > threshold:
                    warnings.append((t1, t2, corr_matrix.loc[t1, t2]))
        return warnings
```

### 2.4 RiskManager 확장

**수정 파일**: `src/risk/risk_manager.py`

```python
class RiskManager:
    # 기존 코드...

    def __init__(self, ...):
        # 기존 초기화...
        self.corr_calc = CorrelationCalculator()
        self.optimizer = PortfolioOptimizer(self.corr_calc)
        self.div_score = DiversificationScore(self.corr_calc)

    def check_correlation_risk(self, new_ticker: str) -> bool:
        """신규 종목 상관관계 리스크 체크"""
        current_positions = list(self.positions.keys())
        if not current_positions:
            return True

        for pos in current_positions:
            corr = self.corr_calc.get_correlation(new_ticker, pos)
            if corr > 0.8:  # 고상관 임계값
                return False
        return True

    def get_optimal_allocation(self, candidates: List[str]) -> Dict[str, float]:
        """최적 포트폴리오 할당 제안"""
        return self.optimizer.optimize_min_variance(candidates)

    def get_portfolio_risk_metrics(self) -> Dict:
        """포트폴리오 리스크 지표"""
        return {
            'diversification_ratio': self.div_score.calculate_diversification_ratio(self.positions),
            'effective_n': self.div_score.calculate_effective_n(self.positions),
            'correlation_warnings': self.div_score.get_correlation_warning(self.positions)
        }
```

---

## 3. UI 연동

### 3.1 새 API 엔드포인트

**수정 파일**: `src/api/routes/portfolio.py` (신규)

```python
@router.get("/portfolio/correlation-matrix")
async def get_correlation_matrix(tickers: List[str] = Query(...)):
    """상관관계 행렬 조회"""
    return correlation_calc.get_correlation_matrix(tickers).to_dict()

@router.get("/portfolio/optimization")
async def get_optimal_weights(tickers: List[str] = Query(...)):
    """최적 포트폴리오 가중치"""
    return optimizer.optimize_min_variance(tickers)

@router.get("/portfolio/diversification")
async def get_diversification_score():
    """다각화 점수 조회"""
    return risk_manager.get_portfolio_risk_metrics()
```

### 3.2 대시보드 표시

**통합 백테스트 UI에 추가**:
- 상관관계 히트맵 차트
- 다각화 점수 게이지
- 고상관 경고 배지

---

## 4. 구현 우선순위

### Phase 1 (필수)
| 항목 | 설명 | 예상 난이도 |
|------|------|------------|
| CorrelationCalculator | 상관관계 행렬 계산 | 중 |
| 고상관 경고 | threshold 초과 알림 | 하 |
| RiskManager 연동 | 진입 전 상관관계 체크 | 중 |

### Phase 2 (권장)
| 항목 | 설명 | 예상 난이도 |
|------|------|------------|
| 최소분산 최적화 | scipy minimize 활용 | 중상 |
| 다각화 점수 | DR, Effective N | 중 |
| API 엔드포인트 | REST API 추가 | 하 |

### Phase 3 (선택)
| 항목 | 설명 | 예상 난이도 |
|------|------|------------|
| 샤프비율 최적화 | 수익률 예측 필요 | 상 |
| 조건부 상관 | 시장 스트레스 감지 | 상 |
| 동적 리밸런싱 | 자동 포지션 조정 | 상 |

---

## 5. 데이터 요구사항

### 필요 데이터
| 데이터 | 소스 | 빈도 |
|--------|------|------|
| 종가 히스토리 | yfinance | 일별 |
| 분봉 데이터 | 기존 수집기 | 5분 |
| 섹터 정보 | enhanced_features.py | 이미 있음 |

### 저장 방식
- SQLite: `data/portfolio/correlation_cache.db`
- 롤링 업데이트: 매일 장 마감 후

---

## 6. 기대 효과

### 리스크 감소
- 고상관 종목 동시 보유 방지
- 포트폴리오 변동성 20~30% 감소 예상

### 수익 안정화
- 다각화를 통한 일관된 수익
- 드로다운 완화

### 의사결정 지원
- 객관적인 종목 선택 기준
- 백테스트 정확도 향상

---

## 7. 파일 구조 (신규)

```
src/
├── portfolio/              # 신규 폴더
│   ├── __init__.py
│   ├── correlation.py      # 상관관계 계산
│   ├── optimizer.py        # 포트폴리오 최적화
│   └── diversification.py  # 다각화 점수
├── risk/
│   └── risk_manager.py     # 수정 (portfolio 연동)
└── api/
    └── routes/
        └── portfolio.py    # 신규 API
```

---

**작성**: 분석팀장
**작성일**: 2025-12-26
**버전**: 1.0
