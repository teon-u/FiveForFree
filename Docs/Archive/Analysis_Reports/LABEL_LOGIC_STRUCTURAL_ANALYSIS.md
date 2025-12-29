# Label Generation Logic - Structural Analysis

**작성일**: 2025-12-23
**요청**: 비서실장 (개발팀장 발견 이슈 검토)
**긴급도**: 높음

---

## 1. 현재 상황 요약

### 발견된 문제
| 항목 | 예상값 | 실제값 | 차이 |
|------|--------|--------|------|
| Up Label Rate | 3.5% | **0.9%** | -74% |
| Down Label Rate | 4.2% | **0.6%** | -86% |
| Signal Rate 목표 | 10% | **구조적 불가** | - |

### 핵심 원인
**레이블이 1인 데이터가 전체의 1.5%에 불과** → 모델이 아무리 잘 예측해도 Signal Rate 10% 달성 불가능

---

## 2. 현재 레이블 생성 로직 분석

### 2.1 파일 위치
```
src/processor/label_generator.py
config/settings.py
```

### 2.2 핵심 로직
```python
# label_generator.py:112-113
label_up = 1 if max_gain >= self.target_percent else 0
label_down = 1 if max_loss <= -self.target_percent else 0
```

### 2.3 실제 적용 조건
```
Entry Price에서 60분 내:
- label_up=1: (max_high - entry) / entry >= 1.0% + 0.2%(commission) = 1.2%
- label_down=1: (min_low - entry) / entry <= -1.0% - 0.2%(commission) = -1.2%
```

### 2.4 Commission 적용 방식
```python
# label_generator.py:100-105
commission_total = self.commission_pct * 2  # Round-trip: 0.2%
future_bars['gain_pct_net'] = future_bars['gain_pct'] - commission_total
future_bars['loss_pct_net'] = future_bars['loss_pct'] + commission_total
```

---

## 3. TARGET_PERCENT (1%) 적절성 검토

### 3.1 변경 이력
| 시점 | TARGET_PERCENT | 이유 |
|------|----------------|------|
| 초기 | 5.0% | 높은 수익 목표 |
| 2024-12-15 | 1.0% | 클래스 불균형 완화 시도 |
| **현재** | 1.0% | 여전히 불균형 심각 |

### 3.2 문제점 분석

#### 실제 임계값 계산
```
명시된 TARGET_PERCENT: 1.0%
Commission 왕복: 0.2%
실제 필요한 이동폭: 1.2%

→ NASDAQ 주요 종목의 60분 내 ±1.2% 이동 빈도: ~1.5%
```

#### 왜 예상(3.5~4.2%)보다 낮은가?
1. **Commission 반영**: 문서 예상치는 Commission 제외 기준일 가능성
2. **데이터 기간**: 최근 데이터가 저변동성 기간일 수 있음
3. **종목 선택**: 대형주 위주로 변동성이 낮음

### 3.3 결론
> **TARGET_PERCENT 1%는 현재 조건에서 너무 높음**
> Commission 포함 시 실질 1.2% 이동 필요 → 발생 빈도 1.5%

---

## 4. 레이블 완화 방안

### 방안 A: TARGET_PERCENT 낮추기 (권장)
```python
# settings.py 수정
TARGET_PERCENT: float = 0.5  # 1.0 → 0.5

# 예상 효과:
# - 실질 임계값: 0.5% + 0.2% = 0.7%
# - 예상 레이블 비율: 5~8%
```

**장점**: 간단한 설정 변경으로 해결
**단점**: 목표 수익률 감소

### 방안 B: Commission 제외 레이블 생성
```python
# label_generator.py 수정
# commission_total = self.commission_pct * 2  # 주석 처리
commission_total = 0  # 레이블 생성 시 Commission 무시

# 예상 효과:
# - 실질 임계값: 1.0%
# - 예상 레이블 비율: 3~5%
```

**장점**: 현재 TARGET_PERCENT 유지 가능
**단점**: 실거래 시 Commission으로 인한 손실 발생 가능

### 방안 C: 변동성 기반 적응형 임계값
```python
# 새로운 접근법
def get_adaptive_threshold(volatility):
    """변동성에 따른 동적 임계값"""
    if volatility > 2.0:  # 고변동성
        return 1.5
    elif volatility > 1.0:  # 중변동성
        return 1.0
    else:  # 저변동성
        return 0.5

# 예상 효과:
# - 시장 상황에 맞는 유연한 레이블링
# - 예상 레이블 비율: 5~10%
```

**장점**: 시장 상황 반영
**단점**: 구현 복잡도 높음

### 방안 D: 시간대별 필터링
```python
# 변동성 높은 시간대만 타겟팅
# 미국 시장 기준: 09:30-10:30, 15:00-16:00

# 예상 효과:
# - 해당 시간대 레이블 비율: 3~5%
# - 전체 데이터 감소 (트레이드오프)
```

**장점**: 실제 트레이딩 시간과 일치
**단점**: 훈련 데이터 감소

---

## 5. Signal Rate 목표 조정 분석

### 5.1 현재 구조적 한계

```
전체 데이터: 100%
├── label=1 (positive): 1.5%
└── label=0 (negative): 98.5%

모델 예측 결과:
├── Predicted=1 (signal): X%
└── Predicted=0 (no signal): (100-X)%

Signal Rate = Predicted=1 / Total = X%
```

### 5.2 수학적 분석

**시나리오 1: 완벽한 모델**
```
- 모든 label=1을 정확히 예측 (Recall=100%)
- 모든 label=0을 정확히 예측 (Specificity=100%)
- Signal Rate = 1.5% (불가피)
- Precision = 100%
```

**시나리오 2: Signal Rate 10% 강제 달성**
```
- Predicted=1: 10%
- 실제 label=1: 1.5%
- 최대 True Positive: 1.5%
- False Positive: 최소 8.5%
- 최대 Precision: 1.5% / 10% = 15%
```

### 5.3 달성 가능한 Signal Rate

| 레이블 비율 | 현실적 Signal Rate | 기대 Precision |
|------------|-------------------|----------------|
| 1.5% | 2~3% | 50~75% |
| 3% | 4~5% | 60~75% |
| 5% | 6~8% | 60~80% |
| 8% | 10~12% | 65~80% |

### 5.4 권장 목표 조정

| 항목 | 현재 | 권장 | 사유 |
|------|------|------|------|
| Signal Rate | 10% | **5%** | 레이블 비율 1.5%에서 현실적 |
| MIN_PRECISION | 60% | **55%** | Signal Rate 상향 시 하향 조정 |
| TARGET_PERCENT | 1.0% | **0.5%** | 레이블 비율 향상 |

---

## 6. 권장 실행 계획

### 즉시 (Phase 1)
```yaml
변경 사항:
  - MIN_SIGNAL_RATE: 0.10 → 0.05
  - SIGNAL_THRESHOLD: 0.40 → 0.35

효과: 현재 레이블 구조에서 즉시 실행 가능
리스크: 낮음
```

### 단기 (Phase 2)
```yaml
변경 사항:
  - TARGET_PERCENT: 1.0 → 0.5
  - 전체 모델 재훈련 필요

효과: 레이블 비율 5~8%로 향상
리스크: 중간 (수익 목표 축소)
```

### 중기 (Phase 3)
```yaml
변경 사항:
  - 적응형 임계값 시스템 구현
  - 변동성 기반 동적 TARGET_PERCENT

효과: 시장 상황에 맞는 유연한 운영
리스크: 높음 (개발 복잡도)
```

---

## 7. 결론

### 핵심 발견
1. **TARGET_PERCENT 1%는 Commission 포함 시 너무 높음** (실질 1.2%)
2. **레이블 비율 1.5%는 Signal Rate 10%와 구조적으로 양립 불가**
3. **현재 scale_pos_weight 적용으로도 Signal Rate 개선 한계**

### 최종 권장 사항

| 우선순위 | 조치 | 담당 |
|---------|------|------|
| 1 | MIN_SIGNAL_RATE를 5%로 조정 | 개발팀장 |
| 2 | TARGET_PERCENT를 0.5%로 변경 | 개발팀장 |
| 3 | 전체 모델 재훈련 | 개발팀장 |
| 4 | 적응형 임계값 R&D | 분석팀장 |

### 의사결정 필요 사항
- **옵션 A**: 목표만 조정 (Signal Rate 10% → 5%)
- **옵션 B**: TARGET_PERCENT도 함께 조정 (1% → 0.5%)
- **옵션 C**: 전면 재설계 (적응형 시스템)

> **분석팀장 의견**: 옵션 B 권장
> - 단기적으로 실현 가능
> - 레이블 비율과 Signal Rate 목표 모두 개선
> - 수익 목표 감소는 거래 빈도 증가로 상쇄 가능

---

## 8. 참고 자료

- `Docs/History/2024-12-15_target_percent_change.md`: TARGET_PERCENT 변경 이력
- `Docs/HYBRID_ENSEMBLE_ARCHITECTURE.md`: 예상 레이블 비율
- `Docs/SIGNAL_RATE_IMPROVEMENT_GUIDE.md`: Signal Rate 개선 가이드
- `src/processor/label_generator.py`: 레이블 생성 로직
- `config/settings.py`: 시스템 설정
