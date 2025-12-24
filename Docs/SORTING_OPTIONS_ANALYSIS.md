# 정렬 옵션 다양화 분석

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 현재 상태 분석

### 1.1 기존 정렬 로직

```javascript
// Dashboard.jsx (현재 구현)
const sortPredictions = (predictions, category) => {
  if (!predictions) return []
  return [...predictions].sort((a, b) => {
    if (category === 'gainers') {
      return b.change_percent - a.change_percent // 변동률순
    } else {
      return b.probability - a.probability // 확률순
    }
  })
}
```

**현재 정렬:**
- Gainers 카테고리: 변동률 내림차순 (고정)
- Volume 카테고리: 예측 확률 내림차순 (고정)

### 1.2 현재 한계점

| 문제 | 설명 | 사용자 영향 |
|------|------|-------------|
| 정렬 옵션 선택 불가 | 카테고리별 고정 정렬만 가능 | 원하는 기준으로 탐색 어려움 |
| Precision 정렬 없음 | 적중률 기준 정렬 불가 | 신뢰도 높은 예측 찾기 어려움 |
| Signal Rate 정렬 없음 | 신호 발생률 정렬 불가 | 활발한 예측 모델 찾기 어려움 |
| 등급 정렬 없음 | 실용성 등급 정렬 불가 | 종합적 우선순위 파악 어려움 |
| 다중 정렬 미지원 | 1차/2차 정렬 불가 | 세밀한 필터링 어려움 |

---

## 2. 정렬 옵션 설계

### 2.1 제안 정렬 기준

| 정렬 기준 | 설명 | 데이터 필드 | 사용 시나리오 |
|-----------|------|-------------|--------------|
| 예측 확률 | 모델 예측 확률 | `probability` | 강한 신호 우선 탐색 |
| Precision | 모델 적중률 | `hit_rate` | 신뢰도 높은 예측 탐색 |
| Signal Rate | 신호 발생률 | `signal_rate` | 활발한 모델 탐색 |
| 등급 | 실용성 등급 | `practicality_grade` | 종합 평가 기준 |
| 변동률 | 일일 가격 변동 | `change_percent` | 모멘텀 기준 탐색 |
| 가격 | 현재 주가 | `current_price` | 가격대별 탐색 |
| 티커명 | 알파벳 순서 | `ticker` | 특정 종목 빠른 찾기 |

### 2.2 정렬 방향

```
┌─────────────────────────────────────┐
│ 정렬: [확률순 ▼]  방향: [▼] [▲]    │
└─────────────────────────────────────┘
```

| 방향 | 아이콘 | 설명 |
|------|--------|------|
| 내림차순 | ▼ | 높은 값 → 낮은 값 |
| 오름차순 | ▲ | 낮은 값 → 높은 값 |

---

## 3. 정렬 로직 상세

### 3.1 단일 정렬 구현

```javascript
// sortUtils.js

export const SORT_OPTIONS = {
  probability: {
    label: { ko: '예측 확률', en: 'Probability' },
    getValue: (item) => item.probability,
    defaultOrder: 'desc'
  },
  precision: {
    label: { ko: 'Precision (적중률)', en: 'Precision' },
    getValue: (item) => item.hit_rate,
    defaultOrder: 'desc'
  },
  signalRate: {
    label: { ko: 'Signal Rate', en: 'Signal Rate' },
    getValue: (item) => item.signal_rate,
    defaultOrder: 'desc'
  },
  grade: {
    label: { ko: '등급', en: 'Grade' },
    getValue: (item) => gradeToNumber(item.practicality_grade),
    defaultOrder: 'desc'
  },
  changePercent: {
    label: { ko: '변동률', en: 'Change %' },
    getValue: (item) => item.change_percent,
    defaultOrder: 'desc'
  },
  price: {
    label: { ko: '가격', en: 'Price' },
    getValue: (item) => item.current_price,
    defaultOrder: 'desc'
  },
  ticker: {
    label: { ko: '티커명', en: 'Ticker' },
    getValue: (item) => item.ticker,
    defaultOrder: 'asc'
  }
}

// 등급을 숫자로 변환 (정렬용)
const gradeToNumber = (grade) => {
  const gradeMap = { 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'N/A': 0 }
  return gradeMap[grade] || 0
}

// 정렬 함수
export const sortPredictions = (predictions, sortBy, order = 'desc') => {
  if (!predictions || !SORT_OPTIONS[sortBy]) return predictions

  const option = SORT_OPTIONS[sortBy]
  const multiplier = order === 'desc' ? -1 : 1

  return [...predictions].sort((a, b) => {
    const valueA = option.getValue(a)
    const valueB = option.getValue(b)

    // 문자열 비교 (티커명)
    if (typeof valueA === 'string') {
      return multiplier * valueA.localeCompare(valueB)
    }

    // 숫자 비교
    return multiplier * (valueA - valueB)
  })
}
```

### 3.2 다중 정렬 (2차 정렬)

```javascript
// 다중 정렬 지원
export const multiSort = (predictions, sortConfigs) => {
  // sortConfigs: [{ field: 'grade', order: 'desc' }, { field: 'probability', order: 'desc' }]

  return [...predictions].sort((a, b) => {
    for (const config of sortConfigs) {
      const option = SORT_OPTIONS[config.field]
      if (!option) continue

      const valueA = option.getValue(a)
      const valueB = option.getValue(b)
      const multiplier = config.order === 'desc' ? -1 : 1

      let comparison = 0
      if (typeof valueA === 'string') {
        comparison = valueA.localeCompare(valueB)
      } else {
        comparison = valueA - valueB
      }

      if (comparison !== 0) {
        return multiplier * comparison
      }
    }
    return 0
  })
}
```

### 3.3 정렬 프리셋

자주 사용되는 정렬 조합을 프리셋으로 제공

```javascript
export const SORT_PRESETS = {
  bestOpportunity: {
    label: { ko: '최고 기회순', en: 'Best Opportunity' },
    configs: [
      { field: 'grade', order: 'desc' },
      { field: 'probability', order: 'desc' }
    ]
  },
  mostReliable: {
    label: { ko: '신뢰도순', en: 'Most Reliable' },
    configs: [
      { field: 'precision', order: 'desc' },
      { field: 'signalRate', order: 'desc' }
    ]
  },
  highMomentum: {
    label: { ko: '모멘텀순', en: 'High Momentum' },
    configs: [
      { field: 'changePercent', order: 'desc' },
      { field: 'probability', order: 'desc' }
    ]
  },
  activeSignals: {
    label: { ko: '활성 신호순', en: 'Active Signals' },
    configs: [
      { field: 'signalRate', order: 'desc' },
      { field: 'precision', order: 'desc' }
    ]
  }
}
```

---

## 4. UI 설계

### 4.1 정렬 드롭다운

```
┌─────────────────────────────────────┐
│ 정렬: [확률순 ▼]                    │
├─────────────────────────────────────┤
│ ── 프리셋 ──                        │
│ ● 최고 기회순 (등급 → 확률)         │
│ ○ 신뢰도순 (Precision → Signal)    │
│ ○ 모멘텀순 (변동률 → 확률)         │
│ ○ 활성 신호순 (Signal → Precision) │
├─────────────────────────────────────┤
│ ── 단일 정렬 ──                     │
│ ○ 예측 확률                    ▼▲  │
│ ○ Precision (적중률)           ▼▲  │
│ ○ Signal Rate                  ▼▲  │
│ ○ 등급 (A→D)                   ▼▲  │
│ ○ 변동률                       ▼▲  │
│ ○ 가격                         ▼▲  │
│ ○ 티커명 (A→Z)                 ▼▲  │
└─────────────────────────────────────┘
```

### 4.2 빠른 정렬 버튼 (데스크톱)

테이블 헤더에 정렬 버튼 추가

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ 티커 ▼▲ │ 확률 ▼▲  │ Prec ▼▲  │ Sig ▼▲  │ 등급 ▼▲ │
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ NVDA     │ 82%      │ 68%      │ 15%      │ A        │
│ TSLA     │ 78%      │ 55%      │ 12%      │ A        │
│ AAPL     │ 75%      │ 62%      │ 18%      │ B        │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

### 4.3 정렬 표시 배지

현재 적용된 정렬을 필터 바에 표시

```
┌─────────────────────────────────────────────────────────┐
│ 활성 정렬: [등급 ▼] → [확률 ▼]              [초기화 ✕] │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 상태 관리

### 5.1 sortStore.js

```javascript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useSortStore = create(
  persist(
    (set, get) => ({
      // 현재 정렬 설정
      sortMode: 'preset', // 'preset' | 'single' | 'multi'
      presetKey: 'bestOpportunity',
      singleSort: {
        field: 'probability',
        order: 'desc'
      },
      multiSort: [
        { field: 'grade', order: 'desc' },
        { field: 'probability', order: 'desc' }
      ],

      // Actions
      setPreset: (presetKey) => set({
        sortMode: 'preset',
        presetKey
      }),

      setSingleSort: (field, order) => set({
        sortMode: 'single',
        singleSort: { field, order }
      }),

      toggleSingleSortOrder: () => set((state) => ({
        singleSort: {
          ...state.singleSort,
          order: state.singleSort.order === 'desc' ? 'asc' : 'desc'
        }
      })),

      setMultiSort: (configs) => set({
        sortMode: 'multi',
        multiSort: configs
      }),

      addMultiSort: (field, order) => set((state) => ({
        sortMode: 'multi',
        multiSort: [...state.multiSort, { field, order }].slice(0, 3) // 최대 3개
      })),

      removeMultiSort: (index) => set((state) => ({
        multiSort: state.multiSort.filter((_, i) => i !== index)
      })),

      // 현재 정렬 설정 가져오기
      getCurrentSortConfigs: () => {
        const state = get()
        switch (state.sortMode) {
          case 'preset':
            return SORT_PRESETS[state.presetKey]?.configs || []
          case 'single':
            return [state.singleSort]
          case 'multi':
            return state.multiSort
          default:
            return [{ field: 'probability', order: 'desc' }]
        }
      },

      // 초기화
      resetSort: () => set({
        sortMode: 'preset',
        presetKey: 'bestOpportunity',
        singleSort: { field: 'probability', order: 'desc' },
        multiSort: []
      })
    }),
    {
      name: 'nasdaq-predictor-sort',
      version: 1,
    }
  )
)
```

---

## 6. 정렬 기준별 분석

### 6.1 예측 확률 (Probability)

**특징:**
- 모델이 예측한 방향으로 움직일 확률
- 70% 이상: 신호 발생 기준
- 80% 이상: 강한 신호
- 90% 이상: 매우 강한 신호

**사용 시나리오:**
- 강한 신호를 빠르게 찾고 싶을 때
- 모델 신뢰도보다 신호 강도 우선

**주의점:**
- 높은 확률이 반드시 높은 Precision을 의미하지 않음
- Precision과 함께 확인 필요

### 6.2 Precision (적중률)

**특징:**
- 과거 예측의 실제 적중률
- 50% 이상: 우수
- 30-50%: 보통
- 30% 미만: 개선 필요

**사용 시나리오:**
- 신뢰할 수 있는 예측을 찾을 때
- 리스크 회피형 투자자

**주의점:**
- 낮은 Precision이라도 수익성이 있을 수 있음 (손익비 고려)
- Signal Rate와 함께 확인 필요

### 6.3 Signal Rate (신호 발생률)

**특징:**
- 모델이 신호를 발생시킨 비율
- 10% 이상: 활발
- 5-10%: 보통
- 5% 미만: 보수적

**사용 시나리오:**
- 활발하게 거래 기회를 찾을 때
- 단기 트레이더

**주의점:**
- 높은 Signal Rate는 노이즈 신호 가능성
- Precision과 균형 필요

### 6.4 등급 (Grade)

**특징:**
- Precision + Signal Rate 종합 평가
- A: Precision ≥ 50% & Signal ≥ 10%
- B: Precision ≥ 30% & Signal ≥ 10%
- C: Precision ≥ 30% & Signal < 10%
- D: Precision < 30%

**사용 시나리오:**
- 종합적으로 우수한 예측을 찾을 때
- 초보 투자자

**권장:**
- 대부분의 사용자에게 기본 정렬로 권장

### 6.5 변동률 (Change %)

**특징:**
- 당일 가격 변동률
- 양수: 상승 중
- 음수: 하락 중

**사용 시나리오:**
- 모멘텀 트레이딩
- 현재 움직임이 큰 종목 탐색

**주의점:**
- 과거 움직임이 미래를 보장하지 않음
- 예측 신호와 함께 확인

---

## 7. 정렬 조합 가이드

### 7.1 추천 조합

| 투자 스타일 | 1차 정렬 | 2차 정렬 | 설명 |
|-------------|----------|----------|------|
| 안정 추구 | 등급 ▼ | Precision ▼ | 검증된 예측 우선 |
| 공격적 | 확률 ▼ | 변동률 ▼ | 강한 신호 + 모멘텀 |
| 활발한 거래 | Signal ▼ | Precision ▼ | 많은 기회 탐색 |
| 균형형 | 등급 ▼ | 확률 ▼ | 종합적 판단 |

### 7.2 정렬 조합 효과

```
┌─────────────────────────────────────────────────────────────┐
│ 정렬 조합 비교 (동일 데이터셋)                              │
├─────────────────────────────────────────────────────────────┤
│ 확률순 Top 5:        NVDA(82%), TSLA(78%), AMD(77%), ...    │
│ Precision순 Top 5:   AAPL(68%), MSFT(65%), GOOGL(62%), ... │
│ 등급순 Top 5:        NVDA(A), TSLA(A), AAPL(A), AMD(B), ... │
│ 변동률순 Top 5:      GME(+15%), NVDA(+8%), TSLA(+6%), ...   │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 컴포넌트 구현

### 8.1 SortDropdown.jsx

```jsx
import { useState } from 'react'
import { useSortStore } from '../stores/sortStore'
import { SORT_OPTIONS, SORT_PRESETS } from '../utils/sortUtils'
import { useSettingsStore } from '../stores/settingsStore'

export default function SortDropdown() {
  const [isOpen, setIsOpen] = useState(false)
  const { language } = useSettingsStore()
  const {
    sortMode,
    presetKey,
    singleSort,
    setPreset,
    setSingleSort,
    toggleSingleSortOrder,
    resetSort
  } = useSortStore()

  const getCurrentLabel = () => {
    if (sortMode === 'preset') {
      return SORT_PRESETS[presetKey]?.label[language]
    }
    return SORT_OPTIONS[singleSort.field]?.label[language]
  }

  return (
    <div className="relative">
      {/* 트리거 버튼 */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-surface-light rounded-lg hover:bg-slate-600 transition-colors"
      >
        <span className="text-sm text-gray-400">정렬:</span>
        <span className="font-medium">{getCurrentLabel()}</span>
        <span className="text-gray-400">
          {singleSort.order === 'desc' ? '▼' : '▲'}
        </span>
      </button>

      {/* 드롭다운 */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-72 bg-surface border border-surface-light rounded-lg shadow-xl z-50">
          {/* 프리셋 섹션 */}
          <div className="p-3 border-b border-surface-light">
            <div className="text-xs text-gray-500 mb-2">프리셋</div>
            {Object.entries(SORT_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => {
                  setPreset(key)
                  setIsOpen(false)
                }}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  sortMode === 'preset' && presetKey === key
                    ? 'bg-blue-500/20 text-blue-400'
                    : 'hover:bg-surface-light'
                }`}
              >
                {preset.label[language]}
              </button>
            ))}
          </div>

          {/* 단일 정렬 섹션 */}
          <div className="p-3">
            <div className="text-xs text-gray-500 mb-2">단일 정렬</div>
            {Object.entries(SORT_OPTIONS).map(([key, option]) => (
              <div
                key={key}
                className={`flex items-center justify-between px-3 py-2 rounded-lg text-sm ${
                  sortMode === 'single' && singleSort.field === key
                    ? 'bg-blue-500/20'
                    : 'hover:bg-surface-light'
                }`}
              >
                <button
                  onClick={() => {
                    setSingleSort(key, option.defaultOrder)
                    setIsOpen(false)
                  }}
                  className="flex-1 text-left"
                >
                  {option.label[language]}
                </button>
                {sortMode === 'single' && singleSort.field === key && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      toggleSingleSortOrder()
                    }}
                    className="px-2 py-1 text-blue-400 hover:bg-blue-500/20 rounded"
                  >
                    {singleSort.order === 'desc' ? '▼' : '▲'}
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* 초기화 */}
          <div className="p-3 border-t border-surface-light">
            <button
              onClick={() => {
                resetSort()
                setIsOpen(false)
              }}
              className="w-full py-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              초기화
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
```

---

## 9. 성능 고려사항

### 9.1 정렬 최적화

```javascript
// 대용량 데이터 정렬 최적화
import { useMemo } from 'react'

const SortedList = ({ predictions, sortConfigs }) => {
  // 정렬 결과 메모이제이션
  const sortedPredictions = useMemo(() => {
    return multiSort(predictions, sortConfigs)
  }, [predictions, sortConfigs])

  return <TickerGrid predictions={sortedPredictions} />
}
```

### 9.2 정렬 성능 벤치마크

| 데이터 크기 | 단일 정렬 | 다중 정렬 (2개) | 다중 정렬 (3개) |
|-------------|-----------|-----------------|-----------------|
| 100개 | < 1ms | < 2ms | < 3ms |
| 500개 | < 5ms | < 8ms | < 12ms |
| 1000개 | < 10ms | < 18ms | < 25ms |

---

## 10. 구현 일정

### Phase 1: 기본 인프라 (2일)
- [ ] sortUtils.js 유틸리티 작성
- [ ] sortStore.js 상태 관리 구현
- [ ] SORT_OPTIONS, SORT_PRESETS 정의

### Phase 2: UI 컴포넌트 (2일)
- [ ] SortDropdown.jsx 구현
- [ ] 정렬 표시 배지 구현
- [ ] 다국어 지원

### Phase 3: 통합 (1일)
- [ ] Dashboard.jsx 정렬 로직 연동
- [ ] filterStore와 통합
- [ ] 설정 저장 연동

### Phase 4: 테스트 (1일)
- [ ] 정렬 기능 단위 테스트
- [ ] 성능 벤치마크
- [ ] 사용자 시나리오 테스트

---

## 11. 테스트 케이스

### 11.1 기능 테스트

| 테스트 | 입력 | 예상 결과 |
|--------|------|-----------|
| 확률 내림차순 | 확률순 ▼ | 82%, 78%, 75%... 순서 |
| 확률 오름차순 | 확률순 ▲ | 55%, 60%, 65%... 순서 |
| 등급 정렬 | 등급순 ▼ | A, A, A, B, B, C... 순서 |
| 다중 정렬 | 등급▼ → 확률▼ | 같은 등급 내 확률순 |
| 프리셋 적용 | 최고 기회순 | 등급 → 확률 순서 |
| 초기화 | 초기화 클릭 | 기본 정렬(등급순) 복귀 |

### 11.2 엣지 케이스

| 테스트 | 설명 | 예상 결과 |
|--------|------|-----------|
| 빈 목록 | 티커 0개 | 빈 배열 반환, 에러 없음 |
| 동일값 | 모든 확률 75% | 원래 순서 유지 |
| null 값 | hit_rate: null | 0으로 처리 |
| N/A 등급 | 등급 없음 | 최하위로 정렬 |

---

*이 문서는 분석팀장이 작성하였습니다.*
