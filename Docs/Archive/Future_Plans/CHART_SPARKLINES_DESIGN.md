# Chart Sparklines 개선 분석서

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 개발팀장, 비서실장
**우선순위**: 중간

---

## 1. 현재 구현 상태 분석

### 1.1 기존 구현

**이미 구현된 컴포넌트**:
- `src/components/Sparkline.jsx` - 커스텀 SVG 기반 미니 차트
- `src/hooks/usePriceHistory.js` - 데이터 fetching 및 기술 지표 계산
- `src/components/TickerCard.jsx` - Sparkline 통합 사용

**현재 기술 스택**:
```
Frontend: React + Vite + Zustand
차트 라이브러리: recharts (설치됨, 미사용)
Sparkline: 커스텀 SVG 구현
데이터 캐싱: @tanstack/react-query
```

### 1.2 현재 Sparkline 기능

| 기능 | 상태 | 설명 |
|------|------|------|
| 가격 라인 차트 | ✅ | SVG path로 구현 |
| 그라데이션 영역 | ✅ | linearGradient 사용 |
| 방향 표시 (색상) | ✅ | up=녹색, down=빨강 |
| 현재가 점 | ✅ | 애니메이션 pulse |
| 60분 데이터 | ✅ | Mock 데이터 |
| 반응형 크기 | ✅ | width="100%" |
| **실제 API 연동** | ❌ | 미구현 |

### 1.3 현재 코드 분석

**Sparkline.jsx** (107줄):
```jsx
// 커스텀 SVG 기반 - 매우 경량
// 외부 의존성 없음
// 성능 우수 (useMemo 최적화)

// 장점:
// - 번들 크기 0 (외부 라이브러리 없음)
// - 완전한 커스터마이징 가능
// - 빠른 렌더링

// 단점:
// - 기능 확장 시 직접 구현 필요
// - 복잡한 인터랙션 어려움
```

---

## 2. Sparkline 라이브러리 비교

### 2.1 비교표

| 라이브러리 | 번들 크기 | 성능 | 커스터마이징 | 인터랙션 | 권장도 |
|-----------|----------|------|------------|---------|--------|
| **현재 (Custom SVG)** | 0KB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | **✅ 유지** |
| recharts (mini) | ~45KB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 이미 설치됨 |
| react-sparklines | ~8KB | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 불필요 |
| visx | ~50KB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 오버킬 |
| Canvas (직접) | 0KB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 불필요 |

### 2.2 각 라이브러리 분석

#### 현재 구현 (Custom SVG) - **권장 유지**

```jsx
// 현재 코드 - 이미 최적화됨
<svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none">
  <path d={areaD} fill={`url(#${gradientId})`} />
  <path d={pathD} stroke={color} strokeWidth="1.5" />
  <circle cx="100" cy={...} r="2" className="animate-pulse" />
</svg>
```

**장점**:
- 번들 크기 0 (외부 의존성 없음)
- 완전한 제어권
- 현재 프로젝트에 최적화됨
- 성능 우수 (useMemo 활용)

**단점**:
- 툴팁, 줌 등 고급 기능 수동 구현 필요

---

#### recharts (이미 설치됨)

```jsx
import { LineChart, Line, ResponsiveContainer } from 'recharts'

<ResponsiveContainer width="100%" height={32}>
  <LineChart data={data}>
    <Line
      type="monotone"
      dataKey="value"
      stroke={color}
      strokeWidth={1.5}
      dot={false}
    />
  </LineChart>
</ResponsiveContainer>
```

**장점**:
- 이미 설치되어 있음 (package.json에 recharts ^2.15.4)
- 풍부한 기능 (툴팁, 애니메이션)
- React 친화적 API

**단점**:
- Sparkline에는 오버스펙
- 번들 크기 증가

**결론**: PredictionPanel 등 복잡한 차트에만 사용 권장

---

#### react-sparklines

```jsx
import { Sparklines, SparklinesLine } from 'react-sparklines'

<Sparklines data={data} width={100} height={32}>
  <SparklinesLine color={color} />
</Sparklines>
```

**장점**:
- 전용 Sparkline 라이브러리
- 간단한 API

**단점**:
- 추가 설치 필요
- 현재 구현보다 장점 없음

**결론**: 불필요

---

#### visx (Airbnb)

```jsx
import { LinePath } from '@visx/shape'
import { scaleLinear } from '@visx/scale'

// 저수준 D3 래퍼 - 높은 유연성
```

**장점**:
- 최고 수준의 커스터마이징
- D3 기반 강력한 기능

**단점**:
- 학습 곡선 높음
- 오버엔지니어링

**결론**: 현재 요구사항에 오버킬

---

### 2.3 권장 사항

```
✅ 현재 Custom SVG 구현 유지
✅ recharts는 복잡한 차트(PredictionPanel)에만 사용
❌ 새 라이브러리 추가 불필요
```

---

## 3. 표시 데이터 분석

### 3.1 가격 히스토리 (현재 구현)

| 기간 | 데이터 포인트 | 갱신 주기 | 용도 |
|------|-------------|----------|------|
| 60분 | 60개 | 1분 | 단기 추세 |
| 1일 | 390개 | 1분 | 당일 움직임 |
| 5일 | 1950개 | 5분 | 주간 추세 |
| 20일 | 7800개 | 15분 | 월간 추세 |

**현재 구현**: 60분 (60개 포인트)

**개선 제안**: 사용자 선택 옵션 추가
```jsx
const periodOptions = ['1H', '1D', '1W', '1M']
```

---

### 3.2 예측 확률 변화 (신규 제안)

```jsx
// 시간별 예측 확률 추이
const probabilityData = [
  { time: '09:30', up_prob: 0.65 },
  { time: '10:00', up_prob: 0.72 },
  { time: '10:30', up_prob: 0.68 },
  // ...
]
```

**장점**:
- 모델 신뢰도 변화 시각화
- 매매 타이밍 판단 도움

**구현 복잡도**: 중간
- Backend: 예측 히스토리 저장 필요
- Frontend: 새 Sparkline 타입 추가

---

### 3.3 거래량 추이 (신규 제안)

```jsx
// 거래량 바 차트 또는 영역 차트
const volumeData = [
  { time: '09:30', volume: 1234567 },
  { time: '10:00', volume: 2345678 },
  // ...
]
```

**장점**:
- 유동성 확인
- 가격 움직임과 상관관계 분석

**구현 복잡도**: 낮음 (데이터 이미 있음)

---

## 4. 구현 복잡도 분석

### 4.1 백엔드 요구사항

#### 신규 API 엔드포인트

**1. Sparkline API** (우선순위: 높음)

```python
# src/api/routes/prices.py

@router.get("/{ticker}/sparkline")
async def get_sparkline(
    ticker: str,
    period: str = "1H",  # 1H, 1D, 1W, 1M
    data_loader: DataLoader = Depends(get_data_loader)
) -> SparklineResponse:
    """
    Sparkline 데이터 반환 (60개 포인트로 다운샘플링)
    """
    # 기간별 데이터 조회
    raw_data = data_loader.get_price_history(ticker, period)

    # 60개 포인트로 다운샘플링
    sampled = downsample(raw_data, target_points=60)

    # 방향 계산
    direction = 'up' if sampled[-1] >= sampled[0] else 'down'

    return SparklineResponse(
        symbol=ticker,
        data=[p['close'] for p in sampled],
        direction=direction,
        period=period,
        min=min(d['close'] for d in sampled),
        max=max(d['close'] for d in sampled)
    )
```

**2. 예측 히스토리 API** (우선순위: 낮음)

```python
@router.get("/{ticker}/prediction-history")
async def get_prediction_history(
    ticker: str,
    hours: int = 24
) -> PredictionHistoryResponse:
    """
    예측 확률 변화 히스토리
    """
    # prediction_records 테이블에서 조회
    pass
```

---

### 4.2 프론트엔드 개선사항

#### 1. 기간 선택 기능 (우선순위: 높음)

```jsx
// src/components/Sparkline.jsx 수정

export default function Sparkline({
  data,
  direction = 'up',
  width = '100%',
  height = 32,
  period = '1H',  // 새 prop
  onPeriodChange,  // 새 prop
}) {
  // 기간 선택 UI (호버 시 표시)
  const [showPeriodSelector, setShowPeriodSelector] = useState(false)

  return (
    <div
      className="relative"
      onMouseEnter={() => setShowPeriodSelector(true)}
      onMouseLeave={() => setShowPeriodSelector(false)}
    >
      <svg>...</svg>

      {showPeriodSelector && onPeriodChange && (
        <div className="absolute top-0 right-0 flex gap-1 bg-surface-dark/90 rounded px-1">
          {['1H', '1D', '1W'].map(p => (
            <button
              key={p}
              onClick={() => onPeriodChange(p)}
              className={clsx(
                'text-[10px] px-1',
                period === p ? 'text-blue-400' : 'text-gray-500'
              )}
            >
              {p}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
```

---

#### 2. 툴팁 기능 (우선순위: 중간)

```jsx
// 마우스 호버 시 가격 표시
const [tooltip, setTooltip] = useState(null)

const handleMouseMove = (e) => {
  const rect = e.currentTarget.getBoundingClientRect()
  const x = (e.clientX - rect.left) / rect.width
  const index = Math.floor(x * data.length)
  if (data[index]) {
    setTooltip({
      x: e.clientX,
      y: e.clientY,
      value: data[index]
    })
  }
}

// SVG에 onMouseMove 추가
<svg onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)}>
  ...
</svg>

{tooltip && (
  <div
    className="fixed bg-gray-800 text-white text-xs px-2 py-1 rounded"
    style={{ left: tooltip.x + 10, top: tooltip.y - 30 }}
  >
    ${tooltip.value.toFixed(2)}
  </div>
)}
```

---

#### 3. 렌더링 성능 최적화 (현재 양호)

**현재 최적화 상태**:
```jsx
// useMemo로 계산 캐싱 ✅
const normalized = useMemo(() => {...}, [data, height])
const pathD = useMemo(() => {...}, [normalized, height])

// React Query 캐싱 ✅
staleTime: 60 * 1000,
gcTime: 5 * 60 * 1000,
refetchInterval: 60 * 1000,
```

**추가 최적화 제안**:
```jsx
// 1. 컴포넌트 메모이제이션
export default React.memo(Sparkline)

// 2. 데이터 변경 시에만 리렌더
// (현재 useMemo로 충분하지만, memo 추가로 더 안전)
```

---

### 4.3 반응형 크기 조절 (현재 구현됨)

**현재 구현**:
```jsx
<svg
  viewBox={`0 0 100 ${height}`}
  preserveAspectRatio="none"  // 비율 무시, 컨테이너에 맞춤
  width={width}               // 기본값 '100%'
  height={height}             // 기본값 32
/>
```

**추가 제안**: 컨테이너 크기에 따른 동적 높이

```jsx
// 큰 화면에서는 더 높은 sparkline
const responsiveHeight = useMediaQuery({
  sm: 24,
  md: 32,
  lg: 40,
})
```

---

## 5. 구현 우선순위

### Phase 1: 백엔드 API (2일)

| 순위 | 항목 | 복잡도 |
|------|------|--------|
| 1 | `/prices/{ticker}/sparkline` API | 낮음 |
| 2 | Mock 데이터 제거, 실제 데이터 연동 | 낮음 |
| 3 | 다운샘플링 유틸리티 | 낮음 |

### Phase 2: 기간 선택 (1일)

| 순위 | 항목 | 복잡도 |
|------|------|--------|
| 1 | Sparkline 컴포넌트 period prop 추가 | 낮음 |
| 2 | 호버 시 기간 선택 UI | 낮음 |
| 3 | usePriceHistory 훅 수정 | 낮음 |

### Phase 3: 인터랙션 (2일)

| 순위 | 항목 | 복잡도 |
|------|------|--------|
| 1 | 툴팁 (호버 시 가격 표시) | 중간 |
| 2 | 클릭 시 상세 차트 열기 | 낮음 |

### Phase 4: 추가 데이터 타입 (3일)

| 순위 | 항목 | 복잡도 |
|------|------|--------|
| 1 | 거래량 Sparkline | 중간 |
| 2 | 예측 확률 Sparkline | 높음 |

---

## 6. 권장 구현 방안

### 6.1 최종 권장사항

```
✅ 현재 Custom SVG 구현 유지 (최적)
✅ 백엔드 Sparkline API 추가 (실제 데이터)
✅ 기간 선택 기능 추가 (1H/1D/1W)
✅ 툴팁 기능 추가 (선택적)
❌ 새 라이브러리 도입 불필요
❌ Canvas 전환 불필요
```

### 6.2 예상 작업량

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| Phase 1 | 백엔드 API | 2일 |
| Phase 2 | 기간 선택 | 1일 |
| Phase 3 | 툴팁 | 1일 |
| **총계** | | **4일** |

---

## 7. 백엔드 API 상세 설계

### 7.1 Sparkline API

**엔드포인트**: `GET /api/prices/{ticker}/sparkline`

**파라미터**:
| 이름 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| period | string | "1H" | 기간 (1H, 1D, 1W, 1M) |

**응답**:
```json
{
  "symbol": "AAPL",
  "period": "1H",
  "data": [150.12, 150.45, 150.23, ...],  // 60개 포인트
  "direction": "up",
  "min": 149.50,
  "max": 151.20,
  "change_percent": 0.85,
  "timestamp": "2025-12-23T15:30:00Z"
}
```

### 7.2 다운샘플링 알고리즘

```python
def downsample(data: List[dict], target_points: int = 60) -> List[dict]:
    """
    시계열 데이터를 target_points 개수로 다운샘플링.
    LTTB (Largest Triangle Three Buckets) 알고리즘 사용 권장.
    """
    if len(data) <= target_points:
        return data

    # 간단한 구현: 균등 간격 샘플링
    step = len(data) / target_points
    return [data[int(i * step)] for i in range(target_points)]

    # 고급 구현: LTTB 알고리즘 (시각적 특성 보존)
    # from lttb import downsample
    # return downsample(data, target_points)
```

---

## 8. 요약

### 8.1 현재 상태
- Sparkline 이미 구현됨 (Custom SVG, 최적화됨)
- Mock 데이터 사용 중 (백엔드 API 미연동)
- recharts 설치됨 (미사용)

### 8.2 개선 방향
1. **백엔드 API 추가** - 실제 데이터 연동
2. **기간 선택 기능** - 1H/1D/1W 옵션
3. **툴팁 추가** - 호버 시 가격 표시
4. **라이브러리 변경 불필요** - 현재 구현 최적

### 8.3 예상 총 작업량
- 백엔드: 2일
- 프론트엔드: 2일
- **총: 4일**

---

*분석팀장 작성*
