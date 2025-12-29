# 차트 개선 기획서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 현재 상태 분석

### 1.1 기존 차트 구성

| 컴포넌트 | 위치 | 기능 |
|----------|------|------|
| PriceChart.jsx | 예측 패널 | 분봉 캔들스틱 차트 |
| ModelComparison.jsx | 모델 상세 | 모델별 성능 비교 |

### 1.2 현재 한계점

1. **티커 카드에 차트 없음**: 트렌드 한눈에 파악 어려움
2. **상세 차트 접근 복잡**: 여러 단계 클릭 필요
3. **시간 범위 제한**: 고정된 기간만 표시
4. **지표 부재**: 이동평균, 볼린저 밴드 등 없음

---

## 2. 스파크라인 설계

### 2.1 개념

티커 카드에 소형 차트를 추가하여 **즉각적인 트렌드 파악** 가능

```
┌─────────────────────────────────┐
│ NVDA                   🟢      │
│ ┌─────────────────────────┐    │
│ │    ╱╲   ╱╲              │    │  ← 스파크라인 (60분 트렌드)
│ │   ╱  ╲ ╱  ╲╱╲           │    │
│ │  ╱    ╲      ╲╱╲        │    │
│ └─────────────────────────┘    │
│ 82% ↑  $142.50  +5.2%     [A]  │
└─────────────────────────────────┘
```

### 2.2 스파크라인 스펙

| 속성 | 값 | 설명 |
|------|-----|------|
| 너비 | 100% (카드 너비) | 반응형 |
| 높이 | 32px | 컴팩트 |
| 데이터 포인트 | 60개 | 최근 60분 |
| 라인 두께 | 1.5px | 가독성 |
| 색상 | 상승: 녹색, 하락: 빨강 | 방향 표시 |
| 애니메이션 | 진입 시 Draw 효과 | 시각적 피드백 |

### 2.3 스파크라인 컴포넌트

```jsx
// Sparkline.jsx

import { useMemo } from 'react'

export default function Sparkline({ data, direction, width = '100%', height = 32 }) {
  // 데이터 정규화
  const normalized = useMemo(() => {
    if (!data || data.length === 0) return []
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1
    return data.map(v => ((v - min) / range) * height)
  }, [data, height])

  // SVG 경로 생성
  const pathD = useMemo(() => {
    if (normalized.length === 0) return ''
    const step = 100 / (normalized.length - 1)
    return normalized.reduce((acc, y, i) => {
      const x = i * step
      const yPos = height - y
      return acc + (i === 0 ? `M ${x},${yPos}` : ` L ${x},${yPos}`)
    }, '')
  }, [normalized, height])

  // 그라디언트 경로 (영역 채우기)
  const areaD = pathD + ` L 100,${height} L 0,${height} Z`

  const color = direction === 'up' ? '#22c55e' : '#ef4444'
  const gradientId = `sparkline-gradient-${direction}`

  return (
    <svg
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      width={width}
      height={height}
      className="overflow-visible"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* 영역 채우기 */}
      <path
        d={areaD}
        fill={`url(#${gradientId})`}
        className="transition-all duration-500"
      />

      {/* 라인 */}
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-all duration-500"
      />

      {/* 현재 가격 점 */}
      {normalized.length > 0 && (
        <circle
          cx="100"
          cy={height - normalized[normalized.length - 1]}
          r="2"
          fill={color}
        />
      )}
    </svg>
  )
}
```

### 2.4 데이터 소스

```javascript
// API: GET /api/prices/{symbol}/sparkline
{
  "symbol": "NVDA",
  "interval": "1m",
  "data": [141.50, 141.75, 142.00, 142.10, 141.90, ...], // 60개
  "direction": "up", // 현재 트렌드
  "change": 0.82 // 60분 변동률
}
```

---

## 3. 상세 차트 모달 설계

### 3.1 트리거

```
[티커 카드] → [📈 차트] 버튼 클릭 → [상세 차트 모달]
```

### 3.2 모달 레이아웃

```
┌─────────────────────────────────────────────────────────────────────┐
│ NVDA - NVIDIA Corporation                                      [✕] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ [1분] [5분] [15분] [1시간] [1일]   |   [1D] [1W] [1M] [3M]         │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │                                                                 ││
│ │     ╱╲                                                         ││
│ │    ╱  ╲   ╱╲    ╱╲                                            ││
│ │   ╱    ╲ ╱  ╲  ╱  ╲                                           ││
│ │  ╱      ╲    ╲╱    ╲                                          ││
│ │ ╱                    ╲╱╲                                        ││
│ │                         ╲                                      ││
│ │─────────────────────────────────────────────────────────────────││
│ │ MA5 ─── MA20 ─── BB ═══                                        ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│ │ 현재가        │  │ 변동률        │  │ 거래량        │              │
│ │ $142.50      │  │ +5.2%        │  │ 45.2M        │              │
│ └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 예측 정보                                                       ││
│ │ 방향: ↑ 상승  |  확률: 82%  |  등급: A  |  Model: XGBoost      ││
│ └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 차트 기능

#### 3.3.1 시간 간격 선택

| 버튼 | 데이터 간격 | 표시 범위 |
|------|------------|-----------|
| 1분 | 1분봉 | 최근 60분 |
| 5분 | 5분봉 | 최근 5시간 |
| 15분 | 15분봉 | 최근 1일 |
| 1시간 | 1시간봉 | 최근 5일 |
| 1일 | 일봉 | 최근 3개월 |

#### 3.3.2 기간 선택

| 버튼 | 기간 |
|------|------|
| 1D | 1일 |
| 1W | 1주 |
| 1M | 1개월 |
| 3M | 3개월 |

#### 3.3.3 기술적 지표

| 지표 | 설명 | 토글 |
|------|------|------|
| MA5 | 5일 이동평균 | ON/OFF |
| MA20 | 20일 이동평균 | ON/OFF |
| Bollinger Band | 볼린저 밴드 | ON/OFF |
| Volume | 거래량 바 | ON/OFF |

### 3.4 상호작용

| 제스처 | 동작 |
|--------|------|
| 마우스 호버 | 크로스헤어 + 툴팁 |
| 드래그 | 기간 이동 |
| 스크롤 | 줌 인/아웃 |
| 더블 클릭 | 줌 리셋 |
| 핀치 (모바일) | 줌 인/아웃 |
| 팬 (모바일) | 기간 이동 |

---

## 4. 차트 모달 컴포넌트

### 4.1 ChartModal.jsx

```jsx
import { useState, useEffect } from 'react'
import { usePriceHistory } from '../hooks/usePriceHistory'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Area
} from 'recharts'

const INTERVALS = ['1m', '5m', '15m', '1h', '1d']
const PERIODS = ['1D', '1W', '1M', '3M']

export default function ChartModal({ ticker, prediction, onClose }) {
  const [interval, setInterval] = useState('1m')
  const [period, setPeriod] = useState('1D')
  const [showMA5, setShowMA5] = useState(true)
  const [showMA20, setShowMA20] = useState(true)
  const [showBB, setShowBB] = useState(false)

  const { data, isLoading } = usePriceHistory(ticker, interval, period)

  // 이동평균 계산
  const chartData = useMemo(() => {
    if (!data) return []
    return data.map((item, index, arr) => ({
      ...item,
      ma5: calculateMA(arr, index, 5),
      ma20: calculateMA(arr, index, 20),
      bbUpper: calculateBB(arr, index, 20, 2).upper,
      bbLower: calculateBB(arr, index, 20, 2).lower,
    }))
  }, [data])

  return (
    <>
      {/* 백드롭 */}
      <div className="modal-backdrop" onClick={onClose} />

      {/* 모달 */}
      <div className="fixed inset-4 md:inset-8 lg:inset-16 bg-surface rounded-2xl shadow-2xl z-50 flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between p-4 border-b border-surface-light">
          <div>
            <h2 className="text-xl font-bold">{ticker}</h2>
            <p className="text-sm text-gray-400">{prediction?.name}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ✕
          </button>
        </div>

        {/* 컨트롤 */}
        <div className="flex items-center gap-4 p-4 border-b border-surface-light">
          {/* 시간 간격 */}
          <div className="flex gap-1">
            {INTERVALS.map((int) => (
              <button
                key={int}
                onClick={() => setInterval(int)}
                className={`px-3 py-1 rounded text-sm ${
                  interval === int
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {int}
              </button>
            ))}
          </div>

          <div className="w-px h-6 bg-surface-light" />

          {/* 기간 */}
          <div className="flex gap-1">
            {PERIODS.map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1 rounded text-sm ${
                  period === p
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {p}
              </button>
            ))}
          </div>

          <div className="flex-1" />

          {/* 지표 토글 */}
          <div className="flex gap-2 text-sm">
            <button
              onClick={() => setShowMA5(!showMA5)}
              className={`px-2 py-1 rounded ${showMA5 ? 'bg-blue-500/30 text-blue-400' : 'text-gray-500'}`}
            >
              MA5
            </button>
            <button
              onClick={() => setShowMA20(!showMA20)}
              className={`px-2 py-1 rounded ${showMA20 ? 'bg-yellow-500/30 text-yellow-400' : 'text-gray-500'}`}
            >
              MA20
            </button>
            <button
              onClick={() => setShowBB(!showBB)}
              className={`px-2 py-1 rounded ${showBB ? 'bg-purple-500/30 text-purple-400' : 'text-gray-500'}`}
            >
              BB
            </button>
          </div>
        </div>

        {/* 차트 영역 */}
        <div className="flex-1 p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="spinner" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
                <YAxis stroke="#9ca3af" fontSize={12} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                  labelStyle={{ color: '#9ca3af' }}
                />

                {/* 볼린저 밴드 */}
                {showBB && (
                  <>
                    <Area
                      dataKey="bbUpper"
                      stroke="none"
                      fill="#a855f7"
                      fillOpacity={0.1}
                    />
                    <Line
                      dataKey="bbUpper"
                      stroke="#a855f7"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                    <Line
                      dataKey="bbLower"
                      stroke="#a855f7"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </>
                )}

                {/* 이동평균 */}
                {showMA5 && (
                  <Line
                    dataKey="ma5"
                    stroke="#3b82f6"
                    strokeWidth={1}
                    dot={false}
                  />
                )}
                {showMA20 && (
                  <Line
                    dataKey="ma20"
                    stroke="#eab308"
                    strokeWidth={1}
                    dot={false}
                  />
                )}

                {/* 가격 라인 */}
                <Line
                  dataKey="close"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* 하단 정보 */}
        <div className="p-4 border-t border-surface-light grid grid-cols-3 md:grid-cols-6 gap-4">
          <InfoCard label="현재가" value={`$${prediction?.current_price?.toFixed(2)}`} />
          <InfoCard
            label="변동률"
            value={`${prediction?.change_percent >= 0 ? '+' : ''}${prediction?.change_percent?.toFixed(2)}%`}
            valueClass={prediction?.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}
          />
          <InfoCard label="거래량" value={formatVolume(prediction?.volume)} />
          <InfoCard
            label="예측"
            value={`${prediction?.probability}% ${prediction?.direction === 'up' ? '↑' : '↓'}`}
            valueClass={prediction?.direction === 'up' ? 'text-green-400' : 'text-red-400'}
          />
          <InfoCard label="모델" value={prediction?.best_model?.toUpperCase()} />
          <InfoCard label="등급" value={prediction?.practicality_grade} />
        </div>
      </div>
    </>
  )
}

function InfoCard({ label, value, valueClass = '' }) {
  return (
    <div className="bg-surface-light rounded-lg p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-lg font-bold ${valueClass}`}>{value}</div>
    </div>
  )
}
```

---

## 5. 성능 최적화

### 5.1 데이터 캐싱

```javascript
// usePriceHistory.js
import { useQuery } from '@tanstack/react-query'

export function usePriceHistory(ticker, interval, period) {
  return useQuery({
    queryKey: ['priceHistory', ticker, interval, period],
    queryFn: () => fetchPriceHistory(ticker, interval, period),
    staleTime: 60 * 1000, // 1분
    cacheTime: 5 * 60 * 1000, // 5분
    refetchInterval: interval === '1m' ? 60 * 1000 : false,
  })
}
```

### 5.2 차트 렌더링 최적화

```jsx
// 리렌더링 방지
const MemoizedChart = memo(({ data, showMA5, showMA20, showBB }) => {
  return (
    <ResponsiveContainer>
      <LineChart data={data}>
        {/* ... */}
      </LineChart>
    </ResponsiveContainer>
  )
}, (prevProps, nextProps) => {
  return (
    prevProps.data === nextProps.data &&
    prevProps.showMA5 === nextProps.showMA5 &&
    prevProps.showMA20 === nextProps.showMA20 &&
    prevProps.showBB === nextProps.showBB
  )
})
```

### 5.3 스파크라인 최적화

```jsx
// Canvas 기반 스파크라인 (대용량 데이터)
function CanvasSparkline({ data, direction }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // 데이터 정규화 및 그리기
    ctx.clearRect(0, 0, width, height)
    ctx.strokeStyle = direction === 'up' ? '#22c55e' : '#ef4444'
    ctx.lineWidth = 1.5

    ctx.beginPath()
    data.forEach((value, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - (value * height)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
  }, [data, direction])

  return <canvas ref={canvasRef} width={100} height={32} className="w-full" />
}
```

---

## 6. 모바일 최적화

### 6.1 스파크라인 모바일

```jsx
// 모바일에서 스파크라인 단순화
<Sparkline
  data={data}
  direction={direction}
  simplified={isMobile} // 포인트 수 감소 (60 → 30)
/>
```

### 6.2 차트 모달 모바일

```jsx
// 모바일 풀스크린 모달
<div className={clsx(
  'fixed z-50 bg-surface flex flex-col',
  isMobile
    ? 'inset-0' // 풀스크린
    : 'inset-4 md:inset-8 lg:inset-16 rounded-2xl shadow-2xl'
)}>
  {/* 내용 */}
</div>
```

### 6.3 터치 제스처

```jsx
// 핀치 줌 / 팬 지원
import { useGesture } from '@use-gesture/react'

const bind = useGesture({
  onPinch: ({ offset: [scale] }) => {
    setZoomLevel(scale)
  },
  onDrag: ({ movement: [x] }) => {
    setPanOffset(x)
  },
})
```

---

## 7. API 요구사항

### 7.1 스파크라인 API

#### GET /api/prices/{symbol}/sparkline

```json
{
  "symbol": "NVDA",
  "data": [141.50, 141.75, ...], // 60개
  "direction": "up",
  "min": 140.20,
  "max": 143.50,
  "change": 0.82,
  "timestamp": "2025-12-21T13:45:00Z"
}
```

### 7.2 히스토리 API

#### GET /api/prices/{symbol}/history

**Parameters:**
- `interval`: 1m, 5m, 15m, 1h, 1d
- `period`: 1D, 1W, 1M, 3M

**Response:**
```json
{
  "symbol": "NVDA",
  "interval": "1m",
  "data": [
    {
      "time": "2025-12-21T13:00:00Z",
      "open": 141.50,
      "high": 141.80,
      "low": 141.30,
      "close": 141.75,
      "volume": 125000
    }
  ]
}
```

---

## 8. 구현 일정

### Phase 1: 스파크라인 (2일)
- [ ] Sparkline.jsx 컴포넌트 구현
- [ ] API 엔드포인트 추가
- [ ] TickerCard.jsx 통합
- [ ] 성능 테스트

### Phase 2: 차트 모달 (3일)
- [ ] ChartModal.jsx 구현
- [ ] 시간 간격/기간 선택
- [ ] 기술적 지표 (MA, BB)
- [ ] 반응형 레이아웃

### Phase 3: 인터랙션 (2일)
- [ ] 툴팁 / 크로스헤어
- [ ] 줌 / 팬 기능
- [ ] 모바일 터치 제스처

### Phase 4: 최적화 (1일)
- [ ] 데이터 캐싱
- [ ] 렌더링 최적화
- [ ] 번들 크기 점검

---

## 9. 테스트 체크리스트

| 항목 | 확인 |
|------|------|
| 스파크라인 표시 | [ ] |
| 상승/하락 색상 정확 | [ ] |
| 차트 모달 열기/닫기 | [ ] |
| 시간 간격 변경 | [ ] |
| 기간 변경 | [ ] |
| MA5/MA20/BB 토글 | [ ] |
| 툴팁 표시 | [ ] |
| 줌 인/아웃 | [ ] |
| 모바일 풀스크린 | [ ] |
| 터치 제스처 | [ ] |

---

*이 문서는 분석팀장이 작성하였습니다.*
