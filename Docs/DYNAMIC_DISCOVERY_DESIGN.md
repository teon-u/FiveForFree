# 동적 종목 발견 기능 설계서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 개요

### 1.1 목적

실시간으로 **새로운 급등주/거래량 상위 종목**을 자동 발견하고 사용자에게 알려주는 기능

### 1.2 관련 API

```
GET /api/status/discover
```

### 1.3 주요 기능

| 기능 | 설명 |
|------|------|
| 신규 급등주 발견 | 기존 목록에 없는 새 Gainers 감지 |
| 학습 상태 표시 | 신규 종목의 모델 학습 여부 표시 |
| 자동 알림 | 새 종목 발견 시 알림 |
| 수동 학습 트리거 | 미학습 종목 즉시 학습 요청 |

---

## 2. API 응답 구조

### 2.1 /api/status/discover 응답

```json
{
  "timestamp": "2025-12-21T14:30:00Z",
  "summary": {
    "total_tickers": 150,
    "trained_tickers": 142,
    "model_coverage": 94.7,
    "new_gainers_count": 5,
    "new_volume_count": 3
  },
  "new_gainers": [
    {
      "ticker": "PLTR",
      "name": "Palantir Technologies Inc.",
      "change_percent": 12.5,
      "volume": 85000000,
      "sector": "technology",
      "is_trained": false,
      "discovered_at": "2025-12-21T14:25:00Z"
    },
    {
      "ticker": "RIVN",
      "name": "Rivian Automotive Inc.",
      "change_percent": 8.3,
      "volume": 45000000,
      "sector": "consumer",
      "is_trained": false,
      "discovered_at": "2025-12-21T14:20:00Z"
    }
  ],
  "new_volume_top": [
    {
      "ticker": "GME",
      "name": "GameStop Corp.",
      "change_percent": 5.2,
      "volume": 120000000,
      "sector": "consumer",
      "is_trained": true,
      "discovered_at": "2025-12-21T14:15:00Z"
    }
  ],
  "training_queue": [
    {
      "ticker": "PLTR",
      "status": "pending",
      "position": 1,
      "estimated_time": 120
    }
  ]
}
```

---

## 3. UI 컴포넌트 설계

### 3.1 발견 알림 배너

대시보드 상단에 새 종목 발견 시 표시

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔔 새로운 급등주 발견! PLTR +12.5%, RIVN +8.3%       [자세히 보기]  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 발견 패널 (Discovery Panel)

설정 패널 또는 별도 탭에 위치

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔍 종목 발견                                              [새로고침] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 📊 현황                                                             │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│ │ 전체 종목    │  │ 학습 완료    │  │ 모델 커버리지 │              │
│ │     150      │  │     142      │  │    94.7%     │              │
│ └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│ 🔥 신규 급등주 (5)                                                  │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ PLTR  Palantir    +12.5%  📊 85M  ⚠️ 미학습   [🎓 학습하기]     ││
│ │ RIVN  Rivian      +8.3%   📊 45M  ⚠️ 미학습   [🎓 학습하기]     ││
│ │ SOFI  SoFi Tech   +6.2%   📊 32M  ✅ 완료     [📈 상세보기]     ││
│ │ LCID  Lucid       +5.8%   📊 28M  ⚠️ 미학습   [🎓 학습하기]     ││
│ │ HOOD  Robinhood   +4.5%   📊 22M  ✅ 완료     [📈 상세보기]     ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 📊 신규 거래량 상위 (3)                                             │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ GME   GameStop    +5.2%   📊 120M  ✅ 완료    [📈 상세보기]     ││
│ │ AMC   AMC Ent.    +3.1%   📊 95M   ✅ 완료    [📈 상세보기]     ││
│ │ BBBY  Bed Bath    +2.8%   📊 78M   ⚠️ 미학습  [🎓 학습하기]     ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 📋 학습 대기열 (3)                                                  │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 1. PLTR - 대기중         [예상 2분]                              ││
│ │ 2. RIVN - 대기중         [예상 4분]                              ││
│ │ 3. LCID - 대기중         [예상 6분]                              ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│                      [🎓 전체 미학습 종목 학습]                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 컴포넌트 구현

### 4.1 DiscoveryBanner.jsx

```jsx
import { useState, useEffect } from 'react'
import { useDiscovery } from '../hooks/useDiscovery'
import { useSettingsStore } from '../stores/settingsStore'

export default function DiscoveryBanner({ onOpenPanel }) {
  const { language } = useSettingsStore()
  const { discovery, isLoading } = useDiscovery()
  const [dismissed, setDismissed] = useState(false)

  // 새 종목 없거나 이미 닫았으면 표시 안함
  if (dismissed || isLoading || !discovery?.new_gainers?.length) {
    return null
  }

  const newGainers = discovery.new_gainers.filter(g => !g.is_trained)
  if (newGainers.length === 0) return null

  const topGainers = newGainers.slice(0, 3)
  const moreCount = newGainers.length - 3

  return (
    <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-3 mb-4 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-xl">🔔</span>
        <span className="text-yellow-400 font-medium">
          새로운 급등주 발견!
        </span>
        <span className="text-white">
          {topGainers.map((g, i) => (
            <span key={g.ticker}>
              {i > 0 && ', '}
              <span className="font-bold">{g.ticker}</span>
              <span className="text-green-400 ml-1">+{g.change_percent.toFixed(1)}%</span>
            </span>
          ))}
          {moreCount > 0 && (
            <span className="text-gray-400"> 외 {moreCount}개</span>
          )}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onOpenPanel}
          className="px-3 py-1 bg-yellow-500 text-black rounded-lg text-sm font-medium hover:bg-yellow-400"
        >
          자세히 보기
        </button>
        <button
          onClick={() => setDismissed(true)}
          className="text-gray-400 hover:text-white"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
```

### 4.2 DiscoveryPanel.jsx

```jsx
import { useState } from 'react'
import { useDiscovery } from '../hooks/useDiscovery'
import { endpoints } from '../services/api'

export default function DiscoveryPanel({ onClose }) {
  const { discovery, isLoading, refetch } = useDiscovery()
  const [trainingStatus, setTrainingStatus] = useState({})

  const handleTrain = async (ticker) => {
    setTrainingStatus(prev => ({ ...prev, [ticker]: 'training' }))
    try {
      await endpoints.trainTicker(ticker)
      setTrainingStatus(prev => ({ ...prev, [ticker]: 'queued' }))
      setTimeout(refetch, 2000) // 상태 갱신
    } catch (error) {
      setTrainingStatus(prev => ({ ...prev, [ticker]: 'error' }))
    }
  }

  const handleTrainAll = async () => {
    const untrained = discovery.new_gainers.filter(g => !g.is_trained)
    for (const gainer of untrained) {
      await handleTrain(gainer.ticker)
    }
  }

  if (isLoading) {
    return <div className="p-8 text-center"><div className="spinner" /></div>
  }

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />

      <div className="fixed right-0 top-0 bottom-0 w-[450px] bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto">
        {/* 헤더 */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            🔍 종목 발견
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={refetch}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              새로고침
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white text-2xl"
            >
              ✕
            </button>
          </div>
        </div>

        {/* 콘텐츠 */}
        <div className="p-6 space-y-6">
          {/* 현황 카드 */}
          <div className="grid grid-cols-3 gap-3">
            <StatCard label="전체 종목" value={discovery?.summary?.total_tickers} />
            <StatCard label="학습 완료" value={discovery?.summary?.trained_tickers} />
            <StatCard
              label="커버리지"
              value={`${discovery?.summary?.model_coverage?.toFixed(1)}%`}
              highlight
            />
          </div>

          {/* 신규 급등주 */}
          <section>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              🔥 신규 급등주
              <span className="text-sm text-gray-400">
                ({discovery?.new_gainers?.length || 0})
              </span>
            </h3>
            <div className="space-y-2">
              {discovery?.new_gainers?.map((gainer) => (
                <TickerRow
                  key={gainer.ticker}
                  ticker={gainer}
                  trainingStatus={trainingStatus[gainer.ticker]}
                  onTrain={() => handleTrain(gainer.ticker)}
                />
              ))}
            </div>
          </section>

          {/* 신규 거래량 상위 */}
          <section>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              📊 신규 거래량 상위
              <span className="text-sm text-gray-400">
                ({discovery?.new_volume_top?.length || 0})
              </span>
            </h3>
            <div className="space-y-2">
              {discovery?.new_volume_top?.map((ticker) => (
                <TickerRow
                  key={ticker.ticker}
                  ticker={ticker}
                  trainingStatus={trainingStatus[ticker.ticker]}
                  onTrain={() => handleTrain(ticker.ticker)}
                />
              ))}
            </div>
          </section>

          {/* 학습 대기열 */}
          {discovery?.training_queue?.length > 0 && (
            <section>
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                📋 학습 대기열
                <span className="text-sm text-gray-400">
                  ({discovery?.training_queue?.length})
                </span>
              </h3>
              <div className="bg-surface-light rounded-lg p-3 space-y-2">
                {discovery.training_queue.map((item, index) => (
                  <div
                    key={item.ticker}
                    className="flex items-center justify-between text-sm"
                  >
                    <span>
                      {index + 1}. <span className="font-bold">{item.ticker}</span>
                      <span className="text-gray-400 ml-2">
                        {item.status === 'training' ? '학습중...' : '대기중'}
                      </span>
                    </span>
                    <span className="text-gray-400">
                      예상 {Math.ceil(item.estimated_time / 60)}분
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 전체 학습 버튼 */}
          {discovery?.new_gainers?.some(g => !g.is_trained) && (
            <button
              onClick={handleTrainAll}
              className="w-full py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600"
            >
              🎓 전체 미학습 종목 학습
            </button>
          )}
        </div>
      </div>
    </>
  )
}

function StatCard({ label, value, highlight = false }) {
  return (
    <div className="bg-surface-light rounded-lg p-3 text-center">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-xl font-bold ${highlight ? 'text-green-400' : 'text-white'}`}>
        {value}
      </div>
    </div>
  )
}

function TickerRow({ ticker, trainingStatus, onTrain }) {
  const isTrained = ticker.is_trained || trainingStatus === 'queued'
  const isTraining = trainingStatus === 'training'

  return (
    <div className="flex items-center justify-between p-3 bg-surface-light rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-bold text-white">{ticker.ticker}</span>
          <span className="text-green-400 text-sm">+{ticker.change_percent?.toFixed(1)}%</span>
        </div>
        <div className="text-xs text-gray-400 truncate max-w-[180px]">
          {ticker.name}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400">
          📊 {formatVolume(ticker.volume)}
        </span>
        <span className={`text-xs ${isTrained ? 'text-green-400' : 'text-yellow-400'}`}>
          {isTrained ? '✅' : '⚠️'}
        </span>

        {isTrained ? (
          <button className="px-3 py-1 text-xs bg-blue-500/20 text-blue-400 rounded">
            📈 상세
          </button>
        ) : (
          <button
            onClick={onTrain}
            disabled={isTraining}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              isTraining
                ? 'bg-yellow-500/20 text-yellow-400 cursor-wait'
                : 'bg-green-500 text-white hover:bg-green-600'
            }`}
          >
            {isTraining ? '학습중...' : '🎓 학습'}
          </button>
        )}
      </div>
    </div>
  )
}

function formatVolume(volume) {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(0)}M`
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(0)}K`
  return volume
}
```

### 4.3 useDiscovery.js

```javascript
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '../services/api'

export function useDiscovery() {
  return useQuery({
    queryKey: ['discovery'],
    queryFn: () => endpoints.getDiscovery(),
    staleTime: 60 * 1000, // 1분
    refetchInterval: 5 * 60 * 1000, // 5분마다 자동 갱신
    select: (response) => response.data,
  })
}
```

---

## 5. 자동 갱신 로직

### 5.1 폴링 전략

| 조건 | 갱신 주기 |
|------|-----------|
| 장 마감 | 비활성화 |
| 장중 (정규거래) | 5분 |
| 프리마켓/애프터마켓 | 15분 |
| 새 종목 발견 후 | 2분 (일시적) |

### 5.2 알림 트리거

```javascript
// 새 종목 발견 시 알림
useEffect(() => {
  if (discovery?.new_gainers?.length > previousCount) {
    const newTickers = discovery.new_gainers.filter(g => !g.is_trained)
    if (newTickers.length > 0) {
      showNotification({
        title: '새로운 급등주 발견!',
        body: `${newTickers.map(t => t.ticker).join(', ')} 발견`,
        icon: '/icons/icon-192.png',
      })
    }
  }
}, [discovery?.new_gainers])
```

---

## 6. 반응형 디자인

### 6.1 모바일 레이아웃

```
┌─────────────────────────────┐
│ 🔍 종목 발견           [✕]  │
├─────────────────────────────┤
│ ┌─────┐ ┌─────┐ ┌───────┐  │
│ │ 150 │ │ 142 │ │ 94.7% │  │
│ │전체 │ │완료 │ │커버리지│  │
│ └─────┘ └─────┘ └───────┘  │
├─────────────────────────────┤
│ 🔥 신규 급등주 (5)          │
│ ┌─────────────────────────┐ │
│ │ PLTR   +12.5%  [🎓학습] │ │
│ │ RIVN   +8.3%   [🎓학습] │ │
│ └─────────────────────────┘ │
├─────────────────────────────┤
│ [🎓 전체 미학습 종목 학습]  │
└─────────────────────────────┘
```

---

## 7. 구현 일정

### Phase 1: 기본 구조 (1일)
- [ ] useDiscovery.js 훅 구현
- [ ] API 엔드포인트 연동

### Phase 2: UI 컴포넌트 (1.5일)
- [ ] DiscoveryBanner.jsx 구현
- [ ] DiscoveryPanel.jsx 구현
- [ ] 학습 트리거 연동

### Phase 3: 알림 및 자동화 (0.5일)
- [ ] 자동 갱신 로직
- [ ] 브라우저 알림 연동
- [ ] 폴링 전략 적용

---

## 8. 테스트 체크리스트

| 항목 | 확인 |
|------|------|
| 발견 배너 표시 | [ ] |
| 발견 패널 열기/닫기 | [ ] |
| 학습 버튼 클릭 | [ ] |
| 학습 상태 업데이트 | [ ] |
| 전체 학습 버튼 | [ ] |
| 학습 대기열 표시 | [ ] |
| 자동 갱신 작동 | [ ] |
| 모바일 레이아웃 | [ ] |

---

*이 문서는 분석팀장이 작성하였습니다.*
