# 알림 기능 및 Watchlist 설계서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 개요

### 1.1 목적

- **브라우저 알림**: 중요한 이벤트(새 신호, 급등주 발견)를 실시간 알림
- **Watchlist**: 관심 종목 저장 및 우선 모니터링

### 1.2 주요 기능

| 기능 | 설명 |
|------|------|
| 브라우저 푸시 알림 | 강한 신호, 급등주 발견 시 알림 |
| 인앱 알림 센터 | 알림 히스토리 및 관리 |
| Watchlist | 관심 종목 저장 및 빠른 접근 |
| 조건부 알림 | 사용자 정의 조건 충족 시 알림 |

---

## 2. 브라우저 알림 시스템

### 2.1 알림 권한 요청 플로우

```
[첫 방문] → [알림 허용 배너] → [권한 요청] → [허용/거부]
                                      ↓
                               [설정에 저장]
```

### 2.2 알림 유형

| 유형 | 트리거 조건 | 우선순위 |
|------|------------|----------|
| 강한 상승 신호 | 확률 80%+ AND 등급 A | 높음 |
| 강한 하락 신호 | 확률 80%+ AND 등급 A | 높음 |
| 새 급등주 발견 | 미학습 종목 발견 | 중간 |
| 학습 완료 | 요청한 종목 학습 완료 | 낮음 |
| Watchlist 신호 | 관심 종목에서 신호 발생 | 높음 |

### 2.3 알림 UI

```
┌─────────────────────────────────────────────────┐
│ 🔔 FiveForFree                                  │
│                                                 │
│ 강한 상승 신호!                                  │
│ NVDA 82% ↑ (등급 A)                             │
│                                                 │
│ $142.50 +5.2%                                   │
└─────────────────────────────────────────────────┘
```

### 2.4 알림 권한 배너

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔔 알림을 켜면 중요한 매매 신호를 실시간으로 받을 수 있습니다       │
│                                        [알림 켜기]  [나중에]       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 인앱 알림 센터

### 3.1 알림 센터 UI

헤더의 알림 아이콘 클릭 시 표시

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔔 알림                                    [모두 읽음] [설정]      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 오늘                                                                │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 🟢 14:30  NVDA 강한 상승 신호                                    ││
│ │    82% ↑ (등급 A) • $142.50 +5.2%                    [상세보기] ││
│ ├─────────────────────────────────────────────────────────────────┤│
│ │ 🔥 14:15  새 급등주 발견                                         ││
│ │    PLTR +12.5%, RIVN +8.3%                          [자세히]    ││
│ ├─────────────────────────────────────────────────────────────────┤│
│ │ ✅ 14:00  TSLA 학습 완료                                         ││
│ │    예측 가능                                        [확인]      ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 어제                                                                │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 🔴 16:20  AAPL 하락 신호                                         ││
│ │    75% ↓ (등급 B) • 읽음                                        ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│                        [더 보기]                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 알림 아이콘 배지

```
┌─────────────────────┐
│ 🔔3                 │  ← 미읽은 알림 수
└─────────────────────┘
```

---

## 4. Watchlist 기능

### 4.1 Watchlist 추가 방법

```
1. 티커 카드 더블 탭 (모바일)
2. 티커 카드 스와이프 우측 (모바일)
3. 상세 패널 ⭐ 버튼 클릭
4. 티커 검색 후 추가
```

### 4.2 Watchlist 패널

```
┌─────────────────────────────────────────────────────────────────────┐
│ ⭐ Watchlist                            [편집]  [+ 추가]           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 📊 3개 종목 • 알림: ON                                             │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ ⭐ NVDA      82% ↑ [A]    $142.50 +5.2%    🔔    [⋮]           ││
│ │    NVIDIA Corporation                                           ││
│ ├─────────────────────────────────────────────────────────────────┤│
│ │ ⭐ TSLA      78% ↑ [A]    $245.30 +3.1%    🔔    [⋮]           ││
│ │    Tesla Inc                                                    ││
│ ├─────────────────────────────────────────────────────────────────┤│
│ │ ⭐ AAPL      65% ↓ [B]    $185.20 -1.2%    🔕    [⋮]           ││
│ │    Apple Inc                                                    ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 🔍 종목 검색...                                                 ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│                      [📥 내보내기]                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Watchlist 컨텍스트 메뉴

```
┌───────────────────┐
│ 📈 상세 보기      │
│ 🔔 알림 끄기      │
│ 📊 차트 보기      │
│ ───────────────── │
│ 🗑️ 삭제          │
└───────────────────┘
```

---

## 5. 알림 설정 UI

### 5.1 알림 설정 패널

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🔔 알림 설정                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 브라우저 알림                                                       │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 알림 허용                                              [ON]    ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 알림 유형                                                           │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 🟢 강한 상승 신호 (80%+, A등급)                        [ON]    ││
│ │ 🔴 강한 하락 신호 (80%+, A등급)                        [ON]    ││
│ │ 🔥 새 급등주 발견                                      [ON]    ││
│ │ ⭐ Watchlist 종목 신호                                 [ON]    ││
│ │ ✅ 학습 완료 알림                                      [OFF]   ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 조건부 알림                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 확률 임계값: __80__% 이상                                       ││
│ │ 등급: ☑ A  ☑ B  ☐ C  ☐ D                                       ││
│ │ 방향: ☑ 상승  ☑ 하락                                            ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 조용한 시간                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 야간 알림 끄기                                         [ON]    ││
│ │ 시간: __22:00__ ~ __08:00__                                     ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 컴포넌트 구현

### 6.1 NotificationPermissionBanner.jsx

```jsx
import { useState, useEffect } from 'react'
import { useNotificationStore } from '../stores/notificationStore'

export default function NotificationPermissionBanner() {
  const { permission, requestPermission, dismissBanner, bannerDismissed } = useNotificationStore()
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // 권한 미결정 + 배너 닫지 않음 → 표시
    if (permission === 'default' && !bannerDismissed) {
      setVisible(true)
    }
  }, [permission, bannerDismissed])

  if (!visible) return null

  const handleEnable = async () => {
    const granted = await requestPermission()
    if (granted) {
      setVisible(false)
    }
  }

  const handleDismiss = () => {
    dismissBanner()
    setVisible(false)
  }

  return (
    <div className="bg-blue-500/20 border border-blue-500/50 rounded-lg p-3 mb-4 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-xl">🔔</span>
        <span className="text-blue-400">
          알림을 켜면 중요한 매매 신호를 실시간으로 받을 수 있습니다
        </span>
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={handleEnable}
          className="px-4 py-1.5 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600"
        >
          알림 켜기
        </button>
        <button
          onClick={handleDismiss}
          className="text-gray-400 hover:text-white text-sm"
        >
          나중에
        </button>
      </div>
    </div>
  )
}
```

### 6.2 NotificationCenter.jsx

```jsx
import { useState } from 'react'
import { useNotifications } from '../hooks/useNotifications'
import { formatRelativeTime } from '../utils/dateUtils'

export default function NotificationCenter({ onClose }) {
  const { notifications, markAsRead, markAllAsRead, clearAll } = useNotifications()

  const groupedNotifications = groupByDate(notifications)

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />

      <div className="fixed right-4 top-16 w-96 max-h-[600px] bg-surface border border-surface-light rounded-xl shadow-2xl z-50 overflow-hidden flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between p-4 border-b border-surface-light">
          <h2 className="font-bold flex items-center gap-2">
            🔔 알림
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={markAllAsRead}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              모두 읽음
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>
        </div>

        {/* 알림 목록 */}
        <div className="flex-1 overflow-y-auto">
          {Object.entries(groupedNotifications).map(([date, items]) => (
            <div key={date}>
              <div className="px-4 py-2 text-xs text-gray-500 bg-surface-light/50">
                {date}
              </div>
              {items.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onRead={() => markAsRead(notification.id)}
                />
              ))}
            </div>
          ))}

          {notifications.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <div className="text-4xl mb-2">🔔</div>
              <p>알림이 없습니다</p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

function NotificationItem({ notification, onRead }) {
  const { type, title, message, ticker, time, read } = notification

  const getIcon = (type) => {
    switch (type) {
      case 'signal_up': return '🟢'
      case 'signal_down': return '🔴'
      case 'discovery': return '🔥'
      case 'training_complete': return '✅'
      case 'watchlist': return '⭐'
      default: return '🔔'
    }
  }

  return (
    <div
      className={clsx(
        'p-4 border-b border-surface-light cursor-pointer hover:bg-surface-light/50 transition-colors',
        !read && 'bg-blue-500/5'
      )}
      onClick={onRead}
    >
      <div className="flex items-start gap-3">
        <span className="text-lg">{getIcon(type)}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="font-medium text-sm">{title}</span>
            <span className="text-xs text-gray-500">{formatRelativeTime(time)}</span>
          </div>
          <p className="text-sm text-gray-400 truncate">{message}</p>
        </div>
        {!read && (
          <div className="w-2 h-2 bg-blue-500 rounded-full" />
        )}
      </div>
    </div>
  )
}
```

### 6.3 WatchlistPanel.jsx

```jsx
import { useState } from 'react'
import { useWatchlist } from '../hooks/useWatchlist'
import { usePredictions } from '../hooks/usePredictions'

export default function WatchlistPanel({ onClose }) {
  const { watchlist, addTicker, removeTicker, toggleAlert } = useWatchlist()
  const { predictions } = usePredictions()
  const [searchQuery, setSearchQuery] = useState('')
  const [isEditing, setIsEditing] = useState(false)

  // Watchlist 종목의 예측 데이터 가져오기
  const watchlistPredictions = watchlist
    .map(item => ({
      ...item,
      prediction: predictions?.find(p => p.ticker === item.ticker)
    }))
    .filter(item => item.prediction)

  // 검색 필터링
  const searchResults = searchQuery
    ? predictions?.filter(p =>
        p.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.name?.toLowerCase().includes(searchQuery.toLowerCase())
      ).slice(0, 5)
    : []

  const handleAddTicker = (ticker) => {
    addTicker(ticker)
    setSearchQuery('')
  }

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />

      <div className="fixed right-0 top-0 bottom-0 w-[400px] bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto">
        {/* 헤더 */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold flex items-center gap-2">
            ⭐ Watchlist
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              {isEditing ? '완료' : '편집'}
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
        <div className="p-6 space-y-4">
          {/* 요약 */}
          <div className="text-sm text-gray-400">
            📊 {watchlist.length}개 종목 • 알림: {watchlist.filter(w => w.alertEnabled).length}개 활성
          </div>

          {/* 종목 목록 */}
          <div className="space-y-2">
            {watchlistPredictions.map(({ ticker, alertEnabled, prediction }) => (
              <WatchlistItem
                key={ticker}
                ticker={ticker}
                prediction={prediction}
                alertEnabled={alertEnabled}
                isEditing={isEditing}
                onToggleAlert={() => toggleAlert(ticker)}
                onRemove={() => removeTicker(ticker)}
              />
            ))}

            {watchlist.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <div className="text-4xl mb-2">⭐</div>
                <p>Watchlist가 비어있습니다</p>
                <p className="text-sm">종목을 추가해보세요</p>
              </div>
            )}
          </div>

          {/* 검색 */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="🔍 종목 검색..."
              className="w-full px-4 py-2 bg-surface-light rounded-lg text-white placeholder-gray-500"
            />

            {/* 검색 결과 */}
            {searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-surface border border-surface-light rounded-lg shadow-xl z-10">
                {searchResults.map((pred) => (
                  <button
                    key={pred.ticker}
                    onClick={() => handleAddTicker(pred.ticker)}
                    className="w-full flex items-center justify-between p-3 hover:bg-surface-light text-left"
                  >
                    <div>
                      <span className="font-bold">{pred.ticker}</span>
                      <span className="text-gray-400 text-sm ml-2 truncate">
                        {pred.name}
                      </span>
                    </div>
                    <span className={pred.direction === 'up' ? 'text-green-400' : 'text-red-400'}>
                      {pred.probability}% {pred.direction === 'up' ? '↑' : '↓'}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function WatchlistItem({ ticker, prediction, alertEnabled, isEditing, onToggleAlert, onRemove }) {
  return (
    <div className="flex items-center justify-between p-3 bg-surface-light rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-yellow-400">⭐</span>
          <span className="font-bold">{ticker}</span>
          <span className={clsx(
            'text-sm',
            prediction?.direction === 'up' ? 'text-green-400' : 'text-red-400'
          )}>
            {prediction?.probability}% {prediction?.direction === 'up' ? '↑' : '↓'}
          </span>
          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
            getGradeStyle(prediction?.practicality_grade)
          }`}>
            {prediction?.practicality_grade}
          </span>
        </div>
        <div className="text-xs text-gray-400 mt-0.5">
          ${prediction?.current_price?.toFixed(2)}
          <span className={prediction?.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
            {' '}{prediction?.change_percent >= 0 ? '+' : ''}{prediction?.change_percent?.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onToggleAlert}
          className={alertEnabled ? 'text-blue-400' : 'text-gray-500'}
        >
          {alertEnabled ? '🔔' : '🔕'}
        </button>

        {isEditing && (
          <button
            onClick={onRemove}
            className="text-red-400 hover:text-red-300"
          >
            🗑️
          </button>
        )}
      </div>
    </div>
  )
}
```

### 6.4 notificationStore.js

```javascript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useNotificationStore = create(
  persist(
    (set, get) => ({
      // 권한 상태
      permission: 'default', // 'default' | 'granted' | 'denied'
      bannerDismissed: false,

      // 알림 설정
      settings: {
        enabled: true,
        signalUp: true,
        signalDown: true,
        discovery: true,
        watchlist: true,
        trainingComplete: false,
        probabilityThreshold: 80,
        grades: ['A', 'B'],
        quietHours: {
          enabled: false,
          start: '22:00',
          end: '08:00',
        },
      },

      // 알림 목록
      notifications: [],

      // Actions
      requestPermission: async () => {
        if (!('Notification' in window)) return false

        const permission = await Notification.requestPermission()
        set({ permission })
        return permission === 'granted'
      },

      dismissBanner: () => set({ bannerDismissed: true }),

      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),

      addNotification: (notification) => set((state) => ({
        notifications: [
          { ...notification, id: Date.now(), read: false, time: new Date() },
          ...state.notifications
        ].slice(0, 100) // 최대 100개 유지
      })),

      markAsRead: (id) => set((state) => ({
        notifications: state.notifications.map(n =>
          n.id === id ? { ...n, read: true } : n
        )
      })),

      markAllAsRead: () => set((state) => ({
        notifications: state.notifications.map(n => ({ ...n, read: true }))
      })),

      clearAll: () => set({ notifications: [] }),

      // 알림 보내기
      sendNotification: (title, options = {}) => {
        const state = get()
        if (state.permission !== 'granted' || !state.settings.enabled) return

        // 조용한 시간 체크
        if (state.settings.quietHours.enabled) {
          const now = new Date()
          const hour = now.getHours()
          const [startHour] = state.settings.quietHours.start.split(':').map(Number)
          const [endHour] = state.settings.quietHours.end.split(':').map(Number)

          if (hour >= startHour || hour < endHour) return
        }

        // 브라우저 알림
        new Notification(title, {
          icon: '/icons/icon-192.png',
          badge: '/icons/badge-72.png',
          ...options
        })

        // 인앱 알림 추가
        state.addNotification({
          type: options.type || 'default',
          title,
          message: options.body,
          ticker: options.ticker,
        })
      },

      getUnreadCount: () => {
        const state = get()
        return state.notifications.filter(n => !n.read).length
      },
    }),
    {
      name: 'nasdaq-predictor-notifications',
      version: 1,
    }
  )
)
```

### 6.5 watchlistStore.js

```javascript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useWatchlistStore = create(
  persist(
    (set, get) => ({
      watchlist: [],

      addTicker: (ticker) => set((state) => {
        if (state.watchlist.find(w => w.ticker === ticker)) return state
        return {
          watchlist: [...state.watchlist, { ticker, alertEnabled: true, addedAt: new Date() }]
        }
      }),

      removeTicker: (ticker) => set((state) => ({
        watchlist: state.watchlist.filter(w => w.ticker !== ticker)
      })),

      toggleAlert: (ticker) => set((state) => ({
        watchlist: state.watchlist.map(w =>
          w.ticker === ticker ? { ...w, alertEnabled: !w.alertEnabled } : w
        )
      })),

      isInWatchlist: (ticker) => {
        return get().watchlist.some(w => w.ticker === ticker)
      },

      reorder: (fromIndex, toIndex) => set((state) => {
        const newList = [...state.watchlist]
        const [removed] = newList.splice(fromIndex, 1)
        newList.splice(toIndex, 0, removed)
        return { watchlist: newList }
      }),
    }),
    {
      name: 'nasdaq-predictor-watchlist',
      version: 1,
    }
  )
)
```

---

## 7. 알림 트리거 통합

### 7.1 WebSocket 연동

```javascript
// useWebSocket.js에 알림 트리거 추가
useEffect(() => {
  if (lastMessage?.type === 'prediction_update') {
    const { ticker, probability, direction, practicality_grade } = lastMessage

    // 강한 신호 알림
    if (probability >= 80 && practicality_grade === 'A') {
      const type = direction === 'up' ? 'signal_up' : 'signal_down'
      sendNotification(`강한 ${direction === 'up' ? '상승' : '하락'} 신호!`, {
        body: `${ticker} ${probability}% ${direction === 'up' ? '↑' : '↓'} (등급 A)`,
        type,
        ticker,
      })
    }

    // Watchlist 종목 알림
    if (isInWatchlist(ticker) && getAlertEnabled(ticker)) {
      sendNotification(`Watchlist 신호`, {
        body: `${ticker} ${probability}% ${direction === 'up' ? '↑' : '↓'}`,
        type: 'watchlist',
        ticker,
      })
    }
  }
}, [lastMessage])
```

---

## 8. 구현 일정

### Phase 1: 알림 인프라 (1일)
- [ ] notificationStore.js 구현
- [ ] 권한 요청 플로우
- [ ] 브라우저 알림 발송

### Phase 2: 알림 센터 (1일)
- [ ] NotificationCenter.jsx 구현
- [ ] 알림 아이콘 배지
- [ ] 알림 설정 패널

### Phase 3: Watchlist (1일)
- [ ] watchlistStore.js 구현
- [ ] WatchlistPanel.jsx 구현
- [ ] 추가/삭제/알림 토글

### Phase 4: 통합 (0.5일)
- [ ] WebSocket 알림 트리거
- [ ] 전체 테스트

---

## 9. 테스트 체크리스트

| 항목 | 확인 |
|------|------|
| 알림 권한 요청 | [ ] |
| 브라우저 알림 수신 | [ ] |
| 알림 센터 열기/닫기 | [ ] |
| 알림 읽음 표시 | [ ] |
| Watchlist 추가/삭제 | [ ] |
| Watchlist 알림 토글 | [ ] |
| 조용한 시간 동작 | [ ] |
| 조건부 알림 필터 | [ ] |

---

*이 문서는 분석팀장이 작성하였습니다.*
