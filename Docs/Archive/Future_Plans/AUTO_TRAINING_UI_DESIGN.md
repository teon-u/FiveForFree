# 새 종목 자동 학습 트리거 UI 설계서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 개요

### 1.1 목적

신규 발견된 종목에 대해 **ML 모델 학습을 트리거**하고 **학습 상태를 실시간으로 표시**하는 UI

### 1.2 관련 API

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/train/{ticker}` | POST | 특정 종목 학습 시작 |
| `/api/train/batch` | POST | 여러 종목 일괄 학습 |
| `/api/train/status` | GET | 학습 상태 조회 |
| `/api/train/queue` | GET | 학습 대기열 조회 |

---

## 2. 학습 상태 정의

### 2.1 상태 종류

| 상태 | 코드 | 아이콘 | 색상 | 설명 |
|------|------|--------|------|------|
| 미학습 | `untrained` | ⚠️ | 노랑 | 학습되지 않음 |
| 대기중 | `queued` | 🕐 | 파랑 | 대기열에 있음 |
| 학습중 | `training` | 🔄 | 파랑 (애니메이션) | 현재 학습 진행 |
| 완료 | `trained` | ✅ | 녹색 | 학습 완료 |
| 오류 | `error` | ❌ | 빨강 | 학습 실패 |
| 재학습 필요 | `stale` | 🔄 | 주황 | 데이터 업데이트 필요 |

### 2.2 상태 전환 흐름

```
untrained → queued → training → trained
                 ↓            ↓
               error        stale
                 ↓            ↓
              (재시도)    → queued (재학습)
```

---

## 3. UI 컴포넌트 설계

### 3.1 학습 상태 배지 (TrainingStatusBadge)

```jsx
// 컴팩트 버전 - 카드/리스트에 표시
┌───────────────────┐
│ ⚠️ 미학습         │  → 노랑 배경
│ 🕐 대기중 #3      │  → 파랑 배경 + 순번
│ 🔄 학습중 45%     │  → 파랑 배경 + 프로그레스
│ ✅ 완료           │  → 녹색 배경
│ ❌ 오류           │  → 빨강 배경
└───────────────────┘
```

### 3.2 학습 트리거 버튼 (TrainButton)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 상태별 버튼 표시                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ 미학습:  [🎓 학습하기]                                              │
│ 대기중:  [🕐 대기중 #3] (비활성, 클릭 시 취소 가능)                 │
│ 학습중:  [████░░░░ 45%] (프로그레스 바)                             │
│ 완료:    [📈 예측보기]                                              │
│ 오류:    [🔄 재시도]                                                │
│ 오래됨:  [🔄 재학습]                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 학습 진행률 모달 (TrainingProgressModal)

학습 시작 시 또는 대기열 조회 시 표시

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🎓 모델 학습                                                  [✕]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 현재 학습중                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ PLTR - Palantir Technologies                                    ││
│ │                                                                 ││
│ │ ████████████████████░░░░░░░░░░░░░░░░░░  45%                    ││
│ │                                                                 ││
│ │ 단계: Feature Engineering (2/5)                                 ││
│ │ 경과: 1분 23초 / 예상: 2분 30초                                ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ 학습 단계                                                           │
│ ✅ 1. 데이터 수집          (완료)                                   │
│ 🔄 2. Feature Engineering  (진행중)                                │
│ ⏳ 3. 모델 학습 - XGBoost                                          │
│ ⏳ 4. 모델 학습 - LightGBM/LSTM/Transformer                        │
│ ⏳ 5. 성능 평가 및 저장                                            │
│                                                                     │
│ 대기열 (2개)                                                        │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ 2. RIVN - Rivian Automotive       예상 2분 [취소]               ││
│ │ 3. LCID - Lucid Group            예상 4분 [취소]               ││
│ └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│                           [백그라운드로 전환]                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 미니 학습 상태 표시 (TrainingMiniStatus)

헤더 또는 대시보드 구석에 항상 표시

```
┌───────────────────────────────────────┐
│ 🎓 PLTR 학습중 45%  │ 대기: 2개      │
└───────────────────────────────────────┘
```

---

## 4. 컴포넌트 구현

### 4.1 TrainingStatusBadge.jsx

```jsx
export default function TrainingStatusBadge({ status, progress, queuePosition }) {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'untrained':
        return { icon: '⚠️', text: '미학습', bg: 'bg-yellow-500/20', textColor: 'text-yellow-400' }
      case 'queued':
        return { icon: '🕐', text: `대기중 #${queuePosition}`, bg: 'bg-blue-500/20', textColor: 'text-blue-400' }
      case 'training':
        return { icon: '🔄', text: `학습중 ${progress}%`, bg: 'bg-blue-500/20', textColor: 'text-blue-400', animate: true }
      case 'trained':
        return { icon: '✅', text: '완료', bg: 'bg-green-500/20', textColor: 'text-green-400' }
      case 'error':
        return { icon: '❌', text: '오류', bg: 'bg-red-500/20', textColor: 'text-red-400' }
      case 'stale':
        return { icon: '🔄', text: '재학습 필요', bg: 'bg-orange-500/20', textColor: 'text-orange-400' }
      default:
        return { icon: '❓', text: '알 수 없음', bg: 'bg-gray-500/20', textColor: 'text-gray-400' }
    }
  }

  const config = getStatusConfig(status)

  return (
    <span className={clsx(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
      config.bg, config.textColor,
      config.animate && 'animate-pulse'
    )}>
      <span>{config.icon}</span>
      <span>{config.text}</span>
    </span>
  )
}
```

### 4.2 TrainButton.jsx

```jsx
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { endpoints } from '../services/api'

export default function TrainButton({ ticker, status, progress, queuePosition, onStatusChange }) {
  const queryClient = useQueryClient()
  const [isHovering, setIsHovering] = useState(false)

  const trainMutation = useMutation({
    mutationFn: (ticker) => endpoints.trainTicker(ticker),
    onSuccess: () => {
      queryClient.invalidateQueries(['training-status'])
      onStatusChange?.('queued')
    },
    onError: (error) => {
      console.error('Training failed:', error)
      onStatusChange?.('error')
    }
  })

  const cancelMutation = useMutation({
    mutationFn: (ticker) => endpoints.cancelTraining(ticker),
    onSuccess: () => {
      queryClient.invalidateQueries(['training-status'])
      onStatusChange?.('untrained')
    }
  })

  const handleClick = () => {
    switch (status) {
      case 'untrained':
      case 'error':
      case 'stale':
        trainMutation.mutate(ticker)
        break
      case 'queued':
        if (isHovering) {
          cancelMutation.mutate(ticker)
        }
        break
      case 'trained':
        // 예측 상세 보기로 이동
        window.location.href = `#/predictions/${ticker}`
        break
    }
  }

  const renderButton = () => {
    switch (status) {
      case 'untrained':
        return (
          <button
            onClick={handleClick}
            className="px-3 py-1.5 bg-green-500 text-white rounded-lg text-sm font-medium hover:bg-green-600 transition-colors"
          >
            🎓 학습하기
          </button>
        )

      case 'queued':
        return (
          <button
            onClick={handleClick}
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              isHovering
                ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                : 'bg-blue-500/20 text-blue-400'
            )}
          >
            {isHovering ? '❌ 취소' : `🕐 대기중 #${queuePosition}`}
          </button>
        )

      case 'training':
        return (
          <div className="px-3 py-1.5 bg-blue-500/20 rounded-lg">
            <div className="flex items-center gap-2 text-sm text-blue-400">
              <div className="w-16 h-1.5 bg-blue-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span>{progress}%</span>
            </div>
          </div>
        )

      case 'trained':
        return (
          <button
            onClick={handleClick}
            className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg text-sm font-medium hover:bg-blue-500/30 transition-colors"
          >
            📈 예측보기
          </button>
        )

      case 'error':
        return (
          <button
            onClick={handleClick}
            className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors border border-red-500/50"
          >
            🔄 재시도
          </button>
        )

      case 'stale':
        return (
          <button
            onClick={handleClick}
            className="px-3 py-1.5 bg-orange-500/20 text-orange-400 rounded-lg text-sm font-medium hover:bg-orange-500/30 transition-colors"
          >
            🔄 재학습
          </button>
        )
    }
  }

  return renderButton()
}
```

### 4.3 TrainingProgressModal.jsx

```jsx
import { useEffect, useState } from 'react'
import { useTrainingStatus } from '../hooks/useTrainingStatus'

const TRAINING_STEPS = [
  { id: 'data', name: '데이터 수집' },
  { id: 'features', name: 'Feature Engineering' },
  { id: 'xgboost', name: '모델 학습 - XGBoost' },
  { id: 'others', name: '모델 학습 - LightGBM/LSTM/Transformer' },
  { id: 'evaluate', name: '성능 평가 및 저장' },
]

export default function TrainingProgressModal({ ticker, onClose, onMinimize }) {
  const { status, isLoading } = useTrainingStatus(ticker, {
    refetchInterval: 1000, // 1초마다 상태 갱신
  })

  const currentStepIndex = TRAINING_STEPS.findIndex(s => s.id === status?.current_step)
  const overallProgress = status?.progress || 0

  return (
    <>
      <div className="modal-backdrop" onClick={onMinimize} />

      <div className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-lg mx-auto bg-surface rounded-2xl shadow-2xl z-50">
        {/* 헤더 */}
        <div className="flex items-center justify-between p-4 border-b border-surface-light">
          <h2 className="text-lg font-bold flex items-center gap-2">
            🎓 모델 학습
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* 콘텐츠 */}
        <div className="p-6 space-y-6">
          {/* 현재 학습중 */}
          <div className="bg-surface-light rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="font-bold">{ticker}</span>
              <span className="text-sm text-gray-400">{status?.ticker_name}</span>
            </div>

            {/* 프로그레스 바 */}
            <div className="w-full h-3 bg-surface rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-blue-500 transition-all duration-500"
                style={{ width: `${overallProgress}%` }}
              />
            </div>

            <div className="flex justify-between text-sm">
              <span className="text-gray-400">
                단계: {TRAINING_STEPS[currentStepIndex]?.name} ({currentStepIndex + 1}/{TRAINING_STEPS.length})
              </span>
              <span className="text-blue-400 font-medium">{overallProgress}%</span>
            </div>

            {status?.elapsed_time && (
              <div className="text-xs text-gray-500 mt-2">
                경과: {formatTime(status.elapsed_time)} / 예상: {formatTime(status.estimated_time)}
              </div>
            )}
          </div>

          {/* 학습 단계 */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-400">학습 단계</h3>
            {TRAINING_STEPS.map((step, index) => (
              <div
                key={step.id}
                className={clsx(
                  'flex items-center gap-2 text-sm py-1',
                  index < currentStepIndex && 'text-green-400',
                  index === currentStepIndex && 'text-blue-400',
                  index > currentStepIndex && 'text-gray-500'
                )}
              >
                <span>
                  {index < currentStepIndex && '✅'}
                  {index === currentStepIndex && '🔄'}
                  {index > currentStepIndex && '⏳'}
                </span>
                <span>{index + 1}. {step.name}</span>
                {index === currentStepIndex && (
                  <span className="text-gray-400 ml-auto">
                    {status?.step_progress}%
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* 대기열 */}
          {status?.queue?.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">
                대기열 ({status.queue.length}개)
              </h3>
              <div className="bg-surface-light rounded-lg divide-y divide-surface">
                {status.queue.map((item, index) => (
                  <div
                    key={item.ticker}
                    className="flex items-center justify-between p-3 text-sm"
                  >
                    <span>
                      {index + 2}. <span className="font-medium">{item.ticker}</span>
                      <span className="text-gray-400 ml-2">{item.name}</span>
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">예상 {formatTime(item.estimated_time)}</span>
                      <button className="text-red-400 hover:text-red-300 text-xs">
                        취소
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 푸터 */}
        <div className="p-4 border-t border-surface-light">
          <button
            onClick={onMinimize}
            className="w-full py-2 bg-surface-light text-gray-400 rounded-lg hover:bg-slate-600 transition-colors"
          >
            백그라운드로 전환
          </button>
        </div>
      </div>
    </>
  )
}

function formatTime(seconds) {
  if (!seconds) return '-'
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return mins > 0 ? `${mins}분 ${secs}초` : `${secs}초`
}
```

### 4.4 TrainingMiniStatus.jsx

```jsx
// 헤더에 표시되는 미니 상태
export default function TrainingMiniStatus() {
  const { status } = useTrainingStatus()

  if (!status?.current_ticker) return null

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/20 rounded-lg text-sm">
      <span className="animate-pulse">🎓</span>
      <span className="text-blue-400 font-medium">
        {status.current_ticker} 학습중
      </span>
      <span className="text-white">{status.progress}%</span>
      {status.queue_count > 0 && (
        <span className="text-gray-400">
          │ 대기: {status.queue_count}개
        </span>
      )}
    </div>
  )
}
```

---

## 5. API 요구사항

### 5.1 POST /api/train/{ticker}

**Response:**
```json
{
  "ticker": "PLTR",
  "status": "queued",
  "queue_position": 3,
  "estimated_time": 180,
  "message": "학습 대기열에 추가되었습니다"
}
```

### 5.2 GET /api/train/status

**Response:**
```json
{
  "current": {
    "ticker": "PLTR",
    "ticker_name": "Palantir Technologies",
    "status": "training",
    "current_step": "features",
    "step_progress": 65,
    "progress": 45,
    "elapsed_time": 83,
    "estimated_time": 150
  },
  "queue": [
    {
      "ticker": "RIVN",
      "name": "Rivian Automotive",
      "position": 2,
      "estimated_time": 120
    }
  ],
  "queue_count": 2
}
```

### 5.3 DELETE /api/train/{ticker}

학습 취소 (대기열에서 제거)

---

## 6. 자동 학습 설정

### 6.1 설정 UI

```
┌─────────────────────────────────────────────────────────────────────┐
│ 자동 학습 설정                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ☑ 새 급등주 발견 시 자동 학습                                       │
│   └ 조건: 변동률 __10__% 이상                                       │
│                                                                     │
│ ☑ 새 거래량 상위 종목 자동 학습                                     │
│   └ 조건: 상위 __50__위 이내                                        │
│                                                                     │
│ ☐ 야간 일괄 학습 (장 마감 후)                                       │
│   └ 시간: __20:00__ 시작                                            │
│                                                                     │
│ 최대 동시 학습: __3__개                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. 구현 일정

### Phase 1: 기본 컴포넌트 (1.5일)
- [ ] TrainingStatusBadge.jsx
- [ ] TrainButton.jsx
- [ ] useTrainingStatus.js 훅

### Phase 2: 진행률 모달 (1일)
- [ ] TrainingProgressModal.jsx
- [ ] 실시간 상태 폴링

### Phase 3: 미니 상태 및 설정 (0.5일)
- [ ] TrainingMiniStatus.jsx
- [ ] 자동 학습 설정 UI

---

## 8. 테스트 체크리스트

| 항목 | 확인 |
|------|------|
| 학습 버튼 클릭 | [ ] |
| 대기열 추가 확인 | [ ] |
| 진행률 실시간 업데이트 | [ ] |
| 학습 완료 상태 변경 | [ ] |
| 학습 취소 기능 | [ ] |
| 에러 상태 및 재시도 | [ ] |
| 미니 상태 표시 | [ ] |

---

*이 문서는 분석팀장이 작성하였습니다.*
