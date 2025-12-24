# 모바일 최적화 전략 수립

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 현재 상태 분석

### 1.1 현재 반응형 구현

```javascript
// TickerGrid.jsx
<div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
```

**현재 브레이크포인트:**
- `xs`: 2열 (< 640px)
- `sm`: 3열 (640px+)
- `md`: 4열 (768px+)
- `lg`: 5열 (1024px+)
- `xl`: 6열 (1280px+)

### 1.2 현재 문제점

| 문제 | 영향 | 심각도 |
|------|------|--------|
| 터치 타겟 작음 | 버튼 오작동 빈번 | 높음 |
| 카드 정보 과밀 | 가독성 저하 | 중간 |
| 스크롤 지옥 | 탐색 피로 | 높음 |
| 필터 접근 어려움 | 기능 활용 저조 | 중간 |
| 차트 조작 불편 | 데이터 분석 어려움 | 높음 |

---

## 2. 모바일 최적화 목표

### 2.1 핵심 원칙

1. **Thumb Zone 최적화**: 엄지로 쉽게 닿는 영역에 주요 기능 배치
2. **정보 우선순위화**: 모바일에서 가장 중요한 정보만 우선 표시
3. **제스처 친화적**: 스와이프, 탭, 롱프레스 등 네이티브 제스처 활용
4. **성능 최우선**: 3G 환경에서도 3초 내 로딩

### 2.2 목표 지표

| 지표 | 현재 | 목표 |
|------|------|------|
| LCP (Largest Contentful Paint) | 3.5s | < 2.5s |
| FID (First Input Delay) | 150ms | < 100ms |
| CLS (Cumulative Layout Shift) | 0.15 | < 0.1 |
| 터치 타겟 크기 | 36px | 48px+ |
| 모바일 이탈률 | 65% | < 40% |

---

## 3. 터치 제스처 설계

### 3.1 제스처 매핑

| 제스처 | 대상 | 동작 |
|--------|------|------|
| 탭 | 티커 카드 | 상세 패널 열기 |
| 더블 탭 | 티커 카드 | 빠른 관심목록 추가 |
| 롱 프레스 | 티커 카드 | 컨텍스트 메뉴 표시 |
| 스와이프 좌 | 티커 카드 | 삭제/숨기기 |
| 스와이프 우 | 티커 카드 | 관심목록 추가 |
| 스와이프 하 | 리스트 상단 | 새로고침 (Pull to Refresh) |
| 핀치 | 차트 영역 | 줌 인/아웃 |
| 팬 | 차트 영역 | 기간 이동 |

### 3.2 제스처 UI 가이드

```
┌─────────────────────────────────────┐
│                                     │
│  ← 스와이프: 숨기기                 │
│                    스와이프: 저장 → │
│                                     │
│    ┌───────────────────────┐        │
│    │                       │        │
│    │      티커 카드        │        │
│    │                       │        │
│    │   탭: 상세보기        │        │
│    │   롱프레스: 메뉴      │        │
│    │                       │        │
│    └───────────────────────┘        │
│                                     │
└─────────────────────────────────────┘
```

### 3.3 제스처 피드백

| 제스처 | 시각적 피드백 | 촉각 피드백 |
|--------|--------------|------------|
| 탭 | 리플 이펙트 | 약한 진동 |
| 롱 프레스 | 확대 효과 | 중간 진동 |
| 스와이프 | 배경색 변경 | 짧은 진동 |
| 성공 | 체크 애니메이션 | 성공 패턴 진동 |

```jsx
// 제스처 피드백 예시
const handleLongPress = () => {
  // 햅틱 피드백 (iOS/Android)
  if ('vibrate' in navigator) {
    navigator.vibrate(50);
  }
  showContextMenu();
}
```

---

## 4. 컴팩트 뷰 설계

### 4.1 일반 뷰 vs 컴팩트 뷰

**일반 뷰 (현재)**
```
┌─────────────────────┐
│ NVDA          🟢    │
│                     │
│    82% ↑            │
│                     │
│ $142.50  +5.2%      │
│                     │
│ Model: XGB    [A]   │
│ Precision: 68%      │
│ Signal: 15%         │
│                     │
│ [📈차트] [🔍모델]   │
└─────────────────────┘
```

**컴팩트 뷰 (신규)**
```
┌────────────────────────────────────┐
│ 🟢 NVDA   82%↑   $142.50 +5.2%  A │
└────────────────────────────────────┘
```

### 4.2 컴팩트 뷰 전환

```jsx
// ViewToggle.jsx
┌──────────────────────────┐
│  [█ 그리드]  [≡ 리스트]  │
└──────────────────────────┘
```

### 4.3 컴팩트 카드 컴포넌트

```jsx
// CompactTickerCard.jsx

export default function CompactTickerCard({ prediction, onPress, onLongPress }) {
  const {
    ticker,
    probability,
    direction,
    change_percent,
    practicality_grade,
    current_price
  } = prediction

  return (
    <Pressable
      onPress={onPress}
      onLongPress={onLongPress}
      className="flex items-center justify-between p-3 bg-surface-light rounded-lg"
    >
      {/* 방향 아이콘 */}
      <span className="text-lg mr-2">
        {direction === 'up' ? '🟢' : '🔴'}
      </span>

      {/* 티커 */}
      <span className="font-bold w-16">{ticker}</span>

      {/* 확률 */}
      <span className={`font-semibold w-16 ${
        direction === 'up' ? 'text-green-400' : 'text-red-400'
      }`}>
        {probability}%{direction === 'up' ? '↑' : '↓'}
      </span>

      {/* 가격 정보 */}
      <div className="flex-1 text-right">
        <span className="text-gray-300">${current_price?.toFixed(2)}</span>
        <span className={`ml-2 ${
          change_percent >= 0 ? 'text-green-400' : 'text-red-400'
        }`}>
          {change_percent >= 0 ? '+' : ''}{change_percent?.toFixed(1)}%
        </span>
      </div>

      {/* 등급 */}
      <span className={`ml-2 px-2 py-0.5 rounded text-xs font-bold ${
        getGradeStyle(practicality_grade)
      }`}>
        {practicality_grade}
      </span>

      {/* 펼침 화살표 */}
      <span className="ml-2 text-gray-500">›</span>
    </Pressable>
  )
}
```

### 4.4 정보 계층 구조

**컴팩트 뷰 표시 순서:**

| 우선순위 | 항목 | 표시 여부 |
|----------|------|-----------|
| 1 | 방향 아이콘 | 항상 |
| 2 | 티커 심볼 | 항상 |
| 3 | 확률 | 항상 |
| 4 | 등급 | 항상 |
| 5 | 가격 | 너비 > 360px |
| 6 | 변동률 | 너비 > 360px |

---

## 5. 네비게이션 최적화

### 5.1 Bottom Tab Navigation

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│            (메인 콘텐츠)                │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│   📊        📈        ⭐        ⚙️     │
│  대시보드   차트     관심목록   설정    │
└─────────────────────────────────────────┘
```

### 5.2 탭 아이템 상세

| 탭 | 아이콘 | 기능 |
|----|--------|------|
| 대시보드 | 📊 | 메인 대시보드 (기본) |
| 차트 | 📈 | 선택된 티커 차트 |
| 관심목록 | ⭐ | 저장된 티커 목록 |
| 설정 | ⚙️ | 필터, 언어, 알림 설정 |

### 5.3 FAB (Floating Action Button)

```
                    ┌─────┐
                    │ ▲   │
                    │ TOP │
                    └─────┘

 ┌─────────────────────────┐
 │                         │
 └─────────────────────────┘
```

- 스크롤 시 화면 우하단에 표시
- 탭 시 목록 최상단으로 스크롤

---

## 6. 풀스크린 모달 패턴

### 6.1 필터 모달

```
┌─────────────────────────────────────┐
│ ← 필터                     [초기화] │
├─────────────────────────────────────┤
│                                     │
│ 방향                                │
│ ┌─────────┐ ┌─────────┐             │
│ │ 📈 상승 │ │ 📉 하락 │             │
│ └─────────┘ └─────────┘             │
│                                     │
│ 섹터                           [▼]  │
│ ┌───────────────────────────────┐   │
│ │ Technology, Healthcare        │   │
│ └───────────────────────────────┘   │
│                                     │
│ 확률 범위                           │
│ ○ 전체  ● 70%+  ○ 80%+  ○ 90%+     │
│                                     │
│ 정렬                                │
│ ● 확률순  ○ Precision순  ○ 등급순  │
│                                     │
├─────────────────────────────────────┤
│        [결과 보기 (23개)]           │
└─────────────────────────────────────┘
```

### 6.2 상세 패널 (Bottom Sheet)

```
┌─────────────────────────────────────┐
│                                     │
│        (흐릿한 대시보드 배경)        │
│                                     │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│ ────────────                        │  ← 드래그 핸들
│                                     │
│ NVDA - NVIDIA Corporation           │
│ $142.50  +5.2%                      │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │          예측 확률              │ │
│ │       [====82%====]             │ │
│ │          상승 ↑                 │ │
│ └─────────────────────────────────┘ │
│                                     │
│ 모델 성능                           │
│ ├─ XGBoost   [A] Precision: 68%    │
│ ├─ LightGBM  [B] Precision: 55%    │
│ ├─ LSTM      [C] Precision: 42%    │
│ └─ Trans.    [B] Precision: 58%    │
│                                     │
│ [📈 차트 보기]  [⭐ 관심목록 추가]  │
│                                     │
└─────────────────────────────────────┘
```

**Bottom Sheet 동작:**
- 드래그로 높이 조절 (30%/60%/100%)
- 하단 스와이프로 닫기
- 배경 탭해도 닫기

---

## 7. 성능 최적화

### 7.1 이미지 최적화

```jsx
// next/image 또는 lazy loading 사용
<img
  src={iconUrl}
  loading="lazy"
  decoding="async"
  width={24}
  height={24}
/>
```

### 7.2 가상화 스크롤

100개 이상 티커 렌더링 시 react-virtualized 사용

```jsx
import { FixedSizeList } from 'react-window'

<FixedSizeList
  height={windowHeight}
  itemCount={predictions.length}
  itemSize={60} // 컴팩트 카드 높이
  width="100%"
>
  {({ index, style }) => (
    <div style={style}>
      <CompactTickerCard prediction={predictions[index]} />
    </div>
  )}
</FixedSizeList>
```

### 7.3 데이터 페칭 최적화

```jsx
// 무한 스크롤 구현
const { data, fetchNextPage, hasNextPage } = useInfiniteQuery({
  queryKey: ['predictions'],
  queryFn: ({ pageParam = 0 }) =>
    fetchPredictions({ offset: pageParam, limit: 20 }),
  getNextPageParam: (lastPage, pages) =>
    lastPage.hasMore ? pages.length * 20 : undefined,
})
```

### 7.4 번들 최적화

```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          i18n: ['./src/i18n'],
        }
      }
    }
  }
}
```

---

## 8. PWA 지원

### 8.1 manifest.json

```json
{
  "name": "FiveForFree - NASDAQ Predictor",
  "short_name": "FiveForFree",
  "description": "NASDAQ 주식 예측 대시보드",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1b26",
  "theme_color": "#3b82f6",
  "orientation": "portrait",
  "icons": [
    {
      "src": "/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### 8.2 Service Worker

```javascript
// 오프라인 캐싱 전략
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/api/')) {
    // API: Network First
    event.respondWith(networkFirst(event.request))
  } else {
    // Static: Cache First
    event.respondWith(cacheFirst(event.request))
  }
})
```

### 8.3 푸시 알림 (향후)

```javascript
// 알림 권한 요청
Notification.requestPermission().then((permission) => {
  if (permission === 'granted') {
    // 구독 설정
    subscribeToPush()
  }
})
```

---

## 9. 접근성 (A11y)

### 9.1 터치 타겟 크기

```css
/* 최소 48x48px */
.touch-target {
  min-width: 48px;
  min-height: 48px;
  padding: 12px;
}
```

### 9.2 색상 대비

| 요소 | 전경색 | 배경색 | 대비율 |
|------|--------|--------|--------|
| 일반 텍스트 | #e5e7eb | #1a1b26 | 12.5:1 |
| 상승 표시 | #34d399 | #1a1b26 | 8.7:1 |
| 하락 표시 | #f87171 | #1a1b26 | 5.2:1 |

### 9.3 스크린 리더

```jsx
<button
  aria-label={`${ticker} 상세 정보 보기. 현재 확률 ${probability}% ${direction === 'up' ? '상승' : '하락'} 예측`}
  role="button"
>
  ...
</button>
```

---

## 10. 구현 일정

### Phase 1: 터치 최적화 (3일)
- [ ] 터치 타겟 크기 48px 이상으로 조정
- [ ] 제스처 라이브러리 도입 (react-use-gesture)
- [ ] 햅틱 피드백 구현

### Phase 2: 컴팩트 뷰 (4일)
- [ ] CompactTickerCard.jsx 구현
- [ ] 뷰 전환 토글 구현
- [ ] 설정 저장 연동

### Phase 3: 네비게이션 (3일)
- [ ] Bottom Tab Navigation 구현
- [ ] FAB (맨위로) 버튼
- [ ] 풀스크린 필터 모달

### Phase 4: 성능 최적화 (3일)
- [ ] react-window 가상화 스크롤
- [ ] 번들 스플리팅
- [ ] 이미지 lazy loading

### Phase 5: PWA (2일)
- [ ] manifest.json 설정
- [ ] Service Worker 캐싱
- [ ] 설치 프롬프트

---

## 11. 테스트 체크리스트

### 11.1 디바이스 테스트

| 디바이스 | 해상도 | 테스트 항목 |
|----------|--------|-------------|
| iPhone SE | 375x667 | 최소 너비 테스트 |
| iPhone 14 | 390x844 | 표준 iOS 테스트 |
| Galaxy S21 | 360x800 | 표준 Android 테스트 |
| iPad Mini | 768x1024 | 태블릿 테스트 |

### 11.2 기능 테스트

- [ ] 모든 터치 타겟 48px 이상
- [ ] 스와이프 제스처 작동
- [ ] 풀스크린 모달 열기/닫기
- [ ] 뷰 전환 정상 작동
- [ ] 오프라인 시 캐시 데이터 표시
- [ ] 푸시 알림 수신 (향후)

---

*이 문서는 분석팀장이 작성하였습니다.*
