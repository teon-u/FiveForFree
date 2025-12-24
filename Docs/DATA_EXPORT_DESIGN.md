# 데이터 내보내기 기획서

**작성일**: 2025-12-21
**작성자**: 분석팀장
**버전**: v1.0

---

## 1. 개요

### 1.1 목적

사용자가 예측 데이터를 **CSV/Excel 형식으로 다운로드**하여 외부 분석 도구에서 활용할 수 있도록 지원

### 1.2 사용 시나리오

| 시나리오 | 설명 |
|----------|------|
| 백테스팅 | 다운로드한 예측을 자체 검증 시스템에 활용 |
| 포트폴리오 분석 | Excel에서 예측 기반 투자 전략 시뮬레이션 |
| 기록 보관 | 일일 예측 결과 아카이빙 |
| 리포트 작성 | 투자 의사결정 근거 문서화 |

---

## 2. 내보내기 옵션 설계

### 2.1 파일 형식

| 형식 | 확장자 | 용도 | 장점 |
|------|--------|------|------|
| CSV | .csv | 범용 | 가벼움, 모든 도구 호환 |
| Excel | .xlsx | 분석용 | 서식, 수식, 시트 지원 |
| JSON | .json | 개발자용 | 프로그래밍 연동 |

### 2.2 데이터 범위

| 범위 | 설명 |
|------|------|
| 현재 화면 | 필터/정렬 적용된 결과 |
| 전체 데이터 | 모든 예측 (필터 무시) |
| 선택된 티커 | 체크박스 선택 항목 |
| 관심목록 | 저장된 관심 티커 |

---

## 3. UI/UX 설계

### 3.1 내보내기 버튼 위치

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📊 오늘의 요약                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 필터: [상승] [하락] | [섹터▼] | [확률▼] | [정렬▼]                  │
│                                                     [📥 내보내기]   │  ← 여기
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 내보내기 모달

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📥 데이터 내보내기                                            [✕]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 파일 형식                                                           │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐                              │
│ │ ● CSV   │  │ ○ Excel │  │ ○ JSON  │                              │
│ └─────────┘  └─────────┘  └─────────┘                              │
│                                                                     │
│ 데이터 범위                                                         │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ ● 현재 화면 (23개) - 필터/정렬 적용된 결과                     │ │
│ │ ○ 전체 데이터 (150개) - 모든 예측                              │ │
│ │ ○ 관심목록 (8개) - 저장된 티커                                 │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ 포함 항목                                                           │
│ ☑ 기본 정보 (티커, 가격, 변동률)                                   │
│ ☑ 예측 정보 (확률, 방향, 등급)                                     │
│ ☑ 모델 성능 (Precision, Signal Rate)                               │
│ ☐ 상세 모델별 결과 (XGB, LGBM, LSTM, Transformer)                 │
│ ☐ 가격 히스토리 (60분)                                             │
│                                                                     │
│ 파일명                                                              │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ fiveforfree_predictions_2025-12-21                             │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│                    [취소]        [📥 다운로드]                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 진행률 표시

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📥 다운로드 준비 중...                                              │
│                                                                     │
│ ████████████████████████░░░░░░░░░░░░░░░░  65%                      │
│                                                                     │
│ 150개 티커 중 98개 처리                                            │
│ 예상 시간: 약 5초                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 데이터 컬럼 정의

### 4.1 기본 컬럼

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| ticker | string | 티커 심볼 | NVDA |
| name | string | 회사명 | NVIDIA Corporation |
| current_price | number | 현재가 | 142.50 |
| change_percent | number | 변동률 (%) | 5.20 |
| volume | number | 거래량 | 45200000 |

### 4.2 예측 컬럼

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| prediction_direction | string | 예측 방향 | UP / DOWN |
| probability | number | 예측 확률 (%) | 82.5 |
| practicality_grade | string | 실용성 등급 | A / B / C / D |
| best_model | string | 최적 모델 | xgboost |
| trading_signal | string | 매매 신호 | BUY / SELL / HOLD |

### 4.3 모델 성능 컬럼

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| precision | number | 적중률 (%) | 68.0 |
| signal_rate | number | 신호 발생률 (%) | 15.0 |
| predictions_count | number | 예측 횟수 | 48 |

### 4.4 상세 모델 컬럼 (선택)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| xgb_probability | number | XGBoost 확률 |
| xgb_precision | number | XGBoost 적중률 |
| lgbm_probability | number | LightGBM 확률 |
| lgbm_precision | number | LightGBM 적중률 |
| lstm_probability | number | LSTM 확률 |
| lstm_precision | number | LSTM 적중률 |
| transformer_probability | number | Transformer 확률 |
| transformer_precision | number | Transformer 적중률 |

### 4.5 메타데이터

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| export_timestamp | datetime | 내보내기 시간 |
| data_timestamp | datetime | 데이터 기준 시간 |
| filters_applied | string | 적용된 필터 |

---

## 5. Excel 시트 구조

### 5.1 다중 시트 구성

| 시트명 | 내용 |
|--------|------|
| Summary | 요약 통계 |
| Predictions | 전체 예측 데이터 |
| Models | 모델별 성능 상세 |
| Metadata | 내보내기 정보 |

### 5.2 Summary 시트

```
┌───────────────────────────────────────────┐
│ FiveForFree 예측 리포트                    │
│ 생성일: 2025-12-21 14:30:00               │
├───────────────────────────────────────────┤
│ 총 티커 수:        150                    │
│ 상승 신호:         23 (32%)               │
│ 하락 신호:         12 (17%)               │
│ A등급:             8                      │
│ B등급:             15                     │
│ 평균 확률:         74.5%                  │
│ 평균 Precision:    52.3%                  │
└───────────────────────────────────────────┘
```

### 5.3 조건부 서식 (Excel)

| 규칙 | 조건 | 서식 |
|------|------|------|
| 등급 A | practicality_grade = "A" | 녹색 배경 |
| 등급 D | practicality_grade = "D" | 빨간 배경 |
| 상승 | direction = "UP" | 녹색 텍스트 |
| 하락 | direction = "DOWN" | 빨간 텍스트 |
| 고확률 | probability >= 80 | 굵은 텍스트 |

---

## 6. 컴포넌트 구현

### 6.1 ExportModal.jsx

```jsx
import { useState } from 'react'
import { useSettingsStore } from '../stores/settingsStore'
import { exportToCSV, exportToExcel, exportToJSON } from '../utils/exportUtils'

const FORMATS = ['csv', 'xlsx', 'json']
const RANGES = ['current', 'all', 'watchlist']

export default function ExportModal({ predictions, onClose }) {
  const { language } = useSettingsStore()
  const [format, setFormat] = useState('csv')
  const [range, setRange] = useState('current')
  const [includeBasic, setIncludeBasic] = useState(true)
  const [includePrediction, setIncludePrediction] = useState(true)
  const [includeModel, setIncludeModel] = useState(true)
  const [includeDetailedModels, setIncludeDetailedModels] = useState(false)
  const [includeHistory, setIncludeHistory] = useState(false)
  const [filename, setFilename] = useState(
    `fiveforfree_predictions_${new Date().toISOString().split('T')[0]}`
  )
  const [isExporting, setIsExporting] = useState(false)
  const [progress, setProgress] = useState(0)

  const handleExport = async () => {
    setIsExporting(true)
    setProgress(0)

    try {
      // 데이터 준비
      const data = await prepareExportData(predictions, {
        range,
        includeBasic,
        includePrediction,
        includeModel,
        includeDetailedModels,
        includeHistory,
        onProgress: setProgress,
      })

      // 형식에 따라 내보내기
      switch (format) {
        case 'csv':
          exportToCSV(data, filename)
          break
        case 'xlsx':
          await exportToExcel(data, filename)
          break
        case 'json':
          exportToJSON(data, filename)
          break
      }

      onClose()
    } catch (error) {
      console.error('Export failed:', error)
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />

      <div className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-lg mx-auto bg-surface rounded-2xl shadow-2xl z-50 p-6">
        {isExporting ? (
          // 진행률 표시
          <div className="text-center py-8">
            <div className="text-lg font-bold mb-4">다운로드 준비 중...</div>
            <div className="w-full bg-surface-light rounded-full h-2 mb-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-sm text-gray-400">{progress}%</div>
          </div>
        ) : (
          <>
            {/* 헤더 */}
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">📥 데이터 내보내기</h2>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            {/* 파일 형식 */}
            <div className="mb-6">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                파일 형식
              </div>
              <div className="flex gap-2">
                {FORMATS.map((f) => (
                  <button
                    key={f}
                    onClick={() => setFormat(f)}
                    className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                      format === f
                        ? 'bg-blue-500 text-white'
                        : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                    }`}
                  >
                    {f.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* 데이터 범위 */}
            <div className="mb-6">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                데이터 범위
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="range"
                    checked={range === 'current'}
                    onChange={() => setRange('current')}
                    className="accent-blue-500"
                  />
                  <span>현재 화면 ({predictions?.length || 0}개)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="range"
                    checked={range === 'all'}
                    onChange={() => setRange('all')}
                    className="accent-blue-500"
                  />
                  <span>전체 데이터</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="range"
                    checked={range === 'watchlist'}
                    onChange={() => setRange('watchlist')}
                    className="accent-blue-500"
                  />
                  <span>관심목록</span>
                </label>
              </div>
            </div>

            {/* 포함 항목 */}
            <div className="mb-6">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                포함 항목
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeBasic}
                    onChange={(e) => setIncludeBasic(e.target.checked)}
                    className="accent-blue-500"
                  />
                  <span>기본 정보 (티커, 가격, 변동률)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includePrediction}
                    onChange={(e) => setIncludePrediction(e.target.checked)}
                    className="accent-blue-500"
                  />
                  <span>예측 정보 (확률, 방향, 등급)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeModel}
                    onChange={(e) => setIncludeModel(e.target.checked)}
                    className="accent-blue-500"
                  />
                  <span>모델 성능 (Precision, Signal Rate)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-gray-500">
                  <input
                    type="checkbox"
                    checked={includeDetailedModels}
                    onChange={(e) => setIncludeDetailedModels(e.target.checked)}
                    className="accent-blue-500"
                  />
                  <span>상세 모델별 결과</span>
                </label>
              </div>
            </div>

            {/* 파일명 */}
            <div className="mb-6">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                파일명
              </div>
              <input
                type="text"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                className="w-full px-4 py-2 bg-surface-light rounded-lg text-white"
              />
            </div>

            {/* 버튼 */}
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-surface-light text-gray-400 rounded-lg hover:bg-slate-600"
              >
                취소
              </button>
              <button
                onClick={handleExport}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium"
              >
                📥 다운로드
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}
```

### 6.2 exportUtils.js

```javascript
// CSV 내보내기
export function exportToCSV(data, filename) {
  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map(row =>
      headers.map(h => {
        const val = row[h]
        // 쉼표 포함 시 따옴표로 감싸기
        return typeof val === 'string' && val.includes(',')
          ? `"${val}"`
          : val
      }).join(',')
    )
  ].join('\n')

  downloadFile(csvContent, `${filename}.csv`, 'text/csv;charset=utf-8;')
}

// Excel 내보내기
export async function exportToExcel(data, filename) {
  // xlsx 라이브러리 동적 import
  const XLSX = await import('xlsx')

  const workbook = XLSX.utils.book_new()

  // Summary 시트
  const summaryData = generateSummary(data)
  const summarySheet = XLSX.utils.aoa_to_sheet(summaryData)
  XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary')

  // Predictions 시트
  const predictionSheet = XLSX.utils.json_to_sheet(data)
  XLSX.utils.book_append_sheet(workbook, predictionSheet, 'Predictions')

  // 조건부 서식 (xlsx-style 사용 시)
  // applyConditionalFormatting(predictionSheet)

  // 다운로드
  XLSX.writeFile(workbook, `${filename}.xlsx`)
}

// JSON 내보내기
export function exportToJSON(data, filename) {
  const jsonContent = JSON.stringify(data, null, 2)
  downloadFile(jsonContent, `${filename}.json`, 'application/json')
}

// 파일 다운로드 헬퍼
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// 요약 데이터 생성
function generateSummary(data) {
  const upCount = data.filter(d => d.prediction_direction === 'UP').length
  const downCount = data.filter(d => d.prediction_direction === 'DOWN').length
  const gradeA = data.filter(d => d.practicality_grade === 'A').length
  const avgProb = data.reduce((sum, d) => sum + d.probability, 0) / data.length

  return [
    ['FiveForFree 예측 리포트'],
    ['생성일', new Date().toISOString()],
    [],
    ['총 티커 수', data.length],
    ['상승 신호', upCount, `${((upCount / data.length) * 100).toFixed(1)}%`],
    ['하락 신호', downCount, `${((downCount / data.length) * 100).toFixed(1)}%`],
    ['A등급', gradeA],
    ['평균 확률', `${avgProb.toFixed(1)}%`],
  ]
}

// 데이터 준비
export async function prepareExportData(predictions, options) {
  const { range, includeBasic, includePrediction, includeModel, onProgress } = options

  let data = predictions

  // 범위에 따른 데이터 선택
  if (range === 'all') {
    data = await fetchAllPredictions()
  } else if (range === 'watchlist') {
    data = await fetchWatchlistPredictions()
  }

  // 진행률 업데이트하며 데이터 변환
  const result = []
  for (let i = 0; i < data.length; i++) {
    const item = data[i]
    const row = {}

    if (includeBasic) {
      row.ticker = item.ticker
      row.name = item.name
      row.current_price = item.current_price
      row.change_percent = item.change_percent
      row.volume = item.volume
    }

    if (includePrediction) {
      row.prediction_direction = item.direction?.toUpperCase()
      row.probability = item.probability
      row.practicality_grade = item.practicality_grade
      row.best_model = item.best_model
      row.trading_signal = getTradingSignal(item)
    }

    if (includeModel) {
      row.precision = item.hit_rate
      row.signal_rate = item.signal_rate
      row.predictions_count = item.predictions_count
    }

    result.push(row)

    // 진행률 업데이트
    if (onProgress) {
      onProgress(Math.round(((i + 1) / data.length) * 100))
    }
  }

  return result
}
```

---

## 7. API 요구사항

### 7.1 전체 데이터 API

#### GET /api/predictions/export

**Parameters:**
- `format`: csv, xlsx, json
- `include_models`: boolean

**Response:** 파일 다운로드 또는 JSON

### 7.2 서버사이드 내보내기 (대용량)

```python
# FastAPI 엔드포인트
@router.get("/predictions/export")
async def export_predictions(
    format: str = "csv",
    include_models: bool = False,
):
    predictions = await get_all_predictions()

    if format == "csv":
        return StreamingResponse(
            generate_csv(predictions),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
```

---

## 8. 성능 고려사항

### 8.1 대용량 데이터

| 데이터 크기 | 전략 |
|-------------|------|
| < 100개 | 클라이언트 사이드 처리 |
| 100-1000개 | 프로그레스 바 표시 |
| > 1000개 | 서버사이드 생성 후 다운로드 링크 |

### 8.2 메모리 최적화

```javascript
// 스트리밍 방식 CSV 생성
function* generateCSVRows(data) {
  yield Object.keys(data[0]).join(',') + '\n'
  for (const row of data) {
    yield Object.values(row).join(',') + '\n'
  }
}
```

---

## 9. 구현 일정

### Phase 1: 기본 구조 (1일)
- [ ] ExportModal.jsx 구현
- [ ] 기본 UI/UX

### Phase 2: CSV 내보내기 (반나절)
- [ ] exportToCSV 함수
- [ ] 파일 다운로드

### Phase 3: Excel 내보내기 (1일)
- [ ] xlsx 라이브러리 연동
- [ ] 다중 시트 생성
- [ ] 조건부 서식

### Phase 4: 최적화 (반나절)
- [ ] 프로그레스 바
- [ ] 대용량 데이터 처리
- [ ] 에러 핸들링

---

## 10. 테스트 체크리스트

| 항목 | 확인 |
|------|------|
| 내보내기 모달 열기/닫기 | [ ] |
| CSV 파일 생성 | [ ] |
| CSV 한글 인코딩 | [ ] |
| Excel 파일 생성 | [ ] |
| Excel 다중 시트 | [ ] |
| JSON 파일 생성 | [ ] |
| 파일명 커스터마이징 | [ ] |
| 프로그레스 바 | [ ] |
| 100개+ 데이터 처리 | [ ] |
| 모바일 다운로드 | [ ] |

---

*이 문서는 분석팀장이 작성하였습니다.*
