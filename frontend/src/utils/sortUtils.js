// Grade to number conversion for sorting
const gradeToNumber = (grade) => {
  const gradeMap = { 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'N/A': 0 }
  return gradeMap[grade] || 0
}

export const SORT_OPTIONS = {
  probability: {
    label: { ko: '예측 확률', en: 'Probability' },
    getValue: (item) => item.probability ?? 0,
    defaultOrder: 'desc'
  },
  precision: {
    label: { ko: 'Precision (적중률)', en: 'Precision' },
    getValue: (item) => item.hit_rate ?? 0,
    defaultOrder: 'desc'
  },
  signalRate: {
    label: { ko: 'Signal Rate', en: 'Signal Rate' },
    getValue: (item) => item.signal_rate ?? 0,
    defaultOrder: 'desc'
  },
  grade: {
    label: { ko: '등급', en: 'Grade' },
    getValue: (item) => gradeToNumber(item.practicality_grade),
    defaultOrder: 'desc'
  },
  changePercent: {
    label: { ko: '변동률', en: 'Change %' },
    getValue: (item) => item.change_percent ?? 0,
    defaultOrder: 'desc'
  },
  price: {
    label: { ko: '가격', en: 'Price' },
    getValue: (item) => item.current_price ?? 0,
    defaultOrder: 'desc'
  },
  ticker: {
    label: { ko: '티커명', en: 'Ticker' },
    getValue: (item) => item.ticker ?? '',
    defaultOrder: 'asc'
  }
}

export const SORT_PRESETS = {
  bestOpportunity: {
    label: { ko: '최고 기회순', en: 'Best Opportunity' },
    description: { ko: '등급 → 확률', en: 'Grade → Probability' },
    configs: [
      { field: 'grade', order: 'desc' },
      { field: 'probability', order: 'desc' }
    ]
  },
  mostReliable: {
    label: { ko: '신뢰도순', en: 'Most Reliable' },
    description: { ko: 'Precision → Signal', en: 'Precision → Signal' },
    configs: [
      { field: 'precision', order: 'desc' },
      { field: 'signalRate', order: 'desc' }
    ]
  },
  highMomentum: {
    label: { ko: '모멘텀순', en: 'High Momentum' },
    description: { ko: '변동률 → 확률', en: 'Change → Probability' },
    configs: [
      { field: 'changePercent', order: 'desc' },
      { field: 'probability', order: 'desc' }
    ]
  },
  activeSignals: {
    label: { ko: '활성 신호순', en: 'Active Signals' },
    description: { ko: 'Signal → Precision', en: 'Signal → Precision' },
    configs: [
      { field: 'signalRate', order: 'desc' },
      { field: 'precision', order: 'desc' }
    ]
  }
}

// Single sort function
export const sortPredictions = (predictions, sortBy, order = 'desc') => {
  if (!predictions || !SORT_OPTIONS[sortBy]) return predictions || []

  const option = SORT_OPTIONS[sortBy]
  const multiplier = order === 'desc' ? -1 : 1

  return [...predictions].sort((a, b) => {
    const valueA = option.getValue(a)
    const valueB = option.getValue(b)

    // String comparison (ticker name)
    if (typeof valueA === 'string') {
      return multiplier * valueA.localeCompare(valueB)
    }

    // Numeric comparison
    return multiplier * (valueA - valueB)
  })
}

// Multi-sort function (supports primary and secondary sorting)
export const multiSort = (predictions, sortConfigs) => {
  if (!predictions || !sortConfigs || sortConfigs.length === 0) {
    return predictions || []
  }

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

// Get sort configs from current state
export const getSortConfigs = (sortMode, presetKey, singleSort, multiSortConfigs) => {
  switch (sortMode) {
    case 'preset':
      return SORT_PRESETS[presetKey]?.configs || []
    case 'single':
      return [singleSort]
    case 'multi':
      return multiSortConfigs
    default:
      return [{ field: 'probability', order: 'desc' }]
  }
}
