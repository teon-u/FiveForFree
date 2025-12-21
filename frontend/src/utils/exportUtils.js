/**
 * Export utilities for CSV, Excel, and JSON
 */

/**
 * Download file helper
 */
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

/**
 * Escape CSV value (handle commas, quotes, newlines)
 */
function escapeCSV(value) {
  if (value == null) return ''
  const stringValue = String(value)
  if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
    return `"${stringValue.replace(/"/g, '""')}"`
  }
  return stringValue
}

/**
 * Export data to CSV format
 */
export function exportToCSV(data, filename) {
  if (!data || data.length === 0) {
    throw new Error('No data to export')
  }

  const headers = Object.keys(data[0])
  const csvContent = [
    // BOM for Excel UTF-8 compatibility
    '\uFEFF',
    headers.join(','),
    ...data.map(row =>
      headers.map(h => escapeCSV(row[h])).join(',')
    )
  ].join('\n')

  downloadFile(csvContent, `${filename}.csv`, 'text/csv;charset=utf-8;')
}

/**
 * Export data to Excel format (.xlsx)
 */
export async function exportToExcel(data, filename) {
  if (!data || data.length === 0) {
    throw new Error('No data to export')
  }

  // Dynamic import for xlsx library
  const XLSX = await import('xlsx')

  const workbook = XLSX.utils.book_new()

  // Summary sheet
  const summaryData = generateSummary(data)
  const summarySheet = XLSX.utils.aoa_to_sheet(summaryData)
  XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary')

  // Predictions sheet
  const predictionSheet = XLSX.utils.json_to_sheet(data)
  // Set column widths
  const colWidths = Object.keys(data[0]).map(key => ({
    wch: Math.max(key.length, 12)
  }))
  predictionSheet['!cols'] = colWidths
  XLSX.utils.book_append_sheet(workbook, predictionSheet, 'Predictions')

  // Download
  XLSX.writeFile(workbook, `${filename}.xlsx`)
}

/**
 * Export data to JSON format
 */
export function exportToJSON(data, filename) {
  if (!data || data.length === 0) {
    throw new Error('No data to export')
  }

  const exportData = {
    exportedAt: new Date().toISOString(),
    count: data.length,
    data
  }

  const jsonContent = JSON.stringify(exportData, null, 2)
  downloadFile(jsonContent, `${filename}.json`, 'application/json')
}

/**
 * Generate summary statistics for Excel
 */
function generateSummary(data) {
  const upCount = data.filter(d => d.prediction_direction === 'UP').length
  const downCount = data.filter(d => d.prediction_direction === 'DOWN').length
  const gradeA = data.filter(d => d.practicality_grade === 'A').length
  const gradeB = data.filter(d => d.practicality_grade === 'B').length
  const avgProb = data.length > 0
    ? data.reduce((sum, d) => sum + (d.probability || 0), 0) / data.length
    : 0
  const avgPrecision = data.length > 0
    ? data.reduce((sum, d) => sum + (d.precision || 0), 0) / data.length
    : 0

  return [
    ['FiveForFree Prediction Report'],
    [],
    ['Generated', new Date().toLocaleString('ko-KR')],
    [],
    ['Statistics'],
    ['Total Tickers', data.length],
    ['Up Signals', upCount, `${((upCount / data.length) * 100).toFixed(1)}%`],
    ['Down Signals', downCount, `${((downCount / data.length) * 100).toFixed(1)}%`],
    [],
    ['Grade Distribution'],
    ['Grade A', gradeA],
    ['Grade B', gradeB],
    ['Grade C', data.filter(d => d.practicality_grade === 'C').length],
    ['Grade D', data.filter(d => d.practicality_grade === 'D').length],
    [],
    ['Averages'],
    ['Average Probability', `${avgProb.toFixed(1)}%`],
    ['Average Precision', `${avgPrecision.toFixed(1)}%`]
  ]
}

/**
 * Get trading signal based on prediction
 */
function getTradingSignal(item) {
  if (!item.probability) return 'HOLD'
  if (item.probability >= 70 && item.practicality_grade === 'A') {
    return item.direction === 'up' ? 'BUY' : 'SELL'
  }
  if (item.probability >= 80) {
    return item.direction === 'up' ? 'BUY' : 'SELL'
  }
  return 'HOLD'
}

/**
 * Prepare export data with selected columns
 */
export function prepareExportData(predictions, options) {
  const {
    includeBasic = true,
    includePrediction = true,
    includeModel = true,
    includeDetailedModels = false,
    onProgress
  } = options

  if (!predictions || predictions.length === 0) {
    return []
  }

  const result = []
  const total = predictions.length

  for (let i = 0; i < predictions.length; i++) {
    const item = predictions[i]
    const row = {}

    if (includeBasic) {
      row.ticker = item.ticker
      row.name = item.name || ''
      row.current_price = item.current_price || item.price || null
      row.change_percent = item.change_percent || null
      row.volume = item.volume || null
    }

    if (includePrediction) {
      row.prediction_direction = (item.direction || '').toUpperCase()
      row.probability = item.probability || null
      row.practicality_grade = item.practicality_grade || ''
      row.best_model = item.best_model || ''
      row.trading_signal = getTradingSignal(item)
    }

    if (includeModel) {
      row.precision = item.hit_rate || null
      row.signal_rate = item.signal_rate || null
      row.predictions_count = item.predictions_count || null
    }

    if (includeDetailedModels) {
      row.xgb_probability = item.model_details?.xgboost?.probability || null
      row.xgb_precision = item.model_details?.xgboost?.precision || null
      row.lgbm_probability = item.model_details?.lightgbm?.probability || null
      row.lgbm_precision = item.model_details?.lightgbm?.precision || null
      row.lstm_probability = item.model_details?.lstm?.probability || null
      row.lstm_precision = item.model_details?.lstm?.precision || null
      row.transformer_probability = item.model_details?.transformer?.probability || null
      row.transformer_precision = item.model_details?.transformer?.precision || null
    }

    // Add export timestamp
    row.export_timestamp = new Date().toISOString()

    result.push(row)

    // Progress callback
    if (onProgress) {
      onProgress(Math.round(((i + 1) / total) * 100))
    }
  }

  return result
}

/**
 * Validate export options
 */
export function validateExportOptions(options) {
  const { includeBasic, includePrediction, includeModel } = options
  if (!includeBasic && !includePrediction && !includeModel) {
    throw new Error('At least one data category must be selected')
  }
  return true
}
