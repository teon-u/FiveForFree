import { useState } from 'react'
import clsx from 'clsx'
import {
  exportToCSV,
  exportToExcel,
  exportToJSON,
  prepareExportData,
  validateExportOptions
} from '../utils/exportUtils'

const FORMATS = [
  { key: 'csv', label: 'CSV', icon: '📄', description: 'Universal format' },
  { key: 'xlsx', label: 'Excel', icon: '📊', description: 'With formatting' },
  { key: 'json', label: 'JSON', icon: '{ }', description: 'For developers' }
]

const RANGES = [
  { key: 'current', label: 'Current View', description: 'Filtered results' },
  { key: 'all', label: 'All Data', description: 'All predictions' }
]

export default function ExportModal({ predictions, allPredictions, onClose }) {
  const [format, setFormat] = useState('csv')
  const [range, setRange] = useState('current')
  const [includeBasic, setIncludeBasic] = useState(true)
  const [includePrediction, setIncludePrediction] = useState(true)
  const [includeModel, setIncludeModel] = useState(true)
  const [includeDetailedModels, setIncludeDetailedModels] = useState(false)
  const [filename, setFilename] = useState(
    `fiveforfree_predictions_${new Date().toISOString().split('T')[0]}`
  )
  const [isExporting, setIsExporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  const currentCount = predictions?.length || 0
  const allCount = allPredictions?.length || currentCount

  const handleExport = async () => {
    setIsExporting(true)
    setProgress(0)
    setError(null)

    try {
      // Validate options
      validateExportOptions({
        includeBasic,
        includePrediction,
        includeModel
      })

      // Select data based on range
      const sourceData = range === 'all'
        ? (allPredictions || predictions)
        : predictions

      // Prepare data
      const data = prepareExportData(sourceData, {
        includeBasic,
        includePrediction,
        includeModel,
        includeDetailedModels,
        onProgress: setProgress
      })

      if (data.length === 0) {
        throw new Error('No data to export')
      }

      // Export based on format
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
        default:
          throw new Error(`Unknown format: ${format}`)
      }

      // Close modal on success
      onClose()
    } catch (err) {
      console.error('Export failed:', err)
      setError(err.message || 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  // Close on Escape key
  const handleKeyDown = (e) => {
    if (e.key === 'Escape' && !isExporting) onClose()
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="modal-backdrop"
        onClick={!isExporting ? onClose : undefined}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-label="Close modal"
      />

      {/* Modal */}
      <div
        className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-lg mx-auto bg-surface rounded-2xl shadow-2xl z-50 p-5 md:p-6 max-h-[90vh] overflow-y-auto"
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-modal-title"
      >
        {isExporting ? (
          /* Progress View */
          <div className="text-center py-8">
            <div className="text-lg font-bold mb-4">Preparing download...</div>
            <div className="w-full bg-surface-light rounded-full h-2.5 mb-3">
              <div
                className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-sm text-gray-400">{progress}% complete</div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
              <h2 id="export-modal-title" className="text-xl font-bold flex items-center gap-2">
                <span className="text-2xl">📥</span>
                Export Data
              </h2>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white text-xl p-1 hover:bg-surface-light rounded transition-colors"
                aria-label="Close"
              >
                &times;
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-3 bg-red-500/20 border border-red-500/40 rounded-lg text-red-400 text-sm">
                {error}
              </div>
            )}

            {/* File Format */}
            <div className="mb-5">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                File Format
              </div>
              <div className="grid grid-cols-3 gap-2">
                {FORMATS.map((f) => (
                  <button
                    key={f.key}
                    onClick={() => setFormat(f.key)}
                    className={clsx(
                      'flex flex-col items-center px-3 py-3 rounded-lg font-medium transition-all',
                      format === f.key
                        ? 'bg-blue-500 text-white ring-2 ring-blue-400 ring-offset-2 ring-offset-surface'
                        : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                    )}
                  >
                    <span className="text-lg mb-1">{f.icon}</span>
                    <span className="text-sm">{f.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Data Range */}
            <div className="mb-5">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                Data Range
              </div>
              <div className="space-y-2">
                {RANGES.map((r) => {
                  const count = r.key === 'current' ? currentCount : allCount
                  return (
                    <label
                      key={r.key}
                      className={clsx(
                        'flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors',
                        range === r.key
                          ? 'bg-blue-500/20 border border-blue-500/40'
                          : 'bg-surface-light hover:bg-slate-600'
                      )}
                    >
                      <input
                        type="radio"
                        name="range"
                        checked={range === r.key}
                        onChange={() => setRange(r.key)}
                        className="accent-blue-500 w-4 h-4"
                      />
                      <div className="flex-1">
                        <span className="font-medium">{r.label}</span>
                        <span className="text-gray-400 text-sm ml-2">
                          ({count} items)
                        </span>
                      </div>
                    </label>
                  )
                })}
              </div>
            </div>

            {/* Include Options */}
            <div className="mb-5">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                Include Fields
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded hover:bg-surface-light cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={includeBasic}
                    onChange={(e) => setIncludeBasic(e.target.checked)}
                    className="accent-blue-500 w-4 h-4"
                  />
                  <div>
                    <span>Basic Info</span>
                    <span className="text-gray-500 text-sm ml-2">
                      (Ticker, Price, Change)
                    </span>
                  </div>
                </label>
                <label className="flex items-center gap-3 p-2 rounded hover:bg-surface-light cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={includePrediction}
                    onChange={(e) => setIncludePrediction(e.target.checked)}
                    className="accent-blue-500 w-4 h-4"
                  />
                  <div>
                    <span>Prediction</span>
                    <span className="text-gray-500 text-sm ml-2">
                      (Probability, Direction, Grade)
                    </span>
                  </div>
                </label>
                <label className="flex items-center gap-3 p-2 rounded hover:bg-surface-light cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={includeModel}
                    onChange={(e) => setIncludeModel(e.target.checked)}
                    className="accent-blue-500 w-4 h-4"
                  />
                  <div>
                    <span>Model Stats</span>
                    <span className="text-gray-500 text-sm ml-2">
                      (Precision, Signal Rate)
                    </span>
                  </div>
                </label>
                <label className="flex items-center gap-3 p-2 rounded hover:bg-surface-light cursor-pointer transition-colors text-gray-500">
                  <input
                    type="checkbox"
                    checked={includeDetailedModels}
                    onChange={(e) => setIncludeDetailedModels(e.target.checked)}
                    className="accent-blue-500 w-4 h-4"
                  />
                  <div>
                    <span>Detailed Models</span>
                    <span className="text-gray-600 text-sm ml-2">
                      (XGB, LGBM, LSTM, Transformer)
                    </span>
                  </div>
                </label>
              </div>
            </div>

            {/* Filename */}
            <div className="mb-6">
              <div className="text-sm font-semibold text-gray-300 mb-2">
                Filename
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={filename}
                  onChange={(e) => setFilename(e.target.value)}
                  className="flex-1 px-4 py-2 bg-surface-light rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter filename"
                />
                <span className="text-gray-500">.{format}</span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2.5 bg-surface-light text-gray-400 rounded-lg hover:bg-slate-600 transition-colors font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleExport}
                disabled={!includeBasic && !includePrediction && !includeModel}
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                  (!includeBasic && !includePrediction && !includeModel)
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                )}
              >
                <span>📥</span>
                Download
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}
