import { useState, useMemo, useCallback } from 'react'
import TickerGrid from './TickerGrid'
import ModelDetailModal from './ModelDetailModal'
import ChartModal from './ChartModal'
import ExportModal from './ExportModal'
import FilterBar from './filters/FilterBar'
import { usePredictions } from '../hooks/usePredictions'
import { useSettingsStore } from '../stores/settingsStore'
import { useFilterStore } from '../stores/filterStore'
import { useSortStore } from '../stores/sortStore'
import { multiSort } from '../utils/sortUtils'
import { t } from '../i18n'

export default function Dashboard() {
  const [detailModalTicker, setDetailModalTicker] = useState(null)
  const [chartModalTicker, setChartModalTicker] = useState(null)
  const [showExportModal, setShowExportModal] = useState(false)
  const [activeCategory, setActiveCategory] = useState('gainers') // 'gainers' or 'volume'
  const { volumeTop100, gainersTop100, isLoading, error } = usePredictions()
  const { language } = useSettingsStore()
  const { directions, probabilityRange } = useFilterStore()
  const { getCurrentSortConfigs } = useSortStore()
  const tr = t(language)

  // Filter predictions based on new filter store
  const filterPredictions = useCallback((predictions) => {
    if (!predictions) return []

    return predictions.filter(p => {
      // Direction filter
      if (!directions.includes(p.direction)) {
        return false
      }

      // Probability filter
      if (p.probability < probabilityRange.min || p.probability > probabilityRange.max) {
        return false
      }

      return true
    })
  }, [directions, probabilityRange])

  // Apply filters and sorting
  const processedPredictions = useMemo(() => {
    const rawPredictions = activeCategory === 'gainers' ? gainersTop100 : volumeTop100
    const filtered = filterPredictions(rawPredictions)
    const sortConfigs = getCurrentSortConfigs()
    return multiSort(filtered, sortConfigs)
  }, [activeCategory, gainersTop100, volumeTop100, filterPredictions, getCurrentSortConfigs])

  const filteredGainersCount = useMemo(() => {
    return filterPredictions(gainersTop100)?.length || 0
  }, [gainersTop100, filterPredictions])

  const filteredVolumeCount = useMemo(() => {
    return filterPredictions(volumeTop100)?.length || 0
  }, [volumeTop100, filterPredictions])

  const categoryTitle = activeCategory === 'gainers'
    ? tr('dashboard.gainersTop100')
    : tr('dashboard.volumeTop100')

  const categoryIcon = activeCategory === 'gainers' ? '📈' : '📊'

  // Get all predictions for export
  const allPredictions = useMemo(() => {
    const all = [...(gainersTop100 || []), ...(volumeTop100 || [])]
    // Remove duplicates by ticker
    const unique = all.filter((item, index, self) =>
      index === self.findIndex(t => t.ticker === item.ticker)
    )
    return unique
  }, [gainersTop100, volumeTop100])

  // Find prediction for chart modal
  const chartPrediction = useMemo(() => {
    if (!chartModalTicker) return null
    return allPredictions.find(p => p.ticker === chartModalTicker) || null
  }, [chartModalTicker, allPredictions])

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-2">{tr('dashboard.error')}</h2>
          <p className="text-gray-400">{error.message}</p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="spinner mx-auto mb-4" />
          <p className="text-gray-400">{tr('dashboard.loading')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Filter Bar */}
      <FilterBar />

      {/* Category Toggle Button */}
      <div className="flex justify-center">
        <div className="inline-flex rounded-lg bg-gray-800 p-1 gap-1">
          <button
            onClick={() => setActiveCategory('gainers')}
            className={`
              px-6 py-2.5 rounded-lg font-medium transition-all duration-200
              flex items-center gap-2
              ${activeCategory === 'gainers'
                ? 'bg-green-600 text-white shadow-lg shadow-green-500/50'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }
            `}
          >
            <span className="text-lg">📈</span>
            {tr('dashboard.gainers')}
            <span className="text-xs opacity-75">
              ({filteredGainersCount})
            </span>
          </button>
          <button
            onClick={() => setActiveCategory('volume')}
            className={`
              px-6 py-2.5 rounded-lg font-medium transition-all duration-200
              flex items-center gap-2
              ${activeCategory === 'volume'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }
            `}
          >
            <span className="text-lg">📊</span>
            {tr('dashboard.volume')}
            <span className="text-xs opacity-75">
              ({filteredVolumeCount})
            </span>
          </button>
        </div>
      </div>

      {/* Active Category Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">{categoryIcon}</span>
            {categoryTitle}
            <span className="text-sm text-gray-400 font-normal">
              ({processedPredictions?.length || 0} {tr('dashboard.tickers')})
            </span>
          </h2>
          {/* Export Button */}
          <button
            onClick={() => setShowExportModal(true)}
            className="px-4 py-2 bg-surface-light hover:bg-slate-600 text-gray-300 hover:text-white rounded-lg transition-colors flex items-center gap-2 text-sm font-medium"
            aria-label="Export data"
          >
            <span>📥</span>
            {tr('dashboard.export')}
          </button>
        </div>
        <TickerGrid
          predictions={processedPredictions}
          onTickerClick={setChartModalTicker}
          onDetailClick={setDetailModalTicker}
        />
      </section>

      {/* Model Detail Modal */}
      {detailModalTicker && (
        <ModelDetailModal
          ticker={detailModalTicker}
          onClose={() => setDetailModalTicker(null)}
        />
      )}

      {/* Chart Modal */}
      {chartModalTicker && (
        <ChartModal
          ticker={chartModalTicker}
          prediction={chartPrediction}
          onClose={() => setChartModalTicker(null)}
        />
      )}

      {/* Export Modal */}
      {showExportModal && (
        <ExportModal
          predictions={processedPredictions}
          allPredictions={allPredictions}
          onClose={() => setShowExportModal(false)}
        />
      )}
    </div>
  )
}
