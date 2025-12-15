import { useState } from 'react'
import TickerGrid from './TickerGrid'
import PredictionPanel from './PredictionPanel'
import ModelDetailModal from './ModelDetailModal'
import { usePredictions } from '../hooks/usePredictions'
import { useSettingsStore } from '../stores/settingsStore'

export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState(null)
  const [detailModalTicker, setDetailModalTicker] = useState(null)
  const [activeCategory, setActiveCategory] = useState('volume') // 'volume' or 'gainers'
  const { volumeTop100, gainersTop100, isLoading, error } = usePredictions()
  const { filterMode } = useSettingsStore()

  // Filter predictions based on settings
  const filterPredictions = (predictions) => {
    if (!predictions) return []

    if (filterMode === 'up') {
      return predictions.filter(p => p.direction === 'up')
    } else if (filterMode === 'down') {
      return predictions.filter(p => p.direction === 'down')
    }
    return predictions // 'all' mode
  }

  const filteredVolumeTop100 = filterPredictions(volumeTop100)
  const filteredGainersTop100 = filterPredictions(gainersTop100)

  // Get currently active predictions based on toggle
  const activePredictions = activeCategory === 'volume'
    ? filteredVolumeTop100
    : filteredGainersTop100

  const categoryTitle = activeCategory === 'volume'
    ? 'Volume Top 100'
    : 'Gainers Top 100'

  const categoryIcon = activeCategory === 'volume' ? '📊' : '📈'

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
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
          <p className="text-gray-400">Loading predictions...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Category Toggle Button */}
      <div className="flex justify-center">
        <div className="inline-flex rounded-lg bg-gray-800 p-1 gap-1">
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
            거래량
            <span className="text-xs opacity-75">
              ({filteredVolumeTop100?.length || 0})
            </span>
          </button>
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
            상승률
            <span className="text-xs opacity-75">
              ({filteredGainersTop100?.length || 0})
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
              ({activePredictions?.length || 0} tickers)
            </span>
          </h2>
        </div>
        <TickerGrid
          predictions={activePredictions}
          onTickerClick={setSelectedTicker}
          onDetailClick={setDetailModalTicker}
        />
      </section>

      {/* Prediction Detail Panel */}
      {selectedTicker && (
        <PredictionPanel
          ticker={selectedTicker}
          onClose={() => setSelectedTicker(null)}
        />
      )}

      {/* Model Detail Modal */}
      {detailModalTicker && (
        <ModelDetailModal
          ticker={detailModalTicker}
          onClose={() => setDetailModalTicker(null)}
        />
      )}
    </div>
  )
}
