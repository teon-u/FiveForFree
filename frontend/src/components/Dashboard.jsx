import { useState } from 'react'
import TickerGrid from './TickerGrid'
import PredictionPanel from './PredictionPanel'
import { usePredictions } from '../hooks/usePredictions'
import { useSettingsStore } from '../stores/settingsStore'

export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState(null)
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
    <div className="space-y-8">
      {/* Volume Top 100 Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Volume Top 100
            <span className="text-sm text-gray-400 font-normal">
              ({filteredVolumeTop100?.length || 0} tickers)
            </span>
          </h2>
        </div>
        <TickerGrid
          predictions={filteredVolumeTop100}
          onTickerClick={setSelectedTicker}
        />
      </section>

      {/* Gainers Top 100 Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">📈</span>
            Gainers Top 100
            <span className="text-sm text-gray-400 font-normal">
              ({filteredGainersTop100?.length || 0} tickers)
            </span>
          </h2>
        </div>
        <TickerGrid
          predictions={filteredGainersTop100}
          onTickerClick={setSelectedTicker}
        />
      </section>

      {/* Prediction Detail Panel */}
      {selectedTicker && (
        <PredictionPanel
          ticker={selectedTicker}
          onClose={() => setSelectedTicker(null)}
        />
      )}
    </div>
  )
}
