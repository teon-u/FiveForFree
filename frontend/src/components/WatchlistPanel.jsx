import { useState, useMemo } from 'react'
import clsx from 'clsx'
import { useWatchlistStore } from '../stores/watchlistStore'
import { usePredictions } from '../hooks/usePredictions'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'

function getGradeStyle(grade) {
  switch (grade) {
    case 'A': return 'bg-green-500/20 text-green-400'
    case 'B': return 'bg-blue-500/20 text-blue-400'
    case 'C': return 'bg-yellow-500/20 text-yellow-400'
    default: return 'bg-gray-500/20 text-gray-400'
  }
}

function WatchlistItem({ ticker, prediction, alertEnabled, isEditing, onToggleAlert, onRemove, onViewDetail }) {
  return (
    <div className="flex items-center justify-between p-3 bg-surface-light rounded-lg">
      <div
        className="flex-1 cursor-pointer min-w-0"
        onClick={() => onViewDetail?.(ticker)}
        onKeyDown={(e) => e.key === 'Enter' && onViewDetail?.(ticker)}
        role="button"
        tabIndex={0}
      >
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-yellow-400">⭐</span>
          <span className="font-bold">{ticker}</span>
          {prediction && (
            <>
              <span className={clsx(
                'text-sm',
                prediction.direction === 'up' ? 'text-green-400' : 'text-red-400'
              )}>
                {prediction.probability?.toFixed(0)}% {prediction.direction === 'up' ? '↑' : '↓'}
              </span>
              <span className={clsx('px-1.5 py-0.5 rounded text-xs font-bold', getGradeStyle(prediction.practicality_grade))}>
                {prediction.practicality_grade}
              </span>
            </>
          )}
        </div>
        {prediction && (
          <div className="text-xs text-gray-400 mt-0.5">
            ${prediction.current_price?.toFixed(2) || '-'}
            <span className={prediction.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
              {' '}{prediction.change_percent >= 0 ? '+' : ''}{prediction.change_percent?.toFixed(1) || 0}%
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onToggleAlert}
          className={clsx(
            'p-1.5 rounded transition-colors',
            alertEnabled ? 'text-blue-400 hover:bg-blue-500/20' : 'text-gray-500 hover:bg-gray-500/20'
          )}
          aria-label={alertEnabled ? 'Disable alert' : 'Enable alert'}
        >
          {alertEnabled ? '🔔' : '🔕'}
        </button>

        {isEditing && (
          <button
            onClick={onRemove}
            className="text-red-400 hover:text-red-300 p-1.5"
            aria-label="Remove from watchlist"
          >
            🗑️
          </button>
        )}
      </div>
    </div>
  )
}

export default function WatchlistPanel({ onClose, onViewDetail }) {
  const { language } = useSettingsStore()
  const { watchlist, addTicker, removeTicker, toggleAlert } = useWatchlistStore()
  const { volumeTop100, gainersTop100 } = usePredictions()
  const [searchQuery, setSearchQuery] = useState('')
  const [isEditing, setIsEditing] = useState(false)
  const tr = t(language)

  // Combine all predictions for search
  const allPredictions = useMemo(() => {
    const all = [...(gainersTop100 || []), ...(volumeTop100 || [])]
    // Remove duplicates
    return all.filter((item, index, self) =>
      index === self.findIndex(t => t.ticker === item.ticker)
    )
  }, [gainersTop100, volumeTop100])

  // Watchlist predictions
  const watchlistPredictions = useMemo(() => {
    return watchlist
      .map(item => ({
        ...item,
        prediction: allPredictions.find(p => p.ticker === item.ticker)
      }))
  }, [watchlist, allPredictions])

  // Search results
  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return []
    const query = searchQuery.toLowerCase()
    return allPredictions
      .filter(p =>
        p.ticker.toLowerCase().includes(query) &&
        !watchlist.some(w => w.ticker === p.ticker)
      )
      .slice(0, 5)
  }, [searchQuery, allPredictions, watchlist])

  const handleAddTicker = (ticker) => {
    addTicker(ticker)
    setSearchQuery('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') onClose()
  }

  const enabledAlertCount = watchlist.filter(w => w.alertEnabled).length

  return (
    <>
      <div
        className="modal-backdrop"
        onClick={onClose}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-label="Close"
      />

      <div
        className="fixed right-0 top-0 bottom-0 w-full max-w-[400px] bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between z-10">
          <h2 className="text-xl font-bold flex items-center gap-2">
            ⭐ {tr('watchlist.title')}
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              {isEditing ? tr('watchlist.done') : tr('watchlist.edit')}
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white text-2xl p-1"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Summary */}
          <div className="text-sm text-gray-400">
            📊 {watchlist.length} {tr('watchlist.tickers')} • {tr('watchlist.alerts')}: {enabledAlertCount} {tr('watchlist.active')}
          </div>

          {/* Watchlist Items */}
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
                onViewDetail={onViewDetail}
              />
            ))}

            {watchlist.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <div className="text-4xl mb-2">⭐</div>
                <p>{tr('watchlist.empty')}</p>
                <p className="text-sm">{tr('watchlist.emptyHint')}</p>
              </div>
            )}
          </div>

          {/* Search */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={`🔍 ${tr('watchlist.searchPlaceholder')}`}
              className="w-full px-4 py-2 bg-surface-light rounded-lg text-white placeholder-gray-500 border border-surface-light focus:border-blue-500 outline-none transition-colors"
            />

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-surface border border-surface-light rounded-lg shadow-xl z-10 overflow-hidden">
                {searchResults.map((pred) => (
                  <button
                    key={pred.ticker}
                    onClick={() => handleAddTicker(pred.ticker)}
                    className="w-full flex items-center justify-between p-3 hover:bg-surface-light text-left transition-colors"
                  >
                    <div className="min-w-0">
                      <span className="font-bold">{pred.ticker}</span>
                    </div>
                    <span className={pred.direction === 'up' ? 'text-green-400' : 'text-red-400'}>
                      {pred.probability?.toFixed(0)}% {pred.direction === 'up' ? '↑' : '↓'}
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
