import { useState } from 'react'
import clsx from 'clsx'
import { useDiscovery, useTrainTicker } from '../hooks/useDiscovery'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'

function formatVolume(volume) {
  if (!volume) return '-'
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(0)}M`
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(0)}K`
  return volume.toString()
}

function StatCard({ label, value, highlight = false }) {
  return (
    <div className="bg-surface-light rounded-lg p-3 text-center">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={clsx('text-xl font-bold', highlight ? 'text-green-400' : 'text-white')}>
        {value}
      </div>
    </div>
  )
}

function TickerRow({ ticker, trainingStatus, onTrain, onViewDetail }) {
  const isTrained = ticker.is_trained || trainingStatus === 'queued'
  const isTraining = trainingStatus === 'training'

  return (
    <div className="flex items-center justify-between p-3 bg-surface-light rounded-lg">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-bold text-white">{ticker.ticker}</span>
          <span className="text-green-400 text-sm">+{ticker.change_percent?.toFixed(1)}%</span>
        </div>
        <div className="text-xs text-gray-400 truncate max-w-[180px]">
          {ticker.name}
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-gray-400">
          {formatVolume(ticker.volume)}
        </span>
        <span className={clsx('text-xs', isTrained ? 'text-green-400' : 'text-yellow-400')}>
          {isTrained ? '✅' : '⚠️'}
        </span>

        {isTrained ? (
          <button
            onClick={() => onViewDetail?.(ticker.ticker)}
            className="px-3 py-1 text-xs bg-blue-500/20 text-blue-400 rounded hover:bg-blue-500/30 transition-colors"
          >
            📈
          </button>
        ) : (
          <button
            onClick={onTrain}
            disabled={isTraining}
            className={clsx(
              'px-3 py-1 text-xs rounded transition-colors',
              isTraining
                ? 'bg-yellow-500/20 text-yellow-400 cursor-wait'
                : 'bg-green-500 text-white hover:bg-green-600'
            )}
          >
            {isTraining ? '...' : '🎓'}
          </button>
        )}
      </div>
    </div>
  )
}

export default function DiscoveryPanel({ onClose, onViewDetail }) {
  const { language } = useSettingsStore()
  const { discovery, isLoading, refetch } = useDiscovery()
  const trainMutation = useTrainTicker()
  const [trainingStatus, setTrainingStatus] = useState({})
  const tr = t(language)

  const handleTrain = async (ticker) => {
    setTrainingStatus(prev => ({ ...prev, [ticker]: 'training' }))
    try {
      await trainMutation.mutateAsync(ticker)
      setTrainingStatus(prev => ({ ...prev, [ticker]: 'queued' }))
    } catch {
      setTrainingStatus(prev => ({ ...prev, [ticker]: 'error' }))
    }
  }

  const handleTrainAll = async () => {
    const untrained = discovery?.new_gainers?.filter(g => !g.is_trained) || []
    for (const gainer of untrained) {
      await handleTrain(gainer.ticker)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') onClose()
  }

  if (isLoading) {
    return (
      <>
        <div className="modal-backdrop" onClick={onClose} onKeyDown={handleKeyDown} role="button" tabIndex={0} aria-label="Close" />
        <div className="fixed right-0 top-0 bottom-0 w-[450px] bg-surface border-l border-surface-light shadow-2xl z-50 flex items-center justify-center">
          <div className="spinner" />
        </div>
      </>
    )
  }

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
        className="fixed right-0 top-0 bottom-0 w-full max-w-[450px] bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between z-10">
          <h2 className="text-xl font-bold flex items-center gap-2">
            🔍 {tr('discovery.title')}
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => refetch()}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              {tr('discovery.refresh')}
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
        <div className="p-6 space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-3 gap-3">
            <StatCard
              label={tr('discovery.totalTickers')}
              value={discovery?.summary?.total_tickers || 0}
            />
            <StatCard
              label={tr('discovery.trainedTickers')}
              value={discovery?.summary?.trained_tickers || 0}
            />
            <StatCard
              label={tr('discovery.coverage')}
              value={`${discovery?.summary?.model_coverage?.toFixed(1) || 0}%`}
              highlight
            />
          </div>

          {/* New Gainers */}
          <section>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              🔥 {tr('discovery.newGainers')}
              <span className="text-sm text-gray-400">
                ({discovery?.new_gainers?.length || 0})
              </span>
            </h3>
            <div className="space-y-2">
              {discovery?.new_gainers?.map((gainer) => (
                <TickerRow
                  key={gainer.ticker}
                  ticker={gainer}
                  trainingStatus={trainingStatus[gainer.ticker]}
                  onTrain={() => handleTrain(gainer.ticker)}
                  onViewDetail={onViewDetail}
                />
              ))}
              {(!discovery?.new_gainers || discovery.new_gainers.length === 0) && (
                <div className="text-center py-4 text-gray-500">
                  {tr('discovery.noNewGainers')}
                </div>
              )}
            </div>
          </section>

          {/* New Volume Top */}
          <section>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              📊 {tr('discovery.newVolumeTop')}
              <span className="text-sm text-gray-400">
                ({discovery?.new_volume_top?.length || 0})
              </span>
            </h3>
            <div className="space-y-2">
              {discovery?.new_volume_top?.map((ticker) => (
                <TickerRow
                  key={ticker.ticker}
                  ticker={ticker}
                  trainingStatus={trainingStatus[ticker.ticker]}
                  onTrain={() => handleTrain(ticker.ticker)}
                  onViewDetail={onViewDetail}
                />
              ))}
              {(!discovery?.new_volume_top || discovery.new_volume_top.length === 0) && (
                <div className="text-center py-4 text-gray-500">
                  {tr('discovery.noNewVolume')}
                </div>
              )}
            </div>
          </section>

          {/* Training Queue */}
          {discovery?.training_queue?.length > 0 && (
            <section>
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                📋 {tr('discovery.trainingQueue')}
                <span className="text-sm text-gray-400">
                  ({discovery.training_queue.length})
                </span>
              </h3>
              <div className="bg-surface-light rounded-lg p-3 space-y-2">
                {discovery.training_queue.map((item, index) => (
                  <div
                    key={item.ticker}
                    className="flex items-center justify-between text-sm"
                  >
                    <span>
                      {index + 1}. <span className="font-bold">{item.ticker}</span>
                      <span className="text-gray-400 ml-2">
                        {item.status === 'training' ? tr('discovery.training') : tr('discovery.pending')}
                      </span>
                    </span>
                    <span className="text-gray-400">
                      ~{Math.ceil(item.estimated_time / 60)}{tr('discovery.minutes')}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Train All Button */}
          {discovery?.new_gainers?.some(g => !g.is_trained) && (
            <button
              onClick={handleTrainAll}
              className="w-full py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors"
            >
              🎓 {tr('discovery.trainAll')}
            </button>
          )}
        </div>
      </div>
    </>
  )
}
