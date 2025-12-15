import { useEffect } from 'react'
import ModelComparison from './ModelComparison'
import PriceChart from './PriceChart'
import { useModels } from '../hooks/useModels'
import clsx from 'clsx'

export default function PredictionPanel({ ticker, onClose }) {
  const { modelData, priceData, isLoading } = useModels(ticker)

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [onClose])

  if (isLoading) {
    return (
      <>
        <div className="modal-backdrop" onClick={onClose} />
        <div className="modal-content">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="spinner" />
          </div>
        </div>
      </>
    )
  }

  const { current_price, change_percent, best_prediction } = modelData || {}

  return (
    <>
      {/* Backdrop */}
      <div className="modal-backdrop" onClick={onClose} />

      {/* Panel */}
      <div className="modal-content">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h2 className="text-2xl font-bold">{ticker}</h2>
            {current_price && (
              <div className="flex items-center gap-4 mt-1">
                <span className="text-lg text-gray-300">
                  ${current_price.toFixed(2)}
                </span>
                <span className={clsx(
                  'text-sm',
                  change_percent >= 0 ? 'text-green-400' : 'text-red-400'
                )}>
                  {change_percent >= 0 ? '+' : ''}{change_percent?.toFixed(2)}%
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-2xl"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Best Prediction */}
          {best_prediction && (
            <div className="bg-surface-light rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                Best Model Prediction
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-400">Direction</div>
                  <div className={clsx(
                    'text-2xl font-bold',
                    best_prediction.direction === 'up' ? 'text-green-400' : 'text-red-400'
                  )}>
                    {best_prediction.direction === 'up' ? '↑ UP' : '↓ DOWN'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Probability</div>
                  <div className="text-2xl font-bold">
                    {best_prediction.probability.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Model</div>
                  <div className="text-lg font-semibold text-gray-300">
                    {best_prediction.model_name}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Hit Rate (50h)</div>
                  <div className="text-lg font-semibold text-gray-300">
                    {best_prediction.hit_rate.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Probability bars */}
              <div className="mt-4 space-y-2">
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Up Probability</span>
                    <span>{best_prediction.up_prob?.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill bg-green-500"
                      style={{ width: `${best_prediction.up_prob || 0}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Down Probability</span>
                    <span>{best_prediction.down_prob?.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill bg-red-500"
                      style={{ width: `${best_prediction.down_prob || 0}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Model Comparison Table */}
          <div>
            <h3 className="text-lg font-semibold mb-3">
              Model Comparison (50h Backtest)
            </h3>
            <ModelComparison models={modelData?.models} />
          </div>

          {/* Price Chart */}
          <div>
            <h3 className="text-lg font-semibold mb-3">
              60-Minute Price Chart
            </h3>
            <PriceChart data={priceData} />
          </div>
        </div>
      </div>
    </>
  )
}
