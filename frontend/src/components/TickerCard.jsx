import { useEffect } from 'react'
import clsx from 'clsx'
import { usePriceStore } from '../stores/priceStore'

export default function TickerCard({ prediction, onClick, onDetailClick }) {
  const {
    ticker,
    probability,
    direction,
    change_percent,
    best_model,
    hit_rate,
    signal_rate,
    practicality_grade,
  } = prediction

  // Get real-time price from store
  const priceData = usePriceStore((state) => state.prices[ticker])
  const clearPriceChanged = usePriceStore((state) => state.clearPriceChanged)

  // Use real-time change_percent if available, otherwise use prediction data
  const displayChangePercent = priceData?.change_percent ?? change_percent
  const displayPrice = priceData?.price

  // Clear the price changed flag after animation
  useEffect(() => {
    if (priceData?.priceChanged) {
      const timer = setTimeout(() => {
        clearPriceChanged(ticker)
      }, 1000) // Flash animation duration
      return () => clearTimeout(timer)
    }
  }, [priceData?.priceChanged, ticker, clearPriceChanged])

  // Determine card style based on probability and direction
  const getCardStyle = () => {
    if (probability >= 80) {
      return direction === 'up' ? 'strong-up' : 'strong-down'
    } else if (probability >= 70) {
      return direction === 'up' ? 'up' : 'down'
    }
    return 'neutral'
  }

  const cardStyle = getCardStyle()
  const isSignificant = probability >= 70

  // Warning: High probability but low precision (unreliable prediction)
  const isLowReliability = probability >= 70 && hit_rate < 30

  // Format model name for display
  const formatModelName = (modelName) => {
    const shortNames = {
      'xgboost': 'XGB',
      'lightgbm': 'LGBM',
      'lstm': 'LSTM',
      'transformer': 'TFRM',
      'ensemble': 'ENS'
    }
    return shortNames[modelName] || modelName.toUpperCase()
  }

  // Practicality Grade styling
  const getGradeStyle = (grade) => {
    switch (grade) {
      case 'A':
        return 'text-green-400 bg-green-400/20 border-green-400/40'
      case 'B':
        return 'text-blue-400 bg-blue-400/20 border-blue-400/40'
      case 'C':
        return 'text-yellow-400 bg-yellow-400/20 border-yellow-400/40'
      default:
        return 'text-red-400 bg-red-400/20 border-red-400/40'
    }
  }

  return (
    <div
      className={clsx('ticker-card', cardStyle, 'relative')}
    >
      {/* Ticker Symbol */}
      <div className="flex items-start justify-between mb-2">
        <h3 className="text-lg font-bold">{ticker}</h3>
        {isSignificant && (
          <div className="text-xl">
            {direction === 'up' ? '🟢' : '🔴'}
          </div>
        )}
      </div>

      {/* Probability */}
      <div className={clsx('prob-badge mb-2', cardStyle, isLowReliability && 'opacity-60')}>
        {probability.toFixed(0)}% {direction === 'up' ? '↑' : '↓'}
        {isLowReliability && <span className="ml-1 text-yellow-400" title="Low reliability: High probability but low precision">⚠️</span>}
      </div>

      {/* Price & Change Percent */}
      <div className="mb-2 space-y-0.5">
        {/* Real-time price with flash animation */}
        {displayPrice && (
          <div className={clsx(
            'text-sm font-semibold transition-all duration-300',
            priceData?.priceChanged && priceData?.priceDirection === 'up' && 'animate-flash-green',
            priceData?.priceChanged && priceData?.priceDirection === 'down' && 'animate-flash-red',
            !priceData?.priceChanged && 'text-gray-300'
          )}>
            ${displayPrice.toFixed(2)}
          </div>
        )}
        {/* Change percent */}
        <div className={clsx(
          'text-sm',
          displayChangePercent >= 0 ? 'text-green-400' : 'text-red-400'
        )}>
          {displayChangePercent >= 0 ? '+' : ''}{displayChangePercent.toFixed(2)}%
        </div>
      </div>

      {/* Model Info */}
      <div className="text-xs text-gray-400 space-y-1 mb-3">
        <div className="flex items-center justify-between">
          <span>Model:</span>
          <div className="flex items-center gap-1.5">
            <span className="font-semibold text-gray-300">
              {formatModelName(best_model)}
            </span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${getGradeStyle(practicality_grade)}`}>
              {practicality_grade}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span>Precision:</span>
          <span className={`font-semibold ${hit_rate >= 50 ? 'text-green-400' : hit_rate >= 30 ? 'text-blue-400' : hit_rate > 0 ? 'text-yellow-400' : 'text-red-400'}`}>
            {hit_rate.toFixed(0)}%{hit_rate === 0 && ' ⚠️'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Signal:</span>
          <span className={`font-semibold ${signal_rate >= 10 ? 'text-green-400' : signal_rate >= 5 ? 'text-yellow-400' : 'text-red-400'}`}>
            {signal_rate.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-2 mt-auto">
        <button
          onClick={(e) => {
            e.stopPropagation()
            onClick(ticker)
          }}
          className="px-2 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 text-xs font-medium rounded transition-colors border border-blue-500/30"
        >
          📈 차트
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onDetailClick && onDetailClick(ticker)
          }}
          className="px-2 py-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 text-xs font-medium rounded transition-colors border border-purple-500/30"
        >
          🔍 모델
        </button>
      </div>
    </div>
  )
}
