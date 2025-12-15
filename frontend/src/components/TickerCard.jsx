import clsx from 'clsx'

export default function TickerCard({ prediction, onClick }) {
  const {
    ticker,
    probability,
    direction,
    change_percent,
    best_model,
    hit_rate,
  } = prediction

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

  return (
    <div
      onClick={() => onClick(ticker)}
      className={clsx('ticker-card', cardStyle)}
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
      <div className={clsx('prob-badge mb-2', cardStyle)}>
        {probability.toFixed(0)}% {direction === 'up' ? '↑' : '↓'}
      </div>

      {/* Change Percent */}
      <div className={clsx(
        'text-sm mb-2',
        change_percent >= 0 ? 'text-green-400' : 'text-red-400'
      )}>
        {change_percent >= 0 ? '+' : ''}{change_percent.toFixed(2)}%
      </div>

      {/* Model Info */}
      <div className="text-xs text-gray-400 space-y-1">
        <div className="flex items-center justify-between">
          <span>Model:</span>
          <span className="font-semibold text-gray-300">
            {formatModelName(best_model)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Hit Rate:</span>
          <span className="font-semibold text-gray-300">
            {hit_rate.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Hover indicator */}
      <div className="absolute inset-0 rounded-lg bg-white/0 hover:bg-white/5 transition-colors pointer-events-none" />
    </div>
  )
}
