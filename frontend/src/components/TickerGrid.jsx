import TickerCard from './TickerCard'

export default function TickerGrid({ predictions, onTickerClick, onDetailClick }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <div className="text-4xl mb-2">📭</div>
        <p>No predictions available</p>
      </div>
    )
  }

  return (
    <div className="ticker-grid-container">
      {/* Vertical scroll grid - 4 columns minimum, responsive */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
        {predictions.map((prediction) => (
          <TickerCard
            key={prediction.ticker}
            prediction={prediction}
            onClick={onTickerClick}
            onDetailClick={onDetailClick}
          />
        ))}
      </div>
    </div>
  )
}
