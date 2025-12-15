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
    <div className="relative">
      {/* Horizontal scroll container */}
      <div className="overflow-x-auto custom-scrollbar pb-4">
        <div className="inline-flex gap-4 min-w-full">
          {predictions.map((prediction) => (
            <div key={prediction.ticker} className="flex-shrink-0 w-48">
              <TickerCard
                prediction={prediction}
                onClick={onTickerClick}
                onDetailClick={onDetailClick}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Scroll indicators */}
      <div className="absolute right-0 top-0 bottom-4 w-16 bg-gradient-to-l from-background to-transparent pointer-events-none" />
    </div>
  )
}
