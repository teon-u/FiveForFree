import { useState, useMemo, memo } from 'react'
import clsx from 'clsx'
import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Area,
  ComposedChart
} from 'recharts'
import { usePriceHistory } from '../hooks/usePriceHistory'

const INTERVALS = [
  { key: '1m', label: '1분' },
  { key: '5m', label: '5분' },
  { key: '15m', label: '15분' },
  { key: '1h', label: '1시간' },
  { key: '1d', label: '1일' }
]

const PERIODS = [
  { key: '1D', label: '1D' },
  { key: '1W', label: '1W' },
  { key: '1M', label: '1M' },
  { key: '3M', label: '3M' }
]

// Memoized chart component for performance
const PriceChart = memo(function PriceChart({ data, showMA5, showMA20, showBB }) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        No data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="displayTime"
          stroke="#9ca3af"
          fontSize={11}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          stroke="#9ca3af"
          fontSize={11}
          tickLine={false}
          domain={['auto', 'auto']}
          tickFormatter={(value) => `$${value.toFixed(0)}`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '8px'
          }}
          labelStyle={{ color: '#9ca3af', marginBottom: '4px' }}
          formatter={(value, name) => {
            const labels = {
              close: 'Price',
              ma5: 'MA5',
              ma20: 'MA20',
              upper: 'BB Upper',
              lower: 'BB Lower'
            }
            return [`$${value?.toFixed(2) || 'N/A'}`, labels[name] || name]
          }}
        />

        {/* Bollinger Bands */}
        {showBB && (
          <>
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="#a855f7"
              fillOpacity={0.1}
            />
            <Line
              type="monotone"
              dataKey="upper"
              stroke="#a855f7"
              strokeDasharray="5 5"
              strokeWidth={1}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="lower"
              stroke="#a855f7"
              strokeDasharray="5 5"
              strokeWidth={1}
              dot={false}
            />
          </>
        )}

        {/* Moving Averages */}
        {showMA5 && (
          <Line
            type="monotone"
            dataKey="ma5"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
          />
        )}
        {showMA20 && (
          <Line
            type="monotone"
            dataKey="ma20"
            stroke="#eab308"
            strokeWidth={1.5}
            dot={false}
          />
        )}

        {/* Price Line */}
        <Line
          type="monotone"
          dataKey="close"
          stroke="#22c55e"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#22c55e' }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
})

function InfoCard({ label, value, valueClass = '' }) {
  return (
    <div className="bg-surface-light rounded-lg p-3 min-w-[80px]">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={clsx('text-sm md:text-lg font-bold truncate', valueClass)}>
        {value || '-'}
      </div>
    </div>
  )
}

function formatVolume(volume) {
  if (!volume) return '-'
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`
  return volume.toString()
}

export default function ChartModal({ ticker, prediction, onClose }) {
  const [interval, setInterval] = useState('1m')
  const [period, setPeriod] = useState('1D')
  const [showMA5, setShowMA5] = useState(true)
  const [showMA20, setShowMA20] = useState(true)
  const [showBB, setShowBB] = useState(false)

  const { data: historyData, isLoading, error } = usePriceHistory(ticker, interval, period)

  const chartData = useMemo(() => {
    return historyData?.data || []
  }, [historyData])

  // Close on Escape key
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') onClose()
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="modal-backdrop"
        onClick={onClose}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-label="Close modal"
      />

      {/* Modal */}
      <div
        className="fixed inset-2 md:inset-4 lg:inset-8 bg-surface rounded-2xl shadow-2xl z-50 flex flex-col max-h-[95vh] overflow-hidden"
        role="dialog"
        aria-modal="true"
        aria-labelledby="chart-modal-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-light shrink-0">
          <div>
            <h2 id="chart-modal-title" className="text-lg md:text-xl font-bold">
              {ticker}
            </h2>
            <p className="text-sm text-gray-400 hidden sm:block">
              Price Chart
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl p-2 hover:bg-surface-light rounded-lg transition-colors"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2 md:gap-4 p-3 md:p-4 border-b border-surface-light shrink-0 overflow-x-auto">
          {/* Interval Buttons */}
          <div className="flex gap-1">
            {INTERVALS.map((int) => (
              <button
                key={int.key}
                onClick={() => setInterval(int.key)}
                className={clsx(
                  'px-2 md:px-3 py-1 rounded text-xs md:text-sm transition-colors',
                  interval === int.key
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                )}
              >
                {int.label}
              </button>
            ))}
          </div>

          <div className="w-px h-6 bg-surface-light hidden md:block" />

          {/* Period Buttons */}
          <div className="flex gap-1">
            {PERIODS.map((p) => (
              <button
                key={p.key}
                onClick={() => setPeriod(p.key)}
                className={clsx(
                  'px-2 md:px-3 py-1 rounded text-xs md:text-sm transition-colors',
                  period === p.key
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                )}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="flex-1" />

          {/* Indicator Toggles */}
          <div className="flex gap-1 md:gap-2 text-xs md:text-sm">
            <button
              onClick={() => setShowMA5(!showMA5)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showMA5 ? 'bg-blue-500/30 text-blue-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              MA5
            </button>
            <button
              onClick={() => setShowMA20(!showMA20)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showMA20 ? 'bg-yellow-500/30 text-yellow-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              MA20
            </button>
            <button
              onClick={() => setShowBB(!showBB)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showBB ? 'bg-purple-500/30 text-purple-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              BB
            </button>
          </div>
        </div>

        {/* Chart Area */}
        <div className="flex-1 p-2 md:p-4 min-h-[200px]">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="spinner" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full text-red-400">
              Failed to load chart data
            </div>
          ) : (
            <PriceChart
              data={chartData}
              showMA5={showMA5}
              showMA20={showMA20}
              showBB={showBB}
            />
          )}
        </div>

        {/* Bottom Info */}
        <div className="p-3 md:p-4 border-t border-surface-light grid grid-cols-3 md:grid-cols-6 gap-2 md:gap-4 shrink-0">
          <InfoCard
            label="Current"
            value={prediction?.current_price ? `$${prediction.current_price.toFixed(2)}` : null}
          />
          <InfoCard
            label="Change"
            value={prediction?.change_percent != null
              ? `${prediction.change_percent >= 0 ? '+' : ''}${prediction.change_percent.toFixed(2)}%`
              : null}
            valueClass={prediction?.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}
          />
          <InfoCard
            label="Volume"
            value={formatVolume(prediction?.volume)}
          />
          <InfoCard
            label="Prediction"
            value={prediction?.probability
              ? `${prediction.probability.toFixed(0)}% ${prediction.direction === 'up' ? '↑' : '↓'}`
              : null}
            valueClass={prediction?.direction === 'up' ? 'text-green-400' : 'text-red-400'}
          />
          <InfoCard
            label="Model"
            value={prediction?.best_model?.toUpperCase()}
          />
          <InfoCard
            label="Grade"
            value={prediction?.practicality_grade}
            valueClass={
              prediction?.practicality_grade === 'A' ? 'text-green-400' :
              prediction?.practicality_grade === 'B' ? 'text-blue-400' :
              prediction?.practicality_grade === 'C' ? 'text-yellow-400' : 'text-gray-400'
            }
          />
        </div>
      </div>
    </>
  )
}
