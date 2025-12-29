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
  ComposedChart,
  Bar,
  Cell,
  ReferenceLine
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

// Calculate RSI from price data
function calculateRSI(data, period = 14) {
  if (!data || data.length < period + 1) return data

  return data.map((item, index) => {
    if (index < period) return { ...item, rsi: null }

    let gains = 0
    let losses = 0

    for (let i = index - period + 1; i <= index; i++) {
      const diff = data[i].close - data[i - 1].close
      if (diff > 0) gains += diff
      else losses -= diff
    }

    const avgGain = gains / period
    const avgLoss = losses / period
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
    const rsi = 100 - (100 / (1 + rs))

    return { ...item, rsi }
  })
}

// Calculate MACD from price data
function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (!data || data.length < slowPeriod) return data

  // Calculate EMA
  const ema = (prices, period) => {
    const k = 2 / (period + 1)
    let ema = prices[0]
    return prices.map((price, i) => {
      if (i === 0) return ema
      ema = price * k + ema * (1 - k)
      return ema
    })
  }

  const closes = data.map(d => d.close)
  const emaFast = ema(closes, fastPeriod)
  const emaSlow = ema(closes, slowPeriod)

  const macdLine = emaFast.map((fast, i) => fast - emaSlow[i])
  const signalLine = ema(macdLine.slice(slowPeriod - 1), signalPeriod)

  return data.map((item, index) => {
    if (index < slowPeriod + signalPeriod - 2) {
      return { ...item, macd: null, signal: null, histogram: null }
    }
    const macdIdx = index - slowPeriod + 1
    const signalIdx = macdIdx - signalPeriod + 1
    const macd = macdLine[index]
    const signal = signalLine[signalIdx] || 0
    const histogram = macd - signal

    return { ...item, macd, signal, histogram }
  })
}

// Volume Chart Component
const VolumeChart = memo(function VolumeChart({ data }) {
  if (!data || data.length === 0) return null

  const maxVolume = Math.max(...data.map(d => d.volume || 0))

  return (
    <ResponsiveContainer width="100%" height={60}>
      <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
        <XAxis dataKey="displayTime" hide />
        <YAxis
          stroke="#9ca3af"
          fontSize={9}
          tickLine={false}
          tickFormatter={(v) => v >= 1e6 ? `${(v/1e6).toFixed(0)}M` : `${(v/1e3).toFixed(0)}K`}
          domain={[0, maxVolume * 1.1]}
          width={40}
        />
        <Bar dataKey="volume" fill="#3b82f6" opacity={0.6}>
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.close >= (data[index - 1]?.close || entry.close) ? '#22c55e' : '#ef4444'}
              opacity={0.5}
            />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  )
})

// RSI Chart Component
const RSIChart = memo(function RSIChart({ data }) {
  if (!data || data.length === 0) return null

  return (
    <ResponsiveContainer width="100%" height={80}>
      <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
        <XAxis dataKey="displayTime" hide />
        <YAxis
          stroke="#9ca3af"
          fontSize={9}
          tickLine={false}
          domain={[0, 100]}
          ticks={[30, 50, 70]}
          width={40}
        />
        <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
        <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.5} />
        <Area
          type="monotone"
          dataKey="rsi"
          stroke="#a855f7"
          fill="#a855f7"
          fillOpacity={0.2}
          strokeWidth={1.5}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
})

// MACD Chart Component
const MACDChart = memo(function MACDChart({ data }) {
  if (!data || data.length === 0) return null

  return (
    <ResponsiveContainer width="100%" height={80}>
      <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
        <XAxis dataKey="displayTime" hide />
        <YAxis stroke="#9ca3af" fontSize={9} tickLine={false} width={40} />
        <ReferenceLine y={0} stroke="#6b7280" strokeWidth={0.5} />
        <Bar dataKey="histogram">
          {data.map((entry, index) => (
            <Cell
              key={`histogram-${index}`}
              fill={(entry.histogram || 0) >= 0 ? '#22c55e' : '#ef4444'}
              opacity={0.6}
            />
          ))}
        </Bar>
        <Line type="monotone" dataKey="macd" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="signal" stroke="#f97316" strokeWidth={1.5} dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
})

// Memoized main chart component for performance
const PriceChart = memo(function PriceChart({ data, showMA5, showMA20, showBB, showVolume }) {
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
          yAxisId="price"
          stroke="#9ca3af"
          fontSize={11}
          tickLine={false}
          domain={['auto', 'auto']}
          tickFormatter={(value) => `$${value.toFixed(0)}`}
        />
        {showVolume && (
          <YAxis
            yAxisId="volume"
            orientation="right"
            stroke="#6b7280"
            fontSize={9}
            tickLine={false}
            tickFormatter={(v) => v >= 1e6 ? `${(v/1e6).toFixed(0)}M` : ''}
          />
        )}
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
              lower: 'BB Lower',
              volume: 'Volume'
            }
            if (name === 'volume') {
              return [value?.toLocaleString() || 'N/A', labels[name]]
            }
            return [`$${value?.toFixed(2) || 'N/A'}`, labels[name] || name]
          }}
        />

        {/* Volume Bars (background) */}
        {showVolume && (
          <Bar yAxisId="volume" dataKey="volume" fill="#3b82f6" opacity={0.15} />
        )}

        {/* Bollinger Bands */}
        {showBB && (
          <>
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="#a855f7"
              fillOpacity={0.1}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="upper"
              stroke="#a855f7"
              strokeDasharray="5 5"
              strokeWidth={1}
              dot={false}
            />
            <Line
              yAxisId="price"
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
            yAxisId="price"
            type="monotone"
            dataKey="ma5"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
          />
        )}
        {showMA20 && (
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="ma20"
            stroke="#eab308"
            strokeWidth={1.5}
            dot={false}
          />
        )}

        {/* Price Line */}
        <Line
          yAxisId="price"
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
  const [showVolume, setShowVolume] = useState(true)
  const [showRSI, setShowRSI] = useState(false)
  const [showMACD, setShowMACD] = useState(false)

  const { data: historyData, isLoading, error } = usePriceHistory(ticker, interval, period)

  const chartData = useMemo(() => {
    return historyData?.data || []
  }, [historyData])

  // Calculate RSI and MACD data
  const rsiData = useMemo(() => {
    if (!showRSI || chartData.length === 0) return []
    return calculateRSI(chartData)
  }, [chartData, showRSI])

  const macdData = useMemo(() => {
    if (!showMACD || chartData.length === 0) return []
    return calculateMACD(chartData)
  }, [chartData, showMACD])

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
            <div className="w-px h-5 bg-surface-light mx-1" />
            <button
              onClick={() => setShowVolume(!showVolume)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showVolume ? 'bg-cyan-500/30 text-cyan-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              VOL
            </button>
            <button
              onClick={() => setShowRSI(!showRSI)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showRSI ? 'bg-purple-500/30 text-purple-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              RSI
            </button>
            <button
              onClick={() => setShowMACD(!showMACD)}
              className={clsx(
                'px-2 py-1 rounded transition-colors',
                showMACD ? 'bg-orange-500/30 text-orange-400' : 'text-gray-500 hover:bg-surface-light'
              )}
            >
              MACD
            </button>
          </div>
        </div>

        {/* Chart Area */}
        <div className="flex-1 p-2 md:p-4 min-h-[200px] flex flex-col overflow-hidden">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="spinner" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full text-red-400">
              Failed to load chart data
            </div>
          ) : (
            <>
              {/* Main Price Chart */}
              <div className={clsx(
                'flex-1 min-h-[150px]',
                (showRSI || showMACD) ? 'max-h-[60%]' : ''
              )}>
                <PriceChart
                  data={chartData}
                  showMA5={showMA5}
                  showMA20={showMA20}
                  showBB={showBB}
                  showVolume={showVolume}
                />
              </div>

              {/* Volume Chart (separate panel) */}
              {showVolume && !showRSI && !showMACD && (
                <div className="mt-1 border-t border-surface-light pt-1">
                  <div className="flex items-center gap-1 mb-1">
                    <span className="text-[10px] text-gray-500 uppercase">Volume</span>
                  </div>
                  <VolumeChart data={chartData} />
                </div>
              )}

              {/* RSI Chart */}
              {showRSI && (
                <div className="mt-1 border-t border-surface-light pt-1">
                  <div className="flex items-center gap-1 mb-1">
                    <span className="text-[10px] text-purple-400 uppercase">RSI (14)</span>
                  </div>
                  <RSIChart data={rsiData} />
                </div>
              )}

              {/* MACD Chart */}
              {showMACD && (
                <div className="mt-1 border-t border-surface-light pt-1">
                  <div className="flex items-center gap-1 mb-1">
                    <span className="text-[10px] text-orange-400 uppercase">MACD (12, 26, 9)</span>
                  </div>
                  <MACDChart data={macdData} />
                </div>
              )}
            </>
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
