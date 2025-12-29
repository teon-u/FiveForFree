import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'
import { t } from '../i18n'
import { useSettingsStore } from '../stores/settingsStore'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function IntegratedBacktest() {
  const language = useSettingsStore((state) => state.language)
  const tr = t(language)
  // State for tickers and selection
  const [availableTickers, setAvailableTickers] = useState([])
  const [selectedTickers, setSelectedTickers] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // State for options
  const [options, setOptions] = useState({
    initial_capital: 100000,
    entry_threshold: 0.6,
    stop_loss: { enabled: true, rate: -0.02 },
    take_profit: { enabled: true, rate: 0.01 },
    max_hold_hours: 1,
    max_drawdown: 0.2,
    market_hours_only: true,
    max_concurrent_positions: 5
  })

  // State for results
  const [results, setResults] = useState(null)
  const [isRunning, setIsRunning] = useState(false)

  // State for trades table pagination and sorting
  const [tradesPage, setTradesPage] = useState(1)
  const [tradesPerPage] = useState(20)
  const [sortConfig, setSortConfig] = useState({ key: 'entry_time', direction: 'desc' })

  // State for daily returns
  const [dailyReturnsPeriod, setDailyReturnsPeriod] = useState('all') // '7d', '30d', 'all'
  const [showDailyTable, setShowDailyTable] = useState(false)
  const [dailyPage, setDailyPage] = useState(1)
  const [dailyPerPage, setDailyPerPage] = useState(10)
  const [dailySortConfig, setDailySortConfig] = useState({ key: 'date', direction: 'desc' })

  // Fetch available tickers on mount
  useEffect(() => {
    fetchTickers()
  }, [])

  const fetchTickers = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tickers/active`)
      const data = await response.json()
      setAvailableTickers(data.tickers || [])
    } catch (err) {
      console.error('Failed to fetch tickers:', err)
      // Fallback tickers
      setAvailableTickers(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'HOOD', 'AVGO', 'MU'])
    }
  }

  const toggleTicker = (ticker) => {
    setSelectedTickers(prev =>
      prev.includes(ticker)
        ? prev.filter(t => t !== ticker)
        : [...prev, ticker]
    )
  }

  const selectAllTickers = () => {
    setSelectedTickers([...availableTickers])
  }

  const clearAllTickers = () => {
    setSelectedTickers([])
  }

  const runBacktest = async () => {
    if (selectedTickers.length === 0) {
      setError(tr('backtest.errorSelectTicker'))
      return
    }

    setIsRunning(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/backtest/integrated`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tickers: selectedTickers,
          period: {},
          options: options
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Backtest failed')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsRunning(false)
    }
  }

  const formatPercent = (value) => {
    if (value === null || value === undefined) return 'N/A'
    return `${(value * 100).toFixed(2)}%`
  }

  const formatCurrency = (value) => {
    if (value === null || value === undefined) return 'N/A'
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatDateTime = (dateStr) => {
    if (!dateStr) return 'N/A'
    const date = new Date(dateStr)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  // Sort handler for trades table
  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }

  // Computed sorted and paginated trades
  const sortedTrades = useMemo(() => {
    if (!results?.trades) return []

    const sorted = [...results.trades].sort((a, b) => {
      const aVal = a[sortConfig.key]
      const bVal = b[sortConfig.key]

      if (aVal === null || aVal === undefined) return 1
      if (bVal === null || bVal === undefined) return -1

      let comparison = 0
      if (typeof aVal === 'string') {
        comparison = aVal.localeCompare(bVal)
      } else {
        comparison = aVal - bVal
      }

      return sortConfig.direction === 'asc' ? comparison : -comparison
    })

    return sorted
  }, [results?.trades, sortConfig])

  const paginatedTrades = useMemo(() => {
    const start = (tradesPage - 1) * tradesPerPage
    return sortedTrades.slice(start, start + tradesPerPage)
  }, [sortedTrades, tradesPage, tradesPerPage])

  const totalTradePages = Math.ceil((results?.trades?.length || 0) / tradesPerPage)

  // Prepare equity curve data with trade numbers
  const equityCurveData = useMemo(() => {
    if (!results?.equity_curve) return []
    return results.equity_curve.map((point, index) => ({
      ...point,
      tradeNumber: index + 1,
      returnPercent: ((point.equity - options.initial_capital) / options.initial_capital) * 100
    }))
  }, [results?.equity_curve, options.initial_capital])

  // Filter daily returns by period
  const filteredDailyReturns = useMemo(() => {
    if (!results?.daily_returns) return []

    let data = [...results.daily_returns]

    if (dailyReturnsPeriod !== 'all') {
      const days = dailyReturnsPeriod === '7d' ? 7 : 30
      data = data.slice(-days)
    }

    return data
  }, [results?.daily_returns, dailyReturnsPeriod])

  // Daily returns chart data
  const dailyChartData = useMemo(() => {
    return filteredDailyReturns.map(dr => ({
      ...dr,
      displayReturn: dr.daily_return * 100 // Convert to percentage
    }))
  }, [filteredDailyReturns])

  // Sorted daily returns for table
  const sortedDailyReturns = useMemo(() => {
    if (!results?.daily_returns) return []

    const sorted = [...results.daily_returns].sort((a, b) => {
      const aVal = a[dailySortConfig.key]
      const bVal = b[dailySortConfig.key]

      if (aVal === null || aVal === undefined) return 1
      if (bVal === null || bVal === undefined) return -1

      let comparison = 0
      if (typeof aVal === 'string') {
        comparison = aVal.localeCompare(bVal)
      } else {
        comparison = aVal - bVal
      }

      return dailySortConfig.direction === 'asc' ? comparison : -comparison
    })

    return sorted
  }, [results?.daily_returns, dailySortConfig])

  const paginatedDailyReturns = useMemo(() => {
    const start = (dailyPage - 1) * dailyPerPage
    return sortedDailyReturns.slice(start, start + dailyPerPage)
  }, [sortedDailyReturns, dailyPage, dailyPerPage])

  const totalDailyPages = Math.ceil((results?.daily_returns?.length || 0) / dailyPerPage)

  // Daily sort handler
  const handleDailySort = (key) => {
    setDailySortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">{tr('backtest.title')}</h2>
        <button
          onClick={runBacktest}
          disabled={isRunning || selectedTickers.length === 0}
          className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
            isRunning || selectedTickers.length === 0
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRunning ? tr('backtest.running') : tr('backtest.runBacktest')}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Left Sidebar - Ticker Selection */}
        <div className="col-span-3 bg-surface-light rounded-lg p-4 border border-surface-lighter">
          <h3 className="text-lg font-bold text-white mb-4">{tr('backtest.selectTickers')}</h3>

          <div className="flex gap-2 mb-4">
            <button
              onClick={selectAllTickers}
              className="px-3 py-1 text-sm bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded"
            >
              {tr('backtest.selectAll')}
            </button>
            <button
              onClick={clearAllTickers}
              className="px-3 py-1 text-sm bg-gray-600/20 hover:bg-gray-600/30 text-gray-400 rounded"
            >
              {tr('backtest.clear')}
            </button>
          </div>

          <div className="text-sm text-gray-400 mb-2">
            {tr('backtest.selected')}: {selectedTickers.length} / {availableTickers.length}
          </div>

          <div className="space-y-1 max-h-96 overflow-y-auto">
            {availableTickers.map(ticker => (
              <label
                key={ticker}
                className="flex items-center gap-2 p-2 rounded hover:bg-surface cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedTickers.includes(ticker)}
                  onChange={() => toggleTicker(ticker)}
                  className="w-4 h-4 rounded border-gray-600 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-white">{ticker}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="col-span-9 space-y-6">
          {/* Options Panel */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <h3 className="text-lg font-bold text-white mb-4">{tr('backtest.options')}</h3>

            <div className="grid grid-cols-3 gap-6">
              {/* Stop Loss */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="stop_loss_enabled"
                    checked={options.stop_loss.enabled}
                    onChange={(e) => setOptions(prev => ({
                      ...prev,
                      stop_loss: { ...prev.stop_loss, enabled: e.target.checked }
                    }))}
                    className="w-4 h-4 rounded border-gray-600 text-blue-600"
                  />
                  <label htmlFor="stop_loss_enabled" className="text-white font-medium">
                    {tr('backtest.stopLoss')}
                  </label>
                </div>
                <input
                  type="number"
                  value={options.stop_loss.rate * 100}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    stop_loss: { ...prev.stop_loss, rate: parseFloat(e.target.value) / 100 }
                  }))}
                  disabled={!options.stop_loss.enabled}
                  step="0.5"
                  className="w-full px-3 py-2 bg-surface rounded border border-surface-lighter text-white disabled:opacity-50"
                />
                <span className="text-xs text-gray-400">{tr('backtest.ratePercent')}</span>
              </div>

              {/* Take Profit */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="take_profit_enabled"
                    checked={options.take_profit.enabled}
                    onChange={(e) => setOptions(prev => ({
                      ...prev,
                      take_profit: { ...prev.take_profit, enabled: e.target.checked }
                    }))}
                    className="w-4 h-4 rounded border-gray-600 text-blue-600"
                  />
                  <label htmlFor="take_profit_enabled" className="text-white font-medium">
                    {tr('backtest.takeProfit')}
                  </label>
                </div>
                <input
                  type="number"
                  value={options.take_profit.rate * 100}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    take_profit: { ...prev.take_profit, rate: parseFloat(e.target.value) / 100 }
                  }))}
                  disabled={!options.take_profit.enabled}
                  step="0.5"
                  className="w-full px-3 py-2 bg-surface rounded border border-surface-lighter text-white disabled:opacity-50"
                />
                <span className="text-xs text-gray-400">{tr('backtest.ratePercent')}</span>
              </div>

              {/* Max Drawdown */}
              <div className="space-y-2">
                <label className="text-white font-medium">{tr('backtest.maxDrawdown')}</label>
                <input
                  type="number"
                  value={options.max_drawdown * 100}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    max_drawdown: parseFloat(e.target.value) / 100
                  }))}
                  step="5"
                  className="w-full px-3 py-2 bg-surface rounded border border-surface-lighter text-white"
                />
                <span className="text-xs text-gray-400">{tr('backtest.ratePercent')}</span>
              </div>

              {/* Entry Threshold */}
              <div className="space-y-2">
                <label className="text-white font-medium">{tr('backtest.entryThreshold')}</label>
                <input
                  type="number"
                  value={options.entry_threshold * 100}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    entry_threshold: parseFloat(e.target.value) / 100
                  }))}
                  step="5"
                  min="50"
                  max="95"
                  className="w-full px-3 py-2 bg-surface rounded border border-surface-lighter text-white"
                />
                <span className="text-xs text-gray-400">{tr('backtest.probabilityPercent')}</span>
              </div>

              {/* Max Hold Hours */}
              <div className="space-y-2">
                <label className="text-white font-medium">{tr('backtest.maxHoldTime')}</label>
                <input
                  type="number"
                  value={options.max_hold_hours}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    max_hold_hours: parseFloat(e.target.value)
                  }))}
                  step="0.5"
                  min="0.5"
                  max="8"
                  className="w-full px-3 py-2 bg-surface rounded border border-surface-lighter text-white"
                />
                <span className="text-xs text-gray-400">{tr('backtest.hours')}</span>
              </div>

              {/* Market Hours Only */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="market_hours_only"
                    checked={options.market_hours_only}
                    onChange={(e) => setOptions(prev => ({
                      ...prev,
                      market_hours_only: e.target.checked
                    }))}
                    className="w-4 h-4 rounded border-gray-600 text-blue-600"
                  />
                  <label htmlFor="market_hours_only" className="text-white font-medium">
                    {tr('backtest.marketHoursOnly')}
                  </label>
                </div>
                <span className="text-xs text-gray-400">{tr('backtest.marketHoursDesc')}</span>
              </div>
            </div>
          </div>

          {/* Results */}
          {results && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
                  <p className="text-sm text-gray-400 mb-1">{tr('backtest.totalReturn')}</p>
                  <p className={`text-2xl font-bold ${results.summary.total_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatPercent(results.summary.total_return)}
                  </p>
                </div>
                <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
                  <p className="text-sm text-gray-400 mb-1">{tr('backtest.totalProfit')}</p>
                  <p className={`text-2xl font-bold ${results.summary.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatCurrency(results.summary.total_profit)}
                  </p>
                </div>
                <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
                  <p className="text-sm text-gray-400 mb-1">{tr('backtest.winRate')}</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {formatPercent(results.summary.win_rate)}
                  </p>
                </div>
                <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
                  <p className="text-sm text-gray-400 mb-1">{tr('backtest.maxDrawdown')}</p>
                  <p className="text-2xl font-bold text-yellow-400">
                    {formatPercent(-results.summary.max_drawdown)}
                  </p>
                </div>
              </div>

              {/* Portfolio Summary */}
              {results.portfolio_summary && (
                <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
                  <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                    {tr('backtest.portfolioSummary')}
                  </h3>
                  <div className="grid grid-cols-4 gap-4">
                    {/* Best Day */}
                    <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-green-400 text-lg">+</span>
                        <span className="text-sm text-gray-400">{tr('backtest.bestDay')}</span>
                      </div>
                      <p className="text-xl font-bold text-green-400">
                        {results.portfolio_summary.best_day.return_pct >= 0 ? '+' : ''}{results.portfolio_summary.best_day.return_pct.toFixed(2)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{results.portfolio_summary.best_day.date}</p>
                    </div>

                    {/* Worst Day */}
                    <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-red-400 text-lg">-</span>
                        <span className="text-sm text-gray-400">{tr('backtest.worstDay')}</span>
                      </div>
                      <p className="text-xl font-bold text-red-400">
                        {results.portfolio_summary.worst_day.return_pct.toFixed(2)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{results.portfolio_summary.worst_day.date}</p>
                    </div>

                    {/* Avg Daily Return */}
                    <div className="bg-gray-500/10 rounded-lg p-4 border border-gray-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-blue-400 text-lg">~</span>
                        <span className="text-sm text-gray-400">{tr('backtest.avgDaily')}</span>
                      </div>
                      <p className={`text-xl font-bold ${results.portfolio_summary.avg_daily_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {results.portfolio_summary.avg_daily_return >= 0 ? '+' : ''}{results.portfolio_summary.avg_daily_return.toFixed(2)}%
                      </p>
                    </div>

                    {/* Profit Factor */}
                    <div className="bg-gray-500/10 rounded-lg p-4 border border-gray-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-yellow-400 text-lg">x</span>
                        <span className="text-sm text-gray-400">{tr('backtest.profitFactor')}</span>
                      </div>
                      <p className={`text-xl font-bold ${results.portfolio_summary.profit_factor >= 1 ? 'text-green-400' : 'text-red-400'}`}>
                        {results.portfolio_summary.profit_factor.toFixed(2)}
                      </p>
                    </div>

                    {/* Win Streak */}
                    <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-green-400 text-lg">W</span>
                        <span className="text-sm text-gray-400">{tr('backtest.winStreak')}</span>
                      </div>
                      <p className="text-xl font-bold text-green-400">
                        {results.portfolio_summary.max_win_streak} {tr('backtest.days')}
                      </p>
                    </div>

                    {/* Loss Streak */}
                    <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-red-400 text-lg">L</span>
                        <span className="text-sm text-gray-400">{tr('backtest.lossStreak')}</span>
                      </div>
                      <p className="text-xl font-bold text-red-400">
                        {results.portfolio_summary.max_loss_streak} {tr('backtest.days')}
                      </p>
                    </div>

                    {/* Avg Recovery */}
                    <div className="bg-gray-500/10 rounded-lg p-4 border border-gray-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-blue-400 text-lg">R</span>
                        <span className="text-sm text-gray-400">{tr('backtest.avgRecovery')}</span>
                      </div>
                      <p className="text-xl font-bold text-blue-400">
                        {results.portfolio_summary.avg_recovery_days.toFixed(1)} {tr('backtest.days')}
                      </p>
                    </div>

                    {/* Volatility */}
                    <div className="bg-gray-500/10 rounded-lg p-4 border border-gray-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-purple-400 text-lg">V</span>
                        <span className="text-sm text-gray-400">{tr('backtest.volatility')}</span>
                      </div>
                      <p className="text-xl font-bold text-purple-400">
                        {results.portfolio_summary.volatility.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Daily Returns Chart */}
              {results.daily_returns && results.daily_returns.length > 0 && (
                <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-white">{tr('backtest.dailyReturns')}</h3>
                    <div className="flex gap-2">
                      {['7d', '30d', 'all'].map(period => (
                        <button
                          key={period}
                          onClick={() => setDailyReturnsPeriod(period)}
                          className={`px-3 py-1 text-sm rounded ${
                            dailyReturnsPeriod === period
                              ? 'bg-blue-600 text-white'
                              : 'bg-surface text-gray-400 hover:bg-surface-lighter'
                          }`}
                        >
                          {period === 'all' ? tr('backtest.all') : period.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={dailyChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        dataKey="date"
                        tick={{ fill: '#9CA3AF', fontSize: 11 }}
                        tickFormatter={(val) => val.slice(5)} // MM-DD only
                      />
                      <YAxis
                        tick={{ fill: '#9CA3AF', fontSize: 12 }}
                        tickFormatter={(value) => `${value.toFixed(1)}%`}
                      />
                      <ReferenceLine y={0} stroke="#6B7280" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                        formatter={(value, name, props) => {
                          if (name === 'displayReturn') {
                            return [`${value.toFixed(2)}%`, 'Daily Return']
                          }
                          return [value, name]
                        }}
                        labelFormatter={(label) => `Date: ${label}`}
                        content={({ payload, label }) => {
                          if (!payload || !payload[0]) return null
                          const data = payload[0].payload
                          return (
                            <div className="bg-gray-800 border border-gray-600 rounded p-3 text-sm">
                              <p className="text-white font-semibold mb-2">{label}</p>
                              <p className={data.displayReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
                                Return: {data.displayReturn >= 0 ? '+' : ''}{data.displayReturn.toFixed(2)}%
                              </p>
                              <p className="text-gray-300">Trades: {data.trades}</p>
                              <p className="text-gray-300">W/L: {data.wins}/{data.losses}</p>
                            </div>
                          )
                        }}
                      />
                      <Bar dataKey="displayReturn" name="Daily Return">
                        {dailyChartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.displayReturn >= 0 ? '#22c55e' : '#ef4444'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Toggle for Daily Table */}
                  <div className="mt-4 pt-4 border-t border-surface-lighter">
                    <button
                      onClick={() => setShowDailyTable(!showDailyTable)}
                      className="text-sm text-blue-400 hover:text-blue-300"
                    >
                      {showDailyTable ? tr('backtest.hideDailyDetails') : tr('backtest.showDailyDetails')}
                    </button>
                  </div>

                  {/* Daily Returns Table */}
                  {showDailyTable && (
                    <div className="mt-4">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="text-left text-gray-400 border-b border-surface-lighter">
                              <th
                                className="pb-3 cursor-pointer hover:text-white"
                                onClick={() => handleDailySort('date')}
                              >
                                {tr('backtest.date')} {dailySortConfig.key === 'date' && (dailySortConfig.direction === 'asc' ? '↑' : '↓')}
                              </th>
                              <th
                                className="pb-3 text-right cursor-pointer hover:text-white"
                                onClick={() => handleDailySort('trades')}
                              >
                                {tr('backtest.trades')} {dailySortConfig.key === 'trades' && (dailySortConfig.direction === 'asc' ? '↑' : '↓')}
                              </th>
                              <th className="pb-3 text-right">{tr('backtest.wins')}</th>
                              <th className="pb-3 text-right">{tr('backtest.losses')}</th>
                              <th
                                className="pb-3 text-right cursor-pointer hover:text-white"
                                onClick={() => handleDailySort('win_rate')}
                              >
                                {tr('backtest.winRate')} {dailySortConfig.key === 'win_rate' && (dailySortConfig.direction === 'asc' ? '↑' : '↓')}
                              </th>
                              <th className="pb-3 text-right">{tr('backtest.grossProfit')}</th>
                              <th className="pb-3 text-right">{tr('backtest.grossLoss')}</th>
                              <th
                                className="pb-3 text-right cursor-pointer hover:text-white"
                                onClick={() => handleDailySort('net_pnl')}
                              >
                                {tr('backtest.netPL')} {dailySortConfig.key === 'net_pnl' && (dailySortConfig.direction === 'asc' ? '↑' : '↓')}
                              </th>
                              <th
                                className="pb-3 text-right cursor-pointer hover:text-white"
                                onClick={() => handleDailySort('daily_return')}
                              >
                                {tr('backtest.dailyReturn')} {dailySortConfig.key === 'daily_return' && (dailySortConfig.direction === 'asc' ? '↑' : '↓')}
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {paginatedDailyReturns.map((dr, idx) => (
                              <tr key={dr.date} className="border-b border-surface-lighter/50">
                                <td className="py-3 text-white">{dr.date}</td>
                                <td className="py-3 text-right text-gray-300">{dr.trades}</td>
                                <td className="py-3 text-right text-green-400">{dr.wins}</td>
                                <td className="py-3 text-right text-red-400">{dr.losses}</td>
                                <td className="py-3 text-right">
                                  <span className={dr.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                                    {(dr.win_rate * 100).toFixed(1)}%
                                  </span>
                                </td>
                                <td className="py-3 text-right text-green-400">
                                  ${dr.gross_profit.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </td>
                                <td className="py-3 text-right text-red-400">
                                  ${Math.abs(dr.gross_loss).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </td>
                                <td className={`py-3 text-right font-semibold ${dr.net_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  ${dr.net_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </td>
                                <td className={`py-3 text-right font-semibold ${dr.daily_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {dr.daily_return >= 0 ? '+' : ''}{(dr.daily_return * 100).toFixed(2)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {/* Daily Returns Pagination */}
                      {totalDailyPages > 1 && (
                        <div className="flex items-center justify-between mt-4 pt-4 border-t border-surface-lighter">
                          <div className="flex items-center gap-4">
                            <span className="text-sm text-gray-400">
                              {tr('backtest.page')} {dailyPage} {tr('backtest.of')} {totalDailyPages}
                            </span>
                            <select
                              value={dailyPerPage}
                              onChange={(e) => {
                                setDailyPerPage(parseInt(e.target.value))
                                setDailyPage(1)
                              }}
                              className="bg-surface text-gray-300 rounded px-2 py-1 text-sm"
                            >
                              <option value={10}>10 {tr('backtest.perPage')}</option>
                              <option value={20}>20 {tr('backtest.perPage')}</option>
                              <option value={50}>50 {tr('backtest.perPage')}</option>
                            </select>
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => setDailyPage(p => Math.max(1, p - 1))}
                              disabled={dailyPage === 1}
                              className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                            >
                              {tr('backtest.prev')}
                            </button>
                            <button
                              onClick={() => setDailyPage(p => Math.min(totalDailyPages, p + 1))}
                              disabled={dailyPage === totalDailyPages}
                              className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                            >
                              {tr('backtest.next')}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Equity Curve Chart */}
              <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
                <h3 className="text-lg font-bold text-white mb-4">{tr('backtest.equityCurve')}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={equityCurveData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="tradeNumber"
                      tick={{ fill: '#9CA3AF', fontSize: 12 }}
                      label={{ value: tr('backtest.tradeNumber'), position: 'bottom', fill: '#9CA3AF', offset: -5 }}
                    />
                    <YAxis
                      tick={{ fill: '#9CA3AF', fontSize: 12 }}
                      tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                      label={{ value: tr('backtest.equityDollar'), angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                    />
                    <ReferenceLine y={options.initial_capital} stroke="#6B7280" strokeDasharray="5 5" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                      labelFormatter={(value) => `${tr('backtest.tradeNumber')}${value}`}
                      formatter={(value, name) => {
                        if (name === 'equity') return [formatCurrency(value), tr('backtest.equity')]
                        if (name === 'returnPercent') return [`${value.toFixed(2)}%`, tr('backtest.return')]
                        return [value, name]
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      name={tr('backtest.equity')}
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Ticker Results Table */}
              <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
                <h3 className="text-lg font-bold text-white mb-4">{tr('backtest.resultsByTicker')}</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-surface-lighter">
                        <th className="pb-3">{tr('backtest.ticker')}</th>
                        <th className="pb-3 text-right">{tr('backtest.trades')}</th>
                        <th className="pb-3 text-right">{tr('backtest.wins')}</th>
                        <th className="pb-3 text-right">{tr('backtest.winRate')}</th>
                        <th className="pb-3 text-right">{tr('backtest.return')}</th>
                        <th className="pb-3 text-right">{tr('backtest.contribution')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.by_ticker.map(ticker => (
                        <tr key={ticker.ticker} className="border-b border-surface-lighter/50">
                          <td className="py-3 font-semibold text-white">{ticker.ticker}</td>
                          <td className="py-3 text-right text-gray-300">{ticker.total_trades}</td>
                          <td className="py-3 text-right text-gray-300">{ticker.wins}</td>
                          <td className="py-3 text-right">
                            <span className={ticker.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                              {formatPercent(ticker.win_rate)}
                            </span>
                          </td>
                          <td className="py-3 text-right">
                            <span className={ticker.total_return >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {ticker.total_return.toFixed(2)}%
                            </span>
                          </td>
                          <td className="py-3 text-right text-gray-300">
                            {ticker.contribution.toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Individual Trades Table */}
              {results.trades && results.trades.length > 0 && (
                <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-white">{tr('backtest.tradeHistory')}</h3>
                    <span className="text-sm text-gray-400">
                      {results.trades.length} {tr('backtest.tradesCount')}
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="text-left text-gray-400 border-b border-surface-lighter">
                          <th className="pb-3 w-12">#</th>
                          <th
                            className="pb-3 cursor-pointer hover:text-white"
                            onClick={() => handleSort('ticker')}
                          >
                            {tr('backtest.ticker')} {sortConfig.key === 'ticker' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                          </th>
                          <th
                            className="pb-3 cursor-pointer hover:text-white"
                            onClick={() => handleSort('entry_time')}
                          >
                            {tr('backtest.entryTime')} {sortConfig.key === 'entry_time' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                          </th>
                          <th
                            className="pb-3 cursor-pointer hover:text-white"
                            onClick={() => handleSort('exit_time')}
                          >
                            {tr('backtest.exitTime')} {sortConfig.key === 'exit_time' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="pb-3 text-right">{tr('backtest.entryPrice')}</th>
                          <th className="pb-3 text-right">{tr('backtest.exitPrice')}</th>
                          <th
                            className="pb-3 text-right cursor-pointer hover:text-white"
                            onClick={() => handleSort('profit_pct')}
                          >
                            {tr('backtest.profitPercent')} {sortConfig.key === 'profit_pct' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                          </th>
                          <th
                            className="pb-3 cursor-pointer hover:text-white"
                            onClick={() => handleSort('exit_reason')}
                          >
                            {tr('backtest.exitReason')} {sortConfig.key === 'exit_reason' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="pb-3 text-center">{tr('backtest.result')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedTrades.map((trade, idx) => (
                          <tr
                            key={idx}
                            className="border-b border-surface-lighter/50 hover:bg-surface/50"
                          >
                            <td className="py-3 text-gray-500">
                              {(tradesPage - 1) * tradesPerPage + idx + 1}
                            </td>
                            <td className="py-3 font-semibold text-white">{trade.ticker}</td>
                            <td className="py-3 text-gray-300 text-sm">{formatDateTime(trade.entry_time)}</td>
                            <td className="py-3 text-gray-300 text-sm">{formatDateTime(trade.exit_time)}</td>
                            <td className="py-3 text-right text-gray-300">
                              ${trade.entry_price?.toFixed(2) || 'N/A'}
                            </td>
                            <td className="py-3 text-right text-gray-300">
                              ${trade.exit_price?.toFixed(2) || 'N/A'}
                            </td>
                            <td className={`py-3 text-right font-semibold ${trade.profit_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {trade.profit_pct?.toFixed(2) || '0.00'}%
                            </td>
                            <td className="py-3">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                trade.exit_reason === 'target_hit' ? 'bg-green-500/20 text-green-400' :
                                trade.exit_reason === 'stop_loss' ? 'bg-red-500/20 text-red-400' :
                                'bg-yellow-500/20 text-yellow-400'
                              }`}>
                                {trade.exit_reason || 'unknown'}
                              </span>
                            </td>
                            <td className="py-3 text-center">
                              {trade.is_win ? (
                                <span className="text-green-400 font-bold">W</span>
                              ) : (
                                <span className="text-red-400 font-bold">L</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination */}
                  {totalTradePages > 1 && (
                    <div className="flex items-center justify-between mt-4 pt-4 border-t border-surface-lighter">
                      <span className="text-sm text-gray-400">
                        {tr('backtest.page')} {tradesPage} {tr('backtest.of')} {totalTradePages}
                      </span>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setTradesPage(1)}
                          disabled={tradesPage === 1}
                          className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                        >
                          {tr('backtest.first')}
                        </button>
                        <button
                          onClick={() => setTradesPage(p => Math.max(1, p - 1))}
                          disabled={tradesPage === 1}
                          className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                        >
                          {tr('backtest.prev')}
                        </button>
                        <button
                          onClick={() => setTradesPage(p => Math.min(totalTradePages, p + 1))}
                          disabled={tradesPage === totalTradePages}
                          className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                        >
                          {tr('backtest.next')}
                        </button>
                        <button
                          onClick={() => setTradesPage(totalTradePages)}
                          disabled={tradesPage === totalTradePages}
                          className="px-3 py-1 text-sm bg-surface rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-surface-lighter"
                        >
                          {tr('backtest.last')}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
