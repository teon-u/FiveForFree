import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'

export default function FinancialTab({ data, tr }) {
  const {
    equity_curve,
    risk_metrics,
    trade_metrics,
    trade_distribution,
    recent_trades
  } = data
  // Use translation if provided, otherwise use default English
  const t = tr || ((key) => key.split('.').pop())

  // Format equity curve data for chart
  const equityData = equity_curve.map(point => ({
    time: new Date(point.timestamp).toLocaleTimeString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }),
    equity: point.equity
  }))

  // Calculate profit/loss for color coding
  const initialEquity = equity_curve.length > 0 ? equity_curve[0].equity : 100
  const finalEquity = equity_curve.length > 0 ? equity_curve[equity_curve.length - 1].equity : 100
  const totalPnL = finalEquity - initialEquity
  const totalPnLPct = ((finalEquity - initialEquity) / initialEquity) * 100

  // Get risk level based on Sharpe ratio
  const getRiskLevel = (sharpe) => {
    if (sharpe >= 2.0) return { label: 'Excellent', color: 'text-green-400' }
    if (sharpe >= 1.0) return { label: 'Good', color: 'text-blue-400' }
    if (sharpe >= 0.5) return { label: 'Fair', color: 'text-yellow-400' }
    return { label: 'Poor', color: 'text-red-400' }
  }

  const riskLevel = getRiskLevel(risk_metrics.sharpe_ratio)

  // Format trade distribution for bar chart
  const distributionData = trade_distribution.map(bin => ({
    range: bin.range,
    count: bin.count,
    midpoint: (bin.min + bin.max) / 2
  }))

  return (
    <div className="space-y-6">
      {/* Equity Curve Section */}
      <div className="bg-surface-light rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">{t('financial.equityCurve')}</h3>
            <p className="text-sm text-gray-400">{t('financial.simulatedBacktest')}</p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalPnL >= 0 ? '+' : ''}{totalPnLPct.toFixed(2)}%
            </div>
            <div className="text-sm text-gray-400">{t('financial.totalReturn')}</div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={equityData}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={totalPnL >= 0 ? '#10B981' : '#EF4444'}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={totalPnL >= 0 ? '#10B981' : '#EF4444'}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="time"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 10 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
              label={{
                value: 'Equity ($)',
                angle: -90,
                position: 'insideLeft',
                style: { fill: '#9CA3AF' }
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
              labelStyle={{ color: '#F3F4F6' }}
              itemStyle={{ color: totalPnL >= 0 ? '#10B981' : '#EF4444' }}
            />
            <Area
              type="monotone"
              dataKey="equity"
              stroke={totalPnL >= 0 ? '#10B981' : '#EF4444'}
              strokeWidth={2}
              fill="url(#equityGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Risk Metrics */}
        <div className="bg-surface-light rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">{t('financial.riskMetrics')}</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-400">Sharpe Ratio</div>
                <div className={`text-xl font-bold ${riskLevel.color}`}>
                  {risk_metrics.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                riskLevel.label === 'Excellent' ? 'bg-green-600/20 text-green-400' :
                riskLevel.label === 'Good' ? 'bg-blue-600/20 text-blue-400' :
                riskLevel.label === 'Fair' ? 'bg-yellow-600/20 text-yellow-400' :
                'bg-red-600/20 text-red-400'
              }`}>
                {riskLevel.label}
              </div>
            </div>

            <div className="border-t border-surface pt-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-gray-400">Sortino Ratio</div>
                  <div className="text-lg font-semibold text-white">
                    {risk_metrics.sortino_ratio.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Calmar Ratio</div>
                  <div className="text-lg font-semibold text-white">
                    {risk_metrics.calmar_ratio.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-surface pt-4">
              <div className="text-xs text-gray-400 mb-2">Max Drawdown</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-surface rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full transition-all"
                    style={{ width: `${Math.min(risk_metrics.max_drawdown, 100)}%` }}
                  />
                </div>
                <div className="text-lg font-semibold text-red-400 w-20 text-right">
                  {risk_metrics.max_drawdown.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Trade Metrics */}
        <div className="bg-surface-light rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">{t('financial.tradeStats')}</h3>

          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-gray-400">Total Trades</div>
                <div className="text-xl font-bold text-white">
                  {trade_metrics.total_trades}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400">Win Rate</div>
                <div className={`text-xl font-bold ${
                  trade_metrics.win_rate >= 0.7 ? 'text-green-400' :
                  trade_metrics.win_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {(trade_metrics.win_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400">Profit Factor</div>
                <div className="text-xl font-bold text-blue-400">
                  {trade_metrics.profit_factor.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="border-t border-surface pt-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-gray-400">Avg Win</div>
                  <div className="text-lg font-semibold text-green-400">
                    +{trade_metrics.avg_win.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Avg Loss</div>
                  <div className="text-lg font-semibold text-red-400">
                    {trade_metrics.avg_loss.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Best Trade</div>
                  <div className="text-lg font-semibold text-green-400">
                    +{trade_metrics.best_trade.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Worst Trade</div>
                  <div className="text-lg font-semibold text-red-400">
                    {trade_metrics.worst_trade.toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-surface pt-4">
              <div className="text-xs text-gray-400 mb-2">Average Return</div>
              <div className={`text-2xl font-bold ${
                trade_metrics.avg_return >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {trade_metrics.avg_return >= 0 ? '+' : ''}{trade_metrics.avg_return.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trade Distribution */}
      {distributionData.length > 0 && (
        <div className="bg-surface-light rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">{t('financial.tradeDistribution')}</h3>
          <p className="text-sm text-gray-400 mb-4">
            {t('financial.distributionDesc')}
          </p>

          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={distributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="range"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF', fontSize: 10 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                label={{
                  value: 'Count',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fill: '#9CA3AF' }
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {distributionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.midpoint >= 0 ? '#10B981' : '#EF4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent Trades */}
      {recent_trades.length > 0 && (
        <div className="bg-surface-light rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">{t('financial.recentTrades')}</h3>
          <p className="text-sm text-gray-400 mb-4">
            {t('financial.lastNTrades').replace('{n}', recent_trades.length)}
          </p>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-surface">
                  <th className="text-left text-xs font-medium text-gray-400 pb-2">Time</th>
                  <th className="text-right text-xs font-medium text-gray-400 pb-2">Return</th>
                  <th className="text-right text-xs font-medium text-gray-400 pb-2">Probability</th>
                  <th className="text-center text-xs font-medium text-gray-400 pb-2">Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface">
                {recent_trades.map((trade, idx) => (
                  <tr key={idx}>
                    <td className="py-2 text-sm text-gray-300">
                      {new Date(trade.timestamp).toLocaleString([], {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </td>
                    <td className={`py-2 text-right text-sm font-semibold ${
                      trade.return_pct >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {trade.return_pct >= 0 ? '+' : ''}{trade.return_pct.toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sm text-gray-300">
                      {(trade.probability * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 text-center">
                      <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                        trade.is_win
                          ? 'bg-green-600/20 text-green-400'
                          : 'bg-red-600/20 text-red-400'
                      }`}>
                        {trade.is_win ? 'Win' : 'Loss'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Info Note */}
      <div className="bg-blue-600/10 border border-blue-500/30 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="text-blue-400 text-xl">ℹ️</div>
          <div>
            <div className="text-sm font-medium text-blue-400 mb-1">{t('financial.simulatedNote')}</div>
            <div className="text-xs text-gray-400">
              {t('financial.simulatedDesc')}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
