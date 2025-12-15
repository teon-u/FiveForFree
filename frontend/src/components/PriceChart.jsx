import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts'

export default function PriceChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-surface-light rounded-lg p-8 text-center text-gray-400">
        <div className="text-4xl mb-2">📊</div>
        <p>No price data available</p>
      </div>
    )
  }

  // Format time for display
  const formatTime = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-surface border border-surface-light rounded-lg p-3 shadow-xl">
          <p className="text-sm text-gray-400 mb-1">{formatTime(data.timestamp)}</p>
          <div className="space-y-1">
            <p className="text-sm">
              <span className="text-gray-400">Open:</span>
              <span className="ml-2 font-semibold">${data.open?.toFixed(2)}</span>
            </p>
            <p className="text-sm">
              <span className="text-gray-400">High:</span>
              <span className="ml-2 font-semibold text-green-400">${data.high?.toFixed(2)}</span>
            </p>
            <p className="text-sm">
              <span className="text-gray-400">Low:</span>
              <span className="ml-2 font-semibold text-red-400">${data.low?.toFixed(2)}</span>
            </p>
            <p className="text-sm">
              <span className="text-gray-400">Close:</span>
              <span className="ml-2 font-semibold">${data.close?.toFixed(2)}</span>
            </p>
            <p className="text-sm">
              <span className="text-gray-400">Volume:</span>
              <span className="ml-2 font-semibold">{data.volume?.toLocaleString()}</span>
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="bg-surface-light rounded-lg p-4">
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <defs>
            <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatTime}
            stroke="#64748b"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            yAxisId="price"
            stroke="#64748b"
            style={{ fontSize: '12px' }}
            domain={['auto', 'auto']}
          />
          <YAxis
            yAxisId="volume"
            orientation="right"
            stroke="#64748b"
            style={{ fontSize: '12px' }}
            tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Volume bars */}
          <Area
            yAxisId="volume"
            type="monotone"
            dataKey="volume"
            fill="url(#colorVolume)"
            stroke="none"
          />

          {/* Price line */}
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
