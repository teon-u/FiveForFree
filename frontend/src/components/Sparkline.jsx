import { useMemo, useState, useCallback, useId } from 'react'

/**
 * Sparkline - Interactive mini chart for TickerCard
 *
 * @param {Object} props
 * @param {number[]} props.data - Price data points (60 points for 60 minutes)
 * @param {string} props.direction - 'up' or 'down'
 * @param {string|number} props.width - SVG width (default: '100%')
 * @param {number} props.height - SVG height (default: 32)
 * @param {boolean} props.showMinMax - Show min/max markers (default: true)
 * @param {boolean} props.interactive - Enable hover tooltip (default: true)
 */
export default function Sparkline({
  data,
  direction = 'up',
  width = '100%',
  height = 32,
  showMinMax = true,
  interactive = true
}) {
  const [hoverIndex, setHoverIndex] = useState(null)
  const gradientId = useId()

  // Calculate stats
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null
    const min = Math.min(...data)
    const max = Math.max(...data)
    const minIdx = data.indexOf(min)
    const maxIdx = data.indexOf(max)
    const first = data[0]
    const last = data[data.length - 1]
    const change = ((last - first) / first * 100) || 0
    return { min, max, minIdx, maxIdx, first, last, change }
  }, [data])

  // Normalize data to fit within height
  // Increased padding to prevent H/L markers from being cut off
  const normalized = useMemo(() => {
    if (!data || data.length === 0 || !stats) return []
    const { min, max } = stats
    const range = max - min || 1
    const padding = height * 0.25  // Increased from 0.15 to 0.25 for better H/L marker visibility
    const usableHeight = height - (padding * 2)
    return data.map(v => padding + ((v - min) / range) * usableHeight)
  }, [data, height, stats])

  // Generate SVG path with smooth curve
  const pathD = useMemo(() => {
    if (normalized.length === 0) return ''
    const step = 100 / (normalized.length - 1)

    // Use bezier curves for smoother line
    let path = `M 0,${height - normalized[0]}`
    for (let i = 1; i < normalized.length; i++) {
      const x = i * step
      const yPos = height - normalized[i]
      const prevX = (i - 1) * step
      const prevY = height - normalized[i - 1]
      const cpX = (prevX + x) / 2
      path += ` C ${cpX},${prevY} ${cpX},${yPos} ${x},${yPos}`
    }
    return path
  }, [normalized, height])

  // Area path (for gradient fill)
  const areaD = useMemo(() => {
    if (!pathD) return ''
    return pathD + ` L 100,${height} L 0,${height} Z`
  }, [pathD, height])

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback((e) => {
    if (!interactive || !data) return
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = x / rect.width
    const index = Math.round(percentage * (data.length - 1))
    setHoverIndex(Math.max(0, Math.min(data.length - 1, index)))
  }, [data, interactive])

  const handleMouseLeave = useCallback(() => {
    setHoverIndex(null)
  }, [])

  // Colors based on direction
  const color = direction === 'up' ? '#22c55e' : '#ef4444'
  const minMaxColor = direction === 'up' ? '#16a34a' : '#dc2626'

  if (!data || data.length === 0) {
    return (
      <div
        style={{ width, height }}
        className="bg-surface-light/30 rounded animate-pulse"
      />
    )
  }

  const step = 100 / (normalized.length - 1)

  return (
    <div className="relative" style={{ width, height }}>
      <svg
        viewBox={`0 0 100 ${height}`}
        preserveAspectRatio="none"
        width="100%"
        height={height}
        className="overflow-visible cursor-crosshair"
        aria-label={`Price trend: ${direction === 'up' ? 'upward' : 'downward'}, change: ${stats?.change.toFixed(1)}%`}
        role="img"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.4" />
            <stop offset="100%" stopColor={color} stopOpacity="0.05" />
          </linearGradient>
          <filter id={`${gradientId}-glow`}>
            <feGaussianBlur stdDeviation="1" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Gradient fill area */}
        <path
          d={areaD}
          fill={`url(#${gradientId})`}
          className="transition-all duration-300"
        />

        {/* Line with glow effect */}
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter={`url(#${gradientId}-glow)`}
          className="transition-all duration-300"
        />

        {/* Min/Max markers */}
        {showMinMax && stats && (
          <>
            {/* Min marker (L) */}
            <g>
              <circle
                cx={stats.minIdx * step}
                cy={height - normalized[stats.minIdx]}
                r="2"
                fill={minMaxColor}
                stroke="#1f2937"
                strokeWidth="0.8"
              />
              <text
                x={stats.minIdx * step}
                y={height - normalized[stats.minIdx] + 6}
                fontSize="5"
                fill="#9ca3af"
                textAnchor="middle"
                fontWeight="bold"
              >
                L
              </text>
            </g>
            {/* Max marker (H) */}
            <g>
              <circle
                cx={stats.maxIdx * step}
                cy={height - normalized[stats.maxIdx]}
                r="2"
                fill={minMaxColor}
                stroke="#1f2937"
                strokeWidth="0.8"
              />
              <text
                x={stats.maxIdx * step}
                y={height - normalized[stats.maxIdx] - 3}
                fontSize="5"
                fill="#9ca3af"
                textAnchor="middle"
                fontWeight="bold"
              >
                H
              </text>
            </g>
          </>
        )}

        {/* Hover line and dot */}
        {hoverIndex !== null && (
          <>
            <line
              x1={hoverIndex * step}
              y1={0}
              x2={hoverIndex * step}
              y2={height}
              stroke="#6b7280"
              strokeWidth="0.5"
              strokeDasharray="2 2"
            />
            <circle
              cx={hoverIndex * step}
              cy={height - normalized[hoverIndex]}
              r="3"
              fill={color}
              stroke="white"
              strokeWidth="1.5"
            />
          </>
        )}

        {/* Current price dot (pulsing) */}
        {hoverIndex === null && normalized.length > 0 && (
          <circle
            cx="100"
            cy={height - normalized[normalized.length - 1]}
            r="2.5"
            fill={color}
            className="animate-pulse"
          />
        )}
      </svg>

      {/* Hover tooltip */}
      {interactive && hoverIndex !== null && data[hoverIndex] !== undefined && (
        <div
          className="absolute -top-8 transform -translate-x-1/2 px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-[10px] font-mono text-white whitespace-nowrap z-10 pointer-events-none shadow-lg"
          style={{ left: `${(hoverIndex / (data.length - 1)) * 100}%` }}
        >
          ${data[hoverIndex].toFixed(2)}
        </div>
      )}
    </div>
  )
}
