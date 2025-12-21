import { useMemo } from 'react'

/**
 * Sparkline - Small inline chart for TickerCard
 *
 * @param {Object} props
 * @param {number[]} props.data - Price data points (60 points for 60 minutes)
 * @param {string} props.direction - 'up' or 'down'
 * @param {string|number} props.width - SVG width (default: '100%')
 * @param {number} props.height - SVG height (default: 32)
 */
export default function Sparkline({
  data,
  direction = 'up',
  width = '100%',
  height = 32
}) {
  // Normalize data to fit within height
  const normalized = useMemo(() => {
    if (!data || data.length === 0) return []
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1
    // Add padding (10%)
    const padding = height * 0.1
    const usableHeight = height - (padding * 2)
    return data.map(v => padding + ((v - min) / range) * usableHeight)
  }, [data, height])

  // Generate SVG path
  const pathD = useMemo(() => {
    if (normalized.length === 0) return ''
    const step = 100 / (normalized.length - 1)
    return normalized.reduce((acc, y, i) => {
      const x = i * step
      const yPos = height - y
      return acc + (i === 0 ? `M ${x},${yPos}` : ` L ${x},${yPos}`)
    }, '')
  }, [normalized, height])

  // Area path (for gradient fill)
  const areaD = useMemo(() => {
    if (!pathD) return ''
    return pathD + ` L 100,${height} L 0,${height} Z`
  }, [pathD, height])

  // Colors based on direction
  const color = direction === 'up' ? '#22c55e' : '#ef4444'
  const gradientId = `sparkline-gradient-${direction}-${Math.random().toString(36).substr(2, 9)}`

  if (!data || data.length === 0) {
    return (
      <div
        style={{ width, height }}
        className="bg-surface-light/30 rounded animate-pulse"
      />
    )
  }

  return (
    <svg
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      width={width}
      height={height}
      className="overflow-visible"
      aria-label={`Price trend: ${direction === 'up' ? 'upward' : 'downward'}`}
      role="img"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Gradient fill area */}
      <path
        d={areaD}
        fill={`url(#${gradientId})`}
        className="transition-all duration-500"
      />

      {/* Line */}
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-all duration-500"
      />

      {/* Current price dot */}
      {normalized.length > 0 && (
        <circle
          cx="100"
          cy={height - normalized[normalized.length - 1]}
          r="2"
          fill={color}
          className="animate-pulse"
        />
      )}
    </svg>
  )
}
