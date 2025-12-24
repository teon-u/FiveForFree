import { useMemo } from 'react'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'

const labels = {
  ko: {
    totalTickers: '총 종목',
    aGradeTickers: 'A등급 종목',
    avgGain: '평균 상승률',
    marketSentiment: '시장 분위기',
    bullish: '강세',
    bearish: '약세',
    neutral: '중립',
    upDown: '상승/하락',
    lastUpdate: '마지막 업데이트'
  },
  en: {
    totalTickers: 'Total Tickers',
    aGradeTickers: 'A-Grade',
    avgGain: 'Avg Gain',
    marketSentiment: 'Market Sentiment',
    bullish: 'Bullish',
    bearish: 'Bearish',
    neutral: 'Neutral',
    upDown: 'Up/Down',
    lastUpdate: 'Last Update'
  }
}

export default function DashboardSummary({ predictions, lastUpdateTime }) {
  const { language } = useSettingsStore()
  const tr = labels[language] || labels.ko

  // Calculate summary statistics
  const stats = useMemo(() => {
    if (!predictions || predictions.length === 0) {
      return {
        total: 0,
        aGrade: 0,
        avgGain: 0,
        upCount: 0,
        downCount: 0,
        sentiment: 'neutral',
        sentimentRatio: 50
      }
    }

    // Helper to calculate grade from precision
    const getGrade = (precision) => {
      if (precision >= 80) return 'A'
      if (precision >= 70) return 'B'
      if (precision >= 60) return 'C'
      return 'D'
    }

    const total = predictions.length

    // Count A-grade tickers
    const aGrade = predictions.filter(p => {
      const precision = (p.precision || p.accuracy || 0.5) * 100
      return getGrade(precision) === 'A'
    }).length

    // Calculate average change percent
    const avgGain = predictions.reduce((sum, p) => sum + (p.change_percent || 0), 0) / total

    // Count up/down predictions
    const upCount = predictions.filter(p => p.direction === 'up').length
    const downCount = predictions.filter(p => p.direction === 'down').length

    // Calculate sentiment
    const sentimentRatio = total > 0 ? (upCount / total) * 100 : 50
    let sentiment = 'neutral'
    if (sentimentRatio >= 60) sentiment = 'bullish'
    else if (sentimentRatio <= 40) sentiment = 'bearish'

    return {
      total,
      aGrade,
      avgGain,
      upCount,
      downCount,
      sentiment,
      sentimentRatio
    }
  }, [predictions])

  // Format timestamp
  const formattedTime = useMemo(() => {
    if (!lastUpdateTime) return '--:--:--'
    const date = new Date(lastUpdateTime)
    return date.toLocaleTimeString(language === 'ko' ? 'ko-KR' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }, [lastUpdateTime, language])

  // Sentiment styling
  const getSentimentStyle = () => {
    switch (stats.sentiment) {
      case 'bullish':
        return { bg: 'bg-green-500/20', text: 'text-green-400', icon: '🚀' }
      case 'bearish':
        return { bg: 'bg-red-500/20', text: 'text-red-400', icon: '📉' }
      default:
        return { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: '➖' }
    }
  }

  const sentimentStyle = getSentimentStyle()
  const sentimentLabel = {
    bullish: tr.bullish,
    bearish: tr.bearish,
    neutral: tr.neutral
  }[stats.sentiment]

  return (
    <div className="bg-gradient-to-r from-gray-800/80 to-gray-900/80 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4 mb-6">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {/* Total Tickers */}
        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
          <div className="text-2xl font-bold text-white">{stats.total}</div>
          <div className="text-xs text-gray-400 mt-1">{tr.totalTickers}</div>
        </div>

        {/* A-Grade Tickers */}
        <div className="text-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
          <div className="text-2xl font-bold text-green-400">
            {stats.aGrade}
            <span className="text-sm ml-1">⭐</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">{tr.aGradeTickers}</div>
        </div>

        {/* Average Gain */}
        <div className="text-center p-3 bg-gray-800/50 rounded-lg">
          <div className={`text-2xl font-bold ${stats.avgGain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {stats.avgGain >= 0 ? '+' : ''}{stats.avgGain.toFixed(2)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">{tr.avgGain}</div>
        </div>

        {/* Market Sentiment */}
        <div className={`text-center p-3 rounded-lg ${sentimentStyle.bg}`}>
          <div className={`text-xl font-bold ${sentimentStyle.text}`}>
            {sentimentStyle.icon} {sentimentLabel}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {tr.upDown}: {stats.upCount}/{stats.downCount}
          </div>
          {/* Sentiment Bar */}
          <div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 transition-all duration-500"
              style={{ width: `${stats.sentimentRatio}%` }}
            />
          </div>
        </div>

        {/* Last Update */}
        <div className="text-center p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
          <div className="text-lg font-mono text-blue-400">
            {formattedTime}
          </div>
          <div className="text-xs text-gray-400 mt-1">{tr.lastUpdate}</div>
          <div className="mt-1 flex items-center justify-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            <span className="text-[10px] text-green-400">LIVE</span>
          </div>
        </div>
      </div>
    </div>
  )
}
