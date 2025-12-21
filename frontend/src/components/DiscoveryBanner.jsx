import { useState } from 'react'
import { useDiscovery } from '../hooks/useDiscovery'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'

export default function DiscoveryBanner({ onOpenPanel }) {
  const { language } = useSettingsStore()
  const { discovery, isLoading } = useDiscovery()
  const [dismissed, setDismissed] = useState(false)
  const tr = t(language)

  // Don't show if dismissed, loading, or no new gainers
  if (dismissed || isLoading || !discovery?.new_gainers?.length) {
    return null
  }

  const newGainers = discovery.new_gainers.filter(g => !g.is_trained)
  if (newGainers.length === 0) return null

  const topGainers = newGainers.slice(0, 3)
  const moreCount = newGainers.length - 3

  return (
    <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-3 mb-4 flex items-center justify-between flex-wrap gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xl">🔔</span>
        <span className="text-yellow-400 font-medium">
          {tr('discovery.newFound')}
        </span>
        <span className="text-white">
          {topGainers.map((g, i) => (
            <span key={g.ticker}>
              {i > 0 && ', '}
              <span className="font-bold">{g.ticker}</span>
              <span className="text-green-400 ml-1">+{g.change_percent.toFixed(1)}%</span>
            </span>
          ))}
          {moreCount > 0 && (
            <span className="text-gray-400"> {tr('discovery.andMore', { count: moreCount })}</span>
          )}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onOpenPanel}
          className="px-3 py-1.5 bg-yellow-500 text-black rounded-lg text-sm font-medium hover:bg-yellow-400 transition-colors"
        >
          {tr('discovery.viewDetails')}
        </button>
        <button
          onClick={() => setDismissed(true)}
          className="text-gray-400 hover:text-white p-1"
          aria-label="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
