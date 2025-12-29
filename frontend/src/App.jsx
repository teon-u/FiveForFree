import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import IntegratedBacktest from './components/IntegratedBacktest'
import SettingsPanel from './components/SettingsPanel'
import { useWebSocket } from './hooks/useWebSocket'
import { useMarketStatus } from './hooks/useMarketStatus'
import { useSettingsStore } from './stores/settingsStore'
import { t } from './i18n'

function App() {
  const [showSettings, setShowSettings] = useState(false)
  const [activeTab, setActiveTab] = useState('dashboard')
  const { isConnected, lastUpdate } = useWebSocket()
  const { isMarketOpen, lastCloseKst } = useMarketStatus()
  const { language, theme, getEffectiveTheme } = useSettingsStore()
  const tr = t(language)

  // Apply theme to body
  useEffect(() => {
    const effectiveTheme = getEffectiveTheme()
    if (effectiveTheme === 'light') {
      document.body.classList.add('light')
      document.body.classList.remove('dark')
    } else {
      document.body.classList.add('dark')
      document.body.classList.remove('light')
    }
  }, [theme, getEffectiveTheme])

  // Listen for system theme changes
  useEffect(() => {
    if (theme !== 'system') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      const effectiveTheme = getEffectiveTheme()
      if (effectiveTheme === 'light') {
        document.body.classList.add('light')
        document.body.classList.remove('dark')
      } else {
        document.body.classList.add('dark')
        document.body.classList.remove('light')
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme, getEffectiveTheme])

  return (
    <div className="min-h-screen bg-background text-white">
      {/* Header */}
      <header className="bg-surface border-b border-surface-light">
        <div className="max-w-[1920px] mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                NASDAQ Predictor
              </h1>

              {/* Tab Navigation */}
              <div className="flex items-center gap-1 bg-surface-light rounded-lg p-1">
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === 'dashboard'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-surface'
                  }`}
                >
                  Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('backtest')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === 'backtest'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-surface'
                  }`}
                >
                  Backtest
                </button>
              </div>

              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm text-gray-400">
                  {isConnected ? 'Live' : 'Disconnected'}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Market Status Indicator */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
                isMarketOpen
                  ? 'bg-green-500/20 border border-green-500/30'
                  : 'bg-yellow-500/20 border border-yellow-500/30'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isMarketOpen ? 'bg-green-500' : 'bg-yellow-500'
                } animate-pulse`} />
                <span className={`text-sm font-medium ${
                  isMarketOpen ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  {isMarketOpen ? tr('app.marketOpen') : tr('app.marketClosed')}
                </span>
                {!isMarketOpen && lastCloseKst && (
                  <span className="text-xs text-gray-400">
                    ({tr('app.lastClose')}: {lastCloseKst} KST)
                  </span>
                )}
              </div>
              {lastUpdate && (
                <span className="text-sm text-gray-400">
                  Updated: {new Date(lastUpdate).toLocaleTimeString()}
                </span>
              )}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="px-4 py-2 bg-surface-light hover:bg-slate-600 rounded-lg transition-colors"
              >
                Settings
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Settings Panel */}
      {showSettings && (
        <SettingsPanel onClose={() => setShowSettings(false)} />
      )}

      {/* Main Content */}
      <main className="max-w-[1920px] mx-auto px-4 py-6">
        {activeTab === 'dashboard' ? <Dashboard /> : <IntegratedBacktest />}
      </main>

      {/* Footer */}
      <footer className="bg-surface border-t border-surface-light mt-12">
        <div className="max-w-[1920px] mx-auto px-4 py-4 text-center text-sm text-gray-400">
          <p>Real-time predictions using 5 ML models | Data from Polygon.io | Updates every minute</p>
        </div>
      </footer>
    </div>
  )
}

export default App
