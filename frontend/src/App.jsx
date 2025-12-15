import { useState } from 'react'
import Dashboard from './components/Dashboard'
import SettingsPanel from './components/SettingsPanel'
import { useWebSocket } from './hooks/useWebSocket'

function App() {
  const [showSettings, setShowSettings] = useState(false)
  const { isConnected, lastUpdate } = useWebSocket()

  return (
    <div className="min-h-screen bg-background text-white">
      {/* Header */}
      <header className="bg-surface border-b border-surface-light">
        <div className="max-w-[1920px] mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                NASDAQ Predictor
              </h1>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm text-gray-400">
                  {isConnected ? 'Live' : 'Disconnected'}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4">
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
        <Dashboard />
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
