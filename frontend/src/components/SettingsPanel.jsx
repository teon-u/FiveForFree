import { useSettingsStore } from '../stores/settingsStore'

export default function SettingsPanel({ onClose }) {
  const {
    probabilityThreshold,
    filterMode,
    setProbabilityThreshold,
    setFilterMode,
  } = useSettingsStore()

  return (
    <>
      {/* Backdrop */}
      <div className="modal-backdrop" onClick={onClose} />

      {/* Panel */}
      <div className="fixed right-0 top-0 bottom-0 w-96 bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-2xl"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Probability Threshold */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Probability Threshold
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="30"
                max="90"
                step="5"
                value={probabilityThreshold}
                onChange={(e) => setProbabilityThreshold(parseInt(e.target.value))}
                className="flex-1 h-2 bg-surface-light rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="w-20 px-3 py-2 bg-surface-light rounded-lg text-center font-semibold">
                {probabilityThreshold}%
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              Filter predictions below this probability
            </p>
          </div>

          {/* Filter Mode */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Direction Filter
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setFilterMode('all')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  filterMode === 'all'
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilterMode('up')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  filterMode === 'up'
                    ? 'bg-green-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                Up ↑
              </button>
              <button
                onClick={() => setFilterMode('down')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  filterMode === 'down'
                    ? 'bg-red-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                Down ↓
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              Show only specific prediction directions
            </p>
          </div>

          {/* Divider */}
          <div className="border-t border-surface-light" />

          {/* Info Section */}
          <div className="bg-surface-light rounded-lg p-4 space-y-3">
            <h3 className="font-semibold text-sm text-gray-300">About This System</h3>
            <div className="text-xs text-gray-400 space-y-2">
              <p>
                This system uses 4 ML models (XGBoost, LightGBM, LSTM, Transformer)
                to predict NASDAQ stock movements within 60 minutes.
              </p>
              <p>
                Predictions are generated using historical minute bar data from Yahoo Finance.
              </p>
              <p>
                Hit rates are calculated based on backtesting results.
              </p>
            </div>
          </div>

          {/* Color Legend */}
          <div className="space-y-2">
            <h3 className="font-semibold text-sm text-gray-300">Card Colors</h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-strong-up rounded" />
                <span className="text-gray-400">80%+ Up Probability</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-up rounded" />
                <span className="text-gray-400">70-80% Up Probability</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-strong-down rounded" />
                <span className="text-gray-400">80%+ Down Probability</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-down rounded" />
                <span className="text-gray-400">70-80% Down Probability</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border border-neutral/30 rounded" />
                <span className="text-gray-400">&lt;70% Probability</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
