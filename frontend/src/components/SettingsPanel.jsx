import { useState, useEffect } from 'react'
import { useSettingsStore } from '../stores/settingsStore'
import { t } from '../i18n'
import { endpoints } from '../services/api'

export default function SettingsPanel({ onClose }) {
  const {
    probabilityThreshold,
    filterMode,
    language,
    setProbabilityThreshold,
    setFilterMode,
    setLanguage,
  } = useSettingsStore()

  const tr = t(language)

  // System status state
  const [systemStatus, setSystemStatus] = useState(null)
  const [loadingStatus, setLoadingStatus] = useState(false)
  const [trainingTickers, setTrainingTickers] = useState({})

  // Fetch system status on mount
  useEffect(() => {
    fetchSystemStatus()
  }, [])

  const fetchSystemStatus = async () => {
    setLoadingStatus(true)
    try {
      const response = await endpoints.getSystemStatus()
      setSystemStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    } finally {
      setLoadingStatus(false)
    }
  }

  const handleTrainTicker = async (ticker) => {
    setTrainingTickers(prev => ({ ...prev, [ticker]: 'training' }))
    try {
      await endpoints.trainTicker(ticker)
      setTrainingTickers(prev => ({ ...prev, [ticker]: 'trained' }))
      // Refresh status after training
      setTimeout(fetchSystemStatus, 2000)
    } catch (error) {
      console.error('Failed to train ticker:', error)
      setTrainingTickers(prev => ({ ...prev, [ticker]: 'error' }))
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div className="modal-backdrop" onClick={onClose} />

      {/* Panel */}
      <div className="fixed right-0 top-0 bottom-0 w-96 bg-surface border-l border-surface-light shadow-2xl z-50 overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-surface-light px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold">{tr('settings.title')}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-2xl"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Language Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              {tr('settings.language')}
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setLanguage('ko')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  language === 'ko'
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {tr('settings.korean')}
              </button>
              <button
                onClick={() => setLanguage('en')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  language === 'en'
                    ? 'bg-blue-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {tr('settings.english')}
              </button>
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-surface-light" />

          {/* Probability Threshold */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              {tr('settings.threshold')}
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
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
              {language === 'ko' ? '이 확률 이하의 예측은 필터링됩니다' : 'Filter predictions below this probability'}
            </p>
          </div>

          {/* Filter Mode */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              {tr('settings.filter')}
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
                {tr('settings.filterAll')}
              </button>
              <button
                onClick={() => setFilterMode('up')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  filterMode === 'up'
                    ? 'bg-green-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {tr('settings.filterUp')} ↑
              </button>
              <button
                onClick={() => setFilterMode('down')}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  filterMode === 'down'
                    ? 'bg-red-500 text-white'
                    : 'bg-surface-light text-gray-400 hover:bg-slate-600'
                }`}
              >
                {tr('settings.filterDown')} ↓
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              {language === 'ko' ? '특정 방향의 예측만 표시합니다' : 'Show only specific prediction directions'}
            </p>
          </div>

          {/* Divider */}
          <div className="border-t border-surface-light" />

          {/* System Status Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm text-gray-300">
                {tr('status.title')}
              </h3>
              <button
                onClick={fetchSystemStatus}
                disabled={loadingStatus}
                className="text-xs text-blue-400 hover:text-blue-300 disabled:text-gray-500"
              >
                {loadingStatus ? '...' : tr('status.refresh')}
              </button>
            </div>

            {systemStatus && (
              <div className="bg-surface-light rounded-lg p-4 space-y-3">
                {/* Stats */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-400">{tr('status.trainedTickers')}:</span>
                    <span className="ml-2 text-white font-semibold">{systemStatus.trained_tickers}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">{tr('status.coverage')}:</span>
                    <span className="ml-2 text-green-400 font-semibold">{systemStatus.model_coverage}%</span>
                  </div>
                </div>

                {/* Data Range */}
                {systemStatus.data_date_range && (
                  <div className="text-xs text-gray-400">
                    {tr('status.dataRange')}: {systemStatus.data_date_range.start?.split('T')[0]} ~ {systemStatus.data_date_range.end?.split('T')[0]}
                  </div>
                )}
              </div>
            )}

            {/* New Gainers */}
            {systemStatus?.new_gainers?.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-yellow-400">
                  🔥 {tr('status.newGainers')} ({systemStatus.new_gainers.length})
                </h4>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {systemStatus.new_gainers.map((gainer) => (
                    <div
                      key={gainer.ticker}
                      className="flex items-center justify-between bg-surface rounded-lg p-3 border border-surface-light"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-white">{gainer.ticker}</span>
                          <span className="text-green-400 text-sm">+{gainer.change_percent?.toFixed(1)}%</span>
                        </div>
                        <div className="text-xs text-gray-400 truncate max-w-[180px]">
                          {gainer.name}
                        </div>
                      </div>
                      <button
                        onClick={() => handleTrainTicker(gainer.ticker)}
                        disabled={trainingTickers[gainer.ticker] === 'training' || trainingTickers[gainer.ticker] === 'trained'}
                        className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                          trainingTickers[gainer.ticker] === 'trained'
                            ? 'bg-green-500/20 text-green-400 cursor-default'
                            : trainingTickers[gainer.ticker] === 'training'
                            ? 'bg-yellow-500/20 text-yellow-400 cursor-wait'
                            : 'bg-blue-500 text-white hover:bg-blue-600'
                        }`}
                      >
                        {trainingTickers[gainer.ticker] === 'trained'
                          ? tr('status.trained')
                          : trainingTickers[gainer.ticker] === 'training'
                          ? tr('status.training')
                          : tr('status.train')}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {systemStatus && (!systemStatus.new_gainers || systemStatus.new_gainers.length === 0) && (
              <div className="text-xs text-gray-400 italic">
                {tr('status.noNewGainers')}
              </div>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-surface-light" />

          {/* Info Section */}
          <div className="bg-surface-light rounded-lg p-4 space-y-3">
            <h3 className="font-semibold text-sm text-gray-300">
              {language === 'ko' ? '시스템 정보' : 'About This System'}
            </h3>
            <div className="text-xs text-gray-400 space-y-2">
              <p>
                {language === 'ko'
                  ? '이 시스템은 4개의 ML 모델(XGBoost, LightGBM, LSTM, Transformer)을 사용하여 60분 내 NASDAQ 주식 움직임을 예측합니다.'
                  : 'This system uses 4 ML models (XGBoost, LightGBM, LSTM, Transformer) to predict NASDAQ stock movements within 60 minutes.'}
              </p>
              <p>
                {language === 'ko'
                  ? '예측은 Yahoo Finance의 분봉 데이터를 기반으로 생성됩니다.'
                  : 'Predictions are generated using historical minute bar data from Yahoo Finance.'}
              </p>
              <p>
                {language === 'ko'
                  ? '적중률은 백테스팅 결과를 기반으로 계산됩니다.'
                  : 'Hit rates are calculated based on backtesting results.'}
              </p>
            </div>
          </div>

          {/* Grade Legend */}
          <div className="space-y-2">
            <h3 className="font-semibold text-sm text-gray-300">
              {language === 'ko' ? '실용성 등급' : 'Practicality Grade'}
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-green-400 bg-green-400/20 border border-green-400/40 rounded font-bold">A</span>
                <span className="text-gray-400">{tr('grade.excellent')} - Precision ≥ 50% & Signal ≥ 10%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-blue-400 bg-blue-400/20 border border-blue-400/40 rounded font-bold">B</span>
                <span className="text-gray-400">{tr('grade.good')} - Precision ≥ 30% & Signal ≥ 10%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-yellow-400 bg-yellow-400/20 border border-yellow-400/40 rounded font-bold">C</span>
                <span className="text-gray-400">{tr('grade.lowSignal')} - Precision ≥ 30% & Signal &lt; 10%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-red-400 bg-red-400/20 border border-red-400/40 rounded font-bold">D</span>
                <span className="text-gray-400">{tr('grade.notPractical')} - Precision &lt; 30%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
