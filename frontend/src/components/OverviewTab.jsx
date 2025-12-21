export default function OverviewTab({ data, tr }) {
  const { prediction, ranking, quick_stats, risk_indicators } = data
  // Use translation if provided, otherwise use default English
  const t = tr || ((key) => key.split('.').pop())

  // Risk level styling
  const getRiskStyle = (level) => {
    switch (level) {
      case 'low':
        return 'text-green-400 bg-green-400/10 border-green-400/20'
      case 'moderate':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
      case 'high':
        return 'text-red-400 bg-red-400/10 border-red-400/20'
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20'
    }
  }

  // Trend emoji
  const getTrendEmoji = (trend) => {
    switch (trend) {
      case 'improving':
        return '📈'
      case 'declining':
        return '📉'
      case 'stable':
        return '➡️'
      default:
        return '❓'
    }
  }

  // Model name formatting
  const formatModelName = (name) => {
    const names = {
      xgboost: 'XGBoost',
      lightgbm: 'LightGBM',
      lstm: 'LSTM',
      transformer: 'Transformer',
      ensemble: 'Ensemble',
    }
    return names[name] || name.toUpperCase()
  }

  // Practicality Grade styling
  const getGradeStyle = (grade) => {
    switch (grade) {
      case 'A':
        return 'text-green-400 bg-green-400/20 border-green-400/40'
      case 'B':
        return 'text-blue-400 bg-blue-400/20 border-blue-400/40'
      case 'C':
        return 'text-yellow-400 bg-yellow-400/20 border-yellow-400/40'
      default:
        return 'text-red-400 bg-red-400/20 border-red-400/40'
    }
  }

  return (
    <div className="space-y-6">
      {/* Current Prediction */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          {t('overview.currentPrediction')}
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-400 mb-1">Direction</p>
              <p className={`text-2xl font-bold ${prediction.direction === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                {prediction.direction === 'up' ? '↑ UP' : '↓ DOWN'}
                <span className="text-sm ml-2">
                  ({(prediction.probability * 100).toFixed(0)}%)
                </span>
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">Best Model</p>
              <p className="text-lg font-semibold text-white">
                {formatModelName(prediction.best_model)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">Expected Move</p>
              <p className="text-lg font-semibold text-blue-400">
                {prediction.expected_change >= 0 ? '+' : ''}{prediction.expected_change.toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">Risk Level</p>
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold border ${getRiskStyle(prediction.risk_level)}`}>
                {prediction.risk_level.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Model Ranking */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🏆</span>
          {t('overview.modelRanking')}
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          {/* Header */}
          <div className="grid grid-cols-12 gap-2 text-xs text-gray-400 mb-3 px-2">
            <div className="col-span-1">#</div>
            <div className="col-span-3">Model</div>
            <div className="col-span-2 text-right">Precision</div>
            <div className="col-span-2 text-right">Signal Rate</div>
            <div className="col-span-2 text-right">Signals</div>
            <div className="col-span-2 text-center">Grade</div>
          </div>
          <div className="space-y-2">
            {ranking && ranking.length > 0 ? (
              ranking.map((model, index) => (
                <div key={model.model} className="grid grid-cols-12 gap-2 items-center bg-surface rounded-lg p-2">
                  <div className="col-span-1 text-center">
                    <span className={`text-lg font-bold ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-300' :
                      index === 2 ? 'text-orange-400' :
                      'text-gray-500'
                    }`}>
                      {index + 1}
                    </span>
                  </div>
                  <div className="col-span-3">
                    <span className="text-white font-medium">
                      {formatModelName(model.model)}
                    </span>
                  </div>
                  <div className="col-span-2 text-right">
                    <span className={`font-semibold ${
                      model.hit_rate >= 0.5 ? 'text-green-400' :
                      model.hit_rate >= 0.3 ? 'text-blue-400' :
                      'text-gray-400'
                    }`}>
                      {(model.hit_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="col-span-2 text-right">
                    <span className={`font-semibold ${
                      (model.signal_rate || 0) >= 0.1 ? 'text-green-400' :
                      (model.signal_rate || 0) >= 0.05 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {((model.signal_rate || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="col-span-2 text-right">
                    <span className="text-gray-300">
                      {model.signal_count || 0}
                    </span>
                  </div>
                  <div className="col-span-2 text-center">
                    <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold border ${getGradeStyle(model.practicality_grade || 'D')}`}>
                      {model.practicality_grade || 'D'}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-400 text-center py-4">No model data available</p>
            )}
          </div>
          {/* Legend */}
          <div className="mt-4 pt-4 border-t border-surface-lighter">
            <p className="text-xs text-gray-400">
              <span className="font-semibold">Grade: </span>
              <span className="text-green-400">A</span>=Precision≥50% & Signal≥10% |
              <span className="text-blue-400 ml-1">B</span>=Precision≥30% & Signal≥10% |
              <span className="text-yellow-400 ml-1">C</span>=Precision≥30% but low signal |
              <span className="text-red-400 ml-1">D</span>=Not practical
            </p>
          </div>
        </div>
      </section>

      {/* Quick Stats */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">⚡</span>
          {t('overview.quickStats')}
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
            <p className="text-sm text-gray-400 mb-1">Accuracy</p>
            <p className="text-2xl font-bold text-white">
              {(quick_stats.accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
            <p className="text-sm text-gray-400 mb-1">Win Rate</p>
            <p className="text-2xl font-bold text-green-400">
              {quick_stats.win_rate != null ? `${(quick_stats.win_rate * 100).toFixed(1)}%` : 'N/A'}
            </p>
          </div>
          <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
            <p className="text-sm text-gray-400 mb-1">Avg Return</p>
            <p className="text-2xl font-bold text-blue-400">
              {quick_stats.avg_return >= 0 ? '+' : ''}{quick_stats.avg_return.toFixed(1)}%
            </p>
          </div>
          <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
            <p className="text-sm text-gray-400 mb-1">Sharpe Ratio</p>
            <p className="text-2xl font-bold text-purple-400">
              {quick_stats.sharpe.toFixed(2)}
            </p>
          </div>
        </div>
      </section>

      {/* Risk Indicators */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">⚠️</span>
          {t('overview.riskIndicators')}
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">False Positive Rate:</span>
              <span className="font-semibold">
                <span className={risk_indicators.false_positive_rate < 0.15 ? 'text-green-400' : 'text-yellow-400'}>
                  {(risk_indicators.false_positive_rate * 100).toFixed(1)}%
                </span>
                <span className="text-gray-400 text-sm ml-2">
                  {risk_indicators.false_positive_rate < 0.15 ? '(Good ✓)' : '(Watch ⚠️)'}
                </span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Model Agreement:</span>
              <span className="font-semibold">
                <span className={risk_indicators.model_agreement > 0.7 ? 'text-green-400' : 'text-yellow-400'}>
                  {(risk_indicators.model_agreement * 100).toFixed(0)}%
                </span>
                <span className="text-gray-400 text-sm ml-2">
                  {risk_indicators.model_agreement > 0.7 ? '(Strong ✓)' : '(Weak ⚠️)'}
                </span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Recent Performance:</span>
              <span className="font-semibold">
                <span className="text-white">
                  {getTrendEmoji(risk_indicators.performance_trend)} {risk_indicators.performance_trend}
                </span>
                <span className="text-gray-400 text-sm ml-2">
                  {risk_indicators.performance_trend === 'improving' ? '(✓)' :
                   risk_indicators.performance_trend === 'declining' ? '(⚠️)' : '(-)'}
                </span>
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Investment Decision Helper */}
      <section>
        <div className={`rounded-lg p-6 border ${
          prediction.risk_level === 'low' ? 'bg-green-500/10 border-green-500/30' :
          prediction.risk_level === 'moderate' ? 'bg-yellow-500/10 border-yellow-500/30' :
          'bg-red-500/10 border-red-500/30'
        }`}>
          <h4 className="text-lg font-bold text-white mb-2">💡 {t('overview.investmentSignal')}</h4>
          <p className="text-gray-300">
            {prediction.risk_level === 'low' && (
              <>Strong signal detected. Model shows high confidence ({(prediction.probability * 100).toFixed(0)}%) with {risk_indicators.model_agreement > 0.7 ? 'strong' : 'moderate'} agreement among models.</>
            )}
            {prediction.risk_level === 'moderate' && (
              <>Moderate signal. Consider position sizing carefully. Model confidence is {(prediction.probability * 100).toFixed(0)}% with {risk_indicators.model_agreement > 0.7 ? 'good' : 'mixed'} model agreement.</>
            )}
            {prediction.risk_level === 'high' && (
              <>Weak signal. High risk detected. Model confidence is only {(prediction.probability * 100).toFixed(0)}%. Consider waiting for better setup.</>
            )}
          </p>
        </div>
      </section>
    </div>
  )
}
