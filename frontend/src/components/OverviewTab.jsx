export default function OverviewTab({ data, ticker }) {
  const { prediction, ranking, quick_stats, risk_indicators } = data

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

  return (
    <div className="space-y-6">
      {/* Current Prediction */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Current Prediction
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
          Model Ranking (50h Hit Rate)
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          <div className="space-y-3">
            {ranking && ranking.length > 0 ? (
              ranking.map((model, index) => (
                <div key={model.model} className="flex items-center">
                  <div className="w-8 text-center">
                    <span className={`text-lg font-bold ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-300' :
                      index === 2 ? 'text-orange-400' :
                      'text-gray-500'
                    }`}>
                      {index + 1}.
                    </span>
                  </div>
                  <div className="flex-1 ml-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white font-medium">
                        {formatModelName(model.model)}
                      </span>
                      <span className="text-gray-300 font-semibold">
                        {(model.hit_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          index === 0 ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                          'bg-gradient-to-r from-blue-500 to-purple-500'
                        }`}
                        style={{ width: `${model.hit_rate * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-400 text-center py-4">No model data available</p>
            )}
          </div>
        </div>
      </section>

      {/* Quick Stats */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">⚡</span>
          Quick Stats
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
              {(quick_stats.win_rate * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-surface-light rounded-lg p-4 border border-surface-lighter">
            <p className="text-sm text-gray-400 mb-1">Avg Return</p>
            <p className="text-2xl font-bold text-blue-400">
              +{(quick_stats.avg_return * 100).toFixed(1)}%
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
          Risk Indicators
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
          <h4 className="text-lg font-bold text-white mb-2">💡 Investment Signal</h4>
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
