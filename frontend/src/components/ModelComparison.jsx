export default function ModelComparison({ models }) {
  if (!models || models.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        No model data available
      </div>
    )
  }

  const modelTypes = ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']

  // Format model name for display
  const formatModelName = (name) => {
    const names = {
      'xgboost': 'XGBoost',
      'lightgbm': 'LightGBM',
      'lstm': 'LSTM',
      'transformer': 'Transformer',
      'ensemble': 'Ensemble'
    }
    return names[name] || name
  }

  // Get precision color
  const getPrecisionColor = (precision) => {
    if (precision >= 75) return 'text-green-400'
    if (precision >= 65) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="overflow-x-auto">
      <table className="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th className="text-center">Up Precision</th>
            <th className="text-center">Down Precision</th>
            <th className="text-center">Avg Precision</th>
            <th className="text-center">Status</th>
          </tr>
        </thead>
        <tbody>
          {modelTypes.map((modelType) => {
            const model = models.find(m => m.type === modelType) || {}
            const upHitRate = model.up_hit_rate || 0
            const downHitRate = model.down_hit_rate || 0
            const avgHitRate = (upHitRate + downHitRate) / 2
            const isTrained = model.is_trained !== false

            return (
              <tr key={modelType} className="hover:bg-surface-light/50 transition-colors">
                <td className="font-semibold">{formatModelName(modelType)}</td>
                <td className={`text-center ${getPrecisionColor(upHitRate)}`}>
                  {upHitRate.toFixed(1)}%
                </td>
                <td className={`text-center ${getPrecisionColor(downHitRate)}`}>
                  {downHitRate.toFixed(1)}%
                </td>
                <td className={`text-center font-semibold ${getPrecisionColor(avgHitRate)}`}>
                  {avgHitRate.toFixed(1)}%
                </td>
                <td className="text-center">
                  {isTrained ? (
                    <span className="text-green-400">✓ Trained</span>
                  ) : (
                    <span className="text-yellow-400">⏳ Training</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-6 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-400 rounded" />
          <span>≥75% Precision</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-yellow-400 rounded" />
          <span>65-75% Precision</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-400 rounded" />
          <span>&lt;65% Precision</span>
        </div>
      </div>
    </div>
  )
}
