import {
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'

export default function EnsembleTab({ data, ticker, tr }) {
  const { meta_learner, base_models, current_agreement, ensemble_vs_base } = data
  // Use translation if provided, otherwise use default English
  const t = tr || ((key) => key.split('.').pop())

  // Format model names
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

  // Prepare weights data for chart
  const weightsData = Object.entries(meta_learner.weights || {}).map(([model, weight]) => ({
    model: formatModelName(model),
    weight: Math.abs(weight) * 100,
    rawWeight: weight,
  }))

  // Prepare base models comparison data
  const comparisonData = base_models
    .filter(m => m.is_trained)
    .map(m => ({
      model: formatModelName(m.type),
      Accuracy: m.accuracy * 100,
      Precision: m.precision * 100,
      Recall: m.recall * 100,
      F1: m.f1_score * 100,
    }))

  // Add ensemble to comparison (using actual metrics from API)
  if (ensemble_vs_base.ensemble_accuracy > 0) {
    comparisonData.push({
      model: 'Ensemble',
      Accuracy: ensemble_vs_base.ensemble_accuracy * 100,
      Precision: (ensemble_vs_base.ensemble_precision || 0) * 100,
      Recall: (ensemble_vs_base.ensemble_recall || 0) * 100,
      F1: (ensemble_vs_base.ensemble_f1 || 0) * 100,
    })
  }

  // Colors for charts
  const colors = {
    xgboost: '#F59E0B',
    lightgbm: '#10B981',
    lstm: '#3B82F6',
    transformer: '#8B5CF6',
    ensemble: '#EF4444',
  }

  const getColor = (modelName) => {
    const lowerName = modelName.toLowerCase()
    return colors[lowerName] || '#6B7280'
  }

  return (
    <div className="space-y-6">
      {/* Ensemble Architecture */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🏗️</span>
          Ensemble Architecture (Stacking)
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          {/* Architecture Diagram */}
          <div className="space-y-4">
            {/* Level 1: Meta Learner */}
            <div className="text-center">
              <div className="inline-block bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg px-6 py-3 border-2 border-purple-400">
                <div className="text-white font-bold">Level 1: Meta Learner</div>
                <div className="text-sm text-purple-100 mt-1">{meta_learner.type}</div>
              </div>
            </div>

            {/* Arrow down */}
            <div className="text-center text-gray-400">
              <div className="text-2xl">↑</div>
              <div className="text-xs">Meta Features (4-dim)</div>
            </div>

            {/* Level 0: Base Models */}
            <div>
              <div className="text-center text-gray-400 text-sm mb-2">Level 0: Base Models</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {base_models.map((model) => (
                  <div
                    key={model.type}
                    className={`rounded-lg p-3 border-2 text-center ${
                      model.is_trained
                        ? 'bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-blue-500/40'
                        : 'bg-gray-800/50 border-gray-700'
                    }`}
                  >
                    <div className={`font-semibold ${model.is_trained ? 'text-white' : 'text-gray-500'}`}>
                      {formatModelName(model.type)}
                    </div>
                    <div className={`text-xs mt-1 ${model.is_trained ? 'text-blue-300' : 'text-gray-600'}`}>
                      {model.is_trained ? `${(model.accuracy * 100).toFixed(1)}%` : 'Not Trained'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-surface-lighter">
              <div className="text-center">
                <div className="text-sm text-gray-400">Trained Models</div>
                <div className="text-2xl font-bold text-white">
                  {meta_learner.trained_base_models} / {meta_learner.total_base_models}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-400">Ensemble Improvement</div>
                <div className={`text-2xl font-bold ${ensemble_vs_base.improvement >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {ensemble_vs_base.improvement >= 0 ? '+' : ''}{(ensemble_vs_base.improvement * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Meta Learner Weights */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">⚖️</span>
          Meta Learner Weights
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          {weightsData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={weightsData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="category" dataKey="model" stroke="#9CA3AF" />
                  <YAxis type="number" stroke="#9CA3AF" label={{ value: 'Weight (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#F3F4F6' }}
                    formatter={(value, name, props) => [
                      `${value.toFixed(1)}% (${props.payload.rawWeight.toFixed(3)})`,
                      'Weight'
                    ]}
                  />
                  <Bar dataKey="weight" radius={[4, 4, 0, 0]}>
                    {weightsData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getColor(entry.model)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-400 mt-4 text-center">
                Higher weights indicate greater influence on final prediction
              </p>
            </>
          ) : (
            <div className="text-center py-8 text-gray-400">
              No weight data available
            </div>
          )}
        </div>
      </section>

      {/* Base Models Comparison */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">📊</span>
          Base Models Comparison (50h)
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* Table */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-lighter">
                    <th className="text-left py-2 text-gray-400 font-semibold">Model</th>
                    <th className="text-right py-2 text-gray-400 font-semibold">Accuracy</th>
                    <th className="text-right py-2 text-gray-400 font-semibold">Precision</th>
                    <th className="text-right py-2 text-gray-400 font-semibold">Recall</th>
                    <th className="text-right py-2 text-gray-400 font-semibold">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {base_models.map((model, index) => (
                    <tr key={model.type} className={`border-b border-surface-lighter/50 ${!model.is_trained && 'opacity-50'}`}>
                      <td className="py-2 text-white font-medium">{formatModelName(model.type)}</td>
                      <td className="text-right py-2 text-gray-300">{(model.accuracy * 100).toFixed(1)}%</td>
                      <td className="text-right py-2 text-gray-300">{(model.precision * 100).toFixed(1)}%</td>
                      <td className="text-right py-2 text-gray-300">{(model.recall * 100).toFixed(1)}%</td>
                      <td className="text-right py-2 text-gray-300">{(model.f1_score * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                  <tr className="border-t-2 border-purple-500/50 bg-purple-500/10">
                    <td className="py-2 text-purple-400 font-bold">Ensemble</td>
                    <td className="text-right py-2 text-purple-300 font-bold">{(ensemble_vs_base.ensemble_accuracy * 100).toFixed(1)}%</td>
                    <td className="text-right py-2 text-purple-300 font-bold">{((ensemble_vs_base.ensemble_precision || 0) * 100).toFixed(1)}%</td>
                    <td className="text-right py-2 text-purple-300 font-bold">{((ensemble_vs_base.ensemble_recall || 0) * 100).toFixed(1)}%</td>
                    <td className="text-right py-2 text-purple-300 font-bold">{((ensemble_vs_base.ensemble_f1 || 0) * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Radar Chart */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            {comparisonData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={comparisonData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="model" stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
                  <Radar name="Accuracy" dataKey="Accuracy" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#F3F4F6' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-400">
                No comparison data available
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Model Agreement Analysis */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🤝</span>
          Model Agreement Analysis
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Current Prediction */}
            <div>
              <h4 className="font-semibold text-white mb-3">Current Prediction</h4>
              <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/30 mb-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Ensemble:</span>
                  <span className="text-xl font-bold text-purple-400">
                    {current_agreement.ensemble_prediction.direction.toUpperCase()} ({(current_agreement.ensemble_prediction.probability * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                {current_agreement.base_predictions.map((pred) => (
                  <div key={pred.model} className="flex items-center justify-between py-2 border-b border-surface-lighter/50">
                    <span className="text-gray-300">{formatModelName(pred.model)}:</span>
                    <span className={`font-semibold ${
                      pred.direction === current_agreement.ensemble_prediction.direction
                        ? 'text-green-400'
                        : 'text-red-400'
                    }`}>
                      {pred.direction.toUpperCase()} ({(pred.probability * 100).toFixed(0)}%)
                      {pred.direction === current_agreement.ensemble_prediction.direction ? ' ✓' : ' ✗'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Agreement Metrics */}
            <div>
              <h4 className="font-semibold text-white mb-3">Agreement Metrics</h4>
              <div className="space-y-4">
                <div className="bg-surface rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400">Agreement Rate</span>
                    <span className={`text-2xl font-bold ${
                      current_agreement.agreement_rate > 0.7 ? 'text-green-400' :
                      current_agreement.agreement_rate > 0.5 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {(current_agreement.agreement_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        current_agreement.agreement_rate > 0.7 ? 'bg-green-500' :
                        current_agreement.agreement_rate > 0.5 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${current_agreement.agreement_rate * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    {current_agreement.agreement_rate > 0.7 ? 'Strong consensus among models ✓' :
                     current_agreement.agreement_rate > 0.5 ? 'Moderate agreement ⚠️' :
                     'Models disagree significantly ❌'}
                  </p>
                </div>

                <div className="bg-surface rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Prediction Variance</span>
                    <span className="text-xl font-bold text-white">
                      {current_agreement.variance.toFixed(4)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    Lower variance indicates more consistent predictions
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Ensemble vs Base Performance */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🏆</span>
          Ensemble vs Individual Models
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-purple-500/10 rounded-lg border border-purple-500/30">
              <div className="text-sm text-gray-400 mb-1">Ensemble</div>
              <div className="text-2xl font-bold text-purple-400">
                {(ensemble_vs_base.ensemble_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-4 bg-blue-500/10 rounded-lg border border-blue-500/30">
              <div className="text-sm text-gray-400 mb-1">Best Base</div>
              <div className="text-2xl font-bold text-blue-400">
                {(ensemble_vs_base.best_base_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-4 bg-gray-500/10 rounded-lg border border-gray-500/30">
              <div className="text-sm text-gray-400 mb-1">Avg Base</div>
              <div className="text-2xl font-bold text-gray-300">
                {(ensemble_vs_base.avg_base_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/30">
              <div className="text-sm text-gray-400 mb-1">Improvement</div>
              <div className={`text-2xl font-bold ${ensemble_vs_base.improvement >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {ensemble_vs_base.improvement >= 0 ? '+' : ''}{(ensemble_vs_base.improvement * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className={`mt-6 p-4 rounded-lg ${
            ensemble_vs_base.improvement >= 0.02 ? 'bg-green-500/10 border border-green-500/30' :
            ensemble_vs_base.improvement >= 0 ? 'bg-yellow-500/10 border border-yellow-500/30' :
            'bg-red-500/10 border border-red-500/30'
          }`}>
            <p className="text-white">
              {ensemble_vs_base.improvement >= 0.02 ? (
                <>✓ Ensemble significantly outperforms individual models. The meta learner effectively combines diverse predictions.</>
              ) : ensemble_vs_base.improvement >= 0 ? (
                <>⚠️ Ensemble shows marginal improvement. Consider if ensemble complexity is justified.</>
              ) : (
                <>❌ Ensemble underperforms best base model. Meta learner may need retraining or ensemble disabled.</>
              )}
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
