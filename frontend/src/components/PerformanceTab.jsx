import {
  LineChart,
  Line,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

export default function PerformanceTab({ data, ticker, tr }) {
  const { confusion_matrix, metrics, roc_curve, pr_curve, time_series, calibration } = data
  // Use translation if provided, otherwise use default English
  const t = tr || ((key) => key.split('.').pop())

  // Prepare ROC curve data
  const rocData = roc_curve.fpr && roc_curve.tpr ? roc_curve.fpr.map((fpr, i) => ({
    fpr: fpr * 100,
    tpr: roc_curve.tpr[i] * 100,
  })) : []

  // Prepare PR curve data
  const prData = pr_curve.precision && pr_curve.recall ? pr_curve.precision.map((precision, i) => ({
    recall: pr_curve.recall[i] * 100,
    precision: precision * 100,
  })) : []

  // Prepare calibration data
  const calibrationData = calibration.predicted_probs && calibration.actual_freq ? calibration.predicted_probs.map((pred, i) => ({
    predicted: pred * 100,
    actual: calibration.actual_freq[i] * 100,
    count: calibration.bin_counts[i],
  })) : []

  // Prepare time series data
  const timeSeriesData = time_series && time_series.length > 0 ? time_series.map(item => ({
    time: new Date(item.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    accuracy: item.accuracy * 100,
  })) : []

  // Confusion matrix values
  const { tp, fp, tn, fn } = confusion_matrix || { tp: 0, fp: 0, tn: 0, fn: 0 }
  const total = tp + fp + tn + fn

  return (
    <div className="space-y-6">
      {/* Confusion Matrix Section */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Confusion Matrix (Last 50 hours)
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* Confusion Matrix Visual */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="col-span-1" />
              <div className="text-center font-semibold text-gray-400">Predicted Up</div>
              <div className="text-center font-semibold text-gray-400">Predicted Down</div>

              <div className="text-right pr-2 font-semibold text-gray-400">Actual Up</div>
              <div className="bg-green-500/20 border border-green-500/40 rounded p-4 text-center">
                <div className="text-2xl font-bold text-green-400">{tp}</div>
                <div className="text-xs text-gray-400 mt-1">True Positive</div>
              </div>
              <div className="bg-red-500/20 border border-red-500/40 rounded p-4 text-center">
                <div className="text-2xl font-bold text-red-400">{fn}</div>
                <div className="text-xs text-gray-400 mt-1">False Negative</div>
              </div>

              <div className="text-right pr-2 font-semibold text-gray-400">Actual Down</div>
              <div className="bg-orange-500/20 border border-orange-500/40 rounded p-4 text-center">
                <div className="text-2xl font-bold text-orange-400">{fp}</div>
                <div className="text-xs text-gray-400 mt-1">False Positive</div>
              </div>
              <div className="bg-blue-500/20 border border-blue-500/40 rounded p-4 text-center">
                <div className="text-2xl font-bold text-blue-400">{tn}</div>
                <div className="text-xs text-gray-400 mt-1">True Negative</div>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <h4 className="font-semibold text-white mb-4">Performance Metrics</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Accuracy</span>
                <span className="text-xl font-bold text-white">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Precision</span>
                <span className="text-xl font-bold text-green-400">
                  {(metrics.precision * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Recall</span>
                <span className="text-xl font-bold text-blue-400">
                  {(metrics.recall * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">F1-Score</span>
                <span className="text-xl font-bold text-purple-400">
                  {(metrics.f1_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-px bg-surface-lighter my-2" />
              <div className="flex items-center justify-between">
                <span className="text-gray-300">FP Rate</span>
                <span className={`text-lg font-semibold ${metrics.fp_rate < 0.15 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {(metrics.fp_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">FN Rate</span>
                <span className={`text-lg font-semibold ${metrics.fn_rate < 0.15 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {(metrics.fn_rate * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ROC and PR Curves */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">📊</span>
          ROC & Precision-Recall Curves
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* ROC Curve */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <h4 className="font-semibold text-white mb-4">
              ROC Curve
              <span className="ml-2 text-sm font-normal text-gray-400">
                (AUC: {roc_curve.insufficient_data ? 'N/A' : (roc_curve.auc ? roc_curve.auc.toFixed(3) : '0.000')})
              </span>
            </h4>
            {rocData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rocData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="fpr"
                    stroke="#9CA3AF"
                    label={{ value: 'False Positive Rate (%)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                  />
                  <YAxis
                    stroke="#9CA3AF"
                    label={{ value: 'True Positive Rate (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#F3F4F6' }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                    stroke="#6B7280"
                    strokeDasharray="5 5"
                    label={{ value: 'Random', fill: '#6B7280', fontSize: 12 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex flex-col items-center justify-center text-gray-400">
                <span>No data available</span>
                {roc_curve.insufficient_data && (
                  <span className="text-xs mt-2">⚠️ Requires 10+ samples (current: {roc_curve.sample_count || 0})</span>
                )}
              </div>
            )}
            <p className="text-xs text-gray-400 mt-2 text-center">
              {roc_curve.insufficient_data ? '⚠️ Insufficient data for evaluation' :
               roc_curve.auc >= 0.9 ? '🟢 Excellent' :
               roc_curve.auc >= 0.8 ? '🟡 Good' :
               roc_curve.auc >= 0.7 ? '🟠 Fair' : '🔴 Poor'}
            </p>
          </div>

          {/* PR Curve */}
          <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
            <h4 className="font-semibold text-white mb-4">
              Precision-Recall Curve
              <span className="ml-2 text-sm font-normal text-gray-400">
                (AP: {pr_curve.insufficient_data ? 'N/A' : (pr_curve.average_precision ? pr_curve.average_precision.toFixed(3) : '0.000')})
              </span>
            </h4>
            {prData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={prData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="recall"
                    stroke="#9CA3AF"
                    label={{ value: 'Recall (%)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                  />
                  <YAxis
                    stroke="#9CA3AF"
                    label={{ value: 'Precision (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#F3F4F6' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="precision"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex flex-col items-center justify-center text-gray-400">
                <span>No data available</span>
                {pr_curve.insufficient_data && (
                  <span className="text-xs mt-2">⚠️ Requires 10+ samples (current: {pr_curve.sample_count || 0})</span>
                )}
              </div>
            )}
            <p className="text-xs text-gray-400 mt-2 text-center">
              {pr_curve.insufficient_data ? '⚠️ Insufficient data for evaluation' :
               pr_curve.average_precision >= 0.9 ? '🟢 Excellent' :
               pr_curve.average_precision >= 0.8 ? '🟡 Good' :
               pr_curve.average_precision >= 0.7 ? '🟠 Fair' : '🔴 Poor'}
            </p>
          </div>
        </div>
      </section>

      {/* Performance Over Time */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">📉</span>
          Performance Over Time (50h rolling window)
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          {timeSeriesData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={timeSeriesData}>
                <defs>
                  <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="time"
                  stroke="#9CA3AF"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis
                  stroke="#9CA3AF"
                  domain={[0, 100]}
                  label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#8B5CF6"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorAccuracy)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-gray-400">
              No time series data available
            </div>
          )}
        </div>
      </section>

      {/* Calibration Curve */}
      <section>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Probability Calibration
          <span className="ml-2 text-sm font-normal text-gray-400">
            (Score: {calibration.insufficient_data ? 'N/A' : (calibration.calibration_score ? (calibration.calibration_score * 100).toFixed(1) : '0.0')}%)
          </span>
        </h3>
        <div className="bg-surface-light rounded-lg p-6 border border-surface-lighter">
          {calibrationData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  type="number"
                  dataKey="predicted"
                  domain={[0, 100]}
                  stroke="#9CA3AF"
                  label={{ value: 'Predicted Probability (%)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                />
                <YAxis
                  type="number"
                  dataKey="actual"
                  domain={[0, 100]}
                  stroke="#9CA3AF"
                  label={{ value: 'Actual Frequency (%)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                  formatter={(value, name) => {
                    if (name === 'predicted') return [`${value.toFixed(1)}%`, 'Predicted']
                    if (name === 'actual') return [`${value.toFixed(1)}%`, 'Actual']
                    return [value, name]
                  }}
                />
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                  stroke="#6B7280"
                  strokeDasharray="5 5"
                  label={{ value: 'Perfect Calibration', fill: '#6B7280', fontSize: 12 }}
                />
                <Scatter
                  data={calibrationData}
                  fill="#F59E0B"
                  shape="circle"
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={false}
                />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex flex-col items-center justify-center text-gray-400">
              <span>No calibration data available</span>
              {calibration.insufficient_data && (
                <span className="text-xs mt-2">⚠️ Requires 10+ samples (current: {calibration.sample_count || 0})</span>
              )}
            </div>
          )}
          <p className="text-sm text-gray-400 mt-4 text-center">
            {calibration.insufficient_data ? '⚠️ Insufficient data for calibration evaluation' :
             calibration.calibration_score >= 0.9 ? '✓ Well-calibrated (probabilities are reliable)' :
             calibration.calibration_score >= 0.8 ? '⚠️ Moderately calibrated (use with caution)' :
             '❌ Poorly calibrated (probabilities may be misleading)'}
          </p>
        </div>
      </section>
    </div>
  )
}
