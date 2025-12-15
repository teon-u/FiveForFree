import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'

export function useModels(ticker) {
  // Fetch model comparison data
  const {
    data: modelData,
    isLoading: modelsLoading,
  } = useQuery({
    queryKey: ['models', ticker],
    queryFn: async () => {
      const response = await api.get(`/models/${ticker}`)
      return response.data
    },
    enabled: !!ticker,
    staleTime: 60000, // 1 minute
  })

  // Fetch price chart data (60 minutes)
  const {
    data: priceData,
    isLoading: priceLoading,
  } = useQuery({
    queryKey: ['prices', ticker],
    queryFn: async () => {
      const response = await api.get(`/prices/${ticker}`, {
        params: {
          minutes: 60,
        },
      })
      return response.data.bars || []
    },
    enabled: !!ticker,
    staleTime: 60000, // 1 minute
  })

  return {
    modelData: modelData ? transformModelData(modelData) : null,
    priceData: priceData || [],
    isLoading: modelsLoading || priceLoading,
  }
}

// Transform model data to UI format
function transformModelData(data) {
  const models = []

  // Process each model type
  const modelTypes = ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']

  modelTypes.forEach(type => {
    if (data.models && data.models[type]) {
      const model = data.models[type]
      models.push({
        type,
        up_hit_rate: (model.up_hit_rate || 0) * 100,
        down_hit_rate: (model.down_hit_rate || 0) * 100,
        is_trained: model.is_trained !== false,
      })
    } else {
      // Add placeholder if model doesn't exist
      models.push({
        type,
        up_hit_rate: 0,
        down_hit_rate: 0,
        is_trained: false,
      })
    }
  })

  // Find best prediction
  let bestPrediction = null
  if (data.best_model) {
    bestPrediction = {
      model_name: data.best_model.model_name || 'Unknown',
      direction: data.best_model.direction || 'up',
      probability: (data.best_model.probability || 0) * 100,
      hit_rate: (data.best_model.hit_rate || 0) * 100,
      up_prob: (data.best_model.up_prob || 0) * 100,
      down_prob: (data.best_model.down_prob || 0) * 100,
    }
  }

  return {
    current_price: data.current_price,
    change_percent: data.change_percent,
    models,
    best_prediction: bestPrediction,
  }
}
