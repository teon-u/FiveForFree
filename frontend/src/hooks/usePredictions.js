import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { useSettingsStore } from '../stores/settingsStore'

export function usePredictions() {
  const { probabilityThreshold } = useSettingsStore()

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['predictions', probabilityThreshold],
    queryFn: async () => {
      const response = await api.get('/predictions', {
        params: {
          threshold: probabilityThreshold / 100, // Convert to decimal
        },
      })
      return response.data
    },
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider data stale after 30 seconds
  })

  // Transform data into the format expected by the UI
  const volumeTop100 = data?.volume_top_100?.map(transformPrediction) || []
  const gainersTop100 = data?.gainers_top_100?.map(transformPrediction) || []

  return {
    volumeTop100,
    gainersTop100,
    isLoading,
    error,
    refetch,
  }
}

// Transform backend prediction data to UI format
function transformPrediction(pred) {
  return {
    ticker: pred.ticker,
    probability: pred.probability * 100, // Convert to percentage
    direction: pred.direction,
    change_percent: pred.change_percent || 0,
    best_model: pred.best_model,
    hit_rate: pred.hit_rate * 100, // Convert to percentage
    current_price: pred.current_price,
  }
}
