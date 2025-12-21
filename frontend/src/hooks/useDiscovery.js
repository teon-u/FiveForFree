import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { endpoints } from '../services/api'

// Mock data for development mode
const generateMockDiscoveryData = () => {
  const mockGainers = [
    { ticker: 'PLTR', name: 'Palantir Technologies Inc.', change_percent: 12.5, volume: 85000000, sector: 'technology', is_trained: false },
    { ticker: 'RIVN', name: 'Rivian Automotive Inc.', change_percent: 8.3, volume: 45000000, sector: 'consumer', is_trained: false },
    { ticker: 'SOFI', name: 'SoFi Technologies Inc.', change_percent: 6.2, volume: 32000000, sector: 'finance', is_trained: true },
    { ticker: 'LCID', name: 'Lucid Group Inc.', change_percent: 5.8, volume: 28000000, sector: 'consumer', is_trained: false },
    { ticker: 'HOOD', name: 'Robinhood Markets Inc.', change_percent: 4.5, volume: 22000000, sector: 'finance', is_trained: true },
  ]

  const mockVolumeTop = [
    { ticker: 'GME', name: 'GameStop Corp.', change_percent: 5.2, volume: 120000000, sector: 'consumer', is_trained: true },
    { ticker: 'AMC', name: 'AMC Entertainment Holdings', change_percent: 3.1, volume: 95000000, sector: 'consumer', is_trained: true },
    { ticker: 'BBBY', name: 'Bed Bath & Beyond Inc.', change_percent: 2.8, volume: 78000000, sector: 'consumer', is_trained: false },
  ]

  return {
    timestamp: new Date().toISOString(),
    summary: {
      total_tickers: 150,
      trained_tickers: 142,
      model_coverage: 94.7,
      new_gainers_count: mockGainers.length,
      new_volume_count: mockVolumeTop.length,
    },
    new_gainers: mockGainers.map(g => ({
      ...g,
      discovered_at: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    })),
    new_volume_top: mockVolumeTop.map(v => ({
      ...v,
      discovered_at: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    })),
    training_queue: [
      { ticker: 'PLTR', status: 'pending', position: 1, estimated_time: 120 },
      { ticker: 'RIVN', status: 'pending', position: 2, estimated_time: 240 },
    ],
  }
}

async function fetchDiscoveryData() {
  // Check if we're in development mode or API is not available
  const isDev = import.meta.env.DEV

  try {
    const response = await endpoints.discoverNewTickers()
    return response.data
  } catch (error) {
    // If API fails in dev mode, use mock data
    if (isDev) {
      console.warn('Discovery API not available, using mock data')
      return generateMockDiscoveryData()
    }
    throw error
  }
}

export function useDiscovery() {
  const query = useQuery({
    queryKey: ['discovery'],
    queryFn: fetchDiscoveryData,
    staleTime: 60 * 1000, // 1 minute
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
  })

  return {
    discovery: query.data,
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  }
}

export function useTrainTicker() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (ticker) => {
      const isDev = import.meta.env.DEV

      try {
        const response = await endpoints.trainTicker(ticker)
        return response.data
      } catch (error) {
        if (isDev) {
          // Mock success in dev mode
          console.warn(`Mock training started for ${ticker}`)
          return { success: true, ticker, status: 'queued' }
        }
        throw error
      }
    },
    onSuccess: () => {
      // Invalidate discovery query to refresh data
      queryClient.invalidateQueries({ queryKey: ['discovery'] })
    },
  })
}

export function useTrainingStatus() {
  return useQuery({
    queryKey: ['training-status'],
    queryFn: async () => {
      const isDev = import.meta.env.DEV

      try {
        const response = await endpoints.getTrainingStatus()
        return response.data
      } catch (error) {
        if (isDev) {
          console.warn('Training status API not available, using mock data')
          return {
            queue: [
              { ticker: 'PLTR', status: 'training', progress: 45 },
              { ticker: 'RIVN', status: 'pending', position: 2 },
            ],
            completed_today: ['AAPL', 'MSFT', 'GOOGL'],
          }
        }
        throw error
      }
    },
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
  })
}
