import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'

// Convert ISO time to Korean Standard Time (KST)
function formatToKST(isoString) {
  if (!isoString) return null
  try {
    const date = new Date(isoString)
    return date.toLocaleString('ko-KR', {
      timeZone: 'Asia/Seoul',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    })
  } catch {
    return null
  }
}

export function useMarketStatus() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['market-status'],
    queryFn: async () => {
      const response = await api.get('/health')
      return response.data.market
    },
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000,
  })

  // Convert to Korean time
  const lastCloseKst = formatToKST(data?.last_close_iso)

  return {
    marketStatus: data,
    isLoading,
    error,
    isMarketOpen: data?.is_open ?? false,
    lastCloseEt: data?.last_close_et ?? null,
    lastCloseIso: data?.last_close_iso ?? null,
    lastCloseKst,
    reason: data?.reason ?? null,
  }
}
