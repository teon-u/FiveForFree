import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Calculate Simple Moving Average
 */
export function calculateMA(data, index, period) {
  if (index < period - 1) return null
  const slice = data.slice(index - period + 1, index + 1)
  const sum = slice.reduce((acc, item) => acc + (item.close || item), 0)
  return sum / period
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBB(data, index, period = 20, multiplier = 2) {
  if (index < period - 1) return { upper: null, lower: null, middle: null }

  const slice = data.slice(index - period + 1, index + 1)
  const closes = slice.map(item => item.close || item)
  const mean = closes.reduce((acc, v) => acc + v, 0) / period
  const variance = closes.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / period
  const stdDev = Math.sqrt(variance)

  return {
    upper: mean + (multiplier * stdDev),
    lower: mean - (multiplier * stdDev),
    middle: mean
  }
}

/**
 * Fetch sparkline data (60 data points for mini chart)
 */
async function fetchSparklineData(ticker) {
  // In development, generate mock data
  if (import.meta.env.DEV) {
    return generateMockSparkline(ticker)
  }

  const response = await axios.get(`${API_BASE}/api/prices/${ticker}/sparkline`)
  return response.data
}

/**
 * Fetch price history with interval and period
 */
async function fetchPriceHistory(ticker, interval, period) {
  // In development, generate mock data
  if (import.meta.env.DEV) {
    return generateMockHistory(ticker, interval, period)
  }

  const response = await axios.get(`${API_BASE}/api/prices/${ticker}/history`, {
    params: { interval, period }
  })
  return response.data
}

/**
 * Generate mock sparkline data for development
 */
function generateMockSparkline(ticker) {
  const basePrice = 100 + (ticker.charCodeAt(0) % 100)
  const volatility = 0.02

  const data = []
  let price = basePrice
  for (let i = 0; i < 60; i++) {
    const change = (Math.random() - 0.5) * volatility * basePrice
    price += change
    data.push(Number(price.toFixed(2)))
  }

  const firstPrice = data[0]
  const lastPrice = data[data.length - 1]
  const direction = lastPrice >= firstPrice ? 'up' : 'down'

  return {
    symbol: ticker,
    data,
    direction,
    min: Math.min(...data),
    max: Math.max(...data),
    change: ((lastPrice - firstPrice) / firstPrice * 100).toFixed(2),
    timestamp: new Date().toISOString()
  }
}

/**
 * Generate mock price history for development
 */
function generateMockHistory(ticker, interval, period) {
  const basePrice = 100 + (ticker.charCodeAt(0) % 100)
  const volatility = 0.03

  // Determine number of data points based on interval and period
  const pointsMap = {
    '1m': { '1D': 60, '1W': 420, '1M': 1800, '3M': 5400 },
    '5m': { '1D': 78, '1W': 390, '1M': 1560, '3M': 4680 },
    '15m': { '1D': 26, '1W': 130, '1M': 520, '3M': 1560 },
    '1h': { '1D': 7, '1W': 35, '1M': 150, '3M': 450 },
    '1d': { '1D': 1, '1W': 5, '1M': 22, '3M': 66 }
  }

  const numPoints = Math.min(pointsMap[interval]?.[period] || 60, 200)

  const data = []
  let price = basePrice
  const now = new Date()

  // Calculate time step
  const intervalMs = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
  }

  const step = intervalMs[interval] || 60 * 1000

  for (let i = numPoints - 1; i >= 0; i--) {
    const change = (Math.random() - 0.48) * volatility * basePrice
    price += change

    const open = price
    const high = price * (1 + Math.random() * 0.01)
    const low = price * (1 - Math.random() * 0.01)
    const close = low + Math.random() * (high - low)
    const volume = Math.floor(Math.random() * 1000000) + 100000

    data.push({
      time: new Date(now.getTime() - (i * step)).toISOString(),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume
    })

    price = close
  }

  return {
    symbol: ticker,
    interval,
    data
  }
}

/**
 * Hook for sparkline data
 */
export function useSparkline(ticker) {
  return useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => fetchSparklineData(ticker),
    staleTime: 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 60 * 1000, // Refetch every minute
    enabled: !!ticker
  })
}

/**
 * Hook for price history with technical indicators
 */
export function usePriceHistory(ticker, interval = '1m', period = '1D') {
  return useQuery({
    queryKey: ['priceHistory', ticker, interval, period],
    queryFn: () => fetchPriceHistory(ticker, interval, period),
    staleTime: 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: interval === '1m' ? 60 * 1000 : false,
    enabled: !!ticker,
    select: (data) => {
      if (!data?.data) return data

      // Add technical indicators
      const enrichedData = data.data.map((item, index, arr) => ({
        ...item,
        displayTime: formatTime(item.time, interval),
        ma5: calculateMA(arr, index, 5),
        ma20: calculateMA(arr, index, 20),
        ...calculateBB(arr, index, 20, 2)
      }))

      return {
        ...data,
        data: enrichedData
      }
    }
  })
}

/**
 * Format time based on interval
 */
function formatTime(isoString, interval) {
  const date = new Date(isoString)

  if (interval === '1d') {
    return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
  }

  if (interval === '1h') {
    return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
  }

  return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
}
