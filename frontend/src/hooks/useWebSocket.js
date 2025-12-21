import { useEffect, useRef, useState, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { usePriceStore } from '../stores/priceStore'

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [subscribedTickers, setSubscribedTickers] = useState([])
  const wsRef = useRef(null)
  const queryClient = useQueryClient()
  const reconnectTimeoutRef = useRef(null)
  const updatePrices = usePriceStore((state) => state.updatePrices)
  const setConnected = usePriceStore((state) => state.setConnected)

  useEffect(() => {
    const connect = () => {
      // WebSocket URL (adjust based on your backend configuration)
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

      try {
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('WebSocket connected')
          setIsConnected(true)
          setConnected(true)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log('WebSocket message received:', data.type)

            // Handle different message types
            switch (data.type) {
              case 'predictions_update':
                // Invalidate predictions query to refetch
                queryClient.invalidateQueries({ queryKey: ['predictions'] })
                setLastUpdate(new Date().toISOString())
                break

              case 'prediction_update':
                // Single ticker prediction update
                if (data.ticker) {
                  queryClient.invalidateQueries({ queryKey: ['models', data.ticker] })
                }
                setLastUpdate(new Date().toISOString())
                break

              case 'price_update':
                // Real-time price updates from backend
                if (data.prices && Array.isArray(data.prices)) {
                  updatePrices(data.prices, data.timestamp)
                }
                break

              case 'ticker_update':
                // Invalidate specific ticker data
                if (data.ticker) {
                  queryClient.invalidateQueries({ queryKey: ['models', data.ticker] })
                }
                break

              case 'connected':
                // Welcome message from server
                console.log('Server:', data.message)
                break

              case 'heartbeat':
                // Just to keep connection alive
                break

              case 'subscribed':
                // Subscription confirmed by server
                console.log('Subscription confirmed:', data.tickers)
                break

              case 'unsubscribed':
                // Unsubscription confirmed by server
                console.log('Unsubscription confirmed:', data.tickers)
                break

              case 'error':
                // Error message from server
                console.error('WebSocket server error:', data.message)
                break

              default:
                console.log('Unknown message type:', data.type)
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
        }

        ws.onclose = () => {
          console.log('WebSocket disconnected')
          setIsConnected(false)
          setConnected(false)

          // Attempt to reconnect after 5 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect...')
            connect()
          }, 5000)
        }
      } catch (error) {
        console.error('Error creating WebSocket:', error)
      }
    }

    connect()

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [queryClient, setConnected, updatePrices])

  // Subscribe to specific tickers for price updates
  const subscribeTickers = useCallback((tickers) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        tickers: tickers
      }))
      setSubscribedTickers(prev => [...new Set([...prev, ...tickers])])
      console.log('Subscribed to tickers:', tickers)
    }
  }, [])

  // Unsubscribe from specific tickers
  const unsubscribeTickers = useCallback((tickers) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        tickers: tickers
      }))
      setSubscribedTickers(prev => prev.filter(t => !tickers.includes(t)))
      console.log('Unsubscribed from tickers:', tickers)
    }
  }, [])

  // Send a custom message to the WebSocket server
  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  return {
    isConnected,
    lastUpdate,
    subscribedTickers,
    subscribeTickers,
    unsubscribeTickers,
    sendMessage,
  }
}
