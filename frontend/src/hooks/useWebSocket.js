import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)
  const wsRef = useRef(null)
  const queryClient = useQueryClient()
  const reconnectTimeoutRef = useRef(null)

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

              case 'ticker_update':
                // Invalidate specific ticker data
                if (data.ticker) {
                  queryClient.invalidateQueries({ queryKey: ['models', data.ticker] })
                }
                break

              case 'heartbeat':
                // Just to keep connection alive
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
  }, [queryClient])

  return {
    isConnected,
    lastUpdate,
  }
}
